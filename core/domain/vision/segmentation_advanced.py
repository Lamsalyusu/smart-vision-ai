from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math


class SegmentationMode(Enum):
    SEMANTIC = "semantic"
    INSTANCE = "instance"
    PANOPTIC = "panoptic"
    INTERACTIVE = "interactive"


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    mask: np.ndarray
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    contour: Optional[np.ndarray] = None
    area: float = 0.0
    mask_iou: Optional[float] = None


@dataclass(frozen=True, slots=True)
class Instance:
    instance_id: int
    class_id: int
    class_name: str
    mask: np.ndarray
    bbox: Tuple[float, float, float, float]
    confidence: float
    score: float


@dataclass(frozen=True, slots=True)
class PanopticResult:
    semantic_map: np.ndarray
    instance_map: np.ndarray
    segments_info: List[dict]


class BaseSegmenter(ABC):
    @abstractmethod
    def segment(self, image: np.ndarray) -> Sequence[SegmentationResult]:
        pass

    @abstractmethod
    def segment_batch(self, images: List[np.ndarray]) -> List[List[SegmentationResult]]:
        pass


class DeepLabV3PlusSegmenter(BaseSegmenter):
    def __init__(
        self,
        backbone: str = "resnet101",
        num_classes: int = 21,
        output_stride: int = 16,
        pretrained: bool = True,
        device: str = "cpu",
    ):
        self.backbone = backbone
        self.num_classes = num_classes
        self.output_stride = output_stride
        self.pretrained = pretrained
        self.device = device
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class ASPP(nn.Module):
                def __init__(self, in_channels, out_channels, atrous_rates):
                    super().__init__()
                    modules = []
                    modules.append(nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    ))
                    for rate in atrous_rates:
                        modules.append(nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True),
                        ))
                    modules.append(nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Conv2d(in_channels, out_channels, 1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    ))
                    self.convs = nn.ModuleList(modules)
                    self.project = nn.Sequential(
                        nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5),
                    )

                def forward(self, x):
                    res = [conv(x) for conv in self.convs[:-1]]
                    res.append(F.interpolate(self.convs[-1](x), size=x.shape[-2:], mode='bilinear', align_corners=False))
                    return self.project(torch.cat(res, dim=1))

            class Decoder(nn.Module):
                def __init__(self, low_level_channels, num_classes):
                    super().__init__()
                    self.low_level_proj = nn.Sequential(
                        nn.Conv2d(low_level_channels, 48, 1, bias=False),
                        nn.BatchNorm2d(48),
                        nn.ReLU(inplace=True),
                    )
                    self.fusion = nn.Sequential(
                        nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, 256, 3, padding=1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, num_classes, 1),
                    )

                def forward(self, x, low_level):
                    low_level = self.low_level_proj(low_level)
                    x = F.interpolate(x, size=low_level.shape[-2:], mode='bilinear', align_corners=False)
                    return self.fusion(torch.cat([x, low_level], dim=1))

            class DeepLabV3Plus(nn.Module):
                def __init__(self, backbone, num_classes, output_stride):
                    super().__init__()
                    if backbone == "resnet50":
                        from torchvision.models import resnet50
                        resnet = resnet50(pretrained=True)
                        low_level_channels = 256
                    elif backbone == "resnet101":
                        from torchvision.models import resnet101
                        resnet = resnet101(pretrained=True)
                        low_level_channels = 256
                    else:
                        raise ValueError(f"Unsupported backbone: {backbone}")

                    self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
                    self.layer1 = resnet.layer1
                    self.layer2 = resnet.layer2
                    self.layer3 = resnet.layer3
                    self.layer4 = resnet.layer4

                    if output_stride == 16:
                        atrous_rates = [6, 12, 18]
                    elif output_stride == 8:
                        atrous_rates = [12, 24, 36]
                    else:
                        raise ValueError("output_stride must be 8 or 16")

                    self.aspp = ASPP(2048, 256, atrous_rates)
                    self.decoder = Decoder(low_level_channels, num_classes)

                def forward(self, x):
                    x0 = self.layer0(x)
                    x1 = self.layer1(x0)
                    x2 = self.layer2(x1)
                    x3 = self.layer3(x2)
                    x4 = self.layer4(x3)
                    x = self.aspp(x4)
                    x = self.decoder(x, x1)
                    return F.interpolate(x, size=x.shape[-2:] * 4, mode='bilinear', align_corners=False)

            self._model = DeepLabV3Plus(self.backbone, self.num_classes, self.output_stride)
            self._model.eval()
            self._initialized = True
            self._torch = torch
            self._F = F

        except ImportError:
            self._initialized = False

    def segment(self, image: np.ndarray) -> Sequence[SegmentationResult]:
        if not self._initialized:
            self._initialize()

        if self._initialized:
            return self._segment_torch(image)
        else:
            return self._segment_classical(image)

    def _segment_torch(self, image: np.ndarray) -> Sequence[SegmentationResult]:
        torch = self._torch
        F = self._F

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        if self.device != "cpu" and torch.cuda.is_available():
            tensor = tensor.cuda()
            self._model = self._model.cuda()

        with torch.no_grad():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).cpu().numpy()[0]
            conf = torch.max(probs, dim=1).values.cpu().numpy()[0]

        results = []
        for class_id in range(1, self.num_classes):
            mask = (pred == class_id).astype(np.uint8)
            if mask.sum() < 100:
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 100:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                class_conf = conf[mask == 1].mean() if (mask == 1).any() else 0.0
                results.append(SegmentationResult(
                    mask=mask,
                    class_id=class_id,
                    class_name=f"class_{class_id}",
                    confidence=float(class_conf),
                    bbox=(x, y, x + w, y + h),
                    contour=contour,
                    area=float(cv2.contourArea(contour)),
                ))

        return results

    def _segment_classical(self, image: np.ndarray) -> Sequence[SegmentationResult]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(image, markers)

        results = []
        for marker_id in range(2, markers.max() + 1):
            mask = (markers == marker_id).astype(np.uint8)
            if mask.sum() < 100:
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 100:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                class_conf = conf[mask == 1].mean() if (mask == 1).any() else 0.0
                results.append(SegmentationResult(
                    mask=mask,
                    class_id=marker_id - 1,
                    class_name=f"segment_{marker_id - 1}",
                    confidence=0.5,
                    bbox=(x, y, x + w, y + h),
                    contour=contour,
                    area=float(cv2.contourArea(contour)),
                ))

        return results

    def segment_interactive(self, image: np.ndarray, points: np.ndarray, labels: np.ndarray) -> SegmentationResult:
        return SegmentationResult(
            mask=np.zeros(image.shape[:2], dtype=np.uint8),
            class_id=0,
            class_name="background",
            confidence=0.0,
            bbox=(0, 0, 0, 0),
        )


class SAMSegmenter(BaseSegmenter):
    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._predictor = None
        self._initialized = False

    def _initialize(self):
        try:
            from segment_anything import sam_model_registry, SamPredictor

            model_map = {
                "vit_h": "sam_vit_h_4b8939.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_b": "sam_vit_b_01ec64.pth",
            }

            if self.model_type not in model_map:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            sam.to(device=self.device)
            self._predictor = SamPredictor(sam)
            self._initialized = True

        except ImportError:
            self._initialized = False

    def segment(self, image: np.ndarray) -> Sequence[SegmentationResult]:
        if not self._initialized:
            self._initialize()

        if self._initialized:
            return self._segment_automatic(image)
        else:
            return self._segment_classical_fallback(image)

    def _segment_automatic(self, image: np.ndarray) -> Sequence[SegmentationResult]:
        from segment_anything import SamAutomaticMaskGenerator

        mask_generator = SamAutomaticMaskGenerator(
            model=self._predictor.model,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )

        masks = mask_generator.generate(image)

        results = []
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation']
            bbox = mask_data['bbox']
            area = mask_data['area']
            predicted_iou = mask_data['predicted_iou']

            results.append(SegmentationResult(
                mask=mask.astype(np.uint8),
                class_id=i,
                class_name=f"object_{i}",
                confidence=float(predicted_iou),
                bbox=(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]),
                area=float(area),
                mask_iou=float(predicted_iou),
            ))

        return results

    def _segment_classical_fallback(self, image: np.ndarray) -> Sequence[SegmentationResult]:
        segmenter = DeepLabV3PlusSegmenter()
        return segmenter.segment(image)

    def segment_interactive(self, image: np.ndarray, points: np.ndarray, labels: np.ndarray) -> SegmentationResult:
        if not self._initialized:
            self._initialize()

        if self._initialized:
            self._predictor.set_image(image)
            masks, scores, logits = self._predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )

            best_idx = np.argmax(scores)
            mask = masks[best_idx]

            y_indices, x_indices = np.where(mask)
            if len(x_indices) > 0:
                x1, y1 = x_indices.min(), y_indices.min()
                x2, y2 = x_indices.max(), y_indices.max()
                bbox = (int(x1), int(y1), int(x2), int(y2))
            else:
                bbox = (0, 0, 0, 0)

            return SegmentationResult(
                mask=mask.astype(np.uint8),
                class_id=1,
                class_name="interactive",
                confidence=float(scores[best_idx]),
                bbox=bbox,
                area=float(mask.sum()),
            )
        else:
            return SegmentationResult(
                mask=np.zeros(image.shape[:2], dtype=np.uint8),
                class_id=0,
                class_name="background",
                confidence=0.0,
                bbox=(0, 0, 0, 0),
            )


class MaskRCNNSegmenter(BaseSegmenter):
    def __init__(
        self,
        backbone: str = "resnet50_fpn",
        num_classes: int = 91,
        box_detections_per_img: int = 100,
        box_nms_thresh: float = 0.5,
        box_score_thresh: float = 0.05,
        device: str = "cpu",
    ):
        self.backbone = backbone
        self.num_classes = num_classes
        self.box_detections_per_img = box_detections_per_img
        self.box_nms_thresh = box_nms_thresh
        self.box_score_thresh = box_score_thresh
        self.device = device
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            import torchvision
            from torchvision.models.detection import maskrcnn_resnet50_fpn
            from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

            self._model = maskrcnn_resnet50_fpn(pretrained=True)

            in_features = self._model.roi_heads.box_predictor.cls_score.in_features
            self._model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

            in_features_mask = self._model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = 256
            self._model.roi_heads.mask_predictor = MaskRCNNPredictor(
                in_features_mask, hidden_layer, self.num_classes
            )

            self._model.eval()
            self._initialized = True
            self._torch = torch

        except ImportError:
            self._initialized = False

    def segment(self, image: np.ndarray) -> Sequence[SegmentationResult]:
        if not self._initialized:
            self._initialize()

        if self._initialized:
            return self._segment_torch(image)
        else:
            return self._segment_classical(image)

    def _segment_torch(self, image: np.ndarray) -> Sequence[SegmentationResult]:
        torch = self._torch

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0

        if self.device != "cpu" and torch.cuda.is_available():
            tensor = tensor.cuda()
            self._model = self._model.cuda()

        with torch.no_grad():
            predictions = self._model(tensor)

        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        masks = pred['masks'].cpu().numpy()

        results = []
        for i in range(len(boxes)):
            if scores[i] < self.box_score_thresh:
                continue

            mask = (masks[i, 0] > 0.5).astype(np.uint8)
            if mask.sum() < 50:
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea) if contours else None
            area = cv2.contourArea(contour) if contour is not None else 0

            x1, y1, x2, y2 = boxes[i].astype(int)
            results.append(SegmentationResult(
                mask=mask,
                class_id=int(labels[i]),
                class_name=f"class_{labels[i]}",
                confidence=float(scores[i]),
                bbox=(x1, y1, x2, y2),
                contour=contour,
                area=float(area),
            ))

        return results

    def _segment_classical(self, image: np.ndarray) -> Sequence[SegmentationResult]:
        segmenter = DeepLabV3PlusSegmenter()
        return segmenter.segment(image)

    def segment_interactive(self, image: np.ndarray, points: np.ndarray, labels: np.ndarray) -> SegmentationResult:
        return SegmentationResult(
            mask=np.zeros(image.shape[:2], dtype=np.uint8),
            class_id=0,
            class_name="background",
            confidence=0.0,
            bbox=(0, 0, 0, 0),
        )


class PanopticSegmenter:
    def __init__(
        self,
        semantic_segmenter: BaseSegmenter,
        instance_segmenter: BaseSegmenter,
        thing_classes: list[int],
        stuff_classes: list[int],
    ):
        self.semantic_segmenter = semantic_segmenter
        self.instance_segmenter = instance_segmenter
        self.thing_classes = set(thing_classes)
        self.stuff_classes = set(stuff_classes)

    def segment(self, image: np.ndarray) -> PanopticResult:
        semantic_results = self.semantic_segmenter.segment(image)
        instance_results = self.instance_segmenter.segment(image)

        h, w = image.shape[:2]
        semantic_map = np.zeros((h, w), dtype=np.int32)
        instance_map = np.zeros((h, w), dtype=np.int32)
        segments_info = []

        for result in semantic_results:
            class_id = result.class_id
            if class_id in self.stuff_classes:
                semantic_map[result.mask > 0] = class_id
                segments_info.append({
                    'id': len(segments_info) + 1,
                    'category_id': class_id,
                    'area': int(result.area),
                    'bbox': list(result.bbox),
                    'isthing': False,
                })

        for result in instance_results:
            class_id = result.class_id
            if class_id in self.thing_classes:
                instance_id = len(segments_info) + 1
                instance_map[result.mask > 0] = instance_id
                semantic_map[result.mask > 0] = class_id
                segments_info.append({
                    'id': instance_id,
                    'category_id': class_id,
                    'area': int(result.area),
                    'bbox': list(result.bbox),
                    'isthing': True,
                })

        return PanopticResult(
            semantic_map=semantic_map,
            instance_map=instance_map,
            segments_info=segments_info,
        )


class SegmentationPostProcessor:
    def __init__(
        self,
        min_area: float = 100.0,
        max_area: Optional[float] = None,
        nms_threshold: float = 0.5,
        score_threshold: float = 0.3,
        fill_holes: bool = True,
        remove_small_components: bool = True,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.fill_holes = fill_holes
        self.remove_small_components = remove_small_components

    def process(self, results: Sequence[SegmentationResult]) -> Sequence[SegmentationResult]:
        filtered = self._filter_by_score(results)
        filtered = self._filter_by_area(filtered)
        filtered = self._apply_nms(filtered)
        filtered = self._post_process_masks(filtered)
        return filtered

    def _filter_by_score(self, results: Sequence[SegmentationResult]) -> list[SegmentationResult]:
        return [r for r in results if r.confidence >= self.score_threshold]

    def _filter_by_area(self, results: Sequence[SegmentationResult]) -> list[SegmentationResult]:
        filtered = []
        for r in results:
            if r.area < self.min_area:
                continue
            if self.max_area is not None and r.area > self.max_area:
                continue
            filtered.append(r)
        return filtered

    def _apply_nms(self, results: Sequence[SegmentationResult]) -> list[SegmentationResult]:
        if not results:
            return []

        boxes = np.array([r.bbox for r in results])
        scores = np.array([r.confidence for r in results])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            order = order[np.where(iou <= 0.5)[0] + 1]

        return [results[i] for i in keep]

    def _post_process_masks(self, results: Sequence[SegmentationResult]) -> list[SegmentationResult]:
        processed = []
        for r in results:
            mask = r.mask.copy()

            if self.fill_holes:
                mask = self._fill_holes(mask)

            if self.remove_small_components:
                mask = self._remove_small_components(mask)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour = max(contours, key=cv2.contourArea) if contours else None
            area = cv2.contourArea(contour) if contour is not None else 0
            x, y, w, h = cv2.boundingRect(contour) if contour is not None else (0, 0, 0, 0)

            processed.append(SegmentationResult(
                mask=mask,
                class_id=r.class_id,
                class_name=r.class_name,
                confidence=r.confidence,
                bbox=(x, y, x + w, y + h),
                contour=contour,
                area=float(area),
                mask_iou=r.mask_iou,
            ))

        return processed

    def _fill_holes(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        filled = mask.copy()
        flood_filled = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(filled, flood_filled, (0, 0), 255)
        filled_inv = cv2.bitwise_not(filled)
        return mask | filled_inv

    def _remove_small_components(self, mask: np.ndarray, min_size: int = 50) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        output = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                output[labels == i] = 255
        return output


class CRFPostProcessor:
    def __init__(
        self,
        iter_max: int = 10,
        pos_xy_std: float = 3.0,
        pos_w: float = 3.0,
        bi_xy_std: float = 80.0,
        bi_rgb_std: float = 13.0,
        bi_w: float = 10.0,
    ):
        self.iter_max = iter_max
        self.pos_xy_std = pos_xy_std
        self.pos_w = pos_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std
        self.bi_w = bi_w
        self._crf = None
        self._initialized = False

    def _initialize(self, h: int, w: int, n_labels: int):
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian

            self._crf = dcrf.DenseCRF2D(w, h, n_labels)
            self._initialized = True

        except ImportError:
            self._initialized = False

    def refine(self, image: np.ndarray, probs: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        n_labels = probs.shape[0]

        if not self._initialized:
            self._initialize(h, w, n_labels)

        if self._initialized:
            return self._refine_crf(image, probs)
        else:
            return self._refine_classical(probs)

    def _refine_crf(self, image: np.ndarray, probs: np.ndarray) -> np.ndarray:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian

        unary = -np.log(probs + 1e-8)
        unary = unary.reshape(probs.shape[0], -1)
        self._crf.setUnaryEnergy(unary)

        feats = create_pairwise_gaussian(sdims=(self.pos_xy_std, self.pos_xy_std), shape=(h, w))
        self._crf.addPairwiseEnergy(feats, compat=self.pos_w,
                                     kernel=dcrf.DIAG_KERNEL,
                                     normalization=dcrf.NORMALIZE_SYMMETRIC)

        feats = create_pairwise_bilateral(
            sdims=(self.bi_xy_std, self.bi_xy_std),
            schan=(self.bi_rgb_std, self.bi_rgb_std, self.bi_rgb_std),
            img=image, chdim=2
        )
        self._crf.addPairwiseEnergy(feats, compat=self.bi_w,
                                     kernel=dcrf.DIAG_KERNEL,
                                     normalization=dcrf.NORMALIZE_SYMMETRIC)

        Q = self._crf.inference(self.iter_max)
        return np.array(Q).reshape((probs.shape[0], h, w))

    def _refine_classical(self, probs: np.ndarray) -> np.ndarray:
        refined = np.zeros_like(probs)
        for i in range(probs.shape[0]):
            refined[i] = cv2.bilateralFilter(probs[i].astype(np.float32), 9, 75, 75)
        return refined