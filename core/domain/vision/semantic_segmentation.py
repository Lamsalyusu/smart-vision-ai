from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
import math
import time
from collections import defaultdict


class SegmentationType(Enum):
    SEMANTIC = "semantic"
    INSTANCE = "instance"
    PANOPTIC = "panoptic"
    SALIENCY = "saliency"
    MEDICAL = "medical"
    AERIAL = "aerial"
    SCENE_PARSING = "scene_parsing"


class ModelArchitecture(Enum):
    UNET = "unet"
    DEEPLAB = "deeplab"
    PSPNET = "pspnet"
    FCN = "fcn"
    SEGNET = "segnet"
    ENET = "enet"
    ICNET = "icnet"
    BISENET = "bisenet"
    HRNET = "hrnet"
    SWIN_SEG = "swin_seg"
    MASK2FORMER = "mask2former"
    ONE_FORMER = "one_former"


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    num_classes: int = 21
    ignore_label: int = 255
    input_size: Tuple[int, int] = (512, 512)
    backbone: str = "resnet50"
    pretrained: bool = True
    output_stride: int = 16
    aspp_rates: Tuple[int, ...] = (6, 12, 18)
    decoder_channels: int = 256


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    segmentation: np.ndarray
    confidence: np.ndarray
    class_probabilities: Optional[np.ndarray] = None
    instance_masks: Optional[List[np.ndarray]] = None
    instance_scores: Optional[List[float]] = None
    instance_classes: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSegmentationModel(ABC):
    def __init__(self, config: SegmentationConfig):
        self.config = config
        self._model = None

    @abstractmethod
    def predict(self, image: np.ndarray) -> SegmentationResult:
        pass

    @abstractmethod
    def predict_batch(self, images: List[np.ndarray]) -> List[SegmentationResult]:
        pass


class UNetSegmentation(BaseSegmentationModel):
    def __init__(
        self,
        config: SegmentationConfig,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16)
    ):
        super().__init__(config)
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.decoder_channels = decoder_channels

    def predict(self, image: np.ndarray) -> SegmentationResult:
        return self._forward(image)

    def predict_batch(self, images: List[np.ndarray]) -> List[SegmentationResult]:
        return [self.predict(img) for img in images]

    def _forward(self, image: np.ndarray) -> SegmentationResult:
        h, w = image.shape[:2]

        logits = np.random.randn(self.config.num_classes, h, w).astype(np.float32)
        probs = self._softmax(logits, axis=0)
        segmentation = probs.argmax(axis=0).astype(np.uint8)
        confidence = probs.max(axis=0)

        return SegmentationResult(
            segmentation=segmentation,
            confidence=confidence,
            class_probabilities=probs
        )

    def _softmax(self, x: np.ndarray, axis: int = 0) -> np.ndarray:
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class DeepLabV3Plus(BaseSegmentationModel):
    def __init__(
        self,
        config: SegmentationConfig,
        backbone: str = "xception",
        output_stride: int = 16,
        aspp_dilations: Tuple[int, ...] = (6, 12, 18)
    ):
        super().__init__(config)
        self.backbone = backbone
        self.output_stride = output_stride
        self.aspp_dilations = aspp_dilations

    def predict(self, image: np.ndarray) -> SegmentationResult:
        return self._forward(image)

    def predict_batch(self, images: List[np.ndarray]) -> List[SegmentationResult]:
        return [self.predict(img) for img in images]

    def _forward(self, image: np.ndarray) -> SegmentationResult:
        h, w = image.shape[:2]

        logits = np.random.randn(self.config.num_classes, h, w).astype(np.float32)
        probs = self._softmax(logits, axis=0)
        segmentation = probs.argmax(axis=0).astype(np.uint8)
        confidence = probs.max(axis=0)

        return SegmentationResult(
            segmentation=segmentation,
            confidence=confidence,
            class_probabilities=probs
        )

    def _softmax(self, x: np.ndarray, axis: int = 0) -> np.ndarray:
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class PanopticSegmentation:
    def __init__(
        self,
        semantic_model: BaseSegmentationModel,
        instance_model: BaseSegmentationModel,
        thing_classes: List[int],
        stuff_classes: List[int],
        overlap_threshold: float = 0.5
    ):
        self.semantic_model = semantic_model
        self.instance_model = instance_model
        self.thing_classes = set(thing_classes)
        self.stuff_classes = set(stuff_classes)
        self.overlap_threshold = overlap_threshold

    def predict(self, image: np.ndarray) -> SegmentationResult:
        semantic_result = self.semantic_model.predict(image)
        instance_result = self.instance_model.predict(image)

        panoptic = self._combine(semantic_result, instance_result)

        return SegmentationResult(
            segmentation=panoptic,
            confidence=semantic_result.confidence,
            instance_masks=instance_result.instance_masks,
            instance_scores=instance_result.instance_scores,
            instance_classes=instance_result.instance_classes
        )

    def _combine(
        self,
        semantic: SegmentationResult,
        instance: SegmentationResult
    ) -> np.ndarray:
        h, w = semantic.segmentation.shape
        panoptic = np.zeros((h, w), dtype=np.int32)
        next_id = 1

        if instance.instance_masks:
            for i, mask in enumerate(instance.instance_masks):
                class_id = instance.instance_classes[i] if instance.instance_classes else 0
                score = instance.instance_scores[i] if instance.instance_scores else 1.0

                if class_id in self.thing_classes and score > 0.5:
                    panoptic[mask > 0.5] = next_id
                    next_id += 1

        for class_id in self.stuff_classes:
            mask = semantic.segmentation == class_id
            if np.any(mask):
                panoptic[mask] = next_id
                next_id += 1

        return panoptic


class SegmentationPostProcessor:
    @staticmethod
    def crf_refinement(
        image: np.ndarray,
        probabilities: np.ndarray,
        iterations: int = 5
    ) -> np.ndarray:
        return probabilities.argmax(axis=0).astype(np.uint8)

    @staticmethod
    def morphological_postprocess(
        segmentation: np.ndarray,
        kernel_size: int = 3,
        operations: List[str] = None
    ) -> np.ndarray:
        operations = operations or ['close', 'open']
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        result = segmentation.copy()

        for op in operations:
            if op == 'close':
                result = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            elif op == 'open':
                result = cv2.morphologyEx(result.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            elif op == 'dilate':
                result = cv2.dilate(result.astype(np.uint8), kernel)
            elif op == 'erode':
                result = cv2.erode(result.astype(np.uint8), kernel)

        return result

    @staticmethod
    def remove_small_components(
        segmentation: np.ndarray,
        min_size: int = 100
    ) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            segmentation.astype(np.uint8), connectivity=8
        )

        result = np.zeros_like(segmentation)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                result[labels == i] = segmentation[labels == i][0]

        return result

    @staticmethod
    def fill_holes(
        segmentation: np.ndarray,
        max_hole_size: int = 50
    ) -> np.ndarray:
        result = segmentation.copy()
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (segmentation == 0).astype(np.uint8), connectivity=8
        )

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] <= max_hole_size:
                result[labels == i] = segmentation[labels != i][0]

        return result


class SegmentationTrainer:
    def __init__(
        self,
        model: BaseSegmentationModel,
        loss_function: str = "cross_entropy",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        class_weights: Optional[np.ndarray] = None
    ):
        self.model = model
        self.loss_function = loss_function
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.class_weights = class_weights

    def train_step(self, images: np.ndarray, masks: np.ndarray) -> Dict[str, float]:
        return {"loss": 0.0, "lr": self.lr}

    def validate(self, images: np.ndarray, masks: np.ndarray) -> Dict[str, float]:
        return {"val_loss": 0.0}


class SegmentationMetrics:
    @staticmethod
    def iou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
        ious = []
        for c in range(num_classes):
            pred_c = (pred == c)
            target_c = (target == c)
            intersection = np.logical_and(pred_c, target_c).sum()
            union = np.logical_or(pred_c, target_c).sum()
            iou = intersection / (union + 1e-6)
            ious.append(iou)
        return np.array(ious)

    @staticmethod
    def miou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> float:
        ious = SegmentationMetrics.iou(pred, target, num_classes)
        return float(np.nanmean(ious))

    @staticmethod
    def pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
        return float(np.mean(pred == target))

    @staticmethod
    def class_accuracy(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
        accs = []
        for c in range(num_classes):
            target_c = (target == c)
            if target_c.sum() > 0:
                pred_c = (pred == c)
                acc = np.logical_and(pred_c, target_c).sum() / target_c.sum()
                accs.append(acc)
            else:
                accs.append(0.0)
        return np.array(accs)

    @staticmethod
    def fw_iou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> float:
        ious = SegmentationMetrics.iou(pred, target, num_classes)
        freq = np.bincount(target.flatten(), minlength=num_classes) / target.size
        return float(np.sum(freq * ious))

    @staticmethod
    def boundary_iou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> float:
        return 0.0


class SegmentationVisualizer:
    @staticmethod
    def colorize(segmentation: np.ndarray, palette: List[Tuple[int, int, int]] = None) -> np.ndarray:
        if palette is None:
            palette = SegmentationVisualizer._generate_palette(256)

        h, w = segmentation.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        for i, color in enumerate(palette):
            colored[segmentation == i] = color

        return colored

    @staticmethod
    def _generate_palette(num_classes: int) -> List[Tuple[int, int, int]]:
        np.random.seed(42)
        palette = [(0, 0, 0)]
        for _ in range(1, num_classes):
            palette.append(tuple(np.random.randint(0, 255, 3).tolist()))
        return palette

    @staticmethod
    def overlay(image: np.ndarray, segmentation: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        colored = SegmentationVisualizer.colorize(segmentation)
        return cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)

    @staticmethod
    def draw_legend(segmentation: np.ndarray, class_names: List[str], palette: List[Tuple[int, int, int]] = None) -> np.ndarray:
        if palette is None:
            palette = SegmentationVisualizer._generate_palette(len(class_names))

        unique_classes = np.unique(segmentation)
        legend_h = len(unique_classes) * 25
        legend = np.zeros((legend_h, 200, 3), dtype=np.uint8)

        for i, class_id in enumerate(unique_classes):
            if class_id < len(class_names):
                color = palette[class_id] if class_id < len(palette) else (255, 255, 255)
                y = i * 25
                cv2.rectangle(legend, (10, y + 5), (30, y + 20), color, -1)
                cv2.putText(legend, class_names[class_id], (40, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return legend


class SegmentationExporter:
    @staticmethod
    def export_coco(
        segmentation: np.ndarray,
        image_id: int,
        category_id: int,
        score: float = 1.0
    ) -> Dict[str, Any]:
        from pycocotools import mask as mask_utils
        import json

        rle = mask_utils.encode(np.asfortranarray((segmentation == category_id).astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('utf-8')

        return {
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": rle,
            "score": score
        }

    @staticmethod
    def export_cityscapes(
        segmentation: np.ndarray,
        output_path: str
    ):
        cv2.imwrite(output_path, segmentation.astype(np.uint16))


from core.domain.vision.feature_extraction import FeatureType, Keypoint, FeatureMatch, FeatureSet
