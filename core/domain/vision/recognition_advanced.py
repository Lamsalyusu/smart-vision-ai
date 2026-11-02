from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math
import time


class RecognitionTask(Enum):
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    POSE_ESTIMATION = "pose_estimation"
    ACTION_RECOGNITION = "action_recognition"
    FACE_RECOGNITION = "face_recognition"
    TEXT_RECOGNITION = "text_recognition"
    SCENE_CLASSIFICATION = "scene_classification"
    ATTRIBUTE_PREDICTION = "attribute_prediction"


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    task: RecognitionTask
    labels: List[str]
    confidences: List[float]
    bboxes: Optional[List[Tuple[float, float, float, float]]] = None
    masks: Optional[List[np.ndarray]] = None
    keypoints: Optional[List[np.ndarray]] = None
    embeddings: Optional[np.ndarray] = None
    processing_time_ms: float = 0.0


class BaseRecognizer(ABC):
    @abstractmethod
    def recognize(self, image: np.ndarray) -> RecognitionResult:
        pass

    @abstractmethod
    def recognize_batch(self, images: List[np.ndarray]) -> List[RecognitionResult]:
        pass


class ImageClassifier:
    def __init__(
        self,
        model_path: str = "classifier.pth",
        num_classes: int = 1000,
        input_size: Tuple[int, int] = (224, 224),
        device: str = "cpu",
        top_k: int = 5,
    ):
        self.model_path = model_path
        self.num_classes = num_classes
        self.input_size = input_size
        self.device = device
        self.top_k = top_k
        self._model = None
        self._initialized = False
        self.class_names = [f"class_{i}" for i in range(num_classes)]

    def _initialize(self):
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            self._model = models.resnet50(pretrained=True)
            self._model.eval()
            self._preprocess = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self._initialized = True
            self._torch = torch

        except ImportError:
            self._initialized = False

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        if not self._initialized:
            self._initialize()

        if self._initialized:
            return self._recognize_torch(image)
        else:
            return self._recognize_classical(image)

    def _recognize_torch(self, image: np.ndarray) -> RecognitionResult:
        torch = self._torch

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        input_tensor = self._preprocess(image).unsqueeze(0)

        if self.device != "cpu" and torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            self._model = self._model.cuda()

        with torch.no_grad():
            logits = self._model(input_tensor)
            probs = torch.softmax(logits, dim=1)

        probs = probs.cpu().numpy()[0]
        top_indices = np.argsort(probs)[-self.top_k:][::-1]
        top_probs = probs[top_indices]

        labels = [self.class_names[i] if i < len(self.class_names) else f"class_{i}" for i in top_indices]
        confidences = top_probs.tolist()

        return RecognitionResult(
            task=RecognitionTask.CLASSIFICATION,
            labels=labels,
            confidences=confidences,
            processing_time_ms=0.0,
        )

    def _recognize_classical(self, image: np.ndarray) -> RecognitionResult:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        mean_val = np.mean(gray)
        std_val = np.std(gray)

        class_name = "bright" if mean_val > 128 else "dark"
        confidence = min(abs(mean_val - 128) / 128, 1.0)

        return RecognitionResult(
            task=RecognitionTask.CLASSIFICATION,
            labels=[class_name],
            confidences=[confidence],
            processing_time_ms=0.0,
        )

    def recognize_batch(self, images: List[np.ndarray]) -> List[RecognitionResult]:
        return [self.recognize(img) for img in images]


class ObjectDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 300,
        device: str = "cpu",
        classes: Optional[List[int]] = None,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.device = device
        self.classes = classes
        self._model = None
        self._initialized = False
        self._class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

    def _initialize(self):
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
            self._initialized = True

        except ImportError:
            self._initialized = False

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        if not self._initialized:
            self._initialize()

        if self._initialized:
            return self._detect_ultralytics(image)
        else:
            return self._detect_classical(image)

    def _detect_ultralytics(self, image: np.ndarray) -> RecognitionResult:
        results = self._model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)

            masks = None
            if boxes.masks is not None:
                masks = boxes.masks.data.cpu().numpy()

            keypoints = None
            if boxes.keypoints is not None:
                keypoints = boxes.keypoints.data.cpu().numpy()

            for i in range(len(xyxy)):
                class_id = cls[i]
                class_name = self._class_names[class_id] if class_id < len(self._class_names) else f"class_{class_id}"
                detections.append({
                    'bbox': xyxy[i],
                    'confidence': float(conf[i]),
                    'class_id': class_id,
                    'class_name': class_name,
                    'mask': masks[i] if masks is not None else None,
                    'keypoints': keypoints[i] if keypoints is not None else None,
                })

        return RecognitionResult(
            task=RecognitionTask.DETECTION,
            labels=[d['class_name'] for d in detections],
            confidences=[d['confidence'] for d in detections],
            bboxes=[d['bbox'] for d in detections],
            masks=[d['mask'] for d in detections] if any(d['mask'] is not None for d in detections) else None,
            keypoints=[d['keypoints'] for d in detections] if any(d['keypoints'] is not None for d in detections) else None,
        )

    def _detect_classical(self, image: np.ndarray) -> RecognitionResult:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            detections.append({
                'bbox': [x, y, x + w, y + h],
                'confidence': 0.5,
                'class_id': 0,
                'class_name': 'object',
            })

        return RecognitionResult(
            task=RecognitionTask.DETECTION,
            labels=[d['class_name'] for d in detections],
            confidences=[d['confidence'] for d in detections],
            bboxes=[d['bbox'] for d in detections],
        )

    def recognize_batch(self, images: List[np.ndarray]) -> List[RecognitionResult]:
        return [self.recognize(img) for img in images]


class RTDETRDetector:
    def __init__(
        self,
        model_path: str = "rtdetr_l.pt",
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            from ultralytics import RTDETR

            self._model = RTDETR(self.model_path)
            self._initialized = True

        except ImportError:
            self._initialized = False

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        if not self._initialized:
            self._initialize()

        if self._initialized:
            return self._detect_ultralytics(image)
        else:
            return self._detect_classical(image)

    def _detect_ultralytics(self, image: np.ndarray) -> RecognitionResult:
        results = self._model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(xyxy)):
                detections.append({
                    'bbox': xyxy[i],
                    'confidence': float(conf[i]),
                    'class_id': cls[i],
                    'class_name': f"class_{cls[i]}",
                })

        return RecognitionResult(
            task=RecognitionTask.DETECTION,
            labels=[d['class_name'] for d in detections],
            confidences=[d['confidence'] for d in detections],
            bboxes=[d['bbox'] for d in detections],
        )

    def _detect_classical(self, image: np.ndarray) -> RecognitionResult:
        detector = ObjectDetector()
        return detector._detect_classical(image)

    def recognize_batch(self, images: List[np.ndarray]) -> List[RecognitionResult]:
        return [self.recognize(img) for img in images]


class DETRDetector:
    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        conf_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.device = device
        self._model = None
        self._processor = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            from transformers import DetrImageProcessor, DetrForObjectDetection

            self._processor = DetrImageProcessor.from_pretrained(self.model_name)
            self._model = DetrForObjectDetection.from_pretrained(self.model_name)
            self._model.eval()
            self._initialized = True
            self._torch = torch

        except ImportError:
            self._initialized = False

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        if not self._initialized:
            self._initialize()

        if self._initialized:
            return self._detect_transformers(image)
        else:
            return self._detect_classical(image)

    def _detect_transformers(self, image: np.ndarray) -> RecognitionResult:
        torch = self._torch

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        inputs = self._processor(images=image, return_tensors="pt")

        if self.device != "cpu" and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            self._model = self._model.cuda()

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([image.shape[:2]])
        results = self._processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.conf_threshold
        )[0]

        detections = []
        for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
            detections.append({
                'bbox': box.cpu().numpy(),
                'confidence': float(score),
                'class_id': int(label),
                'class_name': self._model.config.id2label[int(label)],
            })

        return RecognitionResult(
            task=RecognitionTask.DETECTION,
            labels=[d['class_name'] for d in detections],
            confidences=[d['confidence'] for d in detections],
            bboxes=[d['bbox'] for d in detections],
        )

    def _detect_classical(self, image: np.ndarray) -> RecognitionResult:
        detector = ObjectDetector()
        return detector._detect_classical(image)

    def recognize_batch(self, images: List[np.ndarray]) -> List[RecognitionResult]:
        return [self.recognize(img) for img in images]


class FCOSDetector:
    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 80,
        fpn_channels: int = 256,
        strides: List[int] = [8, 16, 32, 64, 128],
        conf_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.backbone = backbone
        self.num_classes = num_classes
        self.fpn_channels = fpn_channels
        self.strides = strides
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class FCOSHead(nn.Module):
                def __init__(self, in_channels, num_classes, num_anchors=1):
                    super().__init__()
                    self.cls_tower = nn.Sequential(
                        *[nn.Sequential(
                            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                            nn.GroupNorm(32, in_channels),
                            nn.ReLU(inplace=True),
                        ) for _ in range(4)]
                    )
                    self.bbox_tower = nn.Sequential(
                        *[nn.Sequential(
                            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                            nn.GroupNorm(32, in_channels),
                            nn.ReLU(inplace=True),
                        ) for _ in range(4)]
                    )
                    self.cls_logits = nn.Conv2d(in_channels, num_classes * num_anchors, 3, padding=1)
                    self.bbox_pred = nn.Conv2d(in_channels, 4 * num_anchors, 3, padding=1)
                    self.centerness = nn.Conv2d(in_channels, num_anchors, 3, padding=1)
                    self.scales = nn.ModuleList([Scale() for _ in range(5)])

                def forward(self, features):
                    cls_logits = []
                    bbox_preds = []
                    centerness = []
                    for i, feat in enumerate(features):
                        cls_tower = self.cls_tower(feat)
                        bbox_tower = self.bbox_tower(feat)
                        cls_logits.append(self.cls_logits(cls_tower))
                        bbox_preds.append(self.scales[i](self.bbox_pred(bbox_tower)))
                        centerness.append(self.centerness(bbox_tower))
                    return cls_logits, bbox_preds, centerness

            class FCOS(nn.Module):
                def __init__(self, backbone, num_classes, fpn_channels):
                    super().__init__()
                    if backbone == "resnet50":
                        from torchvision.models import resnet50
                        resnet = resnet50(pretrained=True)
                        self.backbone = nn.Sequential(
                            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
                        )
                        in_channels_list = [256, 512, 1024, 2048]
                    else:
                        raise ValueError(f"Unsupported backbone: {backbone}")

                    self.fpn = nn.ModuleList([
                        nn.Conv2d(ch, fpn_channels, 1) for ch in in_channels_list
                    ])
                    self.head = FCOSHead(fpn_channels, num_classes)

                def forward(self, x):
                    features = []
                    x = self.backbone[0](x)
                    x = self.backbone[1](x)
                    x = self.backbone[2](x)
                    x = self.backbone[3](x)
                    features.append(self.backbone[4](x))
                    features.append(self.backbone[5](features[-1]))
                    features.append(self.backbone[6](features[-1]))
                    features.append(self.backbone[7](features[-1]))

                    fpn_features = [conv(feat) for conv, feat in zip(self.fpn, features)]
                    return self.head(fpn_features)

            self._model = FCOS(self.backbone, self.num_classes, self.fpn_channels)
            self._model.eval()
            self._initialized = True
            self._torch = torch
            self._F = F

        except ImportError:
            self._initialized = False

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        if not self._initialized:
            self._initialize()

        if self._initialized:
            return self._detect_torch(image)
        else:
            return self._detect_classical(image)

    def _detect_torch(self, image: np.ndarray) -> RecognitionResult:
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
            cls_logits, bbox_preds, centerness = self._model(tensor)

        detections = []
        for level, (cls_logit, bbox_pred, center) in enumerate(zip(cls_logits, bbox_preds, centerness)):
            stride = self.strides[level]
            cls_logit = cls_logit.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            center = center.permute(0, 2, 3, 1).reshape(-1)

            scores = torch.sigmoid(cls_logit)
            center = torch.sigmoid(center)
            scores = scores * center[:, None]

            max_scores, max_labels = scores.max(dim=1)
            keep = max_scores > self.conf_threshold

            if not keep.any():
                continue

            scores = max_scores[keep]
            labels = max_labels[keep]
            bbox_pred = bbox_pred[keep]

            h, w = cls_logit.shape[1], cls_logit.shape[2]
            shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
            shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
            shifts = torch.stack([shift_x, shift_y], dim=-1).reshape(-1, 2)
            shifts = shifts[keep]

            boxes = torch.zeros_like(bbox_pred)
            boxes[:, 0] = shifts[:, 0] - bbox_pred[:, 0]
            boxes[:, 1] = shifts[:, 1] - bbox_pred[:, 1]
            boxes[:, 2] = shifts[:, 0] + bbox_pred[:, 2]
            boxes[:, 3] = shifts[:, 1] + bbox_pred[:, 3]

            boxes = boxes.cpu().numpy()
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()

            keep_nms = self._nms_numpy(boxes, scores, self.nms_threshold)
            for idx in keep_nms:
                detections.append(RecognitionResult(
                    bbox=boxes[idx],
                    confidence=float(scores[idx]),
                    class_id=int(labels[idx]),
                    class_name=f"class_{labels[idx]}",
                ))

        return detections

    def _nms_numpy(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list:
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
            order = order[np.where(iou <= self.nms_threshold)[0] + 1]
        return keep

    def _detect_classical(self, image: np.ndarray) -> RecognitionResult:
        detector = ObjectDetector()
        return detector._detect_classical(image)

    def recognize_batch(self, images: List[np.ndarray]) -> List[RecognitionResult]:
        return [self.recognize(img) for img in images]


class CenterNetDetector:
    def __init__(
        self,
        backbone: str = "dla34",
        num_classes: int = 80,
        head_channels: int = 64,
        conf_threshold: float = 0.3,
        device: str = "cpu",
    ):
        self.backbone = backbone
        self.num_classes = num_classes
        self.head_channels = head_channels
        self.conf_threshold = conf_threshold
        self.device = device
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class CenterNetHead(nn.Module):
                def __init__(self, in_channels, num_classes, head_channels):
                    super().__init__()
                    self.hm = nn.Sequential(
                        nn.Conv2d(in_channels, head_channels, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(head_channels, num_classes, 1),
                    )
                    self.wh = nn.Sequential(
                        nn.Conv2d(in_channels, head_channels, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(head_channels, 2, 1),
                    )
                    self.reg = nn.Sequential(
                        nn.Conv2d(in_channels, head_channels, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(head_channels, 2, 1),
                    )

                def forward(self, x):
                    return {
                        'hm': self.hm(x),
                        'wh': self.wh(x),
                        'reg': self.reg(x),
                    }

            class CenterNet(nn.Module):
                def __init__(self, backbone, num_classes, head_channels):
                    super().__init__()
                    if backbone == "dla34":
                        self.backbone = self._build_dla34()
                        in_channels = 64
                    elif backbone == "resnet18":
                        from torchvision.models import resnet18
                        resnet = resnet18(pretrained=True)
                        self.backbone = nn.Sequential(
                            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
                        )
                        in_channels = 512
                    else:
                        raise ValueError(f"Unsupported backbone: {backbone}")

                    self.head = CenterNetHead(in_channels, num_classes, head_channels)

                def _build_dla34(self):
                    return nn.Sequential(
                        nn.Conv2d(3, 16, 7, 2, 3, bias=False),
                        nn.BatchNorm2d(16),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(3, 2, 1),
                        *[self._make_layer(16, 32, 1, 2)],
                        *[self._make_layer(32, 64, 1, 2)],
                        *[self._make_layer(64, 128, 1, 2)],
                        *[self._make_layer(128, 256, 1, 2)],
                        *[self._make_layer(256, 512, 1, 2)],
                    )

                def _make_layer(self, in_ch, out_ch, blocks, stride):
                    layers = []
                    layers.append(nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True),
                    ))
                    for _ in range(blocks - 1):
                        layers.append(nn.Sequential(
                            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
                            nn.BatchNorm2d(out_ch),
                            nn.ReLU(inplace=True),
                        ))
                    return nn.Sequential(*layers)

                def forward(self, x):
                    feat = self.backbone(x)
                    return self.head(feat)

            self._model = CenterNet(self.backbone, self.num_classes, self.head_channels)
            self._model.eval()
            self._initialized = True
            self._torch = torch
            self._F = F

        except ImportError:
            self._initialized = False

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        if not self._initialized:
            self._initialize()

        if self._initialized:
            return self._detect_torch(image)
        else:
            return self._detect_classical(image)

    def _detect_torch(self, image: np.ndarray) -> RecognitionResult:
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
            outputs = self._model(tensor)

        hm = outputs['hm'].sigmoid_()
        wh = outputs['wh']
        reg = outputs['reg']

        hm = hm.cpu().numpy()[0]
        wh = wh.cpu().numpy()[0]
        reg = reg.cpu().numpy()[0]

        detections = self._decode(hm, wh, reg)
        return detections

    def _decode(self, hm: np.ndarray, wh: np.ndarray, reg: np.ndarray) -> list:
        h, w = hm.shape[1:]
        detections = []

        for c in range(self.num_classes):
            heatmap = hm[c]
            peaks = self._find_peaks(heatmap)

            for y, x in peaks:
                score = heatmap[y, x]
                if score < self.conf_threshold:
                    continue

                rx, ry = reg[:, y, x]
                w_pred, h_pred = wh[:, y, x]

                cx = x + rx
                cy = y + ry

                x1 = (cx - w_pred / 2) * 4
                y1 = (cy - h_pred / 2) * 4
                x2 = (cx + w_pred / 2) * 4
                y2 = (cy + h_pred / 2) * 4

                detections.append(RecognitionResult(
                    bbox=np.array([x1, y1, x2, y2]),
                    confidence=float(score),
                    class_id=c,
                    class_name=f"class_{c}",
                ))

        return detections

    def _find_peaks(self, heatmap: np.ndarray, kernel: int = 3) -> list:
        h, w = heatmap.shape
        pad = kernel // 2
        padded = np.pad(heatmap, pad, mode='constant')
        peaks = []

        for y in range(h):
            for x in range(w):
                patch = padded[y:y+kernel, x:x+kernel]
                if heatmap[y, x] == patch.max() and heatmap[y, x] > 0:
                    peaks.append((y, x))

        return peaks

    def _detect_classical(self, image: np.ndarray) -> RecognitionResult:
        detector = ObjectDetector()
        return detector._detect_classical(image)

    def recognize_batch(self, images: List[np.ndarray]) -> List[RecognitionResult]:
        return [self.recognize(img) for img in images]


class RecognitionEnsemble:
    def __init__(
        self,
        detectors: List,
        weights: Optional[List[float]] = None,
        iou_threshold: float = 0.5,
        use_wbf: bool = True,
    ):
        self.detectors = detectors
        self.weights = weights or [1.0] * len(detectors)
        self.iou_threshold = iou_threshold
        self.use_wbf = use_wbf

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        all_detections = []
        for detector, weight in zip(self.detectors, self.weights):
            detections = detector.recognize(image)
            for det in detections.detections:
                all_detections.append(det)

        if self.use_wbf:
            return self._weighted_boxes_fusion(all_detections)
        else:
            return self._nms(all_detections)

    def _weighted_boxes_fusion(self, detections: list) -> RecognitionResult:
        if not detections:
            return RecognitionResult(task=RecognitionTask.DETECTION, labels=[], confidences=[], bboxes=[])

        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])
        labels = np.array([d.class_id for d in detections])

        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]
        labels = labels[order]

        fused = []
        used = np.zeros(len(boxes), dtype=bool)

        for i in range(len(boxes)):
            if used[i]:
                continue

            cluster_boxes = [boxes[i]]
            cluster_scores = [scores[i]]
            cluster_labels = [labels[i]]
            used[i] = True

            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue

                iou = self._iou(boxes[i], boxes[j])
                if iou > self.iou_threshold and labels[i] == labels[j]:
                    cluster_boxes.append(boxes[j])
                    cluster_scores.append(scores[j])
                    cluster_labels.append(labels[j])
                    used[j] = True

            cluster_boxes = np.array(cluster_boxes)
            cluster_scores = np.array(cluster_scores)
            cluster_labels = np.array(cluster_labels)

            weights = cluster_scores / cluster_scores.sum()
            fused_box = np.sum(cluster_boxes * weights[:, None], axis=0)
            fused_score = np.mean(cluster_scores)
            fused_label = cluster_labels[0]

            fused.append(RecognitionResult(
                bbox=fused_box,
                confidence=float(fused_score),
                class_id=int(fused_label),
                class_name=f"class_{fused_label}",
            ))

        return fused

    def _nms(self, detections: list) -> List[Detection]:
        if not detections:
            return []

        boxes = np.array([d.xyxy for d in detections])
        scores = np.array([d.confidence for d in detections])
        labels = np.array([d.class_id for d in detections])

        keep = []
        for label in np.unique(labels):
            mask = labels == label
            label_boxes = boxes[mask]
            label_scores = scores[mask]
            label_indices = np.where(mask)[0]

            order = label_scores.argsort()[::-1]
            label_boxes = label_boxes[order]
            label_scores = label_scores[order]
            label_indices = label_indices[order]

            while len(label_boxes) > 0:
                i = 0
                keep.append(label_indices[i])
                if len(label_boxes) == 1:
                    break

                ious = self._iou_batch(label_boxes[i], label_boxes[1:])
                label_boxes = label_boxes[1:][ious <= self.iou_threshold]
                label_scores = label_scores[1:][ious <= self.iou_threshold]
                label_indices = label_indices[1:][ious <= self.iou_threshold]

        return [detections[i] for i in keep]

    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return inter / (area1 + area2 - inter + 1e-6)

    def _iou_batch(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        return inter / (area1 + area2 - inter + 1e-6)


class DetectionPostProcessor:
    def __init__(
        self,
        conf_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        max_detections: int = 300,
        class_agnostic_nms: bool = False,
    ):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.class_agnostic_nms = class_agnostic_nms

    def process(self, detections: Sequence[Detection]) -> Sequence[Detection]:
        filtered = [d for d in detections if d.confidence >= self.conf_threshold]
        filtered = self._apply_nms(filtered)
        filtered = filtered[:self.max_detections]
        return filtered

    def _apply_nms(self, detections: list) -> list:
        if not detections:
            return []

        if self.class_agnostic_nms:
            boxes = np.array([d.xyxy for d in detections])
            scores = np.array([d.confidence for d in detections])
            keep = self._nms_numpy(boxes, scores, self.nms_threshold)
            return [detections[i] for i in keep]

        boxes = np.array([d.xyxy for d in detections])
        scores = np.array([d.confidence for d in detections])
        labels = np.array([d.class_id for d in detections])

        keep = []
        for label in np.unique(labels):
            mask = labels == label
            label_boxes = boxes[mask]
            label_scores = scores[mask]
            label_indices = np.where(mask)[0]

            order = label_scores.argsort()[::-1]
            label_boxes = label_boxes[order]
            label_scores = label_scores[order]
            label_indices = label_indices[order]

            while len(label_boxes) > 0:
                i = 0
                keep.append(label_indices[i])
                if len(label_boxes) == 1:
                    break

                ious = self._iou_batch(label_boxes[i], label_boxes[1:])
                label_boxes = label_boxes[1:][ious <= self.nms_threshold]
                label_scores = label_scores[1:][ious <= self.nms_threshold]
                label_indices = label_indices[1:][ious <= self.nms_threshold]

        return [detections[i] for i in keep]

    def _nms_numpy(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list:
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
            order = order[np.where(iou <= threshold)[0] + 1]
        return keep

    def _iou_batch(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        return inter / (area1 + area2 - inter + 1e-6)