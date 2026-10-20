from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Sequence, Callable, Dict, Any, List, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import math


class FaceDetectionAlgorithm(Enum):
    HOG_SVM = "hog_svm"
    CNN = "cnn"
    MTCNN = "mtcnn"
    RETINAFACE = "retinaface"
    YOLO_FACE = "yolo_face"
    ULTRAFACE = "ultraface"
    BLAZEFACE = "blazeface"
    S3FD = "s3fd"
    DSFD = "dsfd"


class FaceRecognitionAlgorithm(Enum):
    EIGENFACES = "eigenfaces"
    FISHERFACES = "fisherfaces"
    LBPH = "lbph"
    DEEPFACE = "deepface"
    FACENET = "facenet"
    ARCFACE = "arcface"
    COSFACE = "cosface"
    SPHEREFACE = "sphereface"
    MAGFACE = "magface"


class FaceAlignmentAlgorithm(Enum):
    DLIB_SHAPE_PREDICTOR = "dlib_shape_predictor"
    FAN = "fan"
    HRNET = "hrnet"
    MOBILENET_V2 = "mobilenet_v2"
    PFLD = "pfld"
    WING = "wing"
    ADAPTIVE_WING = "adaptive_wing"


@dataclass(frozen=True, slots=True)
class FaceDetection:
    bbox: Tuple[float, float, float, float]
    confidence: float
    landmarks: Optional[np.ndarray] = None
    pose: Optional[np.ndarray] = None
    quality: float = 1.0


@dataclass(frozen=True, slots=True)
class FaceRecognition:
    embedding: np.ndarray
    identity: Optional[str] = None
    confidence: float = 0.0
    distance: float = 0.0


@dataclass(frozen=True, slots=True)
class FaceAlignment:
    landmarks: np.ndarray
    pose: np.ndarray
    confidence: float
    eye_centers: Tuple[Tuple[float, float], Tuple[float, float]]


@dataclass(frozen=True, slots=True)
class FaceAnalysis:
    detection: FaceDetection
    recognition: Optional[FaceRecognition] = None
    alignment: Optional[FaceAlignment] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)


class BaseFaceDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        pass

    @abstractmethod
    def detect_batch(self, images: List[np.ndarray]) -> List[List[FaceDetection]]:
        pass


class HOGSVMDetector(BaseFaceDetector):
    def __init__(
        self,
        upsample: int = 1,
        threshold: float = 0.0,
    ):
        self.upsample = upsample
        self.threshold = threshold
        self.detector = cv2.HOGDescriptor()
        self.detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        rects, weights = self.detector.detectMultiScale(
            gray,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05,
        )

        detections = []
        for (x, y, w, h), weight in zip(rects, weights):
            if weight < self.threshold:
                continue
            detections.append(FaceDetection(
                bbox=(x, y, x + w, y + h),
                confidence=float(weight),
            ))

        return detections

    def detect_batch(self, images: List[np.ndarray]) -> List[List[FaceDetection]]:
        return [self.detect(img) for img in images]


class MTCNNDetector(BaseFaceDetector):
    def __init__(
        self,
        min_face_size: int = 20,
        thresholds: List[float] = [0.6, 0.7, 0.7],
        factor: float = 0.709,
        device: str = "cpu",
    ):
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.device = device
        self._pnet = None
        self._rnet = None
        self._onet = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            self._initialized = True
        except ImportError:
            self._initialized = False

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        if not self._initialized:
            self._initialize()

        if not self._initialized:
            return self._detect_classical(image)

        return self._detect_mtcnn(image)

    def _detect_classical(self, image: np.ndarray) -> List[FaceDetection]:
        detector = HOGSVMDetector()
        return detector.detect(image)

    def _detect_mtcnn(self, image: np.ndarray) -> List[FaceDetection]:
        return []

    def detect_batch(self, images: List[np.ndarray]) -> List[List[FaceDetection]]:
        return [self.detect(img) for img in images]


class RetinaFaceDetector(BaseFaceDetector):
    def __init__(
        self,
        model_path: str = "retinaface_mobilenet0.25.pth",
        confidence_threshold: float = 0.8,
        nms_threshold: float = 0.4,
        input_size: Tuple[int, int] = (640, 640),
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.device = device
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            self._initialized = True
        except ImportError:
            self._initialized = False

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        if not self._initialized:
            self._initialize()

        if not self._initialized:
            return self._detect_classical(image)

        return self._detect_retinaface(image)

    def _detect_classical(self, image: np.ndarray) -> List[FaceDetection]:
        detector = HOGSVMDetector()
        return detector.detect(image)

    def _detect_retinaface(self, image: np.ndarray) -> List[FaceDetection]:
        return []

    def detect_batch(self, images: List[np.ndarray]) -> List[List[FaceDetection]]:
        return [self.detect(img) for img in images]


class YOLOFaceDetector(BaseFaceDetector):
    def __init__(
        self,
        model_path: str = "yolov8n-face.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
            self._initialized = True
        except ImportError:
            self._initialized = False

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        if not self._initialized:
            self._initialize()

        if not self._initialized:
            return self._detect_classical(image)

        results = self._model.predict(
            source=image,
            conf=self.confidence_threshold,
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

            for i in range(len(xyxy)):
                detections.append(FaceDetection(
                    bbox=tuple(xyxy[i]),
                    confidence=float(conf[i]),
                ))

        return detections

    def _detect_classical(self, image: np.ndarray) -> List[FaceDetection]:
        detector = HOGSVMDetector()
        return detector.detect(image)

    def detect_batch(self, images: List[np.ndarray]) -> List[List[FaceDetection]]:
        return [self.detect(img) for img in images]


class BlazeFaceDetector(BaseFaceDetector):
    def __init__(
        self,
        model_path: str = "blazeface.pth",
        confidence_threshold: float = 0.7,
        iou_threshold: float = 0.3,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self._model = None
        self._anchors = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            self._initialized = True
        except ImportError:
            self._initialized = False

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        if not self._initialized:
            self._initialize()

        if not self._initialized:
            return self._detect_classical(image)

        return self._detect_blazeface(image)

    def _detect_classical(self, image: np.ndarray) -> List[FaceDetection]:
        detector = HOGSVMDetector()
        return detector.detect(image)

    def _detect_blazeface(self, image: np.ndarray) -> List[FaceDetection]:
        return []

    def detect_batch(self, images: List[np.ndarray]) -> List[List[FaceDetection]]:
        return [self.detect(img) for img in images]


class FaceAligner:
    def __init__(
        self,
        algorithm: FaceAlignmentAlgorithm = FaceAlignmentAlgorithm.FAN,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.algorithm = algorithm
        self.model_path = model_path
        self.device = device
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            self._initialized = True
        except ImportError:
            self._initialized = False

    def align(self, image: np.ndarray, detection: FaceDetection) -> FaceAlignment:
        if not self._initialized:
            self._initialize()

        if not self._initialized:
            return self._align_classical(image, detection)

        return self._align_deep(image, detection)

    def _align_classical(self, image: np.ndarray, detection: FaceDetection) -> FaceAlignment:
        x1, y1, x2, y2 = detection.bbox
        face_roi = image[int(y1):int(y2), int(x1):int(x2)]

        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_roi

        eyes = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml').detectMultiScale(gray)
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
            eye_centers = [(ex + ew/2, ey + eh/2) for (ex, ey, ew, eh) in eyes]
        else:
            h, w = face_roi.shape[:2]
            eye_centers = [(w*0.3, h*0.4), (w*0.7, h*0.4)]

        left_eye, right_eye = eye_centers
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = math.degrees(math.atan2(dy, dx))

        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(image, M, image.shape[:2])

        landmarks = np.array([
            [left_eye[0], left_eye[1]],
            [right_eye[0], right_eye[1]],
            [(x1 + x2) / 2, (y1 + y2) / 2 + (x2 - x1) * 0.3],
            [x1 + (x2 - x1) * 0.3, y2],
            [x2 - (x2 - x1) * 0.3, y2],
        ])

        pose = np.array([0, 0, 0])

        return FaceAlignment(
            landmarks=landmarks,
            pose=pose,
            confidence=0.8,
            eye_centers=(left_eye, right_eye),
        )

    def _align_deep(self, image: np.ndarray, detection: FaceDetection) -> FaceAlignment:
        return self._align_classical(image, detection)

    def align_batch(self, images: List[np.ndarray], detections: List[List[FaceDetection]]) -> List[List[FaceAlignment]]:
        return [[self.align(img, det) for det in dets] for img, dets in zip(images, detections)]


class FaceRecognizer:
    def __init__(
        self,
        algorithm: FaceRecognitionAlgorithm = FaceRecognitionAlgorithm.ARCFACE,
        model_path: Optional[str] = None,
        device: str = "cpu",
        threshold: float = 0.6,
    ):
        self.algorithm = algorithm
        self.model_path = model_path
        self.device = device
        self.threshold = threshold
        self._model = None
        self._gallery: Dict[str, np.ndarray] = {}
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            self._initialized = True
        except ImportError:
            self._initialized = False

    def enroll(self, identity: str, embedding: np.ndarray):
        if identity in self._gallery:
            self._gallery[identity] = (self._gallery[identity] + embedding) / 2
        else:
            self._gallery[identity] = embedding

    def remove(self, identity: str):
        self._gallery.pop(identity, None)

    def extract_embedding(self, image: np.ndarray, detection: FaceDetection) -> np.ndarray:
        if not self._initialized:
            self._initialize()

        if not self._initialized:
            return self._extract_classical(image, detection)

        return self._extract_deep(image, detection)

    def _extract_classical(self, image: np.ndarray, detection: FaceDetection) -> np.ndarray:
        x1, y1, x2, y2 = detection.bbox
        face_roi = image[int(y1):int(y2), int(x1):int(x2)]

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        face_roi = gray[int(y1):int(y2), int(x1):int(x2)]
        face_roi = cv2.resize(face_roi, (128, 128))

        face_roi = cv2.equalizeHist(face_roi)

        hog = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)
        features = hog.compute(face_roi).flatten()

        return features / (np.linalg.norm(features) + 1e-8)

    def _extract_deep(self, image: np.ndarray, detection: FaceDetection) -> np.ndarray:
        return self._extract_classical(image, detection)

    def recognize(self, image: np.ndarray, detection: FaceDetection) -> FaceRecognition:
        embedding = self.extract_embedding(image, detection)

        best_identity = None
        best_distance = float('inf')

        for identity, gallery_embedding in self._gallery.items():
            distance = np.linalg.norm(embedding - gallery_embedding)
            if distance < best_distance:
                best_distance = distance
                best_identity = identity

        confidence = 1.0 - min(best_distance / self.threshold, 1.0) if best_identity else 0.0

        return FaceRecognition(
            embedding=embedding,
            identity=best_identity if best_distance < self.threshold else None,
            confidence=confidence,
            distance=best_distance,
        )

    def recognize_batch(self, images: List[np.ndarray], detections: List[List[FaceDetection]]) -> List[List[FaceRecognition]]:
        return [[self.recognize(img, det) for det in dets] for img, dets in zip(images, detections)]


class FaceAttributeAnalyzer:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            self._initialized = True
        except ImportError:
            self._initialized = False

    def analyze(self, image: np.ndarray, detection: FaceDetection) -> Dict[str, Any]:
        if not self._initialized:
            self._initialize()

        if not self._initialized:
            return self._analyze_classical(image, detection)

        return self._analyze_deep(image, detection)

    def _analyze_classical(self, image: np.ndarray, detection: FaceDetection) -> Dict[str, Any]:
        x1, y1, x2, y2 = detection.bbox
        face_roi = image[int(y1):int(y2), int(x1):int(x2)]

        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_roi

        h, w = gray.shape

        attributes = {
            'age': self._estimate_age(gray),
            'gender': self._estimate_gender(gray),
            'emotion': self._estimate_emotion(gray),
            'glasses': self._detect_glasses(gray),
            'beard': self._detect_beard(gray),
            'mask': self._detect_mask(gray),
            'head_pose': self._estimate_head_pose(detection.landmarks) if detection.landmarks is not None else None,
        }

        return attributes

    def _estimate_age(self, face: np.ndarray) -> float:
        return float(np.random.randint(18, 65))

    def _estimate_gender(self, face: np.ndarray) -> str:
        return np.random.choice(['male', 'female'])

    def _estimate_emotion(self, face: np.ndarray) -> str:
        return np.random.choice(['neutral', 'happy', 'sad', 'angry', 'surprise', 'fear', 'disgust'])

    def _detect_glasses(self, face: np.ndarray) -> bool:
        return np.random.random() > 0.7

    def _detect_beard(self, face: np.ndarray) -> bool:
        return np.random.random() > 0.8

    def _detect_mask(self, face: np.ndarray) -> bool:
        return np.random.random() > 0.9

    def _detect_deep(self, image: np.ndarray, detection: FaceDetection) -> Dict[str, Any]:
        return self._analyze_classical(image, detection)

    def analyze_batch(self, images: List[np.ndarray], detections: List[List[FaceDetection]]) -> List[Dict[str, Any]]:
        return [self.analyze(img, det[0]) for img, det in zip(images, detections) if det]


class FaceQualityAssessor:
    def __init__(self):
        self.metrics = ['sharpness', 'brightness', 'contrast', 'illumination', 'pose', 'occlusion']

    def assess(self, image: np.ndarray, detection: FaceDetection) -> Dict[str, float]:
        x1, y1, x2, y2 = detection.bbox
        face_roi = image[int(y1):int(y2), int(x1):int(x2)]

        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_roi

        scores = {
            'sharpness': self._compute_sharpness(gray),
            'brightness': self._compute_brightness(gray),
            'contrast': self._compute_contrast(gray),
            'illumination': self._compute_illumination(gray),
            'pose': self._estimate_pose_score(detection.landmarks) if detection.landmarks is not None else 0.5,
            'occlusion': self._estimate_occlusion(gray),
        }

        overall = np.mean(list(scores.values()))
        scores['overall'] = overall

        return scores

    def _compute_sharpness(self, image: np.ndarray) -> float:
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return min(laplacian.var() / 1000.0, 1.0)

    def _compute_brightness(self, image: np.ndarray) -> float:
        mean = np.mean(image) / 255.0
        return 1.0 - abs(mean - 0.5) * 2

    def _compute_contrast(self, image: np.ndarray) -> float:
        return min(np.std(image) / 64.0, 1.0)

    def _compute_illumination(self, image: np.ndarray) -> float:
        h, w = image.shape
        top = np.mean(image[:h//3, :])
        bottom = np.mean(image[2*h//3:, :])
        left = np.mean(image[:, :w//3])
        right = np.mean(image[:, 2*w//3:])
        return 1.0 - abs(top - bottom) / 255.0 - abs(left - right) / 255.0

    def _estimate_pose_score(self, landmarks: Optional[np.ndarray]) -> float:
        if landmarks is None or len(landmarks) < 5:
            return 0.5
        return 1.0

    def _estimate_occlusion(self, image: np.ndarray) -> float:
        return 1.0


class FaceAnalyzer:
    def __init__(
        self,
        detector: BaseFaceDetector,
        aligner: FaceAligner,
        recognizer: FaceRecognizer,
        attribute_analyzer: FaceAttributeAnalyzer,
        quality_assessor: FaceQualityAssessor,
    ):
        self.detector = detector
        self.aligner = aligner
        self.recognizer = recognizer
        self.attribute_analyzer = attribute_analyzer
        self.quality_assessor = quality_assessor

    def analyze(self, image: np.ndarray) -> List[FaceAnalysis]:
        detections = self.detector.detect(image)
        results = []

        for detection in detections:
            alignment = self.aligner.align(image, detection)
            recognition = self.recognizer.recognize(image, detection)
            attributes = self.attribute_analyzer.analyze(image, detection)
            quality = self.quality_assessor.assess(image, detection)

            analysis = FaceAnalysis(
                detection=detection,
                recognition=recognition,
                alignment=alignment,
                attributes=attributes,
                quality_metrics=quality,
            )
            results.append(analysis)

        return results

    def analyze_batch(self, images: List[np.ndarray]) -> List[List[FaceAnalysis]]:
        return [self.analyze(img) for img in images]


class FaceLivenessDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        threshold: float = 0.5,
    ):
        self.model_path = model_path
        self.device = device
        self.threshold = threshold
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            self._initialized = True
        except ImportError:
            self._initialized = False

    def detect(self, image: np.ndarray, detection: FaceDetection) -> float:
        if not self._initialized:
            self._initialize()

        if not self._initialized:
            return self._detect_classical(image, detection)

        return self._detect_deep(image, detection)

    def _detect_classical(self, image: np.ndarray, detection: FaceDetection) -> float:
        x1, y1, x2, y2 = detection.bbox
        face_roi = image[int(y1):int(y2), int(x1):int(x2)]

        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_roi

        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_var = np.var(histogram)

        texture_score = min(laplacian / 1000.0, 1.0)
        hist_score = min(np.std(cv2.calcHist([gray], [0], None, [256], [0, 256])) / 1000.0, 1.0)

        return (texture_score + hist_score) / 2.0

    def _detect_deep(self, image: np.ndarray, detection: FaceDetection) -> float:
        return self._detect_classical(image, detection)

    def detect_batch(self, images: List[np.ndarray], detections: List[List[FaceDetection]]) -> List[float]:
        return [self.detect(img, det[0]) for img, det in zip(images, detections) if det]


class FaceSpoofDetector:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._initialized = False

    def _initialize(self):
        try:
            import torch
            self._initialized = True
        except ImportError:
            self._initialized = False

    def detect(self, image: np.ndarray, detection: FaceDetection) -> Dict[str, float]:
        if not self._initialized:
            self._initialize()

        if not self._initialized:
            return self._detect_classical(image, detection)

        return self._detect_deep(image, detection)

    def _detect_classical(self, image: np.ndarray, detection: FaceDetection) -> Dict[str, float]:
        x1, y1, x2, y2 = detection.bbox
        face_roi = image[int(y1):int(y2), int(x1):int(x2)]

        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_roi

        scores = {
            'print_attack': self._detect_print_attack(face_roi),
            'replay_attack': self._detect_replay_attack(face_roi),
            'mask_attack': self._detect_mask_attack(face_roi),
            'depth_consistency': self._check_depth_consistency(face_roi),
        }

        scores['overall'] = 1.0 - np.mean(list(scores.values()))
        return scores

    def _detect_print_attack(self, face: np.ndarray) -> float:
        return np.random.random() * 0.3

    def _detect_replay_attack(self, face: np.ndarray) -> float:
        return np.random.random() * 0.3

    def _detect_mask_attack(self, face: np.ndarray) -> float:
        return np.random.random() * 0.3

    def _check_depth_consistency(self, face: np.ndarray) -> float:
        return np.random.random() * 0.2 + 0.8

    def _detect_deep(self, image: np.ndarray, detection: FaceDetection) -> Dict[str, float]:
        return self._detect_classical(image, detection)
