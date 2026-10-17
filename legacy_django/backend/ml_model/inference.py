import os

import cv2
import numpy as np
import tensorflow as tf
from django.conf import settings

from .model import build_model


class ObjectDetector:
    _instance = None

    IMG_SIZE = 640
    NUM_CLASSES = 10
    GRID_SIZES = (80, 40, 20)

    CLASS_NAMES = (
        "sunglasses",
        "knife",
        "water_bottle",
        "pen",
        "chair",
        "human_face",
        "mobile_phone",
        "helmet",
        "fire",
        "can",
    )

    CLASS_THRESHOLDS = {
        "sunglasses": 0.22,
        "knife": 0.28,
        "water_bottle": 0.25,
        "pen": 0.18,
        "chair": 0.35,
        "human_face": 0.30,
        "mobile_phone": 0.24,
        "helmet": 0.26,
        "fire": 0.28,
        "can": 0.22,
    }

    ANCHORS = (
        np.array([
            [0.02, 0.03],
            [0.04, 0.07],
            [0.08, 0.06],
        ]),
        np.array([
            [0.07, 0.15],
            [0.15, 0.11],
            [0.14, 0.29],
        ]),
        np.array([
            [0.28, 0.22],
            [0.38, 0.48],
            [0.90, 0.78],
        ]),
    )

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()

        return cls._instance

    def _initialize(self):
        """Build the model and load trained weights."""

        weights_path = os.path.join(
            settings.BASE_DIR,
            "backend",
            "ml_model",
            "weights",
            "best_weights.weights.h5",
        )

        try:
            self.model = build_model()
            self.model.load_weights(weights_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load object detection model from "
                f"'{weights_path}': {exc}"
            ) from exc

    def detect_from_file(self, image_path):
        """Run detection on an image stored on disk."""

        image = cv2.imread(str(image_path))

        return self._detect(image)

    def detect_from_bytes(self, image_bytes):
        """Run detection on image data received as bytes."""

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        return self._detect(image)

    def _detect(self, image):
        """Preprocess an image, run inference, and decode predictions."""

        if image is None:
            raise ValueError("Invalid image data")

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        original_height, original_width = image.shape[:2]

        image_resized = cv2.resize(
            image_rgb,
            (self.IMG_SIZE, self.IMG_SIZE),
        )

        image_normalized = (
            image_resized.astype(np.float32) / 255.0
        )

        image_batch = np.expand_dims(
            image_normalized,
            axis=0,
        )

        predictions = self.model(
            image_batch,
            training=False,
        )

        return self._decode_predictions(
            predictions,
            original_width,
            original_height,
        )

    def _decode_predictions(
        self,
        predictions,
        original_width,
        original_height,
    ):
        """Convert raw model predictions into detection results."""

        all_boxes = []
        all_scores = []
        all_classes = []

        thresholds = np.array([self.CLASS_THRESHOLDS[name] for name in self.CLASS_NAMES])

        for scale_index, prediction in enumerate(predictions):
            grid_size = self.GRID_SIZES[scale_index]
            anchors = self.ANCHORS[scale_index]

            pred = prediction.numpy()[0].reshape(
                grid_size,
                grid_size,
                3,
                5 + self.NUM_CLASSES,
            )

            # Objectness and coordinate predictions
            obj = self._sigmoid(pred[..., 4])

            col_grid, row_grid = np.meshgrid(
                np.arange(grid_size),
                np.arange(grid_size),
            )
            col_grid = np.expand_dims(col_grid, -1)
            row_grid = np.expand_dims(row_grid, -1)

            bx = (col_grid + self._sigmoid(pred[..., 0])) / grid_size
            by = (row_grid + self._sigmoid(pred[..., 1])) / grid_size

            anchor_w = anchors[:, 0].reshape(1, 1, 3)
            anchor_h = anchors[:, 1].reshape(1, 1, 3)

            bw = anchor_w * np.exp(np.clip(pred[..., 2], -5.0, 5.0))
            bh = anchor_h * np.exp(np.clip(pred[..., 3], -5.0, 5.0))

            x1 = np.clip(bx - bw / 2.0, 0.0, 1.0)
            y1 = np.clip(by - bh / 2.0, 0.0, 1.0)
            x2 = np.clip(bx + bw / 2.0, 0.0, 1.0)
            y2 = np.clip(by + bh / 2.0, 0.0, 1.0)

            class_probs = self._sigmoid(pred[..., 5:])
            class_ids = np.argmax(class_probs, axis=-1)
            class_scores = np.max(class_probs, axis=-1)
            confidence = obj * class_scores

            cell_thresholds = thresholds[class_ids]
            mask = (
                (obj >= 0.10)
                & (confidence >= cell_thresholds)
                & (bw >= 0.01)
                & (bh >= 0.01)
            )

            if np.any(mask):
                boxes = np.stack(
                    [x1[mask], y1[mask], x2[mask], y2[mask]],
                    axis=-1,
                )
                all_boxes.extend(boxes.tolist())
                all_scores.extend(confidence[mask].tolist())
                all_classes.extend(class_ids[mask].tolist())

        nms_results = self._apply_nms(
            all_boxes,
            all_scores,
            all_classes,
        )

        detections = []

        for box, score, class_id in zip(
            nms_results["boxes"],
            nms_results["scores"],
            nms_results["classes"],
        ):
            detections.append({
                "class": self.CLASS_NAMES[class_id],
                "class_id": class_id,
                "confidence": float(score),
                "bbox": [
                    float(box[0] * original_width),
                    float(box[1] * original_height),
                    float(box[2] * original_width),
                    float(box[3] * original_height),
                ],
            })

        return detections

    @staticmethod
    def _sigmoid(value):
        """Numerically stable sigmoid for float32 arrays."""
        return 1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))

    def _apply_nms(
        self,
        boxes,
        scores,
        classes,
        iou_threshold=0.3,
    ):
        """Apply class-wise non-maximum suppression."""

        if not boxes:
            return {
                "boxes": [],
                "scores": [],
                "classes": [],
            }

        result_boxes = []
        result_scores = []
        result_classes = []

        boxes_array = np.array(boxes)
        scores_array = np.array(scores)
        classes_array = np.array(classes)

        for class_id in range(self.NUM_CLASSES):
            class_mask = classes_array == class_id

            if not class_mask.any():
                continue

            class_boxes = boxes_array[class_mask]
            class_scores = scores_array[class_mask]
            class_indices = np.where(class_mask)[0]

            keep_indices = tf.image.non_max_suppression(
                class_boxes,
                class_scores,
                3,
                iou_threshold=iou_threshold,
            ).numpy()

            for index in keep_indices:
                original_index = class_indices[index]

                result_boxes.append(
                    boxes[original_index]
                )
                result_scores.append(
                    scores[original_index]
                )
                result_classes.append(
                    classes[original_index]
                )

        return {
            "boxes": result_boxes,
            "scores": result_scores,
            "classes": result_classes,
        }


detector = ObjectDetector()
