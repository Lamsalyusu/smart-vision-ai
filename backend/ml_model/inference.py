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
        "sunglasses": 0.25,
        "knife": 0.25,
        "water_bottle": 0.25,
        "pen": 0.25,
        "chair": 0.25,
        "human_face": 0.25,
        "mobile_phone": 0.25,
        "helmet": 0.25,
        "fire": 0.25,
        "can": 0.25,
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

        for scale_index, prediction in enumerate(predictions):
            grid_size = self.GRID_SIZES[scale_index]
            anchors = self.ANCHORS[scale_index]

            prediction = prediction.numpy()[0]

            prediction = prediction.reshape(
                grid_size,
                grid_size,
                3,
                5 + self.NUM_CLASSES,
            )

            for row in range(grid_size):
                for column in range(grid_size):
                    for anchor_index in range(3):
                        values = prediction[
                            row,
                            column,
                            anchor_index,
                        ]

                        objectness = self._sigmoid(
                            values[4]
                        )

                        if objectness < 0.10:
                            continue

                        class_probabilities = self._sigmoid(
                            values[5:]
                        )

                        class_id = int(
                            np.argmax(class_probabilities)
                        )

                        class_score = class_probabilities[
                            class_id
                        ]

                        confidence = (
                            objectness * class_score
                        )

                        class_name = self.CLASS_NAMES[class_id]
                        threshold = self.CLASS_THRESHOLDS[
                            class_name
                        ]

                        if confidence < threshold:
                            continue

                        tx, ty, tw, th = values[:4]

                        center_x = (
                            column + self._sigmoid(tx)
                        ) / grid_size

                        center_y = (
                            row + self._sigmoid(ty)
                        ) / grid_size

                        width = anchors[
                            anchor_index,
                            0,
                        ] * np.exp(
                            np.clip(tw, -5, 5)
                        )

                        height = anchors[
                            anchor_index,
                            1,
                        ] * np.exp(
                            np.clip(th, -5, 5)
                        )

                        if width < 0.01 or height < 0.01:
                            continue

                        x1 = max(
                            0,
                            center_x - width / 2,
                        )

                        y1 = max(
                            0,
                            center_y - height / 2,
                        )

                        x2 = min(
                            1,
                            center_x + width / 2,
                        )

                        y2 = min(
                            1,
                            center_y + height / 2,
                        )

                        all_boxes.append(
                            [x1, y1, x2, y2]
                        )
                        all_scores.append(
                            float(confidence)
                        )
                        all_classes.append(
                            class_id
                        )

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
        """Apply a numerically bounded sigmoid transformation."""

        return 1 / (
            1 + np.exp(
                -np.clip(value, -500, 500)
            )
        )

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
