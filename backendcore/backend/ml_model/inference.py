import numpy as np              # Numerical operations (arrays, math)
import cv2                     # Image processing (OpenCV)
import tensorflow as tf        # Deep learning framework
import os                      # File path handling
from django.conf import settings  # Django settings (to locate weights file)

from .model import build_model  # Your YOLO-style model architecture


class ObjectDetector:
    _instance = None   # Used for singleton pattern (only one model instance)

    def __new__(cls):
        # Ensures only ONE instance of model is created (memory efficient)
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print("Loading model")

        # Input size expected by model
        self.IMG_SIZE = 640

        # Number of classes your model predicts
        self.NUM_CLASSES = 10

        # Class labels (index → name)
        self.CLASS_NAMES = [
            "sunglasses", "knife", "water_bottle", "pen", "chair",
            "human_face", "mobile_phone", "helmet", "fire", "can"
        ]

        # Confidence threshold per class 
        self.CLASS_THRESHOLDS = {
            "sunglasses": 0.20,
            "knife": 0.20,
            "water_bottle": 0.20,
            "pen": 0.20,
            "chair": 0.20,
            "human_face": 0.20,
            "mobile_phone": 0.20,
            "helmet": 0.20,
            "fire": 0.20,
            "can": 0.20
        }

        # Anchor boxes (predefined shapes for bounding boxes)
        self.ANCHORS = [
            np.array([[0.02, 0.03], [0.04, 0.07], [0.08, 0.06]]),  # small scale
            np.array([[0.07, 0.15], [0.15, 0.11], [0.14, 0.29]]),  # medium
            np.array([[0.28, 0.22], [0.38, 0.48], [0.90, 0.78]])   # large
        ]

        # Grid sizes corresponding to model outputs
        self.GRID_SIZES = [80, 40, 20]

        # Path to trained weights
        weights_path = os.path.join(
            settings.BASE_DIR, 'backend', 'ml_model', 'weights', 'best_weights.weights.h5'
        )

        try:
            self.model = build_model()            # Build architecture
            self.model.load_weights(weights_path) # Load trained weights
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    # Detect from image file
    def detect_from_file(self, image_path):
        img = cv2.imread(str(image_path))  # Read image from disk
        return self._detect(img)

    # Detect from uploaded image bytes (e.g., API request)
    def detect_from_bytes(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)  # Convert bytes → array
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)   # Decode into image
        return self._detect(img)

    # Main detection pipeline
    def _detect(self, img):
        if img is None:
            raise ValueError("Invalid image")

        # Convert BGR → RGB (OpenCV loads BGR, model expects RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Save original size (for later scaling)
        h0, w0 = img.shape[:2]

        # Resize image to model input size
        img_resized = cv2.resize(img_rgb, (self.IMG_SIZE, self.IMG_SIZE))

        # Normalize pixel values (0–255 → 0–1)
        img_norm = img_resized.astype(np.float32) / 255.0

        # Add batch dimension → (1, 640, 640, 3)
        img_batch = np.expand_dims(img_norm, 0)

        # Run model prediction
        preds = self.model(img_batch, training=False)

        # Decode predictions into usable boxes
        detections = self._decode_predictions(preds, w0, h0)
        return detections

    # Convert raw model output into bounding boxes
    def _decode_predictions(self, preds, orig_w, orig_h):
        all_boxes = []
        all_scores = []
        all_classes = []

        # Loop through 3 scales (80x80, 40x40, 20x20)
        for scale_idx, pred in enumerate(preds):
            grid_size = self.GRID_SIZES[scale_idx]
            anchors = self.ANCHORS[scale_idx]

            pred = pred.numpy()[0]

            # Reshape → (grid, grid, anchors, attributes)
            pred = pred.reshape(grid_size, grid_size, 3, 5 + self.NUM_CLASSES)

            # Loop through each grid cell
            for r in range(grid_size):
                for c in range(grid_size):
                    for a in range(3):  # 3 anchors

                        # Objectness score (is there an object?)
                        obj = self._sigmoid(pred[r, c, a, 4])
                        if obj < 0.10:
                            continue  # skip weak detections

                        # Class probabilities
                        cls_probs = self._sigmoid(pred[r, c, a, 5:])
                        cls_id = np.argmax(cls_probs)
                        cls_score = cls_probs[cls_id]

                        # Final confidence
                        conf = obj * cls_score

                        threshold = self.CLASS_THRESHOLDS[self.CLASS_NAMES[cls_id]]
                        if conf < threshold:
                            continue

                        # Bounding box offsets
                        tx, ty = pred[r, c, a, 0:2]
                        tw, th = pred[r, c, a, 2:4]

                        # Convert to actual position
                        cx = (c + self._sigmoid(tx)) / grid_size
                        cy = (r + self._sigmoid(ty)) / grid_size

                        # Width & height using anchors
                        bw = anchors[a, 0] * np.exp(np.clip(tw, -5, 5))
                        bh = anchors[a, 1] * np.exp(np.clip(th, -5, 5))

                        if bw < 0.01 or bh < 0.01:
                            continue

                        # Convert center → corner format
                        x1 = max(0, cx - bw / 2)
                        y1 = max(0, cy - bh / 2)
                        x2 = min(1, cx + bw / 2)
                        y2 = min(1, cy + bh / 2)

                        # Store results
                        all_boxes.append([x1, y1, x2, y2])
                        all_scores.append(float(conf))
                        all_classes.append(int(cls_id))

        # Apply Non-Max Suppression (remove duplicate boxes)
        result = self._apply_nms(all_boxes, all_scores, all_classes)

        detections = []

        # Scale boxes back to original image size
        for box, score, cls_id in zip(result['boxes'], result['scores'], result['classes']):
            detections.append({
                'class': self.CLASS_NAMES[cls_id],
                'class_id': cls_id,
                'confidence': float(score),
                'bbox': [
                    float(box[0] * orig_w),
                    float(box[1] * orig_h),
                    float(box[2] * orig_w),
                    float(box[3] * orig_h)
                ]
            })

        return detections

    # Sigmoid activation
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    # Non-Max Suppression (remove overlapping boxes)
    def _apply_nms(self, boxes, scores, classes, iou_threshold=0.3):
        if len(boxes) == 0:
            return {'boxes': [], 'scores': [], 'classes': []}

        result_boxes = []
        result_scores = []
        result_classes = []

        boxes_np = np.array(boxes)
        scores_np = np.array(scores)

        # Apply NMS per class
        for cls_id in range(self.NUM_CLASSES):
            cls_mask = np.array(classes) == cls_id
            if not cls_mask.any():
                continue

            cls_boxes = boxes_np[cls_mask]
            cls_scores = scores_np[cls_mask]
            cls_indices = np.where(cls_mask)[0]

            # TensorFlow NMS
            keep = tf.image.non_max_suppression(
                cls_boxes,
                cls_scores,
                3,              # max boxes per class
                iou_threshold=0.3
            ).numpy()

            for i in keep:
                orig_idx = cls_indices[i]
                result_boxes.append(boxes[orig_idx])
                result_scores.append(scores[orig_idx])
                result_classes.append(classes[orig_idx])

        return {
            'boxes': result_boxes,
            'scores': result_scores,
            'classes': result_classes
        }
# Create global detector instance
detector = ObjectDetector()