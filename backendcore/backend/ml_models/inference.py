# backend/ml_model/inference.py

import numpy as np
import cv2
import tensorflow as tf
import os
from django.conf import settings

# Import your model architecture
from .model import build_model

class ObjectDetector:
    """Singleton detector for object detection"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Load model once at startup"""
        print("🔧 Loading SmartVision AI model...")
        
        # Configuration
        self.IMG_SIZE = 640
        self.NUM_CLASSES = 10
        
        self.CLASS_NAMES = [
            "sunglasses", "knife", "water_bottle", "pen", "chair",
            "human_face", "mobile_phone", "helmet", "fire", "can"
        ]
        
        self.CLASS_THRESHOLDS = {
            "sunglasses": 0.20,
            "knife": 0.20,
            "water_bottle": 0.20,
            "pen": 0.60,
            "chair": 0.20,
            "human_face": 0.15,
            "mobile_phone": 0.10,
            "helmet": 0.20,
            "fire": 0.60,
            "can": 0.20
        }
        
        self.ANCHORS = [
            np.array([[0.02, 0.03], [0.04, 0.07], [0.08, 0.06]]),
            np.array([[0.07, 0.15], [0.15, 0.11], [0.14, 0.29]]),
            np.array([[0.28, 0.22], [0.38, 0.48], [0.90, 0.78]])
        ]
        
        self.GRID_SIZES = [80, 40, 20]
        
        # Build model
        self.model = build_model()
        
        # Load weights
        weights_path = os.path.join(
            settings.BASE_DIR, 
            'backend', 
            'ml_model', 
            'weights', 
            'best_weights.weights.h5'
        )
        
        try:
            self.model.load_weights(weights_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            raise
    
    def detect_from_file(self, image_path):
        """Detect objects in an image file"""
        img = cv2.imread(str(image_path))
        return self._detect(img)
    
    def detect_from_bytes(self, image_bytes):
        """Detect objects from image bytes (for uploads)"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return self._detect(img)
    
    def _detect(self, img):
        """Internal detection logic"""
        if img is None:
            raise ValueError("Invalid image")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        
        # Preprocess
        img_resized = cv2.resize(img_rgb, (self.IMG_SIZE, self.IMG_SIZE))
        img_norm = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_norm, 0)
        
        # Inference
        preds = self.model(img_batch, training=False)
        
        # Decode
        detections = self._decode_predictions(preds, w0, h0)
        
        return detections
    
    def _decode_predictions(self, preds, orig_w, orig_h):
        """Decode model outputs to detections"""
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for scale_idx, pred in enumerate(preds):
            grid_size = self.GRID_SIZES[scale_idx]
            anchors = self.ANCHORS[scale_idx]
            
            pred = pred.numpy()[0]
            pred = pred.reshape(grid_size, grid_size, 3, 5 + self.NUM_CLASSES)
            
            for r in range(grid_size):
                for c in range(grid_size):
                    for a in range(3):
                        # Objectness
                        obj = self._sigmoid(pred[r, c, a, 4])
                        if obj < 0.10:
                            continue
                        
                        # Class prediction
                        cls_probs = self._sigmoid(pred[r, c, a, 5:])
                        cls_id = np.argmax(cls_probs)
                        cls_score = cls_probs[cls_id]
                        
                        conf = obj * cls_score
                        threshold = self.CLASS_THRESHOLDS[self.CLASS_NAMES[cls_id]]
                        
                        if conf < threshold:
                            continue
                        
                        # Decode bounding box
                        tx, ty = pred[r, c, a, 0:2]
                        tw, th = pred[r, c, a, 2:4]
                        
                        cx = (c + self._sigmoid(tx)) / grid_size
                        cy = (r + self._sigmoid(ty)) / grid_size
                        
                        bw = anchors[a, 0] * np.exp(np.clip(tw, -5, 5))
                        bh = anchors[a, 1] * np.exp(np.clip(th, -5, 5))
                        
                        if bw < 0.01 or bh < 0.01:
                            continue
                        
                        x1 = max(0, cx - bw / 2)
                        y1 = max(0, cy - bh / 2)
                        x2 = min(1, cx + bw / 2)
                        y2 = min(1, cy + bh / 2)
                        
                        all_boxes.append([x1, y1, x2, y2])
                        all_scores.append(float(conf))
                        all_classes.append(int(cls_id))
        
        # Apply NMS
        result = self._apply_nms(all_boxes, all_scores, all_classes)
        
        # Format output
        detections = []
        for box, score, cls_id in zip(result['boxes'], result['scores'], result['classes']):
            detections.append({
                'class': self.CLASS_NAMES[cls_id],
                'class_id': cls_id,
                'confidence': float(score),
                'bbox': {
                    'x1': float(box[0]),
                    'y1': float(box[1]),
                    'x2': float(box[2]),
                    'y2': float(box[3])
                }
            })
        
        return detections
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _apply_nms(self, boxes, scores, classes, iou_threshold=0.45):
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return {'boxes': [], 'scores': [], 'classes': []}
        
        result_boxes = []
        result_scores = []
        result_classes = []
        
        boxes_np = np.array(boxes)
        scores_np = np.array(scores)
        
        # Per-class NMS
        for cls_id in range(self.NUM_CLASSES):
            cls_mask = np.array(classes) == cls_id
            if not cls_mask.any():
                continue
            
            cls_boxes = boxes_np[cls_mask]
            cls_scores = scores_np[cls_mask]
            cls_indices = np.where(cls_mask)[0]
            
            keep = tf.image.non_max_suppression(
                cls_boxes,
                cls_scores,
                20,
                iou_threshold=iou_threshold
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


# Create singleton instance
detector = ObjectDetector()