import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ========== CONFIG ==========
IMG_SIZE = 640
NUM_CLASSES = 10
IOU_THRESH = 0.45
MAX_DETECTIONS = 20
DEBUG_MODE = True

CLASS_NAMES = ["sunglasses", "knife", "water_bottle", "pen", "chair",
               "human_face", "mobile_phone", "helmet", "fire", "can"]

# PER-CLASS CONFIDENCE THRESHOLDS
CLASS_THRESHOLDS = {
    "sunglasses": 0.20,
    "knife": 0.20,
    "water_bottle": 0.20,
    "pen": 0.60,              # Higher - reduces false positives
    "chair": 0.20,
    "human_face": 0.15,
    "mobile_phone": 0.10,     # Lower - catches mobile phones
    "helmet": 0.20,
    "fire": 0.60,             # Higher - reduces false positives
    "can": 0.20
}

COLORS = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255),
          (0,255,255), (128,0,0), (0,128,0), (255,128,0), (128,0,128)]

# CHECKPOINT PATH - Windows format
CHECKPOINT = "best_weights.weights.h5"

ANCHORS = [
    np.array([[0.02, 0.03], [0.04, 0.07], [0.08, 0.06]]),
    np.array([[0.07, 0.15], [0.15, 0.11], [0.14, 0.29]]),
    np.array([[0.28, 0.22], [0.38, 0.48], [0.90, 0.78]])
]


GRID_SIZES = [80, 40, 20]


# ========== YOLO BACKBONE MODEL ARCHITECTURE ==========

def conv_block(x, filters, kernel=3, strides=1, name=None):
    """Standard Conv-BN-ReLU block"""
    x = layers.Conv2D(filters, kernel, strides=strides, padding='same', 
                      use_bias=False, name=f"{name}_conv" if name else None)(x)
    x = layers.BatchNormalization(name=f"{name}_bn" if name else None)(x)
    x = layers.ReLU(name=f"{name}_relu" if name else None)(x)
    return x

def csp_block(x, filters, name):
    """Cross Stage Partial Block (YOLO's key innovation)"""
    split_filters = filters // 2
    
    # Path 1: Direct shortcut
    path1 = layers.Conv2D(split_filters, 1, padding='same', use_bias=False, 
                          name=f"{name}_path1")(x)
    path1 = layers.BatchNormalization(name=f"{name}_bn1")(path1)
    path1 = layers.ReLU(name=f"{name}_relu1")(path1)
    
    # Path 2: Bottleneck processing
    path2 = layers.Conv2D(split_filters, 1, padding='same', use_bias=False,
                          name=f"{name}_path2_conv1")(x)
    path2 = layers.BatchNormalization(name=f"{name}_bn2")(path2)
    path2 = layers.ReLU(name=f"{name}_relu2")(path2)
    
    path2 = layers.Conv2D(split_filters, 3, padding='same', use_bias=False,
                          name=f"{name}_path2_conv2")(path2)
    path2 = layers.BatchNormalization(name=f"{name}_bn3")(path2)
    path2 = layers.ReLU(name=f"{name}_relu3")(path2)
    
    # Concatenate paths
    out = layers.Concatenate(name=f"{name}_concat")([path1, path2])
    
    # Final transition
    out = layers.Conv2D(filters, 1, padding='same', use_bias=False,
                        name=f"{name}_transition")(out)
    out = layers.BatchNormalization(name=f"{name}_bn_out")(out)
    out = layers.ReLU(name=f"{name}_relu_out")(out)
    
    return out

def sppf_block(x, filters, name):
    """Spatial Pyramid Pooling - Fast"""
    conv1 = layers.Conv2D(filters // 2, 1, padding='same', name=f"{name}_conv1")(x)
    conv1 = layers.BatchNormalization(name=f"{name}_bn1")(conv1)
    conv1 = layers.ReLU(name=f"{name}_relu1")(conv1)
    
    # Multiple pooling operations
    pool1 = layers.MaxPooling2D(5, strides=1, padding='same', name=f"{name}_pool1")(conv1)
    pool2 = layers.MaxPooling2D(5, strides=1, padding='same', name=f"{name}_pool2")(pool1)
    pool3 = layers.MaxPooling2D(5, strides=1, padding='same', name=f"{name}_pool3")(pool2)
    
    # Concatenate all
    concat = layers.Concatenate(name=f"{name}_concat")([conv1, pool1, pool2, pool3])
    
    # Final convolution
    out = layers.Conv2D(filters, 1, padding='same', name=f"{name}_conv2")(concat)
    out = layers.BatchNormalization(name=f"{name}_bn2")(out)
    out = layers.ReLU(name=f"{name}_relu2")(out)
    
    return out

def build_model():
    """YOLO-inspired model - MATCHES YOUR YOLO TRAINING!"""
    
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input")
    
    # ========== BACKBONE (CSPDarknet - YOLO Style) ==========
    
    # Stem
    x = conv_block(inputs, 32, kernel=3, strides=2, name="stem1")
    x = conv_block(x, 64, kernel=3, strides=2, name="stem2")
    
    # Stage 1
    x = csp_block(x, 64, name="stage1")
    x = conv_block(x, 128, kernel=3, strides=2, name="down1")
    
    # Stage 2 -> Small features (80x80)
    x = csp_block(x, 128, name="stage2a")
    x = csp_block(x, 128, name="stage2b")
    small_features = x
    x = conv_block(x, 256, kernel=3, strides=2, name="down2")
    
    # Stage 3 -> Medium features (40x40)
    x = csp_block(x, 256, name="stage3a")
    x = csp_block(x, 256, name="stage3b")
    medium_features = x
    x = conv_block(x, 512, kernel=3, strides=2, name="down3")
    
    # Stage 4 -> Large features (20x20)
    x = csp_block(x, 512, name="stage4a")
    x = csp_block(x, 512, name="stage4b")
    x = sppf_block(x, 512, name="sppf")
    large_features = x
    
    # ========== NECK (FPN - Feature Pyramid Network) ==========
    
    # P5 (20x20)
    p5 = conv_block(large_features, 256, kernel=1, name="neck_p5")
    p5_up = layers.UpSampling2D(2, name="p5_upsample")(p5)
    
    # P4 (40x40)
    p4 = conv_block(medium_features, 256, kernel=1, name="neck_p4_reduce")
    p4 = layers.Concatenate(name="p4_concat")([p4, p5_up])
    p4 = csp_block(p4, 256, name="neck_p4")
    p4_up = layers.UpSampling2D(2, name="p4_upsample")(p4)
    
    # P3 (80x80)
    p3 = conv_block(small_features, 256, kernel=1, name="neck_p3_reduce")
    p3 = layers.Concatenate(name="p3_concat")([p3, p4_up])
    p3 = csp_block(p3, 256, name="neck_p3")
    
    # ========== HEAD (Detection Heads) ==========
    
    num_anchors = 3
    out_channels = num_anchors * (5 + NUM_CLASSES)  # 3 * 15 = 45
    
    # Small object head (80x80)
    head_s = conv_block(p3, 256, kernel=3, name="head_s")
    head_s = layers.Dropout(0.2, name="head_s_dropout")(head_s)
    head_s = layers.Conv2D(out_channels, 1, name="detect_s")(head_s)
    
    # Medium object head (40x40)
    head_m = conv_block(p4, 256, kernel=3, name="head_m")
    head_m = layers.Dropout(0.2, name="head_m_dropout")(head_m)
    head_m = layers.Conv2D(out_channels, 1, name="detect_m")(head_m)
    
    # Large object head (20x20)
    head_l = conv_block(p5, 256, kernel=3, name="head_l")
    head_l = layers.Dropout(0.2, name="head_l_dropout")(head_l)
    head_l = layers.Conv2D(out_channels, 1, name="detect_l")(head_l)
    
    return keras.Model(inputs, [head_s, head_m, head_l], name="SmartVision_YOLO")


# ========== LOAD MODEL ==========

print("=" * 60)
print("🔧 Building YOLO-style model...")
model = build_model()

print("📦 Loading weights from:", CHECKPOINT)
try:
    model.load_weights(CHECKPOINT)
    print("✅ Weights loaded successfully!")
except Exception as e:
    print(f"❌ ERROR loading weights: {e}")
    print("\n📋 Make sure:")
    print("1. File exists at:", CHECKPOINT)
    print("2. File is the YOLO backbone weights (not old model)")
    print("3. File was downloaded completely")
    exit(1)

print("📊 Per-class thresholds:")
for name, thresh in CLASS_THRESHOLDS.items():
    print(f"   {name}: {thresh}")
print("=" * 60)


# ========== DETECTION FUNCTIONS ==========

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def get_class_threshold(class_id):
    """Get threshold for specific class"""
    class_name = CLASS_NAMES[class_id]
    return CLASS_THRESHOLDS.get(class_name, 0.20)

def detect(frame):
    h0, w0 = frame.shape[:2]
    
    # Preprocess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict
    preds = model(img, training=False)
    
    all_boxes = []
    all_scores = []
    all_classes = []
    
    # Debug counters
    total_checked = 0
    passed_checks = 0
    class_detections = {name: 0 for name in CLASS_NAMES}
    
    for scale_idx, pred in enumerate(preds):
        grid_size = GRID_SIZES[scale_idx]
        anchors = ANCHORS[scale_idx]
        
        pred = pred.numpy()[0]
        pred = pred.reshape(grid_size, grid_size, 3, 5 + NUM_CLASSES)
        
        for r in range(grid_size):
            for c in range(grid_size):
                for a in range(3):
                    total_checked += 1
                    
                    # Get raw values
                    tx = pred[r, c, a, 0]
                    ty = pred[r, c, a, 1]
                    tw = pred[r, c, a, 2]
                    th = pred[r, c, a, 3]
                    obj_logit = pred[r, c, a, 4]
                    cls_logits = pred[r, c, a, 5:]
                    
                    # Objectness
                    obj = sigmoid(obj_logit)
                    if obj < 0.10:
                        continue
                    
                    # Class
                    cls_probs = sigmoid(cls_logits)
                    cls_id = np.argmax(cls_probs)
                    cls_score = cls_probs[cls_id]
                    
                    # Final confidence
                    conf = obj * cls_score
                    
                    # Use class-specific threshold
                    class_threshold = get_class_threshold(cls_id)
                    if conf < class_threshold:
                        continue
                    
                    passed_checks += 1
                    
                    # Decode box
                    cx = (c + sigmoid(tx)) / grid_size
                    cy = (r + sigmoid(ty)) / grid_size
                    
                    tw = np.clip(tw, -5, 5)
                    th = np.clip(th, -5, 5)
                    bw = anchors[a, 0] * np.exp(tw)
                    bh = anchors[a, 1] * np.exp(th)
                    
                    # Skip bad boxes
                    if bw < 0.01 or bh < 0.01 or bw > 1.2 or bh > 1.2:
                        continue
                    
                    x1 = max(0, cx - bw / 2)
                    y1 = max(0, cy - bh / 2)
                    x2 = min(1, cx + bw / 2)
                    y2 = min(1, cy + bh / 2)
                    
                    # Skip tiny boxes
                    if (x2 - x1) < 0.02 or (y2 - y1) < 0.02:
                        continue
                    
                    all_boxes.append([x1, y1, x2, y2])
                    all_scores.append(float(conf))
                    all_classes.append(int(cls_id))
                    class_detections[CLASS_NAMES[cls_id]] += 1
    
    if DEBUG_MODE:
        print(f"🔍 Checked: {total_checked} | Passed: {passed_checks} | Raw: {len(all_boxes)}")
        if any(class_detections.values()):
            detected = {k: v for k, v in class_detections.items() if v > 0}
            print(f"   Classes: {detected}")
    
    # NMS (Non-Maximum Suppression)
    result_boxes = []
    result_scores = []
    result_classes = []
    
    if len(all_boxes) > 0:
        boxes_np = np.array(all_boxes, dtype=np.float32)
        scores_np = np.array(all_scores, dtype=np.float32)
        
        # Convert to pixel coords for NMS
        boxes_pixel = boxes_np.copy()
        boxes_pixel[:, 0] *= w0
        boxes_pixel[:, 1] *= h0
        boxes_pixel[:, 2] *= w0
        boxes_pixel[:, 3] *= h0
        
        # Per-class NMS
        for cls_id in range(NUM_CLASSES):
            cls_mask = np.array(all_classes) == cls_id
            if not cls_mask.any():
                continue
            
            cls_boxes = boxes_pixel[cls_mask]
            cls_scores = scores_np[cls_mask]
            cls_indices = np.where(cls_mask)[0]
            
            keep = tf.image.non_max_suppression(
                cls_boxes,
                cls_scores,
                MAX_DETECTIONS,
                iou_threshold=IOU_THRESH,
                score_threshold=get_class_threshold(cls_id)
            ).numpy()
            
            for i in keep:
                orig_idx = cls_indices[i]
                result_boxes.append(all_boxes[orig_idx])
                result_scores.append(all_scores[orig_idx])
                result_classes.append(all_classes[orig_idx])
    
    if DEBUG_MODE and len(result_classes) > 0:
        print(f"✅ After NMS: {len(result_boxes)}")
        for cls_id, score in zip(result_classes, result_scores):
            print(f"   - {CLASS_NAMES[cls_id]}: {score:.3f}")
    
    # Draw detections
    for box, score, cls_id in zip(result_boxes, result_scores, result_classes):
        x1 = int(box[0] * w0)
        y1 = int(box[1] * h0)
        x2 = int(box[2] * w0)
        y2 = int(box[3] * h0)
        
        color = COLORS[cls_id]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        label = f"{CLASS_NAMES[cls_id]}: {score:.2f}"
        
        # Background for text
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    # Add detection counter
    cv2.putText(frame, f"Detections: {len(result_boxes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return frame, len(result_boxes)


# ========== WEBCAM LOOP ==========

cap = cv2.VideoCapture(0)
frame_count = 0

print("\n🎥 Starting webcam... Press ESC to quit")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result, num_detections = detect(frame)
    
    cv2.imshow("SmartVision YOLO", result)
    
    frame_count += 1
    if DEBUG_MODE and frame_count % 30 == 0:
        print("-" * 60)
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
print("\n👋 Webcam closed")