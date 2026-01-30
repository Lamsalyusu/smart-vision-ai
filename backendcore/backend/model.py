# model.py - YOLO BACKBONE VERSION

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = 640
NUM_CLASSES = 10

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
    """YOLO-inspired model - MUST MATCH YOUR TRAINING!"""
    
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


if __name__ == "__main__":
    m = build_model()
    m.summary()