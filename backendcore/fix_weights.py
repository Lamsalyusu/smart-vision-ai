import h5py
import numpy as np
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tf_keras as keras
from tf_keras import layers
import sys
sys.path.insert(0, '.')
from backend.ml_model.model import build_model
import tensorflow as tf

model = build_model()
dummy = tf.zeros((1, 640, 640, 3))
_ = model(dummy)

f = h5py.File('backend/ml_model/weights/best_weights.weights.h5', 'r')
root = f['layers']

def sort_key(name):
    parts = name.split('_')
    if parts[-1].isdigit():
        return int(parts[-1])
    return -1

h5_conv = sorted([n for n in root.keys() if n.startswith('conv2d')], key=sort_key)
h5_bn = sorted([n for n in root.keys() if n.startswith('batch_normalization')], key=sort_key)

model_conv = []
model_bn = []
for l in model.layers:
    w = l.get_weights()
    if len(w) == 1 and len(w[0].shape) == 4:
        model_conv.append(l)
    elif len(w) == 2 and len(w[0].shape) == 4:
        model_conv.append(l)
    elif len(w) == 4:
        model_bn.append(l)

print(f"H5 conv: {len(h5_conv)}, Model conv: {len(model_conv)}")
print(f"H5 BN: {len(h5_bn)}, Model BN: {len(model_bn)}")

loaded = 0
for h5_name, model_layer in zip(h5_conv, model_conv):
    group = root[h5_name]['vars']
    weights = [np.array(group[str(i)]) for i in range(len(group))]
    if len(weights) == len(model_layer.get_weights()):
        model_layer.set_weights(weights)
        loaded += 1

for h5_name, model_layer in zip(h5_bn, model_bn):
    group = root[h5_name]['vars']
    weights = [np.array(group[str(i)]) for i in range(len(group))]
    if len(weights) == len(model_layer.get_weights()):
        model_layer.set_weights(weights)
        loaded += 1

f.close()
print(f"Loaded {loaded} layers")

model.save_weights('backend/ml_model/weights/fixed_weights.weights.h5')
print("Saved fixed_weights.weights.h5")

test_img = np.random.randn(1, 640, 640, 3).astype(np.float32)
outputs = model(test_img, training=False)
for i, out in enumerate(outputs):
    vals = out.numpy().flatten()
    print(f"Head {i}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}")