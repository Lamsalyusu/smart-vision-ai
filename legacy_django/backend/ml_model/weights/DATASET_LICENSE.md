# Dataset Licence — Training Data for `best_weights.weights.h5`

## Trained model weights

File: `best_weights.weights.h5`

- **Architecture**: Custom CSPDarknet backbone + SPPF + FPN neck YOLO (7.9M parameters)
- **Framework**: TensorFlow/Keras 2.x
- **Trained by**: Yuyutsu Lamsal (project author), on Google Colab (A100 GPU)
- **Weights licence**: MIT License (project author) — see [`../../../../LICENSE`](../../../../LICENSE)

## Training data licence

The model was trained on public community object-detection datasets sourced via [Roboflow Universe](https://universe.roboflow.com/), one dataset per class. All training data is licensed under:

**Creative Commons Attribution 4.0 International (CC BY 4.0)**
- Licence text: https://creativecommons.org/licenses/by/4.0/
- Licence deed: https://creativecommons.org/licenses/by/4.0/deed.en

## Attribution (required by CC BY 4.0 §3)

When redistributing or using these weights (an adaptation of the CC BY 4.0 training data), you must:

1. Give appropriate credit to the original dataset creators (Roboflow Universe community contributors) and to Roboflow.
2. Provide a link to the licence: https://creativecommons.org/licenses/by/4.0/
3. Indicate that changes were made (datasets merged, filtered, re-annotated into a single 10-class COCO-format dataset; off-target labels dropped; images resized to 640×640, bounding boxes normalised).
4. Not imply that Roboflow or the original dataset creators endorse this project or its use.

Full per-project attribution and source URLs for all 10 classes: see [`../../../../DATASET_ATTRIBUTION.md`](../../../../DATASET_ATTRIBUTION.md).

## Classes

sunglasses, knife, water_bottle, pen, chair, human_face, mobile_phone, helmet, fire, can.

---

**Attributed by:** Yuyutsu Lamsal
**Project:** SmartVision AI
**Date:** 2026-11-04
