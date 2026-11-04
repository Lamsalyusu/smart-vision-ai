# Dataset Attribution — Roboflow Universe (CC BY 4.0)

## Summary

The SmartVision AI object-detection model was trained on public community training datasets sourced via [Roboflow Universe](https://universe.roboflow.com/). These datasets are licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

This document is the canonical attribution notice required by CC BY 4.0 Section 3(a). It accompanies the adapted training data and the author-trained model weights derived from it.

---

## 1. Appropriate Credit

Give appropriate credit to:

- **Original dataset creators**: Roboflow Universe community contributors (one community project per object class)
- **Platform / distributor**: Roboflow (Roboflow Universe)

When redistributing or publishing this project or its trained model, reproduce this attribution (or a link to this file) in a prominent location, in any reasonable manner, but **not** in a way that suggests Roboflow or the original dataset creators endorse SmartVision AI or its use.

## 2. License

**Creative Commons Attribution 4.0 International (CC BY 4.0)**

- License text: https://creativecommons.org/licenses/by/4.0/
- License deed: https://creativecommons.org/licenses/by/4.0/deed.en

You are free to share and adapt this material, provided you give appropriate credit, provide a link to the license, and indicate if changes were made.

## 3. Modifications Made

Yes. The source datasets were **adapted** for this project:

- Merged across 10 single-class Roboflow projects into one combined dataset
- Dropped extra/off-target labels (e.g., `table`, `smoke`, `metal`)
- Filtered and re-annotated into a single 10-class COCO-format target dataset
- Resized images to 640×640 and normalised bounding boxes to centre-x, centre-y, width, height

The author-trained YOLO model weights (`legacy_django/backend/ml_model/weights/best_weights.weights.h5`) are a separate adaptation and are licensed by the project author under the project MIT License (see `LICENSE`).

## 4. Source Datasets (Roboflow Universe)

All projects sourced from workspace `yuyutsu-cxcso` on Roboflow Universe. Each project below is CC BY 4.0.

| Class | Roboflow Project | Version | Source URL | License |
|-------|------------------|---------|------------|---------|
| Sunglasses | `sunglasses-tvknx-wlwcu` | v1 | https://universe.roboflow.com/yuyutsu-cxcso/sunglasses-tvknx-wlwcu | CC BY 4.0 |
| Knife | `knifedetectionv4-6pfwh` | v3 | https://universe.roboflow.com/yuyutsu-cxcso/knifedetectionv4-6pfwh | CC BY 4.0 |
| Water Bottle | `water-bottle-qginj-6glhq` | v4 | https://universe.roboflow.com/yuyutsu-cxcso/water-bottle-qginj-6glhq | CC BY 4.0 |
| Pen | `pen-kkqzs-tvftk` | v3 | https://universe.roboflow.com/yuyutsu-cxcso/pen-kkqzs-tvftk | CC BY 4.0 |
| Chair | `chair-walws-ziqgx` | v1 | https://universe.roboflow.com/yuyutsu-cxcso/chair-walws-ziqgx | CC BY 4.0 |
| Human Face | `face-detection-mik1i-iuvar` | v1 | https://universe.roboflow.com/yuyutsu-cxcso/face-detection-mik1i-iuvar | CC BY 4.0 |
| Mobile Phone | `e-waste-mobile-phones-drlac` | v3 | https://universe.roboflow.com/yuyutsu-cxcso/e-waste-mobile-phones-drlac | CC BY 4.0 |
| Helmet | `helmet-motorcycle-detection-pro0q-ozzzn` | v3 | https://universe.roboflow.com/yuyutsu-cxcso/helmet-motorcycle-detection-pro0q-ozzzn | CC BY 4.0 |
| Fire | `fire-8klur-fceqf` | v3 | https://universe.roboflow.com/yuyutsu-cxcso/fire-8klur-fceqf | CC BY 4.0 |
| Can | `cans-avukd-b09pf` | v5 | https://universe.roboflow.com/yuyutsu-cxcso/cans-avukd-b09pf | CC BY 4.0 |

## 5. No Endorsement

This attribution is provided in compliance with CC BY 4.0 and does **not** imply that Roboflow, Roboflow Universe, or the original dataset creators endorse SmartVision AI, its author, or any use of the trained model.

## 6. Related Files

- `THIRD_PARTY_LICENSES.md` — full third-party IP and licensing inventory
- `ASSET_OWNERSHIP.md` — asset ownership and licensing declaration
- `legacy_django/model_evaluation/experiment_log.md` — model card with training details
- `LICENSE` — project MIT License (author code and author-trained weights only)

---

**Attributed by:** Yuyutsu Lamsal
**Project:** SmartVision AI
**Date:** 2026-11-04
