# SmartVision Model Card

**Author: Yuyutsu Lamsal**

## Dataset source

I got the data from 10 Roboflow projects in my workspace (`yuyutsu-cxcso`), one for each class, and merged them into a single dataset in COCO format.

Classes: sunglasses, knife, water bottle, pen, chair, human face, mobile phone, helmet, fire, can.

Each project is at `https://universe.roboflow.com/yuyutsu-cxcso/<project-name>`:

- sunglasses: sunglasses-tvknx-wlwcu (v1)
- knife: knifedetectionv4-6pfwh (v3)
- water bottle: water-bottle-qginj-6glhq (v4)
- pen: pen-kkqzs-tvftk (v3)
- chair: chair-walws-ziqgx (v1)
- human face: face-detection-mik1i-iuvar (v1)
- mobile phone: e-waste-mobile-phones-drlac (v3)
- helmet: helmet-motorcycle-detection-pro0q-ozzzn (v3)
- fire: fire-8klur-fceqf (v3)
- can: cans-avukd-b09pf (v5)

I dropped the extra labels `table`, `smoke` and `metal`. One thing to keep in mind: each image only has its own class labelled, so other objects in the same picture count as background.

## Train / val / test split

I used the splits Roboflow already gave me and joined them together.

| Split | Images | Annotations |
|---|---|---|
| Train | 27841 | 39463 |
| Validation | 2614 | 3229 |
| Test | 1782 |  2358 |

## Preprocessing choices

- Resized every image to 640x640 and scaled pixels to 0-1.
- Converted boxes to normalised centre-x, centre-y, width, height.
- Removed tiny boxes (width or height under 0.001).
- Kept up to 50 boxes per image.
- Augmentation: only a horizontal flip (50% chance).

## Hyperparameters

- Image size: 640
- Batch size: 16
- Epochs: 60
- Optimiser: Adam, learning rate 0.001
- Dropout: 0.2
- Gradient clipping: 10
- Anchors: 9 hand-set anchors, 3 per scale
- Loss: box (x5) + objectness + class
- Model size: 7,924,199 parameters

## Hardware used

- Google Colab with an A100 GPU (high-RAM)
- Training time: 4 hours 18 minutes (60 epochs)

## Final results

Evaluated on the test set at confidence threshold 0.25.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| sunglasses | 0.7187 | 0.3770 | 0.4946 |
| knife | 0.7273 | 0.2883 | 0.4129 |
| water bottle | 0.8929 | 0.4386 | 0.5882 |
| pen | 0.8231 | 0.3645 | 0.5052 |
| chair | 0.8625 | 0.6699 | 0.7541 |
| human face | 0.5579 | 0.4943 | 0.5242 |
| mobile phone | 0.6389 | 0.3333 | 0.4381 |
| helmet | 0.9140 | 0.7391 | 0.8173 |
| fire | 0.1961 | 0.3701 | 0.2564 |
| can | 0.8036 | 0.8654 | 0.8333 |
| **Mean** | **0.7135** | **0.4941** | **0.5624** |

Best validation loss: 1.042 (Epoch 52)

The model does best on helmet and can (F1 above 0.8) and worst on fire (F1 0.26). Precision is much higher than recall overall, so it misses a lot of objects but is usually right when it does detect something.