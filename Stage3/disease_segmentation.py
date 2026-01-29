import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# Allow Stage1 imports
sys.path.append(os.path.abspath("../Stage1"))

from preprocess_chest_xray import load_dataset
from tensorflow.keras.models import load_model
from unet import build_unet
from segnet import build_segnet

# --------------------------------------------------
# PATHS
# --------------------------------------------------
MODEL_PATH = "../models"
UNET_PATH = os.path.join(MODEL_PATH, "unet_model.h5")
SEGNET_PATH = os.path.join(MODEL_PATH, "segnet_model.h5")

os.makedirs(MODEL_PATH, exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
print("Loading dataset...")
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
    "../data/chest_xray"
)

# --------------------------------------------------
# LOAD / TRAIN U-NET
# --------------------------------------------------
if os.path.exists(UNET_PATH):
    print("Loading saved U-Net...")
    unet = load_model(UNET_PATH)
else:
    print("Training U-Net...")
    unet = build_unet()
    unet.fit(X_train, X_train, epochs=3, batch_size=8)
    unet.save(UNET_PATH)
    print("U-Net saved!")

# --------------------------------------------------
# LOAD / TRAIN SEGNET
# --------------------------------------------------
if os.path.exists(SEGNET_PATH):
    print("Loading saved SegNet...")
    segnet = load_model(SEGNET_PATH)
else:
    print("Training SegNet...")
    segnet = build_segnet()
    segnet.fit(X_train, X_train, epochs=3, batch_size=8)
    segnet.save(SEGNET_PATH)
    print("SegNet saved!")

# --------------------------------------------------
# ENSEMBLE SEGMENTATION
# --------------------------------------------------
print("Generating ensemble mask...")

pred_unet = unet.predict(X_test)
pred_segnet = segnet.predict(X_test)

lung_mask = ((pred_unet + pred_segnet) / 2 > 0.5).astype(np.uint8)

# --------------------------------------------------
# DISEASE SEGMENTATION
# --------------------------------------------------
def segment_disease(image, lung_mask):
    lung_only = image * lung_mask
    lung_only = cv2.normalize(lung_only, None, 0, 255, cv2.NORM_MINMAX)
    lung_only = lung_only.astype(np.uint8)

    # Threshold for disease region
    _, disease = cv2.threshold(lung_only, 180, 255, cv2.THRESH_BINARY)

    # Morphological cleaning
    kernel = np.ones((5, 5), np.uint8)
    disease = cv2.morphologyEx(disease, cv2.MORPH_OPEN, kernel)
    disease = cv2.morphologyEx(disease, cv2.MORPH_CLOSE, kernel)

    return disease

# --------------------------------------------------
# APPLY ON SAMPLE IMAGE
# --------------------------------------------------
img = X_test[0].squeeze()
lung = lung_mask[0].squeeze()
disease = segment_disease(img, lung)

# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------
plt.figure(figsize=(14, 4))

plt.subplot(1, 4, 1)
plt.title("Original X-ray")
plt.imshow(img, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("Lung Mask")
plt.imshow(lung, cmap="gray")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("Disease Region")
plt.imshow(disease, cmap="hot")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Overlay")
overlay = img.copy()
overlay[disease > 0] = 1
plt.imshow(overlay, cmap="hot")
plt.axis("off")

plt.tight_layout()
plt.show()
