import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Access Stage1
sys.path.append(os.path.abspath("../Stage1"))

from preprocess_chest_xray import load_dataset
from tensorflow.keras.models import load_model

# -------------------------------
# LOAD DATA
# -------------------------------
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
    "../data/chest_xray"
)

# -------------------------------
# LOAD TRAINED MODELS
# -------------------------------
unet = load_model("../models/unet_model.h5")
segnet = load_model("../models/segnet_model.h5")

# -------------------------------
# PREDICTION
# -------------------------------
pred_unet = unet.predict(X_test)
pred_segnet = segnet.predict(X_test)

lung_mask = ((pred_unet + pred_segnet) / 2 > 0.5).astype(np.uint8)

# -------------------------------
# DISEASE EXTRACTION FUNCTION
# -------------------------------
def extract_disease(image, lung_mask):
    lung_only = image * lung_mask
    lung_only = cv2.normalize(lung_only, None, 0, 255, cv2.NORM_MINMAX)
    lung_only = lung_only.astype(np.uint8)

    _, disease = cv2.threshold(lung_only, 160, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5), np.uint8)
    disease = cv2.morphologyEx(disease, cv2.MORPH_OPEN, kernel)
    disease = cv2.morphologyEx(disease, cv2.MORPH_CLOSE, kernel)

    return disease

# -------------------------------
# VISUALIZE 3 IMAGES
# -------------------------------
num_images = 3
plt.figure(figsize=(14, 4 * num_images))

for i in range(num_images):

    img = X_test[i].squeeze()
    lung = lung_mask[i].squeeze()
    disease = extract_disease(img, lung)

    # Original
    plt.subplot(num_images, 4, i*4 + 1)
    plt.title("Original X-ray")
    plt.imshow(img, cmap='gray')
    plt.axis("off")

    # Lung Mask
    plt.subplot(num_images, 4, i*4 + 2)
    plt.title("Lung Mask")
    plt.imshow(lung, cmap='gray')
    plt.axis("off")

    # Disease Region
    plt.subplot(num_images, 4, i*4 + 3)
    plt.title("Disease Region")
    plt.imshow(disease, cmap='hot')
    plt.axis("off")

    # Overlay
    plt.subplot(num_images, 4, i*4 + 4)
    plt.title("Overlay")
    overlay = img.copy()
    overlay[disease > 0] = 1
    plt.imshow(overlay, cmap='hot')
    plt.axis("off")

plt.tight_layout()
plt.show()
