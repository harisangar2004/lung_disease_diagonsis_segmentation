import sys
import os
sys.path.append(os.path.abspath("../Stage1"))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tensorflow as tf

from preprocess_unified import load_kaggle_as_nih, load_nih_dataset


# -------------------------------
# ROI GENERATION
# -------------------------------
def generate_roi(feature_map, k=4):
    h, w = feature_map.shape
    flat = feature_map.reshape(-1, 1)

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(flat)
    labels = labels.reshape(h, w)

    cluster_means = [feature_map[labels == i].mean() for i in range(k)]
    roi_cluster = np.argmax(cluster_means)

    roi = np.zeros_like(feature_map)
    roi[labels == roi_cluster] = 1

    kernel = np.ones((5, 5), np.uint8)
    roi = cv2.morphologyEx(roi.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)

    return roi


# -------------------------------
# LOAD DATA
# -------------------------------
Xk, _ = load_kaggle_as_nih("../data/chest_xray")
Xn, _ = load_nih_dataset(
    "../data/NIH-chest_xray",
    "../data/NIH-chest_xray/Data_Entry_2017.csv",
    limit=50
)

X = np.concatenate([Xk, Xn])

# -------------------------------
# LOAD GAME MODEL
# -------------------------------
game_model = tf.keras.models.load_model(
    "../models/game_nih_finetuned.h5",
    compile=False
)

# -------------------------------
# VISUALIZATION (3 IMAGES)
# -------------------------------
num_samples = 3
indices = np.random.choice(len(X), num_samples, replace=False)

plt.figure(figsize=(16, 4 * num_samples))

for i, idx in enumerate(indices):
    image = X[idx].squeeze()
    feature_map = game_model.predict(X[idx:idx+1])[0].squeeze()
    roi_mask = generate_roi(feature_map)
    roi_applied = image * roi_mask

    # -------------------
    # ORIGINAL
    plt.subplot(num_samples, 4, i * 4 + 1)
    plt.title("Original X-ray")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    # FEATURE MAP
    plt.subplot(num_samples, 4, i * 4 + 2)
    plt.title("GAME Feature Map")
    plt.imshow(feature_map, cmap="hot")
    plt.axis("off")

    # ROI MASK
    plt.subplot(num_samples, 4, i * 4 + 3)
    plt.title("ROI Mask")
    plt.imshow(roi_mask, cmap="gray")
    plt.axis("off")

    # ROI APPLIED
    plt.subplot(num_samples, 4, i * 4 + 4)
    plt.title("ROI Applied")
    plt.imshow(roi_applied, cmap="hot")
    plt.axis("off")

plt.tight_layout()
plt.show()
