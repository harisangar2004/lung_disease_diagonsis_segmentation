import sys
import os
sys.path.append(os.path.abspath("../Stage1"))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tensorflow as tf

from preprocess_chest_xray import load_dataset

# ----------------------------------
# ROI GENERATION FUNCTION
# ----------------------------------
def generate_roi(image, k=4):
    """
    Applies K-Means clustering to extract ROI
    """
    flat = image.reshape((-1, 1))

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(flat)

    labels = kmeans.labels_.reshape(image.shape)

    # Take the brightest cluster as ROI
    roi = (labels == labels.max()).astype(np.uint8)

    return roi


# ----------------------------------
# MAIN EXECUTION
# ----------------------------------
if __name__ == "__main__":

    # Load dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
        "../data/chest_xray"
    )

    # Load GAME model
    game_model = tf.keras.models.load_model(
        "../models/game_model.h5", compile=False
    )

    # Select one test image
    img = X_test[0]
    img = img.squeeze()

    # Apply GAME
    game_out = game_model.predict(X_test[0:1])[0].squeeze()

    # Generate ROI
    roi = generate_roi(game_out)

    # Apply ROI
    roi_image = game_out * roi

    # ----------------------------------
    # Visualization
    # ----------------------------------
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("ROI Mask")
    plt.imshow(roi, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("ROI Applied")
    plt.imshow(roi_image, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
