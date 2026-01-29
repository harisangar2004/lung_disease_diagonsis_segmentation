import sys
import os
sys.path.append(os.path.abspath("../Stage1"))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from preprocess_chest_xray import load_dataset
from preprocess_nih import load_nih_dataset

# -------------------------------
# CONFIG
# -------------------------------
SHOW_SAMPLES = 5

# -------------------------------
# LOAD MODELS
# -------------------------------
game_model = tf.keras.models.load_model("../models/game_nih_finetuned.h5", compile=False)
kaggle_classifier = tf.keras.models.load_model("../models/cnn_classifier.h5")
nih_classifier = tf.keras.models.load_model("../models/cnn_classifier_nih.h5")

# -------------------------------
# LOAD DATASETS
# -------------------------------
print("Loading Kaggle dataset...")
Xk, yk, _, _, _, _ = load_dataset("../data/chest_xray")

print("Loading NIH dataset...")
Xn, yn = load_nih_dataset(
    "../data/NIH-chest_xray",
    "../data/NIH-chest_xray/Data_Entry_2017.csv",
    limit=5
)

# -------------------------------
# KAGGLE VISUALIZATION
# -------------------------------
print("Visualizing Kaggle Predictions...")

Xk_game = game_model.predict(Xk)
preds_kaggle = kaggle_classifier.predict(Xk_game)

plt.figure(figsize=(12, 6))

for i in range(SHOW_SAMPLES):
    pred = preds_kaggle[i][0]
    pred_label = "PNEUMONIA" if pred > 0.5 else "NORMAL"
    actual = "PNEUMONIA" if yk[i] == 1 else "NORMAL"

    plt.subplot(2, 3, i+1)
    plt.imshow(Xk[i].squeeze(), cmap="gray")
    plt.title(f"Pred: {pred_label}\nActual: {actual}")
    plt.axis("off")

plt.suptitle("Kaggle Chest X-ray Predictions", fontsize=14)
plt.tight_layout()
plt.show()

# -------------------------------
# NIH VISUALIZATION
# -------------------------------
print("Visualizing NIH Predictions...")

Xn_game = game_model.predict(Xn)
preds_nih = nih_classifier.predict(Xn_game)

NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion",
    "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural Thickening",
    "Hernia", "No Finding"
]

plt.figure(figsize=(12, 8))

for i in range(SHOW_SAMPLES):
    pred = preds_nih[i]
    detected = [NIH_LABELS[j] for j in range(len(pred)) if pred[j] > 0.5]
    detected_text = ", ".join(detected) if detected else "No Finding"

    plt.subplot(2, 3, i+1)
    plt.imshow(Xn[i].squeeze(), cmap="gray")
    plt.title(f"Detected:\n{detected_text}", fontsize=9)
    plt.axis("off")

plt.suptitle("NIH Chest X-ray Predictions", fontsize=14)
plt.tight_layout()
plt.show()
