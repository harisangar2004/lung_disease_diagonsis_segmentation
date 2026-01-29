import sys
import os
sys.path.append(os.path.abspath("../Stage1"))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix

from preprocess_unified import load_kaggle_as_nih, load_nih_dataset

# ------------------------------------
# LOAD DATA
# ------------------------------------
Xk, yk = load_kaggle_as_nih("../data/chest_xray")

Xn, yn = load_nih_dataset(
    "../data/NIH-chest_xray",
    "../data/NIH-chest_xray/Data_Entry_2017.csv",
    limit=3000
)

X = np.concatenate([Xk, Xn])
y_true = np.concatenate([yk, yn])

# ------------------------------------
# LOAD MODELS
# ------------------------------------
game = tf.keras.models.load_model(
    "../models/game_nih_finetuned.h5",
    compile=False
)

classifier = tf.keras.models.load_model(
    "../models/unified_classifier.h5"
)

# ------------------------------------
# PREDICTION
# ------------------------------------
X_game = game.predict(X)
y_pred = classifier.predict(X_game)
y_pred_bin = (y_pred > 0.5).astype(int)

# ------------------------------------
# CLASS LABELS
# ------------------------------------
classes = [
    "Pneumonia", "Atelectasis", "Cardiomegaly", "Effusion",
    "Infiltration", "Mass", "Nodule", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia", "No Finding"
]

# ------------------------------------
# CONFUSION MATRICES
# ------------------------------------
for i, disease in enumerate(classes):
    cm = confusion_matrix(y_true[:, i], y_pred_bin[:, i])

    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"]
    )
    plt.title(f"Confusion Matrix – {disease}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
