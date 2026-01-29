import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sys, os
import random

# -----------------------------------
# PATH FIX
# -----------------------------------
sys.path.append(os.path.abspath("../Stage1"))

from preprocess_unified import load_kaggle_as_nih, load_nih_dataset

# -----------------------------------
# LABELS
# -----------------------------------
CLASSES = [
    "Pneumonia", "Atelectasis", "Cardiomegaly", "Effusion",
    "Infiltration", "Mass", "Nodule", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia", "No Finding"
]

# -----------------------------------
# LOAD DATA
# -----------------------------------
print("Loading datasets...")

Xk, yk = load_kaggle_as_nih("../data/chest_xray")
Xn, yn = load_nih_dataset(
    "../data/NIH-chest_xray",
    "../data/NIH-chest_xray/Data_Entry_2017.csv",
    limit=100
)

X = np.concatenate([Xk, Xn])
y_true = np.concatenate([yk, yn])

# -----------------------------------
# LOAD MODELS
# -----------------------------------
print("Loading models...")

game = tf.keras.models.load_model(
    "../models/game_nih_finetuned.h5",
    compile=False
)

classifier = tf.keras.models.load_model(
    "../models/unified_classifier.h5"
)

# -----------------------------------
# PREDICTION
# -----------------------------------
X_game = game.predict(X)
y_pred = classifier.predict(X_game)

# -----------------------------------
# VISUALIZATION (OPTIMIZED)
# -----------------------------------
print("Visualizing predictions...")

for _ in range(5):

    # Prefer NIH samples (more disease variety)
    idx = random.randint(len(Xk), len(X) - 1)

    image = X[idx]
    true_labels = y_true[idx]
    pred_scores = y_pred[idx]

    # ---------- Actual labels ----------
    actual = [CLASSES[i] for i in range(15) if true_labels[i] == 1]
    if not actual:
        actual = ["No Finding"]

    # ---------- Top-K Predictions ----------
    top_k = np.argsort(pred_scores)[-5:][::-1]
    predicted = [
        f"{CLASSES[i]} ({pred_scores[i]:.2f})"
        for i in top_k
    ]

    # ---------- Plot ----------
    plt.figure(figsize=(11, 4))

    # Image
    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title("Chest X-ray")
    plt.axis("off")

    # Actual
    plt.subplot(1, 3, 2)
    plt.title("Actual Labels")
    plt.text(0.05, 0.9, "\n".join(actual),
             fontsize=11, verticalalignment="top")
    plt.axis("off")

    # Predicted
    plt.subplot(1, 3, 3)
    plt.title("Predicted (Top-5)")
    plt.text(0.05, 0.9, "\n".join(predicted),
             fontsize=11, verticalalignment="top")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
