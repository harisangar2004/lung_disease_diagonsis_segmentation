import sys
import os
sys.path.append(os.path.abspath("../Stage1"))

import tensorflow as tf
from preprocess_nih import load_nih_dataset
from classifier_model import build_classifier

# -----------------------------
# LOAD NIH DATA
# -----------------------------
X, y = load_nih_dataset(
    "../data/NIH-chest_xray",
    "../data/NIH-chest_xray/Data_Entry_2017.csv",
    limit=3000
)

# -----------------------------
# LOAD GAME MODEL
# -----------------------------
game_model = tf.keras.models.load_model(
    "../models/game_nih_finetuned.h5",
    compile=False
)

# -----------------------------
# EXTRACT GAME FEATURES
# -----------------------------
X_game = game_model.predict(X)

# -----------------------------
# BUILD CLASSIFIER
# -----------------------------
classifier = build_classifier(num_classes=15)

# -----------------------------
# TRAIN
# -----------------------------
classifier.fit(
    X_game,
    y,
    epochs=10,
    batch_size=16
)

# -----------------------------
# SAVE
# -----------------------------
classifier.save("../models/cnn_classifier_nih.h5")
print("✅ NIH classifier trained successfully")
