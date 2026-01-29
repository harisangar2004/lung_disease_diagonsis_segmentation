import sys
import os
sys.path.append(os.path.abspath("../Stage1"))

import tensorflow as tf
from preprocess_unified import load_kaggle_as_nih, load_nih_dataset
from classifier_unified import build_unified_classifier

# ----------------------------
# Load datasets
# ----------------------------
Xk, yk = load_kaggle_as_nih("../data/chest_xray")

Xn, yn = load_nih_dataset(
    "../data/NIH-chest_xray",
    "../data/NIH-chest_xray/Data_Entry_2017.csv",
    limit=3000
)

X = tf.concat([Xk, Xn], axis=0)
y = tf.concat([yk, yn], axis=0)

print("Final Dataset:", X.shape, y.shape)

# ----------------------------
# Load GAME
# ----------------------------
game = tf.keras.models.load_model(
    "../models/game_nih_finetuned.h5",
    compile=False
)

X_game = game.predict(X)

# ----------------------------
# Train classifier
# ----------------------------
model = build_unified_classifier()
model.fit(X_game, y, epochs=10, batch_size=16)

model.save("../models/unified_classifier.h5")
print("✅ Unified classifier trained successfully")
