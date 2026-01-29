import sys
import os
sys.path.append(os.path.abspath("../Stage1"))

import tensorflow as tf
from preprocess_chest_xray import load_dataset

# Load data
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
    "../data/chest_xray"
)

# Load models
game_model = tf.keras.models.load_model("../models/game_model.h5", compile=False)
classifier = tf.keras.models.load_model("../models/cnn_classifier.h5")

# Extract GAME features
X_test_game = game_model.predict(X_test)

# Evaluate
loss, acc = classifier.evaluate(X_test_game, y_test)

print(f"✅ Test Accuracy: {acc*100:.2f}%")
