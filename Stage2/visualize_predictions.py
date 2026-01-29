import sys
import os
sys.path.append(os.path.abspath("../Stage1"))

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from preprocess_chest_xray import load_dataset

# Load data
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
    "../data/chest_xray"
)

# Load models
game_model = tf.keras.models.load_model("../models/game_model.h5", compile=False)
classifier = tf.keras.models.load_model("../models/cnn_classifier.h5")

# Predict
X_test_game = game_model.predict(X_test)
preds = classifier.predict(X_test_game)
pred_labels = np.argmax(preds, axis=1)

# ----------------------------
# Visualization
# ----------------------------
for i in range(5):
    plt.figure(figsize=(3,3))
    plt.imshow(X_test[i].squeeze(), cmap="gray")

    actual = "PNEUMONIA" if y_test[i] == 1 else "NORMAL"
    predicted = "PNEUMONIA" if pred_labels[i] == 1 else "NORMAL"

    plt.title(f"Predicted: {predicted}\nActual: {actual}")
    plt.axis("off")
    plt.show()
