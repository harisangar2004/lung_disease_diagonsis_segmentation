import sys
import os
sys.path.append(os.path.abspath("../Stage1"))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from preprocess_chest_xray import load_dataset

# ----------------------------
# Load Dataset
# ----------------------------
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
    "../data/chest_xray"
)

# ----------------------------
# Load Models
# ----------------------------
game_model = tf.keras.models.load_model("../models/game_model.h5", compile=False)
classifier = tf.keras.models.load_model("../models/cnn_classifier.h5")

# ----------------------------
# Predict
# ----------------------------
X_test_game = game_model.predict(X_test)
y_pred = classifier.predict(X_test_game)

y_pred = np.argmax(y_pred, axis=1)
y_true = y_test.flatten()

# ----------------------------
# Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NORMAL", "PNEUMONIA"]
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
