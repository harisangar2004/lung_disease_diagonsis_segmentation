import sys, os
sys.path.append(os.path.abspath("../Stage1"))

import numpy as np
import tensorflow as tf
from preprocess_chest_xray import load_dataset
from preprocess_nih import load_nih_dataset

# ----------------------------
# CONFIG
# ----------------------------
DATASET = "kaggle"   # change to "nih"

# ----------------------------
# LOAD DATA
# ----------------------------
if DATASET == "kaggle":
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
        "../data/chest_xray"
    )
    classifier = tf.keras.models.load_model("../models/cnn_classifier.h5")

else:
    X_test, y_test = load_nih_dataset(
        "../data/NIH-chest_xray",
        "../data/NIH-chest_xray/Data_Entry_2017.csv",
        limit=2000
    )
    classifier = tf.keras.models.load_model("../models/cnn_classifier_nih.h5")

# ----------------------------
# EVALUATION
# ----------------------------
loss, acc = classifier.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {acc*100:.2f}%")
