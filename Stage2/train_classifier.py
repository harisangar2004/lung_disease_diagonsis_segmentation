import sys
import os
sys.path.append(os.path.abspath("../Stage1"))

import tensorflow as tf
from preprocess_chest_xray import load_dataset
from game_model import build_GAME
from classifier_model import build_classifier

# Load dataset
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
    "../data/chest_xray"
)

# Load GAME model
game_model = tf.keras.models.load_model("../models/game_model.h5", compile=False)

# Extract features
X_train_game = game_model.predict(X_train)
X_val_game = game_model.predict(X_val)
X_test_game = game_model.predict(X_test)

# Train classifier
classifier = build_classifier(num_classes=2)
classifier.fit(
    X_train_game, y_train,
    validation_data=(X_val_game, y_val),
    epochs=10,
    batch_size=16
)

classifier.save("../models/cnn_classifier.h5")
print("✅ Classifier saved successfully")
