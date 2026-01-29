import sys
import os
sys.path.append(os.path.abspath("../Stage1"))

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from preprocess_chest_xray import load_dataset
from unet import build_unet
from segnet import build_segnet

# -------------------------------
# LOAD DATA
# -------------------------------
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
    "../data/chest_xray"
)

# -------------------------------
# BUILD MODELS
# -------------------------------
unet = build_unet()
segnet = build_segnet()

print("Training U-Net...")
unet.fit(X_train, X_train, epochs=3, batch_size=8)
unet.save("../models/unet1.h5")
print("U-Net saved as unet1.h5")

print("Training SegNet...")
segnet.fit(X_train, X_train, epochs=3, batch_size=8)
segnet.save("../models/segnet2.h5")
print("SegNet saved as segnet2.h5")

# -------------------------------
# ENSEMBLE PREDICTION
# -------------------------------
pred_unet = unet.predict(X_test)
pred_segnet = segnet.predict(X_test)

ensemble_output = 0.5 * pred_unet + 0.5 * pred_segnet
final_mask = (ensemble_output > 0.5).astype("uint8")

# -------------------------------
# VISUALIZATION
# -------------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1,4,1)
plt.title("Original")
plt.imshow(X_test[0].squeeze(), cmap='gray')
plt.axis("off")

plt.subplot(1,4,2)
plt.title("U-Net Output")
plt.imshow(pred_unet[0].squeeze(), cmap='gray')
plt.axis("off")

plt.subplot(1,4,3)
plt.title("SegNet Output")
plt.imshow(pred_segnet[0].squeeze(), cmap='gray')
plt.axis("off")

plt.subplot(1,4,4)
plt.title("Final Ensemble")
plt.imshow(final_mask[0].squeeze(), cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()
