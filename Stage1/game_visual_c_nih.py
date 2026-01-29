import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from preprocess_chest_xray import load_dataset
from preprocess_nih import load_nih_dataset

# --------------------------------------------------
# PATHS
# --------------------------------------------------
GAME_MODEL_PATH = "../models/game_nih_finetuned.h5"

# --------------------------------------------------
# LOAD GAME MODEL
# --------------------------------------------------
print("Loading GAME model...")
game_model = load_model(GAME_MODEL_PATH, compile=False)

# --------------------------------------------------
# LOAD KAGGLE DATA
# --------------------------------------------------
print("Loading Kaggle dataset...")
Xk, _, _, _, Xk_test, _ = load_dataset("../data/chest_xray")

# --------------------------------------------------
# LOAD NIH DATA
# --------------------------------------------------
print("Loading NIH dataset...")
Xn, _ = load_nih_dataset(
    "../data/NIH-chest_xray",
    "../data/NIH-chest_xray/Data_Entry_2017.csv",
    limit=5
)

# --------------------------------------------------
# FUNCTION TO VISUALIZE GAME OUTPUT
# --------------------------------------------------
def visualize_game(img, title):
    img = img[np.newaxis, ...]
    recon = game_model.predict(img)

    diff = np.abs(img - recon)
    diff = (diff - diff.min()) / (diff.max() - diff.min())

    plt.figure(figsize=(14, 4))
    plt.suptitle(title, fontsize=14)

    # Original
    plt.subplot(1, 4, 1)
    plt.title("Original")
    plt.imshow(img[0].squeeze(), cmap="gray")
    plt.axis("off")

    # Reconstructed
    plt.subplot(1, 4, 2)
    plt.title("Reconstructed")
    plt.imshow(recon[0].squeeze(), cmap="gray")
    plt.axis("off")

    # Difference
    plt.subplot(1, 4, 3)
    plt.title("Attention Map")
    plt.imshow(diff[0].squeeze(), cmap="hot")
    plt.axis("off")

    # Overlay
    plt.subplot(1, 4, 4)
    plt.title("Overlay")
    base = np.stack([img[0].squeeze()]*3, axis=-1)
    heat = plt.cm.jet(diff[0].squeeze())[:, :, :3]
    overlay = 0.6 * base + 0.4 * heat
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# VISUALIZE BOTH DATASETS
# --------------------------------------------------
print("Visualizing Kaggle sample...")
visualize_game(Xk_test[0], "Kaggle Chest X-ray")

print("Visualizing NIH sample...")
visualize_game(Xn[0], "NIH Chest X-ray")
