import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocess_nih import load_nih_dataset

# --------------------------------------------------
# PATHS
# --------------------------------------------------
GAME_MODEL_PATH = "../models/game_model.h5"
SAVE_PATH = "../models/game_nih_finetuned.h5"

NIH_PATH = "../data/NIH-chest_xray"
CSV_PATH = "../data/NIH-chest_xray/Data_Entry_2017.csv"

# --------------------------------------------------
# LOAD NIH DATA
# --------------------------------------------------
print("Loading NIH dataset...")
X_nih, _ = load_nih_dataset(
    base_path=NIH_PATH,
    csv_path=CSV_PATH,
    limit=2000
)

# --------------------------------------------------
# LOAD GAME MODEL (IMPORTANT FIX)
# --------------------------------------------------
print("Loading pretrained GAME model...")
game_model = load_model(GAME_MODEL_PATH, compile=False)

# --------------------------------------------------
# FREEZE EARLY LAYERS
# --------------------------------------------------
for layer in game_model.layers[:-4]:
    layer.trainable = False

# --------------------------------------------------
# RE-COMPILE MODEL
# --------------------------------------------------
game_model.compile(
    optimizer="adam",
    loss="mse"
)

# --------------------------------------------------
# FINE-TUNE
# --------------------------------------------------
print("Fine-tuning GAME on NIH dataset...")

game_model.fit(
    X_nih, X_nih,
    epochs=5,
    batch_size=8
)

# --------------------------------------------------
# SAVE UPDATED MODEL
# --------------------------------------------------
game_model.save(SAVE_PATH)
print("✅ GAME fine-tuned and saved as:", SAVE_PATH)
