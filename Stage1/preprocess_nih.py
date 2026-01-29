import os
import cv2
import numpy as np
import pandas as pd

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
IMG_SIZE = (224, 224)

CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
    "No Finding"
]

# --------------------------------------------------
# Encode labels (Multi-label)
# --------------------------------------------------
def encode_labels(label_string):
    labels = label_string.split("|")
    return np.array([1 if cls in labels else 0 for cls in CLASSES])

# --------------------------------------------------
# Load NIH Dataset
# --------------------------------------------------
def load_nih_dataset(base_path, csv_path, limit=None):
    print("Loading NIH Chest X-ray dataset...")

    df = pd.read_csv(csv_path)

    X = []
    y = []

    for idx, row in df.iterrows():
        img_name = row["Image Index"]
        label_text = row["Finding Labels"]

        # Locate image
        img_path = None
        for folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder, "images")
            if not os.path.isdir(folder_path):
                continue

            candidate = os.path.join(folder_path, img_name)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            continue

        # Read image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)

        # Normalize
        img = (img - np.mean(img)) / (np.std(img) + 1e-7)

        X.append(img)
        y.append(encode_labels(label_text))

        if limit and len(X) >= limit:
            break

    X = np.array(X)[..., np.newaxis]
    y = np.array(y)

    print("✅ NIH Dataset Loaded Successfully")
    print("Images shape :", X.shape)
    print("Labels shape :", y.shape)

    return X, y

# --------------------------------------------------
# TEST THE MODULE
# --------------------------------------------------
if __name__ == "__main__":
    X, y = load_nih_dataset(
        base_path="../data/NIH-chest_xray",
        csv_path="../data/NIH-chest_xray/Data_Entry_2017.csv",
        limit=500
    )

    print("\nSample Label Vector:")
    print(y[0])

    print("\nDiseases in sample:")
    for i, val in enumerate(y[0]):
        if val == 1:
            print(" -", CLASSES[i])
