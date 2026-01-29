import os
import cv2
import numpy as np
import pandas as pd

IMG_SIZE = (224, 224)

CLASSES = [
    "Pneumonia",
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
    "No Finding"
]

# -----------------------------------
# Convert Kaggle → NIH format
# -----------------------------------
def load_kaggle_as_nih(base_path):
    images, labels = [], []

    for split in ["train", "val", "test"]:
        split_path = os.path.join(base_path, split)

        for cls in ["NORMAL", "PNEUMONIA"]:
            folder = os.path.join(split_path, cls)

            if not os.path.exists(folder):
                continue

            for file in os.listdir(folder):
                img_path = os.path.join(folder, file)

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, IMG_SIZE)
                img = img / 255.0

                label = np.zeros(15)
                if cls == "PNEUMONIA":
                    label[0] = 1
                else:
                    label[-1] = 1  # No Finding

                images.append(img)
                labels.append(label)

    return np.array(images)[..., None], np.array(labels)


# -----------------------------------
# Load NIH Dataset
# -----------------------------------
def load_nih_dataset(base_path, csv_path, limit=3000):
    df = pd.read_csv(csv_path)

    images, labels = [], []

    for _, row in df.iterrows():
        if len(images) >= limit:
            break

        img_name = row["Image Index"]
        label_text = row["Finding Labels"]

        img_path = None
        for folder in os.listdir(base_path):
            candidate = os.path.join(base_path, folder, "images", img_name)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0

        label = np.zeros(15)
        for d in label_text.split("|"):
            if d in CLASSES:
                label[CLASSES.index(d)] = 1

        images.append(img)
        labels.append(label)

    return np.array(images)[..., None], np.array(labels)
