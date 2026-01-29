import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer

# ==============================
# CONFIGURATION
# ==============================
BASE_PATH = "../data/chest_xray"   # ✅ CORRECT PATH
IMG_SIZE = (224, 224)

# ==============================
# Load images from folder
# ==============================
def load_images_from_folder(folder_path):
    images = []
    labels = []

    for label in os.listdir(folder_path):
        class_path = os.path.join(folder_path, label)

        if not os.path.isdir(class_path):
            continue

        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)

            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, IMG_SIZE)

                # Normalize
                img = (img - np.mean(img)) / (np.std(img) + 1e-7)

                images.append(img)
                labels.append(label)

    return np.array(images), np.array(labels)


# ==============================
# Load Dataset
# ==============================
def load_dataset(base_path):
    print("Loading training data...")
    X_train, y_train = load_images_from_folder(
        os.path.join(base_path, "train")
    )

    print("Loading validation data...")
    X_val, y_val = load_images_from_folder(
        os.path.join(base_path, "val")
    )

    print("Loading test data...")
    X_test, y_test = load_images_from_folder(
        os.path.join(base_path, "test")
    )

    # Expand dims
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Encode labels
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_val = lb.transform(y_val)
    y_test = lb.transform(y_test)

    print("\n✅ Dataset Loaded Successfully")
    print("Classes:", lb.classes_)
    print("Train:", X_train.shape)
    print("Val  :", X_val.shape)
    print("Test :", X_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    load_dataset(BASE_PATH)
