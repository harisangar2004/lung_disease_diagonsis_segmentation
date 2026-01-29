import numpy as np
import matplotlib.pyplot as plt
from preprocess_nih import load_nih_dataset

# ----------------------------------
# LOAD DATASET
# ----------------------------------
X, y = load_nih_dataset(
    base_path="../data/NIH-chest_xray",
    csv_path="../data/NIH-chest_xray/Data_Entry_2017.csv",
    limit=500   # reduce for fast testing
)

# ----------------------------------
# BASIC INFO
# ----------------------------------
print("\n===== DATASET INFO =====")
print("Images shape:", X.shape)
print("Labels shape:", y.shape)
print("Min pixel value:", X.min())
print("Max pixel value:", X.max())
print("Mean pixel value:", X.mean())

# ----------------------------------
# LABEL NAMES
# ----------------------------------
CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion",
    "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening",
    "Hernia", "No Finding"
]

# ----------------------------------
# LABEL DISTRIBUTION
# ----------------------------------
print("\n===== LABEL DISTRIBUTION =====")
label_count = np.sum(y, axis=0)

for cls, count in zip(CLASSES, label_count):
    print(f"{cls:20s} : {count}")

# ----------------------------------
# VISUALIZE SAMPLE IMAGES
# ----------------------------------
plt.figure(figsize=(12, 6))

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(X[i].squeeze(), cmap="gray")
    plt.axis("off")

    # Show label names
    label_names = [CLASSES[j] for j in range(len(CLASSES)) if y[i][j] == 1]
    plt.title(", ".join(label_names), fontsize=8)

plt.suptitle("NIH Chest X-ray Samples", fontsize=14)
plt.tight_layout()
plt.show()

# ----------------------------------
# CHECK SINGLE SAMPLE
# ----------------------------------
print("\n===== SAMPLE LABEL VECTOR =====")
print("Label vector:", y[0])
print("Diseases:", [CLASSES[i] for i in range(15) if y[0][i] == 1])
