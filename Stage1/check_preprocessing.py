import matplotlib.pyplot as plt
from preprocess_chest_xray import load_dataset

# Load dataset
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
    "../data/chest_xray"
)

# ==============================
# 1. SHAPE CHECK
# ==============================
print("\n--- DATA SHAPES ---")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# ==============================
# 2. PIXEL VALUE CHECK
# ==============================
print("\n--- PIXEL STATS ---")
print("Min:", X_train.min())
print("Max:", X_train.max())
print("Mean:", X_train.mean())

# ==============================
# 3. LABEL CHECK
# ==============================
print("\n--- LABEL SAMPLE ---")
print(y_train[:10])

# ==============================
# 4. VISUALIZATION
# ==============================
plt.figure(figsize=(5,5))
plt.imshow(X_train[0].squeeze(), cmap='gray')
plt.title("Sample Preprocessed Image")
plt.axis("off")
plt.show()
