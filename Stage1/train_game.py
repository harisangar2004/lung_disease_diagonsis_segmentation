import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from preprocess_chest_xray import load_dataset
from game_model import build_GAME

# ================================
# Load Dataset
# ================================
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
    "../data/chest_xray"
)

print("Dataset Loaded")

# ================================
# Build GAME Model
# ================================
game_model = build_GAME()
game_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mse"
)

game_model.summary()

# ================================
# Train GAME
# ================================
history = game_model.fit(
    X_train,
    X_train,           # Autoencoder target = input
    validation_data=(X_val, X_val),
    epochs=10,
    batch_size=16
)

# ================================
# Save Model
# ================================
game_model.save("../models/game_model.h5")
print("✅ GAME model saved successfully")

# ================================
# Visualize Reconstruction
# ================================
reconstructed = game_model.predict(X_test[:5])

plt.figure(figsize=(10,4))
for i in range(5):
    # Original
    plt.subplot(2,5,i+1)
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    plt.title("Original")
    plt.axis("off")

    # Reconstructed
    plt.subplot(2,5,i+6)
    plt.imshow(reconstructed[i].squeeze(), cmap='gray')
    plt.title("Reconstructed")
    plt.axis("off")

plt.show()
