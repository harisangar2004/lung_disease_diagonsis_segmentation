from tensorflow.keras import layers, models

def build_classifier(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(224,224,1)),
        layers.MaxPooling2D(),
        layers.BatchNormalization(),

        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),

        # IMPORTANT CHANGE
        layers.Dense(num_classes, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',   # ✅ FIX
        metrics=['accuracy']
    )

    return model
