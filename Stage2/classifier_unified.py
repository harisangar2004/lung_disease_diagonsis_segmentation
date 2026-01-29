from tensorflow.keras import layers, models

def build_unified_classifier():
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(224,224,1)),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),

        layers.Dense(15, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
