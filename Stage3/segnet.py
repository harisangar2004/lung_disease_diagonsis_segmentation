import tensorflow as tf
from tensorflow.keras import layers, Model

def build_segnet(input_shape=(224, 224, 1)):

    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)

    x = layers.UpSampling2D()(x)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
