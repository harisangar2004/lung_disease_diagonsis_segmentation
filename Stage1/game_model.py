import tensorflow as tf
from tensorflow.keras import layers, Model

# ================================
# Attention Block
# ================================
def attention_block(x):
    att = layers.Dense(x.shape[-1], activation='tanh')(x)
    att = layers.Dense(x.shape[-1], activation='softmax')(att)
    return layers.Multiply()([x, att])


# ================================
# GAME Model
# ================================
def build_GAME(input_shape=(224, 224, 1)):

    inputs = layers.Input(shape=input_shape)

    # -------- Encoder --------
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)

    # -------- Attention --------
    x = attention_block(x)

    # -------- Decoder --------
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)

    outputs = layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)

    model = Model(inputs, outputs, name="GAME_Model")

    return model
