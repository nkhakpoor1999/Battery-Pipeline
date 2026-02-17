from __future__ import annotations

from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model


def build_brand_model(input_dim: int, num_classes: int, l2_reg: float = 1e-4) -> Model:
    inp = Input(shape=(input_dim,))
    x = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(inp)
    x = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inp, out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model
