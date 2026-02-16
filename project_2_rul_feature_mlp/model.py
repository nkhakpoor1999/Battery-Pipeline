# model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Dense

def build_rul_model(input_dim, l2_reg=1e-4):
    inp = Input(shape=(input_dim,))
    x = Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(inp)
    x = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = Dense(16, activation="relu", kernel_regularizer=regularizers.l2(l2_reg))(x)
    out = Dense(1, activation="linear")(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def train_final_model(X, y, chosen_groups, feat_idx, final_epochs, batch_size, l2_reg=1e-4, verbose=0):
    from sklearn.preprocessing import StandardScaler

    X_sel = X[:, feat_idx]
    x_scaler = StandardScaler()
    X_sc = x_scaler.fit_transform(X_sel)

    y = np.asarray(y).reshape(-1, 1)

    model = build_rul_model(input_dim=X_sc.shape[1], l2_reg=l2_reg)

    history = model.fit(
        X_sc, y,
        epochs=int(final_epochs),
        batch_size=batch_size,
        verbose=verbose
    )

    print(f"[Final Train] Using epochs = {int(final_epochs)}")
    return model, x_scaler, history
