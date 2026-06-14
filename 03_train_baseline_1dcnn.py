import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X = np.load("data/processed/X_full.npy")
y = np.load("data/processed/y_full.npy")

X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
X = X[..., None]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64, 3, activation="relu", input_shape=X.shape[1:]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(128, 3, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(256, 3, activation="relu"),
    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=256
)

y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

model.save("models/cnn_300bands.h5")
