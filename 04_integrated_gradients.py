import tensorflow as tf
import numpy as np
import pandas as pd

model = tf.keras.models.load_model("models/cnn_300bands.h5")
X = np.load("data/processed/X_full.npy")[:2000]
X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
X = X[..., None]

def integrated_gradients(model, x, steps=100):
    baseline = tf.zeros_like(x)
    grads = []

    for i in range(steps):
        alpha = i / steps
        with tf.GradientTape() as tape:
            x_interp = baseline + alpha * (x - baseline)
            tape.watch(x_interp)
            pred = model(x_interp)
        grads.append(tape.gradient(pred, x_interp))

    avg_grads = tf.reduce_mean(tf.stack(grads), axis=0)
    return (x - baseline) * avg_grads

ig_vals = [integrated_gradients(model, X[i:i+1]) for i in range(len(X))]
ig_vals = tf.reduce_mean(tf.abs(tf.concat(ig_vals, axis=0)), axis=0).numpy()

df = pd.DataFrame({
    "band": np.arange(len(ig_vals)),
    "importance": ig_vals.squeeze()
})
df.to_csv("results/band_importance.csv", index=False)
