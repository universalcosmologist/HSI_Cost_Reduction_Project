import numpy as np
import tensorflow as tf

X = np.load("data/processed/X_full.npy")
y = np.load("data/processed/y_full.npy")
importance = np.loadtxt("results/band_importance.csv", delimiter=",", skiprows=1)[:,1]

top_k = np.argsort(importance)[::-1][:60]
X_k = X[:, top_k]

np.save("data/processed/X_k60.npy", X_k)
np.save("data/processed/y_k60.npy", y)
