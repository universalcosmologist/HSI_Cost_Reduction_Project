import numpy as np
from spectral import envi
import os

DATA_DIR = "data/raw"
SAVE_DIR = "data/processed"
os.makedirs(SAVE_DIR, exist_ok=True)

healthy_pixels = []
unhealthy_pixels = []

THRESHOLD = 0.25

for hdr in os.listdir(DATA_DIR):
    if not hdr.endswith(".hdr"):
        continue

    bil = hdr.replace(".hdr", ".bil")
    cube = envi.open(
        os.path.join(DATA_DIR, hdr),
        os.path.join(DATA_DIR, bil)
    ).load()

    pixels = cube.reshape(-1, cube.shape[-1])
    mask = pixels.mean(axis=1) > THRESHOLD
    pixels = pixels[mask]

    if "healthy" in hdr.lower():
        healthy_pixels.append(pixels)
    else:
        unhealthy_pixels.append(pixels)

healthy_pixels = np.concatenate(healthy_pixels)
unhealthy_pixels = np.concatenate(unhealthy_pixels)

N = min(len(healthy_pixels), len(unhealthy_pixels))
X = np.vstack([healthy_pixels[:N], unhealthy_pixels[:N]])
y = np.array([0]*N + [1]*N)

np.save(f"{SAVE_DIR}/X_full.npy", X)
np.save(f"{SAVE_DIR}/y_full.npy", y)
