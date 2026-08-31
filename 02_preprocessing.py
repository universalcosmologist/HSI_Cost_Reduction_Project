import numpy as np
from spectral import envi
import os

DATA_DIR = "data/raw"
SAVE_DIR = "data/processed"
os.makedirs(SAVE_DIR, exist_ok=True)

healthy_pixels = []
unhealthy_pixels = []

healthy_groups = []
unhealthy_groups = []

THRESHOLD = 0.25

for hdr in os.listdir(DATA_DIR):
    if not hdr.endswith(".hdr"):
        continue

    bil = hdr.replace(".hdr", ".bil")

    cube = envi.open(
        os.path.join(DATA_DIR, hdr),
        os.path.join(DATA_DIR, bil)
    ).load()

    # Convert (height, width, bands) -> (pixels, bands)
    pixels = cube.reshape(-1, cube.shape[-1])

    # Remove background pixels
    mask = pixels.mean(axis=1) > THRESHOLD
    pixels = pixels[mask]

    # Use filename as the leaf/image ID
    leaf_id = os.path.splitext(hdr)[0]

    # Store pixels and corresponding leaf IDs
    if "healthy" in hdr.lower():
        healthy_pixels.append(pixels)
        healthy_groups.append(
            np.full(len(pixels), leaf_id)
        )
    else:
        unhealthy_pixels.append(pixels)
        unhealthy_groups.append(
            np.full(len(pixels), leaf_id)
        )


# Combine pixels from all leaves
healthy_pixels = np.concatenate(healthy_pixels)
unhealthy_pixels = np.concatenate(unhealthy_pixels)

healthy_groups = np.concatenate(healthy_groups)
unhealthy_groups = np.concatenate(unhealthy_groups)


# Balance the number of pixels between classes
N = min(len(healthy_pixels), len(unhealthy_pixels))

healthy_pixels = healthy_pixels[:N]
unhealthy_pixels = unhealthy_pixels[:N]

healthy_groups = healthy_groups[:N]
unhealthy_groups = unhealthy_groups[:N]


# Construct final dataset
X = np.vstack([
    healthy_pixels,
    unhealthy_pixels
])

y = np.array(
    [0] * N +
    [1] * N
)

groups = np.concatenate([
    healthy_groups,
    unhealthy_groups
])


# Save exactly the same X and y as before
np.save(f"{SAVE_DIR}/X_full.npy", X)
np.save(f"{SAVE_DIR}/y_full.npy", y)

# Additional file containing the source leaf/image ID
np.save(f"{SAVE_DIR}/groups_full.npy", groups)
