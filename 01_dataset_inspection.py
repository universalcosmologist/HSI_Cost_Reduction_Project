import os
from spectral import envi

DATA_DIR = "data/raw"

files = [f for f in os.listdir(DATA_DIR) if f.endswith(".hdr")]
print("Total HSI files:", len(files))

hdr = os.path.join(DATA_DIR, files[0])
bil = hdr.replace(".hdr", ".bil")

cube = envi.open(hdr, bil).load()
print("Cube shape:", cube.shape)
