"""
06_evaluation_plots.py

Generates all evaluation and analysis plots used in the project:
1. Average spectral signatures (Healthy vs Unhealthy)
2. Accuracy & F1 vs number of selected bands
3. Training & validation accuracy curves (300 vs 60 bands)
4. Correlation heatmap of top-60 bands
5. Sensor cost vs number of bands

Author: Shubham
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import seaborn as sns

# Paths
# these are the path of folder as per author's 

DATA_DIR = "data/processed"
RESULTS_DIR = "results"
FIG_DIR = "figures"

import os
os.makedirs(FIG_DIR, exist_ok=True)

# 1. Spectral Signature Plot

def plot_spectral_signatures():
    X = np.load(f"{DATA_DIR}/X_full.npy")
    y = np.load(f"{DATA_DIR}/y_full.npy")
    wavelengths = np.load(f"{DATA_DIR}/wavelengths.npy")

    healthy = X[y == 0]
    unhealthy = X[y == 1]

    h_mean, h_std = healthy.mean(axis=0), healthy.std(axis=0)
    u_mean, u_std = unhealthy.mean(axis=0), unhealthy.std(axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, h_mean, label="Healthy (mean)", color="blue")
    plt.plot(wavelengths, u_mean, label="Unhealthy (mean)", color="orange", linestyle="--")

    plt.fill_between(wavelengths, h_mean-h_std, h_mean+h_std, alpha=0.2, color="blue")
    plt.fill_between(wavelengths, u_mean-u_std, u_mean+u_std, alpha=0.2, color="orange")

    plt.axvspan(400, 680, alpha=0.1, label="Visible")
    plt.axvspan(680, 750, alpha=0.15, label="Red-edge")
    plt.axvspan(750, 1000, alpha=0.1, label="NIR")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title("Average Spectral Signatures: Healthy vs Unhealthy Tomato Leaves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/spectral_signatures.png", dpi=300)
    plt.close()


# 2. Accuracy & F1 vs Bands

def plot_accuracy_vs_k():
    ks = [10, 20, 30, 40, 60]
    accs, f1s = [], []

    for k in ks:
        df = pd.read_csv(f"{RESULTS_DIR}/metrics_k{k}.csv")
        accs.append(df["accuracy"].values[0])
        f1s.append(df["f1"].values[0])

    plt.figure(figsize=(7, 5))
    plt.plot(ks, accs, marker="o", label="Accuracy")
    plt.plot(ks, f1s, marker="s", linestyle="--", label="F1-score")

    plt.xlabel("Number of bands (k)")
    plt.ylabel("Metric")
    plt.title("Accuracy & F1 vs Number of Bands (top-k by IG)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/accuracy_vs_k.png", dpi=300)
    plt.close()


# 3. Training Curves


def plot_training_curves():
    hist_300 = np.load(f"{RESULTS_DIR}/history_300.npy", allow_pickle=True).item()
    hist_60 = np.load(f"{RESULTS_DIR}/history_60.npy", allow_pickle=True).item()

    plt.figure(figsize=(8, 5))
    plt.plot(hist_300["accuracy"], label="Acc-300 bands")
    plt.plot(hist_300["val_accuracy"], linestyle="--", label="Val Acc-300")

    plt.plot(hist_60["accuracy"], label="Acc-60 bands")
    plt.plot(hist_60["val_accuracy"], linestyle="--", label="Val Acc-60")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy (300 vs 60 Bands)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/training_curves.png", dpi=300)
    plt.close()


# 4. Correlation Heatmap

def plot_correlation_heatmap():
    X = np.load(f"{DATA_DIR}/X_k60.npy")
    corr = np.corrcoef(X.T)

    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, cmap="viridis", cbar=True)
    plt.title("Correlation Heatmap of Top-60 Bands")
    plt.xlabel("Band index (within top-k set)")
    plt.ylabel("Band index (within top-k set)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/correlation_heatmap.png", dpi=300)
    plt.close()


# 5. Sensor Cost Projection

def plot_sensor_cost():
    bands = [300, 60, 40, 30, 20, 10]
    cost = [100, 20, 12, 10, 7, 4]  # relative cost assumption

    plt.figure(figsize=(7, 5))
    plt.bar([str(b) for b in bands], cost)
    plt.xlabel("Number of spectral bands")
    plt.ylabel("Sensor cost (% of 300-band camera)")
    plt.title("Projected Sensor Cost vs Number of Bands")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/cost_vs_bands.png", dpi=300)
    plt.close()

# Main


if __name__ == "__main__":
    print("Generating evaluation plots...")

    plot_spectral_signatures()
    plot_accuracy_vs_k()
    plot_training_curves()
    plot_correlation_heatmap()
    plot_sensor_cost()

    print("All plots saved in /figures")
