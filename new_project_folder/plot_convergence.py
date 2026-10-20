#!/usr/bin/env python3
"""
plot_convergence.py

Extracts epoch-level training metrics (epoch_t_loss and epoch_i_acc) from TensorBoard 
event files (.tfevents) generated during DIET training, and plots a dual-panel 
training convergence curve saved as training_convergence_curve.png.
"""

import sys
import os
import glob
from pathlib import Path

# Add PyTorch & TensorFlow site-packages path
USER_SITE = r"C:\Users\macau\AppData\Roaming\Python\Python310\site-packages"
if USER_SITE not in sys.path:
    sys.path.insert(0, USER_SITE)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "tensorboard_logs" / "train"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_PLOT_PATH = RESULTS_DIR / "training_convergence_curve.png"

def extract_and_plot_convergence():
    # Find event file
    event_files = list(LOG_DIR.glob("*events.out.tfevents*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {LOG_DIR}")
    
    log_file = str(event_files[0])
    print(f"Reading TensorBoard log file: {log_file}")

    epoch_loss = []
    epoch_acc = []

    # Iterate over binary tfevents protobuf records using TensorFlow
    for e in tf.compat.v1.train.summary_iterator(log_file):
        for v in e.summary.value:
            if v.HasField("tensor"):
                arr = tf.make_ndarray(v.tensor)
                if arr.size == 1:
                    val = float(arr.flat[0])
                    if v.tag == "epoch_t_loss":
                        epoch_loss.append((e.step + 1, val))
                    elif v.tag == "epoch_i_acc":
                        epoch_acc.append((e.step + 1, val))

    epoch_loss.sort(key=lambda x: x[0])
    epoch_acc.sort(key=lambda x: x[0])

    steps_loss = np.array([x[0] for x in epoch_loss])
    vals_loss = np.array([x[1] for x in epoch_loss])

    steps_acc = np.array([x[0] for x in epoch_acc])
    vals_acc = np.array([x[1] for x in epoch_acc])

    print(f"Successfully extracted {len(steps_loss)} epoch loss records and {len(steps_acc)} epoch accuracy records.")

    # Create dual-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Panel A: Training Loss Convergence
    ax1.plot(steps_loss, vals_loss, color="#d32f2f", linewidth=2.5, label="DIET Total Loss")
    ax1.set_title("A: DIET Training Loss Convergence (Epoch 1 to 100)", fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Training Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_xlim(1, 100)
    ax1.axhline(vals_loss[-1], color="#b71c1c", linestyle="--", alpha=0.7, label=f"Final Loss = {vals_loss[-1]:.4f}")
    ax1.legend(fontsize=10, loc="upper right")

    # Panel B: Intent Classification Accuracy Convergence
    ax2.plot(steps_acc, vals_acc * 100.0, color="#1976d2", linewidth=2.5, label="Intent Accuracy (%)")
    ax2.set_title("B: Intent Classification Accuracy Convergence", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Training Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_xlim(1, 100)
    ax2.set_ylim(0, 105)
    ax2.axhline(vals_acc[-1] * 100.0, color="#0d47a1", linestyle="--", alpha=0.7, label=f"Final Accuracy = {vals_acc[-1]*100:.2f}%")
    ax2.legend(fontsize=10, loc="lower right")

    plt.suptitle("Rasa DIET Classifier Training Convergence (AfroXLM-R Backbone, 100 Epochs)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Convergence plot saved successfully to: {OUTPUT_PLOT_PATH}")

if __name__ == "__main__":
    extract_and_plot_convergence()
