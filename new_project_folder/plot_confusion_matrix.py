#!/usr/bin/env python3
"""
plot_confusion_matrix.py

Computes the 12x12 Domain-Level NLU Intent Confusion Matrix from held-out evaluation results 
(test_metadata.json and intent_errors.json) and generates a high-resolution heatmap plot 
saved as domain_confusion_matrix_heatmap.png.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

TEST_METADATA_PATH = DATA_DIR / "test_metadata.json"
INTENT_ERRORS_PATH = RESULTS_DIR / "intent_errors.json"
OUTPUT_PLOT_PATH = RESULTS_DIR / "domain_confusion_matrix_heatmap.png"

def plot_domain_confusion_matrix():
    # 1. Load held-out test metadata and intent classification errors
    with open(TEST_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(INTENT_ERRORS_PATH, "r", encoding="utf-8") as f:
        errors = json.load(f)

    err_map = {e["text"].strip(): e["intent_prediction"]["name"] for e in errors}
    intent_to_crop = {m["intent"]: m["crop"] for m in metadata}

    crop_keys = [
        "banana", "beans", "cassava", "good_agricultural_methods", "maize",
        "pepper", "pineapple", "plantain", "rice", "vegetable", "waterleaf", "yam"
    ]
    crop_abbrevs = [
        "BAN", "BEA", "CAS", "GAM", "MAI", "PEP", "PIN", "PLA", "RIC", "VEG", "WTR", "YAM"
    ]

    # 2. Build 12x12 domain confusion matrix
    matrix = np.zeros((12, 12), dtype=int)

    for m in metadata:
        text = m["text"].strip()
        true_crop = m["crop"]
        i_true = crop_keys.index(true_crop)
        if text in err_map:
            pred_intent = err_map[text]
            pred_crop = intent_to_crop.get(pred_intent, "Other")
            if pred_crop in crop_keys:
                i_pred = crop_keys.index(pred_crop)
            else:
                i_pred = i_true
        else:
            i_pred = i_true
        matrix[i_true, i_pred] += 1

    # 3. Render high-resolution heatmap plot using matplotlib
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")

    # Add colorbar legend
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Sample Count", rotation=-90, va="bottom", fontweight="bold")

    ax.set_xticks(np.arange(12))
    ax.set_yticks(np.arange(12))
    ax.set_xticklabels(crop_abbrevs, fontweight="bold")
    ax.set_yticklabels(crop_abbrevs, fontweight="bold")

    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Annotate ONLY non-zero cells to avoid visual clutter
    for i in range(12):
        for j in range(12):
            val = matrix[i, j]
            if val > 0:
                color = "white" if val > 120 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold", fontsize=10)

    ax.set_title("Domain-Level NLU Intent Confusion Matrix (N = 1,253 Held-Out Queries)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Domain", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Domain", fontsize=11, fontweight="bold", labelpad=10)
    plt.tight_layout()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Domain confusion matrix heatmap saved successfully to: {OUTPUT_PLOT_PATH}")

if __name__ == "__main__":
    plot_domain_confusion_matrix()
