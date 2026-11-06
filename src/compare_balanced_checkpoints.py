
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.common import ROOT
from src.data import get_split, make_dataset


# ------------------------------------------------
# CONFIGURATION
# ------------------------------------------------

THRESHOLD = 0.50

CHECKPOINTS = {
    "Phase 1": ROOT / "artifacts" / "balanced_phase1_best.keras",
    "Phase 2": ROOT / "artifacts" / "balanced_phase2_best.keras",
}

OUTPUT_DIR = ROOT / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "balanced_validation_comparison.csv"

if CSV_PATH.exists():
    raise FileExistsError(f"Output already exists: {CSV_PATH}")

for path in CHECKPOINTS.values():
    if not path.exists():
        raise FileNotFoundError(path)


# ------------------------------------------------
# LOAD VALIDATION DATA
# ------------------------------------------------

print("\n========== VALIDATION CHECKPOINT COMPARISON ==========")

val_df = get_split("validation")
val_ds = make_dataset(val_df, training=False)

y_true = val_df["label"].to_numpy(dtype=np.int32)

results = []

# ------------------------------------------------
# EVALUATE BOTH CHECKPOINTS
# ------------------------------------------------

for name, checkpoint_path in CHECKPOINTS.items():

    print(f"\n========== {name} ==========")
    print(f"Checkpoint: {checkpoint_path.name}")

    model = tf.keras.models.load_model(
        checkpoint_path,
        compile=False
    )

    probabilities = model.predict(
        val_ds,
        verbose=0
    ).reshape(-1)

    if len(probabilities) != len(y_true):
        raise ValueError("Prediction count does not match validation labels.")

    predictions = (probabilities >= THRESHOLD).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(
        y_true, predictions, zero_division=0
    )
    sensitivity = recall_score(
        y_true, predictions, zero_division=0
    )
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = f1_score(y_true, predictions, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, predictions)
    auc = roc_auc_score(y_true, probabilities)

    metrics = {
        "Checkpoint": name,
        "Threshold": THRESHOLD,
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "F1-score": f1,
        "Balanced Accuracy": balanced_acc,
        "ROC-AUC": auc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    results.append(metrics)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nConfusion Matrix:")
    print(f"                 Predicted Healthy  Predicted Parkinson")
    print(f"Actual Healthy          {tn:3d}                 {fp:3d}")
    print(f"Actual Parkinson       {fn:3d}                 {tp:3d}")

    del model


# ------------------------------------------------
# SAVE COMPARISON
# ------------------------------------------------

comparison = pd.DataFrame(results)
comparison.to_csv(CSV_PATH, index=False)

print("\n========== FINAL COMPARISON ==========")
print(comparison.to_string(index=False))

print(f"\nSaved comparison to: {CSV_PATH}")
print("\nValidation comparison completed.")
print("The test set was NOT accessed.")
