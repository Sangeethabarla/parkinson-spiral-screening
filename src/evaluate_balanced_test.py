
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef,
)

from src.common import ROOT
from src.data import get_split, make_dataset


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = (
    ROOT / "artifacts" /
    "parkinson_effnetv2s_balanced_best.keras"
)

OUTPUT_DIR = ROOT / "results" / "balanced_final_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.50

PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "test_metrics.json"
CM_PATH = OUTPUT_DIR / "test_confusion_matrix.png"
ROC_PATH = OUTPUT_DIR / "test_roc_curve.png"

OUTPUT_FILES = [
    PREDICTIONS_PATH,
    METRICS_PATH,
    CM_PATH,
    ROC_PATH,
]

for path in OUTPUT_FILES:
    if path.exists():
        raise FileExistsError(
            f"Output already exists. Not overwriting: {path}"
        )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Selected model not found: {MODEL_PATH}"
    )


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("\n========== FINAL TEST EVALUATION ==========")

test_df = get_split("test")

print("\nTest dataset summary:")
print(f"Total images: {len(test_df)}")
print(f"Healthy: {(test_df['label'] == 0).sum()}")
print(f"Parkinson: {(test_df['label'] == 1).sum()}")

if len(test_df) != 55:
    raise ValueError(
        f"Expected 55 test images, found {len(test_df)}"
    )

test_ds = make_dataset(test_df, training=False)

y_true = test_df["label"].to_numpy(dtype=np.int32)


# ============================================================
# 3. LOAD SELECTED MODEL
# ============================================================

print("\nLoading selected Phase 2 checkpoint:")
print(MODEL_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)


# ============================================================
# 4. GENERATE TEST PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

probabilities = model.predict(
    test_ds,
    verbose=1
).reshape(-1)

if len(probabilities) != len(y_true):
    raise ValueError("Prediction count does not match test labels.")

if not np.isfinite(probabilities).all():
    raise ValueError("Non-finite prediction probabilities detected.")

predictions = (
    probabilities >= THRESHOLD
).astype(np.int32)


# ============================================================
# 5. CALCULATE METRICS
# ============================================================

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

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0 else 0.0
)

f1 = f1_score(
    y_true, predictions, zero_division=0
)

balanced_acc = balanced_accuracy_score(
    y_true, predictions
)

auc = roc_auc_score(
    y_true, probabilities
)

mcc = matthews_corrcoef(
    y_true, predictions
)

metrics = {
    "model": MODEL_PATH.name,
    "threshold": THRESHOLD,
    "test_images": int(len(y_true)),
    "healthy_images": int(np.sum(y_true == 0)),
    "parkinson_images": int(np.sum(y_true == 1)),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "sensitivity": float(sensitivity),
    "specificity": float(specificity),
    "f1_score": float(f1),
    "balanced_accuracy": float(balanced_acc),
    "roc_auc": float(auc),
    "matthews_correlation_coefficient": float(mcc),
    "true_negative": int(tn),
    "false_positive": int(fp),
    "false_negative": int(fn),
    "true_positive": int(tp),
}


# ============================================================
# 6. PRINT RESULTS
# ============================================================

print("\n========== FINAL TEST METRICS ==========")

for key, value in metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_true,
        predictions,
        labels=[0, 1],
        target_names=["Healthy", "Parkinson"],
        zero_division=0
    )
)

print("\n========== CONFUSION MATRIX ==========")

print(
    f"                 Predicted Healthy  Predicted Parkinson"
)
print(f"Actual Healthy          {tn:3d}                 {fp:3d}")
print(f"Actual Parkinson       {fn:3d}                 {tp:3d}")


# ============================================================
# 7. SAVE CONFUSION MATRIX
# ============================================================

cm = np.array([
    [tn, fp],
    [fn, tp]
])

fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
fig.colorbar(im, ax=ax)

ax.set(
    xticks=[0, 1],
    yticks=[0, 1],
    xticklabels=["Healthy", "Parkinson"],
    yticklabels=["Healthy", "Parkinson"],
    xlabel="Predicted label",
    ylabel="True label",
    title="Final Test Confusion Matrix"
)

for i in range(2):
    for j in range(2):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=14
        )

fig.tight_layout()
fig.savefig(CM_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)


# ============================================================
# 8. SAVE ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_true, probabilities
)

fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(
    fpr,
    tpr,
    label=f"EfficientNetV2S (AUC = {auc:.4f})"
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="Final Test ROC Curve",
    xlim=[0, 1],
    ylim=[0, 1.05]
)

ax.legend(loc="lower right")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(ROC_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)


# ============================================================
# 9. SAVE PREDICTIONS AND METRICS
# ============================================================

prediction_df = test_df.copy()

prediction_df["actual_label"] = y_true
prediction_df["pd_probability"] = probabilities
prediction_df["predicted_label"] = predictions
prediction_df["threshold"] = THRESHOLD

prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)


# ============================================================
# 10. COMPLETION
# ============================================================

print("\n========== FINAL EVALUATION COMPLETED ==========")

print(f"Metrics: {METRICS_PATH}")
print(f"Predictions: {PREDICTIONS_PATH}")
print(f"Confusion matrix: {CM_PATH}")
print(f"ROC curve: {ROC_PATH}")

print("\nTest set was used only for final evaluation.")
print("No training or threshold tuning was performed.")
