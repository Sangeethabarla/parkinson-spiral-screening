
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    ROOT / "reports" / "uncertainty"
    / "mc_dropout_test_results.csv"
)

OUTPUT_DIR = ROOT / "reports" / "uncertainty_threshold"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed using validation data; do not tune using test results.
DECISION_THRESHOLD = 0.58
UNCERTAINTY_THRESHOLD = 0.020342412671172

df = pd.read_csv(INPUT_CSV)

required = {
    "actual_label",
    "predicted_label",
    "mc_dropout_variance",
}

missing = required - set(df.columns)
if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}. "
        f"Available columns: {list(df.columns)}"
    )

y = df["actual_label"].astype(int).to_numpy()
pred = df["predicted_label"].astype(int).to_numpy()
variance = df["mc_dropout_variance"].astype(float).to_numpy()

if not np.isfinite(variance).all():
    raise ValueError("Uncertainty values contain invalid numbers.")

if not np.isin(y, [0, 1]).all():
    raise ValueError("actual_label must contain only 0 and 1.")

if not np.isin(pred, [0, 1]).all():
    raise ValueError("predicted_label must contain only 0 and 1.")

# Verify predictions are consistent with the saved decision threshold.
if "raw_pd_probability" in df.columns:
    probabilities = df["raw_pd_probability"].astype(float).to_numpy()
    expected_predictions = (
        probabilities >= DECISION_THRESHOLD
    ).astype(int)

    if not np.array_equal(pred, expected_predictions):
        raise ValueError(
            "Saved predictions do not match the fixed decision "
            "threshold. Check the input CSV before continuing."
        )

# Baseline: use all test images.
tn, fp, fn, tp = confusion_matrix(
    y, pred, labels=[0, 1]
).ravel()

baseline = {
    "coverage": 1.0,
    "retained_images": int(len(y)),
    "rejected_images": 0,
    "accuracy": float(accuracy_score(y, pred)),
    "balanced_accuracy": float(
        balanced_accuracy_score(y, pred)
    ),
    "sensitivity": float(tp / (tp + fn)) if (tp + fn) else None,
    "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
}

# Retain predictions whose variance is below the
# threshold selected on validation data.
retained = variance <= UNCERTAINTY_THRESHOLD
rejected = ~retained

if retained.sum() == 0:
    raise ValueError("No test predictions remain after filtering.")

yr = y[retained]
pr = pred[retained]

tn, fp, fn, tp = confusion_matrix(
    yr, pr, labels=[0, 1]
).ravel()

selected = {
    "uncertainty_method": "MC Dropout variance",
    "decision_threshold": DECISION_THRESHOLD,
    "uncertainty_threshold_from_validation": UNCERTAINTY_THRESHOLD,
    "coverage": float(retained.mean()),
    "retained_images": int(retained.sum()),
    "rejected_images": int(rejected.sum()),
    "rejected_errors": int(
        ((pred != y) & rejected).sum()
    ),
    "retained_errors": int((pr != yr).sum()),
    "accuracy": float(accuracy_score(yr, pr)),
    "balanced_accuracy": float(
        balanced_accuracy_score(yr, pr)
    ),
    "sensitivity": float(tp / (tp + fn)) if (tp + fn) else None,
    "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp),
    "note": (
        "Exploratory test evaluation. The test set has been used "
        "in previous experiments and is not an untouched independent "
        "test set. Rejected predictions require review; this result "
        "does not establish clinical safety."
    ),
}

# Save per-image results and flag uncertain predictions.
output_df = df.copy()
output_df["uncertainty_threshold"] = UNCERTAINTY_THRESHOLD
output_df["review_required"] = rejected
output_df["retained_for_prediction"] = retained

output_df.to_csv(
    OUTPUT_DIR / "test_uncertainty_threshold_results.csv",
    index=False,
)

summary = {
    "baseline_all_predictions": baseline,
    "uncertainty_filtered_predictions": selected,
}

with open(
    OUTPUT_DIR / "test_uncertainty_threshold_metrics.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(summary, f, indent=2)

print("\nBASELINE — all test images")
print(json.dumps(baseline, indent=2))

print("\nUNCERTAINTY-AWARE — validation-selected threshold")
print(json.dumps(selected, indent=2))

print(f"\nSaved results to: {OUTPUT_DIR}")
