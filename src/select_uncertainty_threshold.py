
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    ROOT / "reports" / "validation_uncertainty"
    / "validation_uncertainty_results.csv"
)

OUTPUT_DIR = ROOT / "reports" / "validation_uncertainty"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Minimum coverage to retain at least 80% of validation images.
MIN_COVERAGE = 0.80

df = pd.read_csv(INPUT_CSV)

required = {
    "actual_label",
    "raw_pd_probability",
    "mc_dropout_variance",
}

missing = required - set(df.columns)
if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}. "
        f"Available columns: {list(df.columns)}"
    )

y = df["actual_label"].astype(int).to_numpy()
prob = df["raw_pd_probability"].astype(float).to_numpy()
uncertainty = df["mc_dropout_variance"].astype(float).to_numpy()

if not np.isfinite(prob).all() or not np.isfinite(uncertainty).all():
    raise ValueError("Probabilities or uncertainty contain invalid values.")

if not np.isin(y, [0, 1]).all():
    raise ValueError("actual_label must contain only 0 and 1.")

# Keep the project's previously selected classification threshold.
DECISION_THRESHOLD = 0.58
pred = (prob >= DECISION_THRESHOLD).astype(int)

rows = []

# Retain predictions whose uncertainty is at or below each threshold.
for threshold in np.unique(uncertainty):
    retained = uncertainty <= threshold
    coverage = float(retained.mean())

    if retained.sum() == 0 or coverage < MIN_COVERAGE:
        continue

    yt = y[retained]
    pt = pred[retained]

    tn, fp, fn, tp = confusion_matrix(
        yt, pt, labels=[0, 1]
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    rows.append({
        "uncertainty_threshold": float(threshold),
        "retained_images": int(retained.sum()),
        "coverage": coverage,
        "rejected_images": int((~retained).sum()),
        "rejected_errors": int(((pred != y) & (~retained)).sum()),
        "retained_errors": int(((pt != yt)).sum()),
        "accuracy": accuracy_score(yt, pt),
        "balanced_accuracy": balanced_accuracy_score(yt, pt),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    })

results = pd.DataFrame(rows)

if results.empty:
    raise ValueError(
        "No threshold meets the minimum coverage requirement."
    )

# Choose the threshold with the highest retained balanced accuracy.
# Tie-break by higher coverage.
results = results.sort_values(
    ["balanced_accuracy", "coverage"],
    ascending=[False, False],
).reset_index(drop=True)

results.to_csv(
    OUTPUT_DIR / "uncertainty_threshold_candidates.csv",
    index=False,
)

best = results.iloc[0].to_dict()

summary = {
    "method": "MC Dropout variance",
    "decision_threshold": DECISION_THRESHOLD,
    "minimum_coverage": MIN_COVERAGE,
    "selected_uncertainty_threshold": float(
        best["uncertainty_threshold"]
    ),
    "retained_images": int(best["retained_images"]),
    "coverage": float(best["coverage"]),
    "accuracy": float(best["accuracy"]),
    "balanced_accuracy": float(best["balanced_accuracy"]),
    "sensitivity": float(best["sensitivity"]),
    "specificity": float(best["specificity"]),
    "note": (
        "Threshold selected using validation data only. "
        "This is a small validation set; performance is uncertain. "
        "Do not interpret this as clinical validation."
    ),
}

with open(
    OUTPUT_DIR / "selected_uncertainty_threshold.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(summary, f, indent=2)

print("\nTop uncertainty threshold candidates:")
print(results.head(10).to_string(index=False))

print("\nSelected threshold summary:")
print(json.dumps(summary, indent=2))
