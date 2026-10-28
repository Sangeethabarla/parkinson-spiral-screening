
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]

pred_path = ROOT / "reports" / "test_predictions_expanded.csv"
df = pd.read_csv(pred_path)

THRESHOLD = 0.5

print("\n========== SOURCE-WISE TEST ANALYSIS ==========")

for source, part in df.groupby("source_dataset"):
    actual = (part["actual_label"] == "parkinson").astype(int)
    probability = part["parkinson_probability"].astype(float)
    predicted = (probability >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0

    auc = (
        roc_auc_score(actual, probability)
        if actual.nunique() == 2
        else float("nan")
    )

    print(f"\nSource: {source}")
    print(f"Test images: {len(part)}")
    print(f"Accuracy: {accuracy_score(actual, predicted):.4f}")
    print(f"Precision: {precision_score(actual, predicted, zero_division=0):.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

print("\nNote: These are subgroup results from the same held-out test set.")
print("They are exploratory because each subgroup is small.")
