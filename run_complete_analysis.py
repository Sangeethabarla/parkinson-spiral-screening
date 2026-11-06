
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
)

from src.common import ROOT
from src.data_70_30 import get_split, make_dataset


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(ROOT)

MODEL_PATH = (
    ROOT / "artifacts" / "parkinson_effnetv2s_70_30_best.keras"
)

HISTORY_PATH = (
    ROOT / "artifacts" / "70_30_training_history.json"
)

OUTPUT_DIR = ROOT / "results" / "complete_analysis_70_30"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

# Fixed conventional threshold. Not optimized on validation data.
THRESHOLD = 0.50

CLASS_NAMES = ["Healthy", "Parkinson"]

# Prevent accidental overwriting of previous analysis.
if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        f"Output folder already contains files:\n{OUTPUT_DIR}\n"
        "Rename or back up the folder before running again."
    )

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def heading(message):
    print("\n" + "=" * 70)
    print(message)
    print("=" * 70)


def save_figure(filename):
    path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


def save_table(df, filename):
    path = TABLE_DIR / filename
    df.to_csv(path, index=False)
    print("\n", df.to_string(index=False))
    print("Saved:", path)


def percentage(value):
    return round(float(value) * 100, 2)


# ============================================================
# 3. CHECK REQUIRED FILES
# ============================================================

heading("70:30 PARKINSON'S SPIRAL SCREENING ANALYSIS")

for path in [MODEL_PATH, HISTORY_PATH]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

print("Model:", MODEL_PATH)
print("History:", HISTORY_PATH)
print("Output:", OUTPUT_DIR)
print("Threshold:", THRESHOLD)


# ============================================================
# 4. LOAD TWO-PHASE TRAINING HISTORY
# ============================================================

heading("LOADING TRAINING HISTORY")

with open(HISTORY_PATH, "r", encoding="utf-8") as file:
    history = json.load(file)

if not all(key in history for key in ["phase1", "phase2", "experiment"]):
    raise ValueError("Unexpected 70:30 training-history structure.")

phase1 = history["phase1"]
phase2 = history["phase2"]

required_keys = [
    "accuracy",
    "val_accuracy",
    "loss",
    "val_loss",
]

for phase_name, phase in [
    ("Phase 1", phase1),
    ("Phase 2", phase2),
]:
    for key in required_keys:
        if key not in phase:
            raise KeyError(f"{key} missing from {phase_name} history.")

    lengths = [len(phase[key]) for key in required_keys]

    if len(set(lengths)) != 1:
        raise ValueError(f"Inconsistent history lengths in {phase_name}.")

p1_epochs = len(phase1["accuracy"])
p2_epochs = len(phase2["accuracy"])

epochs1 = np.arange(1, p1_epochs + 1)
epochs2 = np.arange(p1_epochs + 1, p1_epochs + p2_epochs + 1)

print("Phase 1 epochs:", p1_epochs)
print("Phase 2 epochs:", p2_epochs)
print("Total recorded epochs:", p1_epochs + p2_epochs)

history_df = pd.DataFrame({
    "Phase": (
        ["Phase 1"] * p1_epochs
        + ["Phase 2"] * p2_epochs
    ),
    "Epoch": np.concatenate([epochs1, epochs2]),
    "Training Accuracy": (
        phase1["accuracy"] + phase2["accuracy"]
    ),
    "Validation Accuracy": (
        phase1["val_accuracy"] + phase2["val_accuracy"]
    ),
    "Training Loss": (
        phase1["loss"] + phase2["loss"]
    ),
    "Validation Loss": (
        phase1["val_loss"] + phase2["val_loss"]
    ),
})

save_table(history_df, "70_30_training_history.csv")


# ============================================================
# 5. FIGURE 5 - TRAINING AND VALIDATION CURVES
# ============================================================

heading("FIGURE 5 - TRAINING AND VALIDATION CURVES")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for phase_name, phase_epochs, phase in [
    ("Phase 1", epochs1, phase1),
    ("Phase 2", epochs2, phase2),
]:
    axes[0].plot(
        phase_epochs,
        phase["accuracy"],
        marker="o",
        label=f"{phase_name} Training",
    )

    axes[0].plot(
        phase_epochs,
        phase["val_accuracy"],
        marker="o",
        linestyle="--",
        label=f"{phase_name} Validation",
    )

    axes[1].plot(
        phase_epochs,
        phase["loss"],
        marker="o",
        label=f"{phase_name} Training",
    )

    axes[1].plot(
        phase_epochs,
        phase["val_loss"],
        marker="o",
        linestyle="--",
        label=f"{phase_name} Validation",
    )

for ax in axes:
    ax.axvline(
        p1_epochs + 0.5,
        linestyle=":",
        label="Fine-tuning begins",
    )
    ax.set_xlabel("Epoch")
    ax.grid(True)
    ax.legend()

axes[0].set_title("Training and Validation Accuracy")
axes[0].set_ylabel("Accuracy")

axes[1].set_title("Training and Validation Loss")
axes[1].set_ylabel("Loss")

fig.suptitle("EfficientNetV2-S: 70:30 Training-Validation")

save_figure("Figure_5_training_validation.png")


# ============================================================
# 6. LOAD MODEL AND VALIDATION DATA
# ============================================================

heading("LOADING 70:30 MODEL AND VALIDATION DATA")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

print("Model loaded successfully.")

validation_df = get_split("validation")

validation_ds = make_dataset(
    validation_df,
    training=False,
)

y_true_batches = []

for _, labels in validation_ds:
    y_true_batches.append(labels.numpy().reshape(-1))

y_true = np.concatenate(y_true_batches).astype(int)

y_prob = model.predict(
    validation_ds,
    verbose=1,
).reshape(-1)

if len(y_true) != len(y_prob):
    raise ValueError("Validation labels and predictions do not match.")

if not np.all(np.isfinite(y_prob)):
    raise ValueError("Non-finite prediction probabilities detected.")

if not np.all((y_prob >= 0) & (y_prob <= 1)):
    raise ValueError("Predicted probabilities must be between 0 and 1.")

y_pred = (y_prob >= THRESHOLD).astype(int)

print("\nValidation images:", len(y_true))
print("Healthy:", int(np.sum(y_true == 0)))
print("Parkinson:", int(np.sum(y_true == 1)))
print("Threshold:", THRESHOLD)


# ============================================================
# 7. CALCULATE VALIDATION METRICS
# ============================================================

heading("70:30 VALIDATION PERFORMANCE")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
)

tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(y_true, y_pred)

balanced_acc = balanced_accuracy_score(
    y_true,
    y_pred,
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0,
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0,
)

sensitivity = tp / (tp + fn) if (tp + fn) else 0
specificity = tn / (tn + fp) if (tn + fp) else 0

error_rate = 1 - accuracy

if len(np.unique(y_true)) != 2:
    raise ValueError(
        "ROC-AUC and PR-AUC require both validation classes."
    )

roc_auc = roc_auc_score(y_true, y_prob)

pr_precision, pr_recall, _ = precision_recall_curve(
    y_true,
    y_prob,
)

pr_auc = auc(pr_recall, pr_precision)

metrics = {
    "Evaluation Set": "70:30 Validation",
    "Validation Images": int(len(y_true)),
    "Threshold": THRESHOLD,
    "Accuracy (%)": percentage(accuracy),
    "Balanced Accuracy (%)": percentage(balanced_acc),
    "Precision (%)": percentage(precision),
    "Recall (%)": percentage(recall),
    "Sensitivity (%)": percentage(sensitivity),
    "Specificity (%)": percentage(specificity),
    "F1 Score (%)": percentage(f1),
    "ROC-AUC (%)": percentage(roc_auc),
    "PR-AUC (%)": percentage(pr_auc),
    "Error Rate (%)": percentage(error_rate),
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp),
}

for key, value in metrics.items():
    print(f"{key:<28}: {value}")


# ============================================================
# 8. FIGURE 6 - ROC CURVE
# ============================================================

heading("FIGURE 6 - VALIDATION ROC CURVE")

fpr, tpr, _ = roc_curve(y_true, y_prob)

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"EfficientNetV2-S (AUC = {roc_auc:.4f})",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier",
)

plt.title("Validation Receiver Operating Characteristic Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)

save_figure("Figure_6_ROC_curve.png")


# ============================================================
# 9. FIGURE 7 - PRECISION-RECALL CURVE
# ============================================================

heading("FIGURE 7 - VALIDATION PRECISION-RECALL CURVE")

plt.figure(figsize=(7, 6))

plt.plot(
    pr_recall,
    pr_precision,
    linewidth=2,
    label=f"PR-AUC = {pr_auc:.4f}",
)

plt.title("Validation Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)

save_figure("Figure_7_precision_recall.png")


# ============================================================
# 10. FIGURE 8 - CLASSIFICATION METRICS
# ============================================================

heading("FIGURE 8 - CLASSIFICATION METRICS")

metric_names = [
    "Accuracy",
    "Balanced Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
]

metric_values = [
    accuracy,
    balanced_acc,
    precision,
    recall,
    f1,
]

plt.figure(figsize=(10, 5))

bars = plt.bar(metric_names, metric_values)

plt.title("EfficientNetV2-S Validation Classification Metrics")
plt.ylabel("Score")
plt.ylim(0, 1.10)
plt.grid(axis="y")

for bar, value in zip(bars, metric_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{value:.3f}",
        ha="center",
    )

save_figure("Figure_8_classification_metrics.png")


# ============================================================
# 11. FIGURE 9 - SENSITIVITY AND SPECIFICITY
# ============================================================

heading("FIGURE 9 - SENSITIVITY AND SPECIFICITY")

names = ["Sensitivity", "Specificity"]
values = [sensitivity, specificity]

plt.figure(figsize=(8, 5))

bars = plt.bar(names, values)

plt.title("Validation Sensitivity and Specificity")
plt.ylabel("Score")
plt.ylim(0, 1.10)
plt.grid(axis="y")

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{value:.3f}",
        ha="center",
    )

save_figure("Figure_9_sensitivity_specificity.png")


# ============================================================
# 12. FIGURE 10 - PREDICTED PROBABILITY DISTRIBUTION
# ============================================================

heading("FIGURE 10 - VALIDATION PROBABILITY DISTRIBUTION")

healthy_probabilities = y_prob[y_true == 0]
parkinson_probabilities = y_prob[y_true == 1]

plt.figure(figsize=(9, 5))

plt.hist(
    healthy_probabilities,
    bins=10,
    alpha=0.65,
    label="Healthy",
)

plt.hist(
    parkinson_probabilities,
    bins=10,
    alpha=0.65,
    label="Parkinson",
)

plt.axvline(
    THRESHOLD,
    linestyle="--",
    label=f"Decision Threshold = {THRESHOLD}",
)

plt.title("Validation Predicted Probability Distribution")
plt.xlabel("Predicted Parkinson Probability")
plt.ylabel("Number of Validation Images")
plt.legend()
plt.grid(True)

save_figure("Figure_10_probability_distribution.png")


# ============================================================
# 13. FIGURE 11 - CONFUSION MATRIX
# ============================================================

heading("FIGURE 11 - VALIDATION CONFUSION MATRIX")

plt.figure(figsize=(7, 6))

plt.imshow(cm, interpolation="nearest")
plt.title("EfficientNetV2-S Validation Confusion Matrix")
plt.colorbar()

plt.xticks([0, 1], CLASS_NAMES)
plt.yticks([0, 1], CLASS_NAMES)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=15,
        )

save_figure("Figure_11_confusion_matrix.png")


# ============================================================
# 14. TABLE 4 - COMPARISON WITH PREVIOUS WORK
# ============================================================

heading("TABLE 4 - COMPARISON WITH PREVIOUS WORK")

best_val_auc = history["experiment"].get(
    "best_validation_auc"
)

table4 = pd.DataFrame([
    {
        "Study": "Saravanan et al. (2023)",
        "Dataset": "Spiral and wave drawings",
        "Architecture": "VGG19-INC",
        "Split": "As reported in paper",
        "Evaluation Set": "As reported in paper",
        "Accuracy (%)": 98.45,
    },
    {
        "Study": "Current Project",
        "Dataset": "Spiral drawings",
        "Architecture": "EfficientNetV2-S",
        "Split": "70:30",
        "Evaluation Set": "Validation",
        "Accuracy (%)": percentage(accuracy),
    },
])

save_table(
    table4,
    "Table_4_previous_work_comparison.csv",
)

# Save additional experiment details separately.
experiment_details = {
    "architecture": history["experiment"]["architecture"],
    "split_file": history["experiment"]["split_file"],
    "seed": history["experiment"]["seed"],
    "best_checkpoint": history["experiment"]["best_checkpoint"],
    "best_validation_auc_during_selection": best_val_auc,
    "evaluation_threshold": THRESHOLD,
    "evaluation_set": "Validation",
    "separate_test_set": False,
}

with open(
    OUTPUT_DIR / "experiment_details.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(experiment_details, file, indent=4)


# ============================================================
# 15. SAVE METRICS AND VALIDATION PREDICTIONS
# ============================================================

heading("SAVING VALIDATION RESULTS")

with open(
    OUTPUT_DIR / "validation_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metrics, file, indent=4)

predictions_df = pd.DataFrame({
    "Image": validation_df["image_path"].astype(str).to_numpy(),
    "True Label": y_true,
    "Predicted Label": y_pred,
    "Parkinson Probability": y_prob,
    "Predicted Class": [
        "PARKINSON" if value == 1 else "HEALTHY"
        for value in y_pred
    ],
})

predictions_df.to_csv(
    OUTPUT_DIR / "validation_predictions.csv",
    index=False,
)

print("Validation metrics saved.")
print("Validation predictions saved.")


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

heading("FINAL 70:30 EXPERIMENT RESULTS")

print("MODEL: EfficientNetV2-S")
print("EVALUATION SET: VALIDATION")
print("VALIDATION IMAGES:", len(y_true))

print("\nCLASSIFICATION RESULTS")
print(f"Accuracy          : {percentage(accuracy):.2f}%")
print(f"Balanced Accuracy : {percentage(balanced_acc):.2f}%")
print(f"Precision         : {percentage(precision):.2f}%")
print(f"Recall            : {percentage(recall):.2f}%")
print(f"Sensitivity       : {percentage(sensitivity):.2f}%")
print(f"Specificity       : {percentage(specificity):.2f}%")
print(f"F1 Score          : {percentage(f1):.2f}%")
print(f"ROC-AUC           : {percentage(roc_auc):.2f}%")
print(f"PR-AUC            : {percentage(pr_auc):.2f}%")
print(f"Error Rate        : {percentage(error_rate):.2f}%")

print("\nCONFUSION MATRIX")
print("                 Predicted")
print("               Healthy  Parkinson")
print(f"Actual Healthy    {tn:>4}      {fp:>4}")
print(f"Actual Parkinson  {fn:>4}      {tp:>4}")

print("\nGENERATED FIGURES")
for path in sorted(FIGURE_DIR.glob("*.png")):
    print(" -", path.name)

print("\nGENERATED TABLES")
for path in sorted(TABLE_DIR.glob("*.csv")):
    print(" -", path.name)

print("\nALL RESULTS SAVED IN:")
print(OUTPUT_DIR)

print("\nIMPORTANT RESEARCH NOTE:")
print(
    "These are validation-set results from the 70:30 experiment. "
    "The validation set was used for early stopping and checkpoint "
    "selection. No separate test set was used."
)

print("\nCOMPLETE 70:30 ANALYSIS FINISHED.")
