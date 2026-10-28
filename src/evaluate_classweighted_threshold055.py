
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    brier_score_loss,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4
THRESHOLD = 0.55

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "parkinson_effnetv2s_classweighted.keras"
)

SPLIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_split.csv"
)

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Save to new files so previous results remain unchanged.
METRICS_PATH = (
    REPORTS_DIR
    / "test_metrics_classweighted_t055_preprocesscheck.json"
)

PREDICTIONS_PATH = (
    REPORTS_DIR
    / "test_predictions_classweighted_t055_preprocesscheck.csv"
)

# Label convention: 0 = Healthy, 1 = Parkinson

tf.keras.utils.set_random_seed(SEED)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates, description):
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not find {description} column.\n"
        f"Available columns: {list(df.columns)}"
    )


def normalize_split(value):
    value = str(value).strip().lower()

    aliases = {
        "testing": "test",
        "testing_set": "test",
        "test_set": "test",
        "validation": "validation",
        "valid": "validation",
        "validation_set": "validation",
        "val": "validation",
        "val_set": "validation",
        "training": "train",
        "training_set": "train",
        "train_set": "train",
    }

    return aliases.get(value, value)


def normalize_label(value):
    label = str(value).strip().lower()

    healthy_labels = {
        "0", "healthy", "control", "normal",
        "non-parkinson", "non_parkinson", "nonparkinson",
    }

    parkinson_labels = {
        "1", "parkinson", "parkinsons",
        "parkinson's", "pd", "positive", "patient",
    }

    if label in healthy_labels:
        return 0

    if label in parkinson_labels:
        return 1

    raise ValueError(
        f"Unrecognized label: {value!r}. "
        "Verify label mapping against the training code."
    )


def resolve_image_path(path_value):
    path = Path(str(path_value))

    if path.is_absolute():
        if path.is_file():
            return path
        raise FileNotFoundError(path)

    candidate = PROJECT_ROOT / path
    if candidate.is_file():
        return candidate

    candidate = SPLIT_PATH.parent / path
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(
        f"Could not resolve image path: {path_value}"
    )


# ============================================================
# 3. LOAD TEST DATA
# ============================================================

def load_test_data():
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split CSV not found: {SPLIT_PATH}"
        )

    df = pd.read_csv(SPLIT_PATH)

    print("Dataset columns:", list(df.columns))

    path_col = find_column(
        df,
        ["image_path", "filepath", "file_path", "path", "filename"],
        "image path",
    )

    label_col = find_column(
        df,
        ["label", "class", "target", "diagnosis"],
        "label",
    )

    split_col = find_column(
        df,
        ["split", "set", "subset", "partition"],
        "split",
    )

    df[split_col] = df[split_col].apply(normalize_split)

    test_df = df[df[split_col] == "test"].copy()

    if test_df.empty:
        raise ValueError(
            "No test samples found in dataset_split.csv."
        )

    test_df["resolved_path"] = test_df[path_col].apply(
        resolve_image_path
    )

    test_df["binary_label"] = test_df[label_col].apply(
        normalize_label
    )

    print(f"\nTest images: {len(test_df)}")
    print(
        "Class distribution:",
        test_df["binary_label"].value_counts().sort_index().to_dict(),
    )

    return test_df


# ============================================================
# 4. IMAGE PREPROCESSING
# ============================================================

def load_image(path, label):
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    # Match the resize setting in finetune_classweighted_model.py.
    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        antialias=True,
    )

    # Keep float pixel values in the 0-255 range, as in training.
    # Do not add another rescaling step unless the training pipeline
    # also applies that step outside the saved model.
    image = tf.cast(image, tf.float32)

    return image, label


def create_test_dataset(test_df):
    paths = test_df["resolved_path"].astype(str).tolist()

    labels = test_df["binary_label"].to_numpy(
        dtype=np.float32
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, labels


# ============================================================
# 5. EVALUATION
# ============================================================

def main():
    print("=" * 65)
    print("CLASS-WEIGHTED MODEL - THRESHOLD 0.55")
    print("=" * 65)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print(f"\nLoading model:\n{MODEL_PATH}")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print("Model loaded successfully.")

    test_df = load_test_data()
    test_ds, y_true = create_test_dataset(test_df)

    print("\nGenerating predictions...")

    raw_predictions = model.predict(
        test_ds,
        verbose=1,
    )

    probabilities = np.asarray(
        raw_predictions
    ).reshape(-1)

    y_true = y_true.astype(int)

    if len(probabilities) != len(y_true):
        raise ValueError(
            "Prediction count does not match test label count."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Model generated non-finite predictions."
        )

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError(
            "Model outputs are outside [0, 1]. "
            "Check the model's output activation."
        )

    y_pred = (
        probabilities >= THRESHOLD
    ).astype(int)

    # ========================================================
    # 6. METRICS
    # ========================================================

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, zero_division=0
    )

    sensitivity = recall_score(
        y_true, y_pred, pos_label=1, zero_division=0
    )

    specificity = recall_score(
        y_true, y_pred, pos_label=0, zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, zero_division=0
    )

    brier = brier_score_loss(
        y_true, probabilities
    )

    auc = (
        float(roc_auc_score(y_true, probabilities))
        if len(np.unique(y_true)) == 2
        else None
    )

    cm = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model": "parkinson_effnetv2s_classweighted",
        "evaluation_split": "test",
        "threshold": THRESHOLD,
        "resize_antialias": True,
        "test_samples": int(len(y_true)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "roc_auc": auc,
        "brier_score": float(brier),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm.tolist(),
    }

    # ========================================================
    # 7. DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 65)
    print("TEST RESULTS")
    print("=" * 65)

    print(f"Threshold   : {THRESHOLD:.2f}")
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Sensitivity : {sensitivity:.4f}")
    print(f"Specificity : {specificity:.4f}")
    print(f"F1-score    : {f1:.4f}")

    if auc is not None:
        print(f"ROC-AUC     : {auc:.4f}")
    else:
        print("ROC-AUC     : unavailable (only one class in test set)")

    print(f"Brier score : {brier:.4f}")

    print("\nConfusion matrix")
    print("Rows = actual; columns = predicted")
    print("Class 0 = Healthy; Class 1 = Parkinson")
    print(cm)

    print("\nTN:", tn, "| FP:", fp, "| FN:", fn, "| TP:", tp)

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Healthy", "Parkinson"],
            zero_division=0,
        )
    )

    # ========================================================
    # 8. SAVE SEPARATE REPORTS
    # ========================================================

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    predictions_df = pd.DataFrame({
        "image_path": test_df["resolved_path"].astype(str).tolist(),
        "actual_label": y_true,
        "predicted_probability": probabilities,
        "predicted_label": y_pred,
        "threshold": THRESHOLD,
    })

    predictions_df["actual_class"] = predictions_df[
        "actual_label"
    ].map({0: "Healthy", 1: "Parkinson"})

    predictions_df["predicted_class"] = predictions_df[
        "predicted_label"
    ].map({0: "Healthy", 1: "Parkinson"})

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print("\nEvaluation completed.")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Predictions: {PREDICTIONS_PATH}")
    print("Previous reports were not overwritten.")


if __name__ == "__main__":
    main()
