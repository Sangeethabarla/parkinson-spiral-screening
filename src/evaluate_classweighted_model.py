
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
    roc_auc_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

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

METRICS_PATH = REPORTS_DIR / "test_metrics_classweighted.json"
PREDICTIONS_PATH = REPORTS_DIR / "test_predictions_classweighted.csv"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4
THRESHOLD = 0.5
SEED = 42

# Update these only if your dataset_split.csv uses different names.
PATH_COLUMN_CANDIDATES = [
    "filepath",
    "file_path",
    "image_path",
    "path",
    "filename",
]

LABEL_COLUMN_CANDIDATES = [
    "label",
    "class",
    "target",
    "diagnosis",
]

SPLIT_COLUMN_CANDIDATES = [
    "split",
    "set",
    "subset",
    "partition",
]


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def find_column(dataframe, candidates, description):
    """Find a column using a list of possible column names."""

    for column in candidates:
        if column in dataframe.columns:
            return column

    raise ValueError(
        f"Could not find the {description} column.\n"
        f"Available columns: {list(dataframe.columns)}\n"
        f"Expected one of: {candidates}"
    )


def normalize_split(value):
    """Normalize split names for filtering."""

    value = str(value).strip().lower()

    aliases = {
        "testing": "test",
        "testing_set": "test",
        "test_set": "test",
        "validation": "val",
        "valid": "val",
        "validation_set": "val",
        "val_set": "val",
        "training": "train",
        "training_set": "train",
        "train_set": "train",
    }

    return aliases.get(value, value)


def normalize_label(value):
    """
    Convert common binary label formats to integers.

    Class convention:
        0 = Healthy
        1 = Parkinson

    If your dataset uses a different convention, update this
    function to match the training script.
    """

    if isinstance(value, (int, np.integer)):
        if value in (0, 1):
            return int(value)

    if isinstance(value, (float, np.floating)):
        if value in (0.0, 1.0):
            return int(value)

    label = str(value).strip().lower()

    healthy_labels = {
        "0",
        "healthy",
        "control",
        "normal",
        "non-parkinson",
        "non_parkinson",
        "nonparkinson",
    }

    parkinson_labels = {
        "1",
        "parkinson",
        "parkinsons",
        "parkinson's",
        "pd",
        "positive",
        "patient",
    }

    if label in healthy_labels:
        return 0

    if label in parkinson_labels:
        return 1

    raise ValueError(
        f"Unrecognized label: {value!r}. "
        "Update normalize_label() to match your dataset."
    )


def resolve_image_path(path_value):
    """Resolve image paths relative to the project if necessary."""

    path = Path(str(path_value))

    if path.is_absolute():
        return path

    # First try relative to the project root.
    candidate = PROJECT_ROOT / path

    if candidate.exists():
        return candidate

    # Also try relative to the split CSV directory.
    candidate = SPLIT_PATH.parent / path

    if candidate.exists():
        return candidate

    # Return the project-relative candidate for a useful error.
    return PROJECT_ROOT / path


def load_image(image_path):
    """Read and preprocess an image."""

    image_bytes = tf.io.read_file(str(image_path))

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method="bilinear",
    )

    image = tf.cast(image, tf.float32)

    # EfficientNetV2 Keras models commonly include their own
    # preprocessing/rescaling. Keep inputs in the same 0–255
    # range used by the training pipeline.
    return image


# ============================================================
# 3. LOAD TEST SPLIT
# ============================================================

def load_test_data():
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset split file not found: {SPLIT_PATH}"
        )

    dataframe = pd.read_csv(SPLIT_PATH)

    print("\nDataset split columns:")
    print(list(dataframe.columns))

    path_column = find_column(
        dataframe,
        PATH_COLUMN_CANDIDATES,
        "image path",
    )

    label_column = find_column(
        dataframe,
        LABEL_COLUMN_CANDIDATES,
        "label",
    )

    split_column = find_column(
        dataframe,
        SPLIT_COLUMN_CANDIDATES,
        "split",
    )

    dataframe[split_column] = dataframe[split_column].apply(
        normalize_split
    )

    test_df = dataframe[
        dataframe[split_column] == "test"
    ].copy()

    if test_df.empty:
        raise ValueError(
            "No test rows were found in dataset_split.csv. "
            "Check the split column values."
        )

    test_df["resolved_path"] = test_df[path_column].apply(
        resolve_image_path
    )

    missing_files = [
        str(path)
        for path in test_df["resolved_path"]
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Some test images could not be found. "
            "First missing paths:\n"
            + "\n".join(missing_files[:10])
        )

    test_df["binary_label"] = test_df[label_column].apply(
        normalize_label
    )

    print(f"\nTest images: {len(test_df)}")
    print(
        "Test class distribution:",
        test_df["binary_label"].value_counts().sort_index().to_dict(),
    )

    return test_df


# ============================================================
# 4. CREATE TEST DATASET
# ============================================================

def create_test_dataset(test_df):
    image_paths = test_df["resolved_path"].astype(str).tolist()

    labels = test_df["binary_label"].astype(
        np.float32
    ).to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    def load_example(path, label):
        image = tf.io.read_file(path)

        image = tf.io.decode_image(
            image,
            channels=3,
            expand_animations=False,
        )

        image.set_shape([None, None, 3])

        image = tf.image.resize(
            image,
            IMAGE_SIZE,
            method="bilinear",
        )

        image = tf.cast(image, tf.float32)

        return image, label

    dataset = dataset.map(
        load_example,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, labels


# ============================================================
# 5. EVALUATE MODEL
# ============================================================

def main():
    tf.keras.utils.set_random_seed(SEED)

    print("=" * 65)
    print("CLASS-WEIGHTED PARKINSON'S MODEL EVALUATION")
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

    print("\nModel loaded successfully.")

    test_df = load_test_data()

    test_dataset, y_true = create_test_dataset(test_df)

    print("\nGenerating test predictions...")

    predictions = model.predict(
        test_dataset,
        verbose=1,
    )

    predictions = np.asarray(predictions).reshape(-1)

    if len(predictions) != len(y_true):
        raise ValueError(
            f"Prediction count ({len(predictions)}) does not match "
            f"test label count ({len(y_true)})."
        )

    if not np.isfinite(predictions).all():
        raise ValueError(
            "Model produced non-finite predictions."
        )

    if np.any(predictions < 0) or np.any(predictions > 1):
        raise ValueError(
            "Predictions are outside [0, 1]. "
            "Check whether the model outputs probabilities or logits."
        )

    y_pred = (
        predictions >= THRESHOLD
    ).astype(np.int32)

    # ========================================================
    # 6. CALCULATE METRICS
    # ========================================================

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    specificity = recall_score(
        y_true,
        y_pred,
        pos_label=0,
        zero_division=0,
    )

    brier = brier_score_loss(
        y_true,
        predictions,
    )

    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(
            y_true,
            predictions,
        )
    else:
        roc_auc = None
        print(
            "\nWarning: ROC-AUC cannot be calculated "
            "because the test set contains only one class."
        )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model": "parkinson_effnetv2s_classweighted",
        "threshold": THRESHOLD,
        "test_samples": int(len(y_true)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "sensitivity_recall": float(sensitivity),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "roc_auc": (
            float(roc_auc)
            if roc_auc is not None
            else None
        ),
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

    print(f"Threshold    : {THRESHOLD}")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Sensitivity  : {sensitivity:.4f}")
    print(f"Specificity  : {specificity:.4f}")
    print(f"F1-score     : {f1:.4f}")
    print(
        f"ROC-AUC      : "
        f"{roc_auc:.4f}" if roc_auc is not None
        else "ROC-AUC      : Not available"
    )
    print(f"Brier score  : {brier:.4f}")

    print("\nConfusion Matrix")
    print("Rows = actual class; columns = predicted class")
    print("Class 0 = Healthy; Class 1 = Parkinson")
    print(cm)

    print("\nConfusion Matrix Interpretation:")
    print(f"True Negatives  (Healthy correctly identified): {tn}")
    print(f"False Positives (Healthy predicted Parkinson) : {fp}")
    print(f"False Negatives (Parkinson missed)            : {fn}")
    print(f"True Positives  (Parkinson correctly detected): {tp}")

    print("\nClassification Report:")
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
    # 8. SAVE METRICS
    # ========================================================

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    # ========================================================
    # 9. SAVE PREDICTIONS
    # ========================================================

    predictions_df = pd.DataFrame({
        "image_path": test_df["resolved_path"].astype(str).tolist(),
        "actual_label": y_true.astype(int),
        "predicted_probability": predictions,
        "predicted_label": y_pred,
        "threshold": THRESHOLD,
    })

    predictions_df["actual_class"] = predictions_df[
        "actual_label"
    ].map({
        0: "Healthy",
        1: "Parkinson",
    })

    predictions_df["predicted_class"] = predictions_df[
        "predicted_label"
    ].map({
        0: "Healthy",
        1: "Parkinson",
    })

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETED")
    print("=" * 65)

    print(f"\nMetrics saved to:\n{METRICS_PATH}")
    print(f"\nPredictions saved to:\n{PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
