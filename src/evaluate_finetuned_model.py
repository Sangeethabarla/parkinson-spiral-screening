
from pathlib import Path
import csv
import json

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ==================================================
# 1. CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_split.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "parkinson_effnetv2s_expanded.keras"
)

REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "test_metrics_expanded.json"
PREDICTIONS_PATH = REPORTS_DIR / "test_predictions_expanded.csv"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4

# Labels used during training:
# healthy = 0, parkinson = 1
CLASS_NAMES = ["healthy", "parkinson"]
THRESHOLD = 0.5


# ==================================================
# 2. LOAD TEST DATA
# ==================================================

def load_test_rows():
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"Split CSV not found: {SPLIT_CSV}"
        )

    with SPLIT_CSV.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    test_rows = [
        row for row in rows
        if row["split"] == "test"
    ]

    if not test_rows:
        raise ValueError(
            "No test rows found in dataset_split.csv"
        )

    for row in test_rows:
        row["image_path"] = str(
            PROJECT_ROOT / row["image_path"]
        )

        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])

        if row["label"] not in CLASS_NAMES:
            raise ValueError(
                f"Unexpected label: {row['label']}"
            )

    return test_rows


# ==================================================
# 3. IMAGE PREPROCESSING
# ==================================================

def load_image(path, label):
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        antialias=True,
    )

    # Keep pixel range 0-255.
    # EfficientNetV2 includes preprocessing internally.
    image = tf.cast(image, tf.float32)

    return image, label


def make_test_dataset(rows):
    paths = [row["image_path"] for row in rows]

    labels = [
        CLASS_NAMES.index(row["label"])
        for row in rows
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ==================================================
# 4. EVALUATE MODEL
# ==================================================

def main():
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run python src/train_model.py first."
        )

    test_rows = load_test_rows()
    test_ds = make_test_dataset(test_rows)

    y_true = np.array([
        CLASS_NAMES.index(row["label"])
        for row in test_rows
    ])

    print("\nLoading expanded EfficientNetV2-S...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"Test images: {len(test_rows)}")
    print(f"Model: {MODEL_PATH}")

    # Predict probabilities for the Parkinson class.
    probabilities = model.predict(
        test_ds,
        verbose=1,
    ).reshape(-1)

    if len(probabilities) != len(y_true):
        raise ValueError(
            "Prediction count does not match test labels."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "Predictions contain NaN or infinite values."
        )

    y_pred = (
        probabilities >= THRESHOLD
    ).astype(int)

    # Confusion matrix order:
    # [[True Healthy, False Parkinson],
    #  [Missed Parkinson, Detected Parkinson]]
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    sensitivity = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else None
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    # ROC-AUC requires both classes in the test set.
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(
            y_true,
            probabilities,
        )
    else:
        auc = None

    # Report counts and metrics.
    metrics = {
        "model": MODEL_PATH.name,
        "test_images": int(len(y_true)),
        "threshold": THRESHOLD,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "sensitivity_recall": float(sensitivity),
        "specificity": (
            float(specificity)
            if specificity is not None
            else None
        ),
        "f1_score": float(f1),
        "roc_auc": (
            float(auc)
            if auc is not None
            else None
        ),
        "true_healthy": int(tn),
        "false_parkinson_predictions": int(fp),
        "missed_parkinson_cases": int(fn),
        "detected_parkinson_cases": int(tp),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    # Save per-image predictions for inspection.
    with PREDICTIONS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "source_dataset",
                "actual_label",
                "predicted_label",
                "parkinson_probability",
                "correct",
            ],
        )

        writer.writeheader()

        for row, actual, pred, probability in zip(
            test_rows,
            y_true,
            y_pred,
            probabilities,
        ):
            writer.writerow({
                "filename": row.get("filename", ""),
                "source_dataset": row.get(
                    "source_dataset", ""
                ),
                "actual_label": CLASS_NAMES[int(actual)],
                "predicted_label": CLASS_NAMES[int(pred)],
                "parkinson_probability": round(
                    float(probability), 6
                ),
                "correct": bool(actual == pred),
            })

    # Display results.
    print("\n========== TEST RESULTS ==========")
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Sensitivity:    {sensitivity:.4f}")
    print(
        "Specificity:    "
        + (
            f"{specificity:.4f}"
            if specificity is not None
            else "Not available"
        )
    )
    print(f"F1-score:       {f1:.4f}")
    print(
        "ROC-AUC:        "
        + (
            f"{auc:.4f}"
            if auc is not None
            else "Not available"
        )
    )

    print("\nConfusion Matrix:")
    print("Rows = actual [healthy, parkinson]")
    print("Columns = predicted [healthy, parkinson]")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    print(f"Metrics saved: {METRICS_PATH}")
    print(f"Predictions saved: {PREDICTIONS_PATH}")

    print(
        "\nResearch limitation: results from this small "
        "dataset do not establish clinical diagnostic accuracy. "
        "Filename-derived groups have not been independently "
        "verified as participant identities."
    )


if __name__ == "__main__":
    main()
