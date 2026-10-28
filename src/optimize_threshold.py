
from pathlib import Path
import csv
import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSV = ROOT / "data" / "processed" / "dataset_split.csv"
MODEL_PATH = ROOT / "artifacts" / "parkinson_effnetv2s_expanded.keras"
OUTPUT_PATH = ROOT / "reports" / "validation_threshold_analysis.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
    image = tf.cast(image, tf.float32)
    return image, label


def evaluate_threshold(y_true, probabilities, threshold):
    y_pred = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": round(float(threshold), 3),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "sensitivity": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "specificity": float(
            tn / (tn + fp) if (tn + fp) else 0
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    with SPLIT_CSV.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    validation_rows = [
        row for row in rows if row["split"] == "validation"
    ]

    if not validation_rows:
        raise ValueError("No validation rows found.")

    paths = []
    labels = []

    for row in validation_rows:
        path = Path(row["image_path"])
        if not path.is_absolute():
            path = ROOT / path

        if not path.is_file():
            raise FileNotFoundError(path)

        paths.append(str(path))
        labels.append(1 if row["label"] == "parkinson" else 0)

    labels = np.array(labels, dtype=int)

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )
    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.load_model(MODEL_PATH)
    probabilities = model.predict(dataset, verbose=1).reshape(-1)

    if len(np.unique(labels)) != 2:
        raise ValueError(
            "Validation data must contain both classes for ROC-AUC."
        )

    auc = float(roc_auc_score(labels, probabilities))

    # Explore thresholds on validation data only.
    thresholds = np.arange(0.20, 0.81, 0.025)
    results = [
        evaluate_threshold(labels, probabilities, threshold)
        for threshold in thresholds
    ]

    # Candidate policy: maximize specificity while retaining
    # at least 90% validation sensitivity.
    eligible = [
        result for result in results
        if result["sensitivity"] >= 0.90
    ]

    best = max(
        eligible,
        key=lambda result: (
            result["specificity"],
            result["f1_score"],
        ),
        default=None,
    )

    report = {
        "model": MODEL_PATH.name,
        "validation_images": int(len(labels)),
        "validation_roc_auc": auc,
        "selection_rule": (
            "Maximize validation specificity while sensitivity >= 0.90"
        ),
        "threshold_results": results,
        "selected_threshold": best,
        "warning": (
            "Threshold selected on validation data only. "
            "Confirm the final performance once on the held-out test set."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n========== VALIDATION THRESHOLD ANALYSIS ==========")
    print(f"Validation images: {len(labels)}")
    print(f"Validation ROC-AUC: {auc:.4f}")

    print("\nBaseline threshold = 0.500")
    print(evaluate_threshold(labels, probabilities, 0.5))

    if best:
        print("\nSelected validation threshold:")
        print(best)
    else:
        print(
            "\nNo tested threshold achieved validation sensitivity >= 0.90."
        )

    print(f"\nFull report saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
