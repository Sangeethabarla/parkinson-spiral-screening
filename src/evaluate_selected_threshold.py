
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
)

ROOT = Path(__file__).resolve().parents[1]

SPLIT_CSV = ROOT / "data" / "processed" / "dataset_split.csv"
MODEL_PATH = ROOT / "artifacts" / "parkinson_effnetv2s_expanded.keras"
OUTPUT_PATH = ROOT / "reports" / "selected_threshold_test_metrics.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4
THRESHOLD = 0.625


def load_image(path, label):
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
        antialias=True,
    )
    return tf.cast(image, tf.float32), label


def main():
    with SPLIT_CSV.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    test_rows = [
        row for row in rows
        if row["split"] == "test"
    ]

    if not test_rows:
        raise ValueError("No test rows found.")

    paths = []
    y_true = []

    for row in test_rows:
        path = Path(row["image_path"])
        if not path.is_absolute():
            path = ROOT / path

        if not path.is_file():
            raise FileNotFoundError(path)

        paths.append(str(path))
        y_true.append(
            1 if row["label"] == "parkinson" else 0
        )

    y_true = np.asarray(y_true, dtype=int)

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, y_true)
    )
    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(BATCH_SIZE).prefetch(
        tf.data.AUTOTUNE
    )

    model = tf.keras.models.load_model(MODEL_PATH)
    probabilities = model.predict(
        dataset,
        verbose=1,
    ).reshape(-1)

    y_pred = (
        probabilities >= THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "threshold": THRESHOLD,
        "test_images": int(len(y_true)),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true, y_pred, zero_division=0
            )
        ),
        "sensitivity": float(
            recall_score(
                y_true, y_pred, zero_division=0
            )
        ),
        "specificity": float(
            tn / (tn + fp) if tn + fp else 0
        ),
        "f1_score": float(
            f1_score(
                y_true, y_pred, zero_division=0
            )
        ),
        "roc_auc": (
            float(roc_auc_score(y_true, probabilities))
            if len(np.unique(y_true)) == 2
            else None
        ),
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
        "note": (
            "Threshold chosen using validation data only. "
            "Test metrics are from the held-out test set."
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\n========== SELECTED THRESHOLD TEST RESULTS ==========")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
