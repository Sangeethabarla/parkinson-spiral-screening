
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
    roc_auc_score,
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

# NEW output names: original reports are never overwritten.
CSV_PATH = (
    REPORTS_DIR
    / "threshold_analysis_classweighted_preprocesscheck.csv"
)

JSON_PATH = (
    REPORTS_DIR
    / "threshold_analysis_classweighted_preprocesscheck.json"
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4
SEED = 42

THRESHOLDS = [
    0.20, 0.25, 0.30, 0.35, 0.40,
    0.45, 0.50, 0.55, 0.60, 0.65,
    0.70, 0.75, 0.80,
]

# Class convention:
# 0 = Healthy
# 1 = Parkinson


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates, description):
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not find {description} column.\n"
        f"Available columns: {list(df.columns)}\n"
        f"Expected one of: {candidates}"
    )


def normalize_split(value):
    value = str(value).strip().lower()

    aliases = {
        "validation": "val",
        "valid": "val",
        "validation_set": "val",
        "val_set": "val",
        "val": "val",
        "training": "train",
        "training_set": "train",
        "train_set": "train",
        "train": "train",
        "testing": "test",
        "test_set": "test",
        "testing_set": "test",
        "test": "test",
    }

    return aliases.get(value, value)


def normalize_label(value):
    if isinstance(value, (int, np.integer)):
        if value in (0, 1):
            return int(value)

    if isinstance(value, (float, np.floating)):
        if value in (0.0, 1.0):
            return int(value)

    label = str(value).strip().lower()

    healthy_labels = {
        "0", "healthy", "control", "normal",
        "non-parkinson", "non_parkinson", "nonparkinson",
    }

    parkinson_labels = {
        "1", "parkinson", "parkinsons", "parkinson's",
        "pd", "positive", "patient",
    }

    if label in healthy_labels:
        return 0

    if label in parkinson_labels:
        return 1

    raise ValueError(
        f"Unknown label: {value!r}. "
        "Update normalize_label() for your dataset."
    )


def resolve_path(path_value):
    path = Path(str(path_value))

    if path.is_absolute():
        return path

    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate

    candidate = SPLIT_PATH.parent / path
    if candidate.exists():
        return candidate

    return PROJECT_ROOT / path


# ============================================================
# 3. LOAD VALIDATION DATA ONLY
# ============================================================

def load_validation_data():
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_PATH}"
        )

    df = pd.read_csv(SPLIT_PATH)
    print("Dataset columns:", list(df.columns))

    path_column = find_column(
        df,
        ["image_path", "filepath", "file_path", "path", "filename"],
        "image path",
    )

    label_column = find_column(
        df,
        ["label", "class", "target", "diagnosis"],
        "label",
    )

    split_column = find_column(
        df,
        ["split", "set", "subset", "partition"],
        "split",
    )

    df[split_column] = df[split_column].apply(normalize_split)

    # Never use the test set for threshold selection.
    val_df = df[df[split_column] == "val"].copy()

    if val_df.empty:
        raise ValueError(
            "No validation rows found. Check split values "
            "in dataset_split.csv."
        )

    val_df["resolved_path"] = val_df[path_column].apply(resolve_path)
    val_df["binary_label"] = val_df[label_column].apply(normalize_label)

    missing = [
        str(path)
        for path in val_df["resolved_path"]
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Validation images are missing. First paths:\n"
            + "\n".join(missing[:10])
        )

    print(f"\nValidation images: {len(val_df)}")
    print(
        "Class distribution (0=Healthy, 1=Parkinson):",
        val_df["binary_label"].value_counts().sort_index().to_dict(),
    )

    return val_df


# ============================================================
# 4. BUILD VALIDATION DATASET
# ============================================================

def create_validation_dataset(val_df):
    image_paths = val_df["resolved_path"].astype(str).tolist()

    labels = val_df["binary_label"].astype(
        np.float32
    ).to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    def load_image(path, label):
        image_bytes = tf.io.read_file(path)

        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )

        image.set_shape([None, None, 3])

        # Match the training script's resize preprocessing.
        # This changes input resizing, NOT model weights.
        image = tf.image.resize(
            image,
            IMAGE_SIZE,
            method="bilinear",
            antialias=True,
        )

        # Keep the same float32 pixel scale (0-255).
        image = tf.cast(image, tf.float32)

        return image, label

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, labels


# ============================================================
# 5. RUN VALIDATION PREDICTIONS
# ============================================================

def main():
    tf.keras.utils.set_random_seed(SEED)

    print("=" * 65)
    print("CLASS-WEIGHTED VALIDATION PREPROCESSING CHECK")
    print("=" * 65)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print(f"\nLoading existing model:\n{MODEL_PATH}")
    print("Model will NOT be retrained or saved.")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    val_df = load_validation_data()
    val_dataset, y_true = create_validation_dataset(val_df)

    print("\nGenerating validation predictions...")

    predictions = model.predict(val_dataset, verbose=1)
    probabilities = np.asarray(predictions).reshape(-1)

    if len(probabilities) != len(y_true):
        raise ValueError(
            "Prediction count does not match validation labels."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "The model generated non-finite predictions."
        )

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError(
            "Model outputs are not probabilities in [0, 1]. "
            "Check the model output layer."
        )

    y_true = y_true.astype(int)

    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, probabilities))
    else:
        auc = None

    # ========================================================
    # 6. COMPARE THRESHOLDS
    # ========================================================

    results = []

    for threshold in THRESHOLDS:
        y_pred = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        accuracy = accuracy_score(y_true, y_pred)

        precision = precision_score(
            y_true, y_pred, zero_division=0
        )

        sensitivity = recall_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        )

        specificity = recall_score(
            y_true,
            y_pred,
            pos_label=0,
            zero_division=0,
        )

        f1 = f1_score(
            y_true, y_pred, zero_division=0
        )

        balanced_accuracy = (sensitivity + specificity) / 2

        results.append({
            "threshold": float(threshold),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "f1_score": float(f1),
            "balanced_accuracy": float(balanced_accuracy),
            "roc_auc": auc,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        })

    results_df = pd.DataFrame(results)

    display_columns = [
        "threshold",
        "accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1_score",
        "balanced_accuracy",
        "tn", "fp", "fn", "tp",
    ]

    print("\n" + "=" * 65)
    print("VALIDATION THRESHOLD COMPARISON")
    print("=" * 65)

    print(
        results_df[display_columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # ========================================================
    # 7. SELECT A VALIDATION CANDIDATE
    # ========================================================

    # Illustrative selection rule:
    # sensitivity >= 0.90, then maximize specificity.
    eligible = results_df[
        results_df["sensitivity"] >= 0.90
    ].copy()

    recommended = None

    if not eligible.empty:
        eligible = eligible.sort_values(
            by=[
                "specificity",
                "balanced_accuracy",
                "threshold",
            ],
            ascending=[False, False, True],
        )

        recommended = eligible.iloc[0].to_dict()

        print("\n" + "=" * 65)
        print("VALIDATION-BASED CANDIDATE")
        print("=" * 65)

        print(
            "Rule: sensitivity >= 0.90, then maximize specificity."
        )
        print(f"Threshold   : {recommended['threshold']:.2f}")
        print(f"Accuracy    : {recommended['accuracy']:.4f}")
        print(f"Sensitivity : {recommended['sensitivity']:.4f}")
        print(f"Specificity : {recommended['specificity']:.4f}")
        print(f"Precision   : {recommended['precision']:.4f}")
        print(f"F1-score    : {recommended['f1_score']:.4f}")
        print(
            "Balanced accuracy: "
            f"{recommended['balanced_accuracy']:.4f}"
        )

        print("\nValidation confusion matrix:")
        print(np.array([
            [recommended["tn"], recommended["fp"]],
            [recommended["fn"], recommended["tp"]],
        ]))
    else:
        print(
            "\nNo tested threshold achieved sensitivity >= 0.90."
        )

    # ========================================================
    # 8. SAVE TO NEW FILES ONLY
    # ========================================================

    results_df.to_csv(CSV_PATH, index=False)

    output = {
        "model": "parkinson_effnetv2s_classweighted",
        "evaluation_split": "validation",
        "validation_samples": int(len(y_true)),
        "preprocessing": {
            "image_size": list(IMAGE_SIZE),
            "resize_method": "bilinear",
            "antialias": True,
            "pixel_scale": "float32, original 0-255 scale",
        },
        "class_convention": {
            "0": "Healthy",
            "1": "Parkinson",
        },
        "roc_auc": auc,
        "threshold_results": results,
        "selection_rule": (
            "Among tested thresholds with sensitivity >= 0.90, "
            "choose the highest specificity; ties use balanced "
            "accuracy and then lower threshold."
        ),
        "recommended_validation_candidate": recommended,
        "warning": (
            "Validation-based candidate only, not a clinical threshold. "
            "This run does not modify model weights. The preprocessing "
            "change may change predictions and metrics. Keep the "
            "previous reports for comparison and report final test "
            "performance separately."
        ),
    }

    with open(JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print("\n" + "=" * 65)
    print("PREPROCESSING CHECK COMPLETED")
    print("=" * 65)

    print(f"\nNew CSV saved to:\n{CSV_PATH}")
    print(f"New JSON saved to:\n{JSON_PATH}")

    print("\nOriginal threshold reports were not overwritten.")
    print("Model checkpoint was not modified.")
    print("Only validation data was used.")


if __name__ == "__main__":
    main()
