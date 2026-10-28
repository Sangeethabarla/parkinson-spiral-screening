
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
    balanced_accuracy_score,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_split.csv"
)

MODELS = {
    "classweighted": (
        PROJECT_ROOT
        / "artifacts"
        / "parkinson_effnetv2s_classweighted.keras"
    ),
    "mildweighted_exp1": (
        PROJECT_ROOT
        / "artifacts"
        / "parkinson_effnetv2s_mildweighted_exp1.keras"
    ),
}

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = np.round(
    np.arange(0.20, 0.801, 0.05),
    2,
)

LABEL_MAP = {
    "healthy": 0,
    "parkinson": 1,
}

tf.keras.utils.set_random_seed(SEED)


# ============================================================
# 2. LOAD VALIDATION DATA ONLY
# ============================================================

def load_validation_data():

    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found: {DATASET_CSV}"
        )

    df = pd.read_csv(DATASET_CSV)

    required_columns = {
        "image_path",
        "label",
        "split",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing CSV columns: {sorted(missing)}"
        )

    # Select validation rows only.
    split_values = (
        df["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    val_df = df[
        split_values.isin(["validation", "val"])
    ].copy()

    if val_df.empty:
        raise ValueError(
            "No validation rows found in dataset_split.csv."
        )

    val_df = val_df.reset_index(drop=True)

    normalized_labels = (
        val_df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    unknown_labels = (
        set(normalized_labels.unique())
        - set(LABEL_MAP.keys())
    )

    if unknown_labels:
        raise ValueError(
            f"Unknown labels: {unknown_labels}. "
            "Check the labels in dataset_split.csv."
        )

    y_true = normalized_labels.map(
        LABEL_MAP
    ).to_numpy(dtype=np.int32)

    image_paths = []

    for raw_path in val_df["image_path"]:

        path = Path(str(raw_path))

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        path = path.resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"Validation image not found: {path}"
            )

        image_paths.append(path)

    print("\n" + "=" * 60)
    print("VALIDATION DATASET")
    print("=" * 60)
    print(f"Validation images : {len(val_df)}")
    print(f"Healthy           : {np.sum(y_true == 0)}")
    print(f"Parkinson         : {np.sum(y_true == 1)}")
    print("Test split loaded : NO")

    return val_df, image_paths, y_true


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):

    # IMPORTANT:
    # image_path is a TensorFlow string tensor here.
    # Do not convert it using str() or Path().

    image_bytes = tf.io.read_file(image_path)

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

    # Preserve the 0-255 pixel range used during training.
    image = tf.cast(
        image,
        tf.float32,
    )

    return image


# ============================================================
# 4. CREATE VALIDATION DATASET
# ============================================================

def create_validation_dataset(
    image_paths,
    y_true,
):

    path_strings = [
        str(path) for path in image_paths
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            path_strings,
            y_true,
        )
    )

    dataset = dataset.map(
        lambda path, label: (
            preprocess_image(path),
            label,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ============================================================
# 5. GET MODEL PROBABILITIES
# ============================================================

def predict_probabilities(
    model_path,
    dataset,
):

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    print(f"\nLoading model: {model_path.name}")

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    probabilities = []

    for images, _ in dataset:

        outputs = model(
            images,
            training=False,
        )

        if isinstance(
            outputs,
            (list, tuple, dict),
        ):
            raise ValueError(
                "Expected one model output. "
                "Inspect the model architecture."
            )

        outputs = outputs.numpy()

        if outputs.ndim == 1:
            batch_probabilities = outputs

        elif (
            outputs.ndim == 2
            and outputs.shape[1] == 1
        ):
            batch_probabilities = outputs[:, 0]

        elif (
            outputs.ndim == 2
            and outputs.shape[1] == 2
        ):
            # Assumes output order is:
            # healthy, parkinson.
            batch_probabilities = outputs[:, 1]

        else:
            raise ValueError(
                "Unexpected model output shape: "
                f"{outputs.shape}"
            )

        # Validate that outputs are probabilities.
        if (
            np.any(batch_probabilities < 0)
            or np.any(batch_probabilities > 1)
        ):
            raise ValueError(
                "Model outputs are outside [0, 1]. "
                "This script expects sigmoid or softmax "
                "probabilities, not logits."
            )

        probabilities.extend(
            batch_probabilities.tolist()
        )

    del model

    return np.asarray(
        probabilities,
        dtype=np.float64,
    )


# ============================================================
# 6. CALCULATE CLASSIFICATION METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):

    y_pred = (
        probabilities >= threshold
    ).astype(np.int32)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(
            y_true,
            probabilities,
        )
    else:
        auc = float("nan")

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "roc_auc": float(auc),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# 7. ANALYZE ONE MODEL
# ============================================================

def analyze_model(
    model_name,
    model_path,
    dataset,
    y_true,
    validation_paths,
):

    probabilities = predict_probabilities(
        model_path,
        dataset,
    )

    if len(probabilities) != len(y_true):
        raise ValueError(
            "Prediction count does not match "
            "validation label count."
        )

    # Evaluate the same probability predictions
    # across all thresholds.
    rows = [
        calculate_metrics(
            y_true,
            probabilities,
            threshold,
        )
        for threshold in THRESHOLDS
    ]

    results_df = pd.DataFrame(rows)

    # Threshold 0.50 baseline.
    row_050 = results_df[
        np.isclose(
            results_df["threshold"],
            0.50,
        )
    ]

    if row_050.empty:
        raise ValueError(
            "Threshold 0.50 is missing from the grid."
        )

    metrics_050 = row_050.iloc[0]

    # Candidate threshold:
    # sensitivity >= 0.90, then maximize specificity.
    eligible = results_df[
        results_df["sensitivity"] >= 0.90
    ]

    if not eligible.empty:

        candidate = eligible.sort_values(
            by=[
                "specificity",
                "accuracy",
                "threshold",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        ).iloc[0]

    else:

        candidate = results_df.sort_values(
            by=[
                "balanced_accuracy",
                "f1_score",
            ],
            ascending=[
                False,
                False,
            ],
        ).iloc[0]

    selected_threshold = float(
        candidate["threshold"]
    )

    # --------------------------------------------------------
    # SAVE REPORTS WITHOUT OVERWRITING EXISTING FILES
    # --------------------------------------------------------

    csv_path = (
        REPORT_DIR
        / f"threshold_analysis_{model_name}_exp1.csv"
    )

    json_path = (
        REPORT_DIR
        / f"threshold_analysis_{model_name}_exp1.json"
    )

    predictions_path = (
        REPORT_DIR
        / f"validation_predictions_{model_name}_exp1.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False,
    )

    prediction_df = pd.DataFrame({
        "image_path": [
            str(path)
            for path in validation_paths
        ],
        "actual_label": y_true,
        "parkinson_probability": probabilities,
        "prediction_at_050": (
            probabilities >= 0.50
        ).astype(int),
        "prediction_at_candidate_threshold": (
            probabilities >= selected_threshold
        ).astype(int),
    })

    prediction_df.to_csv(
        predictions_path,
        index=False,
    )

    # Convert NumPy scalar values to standard Python types.
    report = {
        "model_name": model_name,
        "model_path": str(model_path),
        "evaluation_split": "validation_only",
        "validation_samples": int(len(y_true)),
        "preprocessing": {
            "image_size": list(IMAGE_SIZE),
            "resize_antialias": True,
            "pixel_range": "0-255",
        },
        "metrics_at_threshold_050": {
            key: (
                float(value)
                if isinstance(
                    value,
                    (np.integer, np.floating),
                )
                else value
            )
            for key, value in metrics_050.to_dict().items()
        },
        "validation_threshold_candidate": {
            key: (
                float(value)
                if isinstance(
                    value,
                    (np.integer, np.floating),
                )
                else value
            )
            for key, value in candidate.to_dict().items()
        },
        "threshold_selection_rule": (
            "Among validation thresholds with sensitivity >= 0.90, "
            "maximize specificity. Ties are resolved using accuracy "
            "and then the lower threshold. This is a development "
            "candidate, not a clinically validated threshold."
        ),
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print(f"MODEL: {model_name}")
    print("=" * 65)

    print("\nMetrics at threshold 0.50:")

    for key, value in metrics_050.to_dict().items():
        print(f"{key}: {value}")

    print("\nValidation threshold candidate:")

    for key, value in candidate.to_dict().items():
        print(f"{key}: {value}")

    print("\nSaved files:")
    print(csv_path)
    print(json_path)
    print(predictions_path)

    return {
        "model_name": model_name,
        "accuracy_at_050": float(
            metrics_050["accuracy"]
        ),
        "precision_at_050": float(
            metrics_050["precision"]
        ),
        "sensitivity_at_050": float(
            metrics_050["sensitivity"]
        ),
        "specificity_at_050": float(
            metrics_050["specificity"]
        ),
        "auc_at_050": float(
            metrics_050["roc_auc"]
        ),
        "candidate_threshold": selected_threshold,
        "candidate_accuracy": float(
            candidate["accuracy"]
        ),
        "candidate_precision": float(
            candidate["precision"]
        ),
        "candidate_sensitivity": float(
            candidate["sensitivity"]
        ),
        "candidate_specificity": float(
            candidate["specificity"]
        ),
        "candidate_f1": float(
            candidate["f1_score"]
        ),
        "candidate_balanced_accuracy": float(
            candidate["balanced_accuracy"]
        ),
    }


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":

    validation_df, validation_paths, y_true = (
        load_validation_data()
    )

    validation_dataset = create_validation_dataset(
        validation_paths,
        y_true,
    )

    comparison_rows = []

    for model_name, model_path in MODELS.items():

        result = analyze_model(
            model_name=model_name,
            model_path=model_path,
            dataset=validation_dataset,
            y_true=y_true,
            validation_paths=validation_paths,
        )

        comparison_rows.append(result)

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_path = (
        REPORT_DIR
        / "model_comparison_validation_exp1.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    print("\n" + "=" * 90)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 90)

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(
        f"\nComparison saved to:\n{comparison_path}"
    )

    print("\nCompleted successfully.")
    print("Evaluation used validation data only.")
    print("The test split was not loaded or evaluated.")
