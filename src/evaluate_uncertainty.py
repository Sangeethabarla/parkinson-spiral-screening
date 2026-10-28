
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.common import ROOT, load_config, load_image


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = (
    ROOT / "artifacts" / "parkinson_effnetv2s_classweighted.keras"
)

MANIFEST_PATH = ROOT / "data" / "processed" / "dataset_split.csv"

REPORT_DIR = ROOT / "reports" / "uncertainty"
CSV_PATH = REPORT_DIR / "mc_dropout_test_results.csv"
JSON_PATH = REPORT_DIR / "mc_dropout_metrics.json"

THRESHOLD = 0.58
MC_DROPOUT_PASSES = 20
BATCH_SIZE = 8


# ============================================================
# 2. MC DROPOUT PROBE
# ============================================================

def build_mc_dropout_probe(model):

    class MCDropoutProbe(tf.keras.Model):

        def __init__(self, trained_model):
            super().__init__(name="classifier_mc_dropout_probe")

            self.backbone = trained_model.get_layer(
                "efficientnetv2-s"
            )
            self.pool = trained_model.get_layer(
                "global_average_pooling"
            )
            self.drop1 = tf.keras.layers.Dropout(
                trained_model.get_layer("classifier_dropout").rate
            )
            self.dense = trained_model.get_layer("feature_dense")
            self.drop2 = tf.keras.layers.Dropout(
                trained_model.get_layer("feature_dropout").rate
            )
            self.output_layer = trained_model.get_layer("pd_probability")

        def call(self, inputs, training=False):
            # Backbone remains in inference mode.
            x = self.backbone(inputs, training=False)
            x = self.pool(x)

            # Dropout stays enabled for every MC pass.
            x = self.drop1(x, training=True)
            x = self.dense(x)
            x = self.drop2(x, training=True)

            return self.output_layer(x)

    return MCDropoutProbe(model)


# ============================================================
# 3. FIND MANIFEST COLUMNS
# ============================================================

def find_column(df, candidates, description):
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not identify {description} column. "
        f"Available columns: {list(df.columns)}"
    )


def normalize_label(value):
    """
    Map healthy to 0 and Parkinson's to 1.
    Supports common numeric and text label formats.
    """
    if pd.isna(value):
        raise ValueError("Encountered an empty class label.")

    if isinstance(value, (int, float, np.integer, np.floating)):
        number = int(value)
        if number in (0, 1) and float(value) == number:
            return number

    text = str(value).strip().lower()

    healthy_labels = {
        "0", "healthy", "control", "normal", "non-parkinson",
        "non_parkinson", "non-parkinson's", "non_parkinsons"
    }

    parkinson_labels = {
        "1", "parkinson", "parkinsons", "parkinson's",
        "pd", "patient"
    }

    if text in healthy_labels:
        return 0

    if text in parkinson_labels:
        return 1

    raise ValueError(
        f"Unrecognized label: {value!r}. "
        "Check the label mapping in the dataset manifest."
    )


# ============================================================
# 4. LOAD TEST DATA
# ============================================================

def load_test_dataframe():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    path_col = find_column(
        df,
        ["path", "image_path", "filepath", "file_path", "image"],
        "image path"
    )

    split_col = find_column(
        df,
        ["split", "subset", "partition", "set"],
        "dataset split"
    )

    label_col = find_column(
        df,
        ["label", "class", "target", "class_name", "diagnosis"],
        "class label"
    )

    test_df = df[
        df[split_col].astype(str).str.strip().str.lower() == "test"
    ].copy()

    if test_df.empty:
        raise ValueError(
            f"No test rows found in '{split_col}'. "
            f"Observed values: {df[split_col].unique().tolist()}"
        )

    test_df["resolved_path"] = test_df[path_col].apply(
        lambda value: str(
            Path(value)
            if Path(value).is_absolute()
            else ROOT / Path(value)
        )
    )

    test_df["actual_label"] = test_df[label_col].apply(
        normalize_label
    )

    missing_files = [
        path for path in test_df["resolved_path"]
        if not Path(path).exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Some test images were not found. First missing paths: "
            + ", ".join(missing_files[:5])
        )

    return test_df.reset_index(drop=True)


# ============================================================
# 5. RUN MC DROPOUT EVALUATION
# ============================================================

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    test_df = load_test_dataframe()

    print("=" * 65)
    print("MC DROPOUT UNCERTAINTY EVALUATION")
    print("=" * 65)
    print(f"Test images: {len(test_df)}")
    print(f"MC passes per image: {MC_DROPOUT_PASSES}")
    print(f"Decision threshold: {THRESHOLD}")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    mc_probe = build_mc_dropout_probe(model)

    result_rows = []

    for start in range(0, len(test_df), BATCH_SIZE):
        batch_df = test_df.iloc[
            start:start + BATCH_SIZE
        ]

        # Preprocess individual images using the project's
        # existing preprocessing function.
        images = [
            load_image(path)
            for path in batch_df["resolved_path"].tolist()
        ]

        image_batch = tf.stack(images, axis=0)

        # Ordinary deterministic predictions.
        raw_probabilities = (
            model(image_batch, training=False)
            .numpy()
            .reshape(-1)
        )

        # Store one prediction per pass, per image.
        mc_pass_results = []

        for pass_index in range(MC_DROPOUT_PASSES):
            batch_probabilities = (
                mc_probe(image_batch)
                .numpy()
                .reshape(-1)
            )

            mc_pass_results.append(batch_probabilities)

        # Shape: [MC passes, images in batch].
        mc_pass_results = np.asarray(
            mc_pass_results,
            dtype=np.float64
        )

        mc_means = mc_pass_results.mean(axis=0)
        mc_variances = mc_pass_results.var(axis=0, ddof=0)
        mc_stds = mc_pass_results.std(axis=0, ddof=0)

        actual_labels = batch_df["actual_label"].to_numpy(
            dtype=int
        )

        for index, (_, row) in enumerate(batch_df.iterrows()):
            raw_probability = float(raw_probabilities[index])
            mc_mean = float(mc_means[index])

            # Keep the evaluated decision rule consistent:
            # classification is based on the ordinary model
            # probability and the selected threshold.
            predicted_label = int(
                raw_probability >= THRESHOLD
            )

            result_rows.append({
                "image_path": row["resolved_path"],
                "actual_label": int(actual_labels[index]),
                "actual_class": (
                    "PARKINSON" if actual_labels[index] == 1
                    else "HEALTHY"
                ),
                "raw_pd_probability": raw_probability,
                "mc_dropout_mean": mc_mean,
                "mc_dropout_variance": float(
                    mc_variances[index]
                ),
                "mc_dropout_standard_deviation": float(
                    mc_stds[index]
                ),
                "mc_dropout_passes": MC_DROPOUT_PASSES,
                "predicted_label": predicted_label,
                "predicted_class": (
                    "PARKINSON" if predicted_label == 1
                    else "HEALTHY"
                ),
                "correct": bool(
                    predicted_label == actual_labels[index]
                ),
                "decision_threshold": THRESHOLD
            })

        print(
            f"Processed {min(start + BATCH_SIZE, len(test_df))}"
            f"/{len(test_df)} images"
        )

    results_df = pd.DataFrame(result_rows)

    y_true = results_df["actual_label"].to_numpy(dtype=int)
    y_pred = results_df["predicted_label"].to_numpy(dtype=int)
    y_score = results_df["raw_pd_probability"].to_numpy(
        dtype=float
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    sensitivity = (
        float(tp / (tp + fn))
        if (tp + fn) > 0 else None
    )

    specificity = (
        float(tn / (tn + fp))
        if (tn + fp) > 0 else None
    )

    try:
        roc_auc = (
            float(roc_auc_score(y_true, y_score))
            if len(np.unique(y_true)) == 2
            else None
        )
    except ValueError:
        roc_auc = None

    # Basic summaries of MC Dropout variability.
    uncertainty_summary = {
        "mean_mc_variance": float(
            results_df["mc_dropout_variance"].mean()
        ),
        "median_mc_variance": float(
            results_df["mc_dropout_variance"].median()
        ),
        "mean_mc_standard_deviation": float(
            results_df[
                "mc_dropout_standard_deviation"
            ].mean()
        ),
        "median_mc_standard_deviation": float(
            results_df[
                "mc_dropout_standard_deviation"
            ].median()
        )
    }

    metrics = {
        "model": MODEL_PATH.name,
        "test_image_count": int(len(results_df)),
        "mc_dropout_passes": MC_DROPOUT_PASSES,
        "decision_threshold": THRESHOLD,
        "positive_class": "PARKINSON",
        "label_mapping": {
            "HEALTHY": 0,
            "PARKINSON": 1
        },
        "classification_probability_source": (
            "Deterministic model probability; MC Dropout mean "
            "and variance are reported separately."
        ),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": float(
            precision_score(
                y_true, y_pred, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                y_true, y_pred, zero_division=0
            )
        ),
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "uncertainty_summary": uncertainty_summary,
        "warning": (
            "Research evaluation only; not a clinical validation. "
            "The test set has been used in prior experiments, so "
            "these results should not be treated as an untouched "
            "independent final performance estimate."
        )
    }

    results_df.to_csv(CSV_PATH, index=False)

    with open(JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\n" + "=" * 65)
    print("TEST-SET CLASSIFICATION METRICS")
    print("=" * 65)

    print(f"Accuracy          : {metrics['accuracy']:.4f}")
    print(
        f"Balanced accuracy : "
        f"{metrics['balanced_accuracy']:.4f}"
    )
    print(f"Sensitivity       : {sensitivity}")
    print(f"Specificity       : {specificity}")
    print(f"Precision          : {metrics['precision']:.4f}")
    print(f"F1-score           : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC            : {roc_auc}")
    print(f"Confusion matrix   : TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    print("\nMC DROPOUT UNCERTAINTY SUMMARY")
    print("=" * 65)
    print(
        "Mean variance      : "
        f"{uncertainty_summary['mean_mc_variance']:.6f}"
    )
    print(
        "Median variance    : "
        f"{uncertainty_summary['median_mc_variance']:.6f}"
    )
    print(
        "Mean std. deviation: "
        f"{uncertainty_summary['mean_mc_standard_deviation']:.6f}"
    )

    print("\nSaved files:")
    print(f"CSV : {CSV_PATH}")
    print(f"JSON: {JSON_PATH}")
    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
