
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)

from src.common import ROOT, load_config
from src.data import get_split, make_dataset


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = (
    ROOT
    / "artifacts"
    / "parkinson_effnetv2s_classweighted.keras"
)

REPORTS_DIR = ROOT / "reports"

# Class mapping expected by this evaluation:
# 0 = healthy
# 1 = parkinson
#
# Confirm this matches src.data and the model training code.

DEFAULT_THRESHOLD = 0.50
MIN_VALIDATION_SENSITIVITY = 0.80

THRESHOLD_CANDIDATES = np.linspace(
    0.05,
    0.95,
    181,
)


# ============================================================
# 2. LOAD MODEL
# ============================================================

def load_trained_model():
    """Load the existing trained model without retraining."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    print("\nLoading existing trained model...")
    print(f"Model path: {MODEL_PATH}")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

    return model


# ============================================================
# 3. GET PREDICTIONS
# ============================================================

def probs_with_mc(model, dataset, passes=1):
    """
    Return true labels and predicted Parkinson probabilities.

    passes=1:
        Deterministic inference.

    passes>1:
        Repeated training-mode forward passes, typically used
        for MC-dropout uncertainty analysis.

    Note:
        Training-mode inference may affect layers other than
        dropout in some model architectures. Interpret the
        resulting variability cautiously.
    """

    if passes < 1:
        raise ValueError(
            "passes must be at least 1"
        )

    labels = []
    probabilities = []

    for images, targets in dataset:

        batch_labels = (
            targets.numpy()
            .astype(int)
            .reshape(-1)
        )

        labels.extend(
            batch_labels.tolist()
        )

        if passes == 1:

            predictions = model(
                images,
                training=False,
            ).numpy().reshape(-1)

        else:

            draws = []

            for _ in range(passes):
                batch_predictions = model(
                    images,
                    training=True,
                ).numpy().reshape(-1)

                draws.append(
                    batch_predictions
                )

            predictions = np.mean(
                np.stack(draws, axis=0),
                axis=0,
            )

        probabilities.extend(
            predictions.tolist()
        )

    y_true = np.asarray(
        labels,
        dtype=int,
    )

    y_probability = np.asarray(
        probabilities,
        dtype=float,
    )

    if not np.all(np.isfinite(y_probability)):
        raise ValueError(
            "Predictions contain NaN or infinite values."
        )

    if np.any(
        (y_probability < 0)
        | (y_probability > 1)
    ):
        raise ValueError(
            "Predictions are not probabilities in [0, 1]. "
            "Check the model output activation."
        )

    return y_true, y_probability


# ============================================================
# 4. CALCULATE METRICS
# ============================================================

def metric_pack(y_true, probability, threshold=0.50):
    """
    Calculate binary classification metrics.

    Assumes:
        0 = healthy
        1 = parkinson
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    ).reshape(-1)

    probability = np.asarray(
        probability,
        dtype=float,
    ).reshape(-1)

    if len(y_true) != len(probability):
        raise ValueError(
            "Labels and probabilities have different lengths."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Cannot calculate metrics on an empty dataset."
        )

    predicted = (
        probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predicted,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    if len(np.unique(y_true)) == 2:
        auc = float(
            roc_auc_score(
                y_true,
                probability,
            )
        )
    else:
        auc = None

    return {
        "threshold": float(threshold),

        "sample_count": int(len(y_true)),

        "accuracy": float(
            accuracy_score(
                y_true,
                predicted,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predicted,
            )
        ),

        "sensitivity_recall": float(
            recall_score(
                y_true,
                predicted,
                zero_division=0,
            )
        ),

        "specificity": float(
            specificity
        ),

        "precision": float(
            precision_score(
                y_true,
                predicted,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                predicted,
                zero_division=0,
            )
        ),

        "roc_auc": auc,

        "brier_score": float(
            brier_score_loss(
                y_true,
                probability,
            )
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# 5. LOAD SPLIT AND PREDICT
# ============================================================

def evaluate_split(model, split_name):
    """
    Evaluate one named dataset split using the existing
    project data-loading pipeline.
    """

    print(f"\nEvaluating {split_name} split...")

    dataframe = get_split(split_name)

    if dataframe is None or len(dataframe) == 0:
        raise ValueError(
            f"The {split_name} split is empty."
        )

    dataset = make_dataset(
        dataframe,
        False,
    )

    y_true, probability = probs_with_mc(
        model,
        dataset,
        passes=1,
    )

    if len(y_true) != len(dataframe):
        raise ValueError(
            f"{split_name}: dataset rows ({len(dataframe)}) "
            f"do not match prediction count ({len(y_true)}). "
            "Check make_dataset batching and data filtering."
        )

    prediction_frame = dataframe.copy()

    prediction_frame["pd_probability"] = probability

    return y_true, probability, prediction_frame


# ============================================================
# 6. SELECT THRESHOLD USING VALIDATION ONLY
# ============================================================

def select_validation_threshold(
    y_true,
    probability,
):
    """
    Choose a threshold using validation data only.

    First preference:
        Sensitivity >= configured target.

    Among feasible candidates:
        Maximize specificity, then F1.

    If no candidate reaches the sensitivity target:
        Maximize F1 and report that the target was not met.
    """

    candidates = []

    for threshold in THRESHOLD_CANDIDATES:

        metrics = metric_pack(
            y_true,
            probability,
            threshold=float(threshold),
        )

        candidates.append(
            metrics
        )

    feasible = [
        item
        for item in candidates
        if (
            item["sensitivity_recall"]
            >= MIN_VALIDATION_SENSITIVITY
        )
    ]

    if feasible:

        chosen = max(
            feasible,
            key=lambda item: (
                item["specificity"],
                item["f1"],
            ),
        )

        target_met = True

    else:

        chosen = max(
            candidates,
            key=lambda item: item["f1"],
        )

        target_met = False

    return chosen, target_met


# ============================================================
# 7. SAVE PREDICTION REPORTS
# ============================================================

def save_prediction_files(
    prediction_frames,
):
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split_name, frame in prediction_frames.items():

        output_path = (
            REPORTS_DIR
            / f"{split_name}_predictions.csv"
        )

        frame.to_csv(
            output_path,
            index=False,
        )

        print(
            f"Saved {split_name} predictions: "
            f"{output_path}"
        )


# ============================================================
# 8. MAIN EVALUATION
# ============================================================

def main():

    print("=" * 65)
    print("PARKINSON'S SCREENING MODEL EVALUATION")
    print("=" * 65)

    print(f"Project root: {ROOT}")
    print(f"Model: {MODEL_PATH.name}")

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load the existing model.
    # No training takes place in this script.
    # --------------------------------------------------------

    model = load_trained_model()

    results = {}
    prediction_frames = {}
    labels_by_split = {}
    probabilities_by_split = {}

    # --------------------------------------------------------
    # Evaluate validation and test splits.
    # --------------------------------------------------------

    for split_name in [
        "validation",
        "test",
    ]:

        (
            y_true,
            probability,
            prediction_frame,
        ) = evaluate_split(
            model,
            split_name,
        )

        labels_by_split[split_name] = y_true

        probabilities_by_split[
            split_name
        ] = probability

        prediction_frames[
            split_name
        ] = prediction_frame

        results[split_name] = {
            "at_0.50": metric_pack(
                y_true,
                probability,
                threshold=DEFAULT_THRESHOLD,
            )
        }

    # --------------------------------------------------------
    # Choose threshold using validation data only.
    # --------------------------------------------------------

    validation_labels = (
        labels_by_split["validation"]
    )

    validation_probabilities = (
        probabilities_by_split["validation"]
    )

    chosen_metrics, target_met = (
        select_validation_threshold(
            validation_labels,
            validation_probabilities,
        )
    )

    selected_threshold = (
        chosen_metrics["threshold"]
    )

    results["validation"][
        "selected_threshold"
    ] = float(
        selected_threshold
    )

    results["validation"][
        "sensitivity_target"
    ] = MIN_VALIDATION_SENSITIVITY

    results["validation"][
        "sensitivity_target_met"
    ] = bool(
        target_met
    )

    results["validation"][
        "selected_threshold_metrics"
    ] = chosen_metrics

    # --------------------------------------------------------
    # Report test results at the validation-selected threshold.
    # The test set is not used to select the threshold.
    # --------------------------------------------------------

    test_labels = labels_by_split["test"]

    test_probabilities = (
        probabilities_by_split["test"]
    )

    results["test"][
        "validation_selected_threshold"
    ] = float(
        selected_threshold
    )

    results["test"][
        "validation_selected_threshold_metrics"
    ] = metric_pack(
        test_labels,
        test_probabilities,
        threshold=selected_threshold,
    )

    # --------------------------------------------------------
    # Optional MC-dropout prediction analysis.
    # --------------------------------------------------------

    try:

        config = load_config()

        mc_passes = int(
            config.get("mc_passes", 1)
        )

        if mc_passes < 1:
            mc_passes = 1

        test_dataframe = get_split("test")

        test_dataset = make_dataset(
            test_dataframe,
            False,
        )

        _, mc_mean_probability = probs_with_mc(
            model,
            test_dataset,
            passes=mc_passes,
        )

        test_prediction_frame = (
            prediction_frames["test"].copy()
        )

        test_prediction_frame[
            "mc_mean_pd_probability"
        ] = mc_mean_probability

        test_prediction_frame.to_csv(
            REPORTS_DIR
            / "test_predictions_mc.csv",
            index=False,
        )

        results["mc_dropout"] = {
            "passes": mc_passes,
            "note": (
                "Repeated training-mode predictions are an "
                "uncertainty proxy, not a calibrated clinical "
                "confidence guarantee."
            ),
        }

    except Exception as error:

        # Standard evaluation remains available even if the
        # optional MC-dropout analysis is not configured.
        results["mc_dropout"] = {
            "warning": str(error),
            "note": (
                "Optional MC-dropout analysis was skipped. "
                "Deterministic evaluation results were retained."
            ),
        }

        print(
            "\nWARNING: Optional MC-dropout analysis skipped:"
        )

        print(error)

    # --------------------------------------------------------
    # Save all reports.
    # --------------------------------------------------------

    save_prediction_files(
        prediction_frames
    )

    metrics_path = (
        REPORTS_DIR
        / "metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            allow_nan=False,
        )

    # --------------------------------------------------------
    # Print final metrics.
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)

    print(
        json.dumps(
            results,
            indent=2,
        )
    )

    print("\nSaved reports to:")
    print(REPORTS_DIR)

    print(f"\nMetrics JSON: {metrics_path}")

    print(
        "\nIMPORTANT: Threshold was selected using validation "
        "data only. Treat test results cautiously because this "
        "test set has already been examined in earlier experiments."
    )


if __name__ == "__main__":
    main()
