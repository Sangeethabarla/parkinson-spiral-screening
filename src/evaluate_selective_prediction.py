
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

from src.common import ROOT, load_image
from src.evaluate_uncertainty import (
    build_mc_dropout_probe,
    load_test_dataframe,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = (
    ROOT / "artifacts" / "parkinson_effnetv2s_classweighted.keras"
)

OUTPUT_DIR = ROOT / "reports" / "selective_prediction"

CSV_PATH = OUTPUT_DIR / "selective_prediction_results.csv"
JSON_PATH = OUTPUT_DIR / "selective_prediction_metrics.json"

THRESHOLD = 0.58
MC_DROPOUT_PASSES = 20
BATCH_SIZE = 8

# Coverage levels at which to evaluate retained predictions.
COVERAGE_LEVELS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


# ============================================================
# 2. METRIC HELPERS
# ============================================================

def safe_auc(y_true, probabilities):
    if len(np.unique(y_true)) < 2:
        return None

    return float(roc_auc_score(y_true, probabilities))


def classification_metrics(y_true, probabilities, threshold):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    y_pred = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    sensitivity = (
        float(tp / (tp + fn))
        if tp + fn > 0 else None
    )

    specificity = (
        float(tn / (tn + fp))
        if tn + fp > 0 else None
    )

    return {
        "count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": safe_auc(y_true, probabilities),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


# ============================================================
# 3. UNCERTAINTY-ERROR ASSOCIATION
# ============================================================

def evaluate_error_association(df):
    """
    Compare uncertainty scores for correct and incorrect
    predictions. This describes association, not causation.
    """

    correct = df["correct"].to_numpy(dtype=bool)
    variance = df["mc_dropout_variance"].to_numpy(dtype=float)
    std = df["mc_dropout_standard_deviation"].to_numpy(dtype=float)

    correct_count = int(correct.sum())
    error_count = int((~correct).sum())

    result = {
        "correct_prediction_count": correct_count,
        "incorrect_prediction_count": error_count,
        "mean_variance_correct": (
            float(variance[correct].mean())
            if correct_count else None
        ),
        "mean_variance_incorrect": (
            float(variance[~correct].mean())
            if error_count else None
        ),
        "mean_std_correct": (
            float(std[correct].mean())
            if correct_count else None
        ),
        "mean_std_incorrect": (
            float(std[~correct].mean())
            if error_count else None
        ),
        "interpretation": (
            "Compare uncertainty for correct and incorrect cases. "
            "A difference in averages alone does not establish that "
            "uncertainty reliably detects errors."
        ),
    }

    # With both correct and incorrect samples, calculate the
    # rank-based ability of variance to distinguish errors.
    error_labels = (~correct).astype(int)

    if len(np.unique(error_labels)) == 2:
        result["variance_auc_for_error_detection"] = float(
            roc_auc_score(error_labels, variance)
        )
    else:
        result["variance_auc_for_error_detection"] = None

    return result


# ============================================================
# 4. SELECTIVE PREDICTION
# ============================================================

def evaluate_coverage_levels(df):
    """
    Retain the lowest-uncertainty samples at each coverage.
    The ranking uses MC Dropout variance.
    """

    df = df.sort_values(
        "mc_dropout_variance",
        ascending=True
    ).reset_index(drop=True)

    total = len(df)
    rows = []

    for requested_coverage in COVERAGE_LEVELS:
        retained_count = max(
            1,
            int(np.ceil(requested_coverage * total))
        )

        retained = df.iloc[:retained_count]

        y_true = retained["actual_label"].to_numpy(dtype=int)
        probabilities = retained[
            "raw_pd_probability"
        ].to_numpy(dtype=float)

        metrics = classification_metrics(
            y_true,
            probabilities,
            THRESHOLD
        )

        rows.append({
            "requested_coverage": requested_coverage,
            "actual_coverage": retained_count / total,
            "retained_count": retained_count,
            "rejected_count": total - retained_count,
            "selective_accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "precision": metrics["precision"],
            "f1_score": metrics["f1_score"],
            "roc_auc": metrics["roc_auc"],
            "tn": metrics["confusion_matrix"]["tn"],
            "fp": metrics["confusion_matrix"]["fp"],
            "fn": metrics["confusion_matrix"]["fn"],
            "tp": metrics["confusion_matrix"]["tp"],
        })

    return rows


# ============================================================
# 5. MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    test_df = load_test_dataframe()

    print("=" * 68)
    print("UNCERTAINTY VALIDATION AND SELECTIVE PREDICTION")
    print("=" * 68)
    print(f"Test images: {len(test_df)}")
    print(f"MC passes per image: {MC_DROPOUT_PASSES}")
    print(f"Classification threshold: {THRESHOLD}")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    mc_probe = build_mc_dropout_probe(model)

    result_rows = []

    for start in range(0, len(test_df), BATCH_SIZE):
        batch_df = test_df.iloc[start:start + BATCH_SIZE]

        images = [
            load_image(path)
            for path in batch_df["resolved_path"].tolist()
        ]

        image_batch = tf.stack(images, axis=0)

        # Ordinary deterministic probability.
        raw_probabilities = (
            model(image_batch, training=False)
            .numpy()
            .reshape(-1)
        )

        # MC Dropout stochastic predictions.
        passes = []

        for _ in range(MC_DROPOUT_PASSES):
            probabilities = (
                mc_probe(image_batch)
                .numpy()
                .reshape(-1)
            )
            passes.append(probabilities)

        passes = np.asarray(passes, dtype=np.float64)

        mc_mean = passes.mean(axis=0)
        mc_variance = passes.var(axis=0, ddof=0)
        mc_std = passes.std(axis=0, ddof=0)

        actual_labels = batch_df[
            "actual_label"
        ].to_numpy(dtype=int)

        for i, (_, row) in enumerate(batch_df.iterrows()):
            raw_probability = float(raw_probabilities[i])
            prediction = int(raw_probability >= THRESHOLD)
            actual = int(actual_labels[i])

            result_rows.append({
                "image_path": row["resolved_path"],
                "actual_label": actual,
                "actual_class": (
                    "PARKINSON" if actual == 1 else "HEALTHY"
                ),
                "raw_pd_probability": raw_probability,
                "mc_dropout_mean": float(mc_mean[i]),
                "mc_dropout_variance": float(mc_variance[i]),
                "mc_dropout_standard_deviation": float(mc_std[i]),
                "prediction": prediction,
                "predicted_class": (
                    "PARKINSON" if prediction == 1 else "HEALTHY"
                ),
                "correct": bool(prediction == actual),
                "decision_threshold": THRESHOLD,
                "mc_dropout_passes": MC_DROPOUT_PASSES,
            })

        print(
            f"Processed {min(start + BATCH_SIZE, len(test_df))}"
            f"/{len(test_df)} images"
        )

    results_df = pd.DataFrame(result_rows)

    # Save per-image results.
    results_df.to_csv(CSV_PATH, index=False)

    y_true = results_df["actual_label"].to_numpy(dtype=int)
    raw_probabilities = results_df[
        "raw_pd_probability"
    ].to_numpy(dtype=float)

    overall_metrics = classification_metrics(
        y_true,
        raw_probabilities,
        THRESHOLD
    )

    error_association = evaluate_error_association(results_df)

    coverage_results = evaluate_coverage_levels(results_df)

    coverage_df = pd.DataFrame(coverage_results)
    coverage_csv = OUTPUT_DIR / "coverage_accuracy_curve.csv"
    coverage_df.to_csv(coverage_csv, index=False)

    report = {
        "model": MODEL_PATH.name,
        "test_image_count": int(len(results_df)),
        "mc_dropout_passes": MC_DROPOUT_PASSES,
        "decision_threshold": THRESHOLD,
        "overall_classification_metrics": overall_metrics,
        "uncertainty_error_association": error_association,
        "selective_prediction": coverage_results,
        "selection_rule": (
            "At each requested coverage, retain the images with "
            "the lowest MC Dropout variance."
        ),
        "important_limitation": (
            "This is exploratory analysis on a small test set "
            "already used in previous experiments. Coverage-based "
            "selection and resulting metrics are not an independent "
            "clinical validation. Use validation data to choose any "
            "operational uncertainty cutoff, and assess it on a "
            "separate untouched test set."
        ),
    }

    with open(JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 68)
    print("OVERALL CLASSIFICATION")
    print("=" * 68)
    print(f"Accuracy       : {overall_metrics['accuracy']:.4f}")
    print(
        f"Sensitivity    : "
        f"{overall_metrics['sensitivity']}"
    )
    print(
        f"Specificity    : "
        f"{overall_metrics['specificity']}"
    )
    print(f"F1-score       : {overall_metrics['f1_score']:.4f}")

    print("\n" + "=" * 68)
    print("UNCERTAINTY AND ERROR ASSOCIATION")
    print("=" * 68)
    print(
        "Correct predictions: "
        f"{error_association['correct_prediction_count']}"
    )
    print(
        "Incorrect predictions: "
        f"{error_association['incorrect_prediction_count']}"
    )
    print(
        "Mean variance for correct predictions: "
        f"{error_association['mean_variance_correct']}"
    )
    print(
        "Mean variance for incorrect predictions: "
        f"{error_association['mean_variance_incorrect']}"
    )
    print(
        "Variance AUC for error detection: "
        f"{error_association['variance_auc_for_error_detection']}"
    )

    print("\n" + "=" * 68)
    print("SELECTIVE PREDICTION: LOWEST-UNCERTAINTY SAMPLES")
    print("=" * 68)

    print(
        f"{'Coverage':>10} {'Retained':>10} "
        f"{'Accuracy':>10} {'Sensitivity':>12} {'Specificity':>12}"
    )

    for row in coverage_results:
        print(
            f"{row['actual_coverage']:>9.1%} "
            f"{row['retained_count']:>10} "
            f"{row['selective_accuracy']:>10.4f} "
            f"{str(row['sensitivity']):>12} "
            f"{str(row['specificity']):>12}"
        )

    print("\nSaved files:")
    print(f"Per-image CSV : {CSV_PATH}")
    print(f"Coverage CSV  : {coverage_csv}")
    print(f"Metrics JSON  : {JSON_PATH}")
    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
