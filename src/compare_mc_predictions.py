
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

OUTPUT_DIR = ROOT / "reports" / "uncertainty"

COMPARISON_CSV = (
    OUTPUT_DIR / "deterministic_vs_mc_comparison.csv"
)

METRICS_JSON = (
    OUTPUT_DIR / "deterministic_vs_mc_metrics.json"
)

THRESHOLD = 0.58
MC_DROPOUT_PASSES = 20
BATCH_SIZE = 8


# ============================================================
# 2. CALCULATE CLASSIFICATION METRICS
# ============================================================

def calculate_metrics(y_true, probabilities, threshold):
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = (probabilities >= threshold).astype(int)

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

    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, probabilities))
    else:
        auc = None

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": auc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


# ============================================================
# 3. MAIN COMPARISON
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    test_df = load_test_dataframe()

    print("=" * 65)
    print("DETERMINISTIC VS MC DROPOUT MEAN COMPARISON")
    print("=" * 65)
    print(f"Test images: {len(test_df)}")
    print(f"MC passes per image: {MC_DROPOUT_PASSES}")
    print(f"Threshold: {THRESHOLD}")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    mc_probe = build_mc_dropout_probe(model)

    rows = []

    for start in range(0, len(test_df), BATCH_SIZE):
        batch_df = test_df.iloc[
            start:start + BATCH_SIZE
        ]

        images = [
            load_image(path)
            for path in batch_df["resolved_path"].tolist()
        ]

        image_batch = tf.stack(images, axis=0)

        # Method 1: deterministic prediction.
        deterministic_probs = (
            model(image_batch, training=False)
            .numpy()
            .reshape(-1)
        )

        # Method 2: mean of stochastic MC Dropout predictions.
        mc_passes = []

        for _ in range(MC_DROPOUT_PASSES):
            pass_probs = (
                mc_probe(image_batch)
                .numpy()
                .reshape(-1)
            )
            mc_passes.append(pass_probs)

        # Shape: [MC passes, batch size].
        mc_passes = np.asarray(mc_passes, dtype=np.float64)

        mc_means = mc_passes.mean(axis=0)
        mc_variances = mc_passes.var(axis=0, ddof=0)
        mc_stds = mc_passes.std(axis=0, ddof=0)

        for i, (_, row) in enumerate(batch_df.iterrows()):
            actual = int(row["actual_label"])

            deterministic_probability = float(
                deterministic_probs[i]
            )

            mc_mean = float(mc_means[i])

            deterministic_pred = int(
                deterministic_probability >= THRESHOLD
            )

            mc_mean_pred = int(mc_mean >= THRESHOLD)

            rows.append({
                "image_path": row["resolved_path"],
                "actual_label": actual,
                "actual_class": (
                    "PARKINSON" if actual == 1 else "HEALTHY"
                ),

                "deterministic_probability":
                    deterministic_probability,

                "deterministic_prediction":
                    "PARKINSON" if deterministic_pred == 1
                    else "HEALTHY",

                "deterministic_correct":
                    bool(deterministic_pred == actual),

                "mc_dropout_mean": mc_mean,

                "mc_mean_prediction":
                    "PARKINSON" if mc_mean_pred == 1
                    else "HEALTHY",

                "mc_mean_correct":
                    bool(mc_mean_pred == actual),

                "mc_dropout_variance": float(
                    mc_variances[i]
                ),

                "mc_dropout_standard_deviation": float(
                    mc_stds[i]
                ),

                "prediction_changed": bool(
                    deterministic_pred != mc_mean_pred
                ),

                "decision_threshold": THRESHOLD,
                "mc_dropout_passes": MC_DROPOUT_PASSES,
            })

        print(
            f"Processed "
            f"{min(start + BATCH_SIZE, len(test_df))}"
            f"/{len(test_df)} images"
        )

    results_df = pd.DataFrame(rows)

    y_true = results_df["actual_label"].to_numpy(dtype=int)

    deterministic_scores = results_df[
        "deterministic_probability"
    ].to_numpy(dtype=float)

    mc_mean_scores = results_df[
        "mc_dropout_mean"
    ].to_numpy(dtype=float)

    deterministic_metrics = calculate_metrics(
        y_true,
        deterministic_scores,
        THRESHOLD
    )

    mc_mean_metrics = calculate_metrics(
        y_true,
        mc_mean_scores,
        THRESHOLD
    )

    changed_count = int(
        results_df["prediction_changed"].sum()
    )

    results_df.to_csv(COMPARISON_CSV, index=False)

    comparison = {
        "model": MODEL_PATH.name,
        "test_image_count": int(len(results_df)),
        "threshold": THRESHOLD,
        "mc_dropout_passes": MC_DROPOUT_PASSES,

        "methods": {
            "deterministic": deterministic_metrics,
            "mc_dropout_mean": mc_mean_metrics,
        },

        "prediction_comparison": {
            "number_of_images_with_changed_class": changed_count,
            "fraction_with_changed_class": float(
                changed_count / len(results_df)
            ),
            "mean_absolute_probability_difference": float(
                np.mean(
                    np.abs(
                        deterministic_scores - mc_mean_scores
                    )
                )
            ),
        },

        "uncertainty_summary": {
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
        },

        "interpretation_note": (
            "Both methods use the same fixed threshold. "
            "Deterministic metrics use the ordinary model "
            "probability; MC-mean metrics use the average of "
            "the stochastic dropout predictions. The MC mean "
            "is not automatically better. Compare all metrics "
            "and consider the small test-set size."
        ),

        "warning": (
            "Research evaluation only. The test set has been "
            "used in prior experiments and is not an untouched "
            "independent final evaluation set. Results do not "
            "establish clinical readiness."
        ),
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)

    # --------------------------------------------------------
    # PRINT COMPARISON
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("METRIC COMPARISON")
    print("=" * 75)

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1_score",
        "roc_auc",
    ]

    print(
        f"{'Metric':<22}"
        f"{'Deterministic':>20}"
        f"{'MC Dropout Mean':>22}"
    )

    for name in metric_names:
        det_value = deterministic_metrics[name]
        mc_value = mc_mean_metrics[name]

        det_text = (
            f"{det_value:.4f}"
            if det_value is not None else "N/A"
        )

        mc_text = (
            f"{mc_value:.4f}"
            if mc_value is not None else "N/A"
        )

        print(f"{name:<22}{det_text:>20}{mc_text:>22}")

    print("\nPrediction comparison")
    print("-" * 45)
    print(
        "Images whose predicted class changed: "
        f"{changed_count}/{len(results_df)}"
    )
    print(
        "Mean absolute probability difference: "
        f"{comparison['prediction_comparison']['mean_absolute_probability_difference']:.6f}"
    )

    print("\nSaved reports:")
    print(f"CSV : {COMPARISON_CSV}")
    print(f"JSON: {METRICS_JSON}")
    print("\nComparison completed successfully.")


if __name__ == "__main__":
    main()
