
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import roc_auc_score

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

OUTPUT_DIR = ROOT / "reports" / "uncertainty_methods"

CSV_PATH = OUTPUT_DIR / "uncertainty_method_comparison.csv"
JSON_PATH = OUTPUT_DIR / "uncertainty_method_metrics.json"

THRESHOLD = 0.58
MC_DROPOUT_PASSES = 20
BATCH_SIZE = 8
EPSILON = 1e-7


# ============================================================
# 2. UNCERTAINTY MEASURES
# ============================================================

def probability_uncertainty(probability):
    """
    Maximum uncertainty at probability 0.5.
    Range: 0 to 1.
    """
    probability = np.clip(
        probability, EPSILON, 1.0 - EPSILON
    )

    return 1.0 - 2.0 * np.abs(probability - 0.5)


def binary_entropy(probability):
    """
    Normalized binary entropy.
    Range: 0 to 1.
    """
    probability = np.clip(
        probability, EPSILON, 1.0 - EPSILON
    )

    entropy = -(
        probability * np.log2(probability)
        + (1.0 - probability) * np.log2(1.0 - probability)
    )

    return entropy


def binary_entropy_per_sample(probabilities):
    """
    Calculate normalized entropy for an array of probabilities.
    """
    return binary_entropy(np.asarray(probabilities, dtype=float))


# ============================================================
# 3. ERROR-DETECTION AUC
# ============================================================

def error_detection_auc(error_labels, uncertainty_scores):
    """
    AUC for ranking incorrect predictions above correct ones.
    AUC > 0.5 indicates better-than-random ranking on this sample.
    """
    if len(np.unique(error_labels)) < 2:
        return None

    return float(
        roc_auc_score(error_labels, uncertainty_scores)
    )


# ============================================================
# 4. MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    test_df = load_test_dataframe()

    print("=" * 70)
    print("COMPARISON OF UNCERTAINTY METHODS")
    print("=" * 70)
    print(f"Test images: {len(test_df)}")
    print(f"MC Dropout passes per image: {MC_DROPOUT_PASSES}")
    print(f"Classification threshold: {THRESHOLD}")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    mc_probe = build_mc_dropout_probe(model)

    rows = []

    for start in range(0, len(test_df), BATCH_SIZE):
        batch_df = test_df.iloc[start:start + BATCH_SIZE]

        images = [
            load_image(path)
            for path in batch_df["resolved_path"].tolist()
        ]

        image_batch = tf.stack(images, axis=0)

        # Deterministic model probabilities.
        raw_probabilities = (
            model(image_batch, training=False)
            .numpy()
            .reshape(-1)
        )

        # Stochastic predictions from MC Dropout.
        passes = []

        for _ in range(MC_DROPOUT_PASSES):
            probabilities = (
                mc_probe(image_batch)
                .numpy()
                .reshape(-1)
            )
            passes.append(probabilities)

        # Shape: [MC passes, batch size].
        passes = np.asarray(passes, dtype=np.float64)

        mc_means = passes.mean(axis=0)
        mc_variances = passes.var(axis=0, ddof=0)
        mc_stds = passes.std(axis=0, ddof=0)

        # Predictive entropy: entropy of the averaged MC probability.
        predictive_entropies = binary_entropy_per_sample(
            mc_means
        )

        # Probability-based uncertainty from deterministic probability.
        prob_uncertainties = probability_uncertainty(
            raw_probabilities
        )

        # Classification decision based on the same fixed threshold.
        actual_labels = batch_df[
            "actual_label"
        ].to_numpy(dtype=int)

        for i, (_, row) in enumerate(batch_df.iterrows()):
            actual = int(actual_labels[i])
            raw_p = float(raw_probabilities[i])
            prediction = int(raw_p >= THRESHOLD)

            rows.append({
                "image_path": row["resolved_path"],
                "actual_label": actual,
                "raw_pd_probability": raw_p,
                "mc_dropout_mean": float(mc_means[i]),
                "mc_dropout_variance": float(mc_variances[i]),
                "mc_dropout_standard_deviation": float(mc_stds[i]),
                "probability_uncertainty": float(
                    prob_uncertainties[i]
                ),
                "predictive_entropy": float(
                    predictive_entropies[i]
                ),
                "predicted_label": prediction,
                "correct": bool(prediction == actual),
            })

        print(
            f"Processed {min(start + BATCH_SIZE, len(test_df))}"
            f"/{len(test_df)} images"
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(CSV_PATH, index=False)

    error_labels = (
        ~results_df["correct"].to_numpy(dtype=bool)
    ).astype(int)

    uncertainty_columns = {
        "MC Dropout variance": "mc_dropout_variance",
        "Probability uncertainty": "probability_uncertainty",
        "Predictive entropy": "predictive_entropy",
    }

    comparison = {}

    print("\n" + "=" * 70)
    print("UNCERTAINTY-BASED ERROR DETECTION")
    print("=" * 70)
    print(
        f"{'Method':<30}"
        f"{'Mean correct':>16}"
        f"{'Mean incorrect':>18}"
        f"{'Error AUC':>13}"
    )

    correct_mask = results_df["correct"].to_numpy(dtype=bool)

    for method_name, column in uncertainty_columns.items():
        scores = results_df[column].to_numpy(dtype=float)

        mean_correct = (
            float(scores[correct_mask].mean())
            if correct_mask.any() else None
        )

        mean_incorrect = (
            float(scores[~correct_mask].mean())
            if (~correct_mask).any() else None
        )

        auc = error_detection_auc(error_labels, scores)

        comparison[method_name] = {
            "mean_uncertainty_correct": mean_correct,
            "mean_uncertainty_incorrect": mean_incorrect,
            "error_detection_auc": auc,
        }

        correct_text = (
            f"{mean_correct:.6f}"
            if mean_correct is not None else "N/A"
        )
        incorrect_text = (
            f"{mean_incorrect:.6f}"
            if mean_incorrect is not None else "N/A"
        )
        auc_text = f"{auc:.4f}" if auc is not None else "N/A"

        print(
            f"{method_name:<30}"
            f"{correct_text:>16}"
            f"{incorrect_text:>18}"
            f"{auc_text:>13}"
        )

    # Correlation between uncertainty measures.
    correlation_columns = list(uncertainty_columns.values())
    correlation_matrix = results_df[
        correlation_columns
    ].corr(method="spearman")

    report = {
        "model": MODEL_PATH.name,
        "test_image_count": int(len(results_df)),
        "mc_dropout_passes": MC_DROPOUT_PASSES,
        "decision_threshold": THRESHOLD,
        "correct_predictions": int(correct_mask.sum()),
        "incorrect_predictions": int((~correct_mask).sum()),
        "uncertainty_methods": comparison,
        "spearman_correlation": {
            column: {
                other: float(
                    correlation_matrix.loc[column, other]
                )
                for other in correlation_columns
            }
            for column in correlation_columns
        },
        "definitions": {
            "mc_dropout_variance": (
                "Variance of stochastic MC Dropout probabilities."
            ),
            "probability_uncertainty": (
                "1 - 2 * abs(p - 0.5), using deterministic probability."
            ),
            "predictive_entropy": (
                "Normalized binary entropy of the MC-mean probability."
            ),
            "error_detection_auc": (
                "AUC where incorrect predictions are the positive class "
                "and higher uncertainty should rank them higher."
            ),
        },
        "limitations": (
            "Exploratory analysis on a small test set previously used "
            "during experimentation. AUC estimates are uncertain. "
            "The MC-mean entropy is predictive entropy, not a direct "
            "measure of epistemic uncertainty. Use validation data to "
            "choose a method and an untouched test set for final assessment."
        ),
    }

    with open(JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n" + "=" * 70)
    print("SPEARMAN CORRELATION BETWEEN UNCERTAINTY METHODS")
    print("=" * 70)
    print(correlation_matrix.round(3).to_string())

    print("\nSaved files:")
    print(f"CSV : {CSV_PATH}")
    print(f"JSON: {JSON_PATH}")
    print("\nComparison completed.")


if __name__ == "__main__":
    main()
