
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import roc_auc_score

from src.common import ROOT, load_image
from src.evaluate_uncertainty import build_mc_dropout_probe


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = (
    ROOT / "artifacts" / "parkinson_effnetv2s_classweighted.keras"
)

MANIFEST_PATH = ROOT / "data" / "processed" / "dataset_split.csv"

OUTPUT_DIR = ROOT / "reports" / "validation_uncertainty"

CSV_PATH = OUTPUT_DIR / "validation_uncertainty_results.csv"
JSON_PATH = OUTPUT_DIR / "validation_uncertainty_metrics.json"

THRESHOLD = 0.58
MC_DROPOUT_PASSES = 20
BATCH_SIZE = 8
EPSILON = 1e-7


# ============================================================
# 2. DATA HELPERS
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
    if pd.isna(value):
        raise ValueError("Empty class label found.")

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

    raise ValueError(f"Unrecognized class label: {value!r}")


def load_validation_dataframe():
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

    validation_df = df[
        df[split_col].astype(str).str.strip().str.lower().isin(
            ["val", "valid", "validation", "validate"]
        )
    ].copy()

    if validation_df.empty:
        raise ValueError(
            f"No validation rows found in column '{split_col}'. "
            f"Available values: {df[split_col].unique().tolist()}"
        )

    validation_df["resolved_path"] = validation_df[path_col].apply(
        lambda value: str(
            Path(value)
            if Path(value).is_absolute()
            else ROOT / Path(value)
        )
    )

    validation_df["actual_label"] = validation_df[
        label_col
    ].apply(normalize_label)

    missing = [
        path for path in validation_df["resolved_path"]
        if not Path(path).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing validation image(s): "
            + ", ".join(missing[:5])
        )

    return validation_df.reset_index(drop=True)


# ============================================================
# 3. UNCERTAINTY METHODS
# ============================================================

def probability_uncertainty(probability):
    probability = np.clip(
        probability, EPSILON, 1.0 - EPSILON
    )

    return 1.0 - 2.0 * np.abs(probability - 0.5)


def predictive_entropy(probability):
    probability = np.clip(
        probability, EPSILON, 1.0 - EPSILON
    )

    return -(
        probability * np.log2(probability)
        + (1.0 - probability) * np.log2(1.0 - probability)
    )


def error_detection_auc(error_labels, uncertainty_scores):
    if len(np.unique(error_labels)) < 2:
        return None

    return float(
        roc_auc_score(error_labels, uncertainty_scores)
    )


# ============================================================
# 4. MAIN EVALUATION
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    validation_df = load_validation_dataframe()

    print("=" * 70)
    print("VALIDATION-SET UNCERTAINTY COMPARISON")
    print("=" * 70)
    print(f"Validation images: {len(validation_df)}")
    print(f"MC passes per image: {MC_DROPOUT_PASSES}")
    print(f"Classification threshold: {THRESHOLD}")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    mc_probe = build_mc_dropout_probe(model)

    rows = []

    for start in range(0, len(validation_df), BATCH_SIZE):
        batch_df = validation_df.iloc[start:start + BATCH_SIZE]

        images = [
            load_image(path)
            for path in batch_df["resolved_path"].tolist()
        ]

        image_batch = tf.stack(images, axis=0)

        # Deterministic probabilities.
        raw_probabilities = (
            model(image_batch, training=False)
            .numpy()
            .reshape(-1)
        )

        # MC Dropout probabilities.
        passes = []

        for _ in range(MC_DROPOUT_PASSES):
            probabilities = (
                mc_probe(image_batch)
                .numpy()
                .reshape(-1)
            )
            passes.append(probabilities)

        passes = np.asarray(passes, dtype=np.float64)

        mc_means = passes.mean(axis=0)
        mc_variances = passes.var(axis=0, ddof=0)
        mc_stds = passes.std(axis=0, ddof=0)

        prob_uncertainties = probability_uncertainty(
            raw_probabilities
        )

        entropies = predictive_entropy(mc_means)

        actual_labels = batch_df[
            "actual_label"
        ].to_numpy(dtype=int)

        for i, (_, row) in enumerate(batch_df.iterrows()):
            actual = int(actual_labels[i])
            raw_p = float(raw_probabilities[i])
            predicted = int(raw_p >= THRESHOLD)

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
                "predictive_entropy": float(entropies[i]),
                "predicted_label": predicted,
                "correct": bool(predicted == actual),
            })

        print(
            f"Processed "
            f"{min(start + BATCH_SIZE, len(validation_df))}"
            f"/{len(validation_df)} images"
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(CSV_PATH, index=False)

    correct_mask = results_df["correct"].to_numpy(dtype=bool)
    error_labels = (~correct_mask).astype(int)

    methods = {
        "mc_dropout_variance": "mc_dropout_variance",
        "probability_uncertainty": "probability_uncertainty",
        "predictive_entropy": "predictive_entropy",
    }

    method_results = {}

    print("\n" + "=" * 70)
    print("VALIDATION ERROR-DETECTION RESULTS")
    print("=" * 70)
    print(
        f"{'Method':<30}"
        f"{'Mean correct':>16}"
        f"{'Mean incorrect':>18}"
        f"{'Error AUC':>13}"
    )

    for method_name, column in methods.items():
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

        method_results[method_name] = {
            "mean_uncertainty_correct": mean_correct,
            "mean_uncertainty_incorrect": mean_incorrect,
            "error_detection_auc": auc,
        }

        print(
            f"{method_name:<30}"
            f"{mean_correct if mean_correct is not None else 'N/A':>16}"
            f"{mean_incorrect if mean_incorrect is not None else 'N/A':>18}"
            f"{auc if auc is not None else 'N/A':>13}"
        )

    report = {
        "model": MODEL_PATH.name,
        "validation_image_count": int(len(results_df)),
        "mc_dropout_passes": MC_DROPOUT_PASSES,
        "decision_threshold": THRESHOLD,
        "correct_predictions": int(correct_mask.sum()),
        "incorrect_predictions": int((~correct_mask).sum()),
        "uncertainty_methods": method_results,
        "note": (
            "This validation set is used to compare uncertainty "
            "measures. Do not interpret small differences as "
            "conclusive evidence, particularly with a small sample. "
            "Choose a method using validation data, then assess the "
            "chosen method once on an independent test set."
        ),
    }

    with open(JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nSaved files:")
    print(f"CSV : {CSV_PATH}")
    print(f"JSON: {JSON_PATH}")
    print("\nValidation comparison completed.")


if __name__ == "__main__":
    main()
