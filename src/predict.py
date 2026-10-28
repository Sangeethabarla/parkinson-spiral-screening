
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.special import expit, logit

from src.common import ROOT, load_image


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = (
    ROOT / "artifacts" / "parkinson_effnetv2s_classweighted.keras"
)

TEMPERATURE_PATH = ROOT / "artifacts" / "temperature.json"

DEFAULT_THRESHOLD = 0.58
MC_DROPOUT_PASSES = 20


# ============================================================
# 2. TEMPERATURE CALIBRATION
# ============================================================

def load_temperature():
    """
    Load a fitted temperature if the calibration artifact exists.
    Otherwise, use temperature = 1.0 (no calibration change).
    """
    if not TEMPERATURE_PATH.exists():
        return 1.0

    with open(TEMPERATURE_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    temperature = float(data.get("temperature", 1.0))

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            f"Invalid temperature value: {temperature}"
        )

    return temperature


def calibrate_probability(probability, temperature):
    """
    Apply temperature scaling to a sigmoid probability.
    """
    probability = np.clip(
        probability, 1e-6, 1.0 - 1e-6
    )

    calibrated = expit(
        logit(probability) / temperature
    )

    return float(calibrated)


# ============================================================
# 3. MC DROPOUT PROBE
# ============================================================

def build_mc_dropout_probe(model):
    """
    Reuse the trained model's backbone and classifier weights,
    but create fresh dropout layers that are explicitly enabled
    during every stochastic forward pass.
    """

    class MCDropoutProbe(tf.keras.Model):

        def __init__(self, trained_model):
            super().__init__(
                name="classifier_mc_dropout_probe"
            )

            self.backbone = trained_model.get_layer(
                "efficientnetv2-s"
            )

            self.pool = trained_model.get_layer(
                "global_average_pooling"
            )

            self.drop1 = tf.keras.layers.Dropout(
                rate=trained_model.get_layer(
                    "classifier_dropout"
                ).rate
            )

            self.dense = trained_model.get_layer(
                "feature_dense"
            )

            self.drop2 = tf.keras.layers.Dropout(
                rate=trained_model.get_layer(
                    "feature_dropout"
                ).rate
            )

            self.output_layer = trained_model.get_layer(
                "pd_probability"
            )

        def call(self, inputs, training=False):
            # Keep the pretrained backbone in inference mode.
            x = self.backbone(inputs, training=False)

            x = self.pool(x)

            # Explicitly activate dropout for MC sampling.
            x = self.drop1(x, training=True)

            x = self.dense(x)

            x = self.drop2(x, training=True)

            return self.output_layer(x)

    return MCDropoutProbe(model)


# ============================================================
# 4. PREDICT A SINGLE IMAGE
# ============================================================

def predict_image(
    image_path,
    threshold=DEFAULT_THRESHOLD
):
    """
    Predict from one spiral drawing and report probability,
    MC Dropout variability, and a research-only classification.
    """

    image_path = Path(image_path)

    if not image_path.is_absolute():
        image_path = ROOT / image_path

    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    # Load trained model without requiring its compile configuration.
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    # Load and preprocess the spiral drawing.
    image = load_image(str(image_path))

    image_batch = tf.expand_dims(image, axis=0)

    # --------------------------------------------------------
    # A. Deterministic prediction
    # --------------------------------------------------------

    raw_probability = float(
        model(
            image_batch,
            training=False
        ).numpy().reshape(-1)[0]
    )

    # --------------------------------------------------------
    # B. MC Dropout stochastic predictions
    # --------------------------------------------------------

    mc_probe = build_mc_dropout_probe(model)

    mc_predictions = []

    for _ in range(MC_DROPOUT_PASSES):
        probability = float(
            mc_probe(image_batch)
            .numpy()
            .reshape(-1)[0]
        )

        mc_predictions.append(probability)

    mc_predictions = np.asarray(
        mc_predictions,
        dtype=np.float64
    )

    mc_mean = float(np.mean(mc_predictions))

    mc_variance = float(
        np.var(mc_predictions, ddof=0)
    )

    mc_std = float(
        np.std(mc_predictions, ddof=0)
    )

    # --------------------------------------------------------
    # C. Temperature scaling
    # --------------------------------------------------------

    temperature = load_temperature()

    calibrated_probability = calibrate_probability(
        raw_probability,
        temperature
    )

    # --------------------------------------------------------
    # D. Classification decision
    # --------------------------------------------------------

    prediction = (
        "PARKINSON'S"
        if calibrated_probability >= threshold
        else "HEALTHY"
    )

    # Simple probability-based uncertainty proxy.
    # This is not a complete uncertainty estimate.
    uncertainty_proxy = float(
        1.0 - 2.0 * abs(
            calibrated_probability - 0.5
        )
    )

    uncertainty_proxy = float(
        np.clip(uncertainty_proxy, 0.0, 1.0)
    )

    # --------------------------------------------------------
    # E. Return results
    # --------------------------------------------------------

    return {
        "image": str(image_path),
        "model": MODEL_PATH.name,

        "raw_pd_probability": raw_probability,

        "mc_dropout_mean": mc_mean,
        "mc_dropout_variance": mc_variance,
        "mc_dropout_standard_deviation": mc_std,
        "mc_dropout_passes": MC_DROPOUT_PASSES,

        "mc_dropout_predictions": [
            round(float(p), 6)
            for p in mc_predictions
        ],

        "temperature": temperature,
        "calibrated_pd_probability": calibrated_probability,

        "decision_threshold": threshold,
        "prediction": prediction,

        "uncertainty_proxy": uncertainty_proxy,

        "warning": (
            "Research prototype only. This prediction is not "
            "a medical diagnosis and must not be used alone "
            "for clinical decisions."
        )
    }


# ============================================================
# 5. COMMAND-LINE INTERFACE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parkinson's spiral drawing research prediction "
            "with MC Dropout uncertainty estimates."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the spiral drawing image."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Classification threshold; default is 0.58."
    )

    args = parser.parse_args()

    try:
        result = predict_image(
            image_path=args.image,
            threshold=args.threshold
        )

        print(
            json.dumps(
                result,
                indent=2
            )
        )

    except (FileNotFoundError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
