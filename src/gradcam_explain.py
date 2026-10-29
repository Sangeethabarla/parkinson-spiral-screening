
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "parkinson_effnetv2s_classweighted.keras"
)

IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "drawings"
    / "spiral"
    / "testing"
    / "healthy"
    / "V03HE1.png"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "gradcam"

IMAGE_SIZE = (224, 224)
TARGET_LAYER_NAME = "top_conv"

# Existing standalone Grad-CAM default.
# Flask can pass the threshold and target class from predict.py.
DECISION_THRESHOLD = 0.55


# ============================================================
# 2. LOAD AND PREPROCESS IMAGE
# ============================================================

def load_image(image_path):
    """
    Match the existing Grad-CAM preprocessing:
    RGB decoding, 224x224 resizing, antialiasing,
    and float32 pixel values in the 0-255 range.
    """

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    original_image = Image.open(
        image_path
    ).convert("RGB")

    image_bytes = tf.io.read_file(
        str(image_path)
    )

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        antialias=True
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image_batch = tf.expand_dims(
        image,
        axis=0
    )

    return original_image, image_batch.numpy()


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

def load_trained_model():
    """
    Load the existing trained model without retraining.
    Returns model, backbone, and target convolution layer.
    """

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    print("Loading trained model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    try:
        backbone = model.get_layer(
            "efficientnetv2-s"
        )

        target_layer = backbone.get_layer(
            TARGET_LAYER_NAME
        )

    except ValueError as error:
        raise ValueError(
            "Expected EfficientNetV2-S backbone or target "
            f"layer '{TARGET_LAYER_NAME}' was not found."
        ) from error

    print(f"Model loaded: {MODEL_PATH.name}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Backbone: {backbone.name}")
    print(f"Grad-CAM layer: {target_layer.name}")
    print(
        f"Default Grad-CAM threshold: "
        f"{DECISION_THRESHOLD:.2f}"
    )

    return model, backbone, target_layer


# ============================================================
# 4. CREATE BACKBONE PROBE
# ============================================================

def create_backbone_probe(backbone, target_layer):
    """
    Return intermediate convolution maps and final backbone
    features through the existing Functional model graph.
    """

    return tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            target_layer.output,
            backbone.output
        ],
        name="gradcam_backbone_probe"
    )


# ============================================================
# 5. REUSE THE TRAINED CLASSIFICATION HEAD
# ============================================================

def classification_head(model, features):
    """
    Reuse the existing trained classification layers.
    No new trainable weights are created.
    """

    x = model.get_layer(
        "global_average_pooling"
    )(features)

    x = model.get_layer(
        "classifier_dropout"
    )(x, training=False)

    x = model.get_layer(
        "feature_dense"
    )(x)

    x = model.get_layer(
        "feature_dropout"
    )(x, training=False)

    probability = model.get_layer(
        "pd_probability"
    )(x)

    return probability


# ============================================================
# 6. GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    backbone_probe,
    image_batch,
    decision_threshold=DECISION_THRESHOLD,
    target_is_parkinson=None,
):
    """
    Generate a Grad-CAM heatmap for a single image.

    decision_threshold:
        Threshold used when selecting the class automatically.

    target_is_parkinson:
        Optional explicit target class. Flask can pass the class
        selected by the existing calibrated prediction pipeline.

    Returns:
        heatmap,
        uncalibrated PD probability from the Grad-CAM model path,
        selected Parkinson's class boolean.
    """

    image_tensor = tf.convert_to_tensor(
        image_batch,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        conv_outputs, features = backbone_probe(
            image_tensor,
            training=False
        )

        # Watch the convolution activations for Grad-CAM.
        tape.watch(conv_outputs)

        probability_tensor = classification_head(
            model,
            features
        )

        pd_probability = probability_tensor[0, 0]

        epsilon = tf.keras.backend.epsilon()

        clipped_probability = tf.clip_by_value(
            pd_probability,
            epsilon,
            1.0 - epsilon
        )

        logit = tf.math.log(
            clipped_probability
            / (1.0 - clipped_probability)
        )

        # Use the explicit target class when supplied by Flask.
        if target_is_parkinson is None:
            predicted_parkinson = (
                pd_probability >= decision_threshold
            )
        else:
            predicted_parkinson = tf.constant(
                bool(target_is_parkinson)
            )

        # Explain the selected class score.
        target_score = tf.where(
            predicted_parkinson,
            logit,
            -logit
        )

    gradients = tape.gradient(
        target_score,
        conv_outputs
    )

    if gradients is None:
        raise RuntimeError(
            "Gradients are None. Check the backbone and "
            "classification-head connections."
        )

    # Average gradients over spatial dimensions.
    channel_weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    feature_maps = conv_outputs[0]
    channel_weights = channel_weights[0]

    # Weighted sum of feature maps.
    cam = tf.reduce_sum(
        feature_maps
        * channel_weights[tf.newaxis, tf.newaxis, :],
        axis=-1
    )

    # Retain positive contributions.
    cam = tf.nn.relu(cam)

    # Normalize the heatmap to [0, 1].
    heatmap = tf.math.divide_no_nan(
        cam,
        tf.reduce_max(cam)
    )

    return (
        heatmap.numpy(),
        float(pd_probability.numpy()),
        bool(predicted_parkinson.numpy())
    )


# ============================================================
# 7. SAVE HEATMAP, OVERLAY AND JSON REPORT
# ============================================================

def save_gradcam(
    original_image,
    heatmap,
    pd_probability,
    predicted_parkinson,
    image_path=None,
    output_dir=None
):
    """
    Save the standalone heatmap, overlay and JSON report.

    Optional image_path and output_dir arguments allow Flask
    to save each uploaded image's explanation separately.
    """

    if image_path is None:
        image_path = IMAGE_PATH

    if output_dir is None:
        output_dir = OUTPUT_DIR

    image_path = Path(image_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_name = image_path.stem

    heatmap_path = (
        output_dir / f"{image_name}_heatmap.png"
    )

    overlay_path = (
        output_dir / f"{image_name}_gradcam.png"
    )

    report_path = (
        output_dir / f"{image_name}_gradcam.json"
    )

    # Resize the heatmap to the original image dimensions.
    heatmap_image = Image.fromarray(
        np.uint8(
            np.clip(heatmap, 0.0, 1.0) * 255
        )
    )

    heatmap_image = heatmap_image.resize(
        original_image.size,
        Image.Resampling.BILINEAR
    )

    heatmap_array = (
        np.asarray(
            heatmap_image,
            dtype=np.float32
        ) / 255.0
    )

    # Save standalone heatmap.
    plt.figure(figsize=(6, 6))

    plt.imshow(
        heatmap_array,
        cmap="jet",
        vmin=0,
        vmax=1
    )

    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig(
        heatmap_path,
        bbox_inches="tight",
        pad_inches=0,
        dpi=200
    )

    plt.close()

    # Create overlay using the original image.
    original_array = (
        np.asarray(
            original_image,
            dtype=np.float32
        ) / 255.0
    )

    colored_heatmap = plt.get_cmap("jet")(
        heatmap_array
    )[..., :3]

    overlay = np.clip(
        0.60 * original_array
        + 0.40 * colored_heatmap,
        0.0,
        1.0
    )

    plt.figure(figsize=(7, 7))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Grad-CAM: Parkinson's Screening")
    plt.tight_layout()

    plt.savefig(
        overlay_path,
        bbox_inches="tight",
        dpi=200
    )

    plt.close()

    predicted_class = (
        "parkinson"
        if predicted_parkinson
        else "healthy"
    )

    report = {
        "image": str(image_path),
        "model": MODEL_PATH.name,
        "target_layer": TARGET_LAYER_NAME,
        "predicted_class": predicted_class,
        "parkinson_probability": float(pd_probability),
        "healthy_probability": float(1.0 - pd_probability),
        "decision_threshold": float(
            DECISION_THRESHOLD
        ),
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
        "note": (
            "Research visualization only; "
            "not a clinical diagnosis."
        )
    }

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=4
        )

    return (
        heatmap_path,
        overlay_path,
        report_path,
        report
    )


# ============================================================
# 8. MAIN EXECUTION
# ============================================================

def main():

    model, backbone, target_layer = load_trained_model()

    backbone_probe = create_backbone_probe(
        backbone,
        target_layer
    )

    print("\nLoading image...")

    original_image, image_batch = load_image(
        IMAGE_PATH
    )

    print(f"Image: {IMAGE_PATH.name}")
    print(f"Image dimensions: {original_image.size}")

    print("\nGenerating Grad-CAM...")

    (
        heatmap,
        pd_probability,
        predicted_parkinson
    ) = generate_gradcam(
        model,
        backbone_probe,
        image_batch
    )

    (
        heatmap_path,
        overlay_path,
        report_path,
        report
    ) = save_gradcam(
        original_image,
        heatmap,
        pd_probability,
        predicted_parkinson,
        image_path=IMAGE_PATH,
        output_dir=OUTPUT_DIR
    )

    print("\n" + "=" * 60)
    print("GRAD-CAM COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Predicted class       : {report['predicted_class']}")
    print(f"Parkinson probability : {pd_probability:.4f}")
    print(f"Healthy probability   : {1.0 - pd_probability:.4f}")
    print(f"Decision threshold    : {DECISION_THRESHOLD:.2f}")

    print("\nSaved outputs:")
    print(f"Heatmap : {heatmap_path}")
    print(f"Overlay : {overlay_path}")
    print(f"Report  : {report_path}")

    print(
        "\nNote: Grad-CAM is an explanatory visualization, "
        "not a clinical diagnosis."
    )


if __name__ == "__main__":
    main()
