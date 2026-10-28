
from pathlib import Path
import csv
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

import gradcam_explain as gradcam


# =========================================================
# 1. CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "drawings"
    / "spiral"
    / "testing"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "gradcam_batch"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_SIZE = (224, 224)
IMAGES_PER_CLASS = 2

# Must match gradcam_explain.py
DECISION_THRESHOLD = 0.55

CLASS_FOLDERS = {
    "healthy": "healthy",
    "parkinson": "parkinson",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}


# =========================================================
# 2. LOAD IMAGE
# =========================================================

def load_batch_image(image_path):
    """
    Uses the same preprocessing as the training pipeline:
    RGB, 224 x 224, antialias=True, pixel range 0-255.
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
        expand_animations=False,
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        antialias=True,
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    image_batch = tf.expand_dims(
        image,
        axis=0,
    )

    return original_image, image_batch.numpy()


# =========================================================
# 3. FIND SAMPLE IMAGES
# =========================================================

def find_images():
    selected_images = []

    for class_name, folder_name in CLASS_FOLDERS.items():

        class_dir = DATASET_ROOT / folder_name

        if not class_dir.is_dir():
            print(
                f"WARNING: Folder not found: {class_dir}"
            )
            continue

        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        )

        if not image_paths:
            print(
                f"WARNING: No images found in {class_dir}"
            )
            continue

        selected_paths = image_paths[
            :IMAGES_PER_CLASS
        ]

        print(
            f"\n{class_name}: found {len(image_paths)} images; "
            f"selecting {len(selected_paths)}"
        )

        for image_path in selected_paths:
            selected_images.append(
                (class_name, image_path)
            )

    return selected_images


# =========================================================
# 4. SAVE GRAD-CAM OUTPUTS
# =========================================================

def save_visualizations(
    original_image,
    heatmap,
    image_path,
    output_dir,
):
    """
    Saves:
    1. Standalone heatmap
    2. Heatmap overlay
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image_path = Path(image_path)
    image_name = image_path.stem

    heatmap_path = (
        output_dir
        / f"{image_name}_heatmap.png"
    )

    overlay_path = (
        output_dir
        / f"{image_name}_gradcam.png"
    )

    # Normalize and resize the heatmap.
    heatmap_uint8 = np.uint8(
        np.clip(heatmap, 0.0, 1.0) * 255
    )

    heatmap_image = Image.fromarray(
        heatmap_uint8
    ).resize(
        original_image.size,
        Image.Resampling.BILINEAR,
    )

    heatmap_array = (
        np.asarray(
            heatmap_image,
            dtype=np.float32,
        ) / 255.0
    )

    # Save standalone heatmap.
    plt.figure(figsize=(6, 6))

    plt.imshow(
        heatmap_array,
        cmap="jet",
        vmin=0,
        vmax=1,
    )

    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig(
        heatmap_path,
        bbox_inches="tight",
        pad_inches=0,
        dpi=200,
    )

    plt.close()

    # Create overlay.
    original_array = (
        np.asarray(
            original_image,
            dtype=np.float32,
        ) / 255.0
    )

    colored_heatmap = plt.get_cmap(
        "jet"
    )(heatmap_array)[..., :3]

    overlay = np.clip(
        0.60 * original_array
        + 0.40 * colored_heatmap,
        0.0,
        1.0,
    )

    plt.figure(figsize=(7, 7))

    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Grad-CAM: Parkinson's Screening")
    plt.tight_layout()

    plt.savefig(
        overlay_path,
        bbox_inches="tight",
        dpi=200,
    )

    plt.close()

    return heatmap_path, overlay_path


# =========================================================
# 5. PROCESS ONE IMAGE
# =========================================================

def process_image(
    model,
    backbone_probe,
    true_class,
    image_path,
):
    print("\n" + "-" * 60)
    print(f"True class: {true_class}")
    print(f"Image: {image_path.name}")

    original_image, image_batch = load_batch_image(
        image_path
    )

    (
        heatmap,
        pd_probability,
        predicted_parkinson,
    ) = gradcam.generate_gradcam(
        model,
        backbone_probe,
        image_batch,
    )

    # Use the prediction returned by Grad-CAM so the
    # predicted label matches the class being explained.
    predicted_class = (
        "parkinson"
        if predicted_parkinson
        else "healthy"
    )

    sample_name = (
        f"{true_class}_{image_path.stem}"
    )

    sample_dir = OUTPUT_DIR / sample_name

    sample_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    heatmap_path, overlay_path = save_visualizations(
        original_image=original_image,
        heatmap=heatmap,
        image_path=image_path,
        output_dir=sample_dir,
    )

    report_path = (
        sample_dir
        / f"{image_path.stem}_gradcam.json"
    )

    prediction_correct = (
        predicted_class == true_class
    )

    result = {
        "image": str(image_path),
        "true_class": true_class,
        "predicted_class": predicted_class,
        "parkinson_probability": float(
            pd_probability
        ),
        "healthy_probability": float(
            1.0 - pd_probability
        ),
        "decision_threshold": DECISION_THRESHOLD,
        "prediction_correct": prediction_correct,
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
        "report_path": str(report_path),
        "note": (
            "Research visualization only; "
            "not a clinical diagnosis."
        ),
    }

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=4,
        )

    print(f"Predicted class: {predicted_class}")
    print(
        f"Parkinson probability: "
        f"{pd_probability:.4f}"
    )
    print(
        f"Healthy probability: "
        f"{1.0 - pd_probability:.4f}"
    )
    print(
        f"Correct on this image: "
        f"{prediction_correct}"
    )
    print(f"Heatmap: {heatmap_path}")
    print(f"Overlay: {overlay_path}")

    return result


# =========================================================
# 6. SAVE AGGREGATE REPORTS
# =========================================================

def save_aggregate_reports(results):
    csv_path = (
        OUTPUT_DIR
        / "gradcam_batch_results.csv"
    )

    json_path = (
        OUTPUT_DIR
        / "gradcam_batch_results.json"
    )

    fieldnames = [
        "image",
        "true_class",
        "predicted_class",
        "parkinson_probability",
        "healthy_probability",
        "decision_threshold",
        "prediction_correct",
        "heatmap_path",
        "overlay_path",
        "report_path",
        "note",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    return csv_path, json_path


# =========================================================
# 7. MAIN
# =========================================================

def main():

    print("=" * 60)
    print("BATCH GRAD-CAM ANALYSIS")
    print("=" * 60)

    print(f"Dataset: {DATASET_ROOT}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Threshold: {DECISION_THRESHOLD}")

    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(
            f"Testing dataset not found: {DATASET_ROOT}"
        )

    # Prevent accidental threshold mismatch.
    if (
        gradcam.DECISION_THRESHOLD
        != DECISION_THRESHOLD
    ):
        raise ValueError(
            "Threshold mismatch. Set DECISION_THRESHOLD "
            "to 0.55 in gradcam_explain.py too."
        )

    # Load the existing trained model once.
    (
        model,
        backbone,
        target_layer,
    ) = gradcam.load_trained_model()

    backbone_probe = (
        gradcam.create_backbone_probe(
            backbone,
            target_layer,
        )
    )

    selected_images = find_images()

    if not selected_images:
        raise FileNotFoundError(
            "No images found in testing folders."
        )

    print(
        f"\nTotal selected images: "
        f"{len(selected_images)}"
    )

    results = []
    failures = []

    for index, (true_class, image_path) in enumerate(
        selected_images,
        start=1,
    ):

        print(
            f"\nProcessing image "
            f"{index}/{len(selected_images)}"
        )

        try:
            result = process_image(
                model=model,
                backbone_probe=backbone_probe,
                true_class=true_class,
                image_path=image_path,
            )

            results.append(result)

        except Exception as error:
            failures.append({
                "image": str(image_path),
                "error": str(error),
            })

            print(
                f"ERROR processing "
                f"{image_path.name}: {error}"
            )

    # Save reports only when at least one image succeeded.
    if results:
        csv_path, json_path = (
            save_aggregate_reports(results)
        )

    else:
        csv_path = None
        json_path = None

    correct_count = sum(
        result["prediction_correct"]
        for result in results
    )

    print("\n" + "=" * 60)
    print("BATCH GRAD-CAM FINISHED")
    print("=" * 60)

    print(f"Successfully processed: {len(results)}")
    print(f"Failed: {len(failures)}")

    if results:
        print(
            f"Correct among selected images: "
            f"{correct_count}/{len(results)}"
        )

        print(
            "This selected sample is for inspection only; "
            "it is not a reliable overall accuracy estimate."
        )

    if csv_path:
        print(f"\nCSV summary: {csv_path}")

    if json_path:
        print(f"JSON summary: {json_path}")

    if failures:
        failure_path = (
            OUTPUT_DIR
            / "gradcam_batch_failures.json"
        )

        with open(
            failure_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                failures,
                file,
                indent=4,
            )

        print(f"Failure details: {failure_path}")

    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
