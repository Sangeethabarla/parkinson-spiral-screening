import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path
import tempfile

from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename

from src.predict import predict_image
from src.gradcam_explain import (
    load_trained_model,
    load_image as load_gradcam_image,
    create_backbone_probe,
    generate_gradcam,
    save_gradcam,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
GRADCAM_DIR = BASE_DIR / "static" / "gradcam"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({
            "error": "Choose a spiral drawing image first."
        }), 400

    uploaded = request.files["image"]

    if not uploaded or not uploaded.filename:
        return jsonify({
            "error": "Choose a spiral drawing image first."
        }), 400

    if not allowed_file(uploaded.filename):
        return jsonify({
            "error": "Supported formats: PNG, JPG, JPEG, BMP, WEBP."
        }), 400

    temp_path = None

    try:
        safe_name = secure_filename(uploaded.filename) or "spiral.png"
        suffix = Path(safe_name).suffix.lower() or ".png"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
            dir=UPLOAD_DIR,
        ) as temp:
            temp_path = Path(temp.name)
            uploaded.save(temp_path)

        # Existing prediction pipeline: no retraining or model changes.
        result = predict_image(str(temp_path.resolve()))

        response_data = {
            "prediction": result.get("prediction"),
            "raw_pd_probability": result.get("raw_pd_probability"),
            "calibrated_pd_probability": result.get(
                "calibrated_pd_probability"
            ),
            "mc_dropout_mean": result.get("mc_dropout_mean"),
            "mc_dropout_variance": result.get("mc_dropout_variance"),
            "mc_dropout_standard_deviation": result.get(
                "mc_dropout_standard_deviation"
            ),
            "mc_dropout_passes": result.get("mc_dropout_passes"),
            "uncertainty_proxy": result.get("uncertainty_proxy"),
            "decision_threshold": result.get("decision_threshold"),
            "temperature": result.get("temperature"),
            "model": result.get("model"),
            "warning": result.get("warning"),
            "gradcam_url": None,
            "gradcam_error": None,
        }

        # Generate the explanation separately so prediction results
        # remain available even if Grad-CAM fails.
        try:
            model, backbone, target_layer = load_trained_model()

            backbone_probe = create_backbone_probe(
                backbone,
                target_layer,
            )

            original_image, image_batch = load_gradcam_image(
                str(temp_path.resolve())
            )

            target_is_parkinson = (
                str(result.get("prediction", ""))
                .upper()
                .startswith("PARKINSON")
            )

            heatmap, _, _ = generate_gradcam(
                model,
                backbone_probe,
                image_batch,
                decision_threshold=float(
                    result.get("decision_threshold", 0.58)
                ),
                target_is_parkinson=target_is_parkinson,
            )

            _, overlay_path, _, _ = save_gradcam(
                original_image,
                heatmap,
                float(result["raw_pd_probability"]),
                target_is_parkinson,
                image_path=temp_path,
                output_dir=GRADCAM_DIR,
            )

            response_data["gradcam_url"] = url_for(
                "static",
                filename=f"gradcam/{overlay_path.name}",
            )

        except Exception as gradcam_exc:
            app.logger.exception("Grad-CAM generation failed")
            response_data["gradcam_error"] = str(gradcam_exc)

        return jsonify(response_data)

    except Exception as exc:
        app.logger.exception("Prediction request failed")
        return jsonify({
            "error": f"Prediction failed: {exc}"
        }), 500

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    app.run(debug=True)
