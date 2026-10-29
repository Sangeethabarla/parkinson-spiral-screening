# SpiralSense — Lavender Dark UI

A focused Flask + HTML/CSS/JavaScript UI for the existing Parkinson's spiral project.

## Add to the existing project
1. Copy `app.py`, `templates/`, and `static/` into the root of your existing `parkinson_spiral` project. Merge the `templates` and `static` folders if they already exist.
2. Activate your existing `.venv`.
3. Check Flask: `python -c "import flask; print('Flask available')"`; if missing, install with `pip install Flask`.
4. Run from project root: `python app.py`.
5. Open `http://127.0.0.1:5000`.

The backend imports `predict_image` from the existing `src.predict`, so it uses that file's current model path, threshold, temperature calibration, and preprocessing. This UI does not train or replace models. It intentionally does not show training details. The Grad-CAM area is a placeholder until connected to the project's existing Grad-CAM code. The app is for local research demos only; do not expose Flask's debug server publicly.
