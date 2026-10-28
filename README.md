# Parkinson's Disease Detection Using Spiral Hand Drawings

Fresh implementation scaffold based on the uploaded project abstract, Review-I architecture, and the cited IEEE Access base paper. This is a research prototype, not a diagnostic medical device.

## Architecture implemented
1. Spiral image input
2. Grayscale conversion, resize to 224x224, normalization
3. Training-only augmentation (rotation, flip, zoom/translation, noise)
4. Patient-level train/validation/test split (70/15/15)
5. EfficientNetV2-S transfer-learning classifier
6. Grad-CAM visual explanation
7. Monte Carlo dropout uncertainty estimate
8. Validation-only temperature calibration
9. Selective prediction: Healthy / Parkinson's / Abstain
10. Metrics and CSV outputs

## Important source/dataset note
The base paper reports two drawing datasets and discusses spiral and wave drawings. The project abstract and Review-I architecture specify spiral images and EfficientNetV2-S. This implementation follows the project architecture, not the base paper's VGG19-Inception VGG19-INC model. The base paper's reported 98.45% is not a performance promise for this implementation.

The dataset itself is not bundled. Do not redistribute dataset images unless the source license allows it. Verify the exact dataset source, license, label mapping, and participant identifiers before running.

## Expected manifest
Create `data/manifest.csv` with columns:
- `image_path`: path to image, absolute or relative to project root
- `label`: `0` for Healthy, `1` for Parkinson's
- `patient_id`: stable participant ID shared by all drawings from the same person
- `task`: use `spiral` for spiral drawings; wave images are excluded

Example (paths are illustrative only):
```csv
image_path,label,patient_id,task
path/to/healthy_001.png,0,H001,spiral
path/to/pd_001.png,1,P001,spiral
```

If the source dataset has multiple images per participant, every image from that participant must stay in one split. Do not use filenames as patient IDs unless the dataset documentation confirms that they encode participant identity.

## Setup (Windows PowerShell)
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
TensorFlow GPU support varies by Windows/Python/TensorFlow version. Google Colab is a practical option if local GPU setup is difficult.

## Run order
```powershell
python -m src.prepare_splits
python -m src.train
python -m src.evaluate
python -m src.explain --image "path\to\spiral.png"
python -m src.predict --image "path\to\spiral.png"
```

Training is not run automatically. First inspect `data/splits/split_manifest.csv` and confirm patient-level separation. The split script refuses to proceed when participant IDs or labels are missing.

## Outputs
- `data/splits/split_manifest.csv`
- `artifacts/best_model.keras`
- `artifacts/temperature.json`
- `reports/metrics.json`
- `reports/test_predictions.csv`
- `reports/gradcam_*.png`

## Evaluation safeguards
- Augmentation is applied only to training batches, after splitting.
- The test set is not used for threshold or temperature selection.
- Metrics include sensitivity, specificity, precision, recall, F1, ROC-AUC and confusion matrix.
- Results are exploratory and should include participant-level evaluation when several drawings belong to each participant. External validation is needed before clinical claims.
