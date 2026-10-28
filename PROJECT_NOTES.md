# Source alignment and implementation notes

- Base paper: Saravanan et al., IEEE Access (2023), DOI 10.1109/ACCESS.2023.3291406. It describes spiral and wave drawing datasets, transfer-learning CNN comparisons, a VGG19-Inception hybrid (VGG19-INC), and LIME explanations.
- Student project Review-I architecture (page 12): spiral input -> grayscale/224x224/normalization -> augmentation -> patient-level 70/15/15 split -> EfficientNetV2-S -> Grad-CAM, uncertainty estimation, confidence calibration -> selective prediction.
- Abstract: baseline reproduction, Grad-CAM, uncertainty, patient-level splitting, and metrics including accuracy, precision, recall, specificity, F1, ROC-AUC and confusion matrix.

Therefore this code follows the student project's EfficientNetV2-S architecture. It does not claim to reproduce the paper's VGG19-INC result. The source paper's reported performance is not transferable without exact data, protocol, and replication.

## Caveats to review before final report
- The paper text mentions 102 subjects for one drawing dataset and 124 PD / 141 healthy for another; dataset identity and exact downloadable release must be confirmed from cited dataset sources before choosing the manifest.
- Dataset license and original participant IDs must be verified. Do not infer participant IDs from arbitrary filenames.
- The augmentation list in the architecture is illustrative; flip/noise/rotation may alter clinically meaningful orientation or drawing cues. Compare ablations and keep transforms conservative.
- Current implementation supplies a temperature-calibration script, but the prediction helper applies the saved temperature if that script has been run. Run `python -m src.calibrate` after evaluation to create it.
- `uncertainty_proxy` in predict.py is a simple heuristic and should not be described as a calibrated probability of error. MC-dropout variance is also only a proxy.
- Small datasets require confidence intervals and external validation. This is not clinically validated.
