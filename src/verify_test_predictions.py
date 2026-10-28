
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "data" / "processed" / "dataset_split.csv"
PREDICTIONS = (
    ROOT / "reports" / "uncertainty"
    / "mc_dropout_test_results.csv"
)

manifest = pd.read_csv(MANIFEST)
predictions = pd.read_csv(PREDICTIONS)

# Normalize paths so Windows absolute paths and manifest-relative
# paths can be compared.
def normalize_path(value):
    path = Path(str(value).replace("\\", "/"))
    parts = path.parts

    # Keep path from the project data directory onwards.
    parts_lower = [part.lower() for part in parts]
    if "data" in parts_lower:
        index = parts_lower.index("data")
        return "/".join(parts[index:]).lower()

    return str(path).lower()

manifest["normalized_path"] = manifest["image_path"].map(normalize_path)
predictions["normalized_path"] = predictions["image_path"].map(
    normalize_path
)

# The manifest should contain one split assignment per image path.
if manifest["normalized_path"].duplicated().any():
    duplicates = manifest[
        manifest["normalized_path"].duplicated(keep=False)
    ]
    print("WARNING: Duplicate manifest paths found:")
    print(duplicates[["image_path", "split"]].to_string(index=False))

merged = predictions.merge(
    manifest[["normalized_path", "split", "label", "group_id"]],
    on="normalized_path",
    how="left",
    suffixes=("_prediction", "_manifest"),
    indicator=True,
)

print("\nManifest split membership of saved test predictions:")
print(
    merged["split"].fillna("NOT_IN_MANIFEST")
    .value_counts()
    .to_string()
)

print("\nRows not assigned to the manifest test split:")
not_test = merged[
    (merged["_merge"] != "both") |
    (merged["split"].astype(str).str.lower() != "test")
]
print(
    not_test[
        ["image_path", "split", "actual_label", "predicted_label"]
    ].to_string(index=False)
)

print("\nPrediction count:", len(predictions))
print("Manifest matches:", int((merged["_merge"] == "both").sum()))
print("Manifest test rows:", int(
    (merged["split"].astype(str).str.lower() == "test").sum()
))

# Check label agreement for matched images.
matched = merged[merged["_merge"] == "both"].copy()
matched["manifest_label"] = (
    matched["label"].astype(str).str.strip().str.lower()
)
matched["expected_label"] = matched["manifest_label"].map({
    "healthy": 0,
    "parkinson": 1,
    "patient": 1,
})

label_mismatch = matched[
    matched["expected_label"].notna() &
    (matched["actual_label"].astype(int) !=
     matched["expected_label"].astype(int))
]

print("\nLabel mismatches:", len(label_mismatch))
if not label_mismatch.empty:
    print(
        label_mismatch[
            ["image_path", "actual_label", "label", "split"]
        ].to_string(index=False)
    )
