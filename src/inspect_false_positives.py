
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_CSV = (
    PROJECT_ROOT / "reports" / "test_predictions_expanded.csv"
)
SPLIT_CSV = (
    PROJECT_ROOT / "data" / "processed" / "dataset_split.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "reports" / "false_positive_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Read predictions and test split metadata
predictions = pd.read_csv(PREDICTIONS_CSV)
split_data = pd.read_csv(SPLIT_CSV)

# Match filenames and source dataset to recover image paths.
# This is safer than matching filenames alone.
metadata = split_data[
    ["filename", "source_dataset", "image_path", "split"]
].copy()

results = predictions.merge(
    metadata,
    on=["filename", "source_dataset"],
    how="left",
    validate="many_to_one",
)

# 2. Select false positives:
# Actual healthy, predicted Parkinson
false_positives = results[
    (results["actual_label"] == "healthy")
    & (results["predicted_label"] == "parkinson")
].copy()

false_positives = false_positives[
    false_positives["split"] == "test"
]

print("\n========== FALSE POSITIVE ANALYSIS ==========")
print(f"False positives found: {len(false_positives)}")

print("\nImages incorrectly predicted as Parkinson's:")
print(
    false_positives[
        [
            "filename",
            "source_dataset",
            "parkinson_probability",
        ]
    ].to_string(index=False)
)

# Save detailed CSV
false_positives.to_csv(
    OUTPUT_DIR / "false_positives.csv",
    index=False,
)

# 3. Create a contact sheet of false-positive images
if false_positives.empty:
    print("No false positives found.")
else:
    columns = 4
    rows = (len(false_positives) + columns - 1) // columns

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, 4 * rows),
    )

    axes = list(axes.flat)

    for ax, (_, item) in zip(
        axes,
        false_positives.iterrows(),
    ):
        image_path = Path(item["image_path"])

        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path

        ax.axis("off")

        try:
            with Image.open(image_path) as image:
                ax.imshow(image.convert("RGB"))

            probability = float(
                item["parkinson_probability"]
            )

            ax.set_title(
                f'{item["filename"]}\n'
                f'Source: {item["source_dataset"]}\n'
                f'PD probability: {probability:.3f}',
                fontsize=9,
            )

        except Exception as error:
            ax.text(
                0.5,
                0.5,
                f"Could not load image\n{error}",
                ha="center",
                va="center",
            )

    for ax in axes[len(false_positives):]:
        ax.axis("off")

    fig.suptitle(
        "False Positives: Healthy Images Predicted as Parkinson's",
        fontsize=14,
    )
    fig.tight_layout()

    output_image = OUTPUT_DIR / "false_positives_contact_sheet.png"
    fig.savefig(output_image, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nDetailed CSV: {OUTPUT_DIR / 'false_positives.csv'}")
    print(f"Contact sheet: {output_image}")

# 4. Compare false positives by dataset source
print("\nFalse positives grouped by source:")
print(
    false_positives.groupby("source_dataset").size()
)

print("\nInvestigation files generated successfully.")
