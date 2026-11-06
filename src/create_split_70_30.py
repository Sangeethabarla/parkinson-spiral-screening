from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "dataset_inventory.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "dataset_split_70_30.csv"

RANDOM_STATE = 42

# Load inventory
df = pd.read_csv(INPUT_FILE)

required_columns = {
    "image_path", "label", "group_id", "source_dataset", "sha256"
}

missing = required_columns - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

df["group_id"] = df["group_id"].astype(str)
df["source_dataset"] = df["source_dataset"].astype(str)
df["label"] = df["label"].astype(str)

# Check duplicate image hashes
if df["sha256"].duplicated().any():
    raise ValueError("Duplicate image hashes detected. Resolve before splitting.")

# Preserve complete group IDs
df["participant_group"] = (
    df["source_dataset"] + "|" + df["group_id"]
)

# Check that each group has only one class
mixed = df.groupby("participant_group")["label"].nunique()

if (mixed > 1).any():
    raise ValueError(
        "Groups containing multiple labels detected:\n"
        + mixed[mixed > 1].to_string()
    )

# Create group-level table
group_table = (
    df.groupby(["source_dataset", "label", "participant_group"])
    .size()
    .reset_index(name="image_count")
)

assignments = []

# Split separately for each source and class
for (source, label), subset in group_table.groupby(
    ["source_dataset", "label"]
):
    groups = subset["participant_group"].tolist()

    if len(groups) < 2:
        raise ValueError(
            f"At least 2 groups are required for {source}/{label}. "
            f"Found: {len(groups)}"
        )

    train_groups, validation_groups = train_test_split(
        groups,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    assignments.extend(
        (group, "train") for group in train_groups
    )
    assignments.extend(
        (group, "validation") for group in validation_groups
    )

# Apply assignments
assignment_df = pd.DataFrame(
    assignments,
    columns=["participant_group", "split"],
)

result = df.merge(
    assignment_df,
    on="participant_group",
    how="left",
    validate="many_to_one",
)

if result["split"].isna().any():
    raise RuntimeError("Some images were not assigned.")

# Verify group separation
overlap = result.groupby("participant_group")["split"].nunique()
if (overlap > 1).any():
    raise RuntimeError("Group leakage detected.")

# Verify hash separation
hash_overlap = result.groupby("sha256")["split"].nunique()
if (hash_overlap > 1).any():
    raise RuntimeError("Duplicate hash leakage detected.")

# Save to a NEW file
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_FILE, index=False)

# Report
print("\n========== 70:30 GROUP-LEVEL SPLIT ==========")

print("\nIMAGE COUNTS BY SOURCE, SPLIT AND CLASS:")
print(
    result.groupby(["source_dataset", "split", "label"])
    .size().unstack(fill_value=0).to_string()
)

print("\nIMAGE COUNTS:")
print(result["split"].value_counts().to_string())

print("\nGROUP COUNTS:")
print(
    result.groupby("split")["participant_group"]
    .nunique().to_string()
)

print("\nPERCENTAGES:")
print(
    (result["split"].value_counts(normalize=True) * 100)
    .round(2).astype(str).add("%").to_string()
)

print("\nGROUP LEAKAGE: NONE")
print("DUPLICATE HASH LEAKAGE: NONE")
print(f"\nSaved to: {OUTPUT_FILE}")
