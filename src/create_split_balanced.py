
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "processed" / "dataset_inventory.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "dataset_split_balanced.csv"

RANDOM_STATE = 42


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "image_path",
    "label",
    "group_id",
    "source_dataset",
    "sha256",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["group_id"] = df["group_id"].astype(str)
df["source_dataset"] = df["source_dataset"].astype(str)
df["label"] = df["label"].astype(str)

if df["sha256"].duplicated().any():
    duplicates = df[df["sha256"].duplicated(keep=False)]
    raise ValueError(
        "Duplicate image hashes detected. Resolve them before splitting.\n"
        + duplicates[["filename", "sha256"]].to_string(index=False)
    )

# Preserve the complete group ID.
# Example:
# original|V01HE and original|V01PE remain separate.
# handpd|H1 and handpd|P1 remain separate.

df["participant_group"] = (
    df["source_dataset"] + "|" + df["group_id"]
)

# Every group must contain only one class.
mixed = (
    df.groupby("participant_group")["label"]
    .nunique()
)

if (mixed > 1).any():
    raise ValueError(
        "Some groups contain multiple labels:\n"
        + mixed[mixed > 1].to_string()
    )


# ============================================================
# GROUP-LEVEL 70 / 15 / 15 SPLIT
# ============================================================

group_table = (
    df.groupby(
        ["source_dataset", "label", "participant_group"]
    )
    .size()
    .reset_index(name="image_count")
)

assignments = []

for (source, label), subset in group_table.groupby(
    ["source_dataset", "label"]
):

    groups = subset["participant_group"].tolist()

    if len(groups) < 3:
        raise ValueError(
            f"Not enough groups for {source}/{label}: {len(groups)}"
        )

    # First: 70% training, 30% temporary.
    train_groups, temporary_groups = train_test_split(
        groups,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    # Second: divide temporary groups equally.
    validation_groups, test_groups = train_test_split(
        temporary_groups,
        test_size=0.50,
        random_state=RANDOM_STATE,
    )

    for group in train_groups:
        assignments.append((group, "train"))

    for group in validation_groups:
        assignments.append((group, "validation"))

    for group in test_groups:
        assignments.append((group, "test"))


# ============================================================
# APPLY SPLITS
# ============================================================

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
    raise RuntimeError("Some images were not assigned a split.")


# ============================================================
# LEAKAGE VERIFICATION
# ============================================================

group_overlap = (
    result.groupby("participant_group")["split"].nunique()
)

if (group_overlap > 1).any():
    raise RuntimeError("Participant group leakage detected!")

hash_overlap = result.groupby("sha256")["split"].nunique()

if (hash_overlap > 1).any():
    raise RuntimeError("Duplicate hash leakage detected!")


# ============================================================
# SAVE NEW SPLIT
# ============================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# REPORT
# ============================================================

print("\n========== BALANCED GROUP SPLIT ==========")

print("\nIMAGE COUNTS BY SOURCE, SPLIT AND CLASS:")

print(
    result.groupby(
        ["source_dataset", "split", "label"]
    ).size().unstack(fill_value=0).to_string()
)

print("\nPARTICIPANT GROUP COUNTS:")

print(
    result.groupby(
        ["source_dataset", "split"]
    )["participant_group"].nunique().to_string()
)

print("\nTOTAL IMAGE COUNTS:")

print(result["split"].value_counts().to_string())

print("\nTOTAL GROUP COUNTS:")

print(
    result.groupby("split")["participant_group"]
    .nunique()
    .to_string()
)

print("\nGROUP LEAKAGE: NONE")
print("DUPLICATE HASH LEAKAGE: NONE")
print(f"\nSaved new split to: {OUTPUT_FILE}")
print("Original inventory and previous split were not modified.")
