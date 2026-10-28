
from pathlib import Path
import csv
import random
from collections import defaultdict, Counter

# ==================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INVENTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_inventory.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SPLITS = ("train", "validation", "test")


# ==================================================
# 2. LOAD CLEAN INVENTORY
# ==================================================

def load_inventory():
    if not INVENTORY.exists():
        raise FileNotFoundError(
            f"Inventory not found: {INVENTORY}\n"
            "Run: python src/prepare_data.py"
        )

    with INVENTORY.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    required_fields = {
        "image_path",
        "label",
        "group_id",
        "source_dataset",
        "filename",
        "sha256",
    }

    if not rows:
        raise ValueError("The dataset inventory is empty.")

    missing = required_fields - set(rows[0].keys())

    if missing:
        raise ValueError(
            f"Inventory is missing columns: {sorted(missing)}\n"
            "Run the updated src/prepare_data.py first."
        )

    for row in rows:
        if row["label"] not in {"healthy", "parkinson"}:
            raise ValueError(
                f"Unexpected label in inventory: {row['label']}"
            )

        if not row["group_id"]:
            raise ValueError(
                f"Missing group ID for {row['image_path']}. "
                "Resolve group IDs before splitting."
            )

        image_path = PROJECT_ROOT / row["image_path"]

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image listed in inventory is missing:\n"
                f"{image_path}"
            )

    return rows


# ==================================================
# 3. CHECK FOR EXACT DUPLICATES
# ==================================================

def check_duplicate_hashes(rows):
    hash_to_paths = defaultdict(list)

    for row in rows:
        hash_to_paths[row["sha256"]].append(
            row["image_path"]
        )

    duplicates = {
        digest: paths
        for digest, paths in hash_to_paths.items()
        if len(paths) > 1
    }

    if duplicates:
        print("\nERROR: Exact duplicate images remain.")

        for paths in duplicates.values():
            for path in paths:
                print(" ", path)

        raise ValueError(
            "Re-run src/prepare_data.py and inspect duplicates "
            "before creating the split."
        )

    print("Exact duplicate hashes remaining: 0")


# ==================================================
# 4. CREATE SOURCE-SPECIFIC GROUPS
# ==================================================

def create_groups(rows):
    """
    Keep each filename-derived group in one split.

    Include source_dataset in the key so IDs from different
    datasets are not accidentally treated as the same person.

    NOTE: These are filename-derived groups, not verified
    real-world participant identities.
    """

    groups = defaultdict(list)

    for row in rows:
        group_key = (
            row["source_dataset"],
            row["group_id"]
        )

        groups[group_key].append(row)

    # Every group must have a single class label.
    for group_key, images in groups.items():
        labels = {image["label"] for image in images}

        if len(labels) != 1:
            raise ValueError(
                f"Group {group_key} contains multiple labels: "
                f"{labels}"
            )

    return groups


# ==================================================
# 5. STRATIFIED GROUP-LEVEL SPLIT
# ==================================================

def assign_splits(groups):
    rng = random.Random(SEED)

    # Separate groups by class.
    label_groups = defaultdict(list)

    for group_key, images in groups.items():
        label = images[0]["label"]
        label_groups[label].append(group_key)

    assignments = {}

    for label in ("healthy", "parkinson"):

        group_keys = label_groups[label]
        rng.shuffle(group_keys)

        number_of_groups = len(group_keys)

        if number_of_groups < 3:
            raise ValueError(
                f"Class '{label}' has only {number_of_groups} "
                "groups. At least 3 are needed for "
                "train/validation/test."
            )

        # Allocate approximately 70/15/15 by group count.
        n_train = max(
            1,
            round(number_of_groups * TRAIN_RATIO)
        )

        n_validation = max(
            1,
            round(number_of_groups * VAL_RATIO)
        )

        # Always reserve at least one group for testing.
        while n_train + n_validation >= number_of_groups:
            if n_train > 1:
                n_train -= 1
            elif n_validation > 1:
                n_validation -= 1
            else:
                raise ValueError(
                    f"Cannot split class '{label}' safely."
                )

        train_keys = group_keys[:n_train]

        validation_keys = group_keys[
            n_train:n_train + n_validation
        ]

        test_keys = group_keys[
            n_train + n_validation:
        ]

        for key in train_keys:
            assignments[key] = "train"

        for key in validation_keys:
            assignments[key] = "validation"

        for key in test_keys:
            assignments[key] = "test"

        print(
            f"{label}: {number_of_groups} groups -> "
            f"train={len(train_keys)}, "
            f"validation={len(validation_keys)}, "
            f"test={len(test_keys)}"
        )

    return assignments


# ==================================================
# 6. SAVE SPLIT CSV
# ==================================================

def save_split(rows, assignments):
    split_rows = []

    for row in rows:
        group_key = (
            row["source_dataset"],
            row["group_id"]
        )

        updated_row = dict(row)
        updated_row["split"] = assignments[group_key]

        split_rows.append(updated_row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "dataset_split.csv"

    fields = [
        "image_path",
        "label",
        "group_id",
        "source_dataset",
        "original_split",
        "filename",
        "sha256",
        "split",
    ]

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(split_rows)

    return split_rows, output_file


# ==================================================
# 7. VERIFY AND PRINT RESULTS
# ==================================================

def verify_and_print(split_rows, output_file):

    print("\n========== SPLIT SUMMARY ==========")
    print(f"Total images: {len(split_rows)}")

    for split in SPLITS:
        subset = [
            row for row in split_rows
            if row["split"] == split
        ]

        counts = Counter(
            row["label"] for row in subset
        )

        groups = {
            (row["source_dataset"], row["group_id"])
            for row in subset
        }

        print(
            f"{split:10s}: {len(subset):3d} images | "
            f"{len(groups):3d} groups | "
            f"healthy={counts['healthy']}, "
            f"parkinson={counts['parkinson']}"
        )

    # Verify that each source-specific group appears
    # in exactly one split.
    group_splits = defaultdict(set)

    for row in split_rows:
        key = (
            row["source_dataset"],
            row["group_id"]
        )
        group_splits[key].add(row["split"])

    overlapping_groups = {
        key: values
        for key, values in group_splits.items()
        if len(values) > 1
    }

    print(
        "Filename-derived groups crossing splits: "
        f"{len(overlapping_groups)}"
    )

    if overlapping_groups:
        raise ValueError(
            "Group leakage detected. Do not train yet."
        )

    # Verify exact hashes do not cross splits.
    hash_splits = defaultdict(set)

    for row in split_rows:
        hash_splits[row["sha256"]].add(row["split"])

    cross_split_hashes = {
        digest: values
        for digest, values in hash_splits.items()
        if len(values) > 1
    }

    print(
        "Exact hashes crossing splits: "
        f"{len(cross_split_hashes)}"
    )

    if cross_split_hashes:
        raise ValueError(
            "Exact duplicate leakage detected. Do not train yet."
        )

    print(f"Saved split CSV:\n{output_file}")
    print("Split verification passed.")
    print(
        "Caution: filename-derived groups are not proof "
        "of verified participant-level independence."
    )
    print("===================================\n")


# ==================================================
# 8. MAIN
# ==================================================

def main():
    print("Creating train/validation/test split...")

    rows = load_inventory()

    print(f"Loaded clean inventory: {len(rows)} images")

    check_duplicate_hashes(rows)

    groups = create_groups(rows)

    print(f"Source-specific filename groups: {len(groups)}")

    assignments = assign_splits(groups)

    split_rows, output_file = save_split(
        rows,
        assignments
    )

    verify_and_print(split_rows, output_file)


if __name__ == "__main__":
    main()
