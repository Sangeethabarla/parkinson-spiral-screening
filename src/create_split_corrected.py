from pathlib import Path
import csv
import random
import re
from collections import defaultdict, Counter

# ==================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INVENTORY = (
    PROJECT_ROOT / "data" / "processed" / "dataset_inventory.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "dataset_split_corrected.csv"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SPLITS = ("train", "validation", "test")


# ==================================================
# 2. LOAD INVENTORY
# ==================================================

def load_inventory():
    if not INVENTORY.exists():
        raise FileNotFoundError(
            f"Inventory not found: {INVENTORY}"
        )

    with INVENTORY.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    required = {
        "image_path", "label", "group_id",
        "source_dataset", "filename", "sha256"
    }

    if not rows:
        raise ValueError("Dataset inventory is empty.")

    missing = required - set(rows[0].keys())

    if missing:
        raise ValueError(
            f"Inventory missing columns: {sorted(missing)}"
        )

    for row in rows:
        if row["label"] not in {"healthy", "parkinson"}:
            raise ValueError(
                f"Unexpected label: {row['label']}"
            )

        if not row["group_id"]:
            raise ValueError(
                f"Missing group ID: {row['image_path']}"
            )

        image_path = PROJECT_ROOT / row["image_path"]

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        if not row["sha256"]:
            raise ValueError(
                f"Missing image hash: {row['image_path']}"
            )

    return rows


# ==================================================
# 3. CHECK EXACT DUPLICATES
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
        raise ValueError(
            f"Found {len(duplicates)} duplicate image hashes. "
            "Resolve duplicates before creating a split."
        )

    print("Exact duplicate hashes: 0")


# ==================================================
# 4. CREATE PARTICIPANT GROUPS
# ==================================================

def get_group_key(row):
    source = row["source_dataset"].strip().lower()
    filename = Path(row["filename"]).name.upper()

    # Original spiral dataset:
    # V01HE01.png and V01PE01.png -> same group V01
    if source == "original":
        match = re.match(r"^(V\d+)(?:HE|PE)", filename)

        if not match:
            raise ValueError(
                f"Cannot derive participant group from {filename}"
            )

        return source, match.group(1)

    # Retain existing HandPD grouping.
    return source, row["group_id"].strip()


def create_groups(rows):
    groups = defaultdict(list)

    for row in rows:
        groups[get_group_key(row)].append(row)

    print(f"Filename-derived groups: {len(groups)}")

    # A participant group may contain both healthy and
    # Parkinson's drawings. That is allowed.
    return groups


# ==================================================
# 5. ASSIGN GROUPS TO SPLITS
# ==================================================

def assign_splits(groups):
    rng = random.Random(SEED)

    items = list(groups.items())
    rng.shuffle(items)

    # Place larger groups first to reduce imbalance.
    items.sort(
        key=lambda item: len(item[1]),
        reverse=True
    )

    ratios = {
        "train": TRAIN_RATIO,
        "validation": VAL_RATIO,
        "test": TEST_RATIO
    }

    total_images = sum(
        len(group_rows) for _, group_rows in items
    )

    total_by_class = Counter(
        row["label"]
        for _, group_rows in items
        for row in group_rows
    )

    target_total = {
        split: total_images * ratios[split]
        for split in SPLITS
    }

    target_class = {
        split: {
            label: total_by_class[label] * ratios[split]
            for label in ("healthy", "parkinson")
        }
        for split in SPLITS
    }

    current_total = Counter()
    current_class = {
        split: Counter()
        for split in SPLITS
    }

    assignments = {}

    def cost(split, group_rows):
        new_total = current_total[split] + len(group_rows)

        score = (
            (new_total - target_total[split])
            / max(target_total[split], 1)
        ) ** 2

        added_classes = Counter(
            row["label"] for row in group_rows
        )

        for label in ("healthy", "parkinson"):
            new_count = (
                current_class[split][label]
                + added_classes[label]
            )

            target = max(
                target_class[split][label], 1
            )

            score += (
                (new_count - target_class[split][label])
                / target
            ) ** 2

        return score

    for index, (group_key, group_rows) in enumerate(items):
        remaining = len(items) - index

        empty_splits = [
            split for split in SPLITS
            if current_total[split] == 0
        ]

        if remaining <= len(empty_splits) and empty_splits:
            chosen = empty_splits[0]
        else:
            chosen = min(
                SPLITS,
                key=lambda split: (
                    cost(split, group_rows),
                    SPLITS.index(split)
                )
            )

        assignments[group_key] = chosen

        current_total[chosen] += len(group_rows)
        current_class[chosen].update(
            row["label"] for row in group_rows
        )

    return assignments


# ==================================================
# 6. SAVE SEPARATE CORRECTED MANIFEST
# ==================================================

def save_split(rows, assignments):
    split_rows = []

    for row in rows:
        updated = dict(row)
        group_key = get_group_key(row)

        updated["split"] = assignments[group_key]
        updated["participant_group"] = "|".join(group_key)

        split_rows.append(updated)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fields = list(rows[0].keys()) + [
        "split", "participant_group"
    ]

    with OUTPUT_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(split_rows)

    return split_rows


# ==================================================
# 7. VERIFY SPLITS
# ==================================================

def verify_and_print(rows):
    print("\n========== CORRECTED SPLIT SUMMARY ==========")
    print(f"Total images: {len(rows)}")

    for split in SPLITS:
        subset = [
            row for row in rows
            if row["split"] == split
        ]

        counts = Counter(
            row["label"] for row in subset
        )

        groups = {
            row["participant_group"]
            for row in subset
        }

        print(
            f"{split:10s}: {len(subset):3d} images | "
            f"{len(groups):3d} groups | "
            f"healthy={counts['healthy']}, "
            f"parkinson={counts['parkinson']}"
        )

    group_splits = defaultdict(set)
    hash_splits = defaultdict(set)

    for row in rows:
        group_splits[
            row["participant_group"]
        ].add(row["split"])

        hash_splits[
            row["sha256"]
        ].add(row["split"])

    group_overlap = {
        key: splits
        for key, splits in group_splits.items()
        if len(splits) > 1
    }

    hash_overlap = {
        digest: splits
        for digest, splits in hash_splits.items()
        if len(splits) > 1
    }

    print(
        "Participant-group overlap:",
        len(group_overlap)
    )

    print(
        "Exact hashes crossing splits:",
        len(hash_overlap)
    )

    if group_overlap:
        raise ValueError(
            "Group leakage detected. Do not use this split."
        )

    if hash_overlap:
        raise ValueError(
            "Duplicate leakage detected. Do not use this split."
        )

    print(f"\nSaved manifest: {OUTPUT_FILE}")
    print("Split verification passed.")
    print(
        "Note: Filename-derived groups must be verified "
        "against dataset documentation before claiming "
        "true participant-level independence."
    )


# ==================================================
# 8. MAIN
# ==================================================

def main():
    print(
        "Creating corrected split. "
        "Existing dataset_split.csv will not be changed."
    )

    rows = load_inventory()
    print(f"Loaded images: {len(rows)}")

    check_duplicate_hashes(rows)

    groups = create_groups(rows)
    assignments = assign_splits(groups)

    split_rows = save_split(rows, assignments)

    verify_and_print(split_rows)


if __name__ == "__main__":
    main()