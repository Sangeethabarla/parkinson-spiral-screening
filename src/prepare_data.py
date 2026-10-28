
from pathlib import Path
import csv
import re
import hashlib
from collections import Counter, defaultdict

# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = PROJECT_ROOT / "data" / "raw"

ORIGINAL_ROOT = RAW_ROOT / "drawings" / "spiral"
HANDPD_ROOT = RAW_ROOT / "newhandpd"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
VALID_CLASSES = {"healthy", "parkinson"}


# ==================================================
# 2. GROUP ID EXTRACTION
# ==================================================

def get_original_group(filename):
    """
    Example:
        V01HE01.png -> V01HE
        V01PE01.png -> V01PE

    Filename-derived group only.
    Actual participant identity must be verified.
    """
    match = re.match(r"^(V\d+)(HE|PE)", filename.upper())

    if not match:
        return None

    volunteer_id, class_code = match.groups()
    return f"{volunteer_id}{class_code}"


def get_handpd_group(filename):
    """
    Examples:
        sp1-H1.jpg -> H1
        sp2-H1.jpg -> H1
        sp1-P1.jpg -> P1

    These are filename-derived IDs.
    Verify their meaning against original metadata.
    """
    match = re.search(
        r"sp\d+[-_](H|P)(\d+)",
        filename,
        re.IGNORECASE
    )

    if not match:
        return None

    class_code, number = match.groups()
    return f"{class_code.upper()}{number}"


# ==================================================
# 3. FILE HASHING
# ==================================================

def calculate_sha256(image_path):
    """Calculate an exact file-content hash."""

    digest = hashlib.sha256()

    with image_path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


# ==================================================
# 4. LOAD A DATASET SOURCE
# ==================================================

def collect_images(root, source_name):
    records = []

    if not root.exists():
        print(f"WARNING: Folder not found: {root}")
        return records

    for image_path in sorted(root.rglob("*")):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        # ------------------------------------------
        # Original 102-image dataset
        # ------------------------------------------

        if source_name == "original":

            label = image_path.parent.name.lower()

            if label not in VALID_CLASSES:
                continue

            group_id = get_original_group(image_path.stem)

            original_split = (
                image_path.parent.parent.name.lower()
            )

        # ------------------------------------------
        # NewHandPD dataset
        # ------------------------------------------

        elif source_name == "handpd":

            class_folder = image_path.parent.name.lower()

            if class_folder == "healthy":
                label = "healthy"

            elif class_folder in {"patient", "parkinson"}:
                label = "parkinson"

            else:
                print(
                    "WARNING: Unknown HandPD class folder: "
                    f"{image_path}"
                )
                continue

            group_id = get_handpd_group(image_path.stem)
            original_split = "not_provided"

        else:
            continue

        if group_id is None:
            print(
                "WARNING: Could not parse group ID: "
                f"{image_path.name}"
            )

        records.append({
            "image_path": image_path.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "label": label,
            "group_id": group_id or "",
            "source_dataset": source_name,
            "original_split": original_split,
            "filename": image_path.name,
            "sha256": calculate_sha256(image_path),
        })

    return records


# ==================================================
# 5. LOAD BOTH DATASETS
# ==================================================

def load_dataset():

    records = []

    print("\nLoading original dataset...")

    original_records = collect_images(
        ORIGINAL_ROOT,
        "original"
    )

    print(
        f"Original dataset images found: "
        f"{len(original_records)}"
    )

    print("\nLoading HandPD dataset...")

    handpd_records = collect_images(
        HANDPD_ROOT,
        "handpd"
    )

    print(
        f"HandPD dataset images found: "
        f"{len(handpd_records)}"
    )

    records.extend(original_records)
    records.extend(handpd_records)

    if not records:
        raise RuntimeError(
            "No images found. Check these folders:\n"
            f"{ORIGINAL_ROOT}\n{HANDPD_ROOT}"
        )

    return records


# ==================================================
# 6. EXCLUDE EXACT DUPLICATE FILES
# ==================================================

def exclude_duplicate_groups(records):

    hash_groups = defaultdict(list)

    for row in records:
        hash_groups[row["sha256"]].append(row)

    duplicate_groups = {
        digest: rows
        for digest, rows in hash_groups.items()
        if len(rows) > 1
    }

    # Conservatively exclude EVERY file in a duplicate
    # group. This avoids choosing an arbitrary filename
    # as the representative participant image.
    excluded_records = [
        row
        for rows in duplicate_groups.values()
        for row in rows
    ]

    clean_records = [
        row
        for row in records
        if len(hash_groups[row["sha256"]]) == 1
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    excluded_file = OUTPUT_DIR / "duplicates_excluded.csv"

    fields = [
        "image_path",
        "label",
        "group_id",
        "source_dataset",
        "original_split",
        "filename",
        "sha256",
    ]

    with excluded_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        for row in excluded_records:
            writer.writerow({
                field: row.get(field, "")
                for field in fields
            })

    print("\n========== DUPLICATE CHECK ==========")
    print(f"Duplicate groups: {len(duplicate_groups)}")
    print(f"Files excluded: {len(excluded_records)}")
    print(f"Images retained: {len(clean_records)}")

    print(
        f"Excluded-file report:\n{excluded_file}"
    )

    print("Raw images were not deleted.")
    print("=====================================\n")

    return clean_records


# ==================================================
# 7. PRINT DATASET SUMMARY
# ==================================================

def print_summary(records):

    print("\n========== CLEAN DATASET SUMMARY ==========")

    print(f"Total retained images: {len(records)}")

    for source in ("original", "handpd"):

        source_records = [
            row
            for row in records
            if row["source_dataset"] == source
        ]

        counts = Counter(
            row["label"]
            for row in source_records
        )

        print(f"\nSource: {source}")
        print(f"  Total images: {len(source_records)}")
        print(f"  Healthy: {counts['healthy']}")
        print(f"  Parkinson: {counts['parkinson']}")

        missing_groups = sum(
            not row["group_id"]
            for row in source_records
        )

        print(
            f"  Missing group IDs: {missing_groups}"
        )

    total_counts = Counter(
        row["label"]
        for row in records
    )

    print("\nCombined class counts:")
    print(f"  Healthy: {total_counts['healthy']}")
    print(f"  Parkinson: {total_counts['parkinson']}")

    # Verify that no exact hashes remain duplicated.
    hashes = Counter(
        row["sha256"]
        for row in records
    )

    remaining_duplicates = sum(
        count > 1 for count in hashes.values()
    )

    print(
        "\nRemaining duplicate hash groups: "
        f"{remaining_duplicates}"
    )

    print("===========================================\n")


# ==================================================
# 8. SAVE CLEAN INVENTORY
# ==================================================

def save_inventory(records):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "dataset_inventory.csv"

    fields = [
        "image_path",
        "label",
        "group_id",
        "source_dataset",
        "original_split",
        "filename",
        "sha256",
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
        writer.writerows(records)

    print(f"Clean inventory saved to:\n{output_file}")


# ==================================================
# 9. MAIN
# ==================================================

def main():

    print("Preparing combined Parkinson's spiral dataset...")

    all_records = load_dataset()

    print(f"\nTotal images before duplicate filtering: "
          f"{len(all_records)}")

    clean_records = exclude_duplicate_groups(
        all_records
    )

    print_summary(clean_records)
    save_inventory(clean_records)

    print("\nPreparation completed successfully.")
    print("Original images remain unchanged.")
    print("Duplicate files are excluded from the inventory.")
    print("No train/validation/test split has been created yet.")


if __name__ == "__main__":
    main()
