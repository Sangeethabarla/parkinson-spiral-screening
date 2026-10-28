
from pathlib import Path
import csv
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "data" / "processed" / "dataset_inventory.csv"

groups = defaultdict(list)

with INVENTORY.open("r", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        groups[row["sha256"]].append(row)

duplicates = {
    digest: rows
    for digest, rows in groups.items()
    if len(rows) > 1
}

print("Exact duplicate groups:", len(duplicates))
print("Files involved:", sum(len(rows) for rows in duplicates.values()))

for number, rows in enumerate(duplicates.values(), start=1):
    print(f"\nGroup {number}:")
    for row in rows:
        print(
            f"  {row['source_dataset']} | "
            f"{row['label']} | {row['filename']} | "
            f"{row['group_id']}"
        )
