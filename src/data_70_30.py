
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.common import ROOT, load_config, load_image


AUTOTUNE = tf.data.AUTOTUNE

# Use the new balanced, group-based split.
MANIFEST_PATH = (
    ROOT
    / "data"
    / "processed"
    / "dataset_split_70_30.csv"
)


# ============================================================
# 1. LOAD AND VALIDATE DATA SPLITS
# ============================================================

def get_split(name):
    """
    Load a split from dataset_split_70_30.csv.

    Class mapping:
        healthy   -> 0
        parkinson -> 1
    """

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset manifest not found: {MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    required = {
        "image_path",
        "label",
        "split",
        "group_id",
        "source_dataset",
        "sha256",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing manifest columns: {sorted(missing)}"
        )

    # Normalize split names.
    df["split"] = (
        df["split"].astype(str).str.strip().str.lower()
    )

    # Normalize labels.
    label_mapping = {
        "healthy": 0,
        "parkinson": 1,
        "parkinson's": 1,
        "parkinsons": 1,
        "0": 0,
        "1": 1,
    }

    normalized = (
        df["label"].astype(str).str.strip().str.lower()
    )

    mapped = normalized.map(label_mapping)

    if mapped.isna().any():
        raise ValueError(
            f"Unknown labels: "
            f"{normalized[mapped.isna()].unique().tolist()}"
        )

    df["label"] = mapped.astype(np.int32)

    # Construct a source-specific group key.
    df["participant_group"] = (
        df["source_dataset"].astype(str)
        + "|"
        + df["group_id"].astype(str)
    )

    # Each group must have one label.
    group_label_counts = (
        df.groupby("participant_group")["label"].nunique()
    )

    if (group_label_counts > 1).any():
        raise ValueError(
            "Some participant groups contain multiple labels."
        )

    # Check that participant groups do not cross splits.
    group_split_counts = (
        df.groupby("participant_group")["split"].nunique()
    )

    if (group_split_counts > 1).any():
        bad = group_split_counts[group_split_counts > 1]
        raise ValueError(
            "Participant-group leakage detected:\n"
            f"{bad.index.tolist()}"
        )

    # Check duplicate image hashes.
    if df["sha256"].duplicated().any():
        duplicates = df.loc[
            df["sha256"].duplicated(keep=False),
            ["filename", "sha256", "split"],
        ]

        raise ValueError(
            "Duplicate image hashes detected:\n"
            f"{duplicates.to_string(index=False)}"
        )

    requested = str(name).strip().lower()

    aliases = {
        "training": "train",
        "val": "validation",
        "valid": "validation",
        "testing": "test",
    }

    requested = aliases.get(requested, requested)

    if requested not in {"train", "validation", "test"}:
        raise ValueError(f"Invalid split name: {name}")

    split_df = df.loc[df["split"] == requested].copy()

    if split_df.empty:
        raise ValueError(
            f"No images found for split: {requested}"
        )

    # Resolve image paths.
    def resolve_path(value):
        path = Path(str(value).strip())

        if not path.is_absolute():
            path = ROOT / path

        return str(path.resolve())

    split_df["resolved_image_path"] = (
        split_df["image_path"].apply(resolve_path)
    )

    missing_images = [
        path
        for path in split_df["resolved_image_path"]
        if not Path(path).is_file()
    ]

    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} images missing. "
            f"First missing image: {missing_images[0]}"
        )

    counts = (
        split_df["label"]
        .value_counts()
        .reindex([0, 1], fill_value=0)
    )

    print(f"\nLoaded {requested.upper()} split")
    print("Total images:", len(split_df))
    print("Healthy:", int(counts[0]))
    print("Parkinson:", int(counts[1]))
    print(
        "Participant groups:",
        split_df["participant_group"].nunique()
    )

    return split_df.reset_index(drop=True)


# ============================================================
# 2. IMAGE LOADING
# ============================================================

def _load(path, label):
    image = load_image(path)

    label = tf.cast(label, tf.float32)

    return image, label


# ============================================================
# 3. TRAINING AUGMENTATION
# ============================================================

def build_augmentation():
    """
    Mild augmentation for spiral drawings.

    No horizontal flipping:
    it can alter drawing orientation and handedness.

    EfficientNetV2 preprocessing remains unchanged.
    """

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(
                factor=0.02,
                fill_mode="reflect",
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=0.02,
                width_factor=0.02,
                fill_mode="reflect",
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(-0.04, 0.04),
                width_factor=(-0.04, 0.04),
                fill_mode="reflect",
            ),
        ],
        name="training_augmentation",
    )


# ============================================================
# 4. CREATE TENSORFLOW DATASET
# ============================================================

def make_dataset(df, training=False):

    if df is None or len(df) == 0:
        raise ValueError(
            "Cannot create a dataset from an empty DataFrame."
        )

    cfg = load_config()

    paths = (
        df["resolved_image_path"]
        .astype(str)
        .to_numpy()
    )

    labels = (
        df["label"]
        .astype(np.float32)
        .to_numpy()
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=int(cfg["seed"]),
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        _load,
        num_parallel_calls=AUTOTUNE,
    )

    if training:
        augmentation = build_augmentation()

        dataset = dataset.map(
            lambda image, label: (
                augmentation(image, training=True),
                label,
            ),
            num_parallel_calls=AUTOTUNE,
        )

    dataset = dataset.batch(
        int(cfg["batch_size"])
    )

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset

