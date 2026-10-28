
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.common import ROOT, load_config, load_image


AUTOTUNE = tf.data.AUTOTUNE

MANIFEST_PATH = (
    ROOT
    / "data"
    / "processed"
    / "dataset_split.csv"
)


# ============================================================
# 1. LOAD TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def get_split(name):
    """
    Load an existing split from dataset_split.csv.

    Class mapping:
        healthy   -> 0
        parkinson -> 1

    This function does not regenerate the dataset splits.
    """

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset manifest not found: {MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    required_columns = {"split", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Use an existing resolved path if available.
    if "resolved_image_path" not in df.columns:
        if "image_path" not in df.columns:
            raise ValueError(
                "Manifest must contain 'image_path' or "
                "'resolved_image_path'."
            )

        df["resolved_image_path"] = df["image_path"]

    # Normalize split names.
    split_aliases = {
        "train": "train",
        "training": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "test": "test",
        "testing": "test",
    }

    df["_normalized_split"] = (
        df["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(lambda value: split_aliases.get(value, value))
    )

    requested_split = str(name).strip().lower()
    requested_split = split_aliases.get(
        requested_split,
        requested_split,
    )

    split_df = df.loc[
        df["_normalized_split"] == requested_split
    ].copy()

    split_df.drop(
        columns=["_normalized_split"],
        inplace=True,
    )

    if split_df.empty:
        available_splits = sorted(
            df["split"].astype(str).unique().tolist()
        )

        raise ValueError(
            f"No rows found for split '{name}'. "
            f"Available split names: {available_splits}"
        )

    # --------------------------------------------------------
    # Normalize labels.
    # --------------------------------------------------------

    label_mapping = {
        "healthy": 0,
        "parkinson": 1,
        "parkinson's": 1,
        "parkinsons": 1,
        "0": 0,
        "1": 1,
    }

    normalized_labels = (
        split_df["label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapped_labels = normalized_labels.map(label_mapping)

    if mapped_labels.isna().any():
        unknown_labels = (
            normalized_labels[mapped_labels.isna()]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unknown labels in manifest: {unknown_labels}"
        )

    split_df["label"] = mapped_labels.astype(np.int32)

    # --------------------------------------------------------
    # Resolve image paths.
    # --------------------------------------------------------

    def resolve_image_path(value):
        path = Path(str(value).strip())

        if not path.is_absolute():
            path = ROOT / path

        return str(path.resolve())

    split_df["resolved_image_path"] = (
        split_df["resolved_image_path"]
        .apply(resolve_image_path)
    )

    missing_images = [
        path
        for path in split_df["resolved_image_path"]
        if not Path(path).is_file()
    ]

    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} image files were not found. "
            f"First missing file: {missing_images[0]}"
        )

    # Check class representation.
    class_counts = (
        split_df["label"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print(
        f"Loaded {requested_split} split: "
        f"{len(split_df)} images"
    )

    print(
        "Class counts (0=healthy, 1=parkinson): "
        f"{class_counts}"
    )

    return split_df.reset_index(drop=True)


# ============================================================
# 2. LOAD IMAGE AND LABEL
# ============================================================

def _load(path, label):
    """
    Load an image using the project's existing image loader.
    """

    image = load_image(path)

    label = tf.cast(
        label,
        tf.float32,
    )

    return image, label


# ============================================================
# 3. CREATE TENSORFLOW DATASET
# ============================================================

def make_dataset(df, training=False):
    """
    Create a batched TensorFlow dataset.

    Training:
        Shuffle and apply augmentation.

    Validation / test:
        No shuffle and no augmentation.
    """

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
            seed=cfg["seed"],
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        _load,
        num_parallel_calls=AUTOTUNE,
    )

    if training:

        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomRotation(0.04),
                tf.keras.layers.RandomTranslation(
                    0.04,
                    0.04,
                ),
                tf.keras.layers.RandomZoom(0.08),
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.GaussianNoise(2.0),
            ],
            name="training_augmentation",
        )

        dataset = dataset.map(
            lambda image, label: (
                augmentation(image, training=True),
                label,
            ),
            num_parallel_calls=AUTOTUNE,
        )

    batch_size = int(cfg["batch_size"])

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(
        AUTOTUNE
    )

    return dataset
