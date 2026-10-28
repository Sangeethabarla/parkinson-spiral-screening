
from pathlib import Path
import csv
import json
import random
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4

# Keep the original training settings for a fair comparison.
EPOCHS = 8
LEARNING_RATE = 1e-5
UNFREEZE_LAST_LAYERS = 30

# The only intended training change:
# soften balanced class weights to reduce their influence.
WEIGHT_POWER = 0.5

ROOT = Path(__file__).resolve().parents[1]

SPLIT_CSV = (
    ROOT / "data" / "processed" / "dataset_split.csv"
)

# Start from the same expanded baseline.
# Neither existing model will be modified.
BASELINE_PATH = (
    ROOT / "artifacts"
    / "parkinson_effnetv2s_expanded.keras"
)

# NEW checkpoint and history filenames.
MODEL_PATH = (
    ROOT / "artifacts"
    / "parkinson_effnetv2s_mildweighted_exp1.keras"
)

HISTORY_PATH = (
    ROOT / "artifacts"
    / "training_history_mildweighted_exp1.json"
)

CLASS_NAMES = ["healthy", "parkinson"]

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. LOAD DATA
# ============================================================

def load_rows():
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(SPLIT_CSV)

    with SPLIT_CSV.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    for row in rows:
        path = Path(row["image_path"])

        if not path.is_absolute():
            path = ROOT / path

        if not path.is_file():
            raise FileNotFoundError(path)

        row["image_path"] = str(path)
        row["split"] = str(row["split"]).strip().lower()
        row["label"] = str(row["label"]).strip().lower()

        if row["label"] not in CLASS_NAMES:
            raise ValueError(
                f"Unexpected label: {row['label']}"
            )

    return rows


def load_image(path, label):
    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    # Match the original training preprocessing.
    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        antialias=True,
    )

    return tf.cast(image, tf.float32), label


# ============================================================
# 3. MILD CLASS WEIGHTS
# ============================================================

def calculate_class_weights(train_rows):
    labels = [
        CLASS_NAMES.index(row["label"])
        for row in train_rows
    ]

    counts = Counter(labels)
    total = len(labels)
    number_of_classes = len(CLASS_NAMES)

    if len(counts) != number_of_classes:
        raise ValueError(
            "Training data must contain both classes."
        )

    # Original balanced class weights.
    balanced_weights = {
        class_id: total / (
            number_of_classes * counts[class_id]
        )
        for class_id in range(number_of_classes)
    }

    # Soften the weights using a square root.
    # This reduces the difference between class weights.
    mild_weights = {
        class_id: balanced_weights[class_id] ** WEIGHT_POWER
        for class_id in range(number_of_classes)
    }

    print("\nTraining class distribution:")

    for class_id, class_name in enumerate(CLASS_NAMES):
        print(
            f"  {class_name}: {counts[class_id]} images"
            f" | original balanced weight: "
            f"{balanced_weights[class_id]:.4f}"
            f" | mild weight: {mild_weights[class_id]:.4f}"
        )

    return mild_weights


# ============================================================
# 4. CREATE DATASETS
# ============================================================

def make_dataset(
    rows,
    training=False,
    class_weights=None,
):
    paths = [row["image_path"] for row in rows]

    labels = [
        CLASS_NAMES.index(row["label"])
        for row in rows
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            len(rows),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        augmentation = keras.Sequential(
            [
                layers.RandomRotation(0.04),
                layers.RandomZoom(0.08),
                layers.RandomTranslation(0.04, 0.04),
            ],
            name="mildweighted_augmentation",
        )

        dataset = dataset.map(
            lambda image, label: (
                augmentation(image, training=True),
                label,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        weight_lookup = tf.constant(
            [
                class_weights[0],
                class_weights[1],
            ],
            dtype=tf.float32,
        )

        dataset = dataset.map(
            lambda image, label: (
                image,
                label,
                tf.gather(
                    weight_lookup,
                    tf.cast(label, tf.int32),
                ),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return (
        dataset
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


# ============================================================
# 5. FIND EFFICIENTNET BACKBONE
# ============================================================

def find_backbone(model):
    candidates = [
        layer
        for layer in model.layers
        if isinstance(layer, keras.Model)
        and "efficientnetv2" in layer.name.lower()
    ]

    if len(candidates) != 1:
        raise ValueError(
            "Could not uniquely identify the EfficientNetV2 "
            "backbone. Check the saved model architecture."
        )

    return candidates[0]


# ============================================================
# 6. TRAIN THE NEW EXPERIMENT
# ============================================================

def main():
    # Refuse to overwrite any previous experiment output.
    if MODEL_PATH.exists():
        raise FileExistsError(
            f"Experiment checkpoint already exists:\n{MODEL_PATH}\n"
            "Rename the output paths before running another experiment."
        )

    if HISTORY_PATH.exists():
        raise FileExistsError(
            f"Experiment history already exists:\n{HISTORY_PATH}\n"
            "Rename the output paths before running another experiment."
        )

    if not BASELINE_PATH.exists():
        raise FileNotFoundError(BASELINE_PATH)

    rows = load_rows()

    train_rows = [
        row for row in rows
        if row["split"] == "train"
    ]

    val_rows = [
        row for row in rows
        if row["split"] in ("validation", "val")
    ]

    if not train_rows or not val_rows:
        raise ValueError(
            "Training and validation rows are required. "
            "Check split names in dataset_split.csv."
        )

    # Test data is not used for training or model selection.
    test_count = sum(
        row["split"] in ("test", "testing")
        for row in rows
    )

    print("=" * 65)
    print("MILD CLASS-WEIGHTED FINE-TUNING EXPERIMENT")
    print("=" * 65)

    print(f"\nTraining images: {len(train_rows)}")
    print(f"Validation images: {len(val_rows)}")
    print(f"Test images excluded: {test_count}")

    print(f"\nBaseline model:\n{BASELINE_PATH}")
    print(f"New checkpoint:\n{MODEL_PATH}")

    class_weights = calculate_class_weights(train_rows)

    train_dataset = make_dataset(
        train_rows,
        training=True,
        class_weights=class_weights,
    )

    validation_dataset = make_dataset(
        val_rows,
        training=False,
    )

    print("\nLoading expanded baseline...")
    model = keras.models.load_model(
        BASELINE_PATH,
        compile=False,
    )

    backbone = find_backbone(model)
    backbone.trainable = True

    # Freeze every backbone layer first.
    for layer in backbone.layers:
        layer.trainable = False

    # Unfreeze the same number of eligible layers as the
    # previous experiment.
    eligible_layers = [
        layer
        for layer in backbone.layers
        if not isinstance(
            layer, layers.BatchNormalization
        )
    ]

    for layer in eligible_layers[-UNFREEZE_LAST_LAYERS:]:
        layer.trainable = True

    # Keep BatchNormalization layers frozen.
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    trainable_parameters = sum(
        int(tf.size(variable).numpy())
        for variable in model.trainable_variables
    )

    print(f"\nBackbone: {backbone.name}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Maximum epochs: {EPOCHS}")
    print(f"Class-weight power: {WEIGHT_POWER}")

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Save training history separately.
    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with HISTORY_PATH.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(history_data, file, indent=2)

    print("\n" + "=" * 65)
    print("EXPERIMENT COMPLETED")
    print("=" * 65)

    print(f"\nNew model:\n{MODEL_PATH}")
    print(f"Training history:\n{HISTORY_PATH}")

    print("\nOriginal model checkpoints were not modified.")
    print("Original training history was not modified.")
    print("Test data was not used for training or selection.")


if __name__ == "__main__":
    main()
