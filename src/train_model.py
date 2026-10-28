
from pathlib import Path
import csv
import json
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==================================================
# 1. CONFIGURATION
# ==================================================

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 10

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset_split.csv"
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Separate filenames: preserve the original baseline.
MODEL_PATH = (
    ARTIFACTS_DIR
    / "parkinson_effnetv2s_expanded.keras"
)

HISTORY_PATH = (
    ARTIFACTS_DIR
    / "training_history_expanded.json"
)

CLASS_NAMES = ["healthy", "parkinson"]

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==================================================
# 2. LOAD SPLIT CSV
# ==================================================

def load_split():
    if not SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"Split CSV not found: {SPLIT_CSV}\n"
            "Run: python src/create_split.py"
        )

    with SPLIT_CSV.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    required = {
        "image_path",
        "label",
        "split",
        "sha256",
    }

    if not rows:
        raise ValueError("Dataset split CSV is empty.")

    missing = required - set(rows[0].keys())

    if missing:
        raise ValueError(
            f"Split CSV is missing columns: {sorted(missing)}"
        )

    for row in rows:
        row["image_path"] = str(
            PROJECT_ROOT / row["image_path"]
        )

        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])

        if row["label"] not in CLASS_NAMES:
            raise ValueError(
                f"Unexpected label: {row['label']}"
            )

        if row["split"] not in {
            "train", "validation", "test"
        }:
            raise ValueError(
                f"Unexpected split: {row['split']}"
            )

    return rows


# ==================================================
# 3. LOAD AND PREPROCESS IMAGES
# ==================================================

def load_image(path, label):
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        antialias=True,
    )

    image = tf.cast(image, tf.float32)

    return image, label


def make_dataset(rows, training=False):
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
            buffer_size=len(rows),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if training:
        augmentation = keras.Sequential([
            layers.RandomRotation(0.04),
            layers.RandomZoom(0.08),
            layers.RandomTranslation(0.04, 0.04),
        ], name="training_augmentation")

        dataset = dataset.map(
            lambda image, label: (
                augmentation(image, training=True),
                label,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ==================================================
# 4. BUILD EFFICIENTNETV2-S
# ==================================================

def build_model():
    inputs = keras.Input(
        shape=(224, 224, 3),
        name="spiral_image",
    )

    backbone = keras.applications.EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        include_preprocessing=True,
    )

    # Stage 1: freeze pretrained backbone.
    backbone.trainable = False

    features = backbone(inputs, training=False)

    features = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(features)

    features = layers.Dropout(
        0.35,
        name="classifier_dropout"
    )(features)

    features = layers.Dense(
        128,
        activation="relu",
        name="feature_dense",
    )(features)

    features = layers.Dropout(
        0.25,
        name="feature_dropout"
    )(features)

    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="pd_probability",
    )(features)

    model = keras.Model(
        inputs,
        outputs,
        name="Parkinson_EfficientNetV2S_Expanded",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    return model


# ==================================================
# 5. MAIN TRAINING PIPELINE
# ==================================================

def main():
    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_split()

    train_rows = [
        row for row in rows
        if row["split"] == "train"
    ]

    val_rows = [
        row for row in rows
        if row["split"] == "validation"
    ]

    test_rows = [
        row for row in rows
        if row["split"] == "test"
    ]

    if not train_rows or not val_rows or not test_rows:
        raise ValueError(
            "Train, validation, and test sets must all be non-empty."
        )

    print("\n========== DATASET SUMMARY ==========")
    print(f"Training images: {len(train_rows)}")
    print(f"Validation images: {len(val_rows)}")
    print(f"Test images: {len(test_rows)}")
    print("Test images will not be used for training.")
    print("=====================================\n")

    train_ds = make_dataset(
        train_rows,
        training=True,
    )

    val_ds = make_dataset(
        val_rows,
        training=False,
    )

    print("Loading ImageNet-pretrained EfficientNetV2-S...")

    model = build_model()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),

        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Save the best validation-loss checkpoint.
    # Do not overwrite it with the final epoch model.
    with HISTORY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                key: [float(value) for value in values]
                for key, values in history.history.items()
            },
            file,
            indent=2,
        )

    print("\n========== TRAINING COMPLETE ==========")
    print(f"Best-checkpoint path: {MODEL_PATH}")
    print(f"History path: {HISTORY_PATH}")
    print("The original baseline model was not overwritten.")
    print("========================================")

    print(
        "\nNext: evaluate this saved model on the held-out "
        "test set using src/evaluate_model.py."
    )


if __name__ == "__main__":
    main()
