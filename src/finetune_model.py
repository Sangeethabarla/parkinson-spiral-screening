
from pathlib import Path
import csv
import json
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 8
LEARNING_RATE = 1e-5
UNFREEZE_LAST_LAYERS = 30

ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSV = ROOT / "data" / "processed" / "dataset_split.csv"
BASELINE_PATH = ROOT / "artifacts" / "parkinson_effnetv2s_expanded.keras"
MODEL_PATH = ROOT / "artifacts" / "parkinson_effnetv2s_finetuned.keras"
HISTORY_PATH = ROOT / "artifacts" / "training_history_finetuned.json"

CLASS_NAMES = ["healthy", "parkinson"]

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_rows():
    with SPLIT_CSV.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        path = Path(row["image_path"])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(path)
        row["image_path"] = str(path)

    return rows


def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image, channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(
        image, IMAGE_SIZE, antialias=True
    )
    return tf.cast(image, tf.float32), label


def make_dataset(rows, training=False):
    paths = [r["image_path"] for r in rows]
    labels = [CLASS_NAMES.index(r["label"]) for r in rows]

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            len(rows), seed=SEED, reshuffle_each_iteration=True
        )

    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        augmentation = keras.Sequential([
            layers.RandomRotation(0.04),
            layers.RandomZoom(0.08),
            layers.RandomTranslation(0.04, 0.04),
        ], name="finetune_augmentation")

        ds = ds.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def find_backbone(model):
    candidates = [
        layer for layer in model.layers
        if isinstance(layer, keras.Model)
        and "efficientnetv2" in layer.name.lower()
    ]

    if len(candidates) != 1:
        names = [
            (layer.name, type(layer).__name__)
            for layer in model.layers
        ]
        raise ValueError(
            "Could not uniquely identify EfficientNetV2 backbone. "
            f"Model layers: {names}"
        )

    return candidates[0]


def main():
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Baseline model not found: {BASELINE_PATH}"
        )

    rows = load_rows()
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "validation"]

    if not train_rows or not val_rows:
        raise ValueError("Training and validation data are required.")

    print(f"Training images: {len(train_rows)}")
    print(f"Validation images: {len(val_rows)}")
    print("Test data will not be used during fine-tuning.")

    train_ds = make_dataset(train_rows, training=True)
    val_ds = make_dataset(val_rows, training=False)

    print("\nLoading existing expanded baseline...")
    model = keras.models.load_model(BASELINE_PATH)

    backbone = find_backbone(model)
    backbone.trainable = True

    # Keep only the final layers trainable.
    for layer in backbone.layers:
        layer.trainable = False

    eligible_layers = [
        layer for layer in backbone.layers
        if not isinstance(layer, layers.BatchNormalization)
    ]

    for layer in eligible_layers[-UNFREEZE_LAST_LAYERS:]:
        layer.trainable = True

    # Always keep Batch Normalization frozen for this small dataset.
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    trainable_count = sum(
        int(tf.size(variable).numpy())
        for variable in model.trainable_variables
    )

    print(f"Backbone: {backbone.name}")
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Learning rate: {LEARNING_RATE}")

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
            str(MODEL_PATH),
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
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    HISTORY_PATH.write_text(
        json.dumps(
            {
                key: [float(v) for v in values]
                for key, values in history.history.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nFine-tuning finished.")
    print(f"New model: {MODEL_PATH}")
    print(f"Training history: {HISTORY_PATH}")
    print("Original expanded model remains unchanged.")


if __name__ == "__main__":
    main()
