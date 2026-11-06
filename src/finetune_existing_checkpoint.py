
"""
Continue fine-tuning the existing class-weighted EfficientNetV2-S.

SAFETY:
- Does not overwrite the original checkpoint.
- Saves the best new checkpoint separately.
- Saves training history separately.
- Uses train and validation splits only.
- Does not evaluate or train on the test split.

Run from project root:
    python -m src.finetune_existing_checkpoint
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

from src.common import ROOT, set_seed
from src.data import get_split, make_dataset


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(ROOT)

ORIGINAL_MODEL_PATH = (
    ROOT / "artifacts" /
    "parkinson_effnetv2s_classweighted.keras"
)

NEW_MODEL_PATH = (
    ROOT / "artifacts" /
    "parkinson_effnetv2s_finetuned_extra.keras"
)

HISTORY_PATH = (
    ROOT / "artifacts" /
    "finetuned_extra_history.json"
)

CSV_LOG_PATH = (
    ROOT / "artifacts" /
    "finetuned_extra_training_log.csv"
)

MAX_EPOCHS = 20
LEARNING_RATE = 5e-6
UNFREEZE_LAST_LAYERS = 30

EARLY_STOPPING_PATIENCE = 5
LR_PATIENCE = 2
MIN_LEARNING_RATE = 1e-7

SEED = 42

set_seed(SEED)


# ============================================================
# 2. CHECK FILES
# ============================================================

print("\n" + "=" * 70)
print("CONTINUED EFFICIENTNETV2-S FINE-TUNING")
print("=" * 70)

if not ORIGINAL_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Original checkpoint not found:\n{ORIGINAL_MODEL_PATH}"
    )

if NEW_MODEL_PATH.exists():
    raise FileExistsError(
        f"\nNew checkpoint already exists:\n{NEW_MODEL_PATH}\n"
        "Rename or back up that file before running another experiment. "
        "The script will not overwrite it."
    )

if HISTORY_PATH.exists() or CSV_LOG_PATH.exists():
    raise FileExistsError(
        "\nA history or CSV log from this experiment already exists. "
        "Rename or back up the existing output files before rerunning."
    )

print("Original checkpoint:", ORIGINAL_MODEL_PATH)
print("New checkpoint     :", NEW_MODEL_PATH)
print("Maximum epochs     :", MAX_EPOCHS)
print("Learning rate      :", LEARNING_RATE)


# ============================================================
# 3. LOAD TRAINING AND VALIDATION DATA
# ============================================================

print("\nLoading training and validation data...")

train_df = get_split("train")
val_df = get_split("validation")

print("Training images  :", len(train_df))
print("Validation images:", len(val_df))

# The project's data loader uses label 0 = HEALTHY,
# label 1 = PARKINSON.

if "label" not in train_df.columns:
    raise KeyError(
        "The training dataframe does not contain a 'label' column. "
        f"Available columns: {list(train_df.columns)}"
    )

train_labels = train_df["label"].astype(int).to_numpy()

classes = np.unique(train_labels)

if not np.array_equal(classes, np.array([0, 1])):
    raise ValueError(
        f"Expected both classes 0 and 1 in training data. "
        f"Found: {classes}"
    )

# Calculate balanced class weights from the training split only.
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=train_labels,
)

class_weights = {
    0: float(class_weights_array[0]),
    1: float(class_weights_array[1]),
}

print("\nClass weights:")
print("Healthy (0)  :", class_weights[0])
print("Parkinson (1):", class_weights[1])

train_dataset = make_dataset(
    train_df,
    training=True,
)

val_dataset = make_dataset(
    val_df,
    training=False,
)


# ============================================================
# 4. LOAD EXISTING CHECKPOINT
# ============================================================

print("\nLoading existing class-weighted checkpoint...")

model = tf.keras.models.load_model(
    ORIGINAL_MODEL_PATH,
    compile=False,
)

print("Original model loaded successfully.")
print("Model name:", model.name)


# ============================================================
# 5. FIND EFFICIENTNET BACKBONE
# ============================================================

backbone = None

for layer in model.layers:
    if (
        isinstance(layer, tf.keras.Model)
        and "efficientnet" in layer.name.lower()
    ):
        backbone = layer
        break

if backbone is None:
    print("\nAvailable top-level layers:")
    for layer in model.layers:
        print(layer.name, type(layer).__name__)

    raise ValueError(
        "Could not identify the EfficientNetV2-S backbone. "
        "Check the printed model layers before continuing."
    )

print("\nBackbone identified:", backbone.name)


# ============================================================
# 6. CONFIGURE FINE-TUNING
# ============================================================

print("\nConfiguring trainable layers...")

# First freeze the entire backbone.
backbone.trainable = False

# Select the last 30 non-BatchNormalization layers.
eligible_layers = [
    layer
    for layer in backbone.layers
    if not isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    )
]

selected_layers = eligible_layers[-UNFREEZE_LAST_LAYERS:]

# Enable backbone training, then freeze everything individually.
backbone.trainable = True

for layer in backbone.layers:
    layer.trainable = False

# Unfreeze only the selected final layers.
for layer in selected_layers:
    layer.trainable = True

# Keep all BatchNormalization layers frozen.
for layer in backbone.layers:
    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False

# Keep the classifier/head trainable.
for layer in model.layers:
    if layer is not backbone:
        layer.trainable = True

# Explicitly protect any BatchNormalization layers at top level too.
for layer in model.layers:
    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):
        layer.trainable = False

trainable_count = sum(
    int(np.prod(weight.shape))
    for weight in model.trainable_weights
)

non_trainable_count = sum(
    int(np.prod(weight.shape))
    for weight in model.non_trainable_weights
)

print("Unfrozen backbone layers:", len(selected_layers))
print("Trainable parameters    :", trainable_count)
print("Non-trainable parameters:", non_trainable_count)

if trainable_count == 0:
    raise RuntimeError("No trainable model parameters were found.")


# ============================================================
# 7. COMPILE MODEL
# ============================================================

print("\nCompiling model...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)


# ============================================================
# 8. CALLBACKS
# ============================================================

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(NEW_MODEL_PATH),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=LR_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
        verbose=1,
    ),

    tf.keras.callbacks.CSVLogger(
        filename=str(CSV_LOG_PATH),
        append=False,
    ),
]


# ============================================================
# 9. START CONTINUED TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING CONTINUED TRAINING")
print("=" * 70)

print("Starting checkpoint :", ORIGINAL_MODEL_PATH.name)
print("Maximum epochs      :", MAX_EPOCHS)
print("Learning rate       :", LEARNING_RATE)
print("Early stopping      :", EARLY_STOPPING_PATIENCE)
print("Batch normalization : Frozen")
print("Test split          : Not used")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=MAX_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)


# ============================================================
# 10. SAVE TRAINING HISTORY
# ============================================================

history_data = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

history_data["experiment"] = {
    "starting_checkpoint": ORIGINAL_MODEL_PATH.name,
    "new_checkpoint": NEW_MODEL_PATH.name,
    "maximum_epochs": MAX_EPOCHS,
    "epochs_completed": len(history.history["loss"]),
    "learning_rate": LEARNING_RATE,
    "unfrozen_backbone_layers": len(selected_layers),
    "batch_normalization_frozen": True,
    "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    "class_weights": class_weights,
}

with open(
    HISTORY_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(history_data, file, indent=4)


# ============================================================
# 11. DISPLAY TRAINING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CONTINUED TRAINING SUMMARY")
print("=" * 70)

epochs_completed = len(history.history["loss"])

best_epoch = (
    int(np.argmin(history.history["val_loss"])) + 1
)

print("Epochs completed:", epochs_completed)
print("Best validation-loss epoch:", best_epoch)

for metric in [
    "accuracy",
    "val_accuracy",
    "loss",
    "val_loss",
    "auc",
    "val_auc",
    "precision",
    "val_precision",
    "recall",
    "val_recall",
]:
    if metric in history.history:
        values = history.history[metric]

        print(
            f"{metric:<18}: "
            f"{values[-1]:.4f} "
            f"(best recorded: {max(values) if 'loss' not in metric else min(values):.4f})"
        )

print("\nNew best checkpoint:")
print(NEW_MODEL_PATH)

print("\nTraining history:")
print(HISTORY_PATH)

print("\nCSV training log:")
print(CSV_LOG_PATH)

print("\nOriginal checkpoint was not modified.")
print("Continued fine-tuning completed.")
