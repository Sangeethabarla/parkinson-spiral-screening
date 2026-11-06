
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.common import ROOT, load_config, set_seed
from src.data import get_split, make_dataset
from src.model import build_model


# ============================================================
# 1. CONFIGURATION
# ============================================================

cfg = load_config()
set_seed(cfg["seed"])

ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

PHASE1_PATH = ARTIFACTS / "balanced_phase1_best.keras"
PHASE2_PATH = ARTIFACTS / "balanced_phase2_best.keras"
FINAL_PATH = ARTIFACTS / "parkinson_effnetv2s_balanced_best.keras"

HISTORY_PATH = ARTIFACTS / "balanced_training_history.json"

LOG1_PATH = ARTIFACTS / "balanced_phase1_log.csv"
LOG2_PATH = ARTIFACTS / "balanced_phase2_log.csv"

OUTPUT_FILES = [
    PHASE1_PATH,
    PHASE2_PATH,
    FINAL_PATH,
    HISTORY_PATH,
    LOG1_PATH,
    LOG2_PATH,
]

for path in OUTPUT_FILES:
    if path.exists():
        raise FileExistsError(
            f"Output already exists: {path}\n"
            "Back it up or rename it before running."
        )


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n========== LOADING BALANCED DATA ==========")

train_df = get_split("train")
val_df = get_split("validation")

train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df, training=False)

train_labels = train_df["label"].to_numpy(dtype=np.int32)

class_counts = np.bincount(train_labels, minlength=2)

if np.any(class_counts == 0):
    raise ValueError("Both classes must be present in training data.")

# Balanced class weights, calculated using training data only.
total = len(train_labels)

class_weights = {
    0: total / (2.0 * class_counts[0]),
    1: total / (2.0 * class_counts[1]),
}

class_weights = {
    key: float(value)
    for key, value in class_weights.items()
}

print("\nTraining class counts:")
print("Healthy:", class_counts[0])
print("Parkinson:", class_counts[1])

print("\nClass weights:")
print(class_weights)


# ============================================================
# 3. MODEL COMPILATION
# ============================================================

def compile_model(model, learning_rate):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
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
# 4. CALLBACKS
# ============================================================

def make_callbacks(checkpoint_path, log_path):

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),

        tf.keras.callbacks.CSVLogger(
            str(log_path),
        ),
    ]


# ============================================================
# 5. BUILD EFFICIENTNETV2S
# ============================================================

print("\n========== BUILDING EFFICIENTNETV2S ==========")

model, backbone = build_model()

print("Architecture:", model.name)
print("Backbone:", backbone.name)
print("Input shape:", model.input_shape)

compile_model(
    model,
    cfg["phase1_learning_rate"],
)


# ============================================================
# 6. PHASE 1: TRAIN CLASSIFIER HEAD
# ============================================================

print("\n========== PHASE 1: FROZEN BACKBONE ==========")

phase1_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=int(cfg["phase1_epochs"]),
    class_weight=class_weights,
    callbacks=make_callbacks(PHASE1_PATH, LOG1_PATH),
    verbose=1,
)

if not PHASE1_PATH.exists():
    raise RuntimeError("Phase 1 did not produce a checkpoint.")

print("\nBest Phase 1 checkpoint:", PHASE1_PATH)


# ============================================================
# 7. LOAD BEST PHASE 1 CHECKPOINT
# ============================================================

print("\n========== PREPARING FINE-TUNING ==========")

model = tf.keras.models.load_model(
    PHASE1_PATH,
    compile=False,
)

# Find the EfficientNetV2S backbone.
backbone = None

for layer in model.layers:
    if (
        isinstance(layer, tf.keras.Model)
        and "efficientnet" in layer.name.lower()
    ):
        backbone = layer
        break

if backbone is None:
    raise RuntimeError(
        "EfficientNetV2S backbone could not be identified."
    )

# Select the final configured number of non-BN layers.
eligible_layers = [
    layer
    for layer in backbone.layers
    if not isinstance(
        layer,
        tf.keras.layers.BatchNormalization,
    )
]

n_layers = int(cfg["fine_tune_last_layers"])

if n_layers <= 0:
    raise ValueError("fine_tune_last_layers must be positive.")

selected_layers = eligible_layers[-n_layers:]

# Freeze all layers first.
backbone.trainable = True

for layer in backbone.layers:
    layer.trainable = False

# Unfreeze selected final layers.
for layer in selected_layers:
    layer.trainable = True

# Keep BatchNormalization frozen.
for layer in backbone.layers:
    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization,
    ):
        layer.trainable = False

# Keep classifier layers trainable.
for layer in model.layers:
    if layer is not backbone:
        layer.trainable = True

print("Selected backbone layers:", len(selected_layers))

print(
    "Trainable parameters:",
    sum(
        int(np.prod(weight.shape))
        for weight in model.trainable_weights
    ),
)


# ============================================================
# 8. PHASE 2: FINE-TUNE BACKBONE
# ============================================================

print("\n========== PHASE 2: FINE-TUNING ==========")

compile_model(
    model,
    cfg["phase2_learning_rate"],
)

phase2_history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=int(cfg["phase2_epochs"]),
    class_weight=class_weights,
    callbacks=make_callbacks(PHASE2_PATH, LOG2_PATH),
    verbose=1,
)

if not PHASE2_PATH.exists():
    raise RuntimeError("Phase 2 did not produce a checkpoint.")


# ============================================================
# 9. SELECT BEST VALIDATION CHECKPOINT
# ============================================================

print("\n========== SELECTING BEST CHECKPOINT ==========")

candidates = []

for checkpoint in [PHASE1_PATH, PHASE2_PATH]:

    candidate = tf.keras.models.load_model(
        checkpoint,
        compile=False,
    )

    compile_model(
        candidate,
        cfg["phase2_learning_rate"],
    )

    metrics = candidate.evaluate(
        val_ds,
        return_dict=True,
        verbose=0,
    )

    val_auc = float(metrics["auc"])

    print(checkpoint.name, "Validation AUC:", val_auc)

    candidates.append((val_auc, checkpoint))

    del candidate


best_auc, best_checkpoint = max(
    candidates,
    key=lambda item: item[0],
)

best_model = tf.keras.models.load_model(
    best_checkpoint,
    compile=False,
)

best_model.save(FINAL_PATH)


# ============================================================
# 10. SAVE TRAINING HISTORY
# ============================================================

history = {
    "phase1": {
        key: [float(v) for v in values]
        for key, values in phase1_history.history.items()
    },
    "phase2": {
        key: [float(v) for v in values]
        for key, values in phase2_history.history.items()
    },
    "experiment": {
        "architecture": "EfficientNetV2S",
        "split_file": "dataset_split_balanced.csv",
        "seed": int(cfg["seed"]),
        "batch_size": int(cfg["batch_size"]),
        "image_size": cfg["image_size"],
        "class_weights": class_weights,
        "fine_tune_last_layers": len(selected_layers),
        "batch_normalization_frozen": True,
        "best_checkpoint": best_checkpoint.name,
        "best_validation_auc": best_auc,
    },
}

with open(HISTORY_PATH, "w", encoding="utf-8") as file:
    json.dump(history, file, indent=4)


# ============================================================
# 11. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("BALANCED EXPERIMENT COMPLETED")
print("=" * 60)

print("Architecture: EfficientNetV2S")
print("Best checkpoint:", best_checkpoint.name)
print("Best validation AUC:", round(best_auc, 4))
print("Final model:", FINAL_PATH)
print("Training history:", HISTORY_PATH)

print("\nTest data was NOT used for training or checkpoint selection.")
print("Previous checkpoints were not modified.")
