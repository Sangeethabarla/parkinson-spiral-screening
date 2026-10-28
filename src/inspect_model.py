
from pathlib import Path
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "parkinson_effnetv2s_classweighted.keras"
)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

print("\n" + "=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

model.summary(expand_nested=False)

print("\n" + "=" * 70)
print("TOP-LEVEL LAYERS")
print("=" * 70)

for index, layer in enumerate(model.layers):
    print(
        f"{index:03d} | "
        f"{layer.name} | "
        f"{type(layer).__name__}"
    )

    if isinstance(layer, tf.keras.Model):
        print("     Nested model:", layer.name)

        for sublayer in layer.layers:
            if isinstance(
                sublayer,
                tf.keras.layers.Conv2D,
            ):
                print(
                    "       Conv2D:",
                    sublayer.name,
                    "| output:",
                    sublayer.output.shape,
                )

print("\nModel input:", model.input_shape)
print("Model output:", model.output_shape)
