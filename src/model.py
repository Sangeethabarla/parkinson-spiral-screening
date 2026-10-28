import tensorflow as tf
from src.common import load_config

def build_model():
    cfg = load_config(); size = tuple(cfg["image_size"])
    inputs = tf.keras.Input(shape=(*size,3), name="spiral_rgb_224")
    backbone = tf.keras.applications.EfficientNetV2S(include_top=False, weights="imagenet", input_shape=(*size,3), include_preprocessing=True)
    backbone.trainable = False
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(cfg["dropout_rate"], name="classifier_dropout")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="feature_dense")(x)
    x = tf.keras.layers.Dropout(cfg["dropout_rate"], name="mc_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pd_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="parkinsons_efficientnetv2s")
    return model, backbone
