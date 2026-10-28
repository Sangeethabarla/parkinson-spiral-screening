
from pathlib import Path
import json
import random

import numpy as np
import tensorflow as tf


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.json"


# ============================================================
# 2. CONFIGURATION
# ============================================================

def load_config():
    """Load project configuration from config.json."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# 3. REPRODUCIBILITY
# ============================================================

def set_seed(seed=None):
    """Set random seeds for reproducible experiments."""
    if seed is None:
        seed = load_config()["seed"]

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# 4. PATH RESOLUTION
# ============================================================

def resolve_path(value):
    """
    Resolve ordinary Python paths relative to the project root.

    Note:
    This function is intended for Python path values.
    Do not use it on symbolic TensorFlow tensors inside
    tf.data.Dataset.map().
    """
    p = Path(value)

    if p.is_absolute():
        return p

    return ROOT / p


# ============================================================
# 5. IMAGE LOADING AND PREPROCESSING
# ============================================================

def load_image(path, size=None):
    """
    Load a grayscale image and return a 3-channel RGB tensor.

    Supports:
    - Python string paths
    - pathlib.Path values
    - TensorFlow string tensors used by tf.data.Dataset.map()

    Preprocessing:
    - Decode as grayscale
    - Resize with antialiasing
    - Convert to float32
    - Convert grayscale to RGB
    - Preserve pixel values in the 0-255 range
    """

    # Read image dimensions from the project configuration.
    if size is None:
        size = tuple(load_config()["image_size"])

    # Convert only ordinary Python paths to absolute paths.
    # TensorFlow string tensors must be passed directly to
    # tf.io.read_file() without using pathlib.Path or str().
    if isinstance(path, (str, Path)):
        path = str(resolve_path(path))

    # Read the image file.
    image_bytes = tf.io.read_file(path)

    # Decode the image as a single grayscale channel.
    image = tf.io.decode_image(
        image_bytes,
        channels=1,
        expand_animations=False
    )

    # Ensure TensorFlow knows the channel dimension.
    image.set_shape([None, None, 1])

    # Resize to the configured image dimensions.
    image = tf.image.resize(
        image,
        size,
        antialias=True
    )

    # Use float32 while retaining the 0-255 pixel range.
    image = tf.cast(image, tf.float32)

    # Convert grayscale images to three channels.
    image = tf.image.grayscale_to_rgb(image)

    return image
