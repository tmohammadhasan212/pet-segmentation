"""Shared probability inference and exact inverse letterboxing."""

import numpy as np
import torch
from PIL import Image


def probabilities(model, image, horizontal_flip=False):
    probability = model(image).sigmoid()
    if horizontal_flip:
        flipped = model(torch.flip(image, (-1,))).sigmoid()
        probability = 0.5 * (probability + torch.flip(flipped, (-1,)))
    return probability


def restore_probability(probability, bounds, original_size):
    """Remove input padding, then restore probabilities to the original pixel grid."""
    restored = Image.fromarray(np.asarray(probability, dtype=np.float32)).crop(
        tuple(map(int, bounds))
    )
    restored = restored.resize(
        tuple(map(int, original_size)), Image.Resampling.BILINEAR
    )
    return np.clip(np.asarray(restored), 0, 1)
