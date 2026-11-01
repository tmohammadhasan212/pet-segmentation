import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from .data import image_tensor, letterbox
from .prediction import probabilities, restore_probability
from .training import device_for, file_hash, load_checkpoint, sync

MAX_PIXELS = 40_000_000


def open_image(source):
    try:
        if isinstance(source, Image.Image):
            image = source.copy()
        elif isinstance(source, (str, Path)):
            with Image.open(source) as im:
                if im.format not in {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}:
                    raise ValueError("Supported formats: JPEG, PNG, WEBP, BMP, TIFF")
                if im.width * im.height > MAX_PIXELS:
                    raise ValueError("Image exceeds 40 million pixels")
                if getattr(im, "n_frames", 1) != 1:
                    raise ValueError("Animated/multipage images are not supported")
                image = im.copy()
        else:
            raise ValueError("Input must be an image path or PIL image")
        if min(image.size) < 2 or image.width * image.height > MAX_PIXELS:
            raise ValueError(
                "Image must have at least 2 pixels per side and at most 40 million pixels"
            )
        return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as e:
        raise ValueError(f"Cannot read input image: {e}") from e


def make_outputs(image, probability, threshold):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must lie in [0,1]")
    binary = (probability >= threshold).astype(np.uint8) * 255
    rgba = np.dstack((np.asarray(image), binary))
    # Heatmap: blue low probability, yellow high probability.
    heat = np.stack((probability, probability, 1 - probability), axis=-1)
    return {
        "original": image,
        "probability": probability,
        "heatmap": Image.fromarray((heat * 255).astype(np.uint8)),
        "binary": Image.fromarray(binary),
        "cutout": Image.fromarray(rgba),
    }


class Predictor:
    def __init__(self, checkpoint, device="auto"):
        self.device = device_for(device)
        self.model, self.checkpoint = load_checkpoint(checkpoint, self.device)
        self.version = self.checkpoint["version"] + ":" + file_hash(checkpoint)[:12]

    @torch.inference_mode()
    def predict(self, source, threshold=None):
        image = open_image(source)
        threshold = (
            self.checkpoint.get("threshold", 0.5) if threshold is None else threshold
        )
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Threshold must lie in [0,1]")
        sync(self.device)
        start = time.perf_counter()
        size = self.checkpoint.get("inference_size", self.checkpoint["config"]["size"])
        canvas, bounds = letterbox(image, size)
        stats = self.checkpoint["stats"]
        x = image_tensor(canvas, stats["mean"], stats["std"])[None].to(self.device)
        p = (
            probabilities(
                self.model, x, self.checkpoint.get("horizontal_flip_tta", False)
            )[0, 0]
            .cpu()
            .numpy()
        )
        # Remove padding before restoring resolution, never stretch the entire square.
        p = restore_probability(p, bounds, image.size)
        outputs = make_outputs(image, p, threshold)
        sync(self.device)
        outputs.update(
            threshold=threshold,
            version=self.version,
            inference_ms=(time.perf_counter() - start) * 1000,
        )
        return outputs


def export(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "probability.npy", result["probability"])
    Image.fromarray((result["probability"] * 65535).round().astype(np.uint16)).save(
        directory / "probability.png"
    )
    result["heatmap"].save(directory / "heatmap.png")
    result["binary"].save(directory / "mask.png")
    result["cutout"].save(directory / "cutout.png")
    import json

    (directory / "prediction.json").write_text(
        json.dumps(
            {k: result[k] for k in ("threshold", "version", "inference_ms")}, indent=2
        )
    )
