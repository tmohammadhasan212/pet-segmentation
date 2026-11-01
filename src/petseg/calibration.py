"""Select inference resolution, flip averaging and thresholds on validation only."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import PetDataset, validate_splits
from .prediction import restore_probability
from .training import device_for, file_hash, load_checkpoint, save_checkpoint, sync


THRESHOLDS = np.array([i / 1000 for i in range(200, 801, 25)])


def threshold_scores(probability, target, valid, thresholds=THRESHOLDS):
    """Exact per-image region scores for a threshold grid, without giant broadcasts."""
    if not np.any(valid):
        return None
    valid, target = np.asarray(valid, dtype=bool), np.asarray(target, dtype=bool)
    grid = np.asarray(thresholds, dtype=probability.dtype)
    foreground = np.sort(probability[valid & target])
    background = np.sort(probability[valid & ~target])
    tp = len(foreground) - np.searchsorted(foreground, grid, side="left")
    fp = len(background) - np.searchsorted(background, grid, side="left")
    fn = len(foreground) - tp
    union, total = tp + fp + fn, 2 * tp + fp + fn
    iou = np.divide(tp, union, out=np.ones(len(thresholds)), where=union != 0)
    dice = np.divide(2 * tp, total, out=np.ones(len(thresholds)), where=total != 0)
    return iou, dice


@torch.inference_mode()
def calibrate(checkpoint, output, export_path=None, sizes=None, device="auto"):
    torch.set_num_threads(4)
    device = device_for(device)
    model, source = load_checkpoint(checkpoint, device)
    config = source["config"]
    root, manifests = config["data_root"], Path(config["manifests"])
    fingerprint = validate_splits(root, manifests)
    if source["contract_hash"] != fingerprint:
        raise ValueError(
            "Calibration checkpoint has a different split/statistics contract"
        )
    output = Path(output)
    if output.exists() or (export_path and Path(export_path).exists()):
        raise ValueError(
            "Choose new calibration/export paths; existing evidence is protected"
        )
    sizes = sorted(set(sizes or [config["size"]]))
    if any(size < 64 or size > 1024 for size in sizes):
        raise ValueError("Calibration sizes must be between 64 and 1024")
    candidates = []
    start = time.perf_counter()
    for size in sizes:
        dataset = PetDataset(root, manifests / "val.txt", source["stats"], size)
        loader = DataLoader(
            dataset,
            batch_size=8 if size <= 256 else 4 if size <= 384 else 2,
            shuffle=False,
        )
        sums = {
            False: np.zeros((2, len(THRESHOLDS))),
            True: np.zeros((2, len(THRESHOLDS))),
        }
        count = 0
        for batch in loader:
            x = batch["image"].to(device)
            ordinary = model(x).sigmoid()
            reflected = torch.flip(model(torch.flip(x, (-1,))).sigmoid(), (-1,))
            modes = {
                False: ordinary.cpu().numpy(),
                True: (0.5 * (ordinary + reflected)).cpu().numpy(),
            }
            for index, ident in enumerate(batch["id"]):
                with Image.open(
                    Path(root) / "annotations/trimaps" / f"{ident}.png"
                ) as image:
                    labels = np.asarray(image).copy()
                target, valid = labels == 1, labels != 3
                if not valid.any():
                    continue
                for flip, predictions in modes.items():
                    p = restore_probability(
                        predictions[index, 0],
                        batch["content_bounds"][index],
                        (labels.shape[1], labels.shape[0]),
                    )
                    sums[flip] += np.stack(threshold_scores(p, target, valid))
                count += 1
        if not count:
            raise ValueError("Validation has no valid pixels")
        for flip in (False, True):
            scores = sums[flip] / count
            for index, threshold in enumerate(THRESHOLDS):
                candidates.append(
                    {
                        "size": size,
                        "horizontal_flip_tta": flip,
                        "threshold": float(threshold),
                        "mean_iou": float(scores[0, index]),
                        "mean_dice": float(scores[1, index]),
                        "count": count,
                    }
                )
        print(f"calibrated size={size}, validation images={count}", flush=True)
    # Prefer thresholds near .5 and simpler inference when scores tie.
    selected = min(
        candidates,
        key=lambda c: (
            -c["mean_iou"],
            abs(c["threshold"] - 0.5),
            c["horizontal_flip_tta"],
            abs(c["size"] - config["size"]),
        ),
    )
    sync(device)
    report = {
        "split": "val",
        "metric_space": "original",
        "checkpoint_sha256": file_hash(checkpoint),
        "contract_hash": fingerprint,
        "validation_ids_sha256": hashlib.sha256(
            (manifests / "val.txt").read_bytes()
        ).hexdigest(),
        "device": str(device),
        "seconds": time.perf_counter() - start,
        "training_size": config["size"],
        "candidates": candidates,
        "selected": selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if export_path:
        payload = {
            key: source[key]
            for key in (
                "model",
                "config",
                "stats",
                "contract_hash",
                "epoch",
                "val_iou",
                "version",
                "initialization",
            )
            if key in source
        }
        payload.update(
            threshold=selected["threshold"],
            inference_size=selected["size"],
            horizontal_flip_tta=selected["horizontal_flip_tta"],
            calibration=report,
        )
        Path(export_path).parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(export_path, payload)
    return report
