"""Per-image metrics; ignored pixels are excluded before every count."""

import numpy as np
import torch
from torch.nn import functional as F


def region_metrics(probability, target, valid, threshold=0.5):
    pred, target, valid = probability >= threshold, target.bool(), valid.bool()
    tp = int((pred & target & valid).sum())
    fp = int((pred & ~target & valid).sum())
    fn = int((~pred & target & valid).sum())
    n = int(valid.sum())
    if n == 0:
        return {
            k: None
            for k in ("iou", "dice", "precision", "recall", "foreground_fraction")
        }

    def ratio(num, den):
        return num / den if den else 1.0

    return {
        "iou": ratio(tp, tp + fp + fn),
        "dice": ratio(2 * tp, 2 * tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "foreground_fraction": int((target & valid).sum()) / n,
    }


def boundary_f1(probability, target, valid, threshold=0.5, tolerance=2):
    """Morphological boundary F1; exclude unknown pixels AND their tolerance neighborhood.

    Measured in resized input pixels; square/Chebyshev distance tolerance.
    """

    def dilate(x, r):
        return F.max_pool2d(x.float()[None, None], 2 * r + 1, stride=1, padding=r)[
            0, 0
        ].bool()

    p = (
        probability.squeeze().bool()
        if probability.dtype == torch.bool
        else probability.squeeze() >= threshold
    )
    t, v = target.squeeze().bool(), valid.squeeze().bool()
    # Boundaries adjacent to ignored border cannot be judged reliably.
    safe = ~dilate(~v, tolerance + 1)

    def edge(x):
        return (x & dilate(~x, 1)) & safe

    pe, te = edge(p), edge(t)
    np_, nt = int(pe.sum()), int(te.sum())
    if not safe.any():
        return None
    if not np_ or not nt:
        return 1.0 if np_ == nt else 0.0
    precision = int((pe & dilate(te, tolerance)).sum()) / np_
    recall = int((te & dilate(pe, tolerance)).sum()) / nt
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def summarize(rows):
    result = {}
    for metric in ("iou", "dice", "precision", "recall", "boundary_f1", "latency_ms"):
        values = [r[metric] for r in rows if r.get(metric) is not None]
        if values:
            a = np.asarray(values)
            result[metric] = {
                "mean": float(a.mean()),
                "std": float(a.std()),
                "min": float(a.min()),
                "p10": float(np.quantile(a, 0.1)),
                "median": float(np.median(a)),
                "p90": float(np.quantile(a, 0.9)),
                "max": float(a.max()),
            }
    for name, lo, hi in (
        ("small", 0, 0.15),
        ("medium", 0.15, 0.4),
        ("large", 0.4, 1.01),
    ):
        subset = [
            r["iou"]
            for r in rows
            if r.get("iou") is not None and lo <= r["foreground_fraction"] < hi
        ]
        result[f"size_{name}"] = {
            "n": len(subset),
            "iou": float(np.mean(subset)) if subset else None,
        }
    return result


def ellipse_prior(height, width):
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing="ij"
    )
    return (((x / 0.65) ** 2 + (y / 0.85) ** 2) <= 1).float().unsqueeze(0)
