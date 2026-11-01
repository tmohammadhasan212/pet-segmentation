from torch.nn import functional as F


def masked_loss(logits, target, valid, kind="combined"):
    """Per-image macro loss. All-ignored images contribute no gradient or denominator."""
    if kind not in {"bce", "dice", "combined"}:
        raise ValueError(f"Unknown loss: {kind}")
    # CUDA autocast can yield half logits: sums over 256x256 pixels exceed fp16's
    # finite range. Keep loss arithmetic/reductions in fp32 while preserving grads.
    logits, target = logits.float(), target.float()
    valid = valid.to(logits.dtype)
    dims = tuple(range(1, logits.ndim))
    count = valid.sum(dims)
    bce = (
        F.binary_cross_entropy_with_logits(logits, target, reduction="none") * valid
    ).sum(dims) / count.clamp_min(1)
    probs = logits.sigmoid() * valid
    target = target * valid
    dice = 1 - (2 * (probs * target).sum(dims) + 1) / (
        probs.sum(dims) + target.sum(dims) + 1
    )
    loss = bce if kind == "bce" else dice if kind == "dice" else 0.5 * (bce + dice)
    weights = (count > 0).to(logits.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1)
