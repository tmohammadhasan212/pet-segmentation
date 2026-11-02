import hashlib
import json
import math
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import PetDataset, audit, validate_splits
from .losses import masked_loss
from .metrics import region_metrics
from .unet import build_model


def device_for(request="auto"):
    if request == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    return torch.device(request)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if config.get("model") not in {"scratch", "pretrained"} or config.get(
        "loss"
    ) not in {"bce", "dice", "combined"}:
        raise ValueError(
            "Config needs model scratch/pretrained and loss bce/dice/combined"
        )
    for key in ("size", "batch_size", "epochs", "patience"):
        if not isinstance(config.get(key), int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["size"] < 64 or config["lr"] <= 0:
        raise ValueError("size must be >=64 and lr >0")
    if config.get("optimizer", "adamw") not in {"adamw", "sgd"}:
        raise ValueError("optimizer must be adamw or sgd")
    if config.get("scheduler", "cosine") not in {"cosine", "plateau", "none"}:
        raise ValueError("scheduler must be cosine, plateau or none")
    if not 0 < config.get("encoder_lr_multiplier", 1) <= 1:
        raise ValueError("encoder_lr_multiplier must lie in (0,1]")
    return config


def save_checkpoint(path, payload):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def rng_state(generator, device):
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "loader": generator.get_state(),
    }
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, generator, device):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    elif device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def initialize_model(config, fingerprint):
    """Warm-start weights only; each fine-tuning stage has its own optimizer/schedule."""
    source = config.get("init_checkpoint")
    model = build_model(config, initialize=not bool(source))
    provenance = None
    if source:
        checkpoint = torch.load(source, map_location="cpu", weights_only=True)
        if checkpoint["contract_hash"] != fingerprint:
            raise ValueError(
                "Initial checkpoint has a different split/statistics contract"
            )
        if checkpoint["config"]["model"] != config["model"]:
            raise ValueError("Initial checkpoint has a different model architecture")
        model.load_state_dict(checkpoint["model"], strict=True)
        provenance = {
            "path": str(source),
            "sha256": file_hash(source),
            "epoch": checkpoint["epoch"],
        }
    return model, provenance


def parameter_groups(model, config):
    multiplier = config.get("encoder_lr_multiplier", 1)
    if config["model"] != "pretrained" or multiplier == 1:
        return [{"params": list(model.parameters()), "lr": config["lr"], "name": "all"}]
    encoder, decoder = [], []
    for name, parameter in model.named_parameters():
        group = (
            encoder
            if name.startswith(("stem.", "layer1.", "layer2.", "layer3.", "layer4."))
            else decoder
        )
        group.append(parameter)
    return [
        {"params": encoder, "lr": config["lr"] * multiplier, "name": "encoder"},
        {"params": decoder, "lr": config["lr"], "name": "decoder"},
    ]


def train(config_path, output, resume=False):
    config = load_config(config_path)
    torch.set_num_threads(config.get("threads", 4))
    output = Path(output)
    if not resume and (output / "config.json").exists():
        raise ValueError(
            "Run directory already exists; choose a fresh output directory"
        )
    seed_everything(config["seed"])
    # Audit all official pairs, not only the examples visited by this run.
    audit_result = audit(config["data_root"])
    fingerprint = validate_splits(config["data_root"], config["manifests"])
    stats = json.loads((Path(config["manifests"]) / "stats.json").read_text())
    device = device_for(config.get("device", "auto"))
    checkpoint = None
    if resume:
        checkpoint = torch.load(
            output / "last.pt", map_location="cpu", weights_only=True
        )
        saved_config = json.loads((output / "config.json").read_text())
        runtime = {"device", "threads", "workers"}
        for previous in (saved_config, checkpoint["config"]):
            if {k: v for k, v in previous.items() if k not in runtime} != {
                k: v for k, v in config.items() if k not in runtime
            }:
                raise ValueError("Resume config differs from the saved training config")
        if checkpoint["contract_hash"] != fingerprint or checkpoint["stats"] != stats:
            raise ValueError("Resume split/statistics contract differs")
        model = build_model(config, initialize=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        initialization = checkpoint.get("initialization")
    else:
        model, initialization = initialize_model(config, fingerprint)
    model.to(device)
    sets = {
        split: PetDataset(
            config["data_root"],
            Path(config["manifests"]) / f"{split}.txt",
            stats,
            config["size"],
            split == "train" and config["augment"],
            config["seed"],
        )
        for split in ("train", "val")
    }
    generator = torch.Generator().manual_seed(config["seed"])
    loaders = {
        split: DataLoader(
            ds,
            batch_size=config["batch_size"],
            shuffle=split == "train",
            num_workers=config.get("workers", 0),
            generator=generator,
            pin_memory=device.type == "cuda",
        )
        for split, ds in sets.items()
    }
    if config.get("optimizer", "adamw") == "adamw":
        optimizer = torch.optim.AdamW(
            parameter_groups(model, config),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 1e-4),
        )
    else:
        optimizer = torch.optim.SGD(
            parameter_groups(model, config),
            lr=config["lr"],
            momentum=0.9,
            weight_decay=config.get("weight_decay", 1e-4),
        )
    schedule = config.get("scheduler", "cosine")
    if schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"]
        )
    elif schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=config.get("lr_patience", 2),
            min_lr=config.get("min_lr", 1e-6),
        )
    else:
        scheduler = None
    amp = config.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    output.mkdir(parents=True, exist_ok=True)
    best, stale, start = -math.inf, 0, 0
    history = []
    if resume:
        start = checkpoint["epoch"]
        history = checkpoint.get("history")
        if history is None:
            history = json.loads((output / "history.json").read_text())
        if [r["epoch"] for r in history] != list(range(1, start + 1)):
            raise ValueError("Resume history does not match checkpoint epoch")
        for row in history:
            if row["val_iou"] > best:
                best, stale = row["val_iou"], 0
            else:
                stale += 1
        previous_best = torch.load(
            output / "best.pt", map_location="cpu", weights_only=True
        )
        if (
            previous_best["val_iou"] != best
            or previous_best["contract_hash"] != fingerprint
        ):
            raise ValueError("Saved best checkpoint disagrees with resume history")
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler:
            scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        if checkpoint.get("rng_state"):
            restore_rng(checkpoint["rng_state"], generator, device)
        else:
            warnings.warn(
                "Legacy checkpoint has no RNG state; optimizer/schedule resume, "
                "but bitwise random-sequence continuity is unavailable."
            )
        save_json(output / "history.json", history)
        print(
            f"Resuming epoch {start + 1}; best={best:.8f}; stale={stale}; "
            f"learning_rates={[g['lr'] for g in optimizer.param_groups]}",
            flush=True,
        )
    else:
        save_json(output / "config.json", config)
        save_json(output / "audit.json", audit_result)
    if stale >= config["patience"]:
        return output / "best.pt"
    for epoch in range(start, config["epochs"]):
        sets["train"].epoch = epoch
        row = {
            "epoch": epoch + 1,
            "lr": max(g["lr"] for g in optimizer.param_groups),
            "group_lrs": {g["name"]: g["lr"] for g in optimizer.param_groups},
        }
        for split in ("train", "val"):
            model.train(split == "train")
            # Freeze running BN statistics for batch-size-one pretrained experiments.
            if config["model"] == "pretrained" and config["batch_size"] == 1:
                for layer in model.modules():
                    if isinstance(layer, torch.nn.BatchNorm2d):
                        layer.eval()
            losses, scores = [], []
            for step, batch in enumerate(loaders[split]):
                x, y, valid = (
                    batch[k].to(device) for k in ("image", "target", "valid")
                )
                optimizer.zero_grad(set_to_none=True)
                with (
                    torch.set_grad_enabled(split == "train"),
                    torch.autocast(device_type=device.type, enabled=amp),
                ):
                    logits = model(x)
                    loss = masked_loss(logits, y, valid, config["loss"])
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                if split == "train":
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                losses.extend([float(loss.detach())] * len(x))
                if split == "train" and (step + 1) % 50 == 0:
                    print(
                        f"epoch {epoch + 1}: train batch {step + 1}/{len(loaders[split])}",
                        flush=True,
                    )
                if split == "val":
                    # One batched transfer avoids many tiny accelerator synchronizations.
                    for p, t, v in zip(
                        logits.detach().sigmoid().cpu(), batch["target"], batch["valid"]
                    ):
                        score = region_metrics(p, t, v)["iou"]
                        if score is not None:
                            scores.append(score)
            row[f"{split}_loss"] = float(np.mean(losses))
            if split == "val":
                if not scores:
                    raise ValueError(
                        "Validation has no valid pixels; cannot select a checkpoint"
                    )
                row["val_iou"] = float(np.mean(scores))
        if schedule == "plateau":
            scheduler.step(row["val_iou"])
        elif scheduler:
            scheduler.step()
        improved = row["val_iou"] > best
        if improved:
            best, stale = row["val_iou"], 0
        else:
            stale += 1
        history.append(row)
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "scaler": scaler.state_dict(),
            "config": config,
            "stats": stats,
            "contract_hash": fingerprint,
            "epoch": epoch + 1,
            "val_iou": row["val_iou"],
            "threshold": 0.5,
            "version": "petseg-0.1.0",
            "initialization": initialization,
            "history": history,
            "rng_state": rng_state(generator, device),
        }
        if improved:
            save_checkpoint(output / "best.pt", payload)
        save_checkpoint(output / "last.pt", payload)
        save_json(output / "history.json", history)
        print(json.dumps(row), flush=True)
        if stale >= config["patience"]:
            break
    return output / "best.pt"


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["config"], initialize=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
