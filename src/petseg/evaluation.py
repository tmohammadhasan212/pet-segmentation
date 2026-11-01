import csv
import json
import os
import platform
import resource
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from .data import PetDataset, content_bounds, validate_splits
from .metrics import boundary_f1, ellipse_prior, region_metrics, summarize
from .prediction import probabilities, restore_probability
from .training import device_for, file_hash, load_checkpoint, sync


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def evaluate(
    root,
    manifests,
    output,
    checkpoint=None,
    split="test",
    device="auto",
    size=256,
    native=False,
):
    torch.set_num_threads(4)
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test")
    fingerprint = validate_splits(root, manifests)
    output = Path(output)
    if (output / "metadata.json").exists():
        raise ValueError("Evaluation directory already contains results")
    device = device_for(device)
    stats = json.loads((Path(manifests) / "stats.json").read_text())
    model, ckpt = (
        (None, None) if checkpoint is None else load_checkpoint(checkpoint, device)
    )
    if ckpt:
        if ckpt["contract_hash"] != fingerprint:
            raise ValueError(
                "Checkpoint belongs to different split/statistics contract"
            )
        size = ckpt.get("inference_size", ckpt["config"]["size"])
    flip = ckpt.get("horizontal_flip_tta", False) if ckpt else False
    threshold = ckpt.get("threshold", 0.5) if ckpt else 0.5
    ds = PetDataset(root, Path(manifests) / f"{split}.txt", stats, size)
    (output / "predictions").mkdir(parents=True, exist_ok=True)
    # Warmup excluded; latencies are batch-one forward pass plus sigmoid and device sync.
    if model:
        x = ds[0]["image"][None].to(device)
        for _ in range(3):
            probabilities(model, x, flip)
        sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
    rows = []
    for index, sample in enumerate(ds):
        x = sample["image"][None].to(device)
        sync(device)
        start = time.perf_counter()
        p = (
            probabilities(model, x, flip)[0].cpu()
            if model
            else ellipse_prior(size, size)
        )
        sync(device)
        elapsed = (time.perf_counter() - start) * 1000
        y, v = sample["target"], sample["valid"]
        rgb = (
            (
                (
                    sample["image"] * torch.tensor(stats["std"])[:, None, None]
                    + torch.tensor(stats["mean"])[:, None, None]
                ).clamp(0, 1)
                * 255
            )
            .byte()
            .permute(1, 2, 0)
            .numpy()
        )
        bounds = sample["content_bounds"].numpy()
        if native:
            with Image.open(Path(root) / "images" / f"{sample['id']}.jpg") as image:
                rgb = np.asarray(image.convert("RGB")).copy()
            with Image.open(
                Path(root) / "annotations/trimaps" / f"{sample['id']}.png"
            ) as image:
                labels = np.asarray(image).copy()
            original_size = (rgb.shape[1], rgb.shape[0])
            p = torch.from_numpy(
                restore_probability(p[0].numpy(), bounds, original_size).copy()
            )[None]
            y = torch.from_numpy(labels == 1)[None]
            v = torch.from_numpy(labels != 3)[None]
            bounds = np.array((0, 0, *original_size))
        row = {
            "id": sample["id"],
            "breed": sample["id"].rsplit("_", 1)[0],
            **region_metrics(p, y, v, threshold),
            "boundary_f1": boundary_f1(p, y, v, threshold),
            "latency_ms": elapsed,
        }
        rows.append(row)
        np.savez_compressed(
            output / "predictions" / f"{sample['id']}.npz",
            probability=p.numpy().astype(np.float32),
            target=y.numpy().astype(bool),
            valid=v.numpy(),
            image=rgb,
            content_bounds=bounds,
        )
        if (index + 1) % 500 == 0:
            print(f"evaluated {index + 1}/{len(ds)} images", flush=True)
    metadata = {
        "name": Path(checkpoint).parent.name if checkpoint else "ellipse",
        "split": split,
        "data_root": str(root),
        "contract_hash": fingerprint,
        "size": size,
        "threshold": threshold,
        "horizontal_flip_tta": flip,
        "metric_space": "original" if native else "input",
        "training_size": ckpt["config"]["size"] if ckpt else None,
        "device": str(device),
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "cpu_threads": torch.get_num_threads(),
            "accelerator": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "checkpoint_sha256": file_hash(checkpoint) if checkpoint else None,
        "config": ckpt["config"] if ckpt else {"model": "ellipse"},
        "parameters": sum(p.numel() for p in model.parameters()) if model else 0,
        "boundary_tolerance_pixels": 2,
        "count": len(rows),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "mps_allocated_bytes_end": torch.mps.current_allocated_memory()
        if device.type == "mps"
        else None,
        "latency_scope": "batch-one forward + sigmoid + selected flip TTA + device transfer to CPU; 3 warmups; preprocessing and native resizing excluded",
    }
    write_csv(output / "per_image.csv", rows)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (output / "summary.json").write_text(json.dumps(summarize(rows), indent=2))
    gallery(output)
    return summarize(rows)


def gallery(output, selected=None):
    output = Path(output)
    metadata = json.loads((output / "metadata.json").read_text())
    rows = list(csv.DictReader((output / "per_image.csv").open()))
    rows.sort(key=lambda r: float(r["iou"]) if r["iou"] else 1)
    selected = selected or [r["id"] for r in rows[:20]]
    folder = output / "gallery"
    folder.mkdir(exist_ok=True)
    tags = []
    for ident in selected:
        # Restrict identifiers to results, avoiding arbitrary path reads.
        row = next(r for r in rows if r["id"] == ident)
        with np.load(output / "predictions" / f"{ident}.npz") as a:
            p, t, v, rgb = (
                a["probability"].squeeze(),
                a["target"].squeeze(),
                a["valid"].squeeze(),
                a["image"],
            )
            bounds = (
                tuple(map(int, a["content_bounds"])) if "content_bounds" in a else None
            )
        if bounds is None:  # Compatibility with early pilot prediction archives.
            root = metadata.get(
                "data_root", metadata["config"].get("data_root", "data/oxford-pets")
            )
            with Image.open(Path(root) / "images" / f"{ident}.jpg") as original:
                bounds = content_bounds(original.size, metadata["size"])
        threshold = metadata["threshold"]
        w = h = min(256, max(rgb.shape[:2]))
        montage = Image.new("RGB", (w * 4, h + 30), "#202938")
        ground = np.stack((t * 255, t * 255, t * 255), -1).astype(np.uint8)
        ground[~v] = (200, 100, 200)
        heat = np.stack((p, p, 1 - p), -1)
        cutout = rgb.copy()
        cutout[p < threshold] = 230
        for i, panel in enumerate((rgb, ground, (heat * 255).astype(np.uint8), cutout)):
            content = ImageOps.contain(Image.fromarray(panel).crop(bounds), (w, h))
            frame = Image.new("RGB", (w, h), (230, 230, 230))
            frame.paste(content, ((w - content.width) // 2, (h - content.height) // 2))
            montage.paste(frame, (i * w, 30))
        score = f"{float(row['iou']):.3f}" if row["iou"] else "n/a"
        ImageDraw.Draw(montage).text(
            (5, 8),
            f"{ident} IoU={score} | original / truth (purple ignored) / probability / cutout",
            fill="white",
        )
        montage.save(folder / f"{ident}.png")
        tags.append(
            {
                "id": ident,
                "fur": "",
                "tail": "",
                "multiple_pets": "",
                "similar_background": "",
                "small_object": "",
                "motion_blur": "",
                "annotation_includes_all_animals": "",
                "notes": "",
            }
        )
    # Preserve human annotations and add rows for newly curated IDs.
    path = output / "gallery_tags.csv"
    if path.exists():
        with path.open() as stream:
            existing = list(csv.DictReader(stream))
        known = {row["id"] for row in existing}
        tags = existing + [row for row in tags if row["id"] not in known]
    write_csv(path, tags)
    html = "<html><meta charset='utf-8'><title>Error gallery</title><body><h1>Error gallery</h1><p>Original / truth / probability / cutout. Review gallery_tags.csv for category coverage and multi-animal annotation notes.</p>"
    html += "".join(
        f"<figure><img style='max-width:100%' src='gallery/{i}.png'><figcaption>{i}</figcaption></figure>"
        for i in selected
    )
    html += "<p>Oxford-IIIT Pet: Parkhi, Vedaldi, Zisserman and Jawahar, Cats and Dogs, CVPR 2012. Original image owners retain copyright. Derived panels: resize, masks and cutouts. <a href='https://www.robots.ox.ac.uk/~vgg/data/pets/'>Dataset source and terms</a>; <a href='https://creativecommons.org/licenses/by-sa/4.0/'>CC BY-SA 4.0</a>.</p>"
    (output / "gallery.html").write_text(html + "</body></html>")


def regenerate(output):
    """Recompute all region/boundary scores from saved predictions, without model/data."""
    output = Path(output)
    metadata = json.loads((output / "metadata.json").read_text())
    old = {r["id"]: r for r in csv.DictReader((output / "per_image.csv").open())}
    rows = []
    for path in sorted((output / "predictions").glob("*.npz")):
        with np.load(path) as a:
            p, y, v = (
                torch.from_numpy(a[k].astype(np.float32))
                for k in ("probability", "target", "valid")
            )
        rows.append(
            {
                "id": path.stem,
                "breed": path.stem.rsplit("_", 1)[0],
                **region_metrics(p, y, v, metadata["threshold"]),
                "boundary_f1": boundary_f1(p, y, v, metadata["threshold"]),
                "latency_ms": float(old[path.stem]["latency_ms"]),
            }
        )
    if len(rows) != metadata["count"]:
        raise ValueError("Saved prediction count differs from metadata")
    write_csv(output / "per_image.csv", rows)
    (output / "summary.json").write_text(json.dumps(summarize(rows), indent=2))
    gallery(output)


def comparison(directories, destination):
    items = [
        (
            Path(d),
            json.loads((Path(d) / "metadata.json").read_text()),
            json.loads((Path(d) / "summary.json").read_text()),
        )
        for d in directories
    ]
    contracts = {
        (
            m["contract_hash"],
            m["split"],
            m.get("metric_space", "input"),
            m["size"] if m.get("metric_space", "input") == "input" else None,
        )
        for _, m, _ in items
    }
    if len(contracts) != 1:
        raise ValueError(
            "Comparison requires identical contract, split and metric grid"
        )
    text = "| Run | Input px | Metric grid | Device | Parameters | IoU | Dice | Precision | Recall | Boundary F1 | Latency ms |\n|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for _, m, s in items:
        values = [
            s.get(k, {}).get("mean")
            for k in ("iou", "dice", "precision", "recall", "boundary_f1", "latency_ms")
        ]
        text += (
            f"| {m['name']} | {m['size']} | {m.get('metric_space', 'input')} | {m['device']} | {m['parameters']} | "
            + " | ".join("—" if v is None else f"{v:.4f}" for v in values)
            + " |\n"
        )
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text(text)
    distribution_figure(items, Path(destination).with_suffix(".png"))
    return text


def distribution_figure(items, destination):
    """Export full per-image IoU/Dice distributions using a noninteractive backend."""
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="petseg-matplotlib-")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    labels = [m["name"] for _, m, _ in items]
    for ax, metric in zip(axes, ("iou", "dice")):
        values = []
        for directory, _, _ in items:
            with (directory / "per_image.csv").open() as stream:
                values.append(
                    [float(r[metric]) for r in csv.DictReader(stream) if r[metric]]
                )
        ax.boxplot(values, tick_labels=labels, showmeans=True, showfliers=False)
        ax.set(
            ylim=(0, 1.02),
            ylabel=metric.upper(),
            title=f"Per-image {metric.upper()} distribution",
        )
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Held-out images · boxes: quartiles, line: median, triangle: mean · outliers hidden"
    )
    fig.savefig(destination, dpi=160)
    plt.close(fig)
