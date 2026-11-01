"""Data auditing, immutable splits, training-only statistics and paired transforms."""

import hashlib
import json
import random
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

LABELS = {1: "foreground", 2: "background", 3: "ignored border"}


def read_ids(path):
    ids = [s.strip() for s in Path(path).read_text().splitlines() if s.strip()]
    if len(ids) != len(set(ids)) or any(
        not re.fullmatch(r"[A-Za-z_]+_\d+", s) for s in ids
    ):
        raise ValueError(f"Duplicate or unsafe image IDs in {path}")
    return ids


def official_ids(path):
    return [
        line.split()[0]
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


def download(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("annotations", "images"):
        archive = root / f"{name}.tar.gz"
        if not (root / name).exists():
            expected = {
                "annotations": "95a8c909bbe2e81eed6a22bccdf3f68f",
                "images": "5c4f3ee8e5d25df40f4fd59a7f44e54c",
            }[name]
            urls = [
                f"https://www.robots.ox.ac.uk/~vgg/data/pets/data/{name}.tar.gz",
                f"https://thor.robots.ox.ac.uk/pets/{name}.tar.gz",
                f"https://huggingface.co/datasets/choisw0823/datasets/resolve/main/oxford_pets/{name}.tar.gz?download=true",
            ]
            failures = []
            for url in urls:
                try:
                    print(f"Downloading {url}", flush=True)
                    with (
                        urllib.request.urlopen(url, timeout=30) as response,
                        archive.open("wb") as out,
                    ):
                        shutil.copyfileobj(response, out)
                    digest = hashlib.md5()
                    with archive.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    if digest.hexdigest() != expected:
                        raise ValueError("Archive checksum mismatch")
                    break
                except (OSError, ValueError) as e:
                    failures.append(f"{url}: {e}")
                    archive.unlink(missing_ok=True)
            else:
                raise RuntimeError(
                    "All dataset sources failed:\n" + "\n".join(failures)
                )
            with tarfile.open(archive) as tar:
                tar.extractall(root, filter="data")
            archive.unlink()


def audit(root, ids=None):
    root = Path(root)
    annotation_text = (root / "annotations/README").read_text()
    for label, term in ((1, "Foreground"), (2, "Background"), (3, "Not classified")):
        if not re.search(rf"{label}\s*[:=]\s*{term}", annotation_text, re.I):
            raise ValueError(f"Cannot verify label {label} from annotations/README")
    if ids is None:
        ids = official_ids(root / "annotations/trainval.txt") + official_ids(
            root / "annotations/test.txt"
        )
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in official splits")
    errors, counts = [], {str(k): 0 for k in LABELS}
    for ident in ids:
        try:
            with (
                Image.open(root / "images" / f"{ident}.jpg") as image,
                Image.open(root / "annotations/trimaps" / f"{ident}.png") as mask,
            ):
                image.load()
                mask.load()
                if image.size != mask.size:
                    raise ValueError(f"dimensions differ: {image.size} vs {mask.size}")
                values, n = np.unique(np.asarray(mask), return_counts=True)
                if not set(values).issubset(LABELS):
                    raise ValueError(f"unexpected labels {values.tolist()}")
                for v, c in zip(values, n):
                    counts[str(v)] += int(c)
        except (OSError, ValueError) as e:
            errors.append(f"{ident}: {e}")
    if errors:
        raise ValueError("Dataset audit failed:\n" + "\n".join(errors))
    return {
        "pairs": len(ids),
        "pixel_counts": counts,
        "mapping": LABELS,
        "annotation_readme_sha256": hashlib.sha256(
            annotation_text.encode()
        ).hexdigest(),
    }


def contract_hash(manifests):
    h = hashlib.sha256()
    for name in ("train.txt", "val.txt", "test.txt", "stats.json"):
        h.update(name.encode())
        h.update((Path(manifests) / name).read_bytes())
    return h.hexdigest()


def validate_splits(root, manifests):
    m = Path(manifests)
    sets = {s: set(read_ids(m / f"{s}.txt")) for s in ("train", "val", "test")}
    if any(not x for x in sets.values()) or any(
        sets[a] & sets[b]
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("Splits must be nonempty and disjoint")
    if sets["train"] | sets["val"] != set(
        official_ids(Path(root) / "annotations/trainval.txt")
    ):
        raise ValueError("Train/validation must partition official trainval")
    if sets["test"] != set(official_ids(Path(root) / "annotations/test.txt")):
        raise ValueError("Test manifest differs from official test split")
    stats = json.loads((m / "stats.json").read_text())
    train_hash = hashlib.sha256((m / "train.txt").read_bytes()).hexdigest()
    if stats["train_sha256"] != train_hash:
        raise ValueError("Normalization statistics do not match training manifest")
    return contract_hash(m)


def prepare(root, manifests, seed=42, val_fraction=0.2):
    root, manifests = Path(root), Path(manifests)
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    # Existing manifests are immutable: reruns verify them instead of resplitting.
    if (manifests / "stats.json").exists():
        validate_splits(root, manifests)
        return audit(root)
    if any((manifests / f"{s}.txt").exists() for s in ("train", "val", "test")):
        raise ValueError(
            "Partial manifests exist; inspect and remove them before preparing"
        )
    result = audit(root)
    annotation_text = (root / "annotations/README").read_text()
    # Stratify within breed; sort before seeded shuffle so filesystem order is irrelevant.
    groups = {}
    for ident in official_ids(root / "annotations/trainval.txt"):
        groups.setdefault(ident.rsplit("_", 1)[0], []).append(ident)
    train, val = [], []
    rng = random.Random(seed)
    for breed in sorted(groups):
        items = sorted(groups[breed])
        rng.shuffle(items)
        n = max(1, round(len(items) * val_fraction))
        val.extend(items[:n])
        train.extend(items[n:])
    manifests.mkdir(parents=True, exist_ok=True)
    for name, items in (
        ("train", train),
        ("val", val),
        ("test", official_ids(root / "annotations/test.txt")),
    ):
        (manifests / f"{name}.txt").write_text("\n".join(sorted(items)) + "\n")
    total, sums, squares = 0, np.zeros(3), np.zeros(3)
    for ident in train:
        with Image.open(root / "images" / f"{ident}.jpg") as im:
            im = im.convert("RGB")
            im.thumbnail((256, 256), Image.Resampling.BILINEAR)
            a = np.asarray(im, dtype=np.float64).reshape(-1, 3) / 255
        sums += a.sum(0)
        squares += (a * a).sum(0)
        total += len(a)
    mean = sums / total
    stats = {
        "mean": mean.tolist(),
        "std": np.sqrt(np.maximum(squares / total - mean**2, 1e-8)).tolist(),
        "seed": seed,
        "val_fraction": val_fraction,
        "method": "RGB pixels of train images, thumbnail <=256",
        "train_sha256": hashlib.sha256(
            (manifests / "train.txt").read_bytes()
        ).hexdigest(),
    }
    (manifests / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (manifests / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (manifests / "annotation-source.txt").write_text(annotation_text)
    validate_splits(root, manifests)
    return result


def content_bounds(original_size, size):
    w, h = original_size
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    left, top = (size - nw) // 2, (size - nh) // 2
    return left, top, left + nw, top + nh


def letterbox(image, size, fill=0, mask=False):
    """Aspect preserving fit; return content bounds for exact inverse mapping."""
    left, top, right, bottom = content_bounds(image.size, size)
    nw, nh = right - left, bottom - top
    interpolation = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
    resized = image.resize((nw, nh), interpolation)
    canvas = Image.new(image.mode, (size, size), fill)
    canvas.paste(resized, (left, top))
    return canvas, (left, top, left + nw, top + nh)


def paired_transform(image, mask, size, rng=None):
    image, _ = letterbox(image, size)
    mask, _ = letterbox(mask, size, fill=3, mask=True)
    if rng is not None:
        if rng.random() < 0.5:
            image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
        if rng.random() < 0.2:
            image, mask = ImageOps.flip(image), ImageOps.flip(mask)
        angle = rng.uniform(-15, 15)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=3)
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
        image = ImageEnhance.Color(image).enhance(rng.uniform(0.8, 1.2))
    return image, mask


def image_tensor(image, mean, std):
    x = (
        torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1)
        / 255
    )
    return (x - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]


class PetDataset(Dataset):
    def __init__(self, root, manifest, stats, size=256, augment=False, seed=42):
        self.root, self.ids, self.stats = Path(root), read_ids(manifest), stats
        self.size, self.augment, self.seed, self.epoch = size, augment, seed, 0

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ident = self.ids[index]
        with Image.open(self.root / "images" / f"{ident}.jpg") as im:
            image = im.convert("RGB")
        bounds = content_bounds(image.size, self.size)
        with Image.open(self.root / "annotations/trimaps" / f"{ident}.png") as im:
            mask = im.copy()
        rng = (
            random.Random(self.seed + self.epoch * len(self) + index)
            if self.augment
            else None
        )
        image, mask = paired_transform(image, mask, self.size, rng)
        labels = torch.from_numpy(np.asarray(mask).copy()).unsqueeze(0)
        return {
            "image": image_tensor(image, self.stats["mean"], self.stats["std"]),
            "target": (labels == 1).float(),
            "valid": labels != 3,
            "id": ident,
            "content_bounds": torch.tensor(bounds),
        }
