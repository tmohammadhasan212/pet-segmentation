import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture
def pets(tmp_path):
    root = tmp_path / "pets"
    (root / "images").mkdir(parents=True)
    (root / "annotations/trimaps").mkdir(parents=True)
    ids = [f"Cat_{i}" for i in range(1, 7)]
    for i, ident in enumerate(ids):
        mask = np.full((48, 80), 2, np.uint8)
        mask[10:38, 20:60] = 1
        mask[9, 20:60] = 3
        rgb = np.zeros((48, 80, 3), np.uint8)
        rgb[mask == 1] = (220 - i, 140, 100)
        Image.fromarray(rgb).save(root / "images" / f"{ident}.jpg")
        Image.fromarray(mask).save(root / "annotations/trimaps" / f"{ident}.png")
    (root / "annotations/trainval.txt").write_text(
        "\n".join(f"{i} 1 1 1" for i in ids[:5])
    )
    (root / "annotations/test.txt").write_text(f"{ids[-1]} 1 1 1\n")
    (root / "annotations/README").write_text(
        "1: Foreground\n2: Background\n3: Not classified\n"
    )
    return root, tmp_path / "manifests"
