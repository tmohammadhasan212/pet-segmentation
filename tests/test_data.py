import json
import random

import numpy as np
import pytest
from PIL import Image

from petseg.data import (
    PetDataset,
    audit,
    paired_transform,
    prepare,
    read_ids,
    validate_splits,
)


def test_pair_alignment_and_determinism():
    mask = np.full((83, 121), 2, np.uint8)
    mask[25:60, 35:90] = 1
    rgb = np.repeat(((mask == 1) * 255).astype(np.uint8)[..., None], 3, axis=2)
    a, b = paired_transform(
        Image.fromarray(rgb), Image.fromarray(mask), 128, random.Random(9)
    )
    c, d = paired_transform(
        Image.fromarray(rgb), Image.fromarray(mask), 128, random.Random(9)
    )
    assert np.array_equal(a, c) and np.array_equal(b, d)
    labels = np.asarray(b)
    predicted = np.asarray(a)[..., 0] > 100
    valid = labels != 3
    assert (predicted[valid] == (labels[valid] == 1)).mean() > 0.98
    assert set(np.unique(labels)) == {1, 2, 3}


def test_audit_and_locked_splits(pets):
    root, manifests = pets
    assert prepare(root, manifests)["pairs"] == 6
    before = validate_splits(root, manifests)
    prepare(root, manifests, seed=999)
    assert before == validate_splits(root, manifests)
    stats = json.loads((manifests / "stats.json").read_text())
    sample = PetDataset(root, manifests / "train.txt", stats, 64)[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["valid"].dtype.__str__() == "torch.bool"
    assert not sample["valid"][:, :10].any()  # ignored padding
    assert not set(read_ids(manifests / "test.txt")) & set(
        read_ids(manifests / "train.txt")
    )
    # Modifying only test images cannot change training-only statistics.
    test_id = read_ids(manifests / "test.txt")[0]
    Image.new("RGB", (80, 48), "white").save(root / "images" / f"{test_id}.jpg")
    fresh = manifests.parent / "fresh"
    prepare(root, fresh)
    assert json.loads((fresh / "stats.json").read_text())["mean"] == stats["mean"]


@pytest.mark.parametrize("failure", ["missing", "dimensions", "labels"])
def test_audit_rejects_bad_pairs(pets, failure):
    root, _ = pets
    path = root / "annotations/trimaps/Cat_1.png"
    if failure == "missing":
        path.unlink()
    else:
        Image.new(
            "L",
            (1, 1) if failure == "dimensions" else (80, 48),
            9 if failure == "labels" else 1,
        ).save(path)
    with pytest.raises(ValueError, match="audit failed"):
        audit(root)


def test_rejects_split_leakage(pets):
    root, manifests = pets
    prepare(root, manifests)
    (manifests / "val.txt").write_text((manifests / "train.txt").read_text())
    with pytest.raises(ValueError, match="disjoint"):
        validate_splits(root, manifests)


def test_source_mapping_is_verified(pets):
    root, manifests = pets
    (root / "annotations/README").write_text(
        "1: Background 2: Foreground 3: Not classified"
    )
    with pytest.raises(ValueError, match="verify label"):
        prepare(root, manifests)


def test_statistics_manifest_mismatch_is_rejected(pets):
    root, manifests = pets
    prepare(root, manifests)
    stats_path = manifests / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["train_sha256"] = "incorrect"
    stats_path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="statistics"):
        validate_splits(root, manifests)
