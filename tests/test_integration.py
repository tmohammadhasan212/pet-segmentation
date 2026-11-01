import json

import numpy as np
import pytest
import yaml
from PIL import Image

from petseg.data import prepare
from petseg.evaluation import comparison, evaluate, regenerate
from petseg.inference import Predictor, export, open_image
from petseg.training import train


@pytest.fixture
def trained(pets, tmp_path):
    root, manifests = pets
    prepare(root, manifests)
    config = dict(
        model="scratch",
        loss="combined",
        base_channels=8,
        data_root=str(root),
        manifests=str(manifests),
        size=64,
        seed=42,
        batch_size=2,
        epochs=1,
        patience=1,
        optimizer="adamw",
        lr=0.001,
        augment=True,
        device="cpu",
        amp=False,
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    checkpoint = train(path, tmp_path / "run")
    return root, manifests, checkpoint


def test_train_evaluate_regenerate_export(trained, tmp_path):
    root, manifests, checkpoint = trained
    output = tmp_path / "evaluation"
    evaluate(root, manifests, output, checkpoint, device="cpu")
    before = json.loads((output / "summary.json").read_text())
    hidden = root.with_name("temporarily-unavailable-dataset")
    root.rename(hidden)
    try:
        regenerate(output)  # Saved arrays must suffice when source photos are absent.
    finally:
        hidden.rename(root)
    assert json.loads((output / "summary.json").read_text()) == before
    assert (output / "gallery/Cat_6.png").exists()
    assert "iou" in comparison([output], tmp_path / "table.md").lower()
    image = Image.new("RGB", (173, 81), "orange")
    result = Predictor(checkpoint, "cpu").predict(image, 0.4)
    assert result["probability"].shape == (81, 173)
    assert result["cutout"].size == (173, 81)
    assert result["cutout"].mode == "RGBA"
    assert np.array_equal(
        np.asarray(result["cutout"])[..., 3], np.asarray(result["binary"])
    )
    export(result, tmp_path / "export")
    assert Image.open(tmp_path / "export/cutout.png").size == image.size
    assert np.load(tmp_path / "export/probability.npy").shape == (81, 173)
    for threshold in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            Predictor(checkpoint, "cpu").predict(image, threshold)


def test_bad_inputs(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_text("not an image")
    for value in (bad, 3, Image.new("RGB", (1, 1))):
        with pytest.raises(ValueError):
            open_image(value)


def test_demo_builds(trained):
    from petseg.demo import create_demo

    _, _, checkpoint = trained
    demo = create_demo(checkpoint, "cpu")
    assert demo is not None


def test_exif_orientation_preserved(tmp_path):
    image = Image.new("RGB", (100, 40), "orange")
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / "rotated.jpg"
    image.save(path, exif=exif)
    assert open_image(path).size == (40, 100)


def test_prediction_inverse_letterbox(trained):
    import torch

    _, _, checkpoint = trained
    predictor = Predictor(checkpoint, "cpu")

    class EchoForeground(torch.nn.Module):
        def forward(self, x):
            # Exactly known rectangular foreground in model coordinates.
            p = torch.full((1, 1, 64, 64), -20.0)
            p[:, :, 25:39, 16:48] = 20
            return p

    predictor.model = EchoForeground()
    image = Image.new("RGB", (200, 100), "white")
    result = predictor.predict(image)
    # Content is y=16..48 of the square; inverse must remove this padding.
    mask = np.asarray(result["binary"])
    ys, xs = np.where(mask > 0)
    assert abs(xs.min() - 50) < 4 and abs(xs.max() - 149) < 4
    assert abs(ys.min() - 28) < 4 and abs(ys.max() - 71) < 4


def test_warm_start_preserves_weights_and_contract(trained):
    import torch
    from petseg.training import initialize_model

    _, _, path = trained
    checkpoint = torch.load(path, weights_only=True)
    config = {**checkpoint["config"], "init_checkpoint": str(path), "size": 96}
    model, source = initialize_model(config, checkpoint["contract_hash"])
    assert source["sha256"] and source["epoch"] == 1
    for key, value in model.state_dict().items():
        assert torch.equal(value, checkpoint["model"][key])
    with pytest.raises(ValueError, match="contract"):
        initialize_model(config, "wrong-contract")
