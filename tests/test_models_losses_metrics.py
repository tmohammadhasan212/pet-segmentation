import pytest
import torch

from petseg.losses import masked_loss
from petseg.metrics import boundary_f1, region_metrics
from petseg.unet import ResNetUNet, UNet


@pytest.mark.parametrize("kind", ["scratch", "pretrained"])
@pytest.mark.parametrize("shape", [(64, 64), (65, 97)])
def test_output_shapes_and_backward(kind, shape):
    model = UNet(8) if kind == "scratch" else ResNetUNet(pretrained=False)
    model.eval()
    x = torch.randn(1, 3, *shape)
    logits = model(x)
    assert logits.shape == (1, 1, *shape)
    logits.mean().backward()
    assert model.head.weight.grad is not None


@pytest.mark.parametrize("kind", ["bce", "dice", "combined"])
def test_ignored_loss_invariance_and_gradient(kind):
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = torch.zeros_like(logits)
    valid = torch.ones_like(logits, dtype=torch.bool)
    valid[..., :3, :] = False
    loss = masked_loss(logits, target, valid, kind)
    changed = logits.detach().clone()
    changed[~valid] = 100
    changed_target = target.clone()
    changed_target[~valid] = 1
    assert torch.allclose(loss, masked_loss(changed, changed_target, valid, kind))
    loss.backward()
    assert torch.isfinite(loss)
    assert (logits.grad[~valid] == 0).all()
    assert masked_loss(logits, target, torch.zeros_like(valid), kind) == 0


def test_region_metrics_and_empty_masks():
    p = torch.tensor([[[1.0, 1.0], [0.0, 1.0]]])
    t = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    v = torch.tensor([[[True, True], [True, False]]])
    score = region_metrics(p, t, v)
    assert score["iou"] == pytest.approx(1 / 3)
    assert score["dice"] == 0.5 and score["precision"] == 0.5 and score["recall"] == 0.5
    p[~v], t[~v] = 0, 1
    assert region_metrics(p, t, v) == score
    z = torch.zeros(1, 8, 8)
    assert region_metrics(z, z, torch.ones_like(z))["iou"] == 1
    assert region_metrics(z, z, torch.zeros_like(z))["iou"] is None
    assert region_metrics(torch.ones_like(z), z, torch.ones_like(z))["iou"] == 0


def test_boundary_perfect_and_ignored():
    p = torch.zeros(1, 32, 32)
    p[:, 8:24, 8:24] = 1
    valid = torch.ones_like(p, dtype=torch.bool)
    assert boundary_f1(p, p, valid) == 1
    valid[:, :6] = False
    changed = p.clone()
    changed[~valid] = 1
    assert boundary_f1(changed, p, valid) == boundary_f1(p, p, valid)


@pytest.mark.parametrize("kind", ["bce", "dice", "combined"])
def test_half_precision_large_mask_reductions_are_finite(kind):
    logits = torch.zeros(1, 1, 256, 256, dtype=torch.float16, requires_grad=True)
    target = torch.ones_like(logits)
    valid = torch.ones_like(logits, dtype=torch.bool)
    loss = masked_loss(logits, target, valid, kind)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_encoder_learning_rate_groups_cover_every_parameter_once():
    from petseg.training import parameter_groups

    model = ResNetUNet(pretrained=False)
    groups = parameter_groups(
        model, {"model": "pretrained", "lr": 0.001, "encoder_lr_multiplier": 0.2}
    )
    assert [g["lr"] for g in groups] == [0.0002, 0.001]
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert id(model.stem[0].weight) in {id(p) for p in groups[0]["params"]}
    assert id(model.head.weight) in {id(p) for p in groups[1]["params"]}
