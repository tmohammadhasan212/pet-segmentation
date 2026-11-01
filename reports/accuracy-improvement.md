# Accuracy improvement study

**Paused 2026-11-01 at the user's request:** 14 completed 256px epochs; best
validation IoU 0.95617. Epoch fifteen was interrupted. No new-model test results
exist yet. See [the saved progress report](../PROGRESS.md) and
[resume instructions](../NEXT_SESSION.md).

The starting checkpoint is the one-epoch, 128px pretrained-combined pilot.
Its published input-grid test IoU is 0.8402. The original official test has already
been evaluated and visually inspected; follow-up scores are repeated benchmark
measurements, not a newly blinded generalization estimate.

## Registered approach

1. Fine-tune from the pilot at 256px, at most 30 epochs, patience 6. AdamW uses
   3e-4 for decoder/full-resolution skip and 6e-5 for pretrained encoder layers.
   Reduce both learning rates by half after validation plateaus (patience 2).
2. Refine the validation-selected 256px checkpoint at 384px, at most eight epochs,
   patience 4, decoder learning rate 5e-5 and encoder learning rate 1e-5.
3. Compare checkpoints, thresholds and horizontal-flip test-time averaging only
   on the 736-image validation split. Do not modify the locked split or normalization.
4. Compare the selected system against the pilot on a common metric grid at the
   original annotation dimensions. Input-grid metrics at different resolutions
   are not directly interchangeable.
5. Report the selected system once on the official 3,669-image test benchmark.
   Keep the older pilot evidence and checkpoints available for comparison.

The Apple MPS GPU is available outside the sandbox (the earlier pilot's process
could not access it). A 256px batch of eight took about 0.22 seconds after warmup.
No paid compute, cloud uploads, or extra datasets are used.

All hyperparameters are stored in `configs/accuracy/`. Checkpoints retain the
initial weight file's SHA-256 and split/statistics contract. Training logs and
best/last weights live under the ignored `runs/accuracy/` directory.

## Reproducible commands

```bash
petseg train --config configs/accuracy/finetune-256.yaml --output runs/accuracy/finetune-256
petseg train --config configs/accuracy/refine-384.yaml --output runs/accuracy/refine-384
petseg calibrate --checkpoint runs/accuracy/finetune-256/best.pt --sizes 256 384 512 --device mps --output runs/accuracy/finetune-256/calibration.json --export runs/accuracy/finetune-256/calibrated.pt
petseg calibrate --checkpoint runs/accuracy/refine-384/best.pt --sizes 256 384 512 --device mps --output runs/accuracy/refine-384/calibration.json --export runs/accuracy/refine-384/calibrated.pt
```

Calibration evaluates the original-resolution validation masks with 25 thresholds
from 0.2 to 0.8 in steps of 0.025, with and without horizontal-flip averaging.
At each of 256/384/512 input pixels, it calculates plain and flip-averaged
probabilities from two forward passes. Macro IoU selects the settings. The test
split cannot be passed to the calibration command. Exported inference checkpoints
contain selected threshold, inference size and flip-averaging policy; the CLI,
evaluator and demo all use those settings automatically. Training resolution is
retained separately, and the initial/source checkpoint hashes preserve provenance.

Compare only the validation-selected candidate on the final test benchmark and
restore the pilot's probabilities to the same original annotation grid:

```bash
petseg evaluate --checkpoint runs/pilot/pretrained-combined/best.pt --native --device mps --output reports/generated/accuracy/pilot-native
petseg evaluate --checkpoint runs/accuracy/selected.pt --native --device mps --output reports/generated/accuracy/selected-native
petseg report reports/generated/accuracy/pilot-native reports/generated/accuracy/selected-native --output reports/generated/accuracy/results.md
```

`selected.pt` is copied from whichever calibrated candidate has the highest
validation macro IoU, before either new test evaluation is examined. Native
scores are computed on original trimaps after undoing letterbox padding, with
all ambiguous border pixels excluded. The comparison code rejects mixing native
scores with the earlier model-input-grid results. Native boundary F1 uses a
two-original-pixel tolerance and remains separate from region metrics.

Regenerate the learning curves from the recorded stage histories:

```bash
python scripts/plot_training.py runs/accuracy/finetune-256 runs/accuracy/refine-384 --output reports/generated/accuracy/training.png
```
