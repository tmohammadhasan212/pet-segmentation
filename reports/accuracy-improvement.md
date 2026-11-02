# Accuracy improvement study

**Completed 2026-11-03.** The validation-selected refined model measured
**0.967623 mean foreground IoU / 0.980484 Dice** on all 3,669 official test images,
scored at original annotation resolution. The pilot measured **0.840269 IoU /
0.902552 Dice** on the same grid: **+12.735 percentage points IoU**.

| System | Input px | Threshold | Flip averaging | Test IoU | Dice | Precision | Recall |
|---|---:|---:|---|---:|---:|---:|---:|
| One-epoch pilot | 128 | 0.500 | no | 0.840269 | 0.902552 | 0.898080 | 0.930444 |
| Selected refined model | 384 | 0.475 | yes | **0.967623** | **0.980484** | **0.979575** | **0.986456** |

The selected model is `runs/accuracy/selected.pt`, SHA-256
`6690cd31b0d0c00211a3fbd74d1614fdeb994d0bc5f898761a8d0d87911be865`.
[Selection evidence](accuracy/selection.json) was committed (`5715e13`, `8658836`)
before either new test evaluation. It uses native-grid validation IoU alone:
0.969807 versus the completed 256px model's 0.967241, across 736 validation images.
No test outcome changed the model, threshold or inference policy.

## Measured improvement and practical limits

The paired image comparison improved 3,491 images, tied eight and worsened 170.
The mean IoU gain is 0.127354; its paired-image 95% percentile bootstrap interval
is [0.122600, 0.132127] (2,000 resamples, seed 42). This interval is conditional
on these fitted models and this previously inspected benchmark; it does not
measure training-seed variability or establish performance on newly collected data.
See [paired statistics](accuracy/paired-comparison.json),
[breed comparison](accuracy/breed-comparison.csv),
[full metric table](accuracy/results.md), and
[distributions](accuracy/results.png).

| Foreground fraction of valid pixels | Images | Pilot IoU | Selected IoU |
|---|---:|---:|---:|
| Small: <0.15 | 648 | 0.736248 | 0.949622 |
| Medium: 0.15–0.40 | 1,712 | 0.850633 | 0.970170 |
| Large: ≥0.40 | 1,309 | 0.878207 | 0.973203 |

Selected IoU: median 0.995212, 10th percentile 0.914480, standard deviation
0.088360. The wide ignored trimap contour excludes difficult edge pixels; an
IoU of 1 does not mean perfect fur, whiskers, tails or transparent alpha.
The separate two-original-pixel boundary F1 is 0.009245 for the pilot and 0.329051
for the selected model. Its ignored-pixel neighborhood removes much of the true
contour, and its empty-edge convention can return 0 or 1; it is not evidence of
high-quality hair matting. Inspect the visual cutouts as well as region scores.

[Twenty matched reviewed cases](accuracy/selected-native/gallery.html) retain the
pilot's category coverage and multi-animal annotation notes.
[Per-example comparisons](accuracy/reviewed-comparison.csv) show both gains and
regressions. The tiny fisheye dog `basset_hound_197` rises from 0.0621 to 0.9992;
`saint_bernard_24` rises from 0.4275 to 0.9653. In contrast, `chihuahua_82` falls
from 0.9300 to 0.5427 with much of its body missing, and `pomeranian_93` falls from
0.8890 to 0.7851 with framed background retained. `Bombay_71` remains difficult
at 0.7595 in a dark, occluded scene. These examples were reviewed for reporting;
they did not trigger further tuning.

The worst-score review found seven source trimaps containing only label 2
(background) despite visible pets, plus `Persian_273`, whose foreground/background
appear inverted relative to its visible cat. See
[annotation review](accuracy/annotation-review.csv) and the saved
[empty-mask example](accuracy/annotation-examples/Egyptian_Mau_20.png) /
[apparently inverted example](accuracy/annotation-examples/Persian_273.png).
**All eight remain in both 3,669-image primary scores.** No masks or manifests
were corrected, excluded or replaced.

## Training and inference evidence

The 256px stage completed its original 30-epoch limit; best epoch 27 scored
0.964082 input-grid validation IoU. The 384px stage completed eight epochs; best
epoch 6 scored 0.967956 on its own input grid. These different grids are not used
to select between stages. Calibration compared all 150 inference candidates per
stage on the same original-resolution validation masks. Training histories,
configs, hashes, all calibration candidates and [learning curves](accuracy/training.png)
are committed under [accuracy evidence](accuracy/README.md).

Proper optimizer/scheduler continuation is implemented and tested. The original
epoch-14 checkpoints/logs remain under the ignored
`runs/accuracy/finetune-256/paused-epoch-14/`. That legacy checkpoint lacks RNG
state, so exact random-sequence continuity across the old pause cannot be claimed.
New checkpoints save history and Python/NumPy/Torch/accelerator/loader RNG states.
The final suite passed **33 tests**, plus lint/format checks.

Both systems have 14,359,105 parameters. On the same first 50 validation photos,
with three warmups and training/calibration stopped:

| Prediction pipeline | Pilot mean ms | Selected mean ms |
|---|---:|---:|
| CPU, four threads | 21.14 | 210.98 |
| Apple MPS GPU | 11.29 | 52.63 |

These timings start after file decoding and include preprocessing, model passes,
selected flip averaging, CPU transfer, inverse resize and output construction.
They exclude model loading, file exports and browser/upload overhead. The native
test table separately measures forward/sigmoid/TTA/CPU transfer over all 3,669
images (13.62 / 62.57 ms). Peak process RSS during those evaluations was about
724 / 644 MiB; end-of-run Torch MPS allocations were 55.0 / 57.1 MiB, **not peak
GPU memory**. Raw metadata preserves scope and runtime details.

The demo now uses the selected checkpoint. A browser upload of validation photo
`Abyssinian_10` generated all four panels and working downloads; the downloaded
PNG was verified as **375×500 RGBA**, binary alpha 0/255. Its first, cold prediction
was 196.7 ms. [Browser evidence](accuracy/browser-verification.json),
[screenshot](../assets/accuracy/demo-screenshot.jpg), and
[original/pilot/selected comparison](../assets/accuracy/example-comparison.png)
are saved. This is an accuracy improvement over an undertrained pilot, not a claim
that this architecture or benchmark has reached its best possible result.

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
petseg train --config configs/accuracy/finetune-256.yaml --output runs/accuracy/finetune-256 --resume
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
