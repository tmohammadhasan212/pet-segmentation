# Accuracy study evidence

This folder stores small, version-controlled evidence from the follow-up accuracy
study. The earlier one-epoch experiment remains under [pilot](../pilot/README.md).
See [the study protocol](../accuracy-improvement.md) for exact commands and selection
rules. The fixed split has 2,944 train / 736 validation / 3,669 test images.

## Completed 256px stage

- `finetune-256-config.json`: exact training config.
- `finetune-256-history.json`: all 30 epochs; best is epoch 27, validation IoU
  0.9640818122702556 on the 256px model-input grid.
- `finetune-256-summary.json`: checkpoint hashes, split contract and pause/resume
  provenance. Epochs 15–30 resumed optimizer/scheduler state from epoch 14; that
  legacy checkpoint had no RNG state, preventing exact random-sequence continuity.
- `finetune-256-training.png`: train/validation loss, input-grid validation IoU
  and learning-rate schedule.
- `finetune-256-calibration.json`: all 150 validation inference candidates
  (3 sizes × 2 flip policies × 25 thresholds), scored on original-resolution
  masks. Selected: input 256px, flip averaging, threshold 0.525; validation mean
  IoU 0.9672406263111931 and Dice 0.9819132955332215.

| 256px-trained checkpoint input size | Best native validation IoU | Flip averaging | Threshold |
|---|---:|---|---:|
| 256 | 0.967241 | yes | 0.525 |
| 384 | 0.957562 | yes | 0.550 |
| 512 | 0.926663 | yes | 0.800 |

Larger inference inputs are not automatically better. The 384px training stage
was calibrated separately and selected using the same validation criterion.

## Completed refinement and final selection

`refine-384-config.json`, `refine-384-history.json` and
`refine-384-summary.json` record all eight refinement epochs. Epoch 6 was best:
input-grid validation IoU 0.9679561294661133. `training.png` plots both stages.
`refine-384-calibration.json` preserves all 150 native-grid validation candidates.

| Candidate | Native validation IoU | Dice | Inference input | Flip averaging | Threshold |
|---|---:|---:|---:|---|---:|
| Completed 256px model | 0.967241 | 0.981913 | 256 | yes | 0.525 |
| Refined 384px model, selected | **0.969807** | **0.983243** | 384 | yes | 0.475 |

`selection.json` freezes the winner and exported checkpoint SHA-256 before new
test evaluation (Git commits `5715e13`, `8658836`). The final inference checkpoint
is `runs/accuracy/selected.pt`; threshold, inference size and flip policy are
embedded and used automatically by prediction, evaluation and demo.

`*-prediction-benchmark.json` records the same first 50 validation photographs
on CPU and MPS, with training/calibration stopped and three warmups. Timing starts
after file decoding and includes preprocessing, selected model passes, CPU
transfer, inverse resizing and output construction; exports/browser/upload/model
loading are excluded. The explicitly named `concurrent-training` measurement is
preliminary evidence and excluded from final timing comparisons. Reproduce:

```bash
python scripts/benchmark_prediction.py --checkpoint runs/accuracy/selected.pt --device cpu --count 50 --output outputs/selected-cpu-benchmark.json
python scripts/benchmark_prediction.py --checkpoint runs/accuracy/selected.pt --device mps --count 50 --output outputs/selected-mps-benchmark.json
```

## Storage and interpretation

Weights are intentionally ignored under `runs/accuracy/`. Full generated
predictions belong under ignored `reports/generated/accuracy/`; the dataset and
local Python environment are also excluded from Git. Saved histories and JSON
calibration candidates can be inspected without those large artifacts.

Training-grid scores at different resolutions are not directly comparable. Model
selection uses native-grid validation IoU. Final test comparison must evaluate the
pilot and selected system on the same original annotation grid. Ignore trimap
label 3 in all denominators. The official test was inspected during the pilot,
so follow-up test measurements are repeated benchmarks rather than newly blinded
estimates. No new test scores have been used for tuning.
