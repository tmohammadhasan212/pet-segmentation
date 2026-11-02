# Progress report — resumed 2026-11-02

The user resumed the accuracy study on 2026-11-02. Resume support is implemented
and committed as `dace3ed`; all 33 tests pass. Training has been launched from
logical epoch 15 with the saved optimizer/scheduler and original 30-epoch limit.
See `runs/accuracy/finetune-256/training.log` for live progress. The demo remains
stopped until final model selection. The historical pause state below is retained
as evidence; untouched copies are under the ignored `paused-epoch-14/` run folder.

## Implemented and committed

The repository is on branch `main`. It contains the complete data pipeline,
auditing and locked split manifests, handwritten scratch U-Net, ResNet18 encoder
U-Net, masked BCE/Dice losses, training/checkpointing, CLI cutouts, Gradio demo,
per-image evaluation, galleries, and reproducible reports. Dataset, environments,
checkpoints and large generated outputs are deliberately ignored by Git.

The original pilot trained all six model/loss combinations for one epoch at
128px. The pretrained combined-loss pilot measured **0.8402 mean IoU / 0.9025
Dice** on the official test set, scored on the 128px input grid. Its results are
preserved in [reports/pilot](reports/pilot/README.md).

This session added:

- Weight-only initialization from a prior checkpoint, with source hash and
  split/statistics contract checks.
- Separate encoder/decoder learning rates, ReduceLROnPlateau scheduling and
  faster validation transfers from the accelerator.
- Original-annotation-resolution evaluation (`petseg evaluate --native`).
- Validation-only threshold/resolution selection and horizontal-flip averaging
  (`petseg calibrate`); the selected settings are stored in exported checkpoints
  and used by prediction/evaluation/demo automatically.
- Learning-curve rendering from saved histories.

The last complete automated suite passed **32 tests**. Syntax/lint checks passed.
Apple MPS GPU acceleration was verified outside the Codex sandbox; the sandbox
itself reports MPS unavailable. The new training job used MPS float32.

## Exact paused training state

Run: `runs/accuracy/finetune-256/`
Config: `configs/accuracy/finetune-256.yaml`
Initialization: `runs/pilot/pretrained-combined/best.pt`

**Fourteen epochs are complete. Epoch fifteen was interrupted at the user's
request; its partial in-memory updates were discarded.** Both `best.pt` and
`last.pt` are complete, reloadable checkpoints from epoch fourteen.

| Measurement | Saved value |
|---|---:|
| Best epoch | 14 |
| Validation mean foreground IoU | **0.9561734281428408** |
| Training loss at epoch 14 | 0.029302835598608235 |
| Validation loss at epoch 14 | 0.0397806953912115 |
| Encoder learning rate for continuation | 0.00003 |
| Decoder learning rate for continuation | 0.00015 |
| Scheduler bad-epoch count | 0 |
| Maximum configured stage length | 30 epochs |
| Early-stopping patience | 6 epochs |

The 0.95617 score is **validation IoU on the 256px training grid**. It is not a
new test score and must not be presented as directly comparable to the pilot's
128px test metric. No new-model test evaluation has been performed.

Saved checkpoint state includes model, optimizer, scheduler, scaler, epoch,
config, normalization statistics and data contract. **True resume is now implemented** via `train --resume`. Legacy epoch-14
checkpoints lack RNG/loader state, so randomness across the old pause cannot be
reproduced exactly; subsequent checkpoints save these states and embed history.

The two checkpoint files are about 164 MiB each and remain on local disk across
shutdown; they are not in Git. Their hashes and paths are recorded in
[paused-state.json](reports/accuracy-progress/paused-state.json). Preserve the
local `runs/`, `data/oxford-pets/` and `.venv/` directories.

## Saved evidence

- [All 14 epoch records](reports/accuracy-progress/finetune-256-history.json)
- [Exact run config](reports/accuracy-progress/finetune-256-config.json)
- [Learning curves](reports/accuracy-progress/finetune-256-training.png)
- [Machine-readable paused state and checkpoint SHA-256s](reports/accuracy-progress/paused-state.json)
- Full console log: `runs/accuracy/finetune-256/training.log` (includes the
  intentional KeyboardInterrupt). Do not rely on `/tmp` surviving a reboot.
- [Improvement protocol and reproduction commands](reports/accuracy-improvement.md)

## Remaining work

1. Implement and test continuation from `last.pt`, restoring optimizer/scheduler
   state and appending history from epoch fifteen. Document the missing prior RNG
   state and save reproducibility state in future checkpoints.
2. Continue the 256px stage to its stopping rule, then run the configured 384px
   refinement from its validation-selected best checkpoint.
3. Calibrate inference size (256/384/512), threshold and flip averaging using
   validation images only. Select the final system using native-grid validation IoU.
4. Evaluate the selected system and the pilot on the same original-resolution
   test annotation grid; regenerate fair comparison tables and figures.
5. Review cutouts/errors, update demo and documentation, run relevant checks and
   commit the measured results. Keep the existing pilot evidence.

The published test has already been evaluated and inspected during the pilot.
Follow-up test scores are repeated benchmark measurements, not a newly blinded
estimate. All new tuning decisions must use the 736 validation images.

## Git history before this pause report

- `b6d1efb` — implementation, demo, configs and tests
- `4c3b3a0` — locked splits, pilot measurements and examples
- `e5248c7` — high-resolution fine-tuning support/configs
- `df85848` — validation calibration and native-grid evaluation
- `eeef36c` — training convergence plots

The user expects meaningful changes and results to be committed as work proceeds.
