# Current continuation status — 2026-11-02

The user resumed work. Resume support is complete and committed as `dace3ed`
(33 passing tests), and the 256px run was launched with `--resume` from epoch 15.
Before starting another process, check for an existing training job and inspect
`runs/accuracy/finetune-256/training.log` and `history.json`. Do not run two jobs
against the same output. The original epoch-14 checkpoints/logs are preserved at
`runs/accuracy/finetune-256/paused-epoch-14/` with the hashes recorded below.
Continue the study protocol after the 256px stage stops. The older notes below
explain the original pause; the "first engineering task" is now complete.

# Resume note for the next Codex session

**User intent:** improve segmentation accuracy beyond the original one-epoch
pilot. The user then explicitly paused for a system shutdown and asked for this
handoff. Do not restart training until they ask to continue.

When they say “continue,” read `PROGRESS.md`, this file and
`reports/accuracy-progress/paused-state.json`, then inspect `git status`.
The existing Git repository is on `main`; do not initialize another repository.

## Start here

Working directory: `/Users/mohammd/Desktop/pet-segmentation`
Python environment: `.venv/bin/python` (Python 3.13; dependencies installed).
The entire Oxford-IIIT Pet dataset is already prepared and audited. Do not
redownload/resplit it. Manifests lock 2,944 train / 736 val / 3,669 test images.

All training/demo processes were intentionally stopped. Prior terminal session
IDs and `/tmp` logs are not reliable after shutdown. The persistent full training
log is `runs/accuracy/finetune-256/training.log`.

**Apple GPU works outside the Codex sandbox.** For GPU probes/training/calibration
use `functions.exec` -> `exec_command` with `sandbox_permissions:
"require_escalated"`; the sandbox falsely reports MPS unavailable. GPU work and
local training were authorized by the user's “do whatever you can” request.
Normal Terminal execution can also access MPS. Do not silently fall back to CPU
for the long job without checking this.

## Verify preserved work first

- `runs/accuracy/finetune-256/last.pt`: complete epoch 14, validation IoU
  `0.9561734281428408`; SHA-256
  `57a71b7f8bde313abdac6bc90bf7d6e1df61f683e75c4d8d788fcd866edefbc8`.
- `runs/accuracy/finetune-256/best.pt`: also epoch 14, same validation score;
  SHA-256 `80a2bcfc2b0319518e8ade575091b1c2afad758b08289b300fdf01c5c35334f7`.
- Original pilot: `runs/pilot/pretrained-combined/best.pt`; SHA-256
  `070175584262af20e4b6039a6161ed6e40b05864c2c494f06caf7d991a2f71c5`.
- Split/statistics contract:
  `0b081f28c49e40420d0bc739ec591819239966df60104055faef00e090e26a69`.

These files are local and intentionally Git-ignored. **Do not delete or overwrite
them as setup cleanup.** Load with `torch.load(..., map_location="cpu",
weights_only=True)` and verify epoch/config/hash before continuation.

## First engineering task: proper training resume

`src/petseg/training.py:train` currently rejects an existing run directory and
supports only weight-only warm-start (`init_checkpoint`). There is no working
`--resume` flag yet. Do not run a nonexistent command or merely warm-start from
scratch and claim optimizer-continuous resume.

Add a tested resume path to training/CLI that:

1. Loads `last.pt`, verifies config/data contract and restores model, optimizer,
   scheduler and scaler states onto the requested device.
2. Starts at zero-based epoch 14 (logical epoch 15), retaining/appending the
   existing 14 history records and reconstructing best/stale tracking.
3. Preserves reduced learning rates: encoder `3e-5`, decoder `1.5e-4`.
   ReduceLROnPlateau has `last_epoch=14`, `best=0.9561734281428408`,
   `num_bad_epochs=0`; early-stop stale count is zero because epoch 14 was best.
4. Continues the original maximum 30 stage epochs with patience 6. Do not
   accidentally train 30 *additional* epochs or reset the scheduler.
5. Sets dataset augmentation epoch to the continued global stage epoch, and saves
   RNG/loader-generator states in future checkpoints. Existing checkpoints lack
   these states, so bit-for-bit RNG continuity cannot be claimed for this pause.
   Be explicit about that limitation. Preserve seed 42 and deterministic pairing.
6. Protects the previous best checkpoint and history; use atomic checkpoint/log
   updates and meaningful tests for resume behavior.

The unfinished epoch fifteen was discarded. There is no need to recover its
partial gradients. Commit the tested resume support before the next long run.

## Then finish the accuracy study

- Resume `configs/accuracy/finetune-256.yaml` in the existing run safely after
  adding resume support. The current validation metric is on the 256px input
  grid, not a new test result.
- Train `configs/accuracy/refine-384.yaml` to
  `runs/accuracy/refine-384`. Its initializer already points at the 256px best.
- Use `petseg calibrate` on completed checkpoints, validation only. See exact
  commands in `reports/accuracy-improvement.md`. It tests input sizes 256/384/512,
  thresholds 0.2..0.8 by 0.025, with/without horizontal-flip averaging.
- Select the highest native-grid validation IoU candidate and copy its exported
  `calibrated.pt` to `runs/accuracy/selected.pt` **before** inspecting new test
  results. Exports include threshold, inference size and TTA policy.
- Evaluate pilot and selected model using `petseg evaluate --native` on the same
  original annotation grid. Preserve old pilot 128px metrics separately. The
  comparison helper rejects mixing input-grid and native metrics.
- Store small report evidence under version control, full predictions under
  ignored `reports/generated/accuracy/`, weights under ignored `runs/accuracy/`.
- Update examples/report/README and restart the demo with the selected model only
  after the user has resumed work. The prior demo used the pilot at port 7860.
  It is stopped now; any browser page left open is stale after shutdown.
- Validate actual browser upload/download behavior and original PNG dimensions,
  run relevant tests, and commit the completed results.

## Existing capabilities and guardrails

- Last complete suite: **32 passed** (`.venv/bin/python -m pytest -q`).
- `scripts/plot_training.py` regenerates stage loss/IoU/LR figures.
- Shared `src/petseg/prediction.py` handles flip alignment and inverse letterbox.
- `calibration.py` uses exact masked per-image threshold sweeps and only `val.txt`.
- New inference checkpoints can separate training size from `inference_size`.
- Normalization must remain training-only. Trimaps: 1 foreground, 2 background,
  3 ignored. Exclude borders/padding from loss and metrics. Never tune on test.
- The official test was already inspected during the pilot; report follow-up
  scores honestly as repeated benchmark measurements.
- Do not commit dataset, checkpoints, `.venv`, caches or full generated outputs.
- The user specifically asked why there were no commits: keep logical changes
  and measured evidence committed, while preserving their unrelated changes.

The task is **paused**, not complete. No 384px training, validation calibration,
or new-model test evaluation had run when this note was written.
