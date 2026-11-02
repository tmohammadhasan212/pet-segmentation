# Next-session reminder — completed study, 2026-11-03

Read `PROGRESS.md` and inspect `git status` first. The requested accuracy study is
**complete**; do not restart its training or recalibrate using test outcomes.

## Current deliverable

Working directory: `/Users/mohammd/Desktop/pet-segmentation`.
Python: `.venv/bin/python` (3.13). Existing local Git repository: `main`; preserve
unrelated changes and keep meaningful completed work committed. No remote has been
published. Weights, dataset, generated arrays and environments stay outside Git.

Selected model: `runs/accuracy/selected.pt`, SHA-256
`6690cd31b0d0c00211a3fbd74d1614fdeb994d0bc5f898761a8d0d87911be865`.
It embeds 384px inference, horizontal-flip probability averaging and threshold
0.475; the CLI, evaluator and demo load these automatically.

Final original-grid test scores: **IoU 0.9676228798 / Dice 0.9804842858** on all
3,669 images. Pilot on identical masks: 0.8402687784 / 0.9025521015. Selection was
frozen on validation first: IoU 0.9698066408 on 736 validation images. See
`reports/accuracy/selection.json` (committed before new test evaluation).

## Demo and reproducibility

The selected-model demo was left running at http://127.0.0.1:7860/. Check the port
and process before launching another instance. After shutdown it needs restarting:

```bash
.venv/bin/python -m petseg demo --checkpoint runs/accuracy/selected.pt --device mps
```

Apple MPS is available outside the Codex sandbox; sandbox probes can falsely say
it is unavailable. Use `exec_command` with `sandbox_permissions: require_escalated`
for MPS work, as before, or run from ordinary Terminal. Do not silently substitute
CPU for a long GPU job. GPU work was authorized in the accuracy request.
The demo upload/download was verified; `assets/accuracy/demo-screenshot.jpg` and
`reports/accuracy/browser-verification.json` record it. Downloaded PNG: 375×500 RGBA.

Rebuild small report evidence without weights/dataset:

```bash
.venv/bin/python scripts/compare_accuracy.py reports/accuracy/pilot-native reports/accuracy/selected-native --output outputs/accuracy-report
```

`reports/accuracy/` contains configs, full stage histories, checkpoint hashes, all
validation calibration candidates, test metadata/per-image CSVs, paired bootstrap,
breed comparison, training curves, 20 reviewed cases and annotation review.
`reports/accuracy-improvement.md` has methods, commands, results and limitations.
Full saved prediction arrays are only under `reports/generated/accuracy/`; metric
regeneration from probabilities requires those full arrays, not the small snapshot.

## Preserve this state

- Dataset already downloaded/audited: `data/oxford-pets/`. Do not resplit/redownload.
- Locked manifests: 2,944 train / 736 val / 3,669 test, seed 42, training-only stats.
- Split contract: `0b081f28c49e40420d0bc739ec591819239966df60104055faef00e090e26a69`.
- Pilot: `runs/pilot/pretrained-combined/best.pt`, hash
  `070175584262af20e4b6039a6161ed6e40b05864c2c494f06caf7d991a2f71c5`.
- Completed 256px run: `runs/accuracy/finetune-256/`, 30 epochs, best epoch 27,
  best hash `f973ca1312b9caced464f8bb6e40dc5ddc95d7a5f82866997f9e670481b31a75`.
- Completed 384px run: `runs/accuracy/refine-384/`, eight epochs, best epoch 6,
  best hash `92e7bc9ece64254cb1386086286a23803c15ed571758aec9fbc74f8e080b87a1`.
- Original epoch-14 weights/logs: `runs/accuracy/finetune-256/paused-epoch-14/`.
  Historical report evidence remains in `reports/accuracy-progress/` and Git `b94123e`.

Proper `train --resume` is implemented and tested: optimizer/scheduler/scaler,
config/contract checks, embedded history, RNG/loader state. It continues the
original epoch limit. The legacy epoch-14 checkpoint lacked RNG state, so exact
random-sequence continuity over that old pause cannot be claimed. New checkpoints
save it. Last suite: **33 tests pass**, plus lint/format checks.

## If the user asks for further improvement

Do not call this the best possible accuracy. Potential follow-up studies include
repeat seeds, genuinely new photos, better edge/alpha supervision and architectural
comparisons, each with a validation-only selection protocol and an explicit scope.
The current official test was previously inspected and now evaluated again; it is
reporting evidence, not a tuning set. Preserve all eight reviewed annotation
anomalies in primary scores; do not silently fix/exclude them to raise numbers.
Actual failures remain (missing Chihuahua body, Pomeranian background, dark
occlusion), and perfect region IoU does not establish perfect fur edges because
the ambiguous trimap border is ignored. No further work is required for the
completed study unless the user asks for a new follow-up.
