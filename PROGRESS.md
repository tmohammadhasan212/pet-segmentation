# Progress report — completed 2026-11-03

The accuracy study resumed on 2026-11-02 and is complete. Both training stages,
validation-only calibration, frozen model selection, common-grid test comparison,
visual review and browser upload/download verification are finished.

## Final measured result

All 3,669 official test images were scored on their original annotation grid:

| System | Mean foreground IoU | Dice |
|---|---:|---:|
| One-epoch pretrained pilot | 0.8402687784 | 0.9025521015 |
| Selected refined model | **0.9676228798** | **0.9804842858** |

Mean IoU gain: **12.735 percentage points**. Small-pet IoU: 0.736248 → 0.949622.
3,491 images improved, eight tied and 170 worsened. These are repeated public
benchmark measurements: the official test was already inspected during the pilot.
No new test result chose model weights or inference settings.

## Selected model and running demo

- Checkpoint: `runs/accuracy/selected.pt` (local, Git-ignored).
- SHA-256: `6690cd31b0d0c00211a3fbd74d1614fdeb994d0bc5f898761a8d0d87911be865`.
- Selected stage: `refine-384`, best epoch 6.
- Inference: 384px letterbox, horizontal-flip probability averaging, threshold 0.475.
- Validation: native-grid mean IoU 0.9698066408 / Dice 0.9832429278, 736 images.
- Model selection was committed before new test evaluation (`5715e13`, `8658836`).
- Local demo: http://127.0.0.1:7860/ using the selected model on Apple MPS.
- Browser upload and actual PNG download verified: 375×500 RGBA, alpha 0/255,
  using validation photo `Abyssinian_10`. Screenshot and downloaded example are
  under `assets/accuracy/`; `reports/accuracy/browser-verification.json` records it.

Training and evaluation processes have completed. The demo remains running for
user use. Check process/port state before starting another server. Process/session
IDs are not durable across shutdown.

## Completed engineering and training

Proper `train --resume` restores model, optimizer, scheduler and scaler, validates
config/data contract, preserves history and starts at the correct continued epoch.
New checkpoints embed history and RNG/loader state, with atomic file updates.
An interrupted-vs-uninterrupted CPU integration test verifies identical weights,
history and scheduling. All **33 tests** and lint/format checks passed.

The 256px stage resumed from saved epoch 14 and completed the original 30-epoch
limit. Best epoch 27: input-grid validation IoU 0.9640818123. The 384px refinement
completed eight epochs; best epoch 6: input-grid validation IoU 0.9679561295.
The legacy epoch-14 checkpoint lacked RNG state, so exact random-sequence continuity
across that old pause cannot be claimed. Untouched original checkpoint/log copies
remain under `runs/accuracy/finetune-256/paused-epoch-14/`; historical pause evidence
is in `reports/accuracy-progress/` and Git commit `b94123e`.

Both stage-best checkpoints were calibrated on validation only: input sizes
256/384/512, plain/flip averaging, 25 thresholds from 0.2 to 0.8. Native validation
IoU selected the final system before new test evaluation. The model has 14,359,105
parameters. Warm prediction-pipeline mean latency over the same 50 validation
photos: selected 210.98 ms CPU (four threads) / 52.63 ms Apple MPS; pilot 21.14 /
11.29 ms. Timing details and raw measurements are saved.

## Evidence and limitations

- [Full report and commands](reports/accuracy-improvement.md)
- [Evidence index](reports/accuracy/README.md)
- [Result table](reports/accuracy/results.md), [distributions](reports/accuracy/results.png)
- [Training curves](reports/accuracy/training.png)
- [Selection provenance](reports/accuracy/selection.json)
- [20 reviewed cases](reports/accuracy/selected-native/gallery.html)
- [Per-example comparison](reports/accuracy/reviewed-comparison.csv)
- [Annotation anomalies](reports/accuracy/annotation-review.csv)

Full predictions remain under ignored `reports/generated/accuracy/`; weights,
dataset and environment remain outside Git. Committed metric snapshots regenerate
the table, distributions, paired bootstrap and breed comparison without weights.

Real regressions remain: `chihuahua_82` loses much of its body and `pomeranian_93`
retains framed background. Seven all-background test masks despite visible pets,
and one apparently inverted mask, are documented; **all remain in primary scores**.
Wide ignored contours make region scores forgiving; binary alpha does not recover
individual translucent hairs. This is an improvement over an undertrained pilot,
not a proven accuracy ceiling or a newly blinded generalization estimate.

No further training is needed to finish this study. Read `NEXT_SESSION.md` before
future continuation; preserve the selected model, source checkpoints and manifests.
