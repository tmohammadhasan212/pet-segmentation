> Follow-up completed 2026-11-03: the validation-selected refined model achieved
> 0.967623 IoU / 0.980484 Dice on the original-resolution test grid. See the
> [accuracy study](accuracy-improvement.md). This document preserves the original
> six-model, one-epoch pilot and its 128px scoring grid.

# Experimental protocol and completion record

## Status

All 7,349 official pairs passed audit. Locked manifests contain 2,944 training,
736 validation and 3,669 test images. **Six real-data one-epoch CPU pilot runs**
(scratch/pretrained × BCE/Dice/combined) have been trained and scored on the same
locked test set at 128×128 with fixed threshold 0.5. The full 256px, 60-epoch
study and repeated seeds have not been run. Local trained checkpoints are in
`runs/pilot/`; no weights are publicly published.

[Results table](pilot/results.md) · [IoU/Dice distributions](pilot/results.png) ·
[Pretrained gallery](pilot/pretrained-combined/gallery.html) ·
[Scratch gallery](pilot/scratch-combined/gallery.html) ·
[Reviewed categories/annotations](gallery-review.csv)

The combined pretrained pilot scores **0.8402 mean IoU and 0.9025 Dice**, versus
0.5975/0.7312 for combined scratch and 0.4762/0.6193 for the ellipse prior. IoU
standard deviations are 0.1606 (pretrained) and 0.1774 (scratch); their medians
are 0.8915 and 0.6077. Full per-image distributions, breed, size bins, precision,
recall and separate boundary F1 are saved in the CSVs and summaries.

The pilot uses scratch base channels 16 (1,964,097 parameters), compared with
14,359,105 parameters for the pretrained variant. Thus capacity, architecture
and initialization all differ. The pilot establishes that the implementation
learns and exports useful coarse masks; it cannot isolate pretraining's causal
benefit or establish a converged ranking among losses. The small scratch loss
ranking differences should not be interpreted without longer/repeated runs.

CPU forward latency averages 9.11 ms (combined scratch) and 16.44 ms (combined
pretrained), four PyTorch CPU threads on macOS arm64. Process peak RSS was
352.5 MB and 669.2 MB respectively, including framework/checkpoint overhead.
CUDA/MPS were not available to the training/evaluation process, so no GPU timing
or GPU memory measurements are claimed. Runtime versions/platform and exact
checkpoint hashes are stored in each `metadata.json`.

Recomputing the combined pretrained metrics from **all 3,669 saved probability
arrays** reproduced the original summary exactly. The CLI and browser upload
flow were exercised with held-out validation photograph `Abyssinian_10`; the
375×500 binary mask and RGBA cutout retain its original oriented dimensions.

## Visible failures and model differences

Twenty curated photographs and their annotations were visually reviewed, with
matching IDs in both galleries. Their tags are qualitative; summary size bins
use the exact foreground fraction of valid pixels.

- [Small fisheye dog, basset_hound_197](pilot/pretrained-combined/gallery/basset_hound_197.png):
  pretrained IoU 0.062, scratch 0.042. The pretrained mask retains parts of the
  structure rather than isolating the tiny dog. This is a clear failure despite
  strong average test scores.
- [Thin tail, Egyptian_Mau_99](pilot/pretrained-combined/gallery/Egyptian_Mau_99.png):
  pretrained IoU 0.668 versus scratch 0.347. The pretrained mask separates head
  and body but misses much of the thin, stretched tail; tail pixels are also
  heavily covered by the ignored annotation band.
- [Occluding sofa, saint_bernard_24](pilot/pretrained-combined/gallery/saint_bernard_24.png):
  pretrained IoU 0.430 versus scratch 0.278. The cutout includes substantial sofa
  foreground. Similar colors and occlusion remain challenging.
- [Multiple animals, american_bulldog_68](pilot/pretrained-combined/gallery/american_bulldog_68.png):
  the partial second animal at upper right is labeled background. The central
  dog has pretrained IoU 0.945 versus scratch 0.761; this score is not evidence
  that every visible pet is segmented.
- [Fur, samoyed_7](pilot/pretrained-combined/gallery/samoyed_7.png): pretrained
  IoU is 0.947, but the binary silhouette remains smooth and cannot recover
  translucent individual hairs. Ignored borders make region scores forgiving.
- [Running dog, english_cocker_spaniel_61](pilot/pretrained-combined/gallery/english_cocker_spaniel_61.png):
  pretrained IoU 0.995 versus scratch 0.812. The pose and soft paw/fur edges cover
  possible motion blur; capture metadata does not prove motion as the blur cause.
- [Uploaded validation example](../assets/example-comparison.png): the coarse
  mask retains a detached patch of floor. Images without pets may likewise
  produce false positives; this dataset does not validate reliable pet absence.

The pretrained combined model's small-foreground bin has IoU 0.7368, versus
0.8505 for medium and 0.8782 for large; corresponding scratch values are 0.4144,
0.6655 and 0.5996. Small subjects remain the weakest pretrained bin. Gallery
previews crop letterbox padding before presentation; unknown padding predictions
are never interpreted as foreground in the original-size cutout.

## Preregistered comparison

Run a central ellipse, then scratch U-Net and ImageNet-initialized ResNet18 U-Net,
each with masked BCE, masked soft Dice, and their equally weighted combination.
All six configs use seed 42, 256×256 letterboxing, the same train/validation/test
manifests, augmentations, AdamW at 3e-4, cosine decay, at most 60 epochs and
patience 10. The best validation macro IoU at threshold 0.5 selects each checkpoint.
Final threshold remains 0.5. Report one run per config first; repeated seeds are
needed before asserting small differences are reliable.

`python scripts/run_experiments.py` runs the complete matrix. All six models are trained and selected on validation before any final test
evaluation. Never use final test results to modify these preregistered configs. Device and checkpoint hash are
saved. Use `petseg report --regenerate ...` to reproduce metrics and figures from
saved predictions, without reloading a model or source images.

## Measurements

Primary metrics are **macro mean per-image** foreground IoU and Dice. Precision
and recall expose excessive foreground and missed regions. The summary includes
standard deviation, minimum, 10th/50th/90th percentiles and maximum; the CSV retains
the full distribution. Foreground-size bins are small <15%, medium 15–40%, large
≥40% of valid pixels; breed is retained per row. Boundary F1 uses a 2-pixel square
(Chebyshev) tolerance at the model input resolution and excludes a dilated region
around ignored pixels. Because the supplied ambiguous band removes much of the
true pet contour, this boundary measure has limited coverage and should not be
compared directly to region IoU or claimed as full-contour accuracy.

Benchmark each checkpoint separately on CPU and CUDA/MPS where available.
Batch-one latency excludes preprocessing, includes forward/sigmoid/device-to-CPU
transfer, uses three warmups and synchronizes accelerators. Demo timing includes
preprocessing and cutout construction. Peak process RSS is a process-wide high
water mark, not a model-only measurement; CUDA peak allocated memory is reset
after warmup. MPS reports ending allocation, not peak. Unavailable hardware
measurements remain absent rather than estimated.

## Error gallery review

Evaluation exports 20 lowest-IoU examples, an HTML gallery and an editable CSV.
Review the **original, labeled foreground and ignored regions** before assigning
an error to a model. Add curated IDs with `petseg gallery EVALUATION --ids FILE`
to cover fur, tails, multiple pets, similar backgrounds, small pets and motion
blur. Keep at least 20 unique examples overall. The automatic worst-IoU gallery
does not guarantee this category coverage. Fill all applicable tag columns and
`annotation_includes_all_animals`, then discuss whether a missed second animal
is a model error or an annotation limitation. Compare the same IDs across runs.

The observed pilot failures are linked above. The full study should test whether
longer training and higher resolution improve those same preregistered cases;
keep future conclusions separate from this one-epoch pilot.

## Completion checklist

- [x] Official dataset audited; locked manifests/statistics saved for version control (7,349 pairs).
- [x] Scratch and pretrained models trained on real images (one-epoch pilot).
- [x] All six model/loss pilots scored on the identical official test set.
- [x] Generated results table and per-image distribution figures inspected.
- [x] Twenty curated real examples reviewed, with category and annotation notes (blur cause remains uncertain).
- [x] Held-out-from-training photo checked in CLI and browser demo.
- [x] CPU latency and process memory recorded; accelerator unavailable.
- [ ] Full 256px, 60-epoch study and repeated seeds completed.

Use actual numbers only in a final CV claim. Tests, random weights, an ellipse
and synthetic images cannot support a claim of trained pet-segmentation quality.
