# Accuracy improvement study

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
