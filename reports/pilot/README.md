# Real-data CPU pilot (2026-11-01)

Six one-epoch 128px experiments; **not a converged benchmark**. All runs use the
same locked 3,669-image official test set and fixed threshold 0.5. See
[the report](../experiment-report.md), [table](results.md), [distributions](results.png)
and [training logs](training.json). Per-run metadata includes parameter count,
checkpoint SHA-256, preprocessing size, split hash, runtime, CPU latency and RSS.

This small version-control snapshot retains full per-image CSVs, metadata and
summaries for all seven evaluations, and predictions/figures for the 20 curated
scratch-combined and pretrained-combined examples. The complete saved predictions
are local under `reports/generated/pilot/` and excluded from Git. Do not use
`--regenerate` on these partial prediction snapshots: that command requires all
3,669 predictions and intentionally checks the count.

Rebuild the table and distribution figure from the full CSV snapshots:

```bash
petseg report reports/pilot/ellipse reports/pilot/scratch-bce reports/pilot/scratch-dice reports/pilot/scratch-combined reports/pilot/pretrained-bce reports/pilot/pretrained-dice reports/pilot/pretrained-combined --output reports/pilot/results.md
petseg gallery reports/pilot/pretrained-combined --ids reports/gallery-ids.txt
petseg gallery reports/pilot/scratch-combined --ids reports/gallery-ids.txt
```

Recompute all original metrics from the complete local saved probabilities:

```bash
petseg report reports/generated/pilot/ellipse reports/generated/pilot/scratch-bce reports/generated/pilot/scratch-dice reports/generated/pilot/scratch-combined reports/generated/pilot/pretrained-bce reports/generated/pilot/pretrained-dice reports/generated/pilot/pretrained-combined --regenerate --output reports/generated/pilot/results.md
```

To reproduce the experiments from the official data and locked manifests:

```bash
python scripts/run_experiments.py --config-dir configs/pilot --runs-root runs/reproduced-pilot --report-root reports/generated/reproduced-pilot --device cpu
```

Gallery sources: Oxford-IIIT Pet, Parkhi et al., *Cats and Dogs*, CVPR 2012.
Original image owners retain copyright. Changes include resizing, masks,
probability heatmaps and cutouts. Dataset-derived image panels: CC BY-SA 4.0;
see [source, citation and terms](../../DATA.md). No trained weights or full dataset
are included in this small snapshot.
