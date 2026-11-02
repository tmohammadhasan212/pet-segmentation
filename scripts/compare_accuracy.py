"""Regenerate a same-image accuracy comparison from saved metric snapshots."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from petseg.evaluation import comparison, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot")
    parser.add_argument("selected")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for directory in (args.pilot, args.selected):
        with (Path(directory) / "per_image.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        by_id = {r["id"]: r for r in rows}
        if len(by_id) != len(rows):
            raise ValueError("Duplicate prediction IDs")
        records.append(by_id)
    pilot, selected = records
    if set(pilot) != set(selected):
        raise ValueError("Paired comparison requires identical image IDs")
    comparison([args.pilot, args.selected], output / "results.md")
    ids = sorted(pilot)
    delta = np.array([float(selected[i]["iou"]) - float(pilot[i]["iou"]) for i in ids])
    rng = np.random.default_rng(42)
    means = np.array(
        [rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(2000)]
    )
    report = dict(
        count=len(ids),
        metric="per-image foreground IoU",
        mean_gain=float(delta.mean()),
        median_gain=float(np.median(delta)),
        improved=int((delta > 0).sum()),
        unchanged=int((delta == 0).sum()),
        worsened=int((delta < 0).sum()),
        bootstrap=dict(
            seed=42,
            resamples=2000,
            method="paired image percentile bootstrap",
            confidence=0.95,
            interval=np.quantile(means, [0.025, 0.975]).tolist(),
            limitation="Conditional on these fitted models and this previously inspected public benchmark; not training-run variability or a newly blind estimate.",
        ),
    )
    (output / "paired-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    breeds = []
    for breed in sorted({r["breed"] for r in pilot.values()}):
        subset = [i for i in ids if pilot[i]["breed"] == breed]
        a = np.mean([float(pilot[i]["iou"]) for i in subset])
        b = np.mean([float(selected[i]["iou"]) for i in subset])
        breeds.append(
            dict(
                breed=breed,
                count=len(subset),
                pilot_iou=float(a),
                selected_iou=float(b),
                gain=float(b - a),
            )
        )
    write_csv(output / "breed-comparison.csv", breeds)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
