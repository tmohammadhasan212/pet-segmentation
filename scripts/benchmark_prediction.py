"""Benchmark the full prediction pipeline on fixed validation photographs."""

import argparse
import json
import platform
from pathlib import Path
import resource
import sys

import numpy as np
import torch

from petseg.data import read_ids, validate_splits
from petseg.inference import Predictor
from petseg.training import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("count must be positive")
    output = Path(args.output)
    if output.exists():
        parser.error("choose a new output path; existing measurements are protected")
    torch.set_num_threads(4)
    predictor = Predictor(args.checkpoint, args.device)
    checkpoint = predictor.checkpoint
    config = checkpoint["config"]
    root, manifests = config["data_root"], config["manifests"]
    if validate_splits(root, manifests) != checkpoint["contract_hash"]:
        raise ValueError("Checkpoint split/statistics contract differs")
    ids = read_ids(Path(manifests) / "val.txt")[: args.count]
    photos = [Path(root) / "images" / f"{ident}.jpg" for ident in ids]
    for _ in range(3):
        predictor.predict(photos[0])
    rows = [
        {"id": ident, "ms": predictor.predict(photo)["inference_ms"]}
        for ident, photo in zip(ids, photos)
    ]
    times = [row["ms"] for row in rows]
    report = dict(
        checkpoint_sha256=file_hash(args.checkpoint),
        contract_hash=checkpoint["contract_hash"],
        split="val",
        count=len(rows),
        device=str(predictor.device),
        threads=4,
        platform=platform.platform(),
        torch=str(torch.__version__),
        inference_size=checkpoint.get("inference_size", config["size"]),
        horizontal_flip_tta=checkpoint.get("horizontal_flip_tta", False),
        threshold=checkpoint.get("threshold", 0.5),
        scope="Prediction after file decoding: preprocessing, forward/sigmoid, selected flip averaging, CPU transfer, inverse resize and output construction; excludes disk exports/browser/upload/model load; 3 warmups",
        mean_ms=float(np.mean(times)),
        median_ms=float(np.median(times)),
        p90_ms=float(np.quantile(times, 0.9)),
        per_image=rows,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"{args.device}: {report['mean_ms']:.2f} ms mean over {len(rows)} validation images"
    )


if __name__ == "__main__":
    main()
