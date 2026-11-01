"""Train all six variants before evaluating the locked test set. Run from repo root."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Evaluation device; training device comes from configs",
    )
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--report-root", default="reports/generated")
    args = parser.parse_args()

    def run(*arguments):
        subprocess.run(
            [sys.executable, "-m", "petseg", *map(str, arguments)], check=True
        )

    configs, runs, report = (
        Path(args.config_dir),
        Path(args.runs_root),
        Path(args.report_root),
    )
    # Keep pretrained weight downloads within the ignored run directory.
    os.environ.setdefault("TORCH_HOME", str(Path("runs/torch-cache").resolve()))
    names = [
        f"{model}-{loss}"
        for model in ("scratch", "pretrained")
        for loss in ("bce", "dice", "combined")
    ]
    loaded = [yaml.safe_load((configs / f"{name}.yaml").read_text()) for name in names]
    contracts = {(c["data_root"], c["manifests"], c["size"]) for c in loaded}
    if len(contracts) != 1:
        raise ValueError("Matrix configs must share data_root, manifests and size")
    root, manifests, size = contracts.pop()
    for name in names:
        run("train", "--config", configs / f"{name}.yaml", "--output", runs / name)
    common = ["--data-root", root, "--manifests", manifests, "--device", args.device]
    run("evaluate", "--output", report / "ellipse", "--size", size, *common)
    destinations = [report / "ellipse"]
    for name in names:
        destination = report / name
        run(
            "evaluate",
            "--checkpoint",
            runs / name / "best.pt",
            "--output",
            destination,
            *common,
        )
        destinations.append(destination)
    run("report", *destinations, "--output", report / "results.md")


if __name__ == "__main__":
    main()
