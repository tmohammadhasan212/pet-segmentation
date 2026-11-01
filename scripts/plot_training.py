"""Render recorded loss, validation IoU and learning rates for each training stage."""

import argparse
import json
import os
from pathlib import Path
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ.setdefault(
        "MPLCONFIGDIR", tempfile.mkdtemp(prefix="petseg-training-plot-")
    )
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(
        len(args.runs),
        3,
        figsize=(13, 4 * len(args.runs)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, directory in enumerate(args.runs):
        directory = Path(directory)
        history = json.loads((directory / "history.json").read_text())
        config = json.loads((directory / "config.json").read_text())
        epochs = [entry["epoch"] for entry in history]
        for name in ("train_loss", "val_loss"):
            axes[row, 0].plot(epochs, [entry[name] for entry in history], label=name)
        axes[row, 0].set(
            title=f"{directory.name} · {config['size']}px",
            xlabel="Stage epoch",
            ylabel="Loss",
        )
        axes[row, 0].legend()
        scores = [entry["val_iou"] for entry in history]
        best = max(range(len(scores)), key=scores.__getitem__)
        axes[row, 1].plot(epochs, scores)
        axes[row, 1].scatter(
            [epochs[best]],
            [scores[best]],
            color="darkgreen",
            label=f"best {scores[best]:.4f}",
        )
        axes[row, 1].set(
            xlabel="Stage epoch",
            ylabel="Validation mean IoU",
            title="Training-resolution grid",
        )
        axes[row, 1].legend()
        groups = history[0].get("group_lrs", {"all": history[0]["lr"]})
        for name in groups:
            axes[row, 2].plot(
                epochs,
                [
                    entry.get("group_lrs", {"all": entry["lr"]})[name]
                    for entry in history
                ],
                label=name,
            )
        axes[row, 2].set(
            xlabel="Stage epoch",
            ylabel="Learning rate",
            yscale="log",
            title="Optimizer schedule",
        )
        axes[row, 2].legend()
        for ax in axes[row]:
            ax.grid(alpha=0.2)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
