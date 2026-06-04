#!/usr/bin/env python
"""Plot training curves from PyTorch Lightning TensorBoard logs."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_TAGS = [
    "train_loss",
    "train_ce_loss",
    "train_fisher_loss",
    "train_fisher_weighted",
    "val_loss",
    "val_ExpRate",
]


def resolve_logdir(logdir: str, log_root: Path) -> Path:
    path = Path(logdir)
    if path.exists():
        return path

    if logdir.isdigit():
        candidate = log_root / f"version_{logdir}"
        if candidate.exists():
            return candidate

    versions = sorted(
        log_root.glob("version_*"),
        key=lambda p: int(p.name.split("_", 1)[1]),
    )
    if not versions:
        raise FileNotFoundError(f"No runs found under {log_root}")
    return versions[-1]


def load_hparams(logdir: Path) -> dict:
    hparams_path = logdir / "hparams.yaml"
    if not hparams_path.exists():
        return {}
    with open(hparams_path) as f:
        return yaml.safe_load(f) or {}


def get_scalar_series(
    ea: EventAccumulator, tag: str
) -> Optional[Tuple[List[int], List[float]]]:
    if tag not in ea.Tags().get("scalars", []):
        return None
    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def fisher_weighted_series(
    ea: EventAccumulator, logdir: Path
) -> Optional[Tuple[List[int], List[float], str]]:
    direct = get_scalar_series(ea, "train_fisher_weighted")
    if direct is not None:
        return direct[0], direct[1], "train_fisher_weighted"

    fisher = get_scalar_series(ea, "train_fisher_loss")
    if fisher is None:
        return None

    hparams = load_hparams(logdir)
    lambda_fisher = hparams.get("lambda_fisher")
    if lambda_fisher is None:
        return None

    steps, vals = fisher
    weighted = [v * lambda_fisher for v in vals]
    label = f"train_fisher_loss * {lambda_fisher} (from hparams)"
    return steps, weighted, label


def plot_curves(logdir: Path, output: Path, tags: list) -> None:
    ea = EventAccumulator(str(logdir))
    ea.Reload()

    series: Dict[str, Tuple[List[int], List[float], str]] = {}
    for tag in tags:
        if tag == "train_fisher_weighted":
            computed = fisher_weighted_series(ea, logdir)
            if computed is not None:
                steps, vals, label = computed
                series[tag] = (steps, vals, label)
            continue
        data = get_scalar_series(ea, tag)
        if data is not None:
            series[tag] = (data[0], data[1], tag)

    fig, axes = plt.subplots(len(tags), 1, figsize=(8, 2.5 * len(tags)), sharex=True)
    if len(tags) == 1:
        axes = [axes]

    for ax, tag in zip(axes, tags):
        if tag not in series:
            ax.set_title(f"{tag} (not logged)")
            ax.axis("off")
            continue
        steps, vals, label = series[tag]
        ax.plot(steps, vals)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("step")
    fig.suptitle(str(logdir), fontsize=10)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    print(f"Saved plot to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logdir",
        default="11",
        help="Path to lightning_logs/version_N, version index (e.g. 6), or 'latest'",
    )
    parser.add_argument(
        "--log-root",
        default="lightning_logs",
        help="Root folder containing version_* runs",
    )
    parser.add_argument(
        "--output",
        default="training_curves.png",
        help="Output image path",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        default=DEFAULT_TAGS,
        help="TensorBoard scalar tags to plot",
    )
    args = parser.parse_args()

    log_root = Path(args.log_root)
    logdir = args.logdir
    if logdir == "latest":
        logdir = str(resolve_logdir("", log_root))
    else:
        logdir = str(resolve_logdir(logdir, log_root))

    plot_curves(Path(logdir), Path(args.output), args.tags)


if __name__ == "__main__":
    main()
