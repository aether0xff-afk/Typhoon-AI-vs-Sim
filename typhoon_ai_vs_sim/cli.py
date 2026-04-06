from __future__ import annotations

import argparse
from pathlib import Path

from typhoon_ai_vs_sim.config import ExperimentConfig
from typhoon_ai_vs_sim.experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a typhoon intensity physics simulation with AI residual correction."
    )
    parser.add_argument("--storms", type=int, default=120, help="Number of synthetic storms.")
    parser.add_argument("--epochs", type=int, default=40, help="Maximum training epochs.")
    parser.add_argument("--window-size", type=int, default=6, help="Sequence length per sample.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where metrics, predictions, and plots will be saved.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller, faster experiment for smoke testing.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        seed=args.seed,
        n_storms=72 if args.quick else args.storms,
        epochs=25 if args.quick else args.epochs,
        window_size=args.window_size,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        checkpoint_dir=args.output_dir / "checkpoints",
    )
    run_experiment(config)
    return 0
