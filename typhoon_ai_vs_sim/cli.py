from __future__ import annotations

import argparse
from pathlib import Path

from typhoon_ai_vs_sim.config import ExperimentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a typhoon intensity physics simulation with AI residual correction."
    )
    parser.add_argument(
        "--data-source",
        choices=("synthetic", "ibtracs"),
        default="ibtracs",
        help="Choose between synthetic storm generation and real IBTrACS best-track data.",
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
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory used to cache downloaded real-data files.",
    )
    parser.add_argument(
        "--ibtracs-start-year",
        type=int,
        default=2018,
        help="First season to include when using real IBTrACS tracks.",
    )
    parser.add_argument(
        "--ibtracs-end-year",
        type=int,
        default=2020,
        help="Last season to include when using real IBTrACS tracks.",
    )
    parser.add_argument(
        "--max-real-storms",
        type=int,
        default=24,
        help="Maximum number of real storms to load after filtering.",
    )
    parser.add_argument(
        "--min-real-track-steps",
        type=int,
        default=12,
        help="Minimum number of 6-hourly best-track points required per real storm.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller, faster experiment for smoke testing.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    quick_storms = 72 if args.quick else args.storms
    quick_epochs = 25 if args.quick else args.epochs
    quick_real_limit = min(args.max_real_storms, 8) if args.quick else args.max_real_storms
    config = ExperimentConfig(
        seed=args.seed,
        data_source=args.data_source,
        n_storms=quick_storms,
        epochs=quick_epochs,
        window_size=args.window_size,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        checkpoint_dir=args.output_dir / "checkpoints",
        data_dir=args.data_dir,
        ibtracs_start_year=args.ibtracs_start_year,
        ibtracs_end_year=args.ibtracs_end_year,
        max_real_storms=quick_real_limit,
        min_real_track_steps=args.min_real_track_steps,
    )
    from typhoon_ai_vs_sim.experiment import run_experiment

    run_experiment(config)
    return 0
