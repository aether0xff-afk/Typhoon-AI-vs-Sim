from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    test_metrics = metrics.loc[metrics["split"] == "test"].copy()
    test_metrics = test_metrics.sort_values("mae")

    colors = ["#455A64", "#1976D2", "#00897B", "#EF6C00"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(test_metrics["model"], test_metrics["mae"], color=colors[: len(test_metrics)])
    axes[0].set_title("Test MAE")
    axes[0].set_ylabel("m/s")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(
        test_metrics["model"], test_metrics["rmse"], color=colors[: len(test_metrics)]
    )
    axes[1].set_title("Test RMSE")
    axes[1].set_ylabel("m/s")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    for axis in axes:
        axis.tick_params(axis="x", rotation=20)

    fig.suptitle("Physics Baseline vs AI Residual Correction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sample_trajectories(
    storm_frame: pd.DataFrame,
    predictions: pd.DataFrame,
    output_path: Path,
    max_storms: int = 3,
) -> None:
    test_ids = (
        predictions.loc[predictions["split"] == "test", "storm_id"].drop_duplicates().tolist()
    )
    selected_ids = test_ids[:max_storms]

    if not selected_ids:
        return

    model_order = ["Physics", "MLP", "LSTM", "Transformer"]
    model_styles = {
        "Physics": ("#455A64", "--"),
        "MLP": ("#1976D2", "-."),
        "LSTM": ("#00897B", ":"),
        "Transformer": ("#EF6C00", "-"),
    }

    fig, axes = plt.subplots(len(selected_ids), 1, figsize=(12, 4 * len(selected_ids)))
    if len(selected_ids) == 1:
        axes = [axes]

    for axis, storm_id in zip(axes, selected_ids, strict=True):
        storm_rows = storm_frame.loc[storm_frame["storm_id"] == storm_id].sort_values("time_step")
        axis.plot(
            storm_rows["time_step"],
            storm_rows["intensity_ms"],
            color="#111111",
            linewidth=2.2,
            label="True intensity",
        )

        storm_predictions = predictions.loc[
            (predictions["storm_id"] == storm_id) & (predictions["split"] == "test")
        ]
        for model_name in model_order:
            model_rows = storm_predictions.loc[storm_predictions["model"] == model_name]
            if model_rows.empty:
                continue
            color, linestyle = model_styles[model_name]
            axis.plot(
                model_rows["target_step"],
                model_rows["prediction_ms"],
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=model_name,
            )

        axis.set_title(f"Trajectory comparison: {storm_id}")
        axis.set_xlabel("Time step")
        axis.set_ylabel("Intensity (m/s)")
        axis.grid(alpha=0.25)
        axis.legend(loc="best", ncol=3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: pd.DataFrame, output_path: Path) -> None:
    if history.empty:
        return

    fig, axes = plt.subplots(1, history["model"].nunique(), figsize=(15, 4), squeeze=False)
    axes = axes[0]

    for axis, (model_name, model_history) in zip(
        axes, history.groupby("model", sort=False), strict=True
    ):
        axis.plot(model_history["epoch"], model_history["train_loss"], label="Train", linewidth=2)
        axis.plot(model_history["epoch"], model_history["val_loss"], label="Validation", linewidth=2)
        axis.set_title(model_name)
        axis.set_xlabel("Epoch")
        axis.set_ylabel("SmoothL1 loss")
        axis.grid(alpha=0.25)
        axis.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
