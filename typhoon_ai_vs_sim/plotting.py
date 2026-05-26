from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics(metrics: pd.DataFrame, output_path: Path) -> None:
    test_metrics = metrics.loc[metrics["split"] == "test"].copy()
    test_metrics = test_metrics.sort_values("mae")

    colors = ["#455A64", "#1976D2", "#00897B", "#EF6C00", "#7B1FA2", "#C2185B", "#5D4037"]
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

    fig.suptitle("Physics Baseline vs Direct and Hybrid AI Models")
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

    model_order = [
        "Physics",
        "Hybrid-MLP",
        "Hybrid-LSTM",
        "Hybrid-Transformer",
        "Direct-MLP",
        "Direct-LSTM",
        "Direct-Transformer",
    ]
    model_styles = {
        "Physics": ("#455A64", "--"),
        "Hybrid-MLP": ("#1976D2", "-."),
        "Hybrid-LSTM": ("#00897B", ":"),
        "Hybrid-Transformer": ("#EF6C00", "-"),
        "Direct-MLP": ("#7B1FA2", "-."),
        "Direct-LSTM": ("#C2185B", ":"),
        "Direct-Transformer": ("#5D4037", "-"),
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


def plot_diagnostic_relationships(
    predictions: pd.DataFrame,
    storm_frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    physics = predictions.loc[
        (predictions["split"] == "test") & (predictions["model"] == "Physics")
    ].copy()
    if physics.empty:
        return

    physics["residual_ms"] = physics["true_target_ms"] - physics["prediction_ms"]
    merged = physics.merge(
        storm_frame[
            [
                "storm_id",
                "time_step",
                "sst_c",
                "latitude_deg",
            ]
        ],
        left_on=["storm_id", "target_step"],
        right_on=["storm_id", "time_step"],
        how="left",
    )

    _plot_physics_vs_actual(physics, output_dir / "physics_vs_actual.png")
    _plot_residual_distribution(physics, output_dir / "residual_distribution.png")
    _plot_residual_scatter(
        merged,
        x_column="sst_c",
        x_label="SST (deg C)",
        output_path=output_dir / "sst_vs_residual.png",
    )
    _plot_residual_scatter(
        merged,
        x_column="latitude_deg",
        x_label="Latitude (deg)",
        output_path=output_dir / "latitude_vs_residual.png",
    )


def _plot_physics_vs_actual(physics: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(
        physics["true_target_ms"],
        physics["prediction_ms"],
        color="#1976D2",
        alpha=0.75,
        edgecolor="none",
    )
    lower = min(physics["true_target_ms"].min(), physics["prediction_ms"].min())
    upper = max(physics["true_target_ms"].max(), physics["prediction_ms"].max())
    axis.plot([lower, upper], [lower, upper], color="#111111", linestyle="--", linewidth=1.5)
    axis.set_title("Actual Intensity vs Physics Prediction")
    axis.set_xlabel("Actual intensity (m/s)")
    axis.set_ylabel("Physics prediction (m/s)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_residual_distribution(physics: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 5))
    axis.hist(physics["residual_ms"], bins=18, color="#00897B", alpha=0.85)
    axis.axvline(0.0, color="#111111", linestyle="--", linewidth=1.5)
    axis.set_title("Physics Residual Distribution")
    axis.set_xlabel("Actual - physics prediction (m/s)")
    axis.set_ylabel("Count")
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_residual_scatter(
    frame: pd.DataFrame,
    x_column: str,
    x_label: str,
    output_path: Path,
) -> None:
    valid = frame.dropna(subset=[x_column, "residual_ms"])
    if valid.empty:
        return

    fig, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(
        valid[x_column],
        valid["residual_ms"],
        color="#EF6C00",
        alpha=0.75,
        edgecolor="none",
    )
    axis.axhline(0.0, color="#111111", linestyle="--", linewidth=1.5)
    axis.set_title(f"{x_label} vs Physics Residual")
    axis.set_xlabel(x_label)
    axis.set_ylabel("Actual - physics prediction (m/s)")
    axis.grid(alpha=0.25)
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
