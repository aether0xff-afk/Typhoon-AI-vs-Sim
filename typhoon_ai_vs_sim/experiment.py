from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from typhoon_ai_vs_sim.config import ExperimentConfig
from typhoon_ai_vs_sim.data import FEATURE_NAMES, prepare_data
from typhoon_ai_vs_sim.data_sources import load_storm_tracks
from typhoon_ai_vs_sim.models import (
    LSTMResidualRegressor,
    MLPResidualRegressor,
    TransformerResidualRegressor,
)
from typhoon_ai_vs_sim.plotting import (
    plot_metrics,
    plot_sample_trajectories,
    plot_training_curves,
)
from typhoon_ai_vs_sim.simulation import storms_to_frame
from typhoon_ai_vs_sim.train import (
    predict_physics_baseline,
    predict_with_model,
    summarize_metrics,
    train_model,
)
from typhoon_ai_vs_sim.utils import save_json, set_seed


def run_experiment(config: ExperimentConfig) -> None:
    set_seed(config.seed)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.ensure_directories()

    storms, source_metadata = load_storm_tracks(config)
    raw_frame = storms_to_frame(storms)
    prepared = prepare_data(config, storms)
    raw_frame = raw_frame.merge(prepared.storm_frame, on="storm_id", how="left")

    raw_frame.to_csv(config.output_dir / "storm_tracks.csv", index=False)
    if config.data_source == "synthetic":
        raw_frame.to_csv(config.output_dir / "synthetic_storm_tracks.csv", index=False)
    save_json(source_metadata, config.output_dir / "data_source_summary.json")

    input_dim = len(FEATURE_NAMES)
    models = {
        "MLP": MLPResidualRegressor(
            window_size=config.window_size,
            input_dim=input_dim,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
        ),
        "LSTM": LSTMResidualRegressor(
            input_dim=input_dim,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
        ),
        "Transformer": TransformerResidualRegressor(
            input_dim=input_dim,
            model_dim=config.transformer_dim,
            num_heads=config.transformer_heads,
            num_layers=config.transformer_layers,
            dropout=config.dropout,
            max_len=config.window_size,
        ),
    }

    prediction_frames: list[pd.DataFrame] = [
        predict_physics_baseline(prepared.val),
        predict_physics_baseline(prepared.test),
    ]
    history_frames: list[pd.DataFrame] = []
    checkpoint_summary: dict[str, str] = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name} on {config.device}...")
        training_result = train_model(
            name=model_name,
            model=model,
            train_split=prepared.train,
            val_split=prepared.val,
            config=config,
        )
        history_frames.append(training_result.history)
        checkpoint_summary[model_name] = str(training_result.checkpoint_path)

        prediction_frames.append(
            predict_with_model(
                model_name=model_name,
                model=model,
                split=prepared.val,
                scaler=prepared.scaler,
                device_name=config.device,
            )
        )
        prediction_frames.append(
            predict_with_model(
                model_name=model_name,
                model=model,
                split=prepared.test,
                scaler=prepared.scaler,
                device_name=config.device,
            )
        )

    all_predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = summarize_metrics(all_predictions)
    training_history = pd.concat(history_frames, ignore_index=True)

    metrics.to_csv(config.output_dir / "metrics.csv", index=False)
    all_predictions.to_csv(config.output_dir / "predictions.csv", index=False)
    training_history.to_csv(config.output_dir / "training_history.csv", index=False)

    plot_metrics(metrics, config.output_dir / "metrics_comparison.png")
    plot_sample_trajectories(
        storm_frame=raw_frame,
        predictions=all_predictions,
        output_path=config.output_dir / "sample_trajectories.png",
    )
    plot_training_curves(training_history, config.output_dir / "training_curves.png")

    summary_payload = {
        "seed": config.seed,
        "data_source": config.data_source,
        "device": config.device,
        "n_storms": len(storms),
        "window_size": config.window_size,
        "feature_names": list(FEATURE_NAMES),
        "source_metadata": source_metadata,
        "checkpoints": checkpoint_summary,
        "test_metrics": metrics.loc[metrics["split"] == "test"].to_dict(orient="records"),
    }
    save_json(summary_payload, config.output_dir / "summary.json")

    print("\nTest metrics")
    print(metrics.loc[metrics["split"] == "test"].to_string(index=False))
    print(f"\nArtifacts saved to: {Path(config.output_dir).resolve()}")
