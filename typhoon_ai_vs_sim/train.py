from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn

from typhoon_ai_vs_sim.config import ExperimentConfig
from typhoon_ai_vs_sim.data import DataSplit, StandardScalerBundle


@dataclass(slots=True)
class TrainedModelResult:
    name: str
    checkpoint_path: Path
    history: pd.DataFrame
    best_val_loss: float


def train_model(
    name: str,
    model: nn.Module,
    train_split: DataSplit,
    val_split: DataSplit,
    config: ExperimentConfig,
    target_key: str,
) -> TrainedModelResult:
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.SmoothL1Loss()

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    history_rows: list[dict[str, float | int | str]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses: list[float] = []

        for batch in train_split.loader:
            inputs = batch["inputs"].to(device)
            targets = batch[target_key].to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_loss = _evaluate_loss(model, val_split, criterion, device, target_key)
        train_loss = float(np.mean(train_losses))
        history_rows.append(
            {
                "model": name,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"[{name}] epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= config.patience:
            print(f"[{name}] early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError(f"Model {name} did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    checkpoint_path = config.checkpoint_dir / f"{name.lower()}_best.pt"
    torch.save(model.state_dict(), checkpoint_path)

    return TrainedModelResult(
        name=name,
        checkpoint_path=checkpoint_path,
        history=pd.DataFrame(history_rows),
        best_val_loss=best_val_loss,
    )


def _evaluate_loss(
    model: nn.Module,
    split: DataSplit,
    criterion: nn.Module,
    device: torch.device,
    target_key: str,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in split.loader:
            inputs = batch["inputs"].to(device)
            targets = batch[target_key].to(device)
            predictions = model(inputs)
            losses.append(float(criterion(predictions, targets).item()))
    return float(np.mean(losses))


def predict_with_model(
    model_name: str,
    model: nn.Module,
    split: DataSplit,
    scaler: StandardScalerBundle,
    device_name: str,
    prediction_mode: str,
) -> pd.DataFrame:
    device = torch.device(device_name)
    model.to(device)
    model.eval()

    records: list[dict[str, float | int | str]] = []
    with torch.no_grad():
        for batch in split.loader:
            inputs = batch["inputs"].to(device)
            physics_predictions = batch["physics_prediction"].numpy()
            true_targets = batch["true_target"].numpy()
            scaled_predictions = model(inputs).cpu().numpy()

            if prediction_mode == "hybrid":
                residual_predictions = scaler.inverse_residual_target(scaled_predictions)
                final_predictions = physics_predictions + residual_predictions
            elif prediction_mode == "direct":
                final_predictions = scaler.inverse_direct_target(scaled_predictions)
                residual_predictions = final_predictions - physics_predictions
            else:
                raise ValueError(f"Unsupported prediction mode: {prediction_mode}")

            for index in range(len(residual_predictions)):
                records.append(
                    {
                        "model": model_name,
                        "split": split.name,
                        "storm_id": batch["storm_id"][index],
                        "target_step": int(batch["target_step"][index]),
                        "physics_prediction_ms": float(physics_predictions[index]),
                        "residual_prediction_ms": float(residual_predictions[index]),
                        "prediction_ms": float(final_predictions[index]),
                        "true_target_ms": float(true_targets[index]),
                    }
                )

    return pd.DataFrame.from_records(records)


def predict_physics_baseline(split: DataSplit) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for idx in range(len(split.dataset)):
        item = split.dataset[idx]
        physics_prediction = float(item["physics_prediction"].item())
        true_target = float(item["true_target"].item())
        records.append(
            {
                "model": "Physics",
                "split": split.name,
                "storm_id": item["storm_id"],
                "target_step": int(item["target_step"]),
                "physics_prediction_ms": physics_prediction,
                "residual_prediction_ms": 0.0,
                "prediction_ms": physics_prediction,
                "true_target_ms": true_target,
            }
        )
    return pd.DataFrame.from_records(records)


def summarize_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (split, model_name), frame in predictions.groupby(["split", "model"], sort=False):
        mae = mean_absolute_error(frame["true_target_ms"], frame["prediction_ms"])
        rmse = np.sqrt(mean_squared_error(frame["true_target_ms"], frame["prediction_ms"]))
        rows.append(
            {
                "split": split,
                "model": model_name,
                "mae": round(float(mae), 4),
                "rmse": round(float(rmse), 4),
            }
        )

    metrics = pd.DataFrame(rows)
    enriched_rows: list[dict[str, float | str]] = []
    for split, frame in metrics.groupby("split", sort=False):
        physics_row = frame.loc[frame["model"] == "Physics"].iloc[0]
        baseline_mae = float(physics_row["mae"])
        baseline_rmse = float(physics_row["rmse"])
        for _, row in frame.iterrows():
            enriched_rows.append(
                {
                    "split": split,
                    "model": row["model"],
                    "mae": float(row["mae"]),
                    "rmse": float(row["rmse"]),
                    "mae_improvement_pct": round(
                        100.0 * (baseline_mae - float(row["mae"])) / baseline_mae, 2
                    ),
                    "rmse_improvement_pct": round(
                        100.0 * (baseline_rmse - float(row["rmse"])) / baseline_rmse,
                        2,
                    ),
                }
            )

    return pd.DataFrame(enriched_rows)
