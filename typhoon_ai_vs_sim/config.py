from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ExperimentConfig:
    seed: int = 42
    data_source: str = "ibtracs"
    n_storms: int = 120
    min_steps: int = 28
    max_steps: int = 42
    step_hours: int = 6
    window_size: int = 6
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 64
    epochs: int = 40
    patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    hidden_size: int = 64
    transformer_dim: int = 64
    transformer_heads: int = 4
    transformer_layers: int = 2
    dropout: float = 0.1
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    data_dir: Path = field(default_factory=lambda: Path("data"))
    ibtracs_basin: str = "WP"
    ibtracs_start_year: int = 2018
    ibtracs_end_year: int = 2020
    max_real_storms: int = 24
    min_real_track_steps: int = 12
    real_wind_source: str = "TOKYO_WIND"
    device: str = "cpu"

    def ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
