from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from typhoon_ai_vs_sim.config import ExperimentConfig
from typhoon_ai_vs_sim.simulation import StormTrack, physics_predict_next


FEATURE_NAMES = (
    "intensity_ms",
    "sst_c",
    "latitude_deg",
    "physics_next_intensity_ms",
    "physics_delta_ms",
)


@dataclass(slots=True)
class SampleBundle:
    features: np.ndarray
    residual_targets: np.ndarray
    physics_predictions: np.ndarray
    true_targets: np.ndarray
    storm_ids: np.ndarray
    target_steps: np.ndarray


@dataclass(slots=True)
class StandardScalerBundle:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    residual_target_mean: float
    residual_target_std: float
    direct_target_mean: float
    direct_target_std: float

    def transform_features(self, values: np.ndarray) -> np.ndarray:
        return (values - self.feature_mean) / self.feature_std

    def transform_residual_target(self, values: np.ndarray) -> np.ndarray:
        return (values - self.residual_target_mean) / self.residual_target_std

    def inverse_residual_target(self, values: np.ndarray) -> np.ndarray:
        return values * self.residual_target_std + self.residual_target_mean

    def transform_direct_target(self, values: np.ndarray) -> np.ndarray:
        return (values - self.direct_target_mean) / self.direct_target_std

    def inverse_direct_target(self, values: np.ndarray) -> np.ndarray:
        return values * self.direct_target_std + self.direct_target_mean


@dataclass(slots=True)
class DataSplit:
    name: str
    bundle: SampleBundle
    dataset: "SequenceDataset"
    loader: DataLoader


@dataclass(slots=True)
class PreparedData:
    train: DataSplit
    val: DataSplit
    test: DataSplit
    scaler: StandardScalerBundle
    storm_frame: pd.DataFrame
    split_map: dict[str, str]


class SequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        residual_targets: np.ndarray,
        direct_targets: np.ndarray,
        physics_predictions: np.ndarray,
        true_targets: np.ndarray,
        storm_ids: np.ndarray,
        target_steps: np.ndarray,
    ) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.residual_targets = torch.tensor(residual_targets, dtype=torch.float32)
        self.direct_targets = torch.tensor(direct_targets, dtype=torch.float32)
        self.physics_predictions = torch.tensor(physics_predictions, dtype=torch.float32)
        self.true_targets = torch.tensor(true_targets, dtype=torch.float32)
        self.storm_ids = storm_ids
        self.target_steps = target_steps.astype(np.int32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        return {
            "inputs": self.features[index],
            "residual_target": self.residual_targets[index],
            "direct_target": self.direct_targets[index],
            "physics_prediction": self.physics_predictions[index],
            "true_target": self.true_targets[index],
            "storm_id": str(self.storm_ids[index]),
            "target_step": int(self.target_steps[index]),
        }


def split_storms(
    storms: list[StormTrack], train_ratio: float, val_ratio: float, seed: int
) -> dict[str, str]:
    if len(storms) < 3:
        raise ValueError("At least 3 storms are required to create train/val/test splits.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(storms))
    n_train = max(1, int(len(storms) * train_ratio))
    n_val = max(1, int(len(storms) * val_ratio))

    if n_train + n_val >= len(storms):
        n_train = max(1, len(storms) - 2)
        n_val = 1

    split_map: dict[str, str] = {}
    for order, storm_index in enumerate(indices):
        if order < n_train:
            split = "train"
        elif order < n_train + n_val:
            split = "val"
        else:
            split = "test"
        split_map[storms[storm_index].storm_id] = split

    return split_map


def build_sample_bundle(storms: list[StormTrack], window_size: int) -> SampleBundle:
    feature_rows: list[np.ndarray] = []
    residual_targets: list[float] = []
    physics_predictions: list[float] = []
    true_targets: list[float] = []
    storm_ids: list[str] = []
    target_steps: list[int] = []

    for storm in storms:
        physics_next = physics_predict_next(
            storm.intensity[:-1], storm.sst[:-1], storm.latitude[:-1]
        )
        physics_delta = physics_next - storm.intensity[:-1]

        for current_step in range(window_size - 1, len(storm.time) - 1):
            window_slice = slice(current_step - window_size + 1, current_step + 1)
            sequence = np.stack(
                [
                    storm.intensity[:-1][window_slice],
                    storm.sst[:-1][window_slice],
                    storm.latitude[:-1][window_slice],
                    physics_next[window_slice],
                    physics_delta[window_slice],
                ],
                axis=-1,
            )
            true_target = float(storm.intensity[current_step + 1])
            physics_prediction = float(physics_next[current_step])
            residual_target = true_target - physics_prediction

            feature_rows.append(sequence.astype(np.float32))
            residual_targets.append(residual_target)
            physics_predictions.append(physics_prediction)
            true_targets.append(true_target)
            storm_ids.append(storm.storm_id)
            target_steps.append(int(current_step + 1))

    return SampleBundle(
        features=np.asarray(feature_rows, dtype=np.float32),
        residual_targets=np.asarray(residual_targets, dtype=np.float32),
        physics_predictions=np.asarray(physics_predictions, dtype=np.float32),
        true_targets=np.asarray(true_targets, dtype=np.float32),
        storm_ids=np.asarray(storm_ids, dtype=object),
        target_steps=np.asarray(target_steps, dtype=np.int32),
    )


def fit_scaler(train_bundle: SampleBundle) -> StandardScalerBundle:
    feature_mean = train_bundle.features.mean(axis=(0, 1), keepdims=True)
    feature_std = train_bundle.features.std(axis=(0, 1), keepdims=True) + 1e-6
    residual_target_mean = float(train_bundle.residual_targets.mean())
    residual_target_std = float(train_bundle.residual_targets.std() + 1e-6)
    direct_target_mean = float(train_bundle.true_targets.mean())
    direct_target_std = float(train_bundle.true_targets.std() + 1e-6)
    return StandardScalerBundle(
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        residual_target_mean=residual_target_mean,
        residual_target_std=residual_target_std,
        direct_target_mean=direct_target_mean,
        direct_target_std=direct_target_std,
    )


def make_dataset(bundle: SampleBundle, scaler: StandardScalerBundle) -> SequenceDataset:
    scaled_features = scaler.transform_features(bundle.features).astype(np.float32)
    scaled_residual_targets = scaler.transform_residual_target(
        bundle.residual_targets
    ).astype(np.float32)
    scaled_direct_targets = scaler.transform_direct_target(bundle.true_targets).astype(
        np.float32
    )
    return SequenceDataset(
        features=scaled_features,
        residual_targets=scaled_residual_targets,
        direct_targets=scaled_direct_targets,
        physics_predictions=bundle.physics_predictions,
        true_targets=bundle.true_targets,
        storm_ids=bundle.storm_ids,
        target_steps=bundle.target_steps,
    )


def _make_loader(
    dataset: SequenceDataset, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def prepare_data(config: ExperimentConfig, storms: list[StormTrack]) -> PreparedData:
    split_map = split_storms(storms, config.train_ratio, config.val_ratio, config.seed)
    grouped_storms = {
        "train": [storm for storm in storms if split_map[storm.storm_id] == "train"],
        "val": [storm for storm in storms if split_map[storm.storm_id] == "val"],
        "test": [storm for storm in storms if split_map[storm.storm_id] == "test"],
    }

    train_bundle = build_sample_bundle(grouped_storms["train"], config.window_size)
    val_bundle = build_sample_bundle(grouped_storms["val"], config.window_size)
    test_bundle = build_sample_bundle(grouped_storms["test"], config.window_size)

    scaler = fit_scaler(train_bundle)

    train_dataset = make_dataset(train_bundle, scaler)
    val_dataset = make_dataset(val_bundle, scaler)
    test_dataset = make_dataset(test_bundle, scaler)

    train_split = DataSplit(
        name="train",
        bundle=train_bundle,
        dataset=train_dataset,
        loader=_make_loader(train_dataset, config.batch_size, True, config.seed),
    )
    val_split = DataSplit(
        name="val",
        bundle=val_bundle,
        dataset=val_dataset,
        loader=_make_loader(val_dataset, config.batch_size, False, config.seed),
    )
    test_split = DataSplit(
        name="test",
        bundle=test_bundle,
        dataset=test_dataset,
        loader=_make_loader(test_dataset, config.batch_size, False, config.seed),
    )

    storm_frame = pd.DataFrame(
        [{"storm_id": storm_id, "split": split} for storm_id, split in split_map.items()]
    )

    return PreparedData(
        train=train_split,
        val=val_split,
        test=test_split,
        scaler=scaler,
        storm_frame=storm_frame,
        split_map=split_map,
    )
