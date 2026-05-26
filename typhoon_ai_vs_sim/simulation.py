from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from typhoon_ai_vs_sim.config import ExperimentConfig


INTENSITY_MIN = 15.0
INTENSITY_MAX = 85.0

SYNTHETIC_PARAMETER_GUIDE = {
    "time_step_hours": {
        "value": 6,
        "rationale": "Matches the standard synoptic spacing used in operational best-track archives.",
    },
    "track_length_steps": {
        "range": [28, 42],
        "rationale": "Represents roughly 7 to 10.5 days, a typical window that captures intensification and decay within one storm life cycle.",
    },
    "initial_latitude_deg": {
        "range": [8.0, 14.0],
        "rationale": "Keeps genesis in the tropical western North Pacific belt while leaving room for poleward recurvature.",
    },
    "latitude_bounds_deg": {
        "range": [6.5, 32.0],
        "rationale": "Covers low-latitude genesis through the poleward weakening stage without extending into implausible extratropical latitudes for this toy setup.",
    },
    "northward_drift_deg_per_step": {
        "range": [0.18, 0.42],
        "rationale": "Equivalent to about 13 to 31 km/h of meridional motion at 6-hour spacing, consistent with observed tropical cyclone translation speeds.",
    },
    "sst_c": {
        "range": [24.6, 31.2],
        "rationale": "Spans marginally supportive water through very warm tropical ocean conditions while staying inside the daily OISST range commonly seen along western North Pacific tracks.",
    },
    "initial_intensity_ms": {
        "range": [24.0, 42.0],
        "rationale": "Starts storms at tropical-storm to lower-typhoon intensity so the sequence contains meaningful strengthening and weakening behavior.",
    },
    "intensity_bounds_ms": {
        "range": [INTENSITY_MIN, INTENSITY_MAX],
        "rationale": "Avoids physically trivial weak disturbances and caps intensity near the upper end of observed super-typhoon strength.",
    },
    "vertical_shear_ms": {
        "range": [2.0, 18.0],
        "rationale": "Covers low-shear environments favorable for intensification up to hostile shear values that suppress storm organization.",
    },
    "relative_humidity_fraction": {
        "range": [0.40, 0.96],
        "rationale": "Keeps humidity in a realistic marine-tropospheric range while allowing dry-air penalties and moist-core boosts.",
    },
}


@dataclass(slots=True)
class StormTrack:
    storm_id: str
    time: np.ndarray
    latitude: np.ndarray
    sst: np.ndarray
    intensity: np.ndarray
    shear: np.ndarray
    humidity: np.ndarray


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def build_synthetic_source_metadata(config: ExperimentConfig) -> dict[str, object]:
    return {
        "data_source": "synthetic",
        "seed": config.seed,
        "step_hours": config.step_hours,
        "n_storms": config.n_storms,
        "parameter_guide": SYNTHETIC_PARAMETER_GUIDE,
        "physics_baseline": {
            "description": "Interpretable simplified baseline model with rule-based coefficients, not a calibrated operational forecast model.",
            "intensity_min_ms": INTENSITY_MIN,
            "intensity_max_ms": INTENSITY_MAX,
            "thermal_reference_c": 27.0,
            "cold_penalty_threshold_c": 26.4,
            "poleward_drag_threshold_deg": 18.0,
            "saturation_threshold_ms": 60.0,
            "coefficient_note": "Coefficients are fixed simplified rule-based values that encode warm-water support, cold-water penalty, poleward weakening, and intensity saturation.",
        },
    }


def physics_delta(intensity: np.ndarray, sst: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    thermal_term = 1.55 * (sst - 27.0)
    cold_penalty = 0.95 * np.maximum(26.4 - sst, 0.0)
    poleward_drag = 0.55 * np.maximum(latitude - 18.0, 0.0)
    saturation_penalty = 0.045 * np.maximum(intensity - 60.0, 0.0)
    background_decay = 0.40 + 0.05 * np.maximum(latitude - 24.0, 0.0)
    return thermal_term - cold_penalty - poleward_drag - saturation_penalty - background_decay


def physics_predict_next(
    intensity: np.ndarray, sst: np.ndarray, latitude: np.ndarray
) -> np.ndarray:
    return np.clip(
        intensity + physics_delta(intensity, sst, latitude),
        INTENSITY_MIN,
        INTENSITY_MAX,
    )


def generate_storm_tracks(config: ExperimentConfig) -> list[StormTrack]:
    rng = np.random.default_rng(config.seed)
    storms: list[StormTrack] = []

    for storm_index in range(config.n_storms):
        steps = int(rng.integers(config.min_steps, config.max_steps + 1))
        time = np.arange(steps, dtype=np.int32)

        phase_1, phase_2, phase_3, phase_4 = rng.uniform(0.0, 2.0 * np.pi, size=4)
        lat_start = rng.uniform(8.0, 14.0)
        drift = rng.uniform(0.18, 0.42)
        meander = rng.uniform(0.15, 0.55)
        latitude = (
            lat_start
            + drift * time
            + meander * np.sin(rng.uniform(0.14, 0.32) * time + phase_1)
            + rng.normal(0.0, 0.08, size=steps)
        )
        latitude = np.clip(latitude, 6.5, 32.0)

        warm_pool = rng.uniform(28.7, 30.8)
        sst = (
            warm_pool
            - 0.10 * latitude
            + 0.42 * np.sin(0.24 * time + phase_2)
            + 0.18 * np.cos(0.11 * time + phase_3)
            + rng.normal(0.0, 0.14, size=steps)
        )
        sst = np.clip(sst, 24.6, 31.2)

        shear = (
            6.0
            + 0.42 * np.maximum(latitude - 14.0, 0.0)
            + 1.8 * np.sin(0.33 * time + phase_4)
            + rng.normal(0.0, 0.55, size=steps)
        )
        shear = np.clip(shear, 2.0, 18.0)

        humidity = (
            0.67
            + 0.09 * np.tanh((sst - 27.6) * 1.7)
            - 0.018 * np.maximum(latitude - 18.0, 0.0)
            + 0.05 * np.sin(0.27 * time + phase_1)
            + rng.normal(0.0, 0.025, size=steps)
        )
        humidity = np.clip(humidity, 0.40, 0.96)

        ocean_cycle = 0.65 * np.sin(0.52 * time + phase_2) + 0.25 * np.cos(
            0.19 * time + phase_3
        )

        intensity = np.zeros(steps, dtype=np.float32)
        intensity[0] = rng.uniform(24.0, 42.0)

        for t in range(steps - 1):
            current = float(intensity[t])
            prev_intensity = float(intensity[t - 1]) if t > 0 else current
            prev_sst = float(sst[t - 1]) if t > 0 else float(sst[t])

            thermal_drive = 1.85 * np.tanh((sst[t] - 27.4) * 1.65)
            rapid_intensification = 3.1 * _sigmoid((sst[t] - 28.7) * 3.4)
            rapid_intensification *= _sigmoid((humidity[t] - 0.72) * 10.0)
            rapid_intensification *= _sigmoid((60.0 - current) / 6.0)
            poleward_drag = 0.65 * np.maximum(latitude[t] - 18.5, 0.0)
            shear_drag = 0.24 * np.maximum(shear[t] - 9.0, 0.0) ** 1.2
            saturation = 0.08 * np.maximum(current - 65.0, 0.0) ** 1.05
            memory_bonus = 1.25 * np.tanh(
                (sst[t] - prev_sst) * 3.2 + (current - prev_intensity) / 8.0
            )
            eyewall_cycle = 0.95 * np.sin(0.57 * t + phase_4) * _sigmoid(
                (current - 52.0) / 7.0
            )
            humidity_bonus = 1.45 * (humidity[t] - 0.65)
            structured_noise = rng.normal(0.0, 0.35)

            delta_true = (
                0.55
                + thermal_drive
                + rapid_intensification
                + memory_bonus
                + humidity_bonus
                + 0.30 * ocean_cycle[t]
                + eyewall_cycle
                - poleward_drag
                - shear_drag
                - saturation
                + structured_noise
            )
            intensity[t + 1] = np.clip(
                current + delta_true,
                INTENSITY_MIN,
                INTENSITY_MAX,
            )

        storms.append(
            StormTrack(
                storm_id=f"storm_{storm_index:03d}",
                time=time,
                latitude=latitude.astype(np.float32),
                sst=sst.astype(np.float32),
                intensity=intensity,
                shear=shear.astype(np.float32),
                humidity=humidity.astype(np.float32),
            )
        )

    return storms


def storms_to_frame(storms: list[StormTrack]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for storm in storms:
        physics_next = np.full(storm.time.shape, np.nan, dtype=np.float32)
        physics_change = np.full(storm.time.shape, np.nan, dtype=np.float32)
        physics_next[:-1] = physics_predict_next(
            storm.intensity[:-1], storm.sst[:-1], storm.latitude[:-1]
        )
        physics_change[:-1] = physics_next[:-1] - storm.intensity[:-1]

        for idx, step in enumerate(storm.time):
            rows.append(
                {
                    "storm_id": storm.storm_id,
                    "time_step": int(step),
                    "latitude_deg": float(storm.latitude[idx]),
                    "sst_c": float(storm.sst[idx]),
                    "intensity_ms": float(storm.intensity[idx]),
                    "hidden_shear": float(storm.shear[idx]),
                    "hidden_humidity": float(storm.humidity[idx]),
                    "physics_next_intensity_ms": float(physics_next[idx]),
                    "physics_delta_ms": float(physics_change[idx]),
                }
            )

    return pd.DataFrame.from_records(rows)
