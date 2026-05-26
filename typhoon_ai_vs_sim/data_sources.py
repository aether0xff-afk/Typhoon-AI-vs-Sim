from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from typhoon_ai_vs_sim.config import ExperimentConfig
from typhoon_ai_vs_sim.simulation import (
    StormTrack,
    build_synthetic_source_metadata,
    generate_storm_tracks,
)


IBTRACS_BASE_URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv"
)
OISST_FILESERVER_URL = (
    "https://www.ncei.noaa.gov/thredds/fileServer/OisstBase/NetCDF/V2.1/AVHRR"
)
KNOT_TO_MS = 0.514444


def load_storm_tracks(config: ExperimentConfig) -> tuple[list[StormTrack], dict[str, object]]:
    if config.data_source == "synthetic":
        storms = generate_storm_tracks(config)
        return storms, build_synthetic_source_metadata(config)
    if config.data_source == "ibtracs":
        return load_ibtracs_storm_tracks(config)
    raise ValueError(f"Unsupported data source: {config.data_source}")


def load_ibtracs_storm_tracks(
    config: ExperimentConfig,
) -> tuple[list[StormTrack], dict[str, object]]:
    csv_path = _ensure_ibtracs_csv(config)
    storm_frame = _read_ibtracs_frame(csv_path, config)
    selected = _select_candidate_storms(storm_frame, config)
    if selected.empty:
        raise RuntimeError("No real storms remained after applying the current filters.")

    enriched = _attach_oisst(selected, config)
    enriched, era5_metadata = _attach_era5_if_available(enriched, config)
    storms = _frame_to_storm_tracks(enriched, config)
    if not storms:
        raise RuntimeError("Real-data preprocessing did not yield any usable storm tracks.")

    metadata = {
        "data_source": "ibtracs",
        "basin": config.ibtracs_basin,
        "wind_source": config.real_wind_source,
        "season_range": [config.ibtracs_start_year, config.ibtracs_end_year],
        "step_hours": config.step_hours,
        "max_real_storms": config.max_real_storms,
        "min_real_track_steps": config.min_real_track_steps,
        "ibtracs_csv": str(csv_path.resolve()),
        "oisst_cache_dir": str((config.data_dir / "oisst_cache").resolve()),
        "storms_loaded": len(storms),
        "rows_loaded": int(sum(len(storm.time) for storm in storms)),
        "unique_oisst_days": int(enriched["oisst_date"].nunique()),
        "era5": era5_metadata,
        "physics_baseline": {
            "description": "Interpretable simplified baseline model with rule-based coefficients, not a calibrated operational forecast model.",
            "inputs": ["intensity_ms", "sst_c", "latitude_deg"],
            "coefficient_note": "Coefficients are fixed simplified rule-based values that encode warm-water support, cold-water penalty, poleward weakening, and intensity saturation.",
        },
        "notes": [
            "Tracks come from NOAA IBTrACS western North Pacific CSV.",
            "Intensity uses JMA/Tokyo 10-minute sustained wind when available, with WMO wind as fallback.",
            "Daily NOAA OISST v2.1 AVHRR fields are sampled at the nearest grid cell to each storm-center position.",
            "ERA5 humidity and vertical wind shear are optional cache-based enrichments; the real-data path still runs when no ERA5 cache is present.",
        ],
    }
    return storms, metadata


def _ensure_ibtracs_csv(config: ExperimentConfig) -> Path:
    basin = config.ibtracs_basin.upper()
    file_name = f"ibtracs.{basin}.list.v04r01.csv"
    target_path = config.data_dir / file_name
    if target_path.exists():
        return target_path

    url = f"{IBTRACS_BASE_URL}/{file_name}"
    _download_file(url, target_path)
    return target_path


def _read_ibtracs_frame(csv_path: Path, config: ExperimentConfig) -> pd.DataFrame:
    usecols = [
        "SID",
        "SEASON",
        "NAME",
        "ISO_TIME",
        "LAT",
        "LON",
        "WMO_WIND",
        "WMO_PRES",
        "TOKYO_WIND",
        "TOKYO_PRES",
    ]
    frame = pd.read_csv(csv_path, skiprows=[1], usecols=usecols, low_memory=False)
    frame["ISO_TIME"] = pd.to_datetime(frame["ISO_TIME"], utc=True, errors="coerce")
    frame["SEASON"] = pd.to_numeric(frame["SEASON"], errors="coerce")
    frame["LAT"] = pd.to_numeric(frame["LAT"], errors="coerce")
    frame["LON"] = pd.to_numeric(frame["LON"], errors="coerce")
    frame["TOKYO_WIND"] = pd.to_numeric(frame["TOKYO_WIND"], errors="coerce")
    frame["WMO_WIND"] = pd.to_numeric(frame["WMO_WIND"], errors="coerce")
    frame["TOKYO_PRES"] = pd.to_numeric(frame["TOKYO_PRES"], errors="coerce")
    frame["WMO_PRES"] = pd.to_numeric(frame["WMO_PRES"], errors="coerce")

    frame["wind_kt"] = frame[config.real_wind_source].combine_first(frame["WMO_WIND"])
    frame["pressure_hpa"] = frame["TOKYO_PRES"].combine_first(frame["WMO_PRES"])
    frame = frame.dropna(subset=["SID", "ISO_TIME", "SEASON", "LAT", "LON", "wind_kt"])

    synoptic = (
        (frame["ISO_TIME"].dt.minute == 0)
        & (frame["ISO_TIME"].dt.second == 0)
        & (frame["ISO_TIME"].dt.hour % config.step_hours == 0)
    )
    frame = frame.loc[synoptic].copy()
    frame = frame.loc[
        (frame["SEASON"] >= config.ibtracs_start_year)
        & (frame["SEASON"] <= config.ibtracs_end_year)
    ].copy()
    frame["wind_ms"] = frame["wind_kt"] * KNOT_TO_MS
    frame["storm_name"] = frame["NAME"].fillna("UNNAMED").astype(str).str.strip()
    frame["oisst_date"] = frame["ISO_TIME"].dt.floor("D")
    frame = frame.sort_values(["SEASON", "SID", "ISO_TIME"]).reset_index(drop=True)
    return frame


def _select_candidate_storms(frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    counts = (
        frame.groupby(["SID", "SEASON", "storm_name"], sort=False)
        .size()
        .reset_index(name="n_points")
    )
    counts = counts.loc[counts["n_points"] >= config.min_real_track_steps].copy()
    counts = counts.sort_values(
        ["SEASON", "n_points", "SID"], ascending=[False, False, True]
    ).reset_index(drop=True)
    selected_ids = counts["SID"].head(config.max_real_storms).tolist()
    return frame.loc[frame["SID"].isin(selected_ids)].copy()


def _attach_oisst(frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    cache_dir = config.data_dir / "oisst_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    result = frame.copy()
    result["sst_c"] = np.nan
    for oisst_date, date_rows in result.groupby("oisst_date", sort=True):
        dataset_path = _ensure_oisst_file(pd.Timestamp(oisst_date), cache_dir)
        with xr.open_dataset(dataset_path) as dataset:
            sst_field = dataset["sst"].isel(time=0, zlev=0)
            for row_index, row in date_rows.iterrows():
                lon = float(row["LON"]) % 360.0
                lat = float(row["LAT"])
                sst_value = float(sst_field.sel(lat=lat, lon=lon, method="nearest").values)
                result.at[row_index, "sst_c"] = sst_value

    result = result.dropna(subset=["sst_c"]).copy()
    return result


def _attach_era5_if_available(
    frame: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, dict[str, object]]:
    result = frame.copy()
    result["relative_humidity"] = np.nan
    result["vertical_shear"] = np.nan

    cache_dir = config.era5_cache_dir
    if cache_dir is None:
        return result, {"enabled": False, "reason": "No ERA5 cache directory configured."}

    era5_files = sorted(Path(cache_dir).glob("*.nc"))
    if not era5_files:
        return result, {
            "enabled": False,
            "cache_dir": str(Path(cache_dir).resolve()),
            "reason": "No ERA5 NetCDF files found.",
        }

    try:
        enriched = result
        for era5_file in era5_files:
            with xr.open_dataset(era5_file) as dataset:
                enriched = _sample_era5_dataset(enriched, dataset)
    except Exception as exc:
        return result, {
            "enabled": False,
            "cache_dir": str(Path(cache_dir).resolve()),
            "files": [path.name for path in era5_files],
            "reason": f"ERA5 cache could not be sampled: {exc}",
        }

    return enriched, {
        "enabled": True,
        "cache_dir": str(Path(cache_dir).resolve()),
        "files": [path.name for path in era5_files],
        "relative_humidity": "Nearest 700 hPa or 850 hPa relative humidity from ERA5 cache.",
        "vertical_shear": "sqrt((u200 - u850)^2 + (v200 - v850)^2) from ERA5 cache.",
        "rows_with_relative_humidity": int(enriched["relative_humidity"].notna().sum()),
        "rows_with_vertical_shear": int(enriched["vertical_shear"].notna().sum()),
    }


def _sample_era5_dataset(frame: pd.DataFrame, dataset: xr.Dataset) -> pd.DataFrame:
    result = frame.copy()
    coord_names = _era5_coord_names(dataset)
    rh_var = _first_present(dataset, ("r", "relative_humidity"))
    u_var = _first_present(dataset, ("u", "u_component_of_wind"))
    v_var = _first_present(dataset, ("v", "v_component_of_wind"))

    if rh_var is None and (u_var is None or v_var is None):
        raise ValueError("Expected ERA5 variables r/relative_humidity and u/v wind fields.")

    time_values = pd.to_datetime(dataset[coord_names["time"]].values, utc=True)
    min_time = time_values.min() - pd.Timedelta(hours=12)
    max_time = time_values.max() + pd.Timedelta(hours=12)
    candidate_rows = result.loc[
        (result["ISO_TIME"] >= min_time) & (result["ISO_TIME"] <= max_time)
    ]

    for row_index, row in candidate_rows.iterrows():
        time_value = pd.Timestamp(row["ISO_TIME"]).to_datetime64()
        lat_value = float(row["LAT"])
        lon_value = _normalize_lon_for_dataset(float(row["LON"]), dataset, coord_names["lon"])

        if rh_var is not None:
            rh_level = _select_level(dataset, coord_names["level"], (700, 850))
            if rh_level is not None:
                rh_value = _sample_era5_value(
                    dataset[rh_var],
                    coord_names,
                    time_value,
                    lat_value,
                    lon_value,
                    rh_level,
                )
                if np.isfinite(rh_value):
                    result.at[row_index, "relative_humidity"] = (
                        rh_value / 100.0 if rh_value > 1.5 else rh_value
                    )

        if u_var is not None and v_var is not None:
            level_200 = _select_level(dataset, coord_names["level"], (200,))
            level_850 = _select_level(dataset, coord_names["level"], (850,))
            if level_200 is not None and level_850 is not None:
                u200 = _sample_era5_value(
                    dataset[u_var], coord_names, time_value, lat_value, lon_value, level_200
                )
                v200 = _sample_era5_value(
                    dataset[v_var], coord_names, time_value, lat_value, lon_value, level_200
                )
                u850 = _sample_era5_value(
                    dataset[u_var], coord_names, time_value, lat_value, lon_value, level_850
                )
                v850 = _sample_era5_value(
                    dataset[v_var], coord_names, time_value, lat_value, lon_value, level_850
                )
                values = np.asarray([u200, v200, u850, v850], dtype=np.float32)
                if np.isfinite(values).all():
                    result.at[row_index, "vertical_shear"] = float(
                        np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2)
                    )

    return result


def _era5_coord_names(dataset: xr.Dataset) -> dict[str, str]:
    names = set(dataset.coords) | set(dataset.dims)
    candidates = {
        "time": ("valid_time", "time"),
        "lat": ("latitude", "lat"),
        "lon": ("longitude", "lon"),
        "level": ("pressure_level", "level", "isobaricInhPa"),
    }
    found: dict[str, str] = {}
    for key, options in candidates.items():
        for option in options:
            if option in names:
                found[key] = option
                break
        if key not in found:
            raise ValueError(f"ERA5 coordinate not found: {key}")
    return found


def _first_present(dataset: xr.Dataset, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in dataset:
            return name
    return None


def _select_level(
    dataset: xr.Dataset, level_coord: str, preferred_levels: tuple[int, ...]
) -> float | None:
    levels = np.asarray(dataset[level_coord].values, dtype=float)
    for preferred in preferred_levels:
        if levels.size == 0:
            return None
        nearest = float(levels[np.argmin(np.abs(levels - preferred))])
        if abs(nearest - preferred) <= 75:
            return nearest
    return None


def _normalize_lon_for_dataset(lon: float, dataset: xr.Dataset, lon_coord: str) -> float:
    lon_values = np.asarray(dataset[lon_coord].values, dtype=float)
    if lon_values.size and lon_values.max() > 180.0:
        return lon % 360.0
    if lon > 180.0:
        return lon - 360.0
    return lon


def _sample_era5_value(
    field: xr.DataArray,
    coord_names: dict[str, str],
    time_value: np.datetime64,
    lat_value: float,
    lon_value: float,
    level_value: float,
) -> float:
    selectors = {
        coord_names["time"]: time_value,
        coord_names["lat"]: lat_value,
        coord_names["lon"]: lon_value,
        coord_names["level"]: level_value,
    }
    return float(field.sel(selectors, method="nearest").values)


def _ensure_oisst_file(date_value: pd.Timestamp, cache_dir: Path) -> Path:
    yyyymm = date_value.strftime("%Y%m")
    yyyymmdd = date_value.strftime("%Y%m%d")
    file_name = f"oisst-avhrr-v02r01.{yyyymmdd}.nc"
    target_path = cache_dir / file_name
    if target_path.exists():
        return target_path

    url = f"{OISST_FILESERVER_URL}/{yyyymm}/{file_name}"
    _download_file(url, target_path)
    return target_path


def _download_file(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")

    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    temp_path.replace(target_path)


def _frame_to_storm_tracks(frame: pd.DataFrame, config: ExperimentConfig) -> list[StormTrack]:
    storms: list[StormTrack] = []

    for sid, storm_rows in frame.groupby("SID", sort=False):
        storm_rows = storm_rows.sort_values("ISO_TIME").reset_index(drop=True)
        if len(storm_rows) < config.min_real_track_steps:
            continue

        times = np.arange(len(storm_rows), dtype=np.int32)
        storms.append(
            StormTrack(
                storm_id=str(sid),
                time=times,
                latitude=storm_rows["LAT"].to_numpy(dtype=np.float32),
                sst=storm_rows["sst_c"].to_numpy(dtype=np.float32),
                intensity=storm_rows["wind_ms"].to_numpy(dtype=np.float32),
                shear=storm_rows["vertical_shear"].to_numpy(dtype=np.float32),
                humidity=storm_rows["relative_humidity"].to_numpy(dtype=np.float32),
            )
        )

    return storms
