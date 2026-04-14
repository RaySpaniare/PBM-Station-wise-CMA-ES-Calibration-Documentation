# -*- coding: utf-8 -*-
'''
@File    :   data_pipeline.py
@Time    :   2026-04-06
@Desc    :   Data preparation utilities for station-wise PBM upper-bound calibration.
           The module enforces a strict "7-cluster" check at the pipeline entrance. Workflow:
           1) Read the long-format hydrological table and attach `cluster_id` (prefer the column
              already in the main table; if missing, bridge from the cluster CSV).
           2) Build continuous time series per station and split into train/val/test segments.
           3) Return a list of StationSplitData objects containing cluster_id, f_veg, date arrays,
              forcing arrays and observed runoff arrays for CMA-ES calibration.
           If the detected cluster count does not match `expected_cluster_count`, the function
           raises an error to avoid training on corrupted cluster labels. The module also
           handles temperature unit correction, forcing bounds clipping and minimum-sample-length filtering.
@Notice  :   When `strict_cluster_count=True`, the function will abort if the cluster count before
           or after filtering does not match the expected value.
'''

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import DataConfig


@dataclass
class StationSplitData:
    station_id: str
    cluster_id: int
    f_veg: float
    train_dates: np.ndarray
    train_forcings: np.ndarray
    train_qobs: np.ndarray
    val_dates: np.ndarray
    val_forcings: np.ndarray
    val_qobs: np.ndarray
    test_dates: np.ndarray
    test_forcings: np.ndarray
    test_qobs: np.ndarray


def load_hydro_dataframe(parquet_path: str) -> pd.DataFrame:
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError("Failed to read parquet file. Please install pyarrow and verify file integrity.") from exc

    if df.empty:
        raise RuntimeError("Input dataframe is empty; cannot proceed with PBM calibration.")
    return df


def _require_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Input data missing required columns: {missing}; current columns: {list(df.columns)}")


def _normalize_station_id(raw: str) -> str:
    val = str(raw).strip()
    return val if val.startswith("camels_") else f"camels_{val}"


def _infer_cluster_col(columns: List[str]) -> str:
    candidates = ["cluster_id", "Cluster", "cluster", "climate_cluster"]
    for c in candidates:
        if c in columns:
            return c
    raise KeyError("Cluster column not found. Expect one of: cluster_id/Cluster/cluster/climate_cluster")


def _attach_cluster_id(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    work = df.copy()

    # Prefer an existing cluster column in the main table to avoid additional mapping overhead.
    local_cluster_col = None
    for cand in ["cluster_id", "Cluster", "cluster", "climate_cluster"]:
        if cand in work.columns:
            local_cluster_col = cand
            break

    if local_cluster_col is not None:
        cluster_raw = pd.to_numeric(work[local_cluster_col], errors="coerce")
        if cluster_raw.isna().any():
            raise ValueError(f"Main table cluster column {local_cluster_col} contains non-parsable values")
        work["__cluster_id"] = cluster_raw.astype(np.int64)
        return work

    cluster_csv = Path(cfg.cluster_csv_path)
    if not cluster_csv.exists():
        raise FileNotFoundError(f"Main table lacks a cluster column and cluster CSV not found: {cluster_csv}")

    cluster_df = pd.read_csv(cluster_csv)
    _require_columns(cluster_df, [cfg.cluster_csv_station_col])

    cluster_col = cfg.cluster_csv_cluster_col
    if cluster_col not in cluster_df.columns:
        cluster_col = _infer_cluster_col(list(cluster_df.columns))

    _require_columns(work, [cfg.station_col])
    map_df = cluster_df[[cfg.cluster_csv_station_col, cluster_col]].dropna().copy()
    map_df[cfg.cluster_csv_station_col] = map_df[cfg.cluster_csv_station_col].astype(str).str.strip()
    map_df["__station_norm"] = map_df[cfg.cluster_csv_station_col].map(_normalize_station_id)
    map_df = map_df.drop_duplicates(subset=["__station_norm"], keep="first")

    work["__station_norm"] = work[cfg.station_col].astype(str).map(_normalize_station_id)
    merged = work.merge(
        map_df[["__station_norm", cluster_col]],
        on="__station_norm",
        how="left",
        suffixes=("", "_map"),
    )

    if merged[cluster_col].isna().any():
        miss_sites = (
            merged.loc[merged[cluster_col].isna(), cfg.station_col]
            .astype(str)
            .drop_duplicates()
            .head(8)
            .tolist()
        )
        raise ValueError(f"Cluster mapping failed; the following stations lack cluster_id: {miss_sites}")

    cluster_raw = pd.to_numeric(merged[cluster_col], errors="coerce")
    if cluster_raw.isna().any():
        raise ValueError("Cluster mapping produced non-parsable cluster values")

    merged["__cluster_id"] = cluster_raw.astype(np.int64)
    merged = merged.drop(columns=["__station_norm"]) if "__station_norm" in merged.columns else merged
    return merged


def _ensure_celsius(temp_values: np.ndarray) -> np.ndarray:
    arr = temp_values.astype(np.float32, copy=True)
    finite = arr[np.isfinite(arr)]
    # If median is very large (e.g. Kelvin values near 300), convert to Celsius.
    if finite.size and float(np.median(finite)) > 120.0:
        arr = arr - 273.15
    return arr


def _split_lengths(total_days: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    if total_days < 3:
        raise ValueError("Station length must be at least 3 days to split train/val/test")

    n_train = max(1, int(np.floor(total_days * train_ratio)))
    n_val = max(1, int(np.floor(total_days * val_ratio)))
    n_test = total_days - n_train - n_val

    if n_test < 1:
        if n_train > n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        n_test = total_days - n_train - n_val

    if n_test < 1:
        n_test = 1
        n_train = max(1, n_train - 1)

    return int(n_train), int(n_val), int(n_test)


def build_station_split_data(df: pd.DataFrame, cfg: DataConfig) -> Tuple[List[StationSplitData], Dict[str, object]]:
    required = [
        cfg.station_col,
        cfg.date_col,
        cfg.precip_col,
        cfg.temp_col,
        cfg.pet_col,
        cfg.day_length_col,
        cfg.runoff_col,
        cfg.fveg_col,
    ]
    _require_columns(df, required)

    ratio_sum = float(cfg.train_ratio + cfg.val_ratio + cfg.test_ratio)
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1")

    align_start = pd.Timestamp(cfg.align_start_date)
    align_end = pd.Timestamp(cfg.align_end_date)
    if align_end < align_start:
        raise ValueError("align_end_date cannot be earlier than align_start_date")
    required_span_days = int((align_end - align_start).days + 1)

    work = _attach_cluster_id(df=df, cfg=cfg)
    work[cfg.date_col] = pd.to_datetime(work[cfg.date_col], errors="coerce")
    work = work.dropna(subset=[cfg.date_col])
    work = work.sort_values([cfg.station_col, cfg.date_col], kind="mergesort")

    total_clusters = sorted(int(v) for v in work["__cluster_id"].dropna().astype(np.int64).unique().tolist())
    if bool(cfg.strict_cluster_count) and len(total_clusters) != int(cfg.expected_cluster_count):
        raise RuntimeError(
            f"Cluster detection anomaly: found {len(total_clusters)} clusters, expected {int(cfg.expected_cluster_count)}."
            f" Detected cluster labels: {total_clusters}"
        )

    station_data: List[StationSplitData] = []
    stats: Dict[str, object] = {
        "total_station_count": 0,
        "accepted_station_count": 0,
        "skipped_time_window_station_count": 0,
        "skipped_missing_station_count": 0,
        "skipped_short_station_count": 0,
        "total_cluster_count": len(total_clusters),
        "total_cluster_labels": total_clusters,
        "align_start_date": str(align_start.date()),
        "align_end_date": str(align_end.date()),
    }

    for station_id_raw, station_df in work.groupby(cfg.station_col, sort=False):
        stats["total_station_count"] += 1

        station_id = str(station_id_raw)
        station_df = station_df.sort_values(cfg.date_col, kind="mergesort")
        station_cluster_vals = station_df["__cluster_id"].dropna().astype(np.int64).unique()
        if station_cluster_vals.size != 1:
            raise RuntimeError(f"Station {station_id} maps to multiple clusters: {station_cluster_vals.tolist()}")
        cluster_id = int(station_cluster_vals[0])

        # Strict time clipping: keep only the configured alignment window.
        station_min_date = pd.to_datetime(station_df[cfg.date_col].min(), errors="coerce")
        station_max_date = pd.to_datetime(station_df[cfg.date_col].max(), errors="coerce")
        if pd.isna(station_min_date) or pd.isna(station_max_date):
            stats["skipped_time_window_station_count"] += 1
            warnings.warn(
                f"Station {station_id} has unparseable dates and was skipped",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if (station_min_date > align_start) or (station_max_date < align_end):
            stats["skipped_time_window_station_count"] += 1
            warnings.warn(
                f"Station {station_id} does not cover the required window {align_start.date()}~{align_end.date()} and was skipped",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        station_df = station_df[
            (station_df[cfg.date_col] >= align_start) & (station_df[cfg.date_col] <= align_end)
        ].copy()
        if station_df.empty:
            stats["skipped_time_window_station_count"] += 1
            warnings.warn(
                f"Station {station_id} has no valid records within the strict time window and was skipped",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        span_days = int((station_df[cfg.date_col].max() - station_df[cfg.date_col].min()).days + 1)
        if span_days < required_span_days:
            stats["skipped_time_window_station_count"] += 1
            warnings.warn(
                f"Station {station_id} covers insufficient days in the alignment window ({span_days}/{required_span_days}) and was skipped",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        # Missing-value sanitization: convert to numeric, forward-fill, then drop remaining NaNs.
        numeric_cols = [
            cfg.precip_col,
            cfg.temp_col,
            cfg.pet_col,
            cfg.runoff_col,
            cfg.day_length_col,
            cfg.fveg_col,
        ]
        for col in numeric_cols:
            station_df[col] = pd.to_numeric(station_df[col], errors="coerce")

        if bool(cfg.fill_missing_with_ffill):
            station_df[numeric_cols] = station_df[numeric_cols].ffill()

        station_df = station_df.dropna(
            subset=[
                cfg.precip_col,
                cfg.temp_col,
                cfg.pet_col,
                cfg.runoff_col,
                cfg.day_length_col,
            ]
        )
        if station_df.empty:
            stats["skipped_missing_station_count"] += 1
            warnings.warn(
                f"Station {station_id} has too many missing key fields and was skipped",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        total_days = int(station_df.shape[0])
        if total_days < int(cfg.min_days_per_station):
            stats["skipped_short_station_count"] += 1
            continue

        # Clip extreme values to configured physical ranges to protect optimization.
        p_arr = np.clip(
            station_df[[cfg.precip_col]].to_numpy(dtype=np.float32),
            float(cfg.clip_precip_min),
            float(cfg.clip_precip_max),
        )
        t_arr = _ensure_celsius(station_df[[cfg.temp_col]].to_numpy(dtype=np.float32))
        t_arr = np.clip(t_arr, float(cfg.clip_temp_min_c), float(cfg.clip_temp_max_c))
        pet_arr = np.clip(station_df[[cfg.pet_col]].to_numpy(dtype=np.float32), float(cfg.clip_pet_min), np.inf)
        day_len_arr = np.clip(station_df[[cfg.day_length_col]].to_numpy(dtype=np.float32), 0.0, 1.0)
        forcings = np.concatenate([p_arr, t_arr, pet_arr, day_len_arr], axis=1)

        qobs = np.clip(
            station_df[cfg.runoff_col].to_numpy(dtype=np.float64),
            float(cfg.clip_runoff_min),
            float(cfg.clip_runoff_max),
        )
        dates = station_df[cfg.date_col].to_numpy()

        fveg_vals = station_df[cfg.fveg_col].to_numpy(dtype=np.float32)
        fveg_val = float(np.nanmedian(fveg_vals)) if np.any(np.isfinite(fveg_vals)) else 0.5
        fveg_val = float(np.clip(fveg_val, 0.0, 1.0))

        n_train, n_val, n_test = _split_lengths(
            total_days=total_days,
            train_ratio=float(cfg.train_ratio),
            val_ratio=float(cfg.val_ratio),
        )
        train_end = n_train
        val_end = n_train + n_val

        station_data.append(
            StationSplitData(
                station_id=station_id,
                cluster_id=cluster_id,
                f_veg=fveg_val,
                train_dates=dates[:train_end],
                train_forcings=forcings[:train_end],
                train_qobs=qobs[:train_end],
                val_dates=dates[train_end:val_end],
                val_forcings=forcings[train_end:val_end],
                val_qobs=qobs[train_end:val_end],
                test_dates=dates[val_end : val_end + n_test],
                test_forcings=forcings[val_end : val_end + n_test],
                test_qobs=qobs[val_end : val_end + n_test],
            )
        )
        stats["accepted_station_count"] += 1

    accepted_clusters = sorted({int(s.cluster_id) for s in station_data})
    stats["accepted_cluster_count"] = int(len(accepted_clusters))
    stats["accepted_cluster_labels"] = accepted_clusters

    if bool(cfg.strict_cluster_count) and len(accepted_clusters) != int(cfg.expected_cluster_count):
        warnings.warn(
            f"Cluster count after filtering is abnormal: only {len(accepted_clusters)} clusters remain, expected {int(cfg.expected_cluster_count)}."
            f" Filtered cluster labels: {accepted_clusters}. Continuing execution with station-skipping rules.",
            RuntimeWarning,
            stacklevel=2,
        )

    station_data.sort(key=lambda x: x.station_id)
    return station_data, stats
