# -*- coding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2026-04-06
@Desc    :   Entry point for the station-wise PBM upper-bound calibration (Upper Bound / Oracle).
           The program reads the long-format hydrological table, performs station-wise continuous
           time splits (train/val/test), and then runs CMA-ES per station to search for PBM physical
           parameters. The default objective is 1 - NSE. This flow intentionally avoids Global/Cluster
           batch training and gradient-based optimizers to objectively evaluate the maximum accuracy
           achievable by the PBM structure under per-station free calibration. After calibration the
           script exports station metrics, best parameters, test timeseries predictions, optimization
           history, summarized reports and plots; results feed directly into HPA-MoE sample selection
           and upper-bound comparisons.
@Notice  :   It is recommended to run a small-scale trial with `--max_station_count` to check parameter
           ranges and runtime before calibrating all stations.
'''

from __future__ import annotations

import os
import random
from pathlib import Path

# Set numeric library thread caps before importing numpy/torch to avoid oversubscription in multiprocess runs.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from runtime_env import configure_openmp_runtime

configure_openmp_runtime()

import numpy as np
import torch

from config import DataConfig, ExportConfig, TrainConfig, build_config_from_args, parse_args
from data_pipeline import build_station_split_data, load_hydro_dataframe
from trainer import run_stationwise_cmaes
from visualization import plot_upper_bound_visualizations


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_path_from_script(path_like: str) -> Path:
    p = Path(path_like)
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parent / p


def main() -> None:
    args = parse_args()

    data_cfg = build_config_from_args(DataConfig, args)
    train_cfg = build_config_from_args(TrainConfig, args)
    export_cfg = build_config_from_args(ExportConfig, args)

    data_cfg.parquet_path = str(_resolve_path_from_script(data_cfg.parquet_path))
    data_cfg.cluster_csv_path = str(_resolve_path_from_script(data_cfg.cluster_csv_path))
    export_cfg.results_root = str(_resolve_path_from_script(export_cfg.results_root))

    results_root = Path(export_cfg.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    _set_seed(int(train_cfg.seed))

    print("=" * 72)
    print("PBM Station-wise Upper Bound Calibration")
    print(f"Input parquet: {data_cfg.parquet_path}")
    print(f"Cluster csv : {data_cfg.cluster_csv_path}")
    print(f"Expected clusters: {data_cfg.expected_cluster_count} | strict={data_cfg.strict_cluster_count}")
    print(f"Output root : {results_root}")
    print("=" * 72)

    df = load_hydro_dataframe(data_cfg.parquet_path)
    station_data, _ = build_station_split_data(df=df, cfg=data_cfg)

    if len(station_data) == 0:
        raise RuntimeError("No usable station samples were constructed; please check min_days_per_station or input data quality")

    artifacts = run_stationwise_cmaes(
        station_data=station_data,
        train_cfg=train_cfg,
        export_cfg=export_cfg,
    )

    plot_upper_bound_visualizations(
        station_metrics_df=artifacts.station_metrics_df,
        out_dir=results_root,
        jpg_dpi=int(export_cfg.save_fig_jpg_dpi),
    )

    print("\n" + "=" * 72)
    print("PBM station-wise upper bound run complete")
    print(f"Results root: {results_root}")
    print("Generated: station_metrics.csv")
    print("Generated: station_best_parameters.csv")
    print("Generated: cmaes_optimization_history.csv")
    print("Generated: performance_summary.json")
    print("Generated: test_timeseries_predictions.parquet")
    print("Generated: train_val_timeseries_predictions.parquet")
    print("=" * 72)


if __name__ == "__main__":
    main()
