# -*- coding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2026-04-06
@Desc    :   This module defines all configuration entries required for the station-wise PBM upper-bound
           calibration (Upper Bound / Oracle), and centralizes the management of physical parameter bounds.
           Unlike the previous Global/Cluster + AdamW workflow, this version targets gradient-free CMA-ES
           calibration. The core goal is to independently solve optimal PBM parameters per station and then
           evaluate whether the model structure itself can reach acceptable interpretable accuracy, thereby
           providing a trustworthy performance upper bound reference for HPA-MoE.
           Configuration is organized into three dataclasses: DataConfig, TrainConfig and ExportConfig,
           covering data field mapping, time split ratios, minimum station length, CMA-ES iterations and
           population size, objective type (NSE/MSE), and export options for plots. All options are
           overrideable via command line for experiment auditing and reproducibility. This module does not
           couple to simulation details or implement optimizers; it only offers structured configuration
           entry points and a generic argparse builder for main.py, data_pipeline.py and trainer.py.
@Notice  :   If you change the objective function or time-splitting, document the experimental protocol
           to avoid mixing results with different evaluation setups.
'''

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from typing import Any, Dict, Iterable, Optional, Tuple, Type, TypeVar


_CONFIG_T = TypeVar("_CONFIG_T")

# Physical parameter ranges: constrain raw parameters to interpretable physical intervals.
# CMA-ES operates in the unit hypercube and values are linearly mapped to these ranges.
PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "s_max": (2.0, 800.0),
    "s_wilt_frac": (0.05, 0.45),
    "k_drain": (1.0e-4, 0.5),
    "lag_srf": (0.0, 5.0),
    "lag_int": (1.0, 15.0),
    "k_perc": (1.0e-3, 0.1),
    "lag_gw": (1.0, 180.0),
    "beta_t": (1.0, 12.0),
    "gamma_t": (0.2, 1.8),
    "t_snow_thresh": (-2.0, 5.0),
    "t_snow_range": (1.0, 8.0),
    "k_melt_day": (2.0, 15.0),
    "k_melt_base": (0.0, 3.0),
    "s_can_max": (0.1, 4.0),
    "f_et_thresh": (0.3, 1.0),
}


@dataclass
class DataConfig:
    # Input data and field mapping.
    parquet_path: str = r"F:\python项目\Science科研项目\流域迁移\数据处理\hydro_pbm_data(598).parquet"
    cluster_csv_path: str = r"F:\python项目\Science科研项目\流域迁移\数据处理\Kmeans聚类分析\Clustering_Results.csv"
    cluster_csv_station_col: str = "station_id"
    cluster_csv_cluster_col: str = "Cluster"
    expected_cluster_count: int = 7
    strict_cluster_count: bool = True
    station_col: str = "Station_ID"
    date_col: str = "Date"
    precip_col: str = "P"
    temp_col: str = "T"
    pet_col: str = "PET"
    day_length_col: str = "Day_Length_Frac"
    runoff_col: str = "Runoff"
    fveg_col: str = "f_veg"

    # Strict time alignment: keep only 1986-05-16 to 2013-09-30 (as defined by the parquet file).
    align_start_date: str = "1986-05-16"
    align_end_date: str = "2013-09-30"

    # Data sanitization: forward-fill key fields first, then clip extreme values.
    fill_missing_with_ffill: bool = True
    clip_precip_min: float = 0.0
    clip_precip_max: float = 1000.0
    clip_temp_min_c: float = -60.0
    clip_temp_max_c: float = 60.0
    clip_pet_min: float = 0.0
    clip_runoff_min: float = 0.0
    clip_runoff_max: float = 500.0

    # Station-wise time splitting (continuous segments to avoid leakage). Default 7:2:1 (train/val/test).
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1

    # Minimum number of days per station; too-short series will be skipped.
    min_days_per_station: int = 900


@dataclass
class TrainConfig:
    # Optimization and filtering strategies.
    seed: int = 42
    objective: str = "nse"
    warmup_days: int = 365
    nse_screen_threshold: float = 0.2
    max_station_count: int = 0
    num_workers: int = 0
    progress_print_every_generation: int = 20

    # Resume behavior: skip completed stations only when both config and station data hashes match.
    resume_skip_completed: bool = False
    resume_require_hash_match: bool = True

    # Cluster pretraining (Warm Start): pretrain per-cluster initial parameters, then per-station fine-tune.
    enable_cluster_warmstart: bool = True
    cluster_pretrain_population_size: int = 12
    cluster_pretrain_max_iterations: int = 40
    cluster_pretrain_patience: int = 10
    cluster_pretrain_station_sample: int = 5
    warmstart_use_process_pool: bool = True
    warmstart_workers: int = 0

    # Batch evaluation: evaluate a population as a batch to reduce small-tensor dispatch overhead.
    enable_population_batch_eval: bool = True

    # Limit PyTorch CPU threads per worker to avoid oversubscription when using multiprocessing.
    worker_torch_num_threads: int = 1
    worker_torch_num_interop_threads: int = 1

    # CMA-ES hyperparameters.
    cma_population_size: int = 24
    cma_max_iterations: int = 180
    cma_sigma: float = 0.25
    cma_patience: int = 35
    cma_restarts: int = 1

    # Retry behavior: if no valid solution is found, automatically increase search budget and retry once.
    cma_retry_failed: bool = True
    cma_retry_population_scale: float = 1.6
    cma_retry_iteration_scale: float = 1.8


@dataclass
class ExportConfig:
    # Root directory for results; the main program will create subfolders as needed.
    results_root: str = "results_pbm_upper_bound"
    save_fig_jpg_dpi: int = 800


def add_dataclass_arguments(
    parser: argparse.ArgumentParser,
    config_cls: Type[_CONFIG_T],
    *,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> None:
    include_set = None if include is None else set(include)
    exclude_set = set() if exclude is None else set(exclude)

    defaults = config_cls()
    for cfg_field in fields(defaults):
        field_name = str(cfg_field.name)
        if include_set is not None and field_name not in include_set:
            continue
        if field_name in exclude_set:
            continue

        default_value = getattr(defaults, field_name)
        arg_name = f"--{field_name}"
        kwargs: Dict[str, Any] = {"default": default_value}

        if isinstance(default_value, bool):
            parser.add_argument(arg_name, action=argparse.BooleanOptionalAction, **kwargs)
            continue

        parser.add_argument(arg_name, type=type(default_value), **kwargs)


def build_config_from_args(config_cls: Type[_CONFIG_T], args: argparse.Namespace) -> _CONFIG_T:
    defaults = config_cls()
    kwargs: Dict[str, Any] = {}
    for cfg_field in fields(defaults):
        field_name = str(cfg_field.name)
        kwargs[field_name] = getattr(args, field_name, getattr(defaults, field_name))
    return config_cls(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pure PBM station-wise upper bound calibration: per-station CMA-ES calibration producing an Oracle baseline for HPA-MoE"
    )

    add_dataclass_arguments(parser, DataConfig)
    add_dataclass_arguments(parser, TrainConfig)
    add_dataclass_arguments(parser, ExportConfig)

    return parser.parse_args()
