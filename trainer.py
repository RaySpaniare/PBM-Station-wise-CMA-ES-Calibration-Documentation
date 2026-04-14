# -*- coding: utf-8 -*-
'''
@File    :   trainer.py
@Time    :   2026-04-06
@Desc    :   Efficient trainer for station-wise PBM upper-bound calibration. Designed with a focus on
           computational efficiency and provides four main capabilities:
           1) Cluster-level CMA-ES pretraining to obtain cluster warm-starts.
           2) Per-station fine-tuning starting from cluster initial guesses to reduce blind global search.
           3) Support for ProcessPool parallel calibration to significantly reduce end-to-end runtime.
           4) Automatic retry for failed samples (increase population/iterations) to avoid single stations
              blocking the overall run.
           The terminal prints continuous updates: completed station counts, individual station status,
           cumulative elapsed time and ETA for long runs.
@Notice  :   Parallel execution on Windows may use substantial memory; adjust `num_workers` according to available RAM.
'''

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from runtime_env import configure_openmp_runtime

configure_openmp_runtime()

import json
import hashlib
import os
import time
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from config import ExportConfig, TrainConfig
from data_pipeline import StationSplitData
from metrics_losses import calc_r2_nse, calc_station_metrics, classify_performance_group
import model_pbm
from model_pbm import (
    active_sim_backend,
    parameter_names,
    simulate_station_population_with_unit_params,
    simulate_station_series,
    simulate_station_with_unit_params,
)


METRIC_KEYS: Tuple[str, ...] = (
    "R2",
    "NSE",
    "KGE",
    "RMSE",
    "MAE",
    "Bias",
    "Bias_Relative",
    "ubRMSE",
)


INVALID_LOSS = 1.0e12


@dataclass
class CalibrationArtifacts:
    output_dir: Path
    station_metrics_path: Path
    station_params_path: Path
    performance_summary_path: Path
    test_timeseries_path: Path
    train_val_timeseries_path: Path
    cma_history_path: Path
    station_metrics_df: pd.DataFrame
    station_params_df: pd.DataFrame
    test_timeseries_df: pd.DataFrame
    train_val_timeseries_df: pd.DataFrame
    cma_history_df: pd.DataFrame
    summary: Dict[str, object]


def _normalize_objective(raw: str) -> str:
    mode = str(raw).strip().lower()
    if mode not in {"nse", "mse"}:
        raise ValueError("objective only supports 'nse' or 'mse'")
    return mode


def _trim_warmup(
    pred: np.ndarray,
    obs: np.ndarray,
    dates: np.ndarray,
    warmup_days: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_arr = np.asarray(pred, dtype=np.float64).reshape(-1)
    obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    date_arr = np.asarray(dates)

    n = min(pred_arr.size, obs_arr.size, date_arr.size)
    if n <= 0:
        return pred_arr[:0], obs_arr[:0], date_arr[:0]

    cut = int(warmup_days) if n > int(warmup_days) else 0
    return pred_arr[cut:n], obs_arr[cut:n], date_arr[cut:n]


def _objective_loss(pred: np.ndarray, obs: np.ndarray, objective_mode: str) -> float:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    o = np.asarray(obs, dtype=np.float64).reshape(-1)
    valid = np.isfinite(p) & np.isfinite(o)
    if not bool(np.any(valid)):
        return float(INVALID_LOSS)

    p = p[valid]
    o = o[valid]
    if p.size < 2:
        return float(INVALID_LOSS)

    if objective_mode == "nse":
        nse = float(calc_r2_nse(pred=p, obs=o))
        if not np.isfinite(nse):
            return float(INVALID_LOSS)
        return float(1.0 - nse)

    return float(np.mean((p - o) ** 2))


def _objective_loss_batch(pred_batch: np.ndarray, obs: np.ndarray, objective_mode: str) -> np.ndarray:
    pred_arr = np.asarray(pred_batch, dtype=np.float64)
    if pred_arr.ndim == 1:
        pred_arr = pred_arr.reshape(1, -1)
    if pred_arr.ndim != 2:
        raise ValueError(f"pred_batch must be a 2D array, got {tuple(pred_arr.shape)}")

    obs_arr = np.asarray(obs, dtype=np.float64).reshape(-1)
    n = min(pred_arr.shape[1], obs_arr.size)
    if n <= 0:
        return np.full(pred_arr.shape[0], float(INVALID_LOSS), dtype=np.float64)

    pred_arr = pred_arr[:, :n]
    obs_arr = obs_arr[:n]

    valid = np.isfinite(pred_arr) & np.isfinite(obs_arr[None, :])
    valid_count = np.sum(valid, axis=1)
    enough = valid_count >= 2

    losses = np.full(pred_arr.shape[0], float(INVALID_LOSS), dtype=np.float64)
    if not bool(np.any(enough)):
        return losses

    diff = np.where(valid, pred_arr - obs_arr[None, :], 0.0)
    sse = np.sum(diff * diff, axis=1, dtype=np.float64)

    if objective_mode == "nse":
        obs_sum = np.sum(np.where(valid, obs_arr[None, :], 0.0), axis=1, dtype=np.float64)
        obs_mean = np.divide(
            obs_sum,
            np.maximum(valid_count, 1),
            out=np.zeros_like(obs_sum, dtype=np.float64),
            where=valid_count > 0,
        )
        centered = np.where(valid, obs_arr[None, :] - obs_mean[:, None], 0.0)
        sst = np.sum(centered * centered, axis=1, dtype=np.float64)
        ok = enough & np.isfinite(sse) & np.isfinite(sst) & (sst > 1.0e-12)
        losses[ok] = sse[ok] / sst[ok]
        return losses

    mse = np.divide(
        sse,
        np.maximum(valid_count, 1),
        out=np.full_like(sse, float(INVALID_LOSS), dtype=np.float64),
        where=valid_count > 0,
    )
    ok = enough & np.isfinite(mse)
    losses[ok] = mse[ok]
    return losses


def _safe_eigh(cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        vals, vecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(cov + np.eye(cov.shape[0], dtype=np.float64) * 1.0e-8)
    vals = np.maximum(vals, 1.0e-20)
    return vals, vecs


def _run_cma_es(
    objective_fn: Callable[[np.ndarray], float],
    objective_batch_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    dim: int,
    rng: np.random.Generator,
    population_size: int,
    max_iterations: int,
    sigma0: float,
    patience: int,
    init_mean: Optional[np.ndarray] = None,
    progress_print_every: int = 0,
    progress_prefix: str = "",
    on_iteration: Optional[Callable[[int, int, float, float, float], None]] = None,
    on_eval: Optional[Callable[[int, int, int, int], None]] = None,
) -> Tuple[np.ndarray, float, List[Dict[str, float]], int, int]:
    lam = max(4, int(population_size))
    mu = lam // 2

    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=np.float64))
    weights = weights / np.sum(weights)
    mueff = 1.0 / np.sum(weights ** 2)

    cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
    cs = (mueff + 2.0) / (dim + mueff + 5.0)
    c1 = 2.0 / (((dim + 1.3) ** 2) + mueff)
    cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / (((dim + 2.0) ** 2) + mueff))
    damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
    chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * (dim ** 2)))

    mean = np.full(dim, 0.5, dtype=np.float64) if init_mean is None else np.asarray(init_mean, dtype=np.float64).reshape(dim)
    mean = np.clip(mean, 0.0, 1.0)

    sigma = float(max(1.0e-4, sigma0))
    cov = np.eye(dim, dtype=np.float64)
    pc = np.zeros(dim, dtype=np.float64)
    ps = np.zeros(dim, dtype=np.float64)

    best_x = mean.copy()
    best_f = float("inf")
    eval_count = 0
    no_improve = 0
    history_rows: List[Dict[str, float]] = []

    for iteration in range(1, int(max_iterations) + 1):
        eigvals, eigvecs = _safe_eigh(cov)
        diag_sqrt = np.sqrt(eigvals)
        bd = eigvecs * diag_sqrt
        inv_sqrt_c = (eigvecs * (1.0 / diag_sqrt)) @ eigvecs.T

        z = rng.standard_normal((lam, dim))
        y = z @ bd.T
        x = np.clip(mean + sigma * y, 0.0, 1.0)

        fitness = np.empty(lam, dtype=np.float64)
        if objective_batch_fn is not None:
            batch_fitness = np.asarray(objective_batch_fn(x), dtype=np.float64).reshape(-1)
            if batch_fitness.size != lam:
                raise ValueError(f"objective_batch_fn returned incorrect length: expected {lam}, got {batch_fitness.size}")
            fitness[:] = np.where(np.isfinite(batch_fitness), batch_fitness, float(INVALID_LOSS))
            base_eval = int(eval_count)
            eval_count += int(lam)
            if on_eval is not None:
                for i in range(lam):
                    on_eval(int(iteration), int(i + 1), int(lam), int(base_eval + i + 1))
        else:
            for i in range(lam):
                val = float(objective_fn(x[i]))
                fitness[i] = val if np.isfinite(val) else float(INVALID_LOSS)
                eval_count += 1
                if on_eval is not None:
                    on_eval(int(iteration), int(i + 1), int(lam), int(eval_count))

        order = np.argsort(fitness)
        x_sel = x[order[:mu]]
        y_sel = (x_sel - mean[None, :]) / max(sigma, 1.0e-12)
        f_sel = fitness[order[:mu]]

        gen_best = float(f_sel[0])
        if gen_best + 1.0e-12 < best_f:
            best_f = gen_best
            best_x = x_sel[0].copy()
            no_improve = 0
        else:
            no_improve += 1

        mean = np.sum(x_sel * weights[:, None], axis=0)
        y_w = np.sum(y_sel * weights[:, None], axis=0)

        ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (inv_sqrt_c @ y_w)
        norm_ps = float(np.linalg.norm(ps))
        hs_cond = norm_ps / np.sqrt(1.0 - (1.0 - cs) ** (2.0 * iteration)) / chi_n
        hsig = 1.0 if hs_cond < (1.4 + 2.0 / (dim + 1.0)) else 0.0

        pc = (1.0 - cc) * pc + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * y_w

        rank_mu = np.zeros((dim, dim), dtype=np.float64)
        for i in range(mu):
            rank_mu += weights[i] * np.outer(y_sel[i], y_sel[i])

        cov = (
            (1.0 - c1 - cmu + (1.0 - hsig) * c1 * cc * (2.0 - cc)) * cov
            + c1 * np.outer(pc, pc)
            + cmu * rank_mu
        )
        cov = 0.5 * (cov + cov.T)

        sigma = float(sigma * np.exp((cs / damps) * (norm_ps / chi_n - 1.0)))

        history_rows.append(
            {
                "generation": float(iteration),
                "generation_best_loss": float(gen_best),
                "generation_mean_loss": float(np.mean(fitness)),
                "global_best_loss": float(best_f),
                "sigma": float(sigma),
            }
        )

        should_print = int(progress_print_every) > 0 and (
            iteration == 1 or iteration % int(progress_print_every) == 0 or iteration == int(max_iterations)
        )
        if should_print and progress_prefix:
            _progress_write(
                f"{progress_prefix} gen={iteration:04d}/{int(max_iterations):04d} "
                f"best={best_f:.6f} gen_best={gen_best:.6f} sigma={sigma:.4f}"
            )

        if on_iteration is not None:
            on_iteration(
                int(iteration),
                int(max_iterations),
                float(best_f),
                float(gen_best),
                float(sigma),
            )

        if no_improve >= int(max(1, patience)):
            break
        if sigma < 1.0e-6:
            break
        if best_f <= 1.0e-6:
            break

    return best_x, float(best_f), history_rows, int(eval_count), int(len(history_rows))


def _metrics_with_warmup(
    pred: np.ndarray,
    obs: np.ndarray,
    dates: np.ndarray,
    warmup_days: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, int]:
    p, o, d = _trim_warmup(pred=pred, obs=obs, dates=dates, warmup_days=warmup_days)
    valid = np.isfinite(p) & np.isfinite(o)
    p_valid = p[valid]
    o_valid = o[valid]
    d_valid = d[valid]

    if p_valid.size < 2:
        metrics = {key: np.nan for key in METRIC_KEYS}
        return metrics, p_valid, o_valid, d_valid, int(p_valid.size)

    metrics = calc_station_metrics(pred=p_valid, obs=o_valid)
    return metrics, p_valid, o_valid, d_valid, int(p_valid.size)


def _flatten_metrics(prefix: str, metrics: Dict[str, float], row: Dict[str, object]) -> None:
    for key in METRIC_KEYS:
        row[f"{prefix}_{key}"] = float(metrics.get(key, np.nan))


def _run_cma_with_restarts(
    objective_fn: Callable[[np.ndarray], float],
    objective_batch_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    dim: int,
    seed: int,
    population_size: int,
    max_iterations: int,
    sigma: float,
    patience: int,
    restarts: int,
    init_mean: Optional[np.ndarray],
    progress_print_every: int,
    progress_prefix: str,
    phase_label: str,
    progress_hook: Optional[Callable[[int, int, int, float, float, float], None]] = None,
    eval_hook: Optional[Callable[[int, int, int, int, int], None]] = None,
) -> Tuple[Optional[np.ndarray], float, List[Dict[str, object]], int, int]:
    best_unit: Optional[np.ndarray] = None
    best_loss = float("inf")
    total_evals = 0
    total_gens = 0
    history_rows: List[Dict[str, object]] = []

    rng_master = np.random.default_rng(int(seed))
    for restart in range(max(1, int(restarts))):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))

        if restart == 0 and init_mean is not None:
            start_mean = np.clip(np.asarray(init_mean, dtype=np.float64).reshape(dim), 0.0, 1.0)
        elif best_unit is not None:
            start_mean = np.clip(best_unit + 0.08 * rng.standard_normal(dim), 0.0, 1.0)
        else:
            start_mean = np.clip(0.5 + 0.15 * rng.standard_normal(dim), 0.0, 1.0)

        prefix = f"{progress_prefix} [{phase_label} R{restart + 1}]"

        def _on_iteration(iter_idx: int, iter_total: int, global_best: float, generation_best: float, cur_sigma: float) -> None:
            if progress_hook is None:
                return
            progress_hook(
                int(restart + 1),
                int(iter_idx),
                int(iter_total),
                float(global_best),
                float(generation_best),
                float(cur_sigma),
            )

        def _on_eval(iter_idx: int, cand_idx: int, cand_total: int, total_eval: int) -> None:
            if eval_hook is None:
                return
            eval_hook(
                int(restart + 1),
                int(iter_idx),
                int(cand_idx),
                int(cand_total),
                int(total_eval),
            )

        cand_unit, cand_loss, hist_rows, evals, gens = _run_cma_es(
            objective_fn=objective_fn,
            objective_batch_fn=objective_batch_fn if use_batch_eval else None,
            dim=dim,
            seed=int(seed),
            population_size=population_size,
            max_iterations=max_iterations,
            sigma0=float(sigma),
            patience=int(patience),
            init_mean=start_mean,
            progress_print_every=int(progress_print_every),
            progress_prefix=prefix,
            phase_label=phase_label,
            progress_hook=_on_iteration,
            eval_hook=_on_eval,
        )

        total_evals += int(evals)
        total_gens += int(gens)

        for row in hist_rows:
            h = dict(row)
            h["Phase"] = phase_label
            h["Restart"] = int(restart + 1)
            history_rows.append(h)

        if np.isfinite(cand_loss) and cand_loss < best_loss:
            best_loss = float(cand_loss)
            best_unit = cand_unit.copy()

    return best_unit, float(best_loss), history_rows, int(total_evals), int(total_gens)


def _build_station_payload(
    station: StationSplitData,
    objective_mode: str,
    warmup_days: int,
) -> Tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    forcings_train = np.ascontiguousarray(np.asarray(station.train_forcings, dtype=np.float32), dtype=np.float32)
    obs_train = np.asarray(station.train_qobs, dtype=np.float64).reshape(-1)
    date_train = np.asarray(station.train_dates)
    n_ref = min(int(forcings_train.shape[0]), int(obs_train.size), int(date_train.size))
    obs_ref = obs_train[:n_ref]

    def _obj_batch(unit_params_batch: np.ndarray) -> np.ndarray:
        pred_train = simulate_station_population_with_unit_params(
            forcings=forcings_train,
            f_veg=station.f_veg,
            unit_params_batch=unit_params_batch,
        )

        n = min(int(pred_train.shape[1]), int(n_ref))
        if n <= 0:
            return np.full(int(pred_train.shape[0]), float(INVALID_LOSS), dtype=np.float64)

        cut = int(warmup_days) if n > int(warmup_days) else 0
        pred_cut = pred_train[:, cut:n]
        obs_cut = obs_ref[cut:n]
        return _objective_loss_batch(pred_batch=pred_cut, obs=obs_cut, objective_mode=objective_mode)

    def _obj(unit_params: np.ndarray) -> float:
        loss = _obj_batch(np.asarray(unit_params, dtype=np.float64).reshape(1, -1))
        return float(loss[0]) if loss.size > 0 else float(INVALID_LOSS)

    return _obj, _obj_batch


def _calibrate_single_station(
    station: StationSplitData,
    cfg_dict: Dict[str, object],
    cluster_init_mean: Optional[np.ndarray],
    seed: int,
    print_generation_progress: bool,
    config_hash: str,
    station_hash: str,
) -> Dict[str, object]:
    _configure_worker_torch_threads(cfg_dict)
    objective_mode = _normalize_objective(str(cfg_dict["objective"]))
    dim = len(parameter_names())

    row: Dict[str, object] = {
        "Station_ID": station.station_id,
        "Cluster_ID": int(station.cluster_id),
        "Performance_Group": "Failed",
        "Objective": objective_mode,
        "Config_Hash": str(config_hash),
        "Station_Data_Hash": str(station_hash),
        "Status": "ok",
        "Best_Loss": np.nan,
        "CMA_Evals": 0,
        "CMA_Generations": 0,
        "f_veg": float(station.f_veg),
    }

    history_records: List[Dict[str, object]] = []
    test_ts_records: List[Dict[str, object]] = []
    train_val_ts_records: List[Dict[str, object]] = []
    param_row: Optional[Dict[str, object]] = None

    try:
        objective_fn, objective_batch_fn = _build_station_payload(
            station=station,
            objective_mode=objective_mode,
            warmup_days=int(cfg_dict["warmup_days"]),
        )
        use_batch_eval = bool(cfg_dict.get("enable_population_batch_eval", True))

        pop = int(cfg_dict["cma_population_size"])
        iters = int(cfg_dict["cma_max_iterations"])
        sigma = float(cfg_dict["cma_sigma"])
        patience = int(cfg_dict["cma_patience"])
        restarts = int(cfg_dict["cma_restarts"])
        print_every = int(cfg_dict["progress_print_every_generation"]) if print_generation_progress else 0
        prefix = f"[Station {station.station_id}]"

        best_unit, best_loss, hist_rows, evals, gens = _run_cma_with_restarts(
            objective_fn=objective_fn,
            objective_batch_fn=objective_batch_fn if use_batch_eval else None,
            dim=dim,
            seed=int(seed),
            population_size=pop,
            max_iterations=iters,
            sigma=sigma,
            patience=patience,
            restarts=restarts,
            init_mean=cluster_init_mean,
            progress_print_every=print_every,
            progress_prefix=prefix,
            phase_label="station_main",
        )

        total_evals = int(evals)
        total_gens = int(gens)
        history_records.extend(hist_rows)

        # If no valid solution was found, retry once with increased search budget.
        if (best_unit is None or not np.isfinite(best_loss)) and bool(cfg_dict["cma_retry_failed"]):
            retry_pop = max(pop + 2, int(np.ceil(pop * float(cfg_dict["cma_retry_population_scale"]))))
            retry_iters = max(iters + 10, int(np.ceil(iters * float(cfg_dict["cma_retry_iteration_scale"]))))

            retry_unit, retry_loss, retry_hist, retry_evals, retry_gens = _run_cma_with_restarts(
                objective_fn=objective_fn,
                objective_batch_fn=objective_batch_fn if use_batch_eval else None,
                dim=dim,
                seed=int(seed) + 97,
                population_size=retry_pop,
                max_iterations=retry_iters,
                sigma=sigma,
                patience=max(patience, 10),
                restarts=1,
                init_mean=cluster_init_mean,
                progress_print_every=print_every,
                progress_prefix=prefix,
                phase_label="station_retry",
            )
            history_records.extend(retry_hist)
            total_evals += int(retry_evals)
            total_gens += int(retry_gens)

            if retry_unit is not None and np.isfinite(retry_loss) and retry_loss < best_loss:
                best_unit = retry_unit
                best_loss = float(retry_loss)

        if best_unit is None or (not np.isfinite(best_loss)):
            raise RuntimeError("CMA-ES did not obtain a valid solution")

        train_pred, best_physical = simulate_station_with_unit_params(
            forcings=station.train_forcings,
            f_veg=station.f_veg,
            unit_params=best_unit,
        )
        val_pred = simulate_station_series(
            forcings=station.val_forcings,
            f_veg=station.f_veg,
            physical_params=best_physical,
        )
        test_pred = simulate_station_series(
            forcings=station.test_forcings,
            f_veg=station.f_veg,
            physical_params=best_physical,
        )

        train_met, p_train, o_train, d_train, n_train = _metrics_with_warmup(
            pred=train_pred,
            obs=station.train_qobs,
            dates=station.train_dates,
            warmup_days=int(cfg_dict["warmup_days"]),
        )
        val_met, p_val, o_val, d_val, n_val = _metrics_with_warmup(
            pred=val_pred,
            obs=station.val_qobs,
            dates=station.val_dates,
            warmup_days=int(cfg_dict["warmup_days"]),
        )
        test_met, p_test, o_test, d_test, n_test = _metrics_with_warmup(
            pred=test_pred,
            obs=station.test_qobs,
            dates=station.test_dates,
            warmup_days=int(cfg_dict["warmup_days"]),
        )

        row["Best_Loss"] = float(best_loss)
        row["CMA_Evals"] = int(total_evals)
        row["CMA_Generations"] = int(total_gens)
        row["Train_N"] = int(n_train)
        row["Val_N"] = int(n_val)
        row["Test_N"] = int(n_test)
        _flatten_metrics("Train", train_met, row)
        _flatten_metrics("Val", val_met, row)
        _flatten_metrics("Test", test_met, row)
        perf_group = classify_performance_group(float(row.get("Test_NSE", np.nan)))
        row["Performance_Group"] = str(perf_group)

        param_row = {
            "Station_ID": station.station_id,
            "Cluster_ID": int(station.cluster_id),
            "Performance_Group": str(perf_group),
            "Config_Hash": str(config_hash),
            "Station_Data_Hash": str(station_hash),
            "Best_Loss": float(best_loss),
            "f_veg": float(station.f_veg),
        }
        for k, v in best_physical.items():
            param_row[k] = float(v)

        if p_test.size > 0:
            date_test = pd.to_datetime(d_test, errors="coerce")
            for i in range(int(p_test.size)):
                if pd.isna(date_test[i]):
                    continue
                test_ts_records.append(
                    {
                        "Station_ID": station.station_id,
                        "Cluster_ID": int(station.cluster_id),
                        "Performance_Group": str(perf_group),
                        "Date": date_test[i],
                        "Obs": float(o_test[i]),
                        "Pred": float(p_test[i]),
                    }
                )

        for phase, pred_seq, obs_seq, date_seq in (
            ("Train", p_train, o_train, d_train),
            ("Val", p_val, o_val, d_val),
        ):
            if pred_seq.size <= 0:
                continue
            date_arr = pd.to_datetime(date_seq, errors="coerce")
            for i in range(int(pred_seq.size)):
                if pd.isna(date_arr[i]):
                    continue
                train_val_ts_records.append(
                    {
                        "Station_ID": station.station_id,
                        "Cluster_ID": int(station.cluster_id),
                        "Performance_Group": str(perf_group),
                        "Phase": str(phase),
                        "Date": date_arr[i],
                        "Obs": float(obs_seq[i]),
                        "Pred": float(pred_seq[i]),
                    }
                )

    except Exception as exc:
        row["Performance_Group"] = "Failed"
        row["Status"] = f"failed: {type(exc).__name__}"
        row["Message"] = str(exc)
        row["Train_N"] = 0
        row["Val_N"] = 0
        row["Test_N"] = 0
        for prefix in ("Train", "Val", "Test"):
            for key in METRIC_KEYS:
                row[f"{prefix}_{key}"] = np.nan

    for h in history_records:
        h["Station_ID"] = station.station_id
        h["Cluster_ID"] = int(station.cluster_id)
        h["Config_Hash"] = str(config_hash)
        h["Station_Data_Hash"] = str(station_hash)

    return {
        "metric_row": row,
        "param_row": param_row,
        "test_ts_records": test_ts_records,
        "train_val_ts_records": train_val_ts_records,
        "history_records": history_records,
    }


def _station_worker(payload: Dict[str, object]) -> Dict[str, object]:
    _configure_worker_torch_threads(payload.get("cfg_dict", {}))
    return _calibrate_single_station(
        station=payload["station"],
        cfg_dict=payload["cfg_dict"],
        cluster_init_mean=payload["cluster_init_mean"],
        seed=int(payload["seed"]),
        print_generation_progress=False,
        config_hash=str(payload["config_hash"]),
        station_hash=str(payload["station_hash"]),
    )


def _resolve_warmstart_workers(cfg_dict: Dict[str, object], total_clusters: int) -> int:
    if int(total_clusters) <= 1:
        return 1
    if not bool(cfg_dict.get("warmstart_use_process_pool", True)):
        return 1

    requested = int(cfg_dict.get("warmstart_workers", 0))
    if requested > 0:
        return max(1, min(requested, int(total_clusters)))

    configured_workers = int(cfg_dict.get("num_workers", 0))
    if configured_workers > 0:
        return max(1, min(configured_workers, int(total_clusters)))

    cpu = os.cpu_count() or 1
    return max(1, min(max(1, cpu - 1), int(total_clusters)))


def _build_cluster_objectives(
    subset: List[StationSplitData],
    objective_mode: str,
    warmup_days: int,
) -> Tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
    station_batch_fns: List[Callable[[np.ndarray], np.ndarray]] = []
    for st in subset:
        _, batch_fn = _build_station_payload(
            station=st,
            objective_mode=objective_mode,
            warmup_days=int(warmup_days),
        )
        station_batch_fns.append(batch_fn)

    def _cluster_obj_batch(unit_params_batch: np.ndarray) -> np.ndarray:
        unit = np.asarray(unit_params_batch, dtype=np.float64)
        if unit.ndim == 1:
            unit = unit.reshape(1, -1)
        if unit.ndim != 2:
            raise ValueError(f"unit_params_batch must be a 2D array, got {tuple(unit.shape)}")

        pop_size = int(unit.shape[0])
        if pop_size <= 0:
            raise ValueError("unit_params_batch cannot be empty")

        loss_sum = np.zeros(pop_size, dtype=np.float64)
        valid_count = np.zeros(pop_size, dtype=np.int64)

        for batch_fn in station_batch_fns:
            losses = np.asarray(batch_fn(unit), dtype=np.float64).reshape(-1)
            if losses.size != pop_size:
                raise ValueError(f"cluster objective batch result length mismatch: expected {pop_size}, got {losses.size}")
            valid = np.isfinite(losses) & (losses < float(INVALID_LOSS))
            if bool(np.any(valid)):
                loss_sum[valid] += losses[valid]
                valid_count[valid] += 1

        out = np.full(pop_size, float(INVALID_LOSS), dtype=np.float64)
        ok = valid_count > 0
        out[ok] = loss_sum[ok] / valid_count[ok]
        return out

    def _cluster_obj(unit_params: np.ndarray) -> float:
        vals = _cluster_obj_batch(np.asarray(unit_params, dtype=np.float64).reshape(1, -1))
        return float(vals[0]) if vals.size > 0 else float(INVALID_LOSS)

    return _cluster_obj, _cluster_obj_batch


def _run_cluster_pretrain(
    cluster_id: int,
    subset: List[StationSplitData],
    cfg_dict: Dict[str, object],
    objective_mode: str,
    dim: int,
    seed: int,
    progress_print_every: int,
    progress_hook: Optional[Callable[[int, int, int, float, float, float], None]] = None,
    eval_hook: Optional[Callable[[int, int, int, int, int], None]] = None,
) -> Tuple[Optional[np.ndarray], float, List[Dict[str, object]], int, int]:
    _configure_worker_torch_threads(cfg_dict)
    objective_fn, objective_batch_fn = _build_cluster_objectives(
        subset=subset,
        objective_mode=objective_mode,
        warmup_days=int(cfg_dict["warmup_days"]),
    )
    use_batch_eval = bool(cfg_dict.get("enable_population_batch_eval", True))

    return _run_cma_with_restarts(
        objective_fn=objective_fn,
        objective_batch_fn=objective_batch_fn if use_batch_eval else None,
        dim=dim,
        seed=int(seed),
        population_size=int(cfg_dict["cluster_pretrain_population_size"]),
        max_iterations=int(cfg_dict["cluster_pretrain_max_iterations"]),
        sigma=float(cfg_dict["cma_sigma"]),
        patience=int(cfg_dict["cluster_pretrain_patience"]),
        restarts=1,
        init_mean=np.full(dim, 0.5, dtype=np.float64),
        progress_print_every=int(progress_print_every),
        progress_prefix=f"[WarmStart Cluster {cluster_id}]",
        phase_label="cluster_pretrain",
        progress_hook=progress_hook,
        eval_hook=eval_hook,
    )


def _cluster_pretrain_worker(payload: Dict[str, object]) -> Dict[str, object]:
    _configure_worker_torch_threads(payload.get("cfg_dict", {}))
    cluster_id = int(payload["cluster_id"])
    subset = payload["subset"]
    cfg_dict = payload["cfg_dict"]
    objective_mode = str(payload["objective_mode"])
    dim = int(payload["dim"])
    seed = int(payload["seed"])

    best_unit, best_loss, hist_rows, evals, gens = _run_cluster_pretrain(
        cluster_id=cluster_id,
        subset=subset,
        cfg_dict=cfg_dict,
        objective_mode=objective_mode,
        dim=dim,
        seed=seed,
        progress_print_every=0,
        progress_hook=None,
        eval_hook=None,
    )
    return {
        "cluster_id": int(cluster_id),
        "best_unit": best_unit,
        "best_loss": float(best_loss),
        "hist_rows": hist_rows,
        "evals": int(evals),
        "gens": int(gens),
    }


def _build_cluster_warmstarts(
    station_list: List[StationSplitData],
    cfg_dict: Dict[str, object],
) -> Tuple[Dict[int, np.ndarray], List[Dict[str, object]]]:
    if not bool(cfg_dict["enable_cluster_warmstart"]):
        return {}, []

    objective_mode = _normalize_objective(str(cfg_dict["objective"]))
    dim = len(parameter_names())
    rng = np.random.default_rng(int(cfg_dict["seed"]) + 77)

    cluster_to_stations: Dict[int, List[StationSplitData]] = {}
    for st in station_list:
        cluster_to_stations.setdefault(int(st.cluster_id), []).append(st)

    warmstart_map: Dict[int, np.ndarray] = {}
    history_records: List[Dict[str, object]] = []

    _progress_write("\n[WarmStart] Start cluster-level pretraining")
    cluster_ids = sorted(cluster_to_stations.keys())
    cluster_bar = _new_progress_bar(total=len(cluster_ids), desc="WarmStart-Cluster")

    jobs: List[Dict[str, object]] = []
    use_batch_eval = bool(cfg_dict.get("enable_population_batch_eval", True))

    for cluster_id in cluster_ids:
        stations = cluster_to_stations[cluster_id]
        stations = sorted(stations, key=lambda s: int(s.train_forcings.shape[0]), reverse=True)
        sample_n = int(min(len(stations), int(cfg_dict["cluster_pretrain_station_sample"])))
        if sample_n <= 0:
            if cluster_bar is not None:
                cluster_bar.update(1)
                cluster_bar.set_postfix_str(f"cluster={cluster_id} skipped", refresh=False)
            continue

        subset = stations[:sample_n]
        est_eval_total = int(cfg_dict["cluster_pretrain_population_size"]) * int(cfg_dict["cluster_pretrain_max_iterations"])
        if use_batch_eval:
            est_sim_calls = int(cfg_dict["cluster_pretrain_max_iterations"]) * sample_n
        else:
            est_sim_calls = int(est_eval_total * sample_n)

        _progress_write(
            f"[WarmStart][Cluster {cluster_id}] stations={len(stations)} sample={sample_n} "
            f"est_eval={est_eval_total} est_sim_calls={est_sim_calls} batch_eval={use_batch_eval}"
        )

        jobs.append(
            {
                "cluster_id": int(cluster_id),
                "subset": subset,
                "seed": int(rng.integers(0, 2**31 - 1)),
            }
        )

    warm_workers = _resolve_warmstart_workers(cfg_dict=cfg_dict, total_clusters=len(jobs))
    use_pool = warm_workers > 1 and len(jobs) > 1

    if use_pool:
        _progress_write(f"[WarmStart] Use ProcessPoolExecutor workers={warm_workers}")
        future_map = {}
        with ProcessPoolExecutor(max_workers=warm_workers) as executor:
            for job in jobs:
                payload = {
                    "cluster_id": int(job["cluster_id"]),
                    "subset": job["subset"],
                    "cfg_dict": cfg_dict,
                    "objective_mode": objective_mode,
                    "dim": int(dim),
                    "seed": int(job["seed"]),
                }
                fut = executor.submit(_cluster_pretrain_worker, payload)
                future_map[fut] = job

            for fut in as_completed(future_map):
                job = future_map[fut]
                cluster_id = int(job["cluster_id"])
                try:
                    result = fut.result()
                    best_unit = result["best_unit"]
                    best_loss = float(result["best_loss"])
                    hist_rows = result["hist_rows"]
                    evals = int(result["evals"])
                    gens = int(result["gens"])

                    for h in hist_rows:
                        h["Station_ID"] = f"cluster_{cluster_id}"
                        h["Cluster_ID"] = int(cluster_id)
                        history_records.append(h)

                    if best_unit is not None and np.isfinite(best_loss):
                        warmstart_map[int(cluster_id)] = np.asarray(best_unit, dtype=np.float64).copy()
                        _progress_write(
                            f"[WarmStart][Cluster {cluster_id}] best_loss={best_loss:.6f} "
                            f"(ok, evals={evals}, gens={gens})"
                        )
                        if cluster_bar is not None:
                            cluster_bar.update(1)
                            cluster_bar.set_postfix_str(f"cluster={cluster_id} ok", refresh=False)
                    else:
                        _progress_write(f"[WarmStart][Cluster {cluster_id}] failed, fallback to random init")
                        if cluster_bar is not None:
                            cluster_bar.update(1)
                            cluster_bar.set_postfix_str(f"cluster={cluster_id} failed", refresh=False)
                except Exception as exc:
                    _progress_write(
                        f"[WarmStart][Cluster {cluster_id}] failed: {type(exc).__name__}: {exc}; "
                        "fallback to random init"
                    )
                    if cluster_bar is not None:
                        cluster_bar.update(1)
                        cluster_bar.set_postfix_str(f"cluster={cluster_id} failed", refresh=False)
    else:
        for job in jobs:
            cluster_id = int(job["cluster_id"])
            subset = job["subset"]

            iter_bar = _new_progress_bar(
                total=int(cfg_dict["cluster_pretrain_max_iterations"]),
                desc=f"WarmStart-C{cluster_id}-Gen",
            )
            eval_bar = _new_progress_bar(
                total=int(cfg_dict["cluster_pretrain_population_size"]) * int(cfg_dict["cluster_pretrain_max_iterations"]),
                desc=f"WarmStart-C{cluster_id}-Eval",
            )
            last_iter = [0]

            def _cluster_progress_hook(
                restart_idx: int,
                iter_idx: int,
                iter_total: int,
                global_best: float,
                generation_best: float,
                cur_sigma: float,
            ) -> None:
                _ = restart_idx
                _ = iter_total
                _ = generation_best
                if iter_bar is None:
                    return
                delta = int(iter_idx) - int(last_iter[0])
                if delta > 0:
                    iter_bar.update(delta)
                    last_iter[0] = int(iter_idx)
                iter_bar.set_postfix_str(f"best={global_best:.4f} sigma={cur_sigma:.3f}", refresh=False)

            def _cluster_eval_hook(
                restart_idx: int,
                iter_idx: int,
                cand_idx: int,
                cand_total: int,
                total_eval: int,
            ) -> None:
                _ = restart_idx
                _ = iter_idx
                _ = total_eval
                if eval_bar is None:
                    return
                eval_bar.update(1)
                if int(cand_idx) == int(cand_total):
                    eval_bar.set_postfix_str(f"iter={iter_idx} eval={total_eval}", refresh=False)

            best_unit, best_loss, hist_rows, _, _ = _run_cluster_pretrain(
                cluster_id=cluster_id,
                subset=subset,
                cfg_dict=cfg_dict,
                objective_mode=objective_mode,
                dim=dim,
                seed=int(job["seed"]),
                progress_print_every=int(cfg_dict["progress_print_every_generation"]),
                progress_hook=_cluster_progress_hook,
                eval_hook=_cluster_eval_hook,
            )

            if iter_bar is not None:
                iter_bar.close()
            if eval_bar is not None:
                eval_bar.close()

            for h in hist_rows:
                h["Station_ID"] = f"cluster_{cluster_id}"
                h["Cluster_ID"] = int(cluster_id)
                history_records.append(h)

            if best_unit is not None and np.isfinite(best_loss):
                warmstart_map[int(cluster_id)] = best_unit.copy()
                _progress_write(f"[WarmStart][Cluster {cluster_id}] best_loss={best_loss:.6f} (ok)")
                if cluster_bar is not None:
                    cluster_bar.update(1)
                    cluster_bar.set_postfix_str(f"cluster={cluster_id} ok", refresh=False)
            else:
                _progress_write(f"[WarmStart][Cluster {cluster_id}] failed, fallback to random init")
                if cluster_bar is not None:
                    cluster_bar.update(1)
                    cluster_bar.set_postfix_str(f"cluster={cluster_id} failed", refresh=False)

    if cluster_bar is not None:
        cluster_bar.close()

    _progress_write("[WarmStart] Completed\n")
    return warmstart_map, history_records


def _resolve_num_workers(num_workers: int) -> int:
    if int(num_workers) > 0:
        return int(num_workers)
    cpu = os.cpu_count() or 1
    return max(1, cpu - 1)


def _progress_write(message: str) -> None:
    if tqdm is not None:
        try:
            tqdm.write(message)
            return
        except Exception:
            pass
    print(message)


def _new_progress_bar(total: int, desc: str):
    if tqdm is None:
        return None
    if int(total) <= 0:
        return None
    try:
        return tqdm(
            total=int(total),
            desc=desc,
            dynamic_ncols=True,
            mininterval=0.3,
            leave=True,
        )
    except Exception:
        return None


def _safe_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _configure_worker_torch_threads(cfg_dict: Dict[str, object]) -> None:
    num_threads = int(cfg_dict.get("worker_torch_num_threads", 1))
    num_interop = int(cfg_dict.get("worker_torch_num_interop_threads", 1))
    try:
        import torch

        if num_threads > 0:
            torch.set_num_threads(int(num_threads))
        if num_interop > 0:
            try:
                torch.set_num_interop_threads(int(num_interop))
            except RuntimeError:
                # interop thread pool may have been initialized in some contexts; keep defaults then.
                pass
    except Exception:
        pass


def _hash_dict(payload: Dict[str, object]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_content_hash(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _station_data_hash(station: StationSplitData) -> str:
    hasher = hashlib.sha256()
    hasher.update(str(station.station_id).encode("utf-8"))
    hasher.update(np.asarray([int(station.cluster_id)], dtype=np.int64).tobytes())
    hasher.update(np.asarray([float(station.f_veg)], dtype=np.float64).tobytes())

    def _upd_dt(arr: np.ndarray) -> None:
        dt = pd.to_datetime(np.asarray(arr), errors="coerce")
        v = dt.view("int64")
        hasher.update(np.asarray(v, dtype=np.int64).tobytes())

    def _upd_f(arr: np.ndarray) -> None:
        x = np.asarray(arr, dtype=np.float32)
        hasher.update(np.asarray(x.shape, dtype=np.int64).tobytes())
        hasher.update(x.tobytes(order="C"))

    _upd_dt(station.train_dates)
    _upd_dt(station.val_dates)
    _upd_dt(station.test_dates)

    _upd_f(station.train_forcings)
    _upd_f(station.val_forcings)
    _upd_f(station.test_forcings)
    _upd_f(station.train_qobs)
    _upd_f(station.val_qobs)
    _upd_f(station.test_qobs)
    return hasher.hexdigest()


def run_stationwise_cmaes(
    station_data: List[StationSplitData],
    train_cfg: TrainConfig,
    export_cfg: ExportConfig,
) -> CalibrationArtifacts:
    objective_mode = _normalize_objective(train_cfg.objective)
    sim_backend = active_sim_backend()
    out_dir = Path(export_cfg.results_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    station_metrics_path = out_dir / "station_metrics.csv"
    station_params_path = out_dir / "station_best_parameters.csv"
    cma_history_path = out_dir / "cmaes_optimization_history.csv"
    performance_summary_path = out_dir / "performance_summary.json"
    test_timeseries_path = out_dir / "test_timeseries_predictions.parquet"
    train_val_timeseries_path = out_dir / "train_val_timeseries_predictions.parquet"

    max_station = int(train_cfg.max_station_count)
    station_list = station_data[:max_station] if max_station > 0 else station_data
    if len(station_list) == 0:
        raise RuntimeError("Station list is empty; cannot run CMA-ES")

    workers = _resolve_num_workers(train_cfg.num_workers)
    cfg_dict: Dict[str, object] = {
        "seed": int(train_cfg.seed),
        "num_workers": int(workers),
        "objective": str(objective_mode),
        "warmup_days": int(train_cfg.warmup_days),
        "nse_screen_threshold": float(train_cfg.nse_screen_threshold),
        "progress_print_every_generation": int(train_cfg.progress_print_every_generation),
        "resume_skip_completed": bool(train_cfg.resume_skip_completed),
        "resume_require_hash_match": bool(train_cfg.resume_require_hash_match),
        "enable_cluster_warmstart": bool(train_cfg.enable_cluster_warmstart),
        "cluster_pretrain_population_size": int(train_cfg.cluster_pretrain_population_size),
        "cluster_pretrain_max_iterations": int(train_cfg.cluster_pretrain_max_iterations),
        "cluster_pretrain_patience": int(train_cfg.cluster_pretrain_patience),
        "cluster_pretrain_station_sample": int(train_cfg.cluster_pretrain_station_sample),
        "warmstart_use_process_pool": bool(train_cfg.warmstart_use_process_pool),
        "warmstart_workers": int(train_cfg.warmstart_workers),
        "enable_population_batch_eval": bool(train_cfg.enable_population_batch_eval),
        "worker_torch_num_threads": int(train_cfg.worker_torch_num_threads),
        "worker_torch_num_interop_threads": int(train_cfg.worker_torch_num_interop_threads),
        "cma_population_size": int(train_cfg.cma_population_size),
        "cma_max_iterations": int(train_cfg.cma_max_iterations),
        "cma_sigma": float(train_cfg.cma_sigma),
        "cma_patience": int(train_cfg.cma_patience),
        "cma_restarts": int(train_cfg.cma_restarts),
        "cma_retry_failed": bool(train_cfg.cma_retry_failed),
        "cma_retry_population_scale": float(train_cfg.cma_retry_population_scale),
        "cma_retry_iteration_scale": float(train_cfg.cma_retry_iteration_scale),
    }
    code_hashes: Dict[str, str] = {
        "trainer.py": _file_content_hash(__file__),
        "model_pbm.py": _file_content_hash(model_pbm.__file__),
    }
    config_hash = _hash_dict(
        {
            "cfg": cfg_dict,
            "param_names": parameter_names(),
            "code_hashes": code_hashes,
            "version": "pbm-upper-bound-v3",
        }
    )

    station_hash_map: Dict[str, str] = {str(s.station_id): _station_data_hash(s) for s in station_list}

    total_station = int(len(station_list))
    print("=" * 72)
    print("Station-wise PBM CMA-ES Calibration (Efficiency Mode)")
    print(f"Objective={objective_mode.upper()} | Stations={total_station} | Workers={workers}")
    print(f"Simulation backend={sim_backend} | Requested={os.environ.get('PBM_SIM_BACKEND', 'auto')}")
    print(f"Cluster warmstart={bool(train_cfg.enable_cluster_warmstart)}")
    print(
        f"Batch eval={bool(train_cfg.enable_population_batch_eval)} | "
        f"WarmStart pool={bool(train_cfg.warmstart_use_process_pool)}"
    )
    print(
        f"Worker torch threads={int(train_cfg.worker_torch_num_threads)} | "
        f"interop={int(train_cfg.worker_torch_num_interop_threads)}"
    )
    print(f"Resume skip={bool(train_cfg.resume_skip_completed)} | Strict hash={bool(train_cfg.resume_require_hash_match)}")
    print(f"Config hash={config_hash[:12]}...")
    print("=" * 72)

    resume_metric_rows: List[Dict[str, object]] = []
    resume_param_rows: List[Dict[str, object]] = []
    resume_test_ts_records: List[Dict[str, object]] = []
    resume_train_val_ts_records: List[Dict[str, object]] = []
    resume_history_records: List[Dict[str, object]] = []
    skipped_station_ids: set[str] = set()

    if bool(train_cfg.resume_skip_completed) and station_metrics_path.exists():
        try:
            old_metrics = pd.read_csv(station_metrics_path)
            if not old_metrics.empty and "Station_ID" in old_metrics.columns and "Status" in old_metrics.columns:
                old_metrics["Station_ID"] = old_metrics["Station_ID"].astype(str)
                m = old_metrics.copy()

                if bool(train_cfg.resume_require_hash_match):
                    if ("Config_Hash" not in m.columns) or ("Station_Data_Hash" not in m.columns):
                        m = m.iloc[0:0]
                    else:
                        cfg_ok = m["Config_Hash"].astype(str) == str(config_hash)
                        data_ok = m["Station_ID"].map(station_hash_map).fillna("") == m["Station_Data_Hash"].astype(str)
                        m = m[cfg_ok & data_ok]

                m = m[m["Status"].astype(str) == "ok"]
                skipped_station_ids = set(m["Station_ID"].tolist())
                resume_metric_rows = m.to_dict(orient="records")

                if station_params_path.exists():
                    old_params = pd.read_csv(station_params_path)
                    if not old_params.empty and "Station_ID" in old_params.columns:
                        old_params["Station_ID"] = old_params["Station_ID"].astype(str)
                        p = old_params[old_params["Station_ID"].isin(skipped_station_ids)]
                        if bool(train_cfg.resume_require_hash_match):
                            if ("Config_Hash" not in p.columns) or ("Station_Data_Hash" not in p.columns):
                                p = p.iloc[0:0]
                            else:
                                cfg_ok = p["Config_Hash"].astype(str) == str(config_hash)
                                data_ok = p["Station_ID"].map(station_hash_map).fillna("") == p["Station_Data_Hash"].astype(str)
                                p = p[cfg_ok & data_ok]
                        resume_param_rows = p.to_dict(orient="records")

                if test_timeseries_path.exists():
                    old_ts = pd.read_parquet(test_timeseries_path)
                    if not old_ts.empty and "Station_ID" in old_ts.columns:
                        old_ts["Station_ID"] = old_ts["Station_ID"].astype(str)
                        t = old_ts[old_ts["Station_ID"].isin(skipped_station_ids)]
                        if bool(train_cfg.resume_require_hash_match):
                            if ("Config_Hash" not in t.columns) or ("Station_Data_Hash" not in t.columns):
                                t = t.iloc[0:0]
                            else:
                                cfg_ok = t["Config_Hash"].astype(str) == str(config_hash)
                                data_ok = t["Station_ID"].map(station_hash_map).fillna("") == t["Station_Data_Hash"].astype(str)
                                t = t[cfg_ok & data_ok]
                        resume_test_ts_records = t.to_dict(orient="records")

                if train_val_timeseries_path.exists():
                    old_tv = pd.read_parquet(train_val_timeseries_path)
                    if not old_tv.empty and "Station_ID" in old_tv.columns:
                        old_tv["Station_ID"] = old_tv["Station_ID"].astype(str)
                        tv = old_tv[old_tv["Station_ID"].isin(skipped_station_ids)]
                        if bool(train_cfg.resume_require_hash_match):
                            if ("Config_Hash" not in tv.columns) or ("Station_Data_Hash" not in tv.columns):
                                tv = tv.iloc[0:0]
                            else:
                                cfg_ok = tv["Config_Hash"].astype(str) == str(config_hash)
                                data_ok = tv["Station_ID"].map(station_hash_map).fillna("") == tv["Station_Data_Hash"].astype(str)
                                tv = tv[cfg_ok & data_ok]
                        resume_train_val_ts_records = tv.to_dict(orient="records")

                if cma_history_path.exists():
                    old_hist = pd.read_csv(cma_history_path)
                    if not old_hist.empty and "Station_ID" in old_hist.columns:
                        old_hist["Station_ID"] = old_hist["Station_ID"].astype(str)
                        h = old_hist[old_hist["Station_ID"].isin(skipped_station_ids)]
                        if bool(train_cfg.resume_require_hash_match):
                            if ("Config_Hash" not in h.columns) or ("Station_Data_Hash" not in h.columns):
                                h = h.iloc[0:0]
                            else:
                                cfg_ok = h["Config_Hash"].astype(str) == str(config_hash)
                                data_ok = h["Station_ID"].map(station_hash_map).fillna("") == h["Station_Data_Hash"].astype(str)
                                h = h[cfg_ok & data_ok]
                        resume_history_records = h.to_dict(orient="records")

        except Exception as exc:
            print(f"[Resume] Failed to read historical results, falling back to full recompute: {type(exc).__name__}: {exc}")
            resume_metric_rows = []
            resume_param_rows = []
            resume_test_ts_records = []
            resume_train_val_ts_records = []
            resume_history_records = []
            skipped_station_ids = set()

    pending_stations = [s for s in station_list if str(s.station_id) not in skipped_station_ids]
    print(
        f"[Resume] matched_skip={len(skipped_station_ids)} | pending={len(pending_stations)} | "
        f"total={total_station}"
    )

    warmstart_map: Dict[int, np.ndarray] = {}
    warm_hist: List[Dict[str, object]] = []
    if len(pending_stations) > 0:
        warmstart_map, warm_hist = _build_cluster_warmstarts(station_list=pending_stations, cfg_dict=cfg_dict)
        for h in warm_hist:
            h["Config_Hash"] = str(config_hash)
            h["Station_Data_Hash"] = "cluster_pretrain"

    metric_rows: List[Dict[str, object]] = list(resume_metric_rows)
    param_rows: List[Dict[str, object]] = list(resume_param_rows)
    test_ts_records: List[Dict[str, object]] = list(resume_test_ts_records)
    train_val_ts_records: List[Dict[str, object]] = list(resume_train_val_ts_records)
    history_records: List[Dict[str, object]] = list(resume_history_records) + list(warm_hist)

    start_time = time.perf_counter()
    rng = np.random.default_rng(int(train_cfg.seed) + 131)
    pending_total = int(len(pending_stations))

    if pending_total == 0:
        _progress_write("[Resume] all stations matched current hashes; skip calibration and aggregate outputs.")
    elif workers <= 1:
        station_bar = _new_progress_bar(total=pending_total, desc="Stations")
        for i, station in enumerate(pending_stations, start=1):
            station_seed = int(rng.integers(0, 2**31 - 1))
            init_mean = warmstart_map.get(int(station.cluster_id))
            result = _calibrate_single_station(
                station=station,
                cfg_dict=cfg_dict,
                cluster_init_mean=init_mean,
                seed=station_seed,
                print_generation_progress=True,
                config_hash=str(config_hash),
                station_hash=str(station_hash_map[str(station.station_id)]),
            )

            metric_rows.append(result["metric_row"])
            if result["param_row"] is not None:
                param_rows.append(result["param_row"])
            test_ts_records.extend(result["test_ts_records"])
            train_val_ts_records.extend(result["train_val_ts_records"])
            history_records.extend(result["history_records"])

            elapsed = time.perf_counter() - start_time
            avg = elapsed / float(max(i, 1))
            eta = max(0.0, (pending_total - i) * avg)
            done_all = len(skipped_station_ids) + i
            test_nse = _safe_float(result["metric_row"].get("Test_NSE", np.nan))
            best_loss = _safe_float(result["metric_row"].get("Best_Loss", np.nan))
            status = str(result["metric_row"].get("Status", "unknown"))
            if station_bar is not None:
                station_bar.update(1)
                station_bar.set_postfix_str(
                    f"station={station.station_id} status={status} nse={test_nse:.4f} eta={eta/60.0:.1f}m",
                    refresh=False,
                )
            _progress_write(
                f"[Done {done_all:04d}/{total_station:04d}] station={station.station_id} cluster={station.cluster_id} "
                f"status={status} loss={best_loss:.6f} test_nse={test_nse:.4f} "
                f"elapsed={elapsed/60.0:.1f}m eta={eta/60.0:.1f}m"
            )
        if station_bar is not None:
            station_bar.close()
    else:
        station_bar = _new_progress_bar(total=pending_total, desc="Stations")

        def _submit_station(executor: ProcessPoolExecutor, station: StationSplitData):
            station_seed = int(rng.integers(0, 2**31 - 1))
            payload = {
                "station": station,
                "cfg_dict": cfg_dict,
                "cluster_init_mean": warmstart_map.get(int(station.cluster_id)),
                "seed": station_seed,
                "config_hash": str(config_hash),
                "station_hash": str(station_hash_map[str(station.station_id)]),
            }
            return executor.submit(_station_worker, payload)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            station_iter = iter(pending_stations)
            future_map = {}
            max_inflight = max(int(workers) * 2, int(workers) + 1)

            for _ in range(min(max_inflight, pending_total)):
                st = next(station_iter, None)
                if st is None:
                    break
                fut = _submit_station(executor, st)
                future_map[fut] = st

            done_count = 0
            while future_map:
                done_set, _ = wait(set(future_map.keys()), return_when=FIRST_COMPLETED)
                for fut in done_set:
                    station = future_map.pop(fut)
                    done_count += 1

                    try:
                        result = fut.result()
                        metric_rows.append(result["metric_row"])
                        if result["param_row"] is not None:
                            param_rows.append(result["param_row"])
                        test_ts_records.extend(result["test_ts_records"])
                        train_val_ts_records.extend(result["train_val_ts_records"])
                        history_records.extend(result["history_records"])

                        status = str(result["metric_row"].get("Status", "unknown"))
                        best_loss = _safe_float(result["metric_row"].get("Best_Loss", np.nan))
                        test_nse = _safe_float(result["metric_row"].get("Test_NSE", np.nan))
                    except Exception as exc:
                        status = f"failed: {type(exc).__name__}"
                        best_loss = float("nan")
                        test_nse = float("nan")
                        metric_rows.append(
                            {
                                "Station_ID": station.station_id,
                                "Cluster_ID": int(station.cluster_id),
                                "Performance_Group": "Failed",
                                "f_veg": float(station.f_veg),
                                "Objective": objective_mode,
                                "Config_Hash": str(config_hash),
                                "Station_Data_Hash": str(station_hash_map[str(station.station_id)]),
                                "Status": status,
                                "Message": str(exc),
                            }
                        )

                    nxt = next(station_iter, None)
                    if nxt is not None:
                        next_fut = _submit_station(executor, nxt)
                        future_map[next_fut] = nxt

                    elapsed = time.perf_counter() - start_time
                    avg = elapsed / float(max(done_count, 1))
                    eta = max(0.0, (pending_total - done_count) * avg)
                    done_all = len(skipped_station_ids) + done_count
                    if station_bar is not None:
                        station_bar.update(1)
                        station_bar.set_postfix_str(
                            f"station={station.station_id} status={status} nse={test_nse:.4f} eta={eta/60.0:.1f}m",
                            refresh=False,
                        )
                    _progress_write(
                        f"[Done {done_all:04d}/{total_station:04d}] station={station.station_id} cluster={station.cluster_id} "
                        f"status={status} loss={best_loss:.6f} test_nse={test_nse:.4f} "
                        f"elapsed={elapsed/60.0:.1f}m eta={eta/60.0:.1f}m"
                    )
        if station_bar is not None:
            station_bar.close()

    metrics_df = pd.DataFrame(metric_rows)
    if not metrics_df.empty:
        metrics_df = metrics_df.drop_duplicates(subset=["Station_ID"], keep="last")
        metrics_df = metrics_df.sort_values(["Cluster_ID", "Station_ID"], kind="mergesort").reset_index(drop=True)

    if "Performance_Group" not in metrics_df.columns:
        metrics_df["Performance_Group"] = "Failed"
    test_nse_series = pd.to_numeric(metrics_df.get("Test_NSE", pd.Series(dtype=float)), errors="coerce")
    if metrics_df.shape[0] > 0:
        metrics_df["Performance_Group"] = test_nse_series.apply(
            lambda x: classify_performance_group(float(x)) if np.isfinite(x) else "Failed"
        )

    metric_value_cols = ["NSE", "KGE", "RMSE", "Bias", "Bias_Relative", "ubRMSE"]
    metric_required_cols = ["Station_ID", "Cluster_ID", "Performance_Group", "f_veg"]
    for phase in ("Train", "Val", "Test"):
        for metric_name in metric_value_cols:
            metric_required_cols.append(f"{phase}_{metric_name}")
    for col in metric_required_cols:
        if col not in metrics_df.columns:
            metrics_df[col] = np.nan
    metrics_export_df = metrics_df.loc[:, metric_required_cols].copy()

    params_df = pd.DataFrame(param_rows)
    if not params_df.empty:
        params_df = params_df.drop_duplicates(subset=["Station_ID"], keep="last")
        params_df = params_df.sort_values(["Cluster_ID", "Station_ID"], kind="mergesort").reset_index(drop=True)

    perf_map = dict(zip(metrics_export_df.get("Station_ID", pd.Series(dtype=str)).astype(str), metrics_export_df.get("Performance_Group", pd.Series(dtype=str)).astype(str)))
    if "Performance_Group" not in params_df.columns and not params_df.empty:
        params_df["Performance_Group"] = params_df["Station_ID"].astype(str).map(perf_map).fillna("Failed")
    param_required_cols = ["Station_ID", "Cluster_ID", "Performance_Group"] + list(parameter_names())
    for col in param_required_cols:
        if col not in params_df.columns:
            params_df[col] = np.nan
    params_export_df = params_df.loc[:, param_required_cols].copy()

    test_ts_df = pd.DataFrame(test_ts_records)
    if not test_ts_df.empty:
        if "Performance_Group" not in test_ts_df.columns:
            test_ts_df["Performance_Group"] = test_ts_df["Station_ID"].astype(str).map(perf_map).fillna("Failed")
        test_ts_df["Date"] = pd.to_datetime(test_ts_df["Date"], errors="coerce")
        test_ts_df = test_ts_df.dropna(subset=["Date"])
        test_ts_df = test_ts_df.sort_values(["Cluster_ID", "Station_ID", "Date"], kind="mergesort").reset_index(drop=True)
    test_required_cols = ["Station_ID", "Cluster_ID", "Performance_Group", "Date", "Obs", "Pred"]
    for col in test_required_cols:
        if col not in test_ts_df.columns:
            test_ts_df[col] = np.nan
    test_ts_export_df = test_ts_df.loc[:, test_required_cols].copy()

    train_val_ts_df = pd.DataFrame(train_val_ts_records)
    if not train_val_ts_df.empty:
        if "Performance_Group" not in train_val_ts_df.columns:
            train_val_ts_df["Performance_Group"] = train_val_ts_df["Station_ID"].astype(str).map(perf_map).fillna("Failed")
        train_val_ts_df["Date"] = pd.to_datetime(train_val_ts_df["Date"], errors="coerce")
        train_val_ts_df = train_val_ts_df.dropna(subset=["Date"])
        train_val_ts_df = train_val_ts_df.sort_values(["Cluster_ID", "Station_ID", "Phase", "Date"], kind="mergesort").reset_index(drop=True)
    train_val_required_cols = ["Station_ID", "Cluster_ID", "Performance_Group", "Phase", "Date", "Obs", "Pred"]
    for col in train_val_required_cols:
        if col not in train_val_ts_df.columns:
            train_val_ts_df[col] = np.nan
    train_val_ts_export_df = train_val_ts_df.loc[:, train_val_required_cols].copy()

    hist_df = pd.DataFrame(history_records)
    hist_export_df = pd.DataFrame(columns=["Station_ID", "Cluster_ID", "Generation", "Best_Loss", "Sigma"])
    if not hist_df.empty:
        hist_work = hist_df.copy()
        if "Station_ID" in hist_work.columns:
            hist_work = hist_work[~hist_work["Station_ID"].astype(str).str.startswith("cluster_")]
        if not hist_work.empty:
            if "Generation" in hist_work.columns:
                hist_work["Generation"] = pd.to_numeric(hist_work["Generation"], errors="coerce")
            else:
                hist_work["Generation"] = pd.to_numeric(hist_work.get("generation", np.nan), errors="coerce")

            if "Best_Loss" in hist_work.columns:
                hist_work["Best_Loss"] = pd.to_numeric(hist_work["Best_Loss"], errors="coerce")
            elif "global_best_loss" in hist_work.columns:
                hist_work["Best_Loss"] = pd.to_numeric(hist_work["global_best_loss"], errors="coerce")
            else:
                hist_work["Best_Loss"] = pd.to_numeric(hist_work.get("generation_best_loss", np.nan), errors="coerce")

            if "Sigma" in hist_work.columns:
                hist_work["Sigma"] = pd.to_numeric(hist_work["Sigma"], errors="coerce")
            else:
                hist_work["Sigma"] = pd.to_numeric(hist_work.get("sigma", np.nan), errors="coerce")

            for col in ["Station_ID", "Cluster_ID", "Generation", "Best_Loss", "Sigma"]:
                if col not in hist_work.columns:
                    hist_work[col] = np.nan
            hist_export_df = hist_work.loc[:, ["Station_ID", "Cluster_ID", "Generation", "Best_Loss", "Sigma"]].copy()
            hist_export_df = hist_export_df.sort_values(["Cluster_ID", "Station_ID", "Generation"], kind="mergesort").reset_index(drop=True)

    metrics_export_df.to_csv(station_metrics_path, index=False, encoding="utf-8-sig")
    params_export_df.to_csv(station_params_path, index=False, encoding="utf-8-sig")
    hist_export_df.to_csv(cma_history_path, index=False, encoding="utf-8-sig")
    try:
        test_ts_export_df.to_parquet(test_timeseries_path, index=False, engine="pyarrow")
        train_val_ts_export_df.to_parquet(train_val_timeseries_path, index=False, engine="pyarrow")
    except ImportError as exc:
        raise RuntimeError("Failed to export Parquet: please install pyarrow") from exc

    elapsed_total = float(time.perf_counter() - start_time)
    group_order = ["Excellent", "Good", "Poor", "Failed"]
    overall_total = int(metrics_export_df.shape[0])

    overall_groups: Dict[str, Dict[str, float]] = {}
    for group_name in group_order:
        count = int(np.sum(metrics_export_df["Performance_Group"].astype(str) == group_name)) if overall_total > 0 else 0
        pct = float(count / overall_total) if overall_total > 0 else 0.0
        overall_groups[group_name] = {
            "count": count,
            "ratio": pct,
            "percentage": pct * 100.0,
        }

    cluster_groups: Dict[str, Dict[str, object]] = {}
    cluster_numeric = pd.to_numeric(metrics_export_df.get("Cluster_ID", pd.Series(dtype=float)), errors="coerce")
    for cluster_id in range(1, 8):
        mask = cluster_numeric == float(cluster_id)
        sub = metrics_export_df.loc[mask]
        sub_total = int(sub.shape[0])
        stat: Dict[str, Dict[str, float]] = {}
        for group_name in group_order:
            count = int(np.sum(sub["Performance_Group"].astype(str) == group_name)) if sub_total > 0 else 0
            pct = float(count / sub_total) if sub_total > 0 else 0.0
            stat[group_name] = {
                "count": count,
                "ratio": pct,
                "percentage": pct * 100.0,
            }
        cluster_groups[str(cluster_id)] = {
            "total_station_count": sub_total,
            "groups": stat,
        }

    summary: Dict[str, object] = {
        "overall": {
            "total_station_count": overall_total,
            "groups": overall_groups,
        },
        "by_cluster": cluster_groups,
        "meta": {
            "resumed_station_count": int(len(skipped_station_ids)),
            "newly_calibrated_station_count": int(pending_total),
            "total_wall_seconds": float(elapsed_total),
            "stations_per_minute": float(max(pending_total, 0) / max(elapsed_total, 1.0) * 60.0),
            "simulation_backend": str(sim_backend),
            "config_hash": str(config_hash),
        },
    }

    with open(performance_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "-" * 72)
    print(
        f"Calibration finished: stations={overall_total}, resumed={int(len(skipped_station_ids))}, "
        f"wall={elapsed_total/60.0:.2f} min, speed={summary['meta']['stations_per_minute']:.2f} station/min"
    )
    print("-" * 72)

    return CalibrationArtifacts(
        output_dir=out_dir,
        station_metrics_path=station_metrics_path,
        station_params_path=station_params_path,
        performance_summary_path=performance_summary_path,
        test_timeseries_path=test_timeseries_path,
        train_val_timeseries_path=train_val_timeseries_path,
        cma_history_path=cma_history_path,
        station_metrics_df=metrics_export_df,
        station_params_df=params_export_df,
        test_timeseries_df=test_ts_export_df,
        train_val_timeseries_df=train_val_ts_export_df,
        cma_history_df=hist_export_df,
        summary=summary,
    )