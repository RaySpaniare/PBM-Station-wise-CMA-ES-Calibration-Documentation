# -*- coding: utf-8 -*-
'''
@File    :   metrics_losses.py
@Time    :   2026-04-04
@Desc    :   Losses and evaluation metrics used in PBM calibration experiments. This module addresses
           common issues in hydrological time series: missing values, extreme flows, and numerical
           instability of log-domain metrics at low flows. The training loss combines MSE and a Log-MSE
           variant (the Log-MSE is more stable during low-flow periods and helps reduce high-flow
           dominance when using MSE alone). Evaluation returns per-station R2/NSE, KGE, RMSE, MAE,
           Bias and ubRMSE computed under finite-value masks. Mathematical definitions are implemented
           to support reproducible scientific experiments: NSE = 1 - SSE/SST, KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2), ubRMSE = sqrt(RMSE^2 - Bias^2).
           To avoid NaN propagation all functions first apply a finite-value mask and then check
           effective sample size before computing metrics. This module is model-agnostic and usable
           across Global/Cluster or HPA-MoE comparisons to ensure consistent metric definitions.
@Notice  :   Bias is implemented here as absolute mean bias (same units); read `Bias_Relative` if a relative bias is needed.
'''

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _valid_np(pred: np.ndarray, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    o = np.asarray(obs, dtype=np.float64).reshape(-1)
    m = np.isfinite(p) & np.isfinite(o)
    return p[m], o[m]


def calc_r2_nse(pred: np.ndarray, obs: np.ndarray) -> float:
    # R2 and NSE use the same formula for a single-variable runoff series: 1 - SSE/SST.
    p, o = _valid_np(pred, obs)
    if p.size < 2:
        return np.nan

    sse = float(np.sum((p - o) ** 2))
    sst = float(np.sum((o - np.mean(o)) ** 2))
    if sst <= 1e-12:
        return np.nan
    return 1.0 - sse / sst


def calc_kge(pred: np.ndarray, obs: np.ndarray) -> float:
    # KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
    p, o = _valid_np(pred, obs)
    if p.size < 2:
        return np.nan

    mean_p = float(np.mean(p))
    mean_o = float(np.mean(o))
    std_p = float(np.std(p))
    std_o = float(np.std(o))

    if std_o <= 1e-12 or std_p <= 1e-12:
        return np.nan

    r = float(np.corrcoef(p, o)[0, 1])
    if not np.isfinite(r):
        return np.nan

    alpha = std_p / std_o
    beta = mean_p / mean_o if abs(mean_o) > 1e-12 else np.nan
    if not np.isfinite(beta):
        return np.nan

    return 1.0 - float(np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def calc_rmse(pred: np.ndarray, obs: np.ndarray) -> float:
    p, o = _valid_np(pred, obs)
    if p.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((p - o) ** 2)))


def calc_mae(pred: np.ndarray, obs: np.ndarray) -> float:
    p, o = _valid_np(pred, obs)
    if p.size == 0:
        return np.nan
    return float(np.mean(np.abs(p - o)))


def calc_bias(pred: np.ndarray, obs: np.ndarray) -> Tuple[float, float]:
    # Bias is implemented as absolute mean bias (same units). Also return relative bias for reporting.
    p, o = _valid_np(pred, obs)
    if p.size == 0:
        return np.nan, np.nan

    bias_abs = float(np.mean(p) - np.mean(o))
    mean_obs = float(np.mean(o))
    bias_rel = float(bias_abs / (mean_obs + 1e-12)) if abs(mean_obs) > 1e-12 else np.nan
    return bias_abs, bias_rel


def calc_ubrmse(pred: np.ndarray, obs: np.ndarray) -> float:
    # ubRMSE = sqrt(RMSE^2 - Bias^2), where Bias uses absolute mean bias.
    rmse = calc_rmse(pred, obs)
    bias_abs, _ = calc_bias(pred, obs)
    if not np.isfinite(rmse) or not np.isfinite(bias_abs):
        return np.nan

    val = max(0.0, float(rmse ** 2 - bias_abs ** 2))
    return float(np.sqrt(val))


def calc_station_metrics(pred: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
    nse = calc_r2_nse(pred, obs)
    bias_abs, bias_rel = calc_bias(pred, obs)
    rmse = calc_rmse(pred, obs)

    return {
        "R2": nse,
        "NSE": nse,
        "KGE": calc_kge(pred, obs),
        "RMSE": rmse,
        "MAE": calc_mae(pred, obs),
        "Bias": bias_abs,
        "Bias_Relative": bias_rel,
        "ubRMSE": calc_ubrmse(pred, obs),
    }


def classify_performance_group(test_nse: float) -> str:
    nse_val = float(test_nse)
    if (not np.isfinite(nse_val)) or (nse_val <= 0.0):
        return "Failed"
    if nse_val >= 0.75:
        return "Excellent"
    if nse_val >= 0.5:
        return "Good"
    return "Poor"
