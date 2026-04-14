# -*- coding: utf-8 -*-
'''
@File    :   model_pbm.py
@Time    :   2026-04-06
@Desc    :   Minimal PBM model interface required for station-wise upper-bound calibration. Given a single-station
           forcing sequence and a set of physical parameters, the module returns simulated runoff.
           Unlike previous versions this file does not include ParameterBank, Global/Cluster mode indexing or
           differentiable training wrappers, avoiding mixing gradient-based training and gradient-free
           CMA-ES calibration in the same code path. The core water-balance equations are preserved and
           utility functions are provided to map parameter vectors from the unit hypercube [0,1]^d to
           their physical bounds so that optimizers can search safely within interpretable ranges.
           Outputs include total runoff and diagnostic fluxes/storages for structural error analysis.
@Notice  :   The module uses CPU + torch forward simulation by default and does not rely on autograd;
           optimization should be implemented in trainer.py.
'''

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

try:
    import numba as nb
except Exception:
    nb = None

from config import PARAM_RANGES


PARAM_NAMES: Tuple[str, ...] = tuple(PARAM_RANGES.keys())
_PARAM_LOWER = np.asarray([float(PARAM_RANGES[k][0]) for k in PARAM_NAMES], dtype=np.float64)
_PARAM_SPAN = np.asarray([float(PARAM_RANGES[k][1] - PARAM_RANGES[k][0]) for k in PARAM_NAMES], dtype=np.float64)
_HAS_NUMBA = nb is not None


@dataclass
class ParameterSnapshot:
    s_max: Tensor
    s_wilt_frac: Tensor
    s_wilt: Tensor
    k_drain: Tensor
    lag_srf: Tensor
    lag_int: Tensor
    k_perc: Tensor
    lag_gw: Tensor
    beta_t: Tensor
    gamma_t: Tensor
    t_snow_thresh: Tensor
    t_snow_range: Tensor
    k_melt_day: Tensor
    k_melt_base: Tensor
    s_can_max: Tensor
    f_et_thresh: Tensor


def _resolve_sim_backend() -> str:
    raw = str(os.environ.get("PBM_SIM_BACKEND", "auto")).strip().lower()
    if raw in {"auto", "numba", "torch"}:
        return raw
    return "auto"


def active_sim_backend() -> str:
    backend = _resolve_sim_backend()
    if backend == "torch":
        return "torch"
    if _HAS_NUMBA:
        return "numba"
    return "torch"


def _unit_batch_to_physical_matrix(unit_params_batch: np.ndarray) -> np.ndarray:
    unit = np.asarray(unit_params_batch, dtype=np.float64)
    if unit.ndim == 1:
        unit = unit.reshape(1, -1)
    if unit.ndim != 2 or unit.shape[1] != len(PARAM_NAMES):
        raise ValueError(
            f"unit_params_batch shape is invalid: expected (N, {len(PARAM_NAMES)}), got {tuple(unit.shape)}"
        )

    unit = np.clip(unit, 0.0, 1.0)
    physical = _PARAM_LOWER[None, :] + unit * _PARAM_SPAN[None, :]
    return np.ascontiguousarray(physical, dtype=np.float64)


if _HAS_NUMBA:

    @nb.njit(cache=True, fastmath=True)
    def _simulate_population_q_only_numba_core(
        forcings: np.ndarray,
        f_veg: float,
        physical_params: np.ndarray,
    ) -> np.ndarray:
        pop_size = int(physical_params.shape[0])
        time_steps = int(forcings.shape[0])
        out = np.zeros((pop_size, time_steps), dtype=np.float64)

        f_veg_val = f_veg
        if f_veg_val < 0.0:
            f_veg_val = 0.0
        elif f_veg_val > 1.0:
            f_veg_val = 1.0

        for b in range(pop_size):
            s_max = physical_params[b, 0]
            s_wilt_frac = physical_params[b, 1]
            s_wilt = s_max * s_wilt_frac
            k_drain = physical_params[b, 2]
            lag_srf = physical_params[b, 3]
            lag_int = physical_params[b, 4]
            k_perc = physical_params[b, 5]
            lag_gw = physical_params[b, 6]
            beta_t = physical_params[b, 7]
            gamma_t = physical_params[b, 8]
            t_snow_thresh = physical_params[b, 9]
            t_snow_range = physical_params[b, 10]
            k_melt_day = physical_params[b, 11]
            k_melt_base = physical_params[b, 12]
            s_can_max = physical_params[b, 13]
            f_et_thresh = physical_params[b, 14]

            s_snow = 0.0
            s_can = 0.0
            s_soil = 0.0
            s_srf = 0.0
            s_gw_upper = 0.0
            s_gw_lower = 0.0

            for t in range(time_steps):
                p_t = forcings[t, 0]
                if p_t < 0.0:
                    p_t = 0.0

                t_air_t = forcings[t, 1]

                pet_t = forcings[t, 2]
                if pet_t < 0.0:
                    pet_t = 0.0

                day_len_frac_t = forcings[t, 3]
                if day_len_frac_t < 0.0:
                    day_len_frac_t = 0.0
                elif day_len_frac_t > 1.0:
                    day_len_frac_t = 1.0

                snow_frac = (t_snow_thresh - t_air_t) / (t_snow_range + 1.0e-6)
                if snow_frac < 0.0:
                    snow_frac = 0.0
                elif snow_frac > 1.0:
                    snow_frac = 1.0

                p_snow = p_t * snow_frac
                p_rain = p_t - p_snow
                available_snow = s_snow + p_snow
                if available_snow < 0.0:
                    available_snow = 0.0

                t_air_pos = t_air_t
                if t_air_pos < 0.0:
                    t_air_pos = 0.0

                melt_potential = (day_len_frac_t * k_melt_day + k_melt_base) * t_air_pos
                melt = melt_potential if melt_potential < available_snow else available_snow
                s_snow_next = available_snow - melt
                if s_snow_next < 0.0:
                    s_snow_next = 0.0

                p_veg = p_rain * f_veg_val
                p_bare = p_rain - p_veg

                canopy_capacity = s_can_max * f_veg_val
                if canopy_capacity < 1.0e-6:
                    canopy_capacity = 1.0e-6

                canopy_input = s_can + p_veg
                f_wet = canopy_input / (canopy_capacity + 1.0e-6)
                if f_wet < 0.0:
                    f_wet = 0.0
                elif f_wet > 1.0:
                    f_wet = 1.0

                e_can_pot = pet_t * f_wet * f_veg_val
                e_can = e_can_pot if e_can_pot < canopy_input else canopy_input
                r_can = canopy_input - e_can - canopy_capacity
                if r_can < 0.0:
                    r_can = 0.0
                r_tr = r_can + p_bare + melt

                s_can_next = canopy_input - e_can - r_can
                if s_can_next < 0.0:
                    s_can_next = 0.0
                if s_can_next > canopy_capacity:
                    s_can_next = canopy_capacity

                sat = s_soil / (s_max + 1.0e-6)
                if sat < 0.0:
                    sat = 0.0
                elif sat > 1.0:
                    sat = 1.0

                sat_pow = 0.0
                if sat > 0.0:
                    sat_eps = sat if sat > 1.0e-6 else 1.0e-6
                    sat_pow = sat_eps ** beta_t

                r_srf_gen = r_tr * sat_pow
                infiltration = r_tr - r_srf_gen
                if infiltration < 0.0:
                    infiltration = 0.0

                soil_after_infil = s_soil + infiltration
                if soil_after_infil < 0.0:
                    soil_after_infil = 0.0

                soil_overflow = soil_after_infil - s_max
                if soil_overflow < 0.0:
                    soil_overflow = 0.0

                soil_store_unbounded = soil_after_infil - soil_overflow
                if soil_store_unbounded < 0.0:
                    soil_store_unbounded = 0.0
                soil_store = soil_store_unbounded if soil_store_unbounded < s_max else s_max
                r_srf_total = r_srf_gen + soil_overflow

                r_gw_recharge = k_drain * soil_store
                if r_gw_recharge > soil_store:
                    r_gw_recharge = soil_store

                et_denom = f_et_thresh * s_max - s_wilt
                if et_denom < 1.0e-6:
                    et_denom = 1.0e-6

                et_factor = (soil_store - s_wilt) / et_denom
                if et_factor < 0.0:
                    et_factor = 0.0
                elif et_factor > 1.0:
                    et_factor = 1.0

                pet_minus_e_can = pet_t - e_can
                if pet_minus_e_can < 0.0:
                    pet_minus_e_can = 0.0
                e_soil_pot = pet_minus_e_can * et_factor * gamma_t

                available_soil = soil_store - r_gw_recharge
                if available_soil < 0.0:
                    available_soil = 0.0

                e_soil = e_soil_pot if e_soil_pot < available_soil else available_soil
                s_soil_next = soil_store - r_gw_recharge - e_soil
                if s_soil_next < 0.0:
                    s_soil_next = 0.0
                if s_soil_next > s_max:
                    s_soil_next = s_max

                k_srf = 1.0 / (lag_srf + 1.0)
                s_srf_available = s_srf + r_srf_total
                if s_srf_available < 0.0:
                    s_srf_available = 0.0
                q_srf_out = k_srf * s_srf_available
                s_srf_next = s_srf_available - q_srf_out
                if s_srf_next < 0.0:
                    s_srf_next = 0.0

                upper_available = s_gw_upper + r_gw_recharge
                if upper_available < 0.0:
                    upper_available = 0.0

                k_int = 1.0 / (lag_int + 1.0)
                q_int = k_int * upper_available
                perc_potential = k_perc * upper_available
                upper_remain = upper_available - q_int
                if upper_remain < 0.0:
                    upper_remain = 0.0
                perc_to_lower = perc_potential if perc_potential < upper_remain else upper_remain
                s_gw_upper_next = upper_available - q_int - perc_to_lower
                if s_gw_upper_next < 0.0:
                    s_gw_upper_next = 0.0

                lower_available = s_gw_lower + perc_to_lower
                if lower_available < 0.0:
                    lower_available = 0.0

                k_base = 1.0 / (lag_gw + 1.0)
                q_base = k_base * lower_available
                s_gw_lower_next = lower_available - q_base
                if s_gw_lower_next < 0.0:
                    s_gw_lower_next = 0.0

                q_total = q_srf_out + q_int + q_base
                if q_total < 0.0:
                    q_total = 0.0
                out[b, t] = q_total

                s_snow = s_snow_next
                s_can = s_can_next
                s_soil = s_soil_next
                s_srf = s_srf_next
                s_gw_upper = s_gw_upper_next
                s_gw_lower = s_gw_lower_next

        return out


def _simulate_population_with_numba(
    forcings: np.ndarray,
    f_veg: float,
    unit_params_batch: np.ndarray,
) -> np.ndarray:
    if not _HAS_NUMBA:
        raise RuntimeError("numba is not installed; numba forward backend is unavailable")

    forcings_arr = np.asarray(forcings, dtype=np.float32)
    if forcings_arr.ndim != 2 or forcings_arr.shape[1] != 4:
        raise ValueError("forcings must be a 2D array with 4 columns: [P, T, PET, Day_Length_Frac]")

    forcings_arr = np.ascontiguousarray(forcings_arr, dtype=np.float32)
    physical = _unit_batch_to_physical_matrix(unit_params_batch)
    f_veg_val = float(np.clip(float(f_veg), 0.0, 1.0))
    return _simulate_population_q_only_numba_core(
        forcings=forcings_arr,
        f_veg=f_veg_val,
        physical_params=physical,
    )


def parameter_names() -> List[str]:
    return list(PARAM_NAMES)


def unit_to_physical_params(unit_params: np.ndarray) -> Dict[str, float]:
    unit = np.asarray(unit_params, dtype=np.float64).reshape(-1)
    if unit.size != len(PARAM_NAMES):
        raise ValueError(f"Parameter dimension mismatch: expected {len(PARAM_NAMES)}, got {unit.size}")

    unit = np.clip(unit, 0.0, 1.0)
    out: Dict[str, float] = {}
    for i, name in enumerate(PARAM_NAMES):
        low, high = PARAM_RANGES[name]
        out[name] = float(low + unit[i] * (high - low))
    return out


def _build_snapshot(
    physical_params: Dict[str, float],
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> ParameterSnapshot:
    def _param_tensor(name: str) -> Tensor:
        low, high = PARAM_RANGES[name]
        raw_val = float(physical_params.get(name, low))
        val = float(np.clip(raw_val, low, high))
        return torch.full((batch_size, 1), val, dtype=dtype, device=device)

    s_max = _param_tensor("s_max")
    s_wilt_frac = _param_tensor("s_wilt_frac")
    return ParameterSnapshot(
        s_max=s_max,
        s_wilt_frac=s_wilt_frac,
        s_wilt=s_max * s_wilt_frac,
        k_drain=_param_tensor("k_drain"),
        lag_srf=_param_tensor("lag_srf"),
        lag_int=_param_tensor("lag_int"),
        k_perc=_param_tensor("k_perc"),
        lag_gw=_param_tensor("lag_gw"),
        beta_t=_param_tensor("beta_t"),
        gamma_t=_param_tensor("gamma_t"),
        t_snow_thresh=_param_tensor("t_snow_thresh"),
        t_snow_range=_param_tensor("t_snow_range"),
        k_melt_day=_param_tensor("k_melt_day"),
        k_melt_base=_param_tensor("k_melt_base"),
        s_can_max=_param_tensor("s_can_max"),
        f_et_thresh=_param_tensor("f_et_thresh"),
    )


def _build_snapshot_from_unit_params(
    unit_params_batch: np.ndarray,
    dtype: torch.dtype,
    device: torch.device,
) -> ParameterSnapshot:
    unit = np.asarray(unit_params_batch, dtype=np.float64)
    if unit.ndim == 1:
        unit = unit.reshape(1, -1)
    if unit.ndim != 2 or unit.shape[1] != len(PARAM_NAMES):
        raise ValueError(
            f"unit_params_batch shape is invalid: expected (N, {len(PARAM_NAMES)}), got {tuple(unit.shape)}"
        )

    unit = np.clip(unit, 0.0, 1.0)
    batch_size = int(unit.shape[0])
    if batch_size <= 0:
        raise ValueError("unit_params_batch cannot be empty")

    tensors: Dict[str, Tensor] = {}
    for i, name in enumerate(PARAM_NAMES):
        low, high = PARAM_RANGES[name]
        vals = low + unit[:, i] * (high - low)
        tensors[name] = torch.as_tensor(vals, dtype=dtype, device=device).reshape(batch_size, 1)

    s_max = tensors["s_max"]
    s_wilt_frac = tensors["s_wilt_frac"]
    return ParameterSnapshot(
        s_max=s_max,
        s_wilt_frac=s_wilt_frac,
        s_wilt=s_max * s_wilt_frac,
        k_drain=tensors["k_drain"],
        lag_srf=tensors["lag_srf"],
        lag_int=tensors["lag_int"],
        k_perc=tensors["k_perc"],
        lag_gw=tensors["lag_gw"],
        beta_t=tensors["beta_t"],
        gamma_t=tensors["gamma_t"],
        t_snow_thresh=tensors["t_snow_thresh"],
        t_snow_range=tensors["t_snow_range"],
        k_melt_day=tensors["k_melt_day"],
        k_melt_base=tensors["k_melt_base"],
        s_can_max=tensors["s_can_max"],
        f_et_thresh=tensors["f_et_thresh"],
    )


def simulate_pbm(forcings: Tensor, f_veg: Tensor, params: ParameterSnapshot) -> Dict[str, Tensor]:
    if forcings.ndim != 3 or forcings.size(-1) != 4:
        raise ValueError("forcings must have shape (B, T, 4)")
    if f_veg.ndim != 2 or f_veg.size(-1) != 1:
        raise ValueError("f_veg must have shape (B, 1)")

    batch_size = forcings.size(0)
    time_steps = forcings.size(1)
    dtype = forcings.dtype
    device = forcings.device

    q_total_steps: List[Tensor] = []
    q_srf_steps: List[Tensor] = []
    q_int_steps: List[Tensor] = []
    q_base_steps: List[Tensor] = []
    q_gw_steps: List[Tensor] = []
    e_total_steps: List[Tensor] = []
    s_total_steps: List[Tensor] = []
    p_steps: List[Tensor] = []

    s_snow = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    s_can = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    s_soil = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    s_srf = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    s_gw_upper = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    s_gw_lower = torch.zeros((batch_size, 1), dtype=dtype, device=device)

    s_max = params.s_max
    s_wilt = params.s_wilt
    k_drain = params.k_drain
    lag_srf = params.lag_srf
    lag_int = params.lag_int
    k_perc = params.k_perc
    lag_gw = params.lag_gw
    beta_t = params.beta_t
    gamma_t = params.gamma_t
    t_snow_thresh = params.t_snow_thresh
    t_snow_range = params.t_snow_range
    k_melt_day = params.k_melt_day
    k_melt_base = params.k_melt_base
    s_can_max = params.s_can_max
    f_et_thresh = params.f_et_thresh

    for t in range(time_steps):
        p_t = torch.clamp(forcings[:, t, 0:1], min=0.0)
        t_air_t = forcings[:, t, 1:2]
        pet_t = torch.clamp(forcings[:, t, 2:3], min=0.0)
        day_len_frac_t = torch.clamp(forcings[:, t, 3:4], min=0.0, max=1.0)

        snow_frac = torch.clamp((t_snow_thresh - t_air_t) / (t_snow_range + 1e-6), min=0.0, max=1.0)
        p_snow = p_t * snow_frac
        p_rain = p_t - p_snow
        available_snow = torch.clamp(s_snow + p_snow, min=0.0)

        melt_potential = (day_len_frac_t * k_melt_day + k_melt_base) * torch.clamp(t_air_t, min=0.0)
        melt = torch.minimum(melt_potential, available_snow)
        s_snow_next = torch.clamp(available_snow - melt, min=0.0)

        p_veg = p_rain * f_veg
        p_bare = p_rain - p_veg

        canopy_capacity = torch.clamp(s_can_max * f_veg, min=1e-6)
        canopy_input = s_can + p_veg
        f_wet = torch.clamp(canopy_input / (canopy_capacity + 1e-6), min=0.0, max=1.0)
        e_can_pot = pet_t * f_wet * f_veg
        e_can = torch.minimum(e_can_pot, canopy_input)
        r_can = torch.clamp(canopy_input - e_can - canopy_capacity, min=0.0)
        r_tr = r_can + p_bare + melt
        s_can_next = torch.clamp(canopy_input - e_can - r_can, min=0.0)
        s_can_next = torch.minimum(s_can_next, canopy_capacity)

        sat = torch.clamp(s_soil / (s_max + 1e-6), min=0.0, max=1.0)
        sat_pow = torch.pow(torch.clamp(sat, min=1e-6), beta_t)
        sat_pow = torch.where(sat > 0.0, sat_pow, torch.zeros_like(sat_pow))

        r_srf_gen = r_tr * sat_pow
        infiltration = torch.clamp(r_tr - r_srf_gen, min=0.0)

        soil_after_infil = torch.clamp(s_soil + infiltration, min=0.0)
        soil_overflow = torch.clamp(soil_after_infil - s_max, min=0.0)
        soil_store_unbounded = torch.clamp(soil_after_infil - soil_overflow, min=0.0)
        soil_store = torch.minimum(soil_store_unbounded, s_max)
        r_srf_total = r_srf_gen + soil_overflow

        r_gw_recharge = torch.minimum(k_drain * soil_store, soil_store)
        et_denom = torch.clamp(f_et_thresh * s_max - s_wilt, min=1e-6)
        et_factor = torch.clamp((soil_store - s_wilt) / et_denom, min=0.0, max=1.0)
        e_soil_pot = torch.clamp(pet_t - e_can, min=0.0) * et_factor * gamma_t

        available_soil = torch.clamp(soil_store - r_gw_recharge, min=0.0)
        e_soil = torch.minimum(e_soil_pot, available_soil)
        s_soil_next_unbounded = torch.clamp(soil_store - r_gw_recharge - e_soil, min=0.0)
        s_soil_next = torch.minimum(s_soil_next_unbounded, s_max)

        k_srf = 1.0 / (lag_srf + 1.0)
        s_srf_available = torch.clamp(s_srf + r_srf_total, min=0.0)
        q_srf_out = k_srf * s_srf_available
        s_srf_next = torch.clamp(s_srf_available - q_srf_out, min=0.0)

        upper_available = torch.clamp(s_gw_upper + r_gw_recharge, min=0.0)
        k_int = 1.0 / (lag_int + 1.0)
        q_int = k_int * upper_available
        perc_potential = k_perc * upper_available
        perc_to_lower = torch.minimum(perc_potential, torch.clamp(upper_available - q_int, min=0.0))
        s_gw_upper_next = torch.clamp(upper_available - q_int - perc_to_lower, min=0.0)

        lower_available = torch.clamp(s_gw_lower + perc_to_lower, min=0.0)
        k_base = 1.0 / (lag_gw + 1.0)
        q_base = k_base * lower_available
        s_gw_lower_next = torch.clamp(lower_available - q_base, min=0.0)

        q_gw = q_int + q_base
        q_total = torch.clamp(q_srf_out + q_int + q_base, min=0.0)
        s_total = s_snow_next + s_can_next + s_soil_next + s_srf_next + s_gw_upper_next + s_gw_lower_next
        e_total = e_can + e_soil

        q_total_steps.append(q_total)
        q_srf_steps.append(q_srf_out)
        q_int_steps.append(q_int)
        q_base_steps.append(q_base)
        q_gw_steps.append(q_gw)
        e_total_steps.append(e_total)
        s_total_steps.append(s_total)
        p_steps.append(p_t)

        s_snow = s_snow_next
        s_can = s_can_next
        s_soil = s_soil_next
        s_srf = s_srf_next
        s_gw_upper = s_gw_upper_next
        s_gw_lower = s_gw_lower_next

    return {
        "q_pred": torch.stack(q_total_steps, dim=1),
        "q_srf": torch.stack(q_srf_steps, dim=1),
        "q_int": torch.stack(q_int_steps, dim=1),
        "q_base": torch.stack(q_base_steps, dim=1),
        "q_gw": torch.stack(q_gw_steps, dim=1),
        "e_total": torch.stack(e_total_steps, dim=1),
        "s_total": torch.stack(s_total_steps, dim=1),
        "p_seq": torch.stack(p_steps, dim=1),
    }


def simulate_pbm_q_only(forcings: Tensor, f_veg: Tensor, params: ParameterSnapshot) -> Tensor:
    if forcings.ndim != 3 or forcings.size(-1) != 4:
        raise ValueError("forcings must have shape (B, T, 4)")
    if f_veg.ndim != 2 or f_veg.size(-1) != 1:
        raise ValueError("f_veg must have shape (B, 1)")

    batch_size = forcings.size(0)
    time_steps = forcings.size(1)
    dtype = forcings.dtype
    device = forcings.device

    q_total_steps: List[Tensor] = []

    s_snow = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    s_can = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    s_soil = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    s_srf = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    s_gw_upper = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    s_gw_lower = torch.zeros((batch_size, 1), dtype=dtype, device=device)

    s_max = params.s_max
    s_wilt = params.s_wilt
    k_drain = params.k_drain
    lag_srf = params.lag_srf
    lag_int = params.lag_int
    k_perc = params.k_perc
    lag_gw = params.lag_gw
    beta_t = params.beta_t
    gamma_t = params.gamma_t
    t_snow_thresh = params.t_snow_thresh
    t_snow_range = params.t_snow_range
    k_melt_day = params.k_melt_day
    k_melt_base = params.k_melt_base
    s_can_max = params.s_can_max
    f_et_thresh = params.f_et_thresh

    for t in range(time_steps):
        p_t = torch.clamp(forcings[:, t, 0:1], min=0.0)
        t_air_t = forcings[:, t, 1:2]
        pet_t = torch.clamp(forcings[:, t, 2:3], min=0.0)
        day_len_frac_t = torch.clamp(forcings[:, t, 3:4], min=0.0, max=1.0)

        snow_frac = torch.clamp((t_snow_thresh - t_air_t) / (t_snow_range + 1e-6), min=0.0, max=1.0)
        p_snow = p_t * snow_frac
        p_rain = p_t - p_snow
        available_snow = torch.clamp(s_snow + p_snow, min=0.0)

        melt_potential = (day_len_frac_t * k_melt_day + k_melt_base) * torch.clamp(t_air_t, min=0.0)
        melt = torch.minimum(melt_potential, available_snow)
        s_snow_next = torch.clamp(available_snow - melt, min=0.0)

        p_veg = p_rain * f_veg
        p_bare = p_rain - p_veg

        canopy_capacity = torch.clamp(s_can_max * f_veg, min=1e-6)
        canopy_input = s_can + p_veg
        f_wet = torch.clamp(canopy_input / (canopy_capacity + 1e-6), min=0.0, max=1.0)
        e_can_pot = pet_t * f_wet * f_veg
        e_can = torch.minimum(e_can_pot, canopy_input)
        r_can = torch.clamp(canopy_input - e_can - canopy_capacity, min=0.0)
        r_tr = r_can + p_bare + melt
        s_can_next = torch.clamp(canopy_input - e_can - r_can, min=0.0)
        s_can_next = torch.minimum(s_can_next, canopy_capacity)

        sat = torch.clamp(s_soil / (s_max + 1e-6), min=0.0, max=1.0)
        sat_pow = torch.pow(torch.clamp(sat, min=1e-6), beta_t)
        sat_pow = torch.where(sat > 0.0, sat_pow, torch.zeros_like(sat_pow))

        r_srf_gen = r_tr * sat_pow
        infiltration = torch.clamp(r_tr - r_srf_gen, min=0.0)

        soil_after_infil = torch.clamp(s_soil + infiltration, min=0.0)
        soil_overflow = torch.clamp(soil_after_infil - s_max, min=0.0)
        soil_store_unbounded = torch.clamp(soil_after_infil - soil_overflow, min=0.0)
        soil_store = torch.minimum(soil_store_unbounded, s_max)
        r_srf_total = r_srf_gen + soil_overflow

        r_gw_recharge = torch.minimum(k_drain * soil_store, soil_store)
        et_denom = torch.clamp(f_et_thresh * s_max - s_wilt, min=1e-6)
        et_factor = torch.clamp((soil_store - s_wilt) / et_denom, min=0.0, max=1.0)
        e_soil_pot = torch.clamp(pet_t - e_can, min=0.0) * et_factor * gamma_t

        available_soil = torch.clamp(soil_store - r_gw_recharge, min=0.0)
        e_soil = torch.minimum(e_soil_pot, available_soil)
        s_soil_next_unbounded = torch.clamp(soil_store - r_gw_recharge - e_soil, min=0.0)
        s_soil_next = torch.minimum(s_soil_next_unbounded, s_max)

        k_srf = 1.0 / (lag_srf + 1.0)
        s_srf_available = torch.clamp(s_srf + r_srf_total, min=0.0)
        q_srf_out = k_srf * s_srf_available
        s_srf_next = torch.clamp(s_srf_available - q_srf_out, min=0.0)

        upper_available = torch.clamp(s_gw_upper + r_gw_recharge, min=0.0)
        k_int = 1.0 / (lag_int + 1.0)
        q_int = k_int * upper_available
        perc_potential = k_perc * upper_available
        perc_to_lower = torch.minimum(perc_potential, torch.clamp(upper_available - q_int, min=0.0))
        s_gw_upper_next = torch.clamp(upper_available - q_int - perc_to_lower, min=0.0)

        lower_available = torch.clamp(s_gw_lower + perc_to_lower, min=0.0)
        k_base = 1.0 / (lag_gw + 1.0)
        q_base = k_base * lower_available
        s_gw_lower_next = torch.clamp(lower_available - q_base, min=0.0)

        q_total = torch.clamp(q_srf_out + q_int + q_base, min=0.0)
        q_total_steps.append(q_total)

        s_snow = s_snow_next
        s_can = s_can_next
        s_soil = s_soil_next
        s_srf = s_srf_next
        s_gw_upper = s_gw_upper_next
        s_gw_lower = s_gw_lower_next

    return torch.stack(q_total_steps, dim=1)


def simulate_station_series(
    forcings: np.ndarray,
    f_veg: float,
    physical_params: Dict[str, float],
) -> np.ndarray:
    forcings_arr = np.asarray(forcings, dtype=np.float32)
    if forcings_arr.ndim != 2 or forcings_arr.shape[1] != 4:
        raise ValueError("forcings must be a 2D array with 4 columns: [P, T, PET, Day_Length_Frac]")

    forcings_t = torch.as_tensor(forcings_arr, dtype=torch.float32).unsqueeze(0)
    f_veg_t = torch.full((1, 1), float(np.clip(f_veg, 0.0, 1.0)), dtype=torch.float32)

    snapshot = _build_snapshot(
        physical_params=physical_params,
        batch_size=1,
        dtype=forcings_t.dtype,
        device=forcings_t.device,
    )
    with torch.no_grad():
        q_pred = simulate_pbm_q_only(forcings=forcings_t, f_veg=f_veg_t, params=snapshot)
    return q_pred[0, :, 0].detach().cpu().numpy().astype(np.float64, copy=False)


def simulate_station_with_unit_params(
    forcings: np.ndarray,
    f_veg: float,
    unit_params: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    physical = unit_to_physical_params(unit_params)
    q_pred = simulate_station_series(forcings=forcings, f_veg=f_veg, physical_params=physical)
    return q_pred, physical


def simulate_station_population_with_unit_params(
    forcings: np.ndarray,
    f_veg: float,
    unit_params_batch: np.ndarray,
) -> np.ndarray:
    backend = _resolve_sim_backend()
    if backend != "torch":
        if _HAS_NUMBA:
            return _simulate_population_with_numba(
                forcings=forcings,
                f_veg=f_veg,
                unit_params_batch=unit_params_batch,
            )
        if backend == "numba":
            raise RuntimeError("PBM_SIM_BACKEND=numba but numba is not installed in the current environment")

    forcings_arr = np.asarray(forcings, dtype=np.float32)
    if forcings_arr.ndim != 2 or forcings_arr.shape[1] != 4:
        raise ValueError("forcings must be a 2D array with 4 columns: [P, T, PET, Day_Length_Frac]")

    unit = np.asarray(unit_params_batch, dtype=np.float64)
    if unit.ndim == 1:
        unit = unit.reshape(1, -1)
    if unit.ndim != 2 or unit.shape[1] != len(PARAM_NAMES):
        raise ValueError(
            f"unit_params_batch shape is invalid: expected (N, {len(PARAM_NAMES)}), got {tuple(unit.shape)}"
        )

    pop_size = int(unit.shape[0])
    if pop_size <= 0:
        raise ValueError("unit_params_batch cannot be empty")

    forcings_t = torch.as_tensor(forcings_arr, dtype=torch.float32).unsqueeze(0).expand(pop_size, -1, -1)
    f_veg_t = torch.full((pop_size, 1), float(np.clip(f_veg, 0.0, 1.0)), dtype=torch.float32)

    snapshot = _build_snapshot_from_unit_params(
        unit_params_batch=unit,
        dtype=forcings_t.dtype,
        device=forcings_t.device,
    )
    with torch.no_grad():
        q_pred = simulate_pbm_q_only(forcings=forcings_t, f_veg=f_veg_t, params=snapshot)
    return q_pred[:, :, 0].detach().cpu().numpy().astype(np.float64, copy=False)