# -*- coding: utf-8 -*-
'''
@File    :   runtime_env.py
@Time    :   2026-04-04
@Desc    :   Utilities to handle common OpenMP runtime conflicts on Windows platforms, especially the
           libiomp5md.dll duplicate-load error that can arise when torch, numpy, scipy, and pandas
           are installed from mixed channels. PBM calibration scripts perform extensive tensor
           operations and may fail at startup with "OMP Error #15"; calling this initialization
           function at the program entry can significantly improve robustness.
           The strategy is a conservative fallback: unless the user explicitly requests strict
           behavior, on Windows the function sets `KMP_DUPLICATE_LIB_OK=TRUE` to avoid startup
           interruption caused by runtime conflicts. This is a pragmatic workaround and should be
           replaced by unifying package installation channels (conda/pip) for a permanent fix.
           The module performs only lightweight environment-variable initialization and does not
           alter model logic.
@Notice  :   For a long-term solution, unify dependency sources; this fallback is intended to keep
           experiments runnable in teaching or migration environments.
'''

from __future__ import annotations

import logging
import os


_OPENMP_CONFIG_DONE_ENV = "PBM_OPENMP_RUNTIME_CONFIG_DONE"
_NUMERIC_THREADS_CONFIG_DONE_ENV = "PBM_NUMERIC_THREADS_CONFIG_DONE"

_NUMERIC_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _resolve_numeric_thread_cap() -> int:
    raw = str(os.environ.get("PBM_NUMERIC_THREAD_CAP", "1")).strip()
    try:
        val = int(raw)
    except ValueError:
        return 1
    return max(1, val)


def _resolve_numeric_thread_override() -> bool:
    raw = str(os.environ.get("PBM_FORCE_NUMERIC_THREAD_CAP", "true")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _configure_numeric_threads() -> None:
    if os.environ.get(_NUMERIC_THREADS_CONFIG_DONE_ENV, "") == "1":
        return

    cap = str(_resolve_numeric_thread_cap())
    force_override = _resolve_numeric_thread_override()

    for key in _NUMERIC_THREAD_ENV_KEYS:
        if force_override:
            os.environ[key] = cap
        else:
            os.environ.setdefault(key, cap)

    os.environ[_NUMERIC_THREADS_CONFIG_DONE_ENV] = "1"


def _resolve_openmp_policy() -> str:
    raw = str(os.environ.get("PBM_OPENMP_DUPLICATE_POLICY", "allow")).strip().lower()
    if raw in {"strict", "deny", "off", "0", "false", "no"}:
        return "strict"
    return "allow"


def configure_openmp_runtime() -> None:
    # Enforce numeric thread caps first to avoid oversubscription from process-level parallelism + internal thread pools.
    _configure_numeric_threads()

    if os.name != "nt":
        return
    if os.environ.get(_OPENMP_CONFIG_DONE_ENV, "") == "1":
        return

    current_flag = str(os.environ.get("KMP_DUPLICATE_LIB_OK", "")).strip().upper() == "TRUE"
    policy = _resolve_openmp_policy()

    if current_flag or policy == "strict":
        os.environ[_OPENMP_CONFIG_DONE_ENV] = "1"
        return

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ[_OPENMP_CONFIG_DONE_ENV] = "1"
    logging.info(
        "[OpenMP] Enabled KMP_DUPLICATE_LIB_OK as compatibility fallback. "
        "Please unify numpy/torch/scipy installation channels for permanent fix."
    )
