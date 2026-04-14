# PBM Station-wise CMA-ES Calibration — Documentation

## 1. What this version does
This version aims to establish the station-wise upper bound (Upper Bound / Oracle) of the PBM and provide a trustworthy reference for subsequent HPA-MoE work.

The calibration workflow is:
1. Read the long hydrological table and attach cluster labels.
2. Enforce that the detected number of clusters equals the configured value (default 7).
3. Optionally run CMA-ES pretraining at the cluster level (warm start).
4. For each station, continue with CMA-ES fine-tuning starting from the corresponding cluster initial guess.
5. Export station parameters, metrics, test timeseries and optimization history.

## 2. Which parameters CMA-ES optimizes
### 2.1 PBM physical parameters (15)
They are defined in `PARAM_RANGES`. CMA-ES searches in the unit hypercube `[0,1]^15` and then linearly maps to the physical ranges.

1. s_max
2. s_wilt_frac
3. k_drain
4. lag_srf
5. lag_int
6. k_perc
7. lag_gw
8. beta_t
9. gamma_t
10. t_snow_thresh
11. t_snow_range
12. k_melt_day
13. k_melt_base
14. s_can_max
15. f_et_thresh

### 2.1.1 Per-parameter English descriptions
1. `s_max`: Maximum soil water storage capacity; larger values mean higher soil storage capacity.
2. `s_wilt_frac`: Fraction representing the wilting point; used to determine soil moisture level where evapotranspiration becomes strongly limited.
3. `k_drain`: Coefficient for drainage from the soil to groundwater; larger values increase groundwater recharge rate.
4. `lag_srf`: Lag parameter for surface quick flow routing; larger values indicate slower and smoother surface responses.
5. `lag_int`: Lag parameter for interflow (upper groundwater layer), controlling intermediate drainage response speed.
6. `k_perc`: Percolation coefficient from upper to lower groundwater layers, affecting baseflow formation strength.
7. `lag_gw`: Lag parameter for lower groundwater (baseflow); larger values indicate slower baseflow recession.
8. `beta_t`: Nonlinearity exponent for runoff generation, controlling the amplification of surface runoff with soil wetness.
9. `gamma_t`: Coefficient that modulates soil evaporation relative to potential evapotranspiration.
10. `t_snow_thresh`: Temperature threshold for rain/snow partitioning; below this threshold precipitation is more likely snow.
11. `t_snow_range`: Transition temperature range for rain/snow partitioning; controls smoothing over the rain-snow transition.
12. `k_melt_day`: Day-length dependent melt coefficient that scales melting during daytime.
13. `k_melt_base`: Baseline melt coefficient independent of day length.
14. `s_can_max`: Maximum canopy interception capacity; larger values mean vegetation can temporarily store more precipitation.
15. `f_et_thresh`: Fractional threshold for evapotranspiration limitation; controls when ET begins to be noticeably limited by soil moisture.

### 2.2 CMA-ES hyperparameters
Configurable in `TrainConfig`:
- `cma_population_size`: Population size per generation.
- `cma_max_iterations`: Maximum number of generations per run.
- `cma_sigma`: Initial step-size.
- `cma_patience`: Patience in generations with no improvement.
- `cma_restarts`: Number of restarts.
- `cma_retry_failed`: Whether to automatically retry failed runs.
- `cma_retry_population_scale`: Population scale factor for retries.
- `cma_retry_iteration_scale`: Iteration scale factor for retries.

Efficiency-related options:
- `enable_cluster_warmstart`: Enable cluster pretraining before per-station fine-tuning.
- `num_workers`: Number of parallel worker processes; `0` means auto.
- `progress_print_every_generation`: In serial mode, print intermediate progress every N generations.
- `resume_skip_completed`: Enable skipping already-completed stations when resuming.
- `resume_require_hash_match`: Require config+data hash match to allow skipping.

## 3. Are there physical constraints?
Yes — strict constraints are enforced.

### 3.1 Parameter bounds
Each parameter is confined to the interval defined in `PARAM_RANGES`. Mapping formula:

- Let unit parameter be `u in [0,1]`, physical lower bound `L`, upper bound `H`.
- Physical parameter `theta = L + u * (H - L)`.

This ensures CMA-ES search stays within physically interpretable ranges.

### 3.2 Process-level constraints
The PBM simulation includes process-level safeguards (non-negative truncation, capacity limits, proportion bounds, etc.) to prevent non-physical states such as negative storage or negative flows.

## 4. Terminal progress logging (real-time monitoring)
There are three levels of progress output:

1. Cluster pretraining progress:
   - Each cluster prints start, periodic generation summaries, and final best loss.

2. Station-wise calibration progress:
   - After each station completes it prints: completed count, station ID, cluster, status, best loss, test NSE, cumulative elapsed time, ETA.

3. Per-station generation progress (serial mode):
   - When `num_workers <= 1`, the trainer prints generation-level updates every `progress_print_every_generation` generations.

## 5. Efficiency-oriented design
This version prioritizes computational efficiency using:

1. Parallel calibration
   - Use `num_workers` to enable multi-process parallelism. `num_workers=0` uses `CPU count - 1`.

2. Cluster pretraining + station fine-tuning (warm start)
   - Each cluster is pre-optimized with a small budget to obtain a starting point.
   - Station fine-tuning starts from the cluster initial guess for faster convergence than random init.

3. Adaptive retry on failure
   - If a main run fails, automatically increase population/iterations and retry once.

4. Strict hash-based resume (prevents accidental reuse)
   - `resume_skip_completed=True` by default.
   - `resume_require_hash_match=True` by default.
   - A completed station is skipped only if BOTH:
     1. `Config_Hash` in historical results matches the current run configuration hash.
     2. `Station_Data_Hash` in historical results matches the current station data hash.
   - `Config_Hash` includes: training config, parameter list, and content hashes of key code files (e.g., `trainer.py`, `model_pbm.py`).
   - If configuration or station data change, the station will be recomputed to avoid incorrectly reusing old results.

## 6. Should I enable cluster pretraining before per-station fine-tuning?
If your goal is computational efficiency, enabling warmstart is recommended (default on):
- Pros: faster and more stable, especially with many stations.
- Cons: adds a pretraining phase, but total runtime usually decreases.

If you need a strict method-comparison experiment, you may disable warmstart to compare "random init vs cluster init".

## 7. Strict 7-cluster detection and abort behavior
The pipeline supports strict cluster-count checking (`strict_cluster_count=True` by default):

1. Detect the number of clusters in input.
2. If it does not equal `expected_cluster_count` (default 7), the run aborts with an error.
3. After station filtering, the cluster count is checked again; if still different, the run aborts.

This prevents running on corrupted cluster labels or mismatched station mappings.

## 8. Typical run examples
Run from the `PBM参数训练` directory:

```bash
python main.py --num_workers 0 --enable_cluster_warmstart --expected_cluster_count 7 --strict_cluster_count
```

Small smoke test:

```bash
python main.py --max_station_count 30 --num_workers 4 --cma_max_iterations 80
```

Disable warmstart for control experiments:

```bash
python main.py --no-enable_cluster_warmstart --num_workers 0
```

## 9. Key output files
- `station_metrics.csv`: station-level Train/Val/Test metrics.
- `station_best_parameters.csv`: best physical parameters per station.
- `test_timeseries_predictions.csv`: daily test-period predictions and observations.
- `cma_history.csv`: generational optimization traces (including cluster pretraining and station fine-tuning).
- `summary.json`: aggregated statistics (including timing metrics).
- `upper_bound_report.txt`: human-readable report.

## 10. NSE/KGE/RMSE/Bias plots (reference style)
The run also exports the following plots (based on test metrics):
1. `NSE_KGE_boxplot.jpg/.pdf`
2. `RMSE_Bias_boxplot.jpg/.pdf`

Plot style follows the reference scripts:
- Boxplot + red mean marker + black dashed median line.
- Jittered scatter overlay with black edge.
- Optional one-sided KDE contour if `scipy` is available.
- Arial font, legend contains Mean/Median.
- Also exports a statistics table: `metrics_statistics.csv`.

