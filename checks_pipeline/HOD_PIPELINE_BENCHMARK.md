# HOD pipeline benchmark

**No successful real-simulation timings are available yet.**

Completed points: **0/140**. Output directory: `/global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/checks_pipeline/results/hod_benchmark/60031882`.

## Measurement definition

- Timed operation: `HOD.make_mock_cat(fix_seed=..., verbose=False)`, including density normalization, central/satellite draws, positions, velocities and catalogue construction.
- Catalogue loading, HOD construction, capacity checks, warm-ups/JIT, mock counting, garbage collection and catalogue deletion are outside the repeated generation timings. Warm-ups also initialize the Uchuu subhalo index. Per-call subhalo mapping inside the sampler remains timed.
- 1 warm-up call(s) per point; 5 timed draws. Plots show medians with the 25th–75th percentile range. Seeds repeat across comparable points; different thread counts need not give identical mock catalogues.
- Thread counts are logical CPUs. Both the Numba mask and HOD/loader thread counts are set and checked. Points exceeding the job CPU allocation or Numba capacity are skipped, never silently clamped.
- CPU eligibility is captured before importing threaded libraries. OpenMP can pin its primary thread to one core; that later mask is not treated as the job CPU budget. Reader/benchmark child processes inherit the original task mask.
- One SHOD shape is used across boxes, profiles and densities. Its low mass cutoff is chosen to support 1e-2; capacity is validated per catalogue. This is an LRG-style SHOD performance model, not a calibrated DESI LRG HOD. Only occupation amplitudes are density-normalized. Assembly bias and conformity are off.
- A point fails if the scaled central amplitude exceeds one or its realized density fails the max(5.0%, 5/sqrt(expected Ngal)) tolerance. Particle/subhalo modes must place at least some satellites on actual auxiliary objects. The warm-up NFW fallback fraction is reported; it is not assumed to be zero.
- Uchuu uses the native loader. Its subhalos are matched to hosts using `upid`, retaining their original IDs as `subhalo_id`; this mapping is outside the timed calls. Compressed HDF5 filters are registered with hdf5plugin before loading.
- Separate fresh processes load each box/profile once, so catalogues do not coexist in memory. Worker logs and partial JSON records survive interruptions.
- Abacus and Uchuu differ in halo finder, resolution and cosmology. Snapshots are explicitly recorded; the default Abacus z=0.5 and Uchuu z=0.49 are close but not identical.
- Clustering is not part of this benchmark.

## Plots

![Time versus threads](results/hod_benchmark/60031882/thread_scaling.png)

![Time versus target density, Abacus base](results/hod_benchmark/60031882/density_scaling.png)

## Saved numbers and provenance

- [Individual timed draws](results/hod_benchmark/60031882/timings.csv)
- [Aggregated timings and failed/skipped points](results/hod_benchmark/60031882/summary.csv)
- [Configuration and hardware](results/hod_benchmark/60031882/manifest.json)

Case JSON files include loading/initialization/warm-up times, realized density, satellite fraction, fallback diagnostics, per-point hardware and worker peak RSS (Linux ru_maxrss, including loading and JIT).

Driver environment at manifest creation is listed below; execution hardware for each measured point is stored in CSV and case JSON files.

| Hardware / software setting | Value |
| --- | --- |
| host | nid200052 |
| cpu_model | AMD EPYC 7763 64-Core Processor |
| available_logical_cpus | 2 |
| python | 3.12.9 |
| platform | Linux-6.4.0-150600.23.135_15.0.44-cray_shasta_c-x86_64-with-glibc2.38 |
| slurm_job | 60031882 |
| slurm_cpus_per_task | — |
| OMP_NUM_THREADS | 256 |
| OMP_THREAD_LIMIT | — |
| OMP_DYNAMIC | — |
| NUMBA_NUM_THREADS | 256 |
| OPENBLAS_NUM_THREADS | 1 |
| MKL_NUM_THREADS | 1 |
| git_revision | 68537bc85c5ee92b6fc2c04f047fe072b9389bab |
| git_dirty | True |

## Catalogue setup

| Box | Path | Simulation | Snapshot z | Expected size [Mpc/h] |
| --- | --- | --- | --- | --- |
| Abacus base 2 Gpc/h | /global/cfs/cdirs/desi/cosmosim/Abacus/AbacusSummit_base_c000_ph000/halos/z0.500 | AbacusSummit_base_c000_ph000 | 0.5 | 2000 |
| Abacus highbase 1 Gpc/h | /global/cfs/cdirs/desi/cosmosim/Abacus/AbacusSummit_highbase_c000_ph100/halos/z0.500 | AbacusSummit_highbase_c000_ph100 | 0.5 | 1000 |
| Uchuu 2 Gpc/h | /global/homes/a/arocher/users_arocher/HODDIES_data/Uchuu/Uchuu_halo_catalogs/halodir_040 | Uchuu2Gpc | 0.49 | 2000 |

## SHOD input parameters

| Parameter | Value |
| --- | --- |
| HOD_model | SHOD |
| Ac | 1 |
| log_Mcent | 11 |
| sigma_M | 0.4 |
| satellites | True |
| sat_HOD_model | Nsat_pow_law |
| As | 1 |
| M_0 | 11 |
| M_1 | 13 |
| alpha | 1 |
| link_sat_to_central | False |
| conformity_bias | False |
| nu | 1 |
| f_sigv | 1 |
| f_vcen | 0 |
| vel_sat | rd_normal |
| v_infall | 0 |
| exp_frac | 0 |
| exp_scale | 1 |
| nfw_rescale | 1 |
| vsmear | — |
| assembly_bias | {'c': [0.0, 0.0]} |

Optional host mass cut: `None` in log10(Msun/h). Uchuu subhalos are not removed by this host selection.

## Fixed density: timings

| Box | Profile | Threads | Median [s] | Q25 [s] | Q75 [s] | Mean density | Warm-up NFW fallback | State |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Abacus base density scan: nfw

| Density [(h/Mpc)³] | 4 threads [s] | 8 threads [s] | 16 threads [s] | 32 threads [s] | 64 threads [s] | 128 threads [s] | 256 threads [s] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0001 | pending | pending | pending | pending | pending | pending | pending |
| 0.0002 | pending | pending | pending | pending | pending | pending | pending |
| 0.0005 | pending | pending | pending | pending | pending | pending | pending |
| 0.001 | pending | pending | pending | pending | pending | pending | pending |
| 0.002 | pending | pending | pending | pending | pending | pending | pending |
| 0.005 | pending | pending | pending | pending | pending | pending | pending |
| 0.01 | pending | pending | pending | pending | pending | pending | pending |

## Abacus base density scan: particles

| Density [(h/Mpc)³] | 4 threads [s] | 8 threads [s] | 16 threads [s] | 32 threads [s] | 64 threads [s] | 128 threads [s] | 256 threads [s] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0001 | pending | pending | pending | pending | pending | pending | pending |
| 0.0002 | pending | pending | pending | pending | pending | pending | pending |
| 0.0005 | pending | pending | pending | pending | pending | pending | pending |
| 0.001 | pending | pending | pending | pending | pending | pending | pending |
| 0.002 | pending | pending | pending | pending | pending | pending | pending |
| 0.005 | pending | pending | pending | pending | pending | pending | pending |
| 0.01 | pending | pending | pending | pending | pending | pending | pending |

## Cases, preparation and issues

### abacus_base__nfw

| Setting | Value |
| --- | --- |
| state | failed |
| boxsize | — |
| snapshot_z | — |
| halo_count | — |
| auxiliary_count | — |
| load_seconds | — |

[Worker log](results/hod_benchmark/60031882/logs/abacus_base__nfw.log); [Case measurements](results/hod_benchmark/60031882/cases/abacus_base__nfw.json).

Case error: None of the requested thread counts fits this CPU allocation.


### abacus_base__particles

| Setting | Value |
| --- | --- |
| state | failed |
| boxsize | — |
| snapshot_z | — |
| halo_count | — |
| auxiliary_count | — |
| load_seconds | — |

[Worker log](results/hod_benchmark/60031882/logs/abacus_base__particles.log); [Case measurements](results/hod_benchmark/60031882/cases/abacus_base__particles.json).

Case error: None of the requested thread counts fits this CPU allocation.


### abacus_highbase__nfw

| Setting | Value |
| --- | --- |
| state | failed |
| boxsize | — |
| snapshot_z | — |
| halo_count | — |
| auxiliary_count | — |
| load_seconds | — |

[Worker log](results/hod_benchmark/60031882/logs/abacus_highbase__nfw.log); [Case measurements](results/hod_benchmark/60031882/cases/abacus_highbase__nfw.json).

Case error: None of the requested thread counts fits this CPU allocation.


### abacus_highbase__particles

| Setting | Value |
| --- | --- |
| state | failed |
| boxsize | — |
| snapshot_z | — |
| halo_count | — |
| auxiliary_count | — |
| load_seconds | — |

[Worker log](results/hod_benchmark/60031882/logs/abacus_highbase__particles.log); [Case measurements](results/hod_benchmark/60031882/cases/abacus_highbase__particles.json).

Case error: None of the requested thread counts fits this CPU allocation.


### uchuu__nfw

| Setting | Value |
| --- | --- |
| state | failed |
| boxsize | — |
| snapshot_z | — |
| halo_count | — |
| auxiliary_count | — |
| load_seconds | — |

[Worker log](results/hod_benchmark/60031882/logs/uchuu__nfw.log); [Case measurements](results/hod_benchmark/60031882/cases/uchuu__nfw.json).

Case error: None of the requested thread counts fits this CPU allocation.


### uchuu__particles

| Setting | Value |
| --- | --- |
| state | failed |
| boxsize | — |
| snapshot_z | — |
| halo_count | — |
| auxiliary_count | — |
| load_seconds | — |

[Worker log](results/hod_benchmark/60031882/logs/uchuu__particles.log); [Case measurements](results/hod_benchmark/60031882/cases/uchuu__particles.json).

Case error: None of the requested thread counts fits this CPU allocation.


## Run / refresh

From the repository root in the cosmodesi environment:

```bash
python checks_pipeline/benchmark_hod_pipeline.py --check-inputs
sbatch checks_pipeline/run_hod_benchmark.sh
python checks_pipeline/benchmark_hod_pipeline.py --report-only --output /global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/checks_pipeline/results/hod_benchmark/60031882
python checks_pipeline/benchmark_hod_pipeline.py --resume --output /global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/checks_pipeline/results/hod_benchmark/60031882
```

A smoke test can be run with `--smoke-test --boxes abacus_base --threads 4 8 --repeats 2`. Its report is saved in its own output directory. Use a compute allocation for real catalogues.
