# DESI Y1 HOD fit results

Generated 2026-10-21T12:39:02+00:00 from saved fit artifacts. This file is regenerated automatically.

## Updating this report

New DESI example fits update this report on MPI rank 0 after best-point reevaluation. Completed runs keep JSON snapshots in `run_history/`, including repeated fits in the same directory. Their data-versus-best-fit plots are archived there and embedded below. Use a new output directory to retain each run’s full arrays and plot. Only outputs of `run_desi_y1_minimizer.py` are included; unrelated optimizer examples are excluded.

From the repository root, with the cosmodesi environment active:

```bash
python HODDIES/fit_functions/scripts/summarize_desi_y1_fits.py
# Add another results directory:
python HODDIES/fit_functions/scripts/summarize_desi_y1_fits.py --root /path/to/fits
```

Search roots: `/global/u1/a/arocher/Code/postdoc/HODDIES_v2/tmp`, `/global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/tmp`, `/global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/tmp/desi_y1_debug`.

## Fit overview

Clustering fits use wp + xi0 + xi2 in the redshift bin listed for each run. The hybrid covariance is C = R_small × outer(σ_JK, σ_JK), with the saved finite-mock precision multiplier applied once to the clustering block. Optional LRG lensing at z=0.5 uses diagonal CSV errors with zero clustering-lensing covariance; the lensing precision is unscaled. χ² is the saved reevaluation using the fit’s fixed mock seed; dof = Ndata − Nfree. Parameter uncertainties are not estimated by this optimizer.

| Fit | Tracers | Redshift bin | Statistics | Run state | Ndata | Nfree | dof | χ² | χ²/dof |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [F1](#f1) | LRG | [0.4, 0.6] | wp+xi0+xi2+delta_sigma | Data checked; no optimization | 82 | 9 | 73 | — | — |
| [F2](#f2) | LRG | [0.8, 1.1] | wp+xi0+xi2 | Completed | 69 | 9 | 60 | 211.0319 | 3.517198 |
| [F3](#f3) | ELG | [0.8, 1.1] | wp+xi0+xi2 | Completed | 69 | 10 | 59 | 94.2892 | 1.598122 |
| [F4](#f4) | ELG | [0.8, 1.1] | wp+xi0+xi2 | Completed | 69 | 10 | 59 | 95.20075 | 1.613572 |
| [F5](#f5) | LRG+ELG | [0.8, 1.1] | wp+xi0+xi2 | Completed | 207 | 19 | 188 | 719.3999 | 3.826595 |
| [F6](#f6) | LRG | [0.4, 0.6] | wp+xi0+xi2+delta_sigma | Completed | 82 | 9 | 73 | 307.1979 | 4.20819 |
| [F7](#f7) | LRG | [0.4, 0.6] | wp+xi0+xi2+delta_sigma | Completed | 82 | 9 | 73 | 279.0103 | 3.82206 |

“Completed” means a result was saved; optimizer convergence is reported separately below. Blank best-fit cells mean that no completed parameter estimate is available.

## Best-fit parameter comparison

| Parameter | F1 | F2 | F3 | F4 | F5 | F6 | F7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M_0_LRG | — | 12.5195 | — | — | 12.7143 | 12.85495 | 12.82845 |
| M_1_LRG | — | 13.56777 | — | — | 13.44363 | 13.6187 | 13.63102 |
| log_Mcent_LRG | — | 12.57812 | — | — | 12.55074 | 12.57175 | 12.57996 |
| alpha_LRG | — | 1.083486 | — | — | 0.960833 | 0.8902968 | 0.9073612 |
| f_sigv_LRG | — | 0.7253601 | — | — | 0.8693954 | 0.6966254 | 0.6935626 |
| sigma_M_LRG | — | 0.1424595 | — | — | 0.1459179 | 0.05116095 | 0.0800144 |
| f_vcen_LRG | — | 0.4548684 | — | — | 0.2841225 | 0.3342753 | 0.3545933 |
| ab_c_cen_LRG | — | 0.2711032 | — | — | 0.2306815 | 0.8963878 | 0.8974561 |
| ab_c_sat_LRG | — | 0.04676444 | — | — | -0.05142641 | 0.1281854 | 0.1927837 |
| M_0_ELG | — | — | 11.12027 | 11.11884 | 11.14952 | — | — |
| As_ELG | — | — | 0.6102395 | 0.6096463 | 0.3076187 | — | — |
| log_Mcent_ELG | — | — | 11.43333 | 11.43335 | 11.48874 | — | — |
| alpha_ELG | — | — | 0.6768215 | 0.6768024 | 0.5442901 | — | — |
| f_sigv_ELG | — | — | 1.921658 | 1.921794 | 1.530279 | — | — |
| sigma_M_ELG | — | — | 0.314283 | 0.3142449 | 0.2656131 | — | — |
| gamma_ELG | — | — | 3.441498 | 3.440723 | 4.975559 | — | — |
| exp_scale_ELG | — | — | 5.193589 | 5.165649 | 6.361973 | — | — |
| exp_frac_ELG | — | — | 0.5233417 | 0.5224129 | 0.5014961 | — | — |
| nfw_rescale_ELG | — | — | 0.5214822 | 0.520319 | 0.6210558 | — | — |

<a id="f1"></a>

## F1: LRG

Run ID: `eff80553a7374e858569f73d4ac714ec`. State: Data checked; no optimization.

| Setting | Value |
| --- | --- |
| Simulation | AbacusSummit_highbase_c000_ph100 |
| Snapshot redshift | 0.5 |
| Halo root | /global/cfs/cdirs/desi/cosmosim/Abacus |
| Halo mass cut | — |
| Data redshift bin | [0.4, 0.6] |
| Statistics | ['wp', 'xi0', 'xi2', 'delta_sigma'] |
| Pair order | ['LRG_LRG'] |
| Cuts | {'rppi': '[::2][4:-3]', 'smu': '[16:-6]', 'pimax': 40} |
| Hybrid correlation matrix | /global/cfs/cdirs/desi/users/arocher/Y1/2PCF_for_corr/Abcaus_small_boxes/z0.500/Mcorr_small_box_z0.5_wp+xi_LRG.txt |
| Covariance mock count | 1883 |
| Precision multiplier | 0.9628055 |
| Mock seed | 10 |
| MPI ranks | 1 |
| Threads/rank | 2 |
| Slurm job | — |
| Started (UTC) | 2026-10-21T12:09:10+00:00 |
| Finished (UTC) | — |
| Elapsed seconds | — |

| Pair | Statistic | Bins | First bin centre [Mpc/h] | Last bin centre [Mpc/h] |
| --- | --- | --- | --- | --- |
| LRG_LRG | wp | 17 | 0.05791791 | 26.872 |
| LRG_LRG | xi0 | 26 | 0.239598 | 29.02725 |
| LRG_LRG | xi2 | 26 | 0.239598 | 29.02725 |
| LRG_LRG | delta_sigma | 13 | 0.1258925 | 31.62278 |

### Lensing measurement

| Setting | Value |
| --- | --- |
| CSV file | /global/u1/a/arocher/Code/postdoc/HODDIES_v2/tmp/bgs2_bgs3_lrg1_lrg2_best_fit_hsc.csv |
| Columns | ['rp', 'ds_obs_lrg1', 'ds_err_lrg1'] |
| Covariance | diagonal CSV errors squared |
| Cross covariance | zero with clustering |
| DeltaSigma units | h Msun/pc^2 |
| Excluded CSV rows (zero-based) | [13] |
| Bin edges | np.logspace(-1, 1.6, 13) |
| Model evaluation | CSV rp centres on the internal fine model grid |
| Precision correction | clustering block only |

### Models

**LRG**: `SHOD`; satellite model `Nsat_pow_law`. Conformity: False; link_sat_to_central: True; global assembly bias: True; configured tracer coefficients: `{'c': [0.0, 0.0]}`.

Configured values below are the input settings; fitted values are in the parameter table. Density normalization can change occupation amplitudes during mock generation.

| Input setting | Value |
| --- | --- |
| Ac | 1 |
| log_Mcent | 12.67 |
| sigma_M | 0.08 |
| satellites | True |
| sat_HOD_model | Nsat_pow_law |
| As | 1 |
| M_0 | 12.61 |
| M_1 | 13.78 |
| alpha | 1.03 |
| nu | 1 |
| density | 0.0007 |
| conformity_bias | False |
| link_sat_to_central | True |
| vel_sat | rd_normal |
| f_sigv | 1 |
| f_vcen | 0.1 |
| vsmear | — |
| exp_frac | 0 |
| exp_scale | 1 |
| nfw_rescale | 1 |

### Optimizer

| Setting / diagnostic | Value |
| --- | --- |
| method | cmaes |
| maxiter | 300 |
| popsize | 5 |
| xtol | 1 |
| ftol | 41 |
| fun | — |
| nit | — |
| nfev | — |
| status | — |
| success | — |
| message | — |

For CMA-ES, ftol is an absolute objective target. The DESI example defaults to ftol = Ndata / 2 and, with MPI, popsize = max(2, number of ranks).

### Fitted parameters

| Parameter | Best fit | Prior minimum | Prior maximum |
| --- | --- | --- | --- |
| M_0_LRG | — | 12.4 | 13.5 |
| M_1_LRG | — | 13 | 14.5 |
| log_Mcent_LRG | — | 12.4 | 13.2 |
| alpha_LRG | — | 0.5 | 1.5 |
| f_sigv_LRG | — | 0.5 | 1.5 |
| sigma_M_LRG | — | 0.02 | 0.5 |
| f_vcen_LRG | — | 0 | 1 |
| ab_c_cen_LRG | — | -1 | 1 |
| ab_c_sat_LRG | — | -1 | 1 |

### Data and artifacts

Output directory: `/global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/tmp/desi_y1_LRG_z0.4_0.6_delta_sigma`.
Latest directory artifacts: [configuration.json](tmp/desi_y1_LRG_z0.4_0.6_delta_sigma/configuration.json), [fit_inputs.npz](tmp/desi_y1_LRG_z0.4_0.6_delta_sigma/fit_inputs.npz).

- LRG_LRG, rppi: `/global/cfs/cdirs/desi/users/arocher/Y1/Y1_data_v2/rppi/allcounts_LRG_GCcomb_0.4_0.6_default_angular_bitwise_log_njack128_nran4_split20.npy`
- LRG_LRG, smu: `/global/cfs/cdirs/desi/users/arocher/Y1/Y1_data_v2/smu/allcounts_LRG_GCcomb_0.4_0.6_default_angular_bitwise_log_njack128_nran4_split20.npy`
- LRG_LRG, delta_sigma: `/global/u1/a/arocher/Code/postdoc/HODDIES_v2/tmp/bgs2_bgs3_lrg1_lrg2_best_fit_hsc.csv`

### Data and best-fit plots

![F1: DESI Y1 measurements of wp, xi0, xi2, delta_sigma](tmp/desi_y1_LRG_z0.4_0.6_delta_sigma/data.png)

Measurements with saved errors (jackknife for clustering). No completed best-fit model is available yet.

<a id="f2"></a>

## F2: LRG

Run ID: `58080481d3f04f5e8c2a2a2ee400815e`. State: Completed.

| Setting | Value |
| --- | --- |
| Simulation | AbacusSummit_highbase_c000_ph100 |
| Snapshot redshift | 0.95 |
| Halo root | /global/cfs/cdirs/desi/cosmosim/Abacus |
| Halo mass cut | — |
| Data redshift bin | [0.8, 1.1] |
| Statistics | ['wp', 'xi0', 'xi2'] |
| Pair order | ['LRG_LRG'] |
| Cuts | {'rppi': '[::2][4:-3]', 'smu': '[16:-6]', 'pimax': 40} |
| Hybrid correlation matrix | /global/cfs/cdirs/desi/users/arocher/Y1/2PCF_for_corr/Abcaus_small_boxes/z0.950/Mcorr_small_box_z0.95_wp+xi_LRG.txt |
| Covariance mock count | 1883 |
| Precision multiplier | 0.9628055 |
| Mock seed | 10 |
| MPI ranks | 64 |
| Threads/rank | 32 |
| Slurm job | 59996609 |
| Started (UTC) | 2026-10-19T13:59:32+00:00 |
| Finished (UTC) | 2026-10-19T14:07:36+00:00 |
| Elapsed seconds | 484.6945 |

| Pair | Statistic | Bins | First bin centre [Mpc/h] | Last bin centre [Mpc/h] |
| --- | --- | --- | --- | --- |
| LRG_LRG | wp | 17 | 0.05795803 | 26.87869 |
| LRG_LRG | xi0 | 26 | 0.2396581 | 29.02873 |
| LRG_LRG | xi2 | 26 | 0.2396581 | 29.02873 |

### Models

**LRG**: `SHOD`; satellite model `Nsat_pow_law`. Conformity: False; link_sat_to_central: True; global assembly bias: True; configured tracer coefficients: `{'c': [0.0, 0.0]}`.

Configured values below are the input settings; fitted values are in the parameter table. Density normalization can change occupation amplitudes during mock generation.

| Input setting | Value |
| --- | --- |
| Ac | 1 |
| log_Mcent | 12.67 |
| sigma_M | 0.08 |
| satellites | True |
| sat_HOD_model | Nsat_pow_law |
| As | 1 |
| M_0 | 12.61 |
| M_1 | 13.78 |
| alpha | 1.03 |
| nu | 1 |
| density | 0.0007 |
| conformity_bias | False |
| link_sat_to_central | True |
| vel_sat | rd_normal |
| f_sigv | 1 |
| f_vcen | 0.1 |
| vsmear | — |
| exp_frac | 0 |
| exp_scale | 1 |
| nfw_rescale | 1 |

### Optimizer

| Setting / diagnostic | Value |
| --- | --- |
| method | cmaes |
| maxiter | 200 |
| popsize | 64 |
| xtol | 1 |
| ftol | 34.5 |
| fun | 211.0319 |
| nit | 200 |
| nfev | 12800 |
| status | -1 |
| success | False |
| message | maximum number of iterations is reached |

For CMA-ES, ftol is an absolute objective target. The DESI example defaults to ftol = Ndata / 2 and, with MPI, popsize = max(2, number of ranks).

### Fitted parameters

| Parameter | Best fit | Prior minimum | Prior maximum |
| --- | --- | --- | --- |
| M_0_LRG | 12.5195 | 12.4 | 13.5 |
| M_1_LRG | 13.56777 | 13 | 14.5 |
| log_Mcent_LRG | 12.57812 | 12.4 | 13.2 |
| alpha_LRG | 1.083486 | 0.5 | 1.5 |
| f_sigv_LRG | 0.7253601 | 0.5 | 1.5 |
| sigma_M_LRG | 0.1424595 | 0.02 | 0.5 |
| f_vcen_LRG | 0.4548684 | 0 | 1 |
| ab_c_cen_LRG | 0.2711032 | -1 | 1 |
| ab_c_sat_LRG | 0.04676444 | -1 | 1 |

### Data and artifacts

Output directory: `/global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/tmp/desi_y1_debug/59996609`.
Archived run metadata: [58080481d3f04f5e8c2a2a2ee400815e.json](tmp/desi_y1_debug/59996609/run_history/58080481d3f04f5e8c2a2a2ee400815e.json).
Latest directory artifacts: [configuration.json](tmp/desi_y1_debug/59996609/configuration.json), [fit_inputs.npz](tmp/desi_y1_debug/59996609/fit_inputs.npz), [best_fit.npy](tmp/desi_y1_debug/59996609/best_fit.npy), [best_fit_statistics.npz](tmp/desi_y1_debug/59996609/best_fit_statistics.npz), [best_fit.png](tmp/desi_y1_debug/59996609/best_fit.png).

- LRG_LRG, rppi: `/global/cfs/cdirs/desi/users/arocher/Y1/Y1_data_v2/rppi/allcounts_LRG_GCcomb_0.8_1.1_default_angular_bitwise_log_njack128_nran4_split20.npy`
- LRG_LRG, smu: `/global/cfs/cdirs/desi/users/arocher/Y1/Y1_data_v2/smu/allcounts_LRG_GCcomb_0.8_1.1_default_angular_bitwise_log_njack128_nran4_split20.npy`

### Data and best-fit plots

![F2: DESI Y1 data and best-fit wp, xi0, xi2](tmp/desi_y1_debug/59996609/run_history/58080481d3f04f5e8c2a2a2ee400815e.png)

Measurements with error bars and the best-fit model for this run.

<a id="f3"></a>

## F3: ELG

Run ID: `fdca8b2f174f41bfac2c56878e3e5b9f`. State: Completed.

| Setting | Value |
| --- | --- |
| Simulation | AbacusSummit_highbase_c000_ph100 |
| Snapshot redshift | 0.95 |
| Halo root | /global/cfs/cdirs/desi/cosmosim/Abacus |
| Halo mass cut | — |
| Data redshift bin | [0.8, 1.1] |
| Statistics | ['wp', 'xi0', 'xi2'] |
| Pair order | ['ELG_ELG'] |
| Cuts | {'rppi': '[::2][4:-3]', 'smu': '[16:-6]', 'pimax': 40} |
| Hybrid correlation matrix | /global/cfs/cdirs/desi/users/arocher/Y1/2PCF_for_corr/Abcaus_small_boxes/z0.950/Mcorr_small_box_z0.95_wp+xi_ELG.txt |
| Covariance mock count | 1883 |
| Precision multiplier | 0.9628055 |
| Mock seed | 10 |
| MPI ranks | 64 |
| Threads/rank | 32 |
| Slurm job | 59997489 |
| Started (UTC) | 2026-10-19T14:57:26+00:00 |
| Finished (UTC) | 2026-10-19T15:09:04+00:00 |
| Elapsed seconds | 698.0126 |

| Pair | Statistic | Bins | First bin centre [Mpc/h] | Last bin centre [Mpc/h] |
| --- | --- | --- | --- | --- |
| ELG_ELG | wp | 17 | 0.05796909 | 26.87869 |
| ELG_ELG | xi0 | 26 | 0.2392322 | 29.02988 |
| ELG_ELG | xi2 | 26 | 0.2392322 | 29.02988 |

### Models

**ELG**: `mHMQ`; satellite model `Nsat_pow_law`. Conformity: True; link_sat_to_central: False; global assembly bias: False; configured tracer coefficients: `{'c': [0.0, 0.0]}`.

Configured values below are the input settings; fitted values are in the parameter table. Density normalization can change occupation amplitudes during mock generation.

| Input setting | Value |
| --- | --- |
| Ac | 0.1 |
| log_Mcent | 11.64 |
| sigma_M | 0.3 |
| gamma | 5.57 |
| satellites | True |
| sat_HOD_model | Nsat_pow_law |
| As | 0.41 |
| M_0 | 11.2 |
| M_1 | 13 |
| alpha | 0.81 |
| nu | 1 |
| density | 0.001 |
| conformity_bias | True |
| link_sat_to_central | False |
| vel_sat | rd_normal |
| f_sigv | 1.63 |
| f_vcen | 0 |
| vsmear | — |
| exp_frac | 0.55 |
| exp_scale | 6.14 |
| nfw_rescale | 0.67 |

### Optimizer

| Setting / diagnostic | Value |
| --- | --- |
| method | cmaes |
| maxiter | 300 |
| popsize | 64 |
| xtol | 1 |
| ftol | 34.5 |
| fun | 94.2892 |
| nit | 300 |
| nfev | 19200 |
| status | -1 |
| success | False |
| message | maximum number of iterations is reached |

For CMA-ES, ftol is an absolute objective target. The DESI example defaults to ftol = Ndata / 2 and, with MPI, popsize = max(2, number of ranks).

### Fitted parameters

| Parameter | Best fit | Prior minimum | Prior maximum |
| --- | --- | --- | --- |
| M_0_ELG | 11.12027 | 11 | 11.8 |
| As_ELG | 0.6102395 | 0.01 | 0.7 |
| log_Mcent_ELG | 11.43333 | 11.4 | 12.2 |
| alpha_ELG | 0.6768215 | 0.3 | 1.5 |
| f_sigv_ELG | 1.921658 | 0.8 | 2.2 |
| sigma_M_ELG | 0.314283 | 0.05 | 1 |
| gamma_ELG | 3.441498 | 1 | 8 |
| exp_scale_ELG | 5.193589 | 0.9 | 10 |
| exp_frac_ELG | 0.5233417 | 0 | 1 |
| nfw_rescale_ELG | 0.5214822 | 0.1 | 2 |

### Data and artifacts

Output directory: `/global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/tmp/desi_y1_debug/59997489`.
Archived run metadata: [fdca8b2f174f41bfac2c56878e3e5b9f.json](tmp/desi_y1_debug/59997489/run_history/fdca8b2f174f41bfac2c56878e3e5b9f.json).
Latest directory artifacts: [configuration.json](tmp/desi_y1_debug/59997489/configuration.json), [fit_inputs.npz](tmp/desi_y1_debug/59997489/fit_inputs.npz), [best_fit.npy](tmp/desi_y1_debug/59997489/best_fit.npy), [best_fit_statistics.npz](tmp/desi_y1_debug/59997489/best_fit_statistics.npz), [best_fit.png](tmp/desi_y1_debug/59997489/best_fit.png).

- ELG_ELG, rppi: `/global/cfs/cdirs/desi/users/arocher/Y1/Y1_data_v2/rppi/allcounts_ELG_LOPnotqso_GCcomb_0.8_1.1_default_angular_bitwise_log_njack128_nran4_split20.npy`
- ELG_ELG, smu: `/global/cfs/cdirs/desi/users/arocher/Y1/Y1_data_v2/smu/allcounts_ELG_LOPnotqso_GCcomb_0.8_1.1_default_angular_bitwise_log_njack128_nran4_split20.npy`

### Data and best-fit plots

![F3: DESI Y1 data and best-fit wp, xi0, xi2](tmp/desi_y1_debug/59997489/run_history/fdca8b2f174f41bfac2c56878e3e5b9f.png)

Measurements with error bars and the best-fit model for this run.

<a id="f4"></a>

## F4: ELG

Run ID: `7a680cc077304bdcb760785cb2c10a16`. State: Completed.

| Setting | Value |
| --- | --- |
| Simulation | AbacusSummit_highbase_c000_ph100 |
| Snapshot redshift | 0.95 |
| Halo root | /global/cfs/cdirs/desi/cosmosim/Abacus |
| Halo mass cut | — |
| Data redshift bin | [0.8, 1.1] |
| Statistics | ['wp', 'xi0', 'xi2'] |
| Pair order | ['ELG_ELG'] |
| Cuts | {'rppi': '[::2][4:-3]', 'smu': '[16:-6]', 'pimax': 40} |
| Hybrid correlation matrix | /global/cfs/cdirs/desi/users/arocher/Y1/2PCF_for_corr/Abcaus_small_boxes/z0.950/Mcorr_small_box_z0.95_wp+xi_ELG.txt |
| Covariance mock count | 1883 |
| Precision multiplier | 0.9628055 |
| Mock seed | 10 |
| MPI ranks | 64 |
| Threads/rank | 32 |
| Slurm job | 60000677 |
| Started (UTC) | 2026-10-19T15:47:41+00:00 |
| Finished (UTC) | 2026-10-19T15:56:01+00:00 |
| Elapsed seconds | 500.6105 |

| Pair | Statistic | Bins | First bin centre [Mpc/h] | Last bin centre [Mpc/h] |
| --- | --- | --- | --- | --- |
| ELG_ELG | wp | 17 | 0.05796909 | 26.87869 |
| ELG_ELG | xi0 | 26 | 0.2392322 | 29.02988 |
| ELG_ELG | xi2 | 26 | 0.2392322 | 29.02988 |

### Models

**ELG**: `mHMQ`; satellite model `Nsat_pow_law`. Conformity: True; link_sat_to_central: False; global assembly bias: False; configured tracer coefficients: `{'c': [0.0, 0.0]}`.

Configured values below are the input settings; fitted values are in the parameter table. Density normalization can change occupation amplitudes during mock generation.

| Input setting | Value |
| --- | --- |
| Ac | 0.1 |
| log_Mcent | 11.64 |
| sigma_M | 0.3 |
| gamma | 5.57 |
| satellites | True |
| sat_HOD_model | Nsat_pow_law |
| As | 0.41 |
| M_0 | 11.2 |
| M_1 | 13 |
| alpha | 0.81 |
| nu | 1 |
| density | 0.001 |
| conformity_bias | True |
| link_sat_to_central | False |
| vel_sat | rd_normal |
| f_sigv | 1.63 |
| f_vcen | 0 |
| vsmear | — |
| exp_frac | 0.55 |
| exp_scale | 6.14 |
| nfw_rescale | 0.67 |

### Optimizer

| Setting / diagnostic | Value |
| --- | --- |
| method | cmaes |
| maxiter | 200 |
| popsize | 64 |
| xtol | 1 |
| ftol | 34.5 |
| fun | 95.20075 |
| nit | 200 |
| nfev | 12800 |
| status | -1 |
| success | False |
| message | maximum number of iterations is reached |

For CMA-ES, ftol is an absolute objective target. The DESI example defaults to ftol = Ndata / 2 and, with MPI, popsize = max(2, number of ranks).

### Fitted parameters

| Parameter | Best fit | Prior minimum | Prior maximum |
| --- | --- | --- | --- |
| M_0_ELG | 11.11884 | 11 | 11.8 |
| As_ELG | 0.6096463 | 0.01 | 0.7 |
| log_Mcent_ELG | 11.43335 | 11.4 | 12.2 |
| alpha_ELG | 0.6768024 | 0.3 | 1.5 |
| f_sigv_ELG | 1.921794 | 0.8 | 2.2 |
| sigma_M_ELG | 0.3142449 | 0.05 | 1 |
| gamma_ELG | 3.440723 | 1 | 8 |
| exp_scale_ELG | 5.165649 | 0.9 | 10 |
| exp_frac_ELG | 0.5224129 | 0 | 1 |
| nfw_rescale_ELG | 0.520319 | 0.1 | 2 |

### Data and artifacts

Output directory: `/global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/tmp/desi_y1_debug/60000677`.
Archived run metadata: [7a680cc077304bdcb760785cb2c10a16.json](tmp/desi_y1_debug/60000677/run_history/7a680cc077304bdcb760785cb2c10a16.json).
Latest directory artifacts: [configuration.json](tmp/desi_y1_debug/60000677/configuration.json), [fit_inputs.npz](tmp/desi_y1_debug/60000677/fit_inputs.npz), [best_fit.npy](tmp/desi_y1_debug/60000677/best_fit.npy), [best_fit_statistics.npz](tmp/desi_y1_debug/60000677/best_fit_statistics.npz), [best_fit.png](tmp/desi_y1_debug/60000677/best_fit.png).

- ELG_ELG, rppi: `/global/cfs/cdirs/desi/users/arocher/Y1/Y1_data_v2/rppi/allcounts_ELG_LOPnotqso_GCcomb_0.8_1.1_default_angular_bitwise_log_njack128_nran4_split20.npy`
- ELG_ELG, smu: `/global/cfs/cdirs/desi/users/arocher/Y1/Y1_data_v2/smu/allcounts_ELG_LOPnotqso_GCcomb_0.8_1.1_default_angular_bitwise_log_njack128_nran4_split20.npy`

### Data and best-fit plots

![F4: DESI Y1 data and best-fit wp, xi0, xi2](tmp/desi_y1_debug/60000677/run_history/7a680cc077304bdcb760785cb2c10a16.png)

Measurements with error bars and the best-fit model for this run.

<a id="f5"></a>

## F5: LRG+ELG

Run ID: `b670f64b9984483bbdeec7675ba8f8dc`. State: Completed.

| Setting | Value |
| --- | --- |
| Simulation | AbacusSummit_highbase_c000_ph100 |
| Snapshot redshift | 0.95 |
| Halo root | /global/cfs/cdirs/desi/cosmosim/Abacus |
| Halo mass cut | — |
| Data redshift bin | [0.8, 1.1] |
| Statistics | ['wp', 'xi0', 'xi2'] |
| Pair order | ['ELG_ELG', 'ELG_LRG', 'LRG_LRG'] |
| Cuts | {'rppi': '[::2][4:-3]', 'smu': '[16:-6]', 'pimax': 40} |
| Hybrid correlation matrix | /global/cfs/cdirs/desi/users/arocher/Y1/2PCF_for_corr/Abcaus_small_boxes/z0.950/Mcorr_small_box_z0.95_wp+xi_ELG_LRG.txt |
| Covariance mock count | 1883 |
| Precision multiplier | 0.8894793 |
| Mock seed | 10 |
| MPI ranks | 64 |
| Threads/rank | 32 |
| Slurm job | 60031900 |
| Started (UTC) | 2026-10-20T11:14:46+00:00 |
| Finished (UTC) | 2026-10-20T11:32:42+00:00 |
| Elapsed seconds | 1075.743 |

| Pair | Statistic | Bins | First bin centre [Mpc/h] | Last bin centre [Mpc/h] |
| --- | --- | --- | --- | --- |
| ELG_ELG | wp | 17 | 0.05796909 | 26.87869 |
| ELG_ELG | xi0 | 26 | 0.2392322 | 29.02988 |
| ELG_ELG | xi2 | 26 | 0.2392322 | 29.02988 |
| ELG_LRG | wp | 17 | 0.05794125 | 26.87867 |
| ELG_LRG | xi0 | 26 | 0.239755 | 29.0299 |
| ELG_LRG | xi2 | 26 | 0.239755 | 29.0299 |
| LRG_LRG | wp | 17 | 0.05795803 | 26.87869 |
| LRG_LRG | xi0 | 26 | 0.2396581 | 29.02873 |
| LRG_LRG | xi2 | 26 | 0.2396581 | 29.02873 |

### Models

**LRG**: `SHOD`; satellite model `Nsat_pow_law`. Conformity: False; link_sat_to_central: True; global assembly bias: True; configured tracer coefficients: `{'c': [0.0, 0.0]}`.

Configured values below are the input settings; fitted values are in the parameter table. Density normalization can change occupation amplitudes during mock generation.

| Input setting | Value |
| --- | --- |
| Ac | 1 |
| log_Mcent | 12.67 |
| sigma_M | 0.08 |
| satellites | True |
| sat_HOD_model | Nsat_pow_law |
| As | 1 |
| M_0 | 12.61 |
| M_1 | 13.78 |
| alpha | 1.03 |
| nu | 1 |
| density | 0.0007 |
| conformity_bias | False |
| link_sat_to_central | True |
| vel_sat | rd_normal |
| f_sigv | 1 |
| f_vcen | 0.1 |
| vsmear | — |
| exp_frac | 0 |
| exp_scale | 1 |
| nfw_rescale | 1 |

**ELG**: `mHMQ`; satellite model `Nsat_pow_law`. Conformity: True; link_sat_to_central: False; global assembly bias: True; configured tracer coefficients: `{'c': [0.0, 0.0]}`.

Configured values below are the input settings; fitted values are in the parameter table. Density normalization can change occupation amplitudes during mock generation.

| Input setting | Value |
| --- | --- |
| Ac | 0.1 |
| log_Mcent | 11.64 |
| sigma_M | 0.3 |
| gamma | 5.57 |
| satellites | True |
| sat_HOD_model | Nsat_pow_law |
| As | 0.41 |
| M_0 | 11.2 |
| M_1 | 13 |
| alpha | 0.81 |
| nu | 1 |
| density | 0.001 |
| conformity_bias | True |
| link_sat_to_central | False |
| vel_sat | rd_normal |
| f_sigv | 1.63 |
| f_vcen | 0 |
| vsmear | — |
| exp_frac | 0.55 |
| exp_scale | 6.14 |
| nfw_rescale | 0.67 |

### Optimizer

| Setting / diagnostic | Value |
| --- | --- |
| method | cmaes |
| maxiter | 100 |
| popsize | 64 |
| xtol | 1 |
| ftol | 103.5 |
| fun | 719.3999 |
| nit | 100 |
| nfev | 6400 |
| status | -1 |
| success | False |
| message | maximum number of iterations is reached |

For CMA-ES, ftol is an absolute objective target. The DESI example defaults to ftol = Ndata / 2 and, with MPI, popsize = max(2, number of ranks).

### Fitted parameters

| Parameter | Best fit | Prior minimum | Prior maximum |
| --- | --- | --- | --- |
| M_0_LRG | 12.7143 | 12.4 | 13.5 |
| M_1_LRG | 13.44363 | 13 | 14.5 |
| log_Mcent_LRG | 12.55074 | 12.4 | 13.2 |
| alpha_LRG | 0.960833 | 0.5 | 1.5 |
| f_sigv_LRG | 0.8693954 | 0.5 | 1.5 |
| sigma_M_LRG | 0.1459179 | 0.02 | 0.5 |
| f_vcen_LRG | 0.2841225 | 0 | 1 |
| ab_c_cen_LRG | 0.2306815 | -1 | 1 |
| ab_c_sat_LRG | -0.05142641 | -1 | 1 |
| M_0_ELG | 11.14952 | 11 | 11.8 |
| As_ELG | 0.3076187 | 0.01 | 0.7 |
| log_Mcent_ELG | 11.48874 | 11.4 | 12.2 |
| alpha_ELG | 0.5442901 | 0.3 | 1.5 |
| f_sigv_ELG | 1.530279 | 0.8 | 2.2 |
| sigma_M_ELG | 0.2656131 | 0.05 | 1 |
| gamma_ELG | 4.975559 | 1 | 8 |
| exp_scale_ELG | 6.361973 | 0.9 | 10 |
| exp_frac_ELG | 0.5014961 | 0 | 1 |
| nfw_rescale_ELG | 0.6210558 | 0.1 | 2 |

### Data and artifacts

Output directory: `/global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/tmp/desi_y1_debug/60031900`.
Archived run metadata: [b670f64b9984483bbdeec7675ba8f8dc.json](tmp/desi_y1_debug/60031900/run_history/b670f64b9984483bbdeec7675ba8f8dc.json).
Latest directory artifacts: [configuration.json](tmp/desi_y1_debug/60031900/configuration.json), [fit_inputs.npz](tmp/desi_y1_debug/60031900/fit_inputs.npz), [best_fit.npy](tmp/desi_y1_debug/60031900/best_fit.npy), [best_fit_statistics.npz](tmp/desi_y1_debug/60031900/best_fit_statistics.npz), [best_fit.png](tmp/desi_y1_debug/60031900/best_fit.png).

- ELG_ELG, rppi: `/global/cfs/cdirs/desi/users/arocher/Y1/Y1_data_v2/rppi/allcounts_ELG_LOPnotqso_GCcomb_0.8_1.1_default_angular_bitwise_log_njack128_nran4_split20.npy`
- ELG_ELG, smu: `/global/cfs/cdirs/desi/users/arocher/Y1/Y1_data_v2/smu/allcounts_ELG_LOPnotqso_GCcomb_0.8_1.1_default_angular_bitwise_log_njack128_nran4_split20.npy`
- ELG_LRG, rppi: `/global/cfs/cdirs/desi/users/arocher/Y1/Y1_data_v2/rppi/allcounts_LRG_ELG_LOPnotqso_GCcomb_0.8_1.1_default_angular_bitwise_log_njack128_nran4_split20.npy`
- ELG_LRG, smu: `/global/cfs/cdirs/desi/users/arocher/Y1/Y1_data_v2/smu/allcounts_LRG_ELG_LOPnotqso_GCcomb_0.8_1.1_default_angular_bitwise_log_njack128_nran4_split20.npy`
- LRG_LRG, rppi: `/global/cfs/cdirs/desi/users/arocher/Y1/Y1_data_v2/rppi/allcounts_LRG_GCcomb_0.8_1.1_default_angular_bitwise_log_njack128_nran4_split20.npy`
- LRG_LRG, smu: `/global/cfs/cdirs/desi/users/arocher/Y1/Y1_data_v2/smu/allcounts_LRG_GCcomb_0.8_1.1_default_angular_bitwise_log_njack128_nran4_split20.npy`

### Data and best-fit plots

![F5: DESI Y1 data and best-fit wp, xi0, xi2](tmp/desi_y1_debug/60031900/run_history/b670f64b9984483bbdeec7675ba8f8dc.png)

Measurements with error bars and the best-fit model for this run.

<a id="f6"></a>

## F6: LRG

Run ID: `84e56fc09dde4368a76d10f46c237268`. State: Completed.

| Setting | Value |
| --- | --- |
| Simulation | AbacusSummit_highbase_c000_ph100 |
| Snapshot redshift | 0.5 |
| Halo root | /global/cfs/cdirs/desi/cosmosim/Abacus |
| Halo mass cut | — |
| Data redshift bin | [0.4, 0.6] |
| Statistics | ['wp', 'xi0', 'xi2', 'delta_sigma'] |
| Pair order | ['LRG_LRG'] |
| Cuts | {'rppi': '[::2][4:-3]', 'smu': '[16:-6]', 'pimax': 40} |
| Hybrid correlation matrix | /global/cfs/cdirs/desi/users/arocher/Y1/2PCF_for_corr/Abcaus_small_boxes/z0.500/Mcorr_small_box_z0.5_wp+xi_LRG.txt |
| Covariance mock count | 1883 |
| Precision multiplier | 0.9628055 |
| Mock seed | 10 |
| MPI ranks | 64 |
| Threads/rank | 32 |
| Slurm job | 60088977 |
| Started (UTC) | 2026-10-21T10:18:54+00:00 |
| Finished (UTC) | 2026-10-21T10:26:15+00:00 |
| Elapsed seconds | 440.8 |

| Pair | Statistic | Bins | First bin centre [Mpc/h] | Last bin centre [Mpc/h] |
| --- | --- | --- | --- | --- |
| LRG_LRG | wp | 17 | 0.05791791 | 26.872 |
| LRG_LRG | xi0 | 26 | 0.239598 | 29.02725 |
| LRG_LRG | xi2 | 26 | 0.239598 | 29.02725 |
| LRG_LRG | delta_sigma | 13 | 0.1258925 | 31.62278 |

### Lensing measurement

| Setting | Value |
| --- | --- |
| CSV file | /global/u1/a/arocher/Code/postdoc/HODDIES_v2/tmp/bgs2_bgs3_lrg1_lrg2_best_fit_hsc.csv |
| Columns | ['rp', 'ds_obs_lrg1', 'ds_err_lrg1'] |
| Covariance | diagonal CSV errors squared |
| Cross covariance | zero with clustering |
| DeltaSigma units | h Msun/pc^2 |
| Excluded CSV rows (zero-based) | [13] |
| Bin edges | inferred from logarithmically spaced rp centres |
| Model evaluation | geometric bin centres |
| Precision correction | clustering block only |

### Models

**LRG**: `SHOD`; satellite model `Nsat_pow_law`. Conformity: False; link_sat_to_central: True; global assembly bias: True; configured tracer coefficients: `{'c': [0.0, 0.0]}`.

Configured values below are the input settings; fitted values are in the parameter table. Density normalization can change occupation amplitudes during mock generation.

| Input setting | Value |
| --- | --- |
| Ac | 1 |
| log_Mcent | 12.67 |
| sigma_M | 0.08 |
| satellites | True |
| sat_HOD_model | Nsat_pow_law |
| As | 1 |
| M_0 | 12.61 |
| M_1 | 13.78 |
| alpha | 1.03 |
| nu | 1 |
| density | 0.0007 |
| conformity_bias | False |
| link_sat_to_central | True |
| vel_sat | rd_normal |
| f_sigv | 1 |
| f_vcen | 0.1 |
| vsmear | — |
| exp_frac | 0 |
| exp_scale | 1 |
| nfw_rescale | 1 |

### Optimizer

| Setting / diagnostic | Value |
| --- | --- |
| method | cmaes |
| maxiter | 100 |
| popsize | 64 |
| xtol | 1 |
| ftol | 41 |
| fun | 307.1979 |
| nit | 100 |
| nfev | 6400 |
| status | -1 |
| success | False |
| message | maximum number of iterations is reached |

For CMA-ES, ftol is an absolute objective target. The DESI example defaults to ftol = Ndata / 2 and, with MPI, popsize = max(2, number of ranks).

### Fitted parameters

| Parameter | Best fit | Prior minimum | Prior maximum |
| --- | --- | --- | --- |
| M_0_LRG | 12.85495 | 12.4 | 13.5 |
| M_1_LRG | 13.6187 | 13 | 14.5 |
| log_Mcent_LRG | 12.57175 | 12.4 | 13.2 |
| alpha_LRG | 0.8902968 | 0.5 | 1.5 |
| f_sigv_LRG | 0.6966254 | 0.5 | 1.5 |
| sigma_M_LRG | 0.05116095 | 0.02 | 0.5 |
| f_vcen_LRG | 0.3342753 | 0 | 1 |
| ab_c_cen_LRG | 0.8963878 | -1 | 1 |
| ab_c_sat_LRG | 0.1281854 | -1 | 1 |

### Data and artifacts

Output directory: `/global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/tmp/desi_y1_debug/60088977`.
Archived run metadata: [84e56fc09dde4368a76d10f46c237268.json](tmp/desi_y1_debug/60088977/run_history/84e56fc09dde4368a76d10f46c237268.json).
Latest directory artifacts: [configuration.json](tmp/desi_y1_debug/60088977/configuration.json), [fit_inputs.npz](tmp/desi_y1_debug/60088977/fit_inputs.npz), [best_fit.npy](tmp/desi_y1_debug/60088977/best_fit.npy), [best_fit_statistics.npz](tmp/desi_y1_debug/60088977/best_fit_statistics.npz), [best_fit.png](tmp/desi_y1_debug/60088977/best_fit.png).

- LRG_LRG, rppi: `/global/cfs/cdirs/desi/users/arocher/Y1/Y1_data_v2/rppi/allcounts_LRG_GCcomb_0.4_0.6_default_angular_bitwise_log_njack128_nran4_split20.npy`
- LRG_LRG, smu: `/global/cfs/cdirs/desi/users/arocher/Y1/Y1_data_v2/smu/allcounts_LRG_GCcomb_0.4_0.6_default_angular_bitwise_log_njack128_nran4_split20.npy`
- LRG_LRG, delta_sigma: `/global/u1/a/arocher/Code/postdoc/HODDIES_v2/tmp/bgs2_bgs3_lrg1_lrg2_best_fit_hsc.csv`

### Data and best-fit plots

![F6: DESI Y1 data and best-fit wp, xi0, xi2, delta_sigma](tmp/desi_y1_debug/60088977/run_history/84e56fc09dde4368a76d10f46c237268.png)

Measurements with error bars and the best-fit model for this run.

<a id="f7"></a>

## F7: LRG

Run ID: `afbf331bb68e447cab850becf301ab8a`. State: Completed.

| Setting | Value |
| --- | --- |
| Simulation | AbacusSummit_highbase_c000_ph100 |
| Snapshot redshift | 0.5 |
| Halo root | /global/cfs/cdirs/desi/cosmosim/Abacus |
| Halo mass cut | — |
| Data redshift bin | [0.4, 0.6] |
| Statistics | ['wp', 'xi0', 'xi2', 'delta_sigma'] |
| Pair order | ['LRG_LRG'] |
| Cuts | {'rppi': '[::2][4:-3]', 'smu': '[16:-6]', 'pimax': 40} |
| Hybrid correlation matrix | /global/cfs/cdirs/desi/users/arocher/Y1/2PCF_for_corr/Abcaus_small_boxes/z0.500/Mcorr_small_box_z0.5_wp+xi_LRG.txt |
| Covariance mock count | 1883 |
| Precision multiplier | 0.9628055 |
| Mock seed | 10 |
| MPI ranks | 64 |
| Threads/rank | 32 |
| Slurm job | 60091536 |
| Started (UTC) | 2026-10-21T12:00:36+00:00 |
| Finished (UTC) | 2026-10-21T12:16:06+00:00 |
| Elapsed seconds | 930.4395 |

| Pair | Statistic | Bins | First bin centre [Mpc/h] | Last bin centre [Mpc/h] |
| --- | --- | --- | --- | --- |
| LRG_LRG | wp | 17 | 0.05791791 | 26.872 |
| LRG_LRG | xi0 | 26 | 0.239598 | 29.02725 |
| LRG_LRG | xi2 | 26 | 0.239598 | 29.02725 |
| LRG_LRG | delta_sigma | 13 | 0.1258925 | 31.62278 |

### Lensing measurement

| Setting | Value |
| --- | --- |
| CSV file | /global/u1/a/arocher/Code/postdoc/HODDIES_v2/tmp/bgs2_bgs3_lrg1_lrg2_best_fit_hsc.csv |
| Columns | ['rp', 'ds_obs_lrg1', 'ds_err_lrg1'] |
| Covariance | diagonal CSV errors squared |
| Cross covariance | zero with clustering |
| DeltaSigma units | h Msun/pc^2 |
| Excluded CSV rows (zero-based) | [13] |
| Bin edges | inferred from logarithmically spaced rp centres |
| Model evaluation | geometric bin centres |
| Precision correction | clustering block only |

### Models

**LRG**: `SHOD`; satellite model `Nsat_pow_law`. Conformity: False; link_sat_to_central: True; global assembly bias: True; configured tracer coefficients: `{'c': [0.0, 0.0]}`.

Configured values below are the input settings; fitted values are in the parameter table. Density normalization can change occupation amplitudes during mock generation.

| Input setting | Value |
| --- | --- |
| Ac | 1 |
| log_Mcent | 12.67 |
| sigma_M | 0.08 |
| satellites | True |
| sat_HOD_model | Nsat_pow_law |
| As | 1 |
| M_0 | 12.61 |
| M_1 | 13.78 |
| alpha | 1.03 |
| nu | 1 |
| density | 0.0007 |
| conformity_bias | False |
| link_sat_to_central | True |
| vel_sat | rd_normal |
| f_sigv | 1 |
| f_vcen | 0.1 |
| vsmear | — |
| exp_frac | 0 |
| exp_scale | 1 |
| nfw_rescale | 1 |

### Optimizer

| Setting / diagnostic | Value |
| --- | --- |
| method | cmaes |
| maxiter | 300 |
| popsize | 64 |
| xtol | 1 |
| ftol | 41 |
| fun | 279.0103 |
| nit | 300 |
| nfev | 19200 |
| status | -1 |
| success | False |
| message | maximum number of iterations is reached |

For CMA-ES, ftol is an absolute objective target. The DESI example defaults to ftol = Ndata / 2 and, with MPI, popsize = max(2, number of ranks).

### Fitted parameters

| Parameter | Best fit | Prior minimum | Prior maximum |
| --- | --- | --- | --- |
| M_0_LRG | 12.82845 | 12.4 | 13.5 |
| M_1_LRG | 13.63102 | 13 | 14.5 |
| log_Mcent_LRG | 12.57996 | 12.4 | 13.2 |
| alpha_LRG | 0.9073612 | 0.5 | 1.5 |
| f_sigv_LRG | 0.6935626 | 0.5 | 1.5 |
| sigma_M_LRG | 0.0800144 | 0.02 | 0.5 |
| f_vcen_LRG | 0.3545933 | 0 | 1 |
| ab_c_cen_LRG | 0.8974561 | -1 | 1 |
| ab_c_sat_LRG | 0.1927837 | -1 | 1 |

### Data and artifacts

Output directory: `/global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/tmp/desi_y1_debug/60091536`.
Archived run metadata: [afbf331bb68e447cab850becf301ab8a.json](tmp/desi_y1_debug/60091536/run_history/afbf331bb68e447cab850becf301ab8a.json).
Latest directory artifacts: [configuration.json](tmp/desi_y1_debug/60091536/configuration.json), [fit_inputs.npz](tmp/desi_y1_debug/60091536/fit_inputs.npz), [best_fit.npy](tmp/desi_y1_debug/60091536/best_fit.npy), [best_fit_statistics.npz](tmp/desi_y1_debug/60091536/best_fit_statistics.npz), [best_fit.png](tmp/desi_y1_debug/60091536/best_fit.png).

- LRG_LRG, rppi: `/global/cfs/cdirs/desi/users/arocher/Y1/Y1_data_v2/rppi/allcounts_LRG_GCcomb_0.4_0.6_default_angular_bitwise_log_njack128_nran4_split20.npy`
- LRG_LRG, smu: `/global/cfs/cdirs/desi/users/arocher/Y1/Y1_data_v2/smu/allcounts_LRG_GCcomb_0.4_0.6_default_angular_bitwise_log_njack128_nran4_split20.npy`
- LRG_LRG, delta_sigma: `/global/u1/a/arocher/Code/postdoc/HODDIES_v2/tmp/bgs2_bgs3_lrg1_lrg2_best_fit_hsc.csv`

### Data and best-fit plots

![F7: DESI Y1 data and best-fit wp, xi0, xi2, delta_sigma](tmp/desi_y1_debug/60091536/run_history/afbf331bb68e447cab850becf301ab8a.png)

Measurements with error bars and the best-fit model for this run.
