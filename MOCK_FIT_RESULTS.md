# Mock HOD fit results

Generated 2026-10-21T09:14:09+00:00 from saved mock-fit artifacts.

Fits from `run_mock_minimizer.py`. Data and covariance use pair, statistic, multipole, bin order; lensing contributes auto-pairs only. The saved finite-mock precision multiplier is applied once. χ² is reevaluated at the optimizer’s best parameters with its fixed mock seed; dof = Ndata − Nfree. Optimizer success is separate from whether a result was saved. Parameter uncertainties are not estimated.

Regenerate from the repository root in the cosmodesi environment:

```bash
python HODDIES/fit_functions/scripts/summarize_mock_fits.py
# Include another output directory:
python HODDIES/fit_functions/scripts/summarize_mock_fits.py --root /path/to/fits
```

Search roots: `/global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/tmp/mock_fits`, `/global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/tmp/mock_fits/LRG_input_check`.

## Fit overview

| Fit | Tracers | Statistics | State | Ndata | Nfree | dof | χ² | χ²/dof | Optimizer success |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [F1](#f1) | LRG | wp+xi0+xi2 | Data checked; no optimization | 69 | 15 | 54 | — | — | — |
| [F2](#f2) | LRG+ELG | wp+xi0+xi2 | Completed | 207 | 34 | 173 | 308.7328 | 1.784583 | False |
| [F3](#f3) | LRG | wp+xi0+xi2+delta_sigma | Completed | 94 | 15 | 79 | 79.31442 | 1.00398 | False |
| [F4](#f4) | ELG | wp+xi0+xi2+delta_sigma | Completed | 94 | 19 | 75 | 132.6977 | 1.769302 | False |
| [F5](#f5) | LRG+ELG | wp+xi0+xi2+delta_sigma | Completed | 257 | 34 | 223 | 332.1401 | 1.489418 | False |

## Best-fit parameter comparison

| Parameter | F1 | F2 | F3 | F4 | F5 |
| --- | --- | --- | --- | --- | --- |
| M_0_LRG | — | 13.43645 | 13.19268 | — | 13.35939 |
| M_1_LRG | — | 13.43365 | 13.50144 | — | 13.48404 |
| alpha_LRG | — | 0.99552 | 1.213578 | — | 1.096772 |
| f_sigv_LRG | — | 1.127846 | 1.196522 | — | 1.036034 |
| log_Mcent_LRG | — | 12.8159 | 12.6427 | — | 12.79952 |
| sigma_M_LRG | — | 0.4397039 | 0.108115 | — | 0.4245562 |
| Ac_LRG | — | 0.7940909 | 0.2686307 | — | 0.6205689 |
| ab_c_cen_LRG | — | 0.4294762 | 0.1079405 | — | -0.3743002 |
| ab_c_sat_LRG | — | 0.8338784 | 0.2902623 | — | 0.5085442 |
| ab_env_cen_LRG | — | -0.1121149 | -0.4828621 | — | -0.6043234 |
| ab_env_sat_LRG | — | 0.09310843 | -0.226065 | — | -0.4368648 |
| ab_shear_cen_LRG | — | -0.4544537 | -0.3484473 | — | 0.2999136 |
| ab_shear_sat_LRG | — | -0.9321036 | 0.08702211 | — | -0.8887347 |
| nu_LRG | — | 0.9430462 | 1.5118 | — | 2.572085 |
| f_vcen_LRG | — | 0.1567233 | 0.2564597 | — | 0.2504786 |
| Ac_ELG | — | 0.421193 | — | 0.2581021 | 0.2708125 |
| M_0_ELG | — | 12.43781 | — | 12.45973 | 11.76985 |
| M_1_ELG | — | 12.79224 | — | 12.80112 | 12.97845 |
| alpha_ELG | — | 1.312402 | — | 0.8886195 | 1.10699 |
| gamma_ELG | — | 3.819652 | — | 7.084022 | 5.179286 |
| f_sigv_ELG | — | 0.6243351 | — | 1.233991 | 1.109234 |
| log_Mcent_ELG | — | 11.82966 | — | 11.68237 | 11.63857 |
| sigma_M_ELG | — | 0.04135843 | — | 0.7022078 | 0.5743173 |
| nu_ELG | — | 2.655036 | — | 3.479612 | 3.149661 |
| f_vcen_ELG | — | 0.6317817 | — | 0.07762088 | 0.472 |
| exp_frac_ELG | — | 0.4406038 | — | 0.5525711 | 0.7224617 |
| exp_scale_ELG | — | 5.23261 | — | 5.524864 | 2.756023 |
| nfw_rescale_ELG | — | 1.772308 | — | 0.5900491 | 1.135004 |
| ab_c_cen_ELG | — | -0.2390358 | — | 0.4415017 | 0.5937362 |
| ab_c_sat_ELG | — | -0.2591729 | — | 0.2361409 | -0.8081712 |
| ab_env_cen_ELG | — | -0.3108003 | — | 0.1423691 | -0.384141 |
| ab_env_sat_ELG | — | 0.2742924 | — | 0.3949187 | -0.3766805 |
| ab_shear_cen_ELG | — | -0.1204465 | — | -0.2947338 | 0.08585045 |
| ab_shear_sat_ELG | — | 0.3017376 | — | -0.1247405 | 0.4113894 |

<a id="f1"></a>

## F1: LRG / wp+xi0+xi2

Run ID: `af5920a22a5a4472b8decda4f258133b`. State: Data checked; no optimization.

| Setting | Value |
| --- | --- |
| Simulation | AbacusSummit_small_c000_ph3000 |
| Snapshot redshift | 0.5 |
| Data file | /global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/test_data/test_data_minimizer.npy |
| Covariance file | /global/cfs/cdirs/desicollab/users/arocher/HODDIES_results/Abcaus_small_boxes_corr/z0.500/Mcov_small_box_z0.5_wp+xi_LRG.txt |
| Source covariance pairs | ['LRG_LRG'] |
| Source covariance statistics | ['wp', 'xi0', 'xi2'] |
| Fitted pair order | ['LRG_LRG'] |
| Fitted statistics | ['wp', 'xi0', 'xi2'] |
| Covariance mock count | 1786 |
| Precision multiplier | 0.9607843 |
| Mock seed | 10 |
| MPI ranks | 1 |
| Threads/rank | 2 |
| Slurm job | — |
| Started (UTC) | 2026-10-21T08:09:43+00:00 |
| Finished (UTC) | — |
| Elapsed seconds | — |

### Models

**LRG**: `SHOD`.

| Input setting | Value |
| --- | --- |
| HOD_model | SHOD |
| Ac | 0.3491289 |
| log_Mcent | 12.66915 |
| sigma_M | 0.1719817 |
| gamma | 1 |
| pmax | 1 |
| Q | 100 |
| satellites | True |
| sat_HOD_model | Nsat_pow_law |
| As | 1 |
| M_0 | 12.97646 |
| M_1 | 13.62722 |
| alpha | 1.325596 |
| f_sigv | 1.19387 |
| f_vcen | 0.09930514 |
| vel_sat | rd_normal |
| v_infall | 0 |
| link_sat_to_central | True |
| assembly_bias | {'c': [0.3605874291029574, -0.006133712448889939], 'env': [0.16673375157378234, 0.7366165366863413], 'shear': [-0.9107959885376563, -0.1875148739088941]} |
| nu | 1.673389 |
| conformity_bias | False |
| exp_frac | 0 |
| exp_scale | 1 |
| nfw_rescale | 1 |
| density | 0.0007 |
| vsmear | 0 |
| ab_c_cen | 0.3605874 |
| ab_c_sat | -0.006133712 |
| ab_env_cen | 0.1667338 |
| ab_env_sat | 0.7366165 |
| ab_shear_cen | -0.910796 |
| ab_shear_sat | -0.1875149 |

### Optimizer

| Setting / diagnostic | Value |
| --- | --- |
| method | cmaes |
| maxiter | 100 |
| popsize | 5 |
| xtol | 1 |
| ftol | 34.5 |
| fun | — |
| nit | — |
| nfev | — |
| status | — |
| success | — |
| message | — |

### Fitted parameters

“Saved data input” is the HOD setting stored with the mock measurements. The optimizer starts from a seeded draw within the priors. Density normalization can change amplitudes.

| Parameter | Best fit | Saved data input | Prior minimum | Prior maximum |
| --- | --- | --- | --- | --- |
| M_0_LRG | — | 12.97646 | 12.5 | 13.5 |
| M_1_LRG | — | 13.62722 | 13 | 14.5 |
| alpha_LRG | — | 1.325596 | 0.5 | 1.5 |
| f_sigv_LRG | — | 1.19387 | 0.5 | 1.5 |
| log_Mcent_LRG | — | 12.66915 | 12.4 | 13.5 |
| sigma_M_LRG | — | 0.1719817 | 0.05 | 1 |
| Ac_LRG | — | 0.3491289 | 0 | 1 |
| ab_c_cen_LRG | — | 0.3605874 | -1 | 1 |
| ab_c_sat_LRG | — | -0.006133712 | -1 | 1 |
| ab_env_cen_LRG | — | 0.1667338 | -1 | 1 |
| ab_env_sat_LRG | — | 0.7366165 | -1 | 1 |
| ab_shear_cen_LRG | — | -0.910796 | -1 | 1 |
| ab_shear_sat_LRG | — | -0.1875149 | -1 | 1 |
| nu_LRG | — | 1.673389 | 0.5 | 4 |
| f_vcen_LRG | — | 0.09930514 | 0 | 1 |

### Artifacts

Output directory: `/global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/tmp/mock_fits/LRG_input_check`.

Current run files: [configuration.json](tmp/mock_fits/LRG_input_check/configuration.json), [fit_inputs.npz](tmp/mock_fits/LRG_input_check/fit_inputs.npz).

No comparison plot was saved for this run.

<a id="f2"></a>

## F2: LRG+ELG / wp+xi0+xi2

Run ID: `e9d2c978567443c387d03371353ef6b9`. State: Completed.

| Setting | Value |
| --- | --- |
| Simulation | AbacusSummit_small_c000_ph3000 |
| Snapshot redshift | 0.5 |
| Data file | /global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/test_data/test_data_minimizer.npy |
| Covariance file | /global/cfs/cdirs/desicollab/users/arocher/HODDIES_results/Abcaus_small_boxes_corr/z0.500/Mcov_small_box_z0.5_wp+xi_ELG_LRG.txt |
| Source covariance pairs | ['ELG_ELG', 'ELG_LRG', 'LRG_LRG'] |
| Source covariance statistics | ['wp', 'xi0', 'xi2'] |
| Fitted pair order | ['ELG_ELG', 'ELG_LRG', 'LRG_LRG'] |
| Fitted statistics | ['wp', 'xi0', 'xi2'] |
| Covariance mock count | 1786 |
| Precision multiplier | 0.8834734 |
| Mock seed | 10 |
| MPI ranks | 16 |
| Threads/rank | 16 |
| Slurm job | 60083108 |
| Started (UTC) | 2026-10-21T08:15:52+00:00 |
| Finished (UTC) | 2026-10-21T08:24:59+00:00 |
| Elapsed seconds | 547.6225 |

### Models

**LRG**: `SHOD`.

| Input setting | Value |
| --- | --- |
| HOD_model | SHOD |
| Ac | 0.3491289 |
| log_Mcent | 12.66915 |
| sigma_M | 0.1719817 |
| gamma | 1 |
| pmax | 1 |
| Q | 100 |
| satellites | True |
| sat_HOD_model | Nsat_pow_law |
| As | 1 |
| M_0 | 12.97646 |
| M_1 | 13.62722 |
| alpha | 1.325596 |
| f_sigv | 1.19387 |
| f_vcen | 0.09930514 |
| vel_sat | rd_normal |
| v_infall | 0 |
| link_sat_to_central | True |
| assembly_bias | {'c': [0.3605874291029574, -0.006133712448889939], 'env': [0.16673375157378234, 0.7366165366863413], 'shear': [-0.9107959885376563, -0.1875148739088941]} |
| nu | 1.673389 |
| conformity_bias | False |
| exp_frac | 0 |
| exp_scale | 1 |
| nfw_rescale | 1 |
| density | 0.0007 |
| vsmear | 0 |
| ab_c_cen | 0.3605874 |
| ab_c_sat | -0.006133712 |
| ab_env_cen | 0.1667338 |
| ab_env_sat | 0.7366165 |
| ab_shear_cen | -0.910796 |
| ab_shear_sat | -0.1875149 |

**ELG**: `mHMQ`.

| Input setting | Value |
| --- | --- |
| HOD_model | mHMQ |
| Ac | 0.2264542 |
| log_Mcent | 11.92708 |
| sigma_M | 0.1304524 |
| gamma | 4.056111 |
| pmax | 1 |
| Q | 100 |
| satellites | True |
| sat_HOD_model | Nsat_pow_law |
| As | 0.11 |
| M_0 | 12.17689 |
| M_1 | 12.82383 |
| alpha | 0.8503974 |
| f_sigv | 1.326974 |
| f_vcen | 0.3669561 |
| vel_sat | rd_normal |
| v_infall | 0 |
| link_sat_to_central | False |
| assembly_bias | {'c': [0.1517711324782447, -0.22123530620496], 'env': [0.4095395762108023, 0.4309918411409557], 'shear': [-0.5418936905119249, -0.2041099364426886]} |
| nu | 2.813184 |
| conformity_bias | False |
| exp_frac | 0.4651047 |
| exp_scale | 6.041057 |
| nfw_rescale | 1.212031 |
| density | 0.001 |
| vsmear | 0 |
| ab_c_cen | 0.1517711 |
| ab_c_sat | -0.2212353 |
| ab_env_cen | 0.4095396 |
| ab_env_sat | 0.4309918 |
| ab_shear_cen | -0.5418937 |
| ab_shear_sat | -0.2041099 |

### Optimizer

| Setting / diagnostic | Value |
| --- | --- |
| method | cmaes |
| maxiter | 200 |
| popsize | 16 |
| xtol | 1 |
| ftol | 103.5 |
| fun | 308.7328 |
| nit | 200 |
| nfev | 3200 |
| status | -1 |
| success | False |
| message | maximum number of iterations is reached |

### Fitted parameters

“Saved data input” is the HOD setting stored with the mock measurements. The optimizer starts from a seeded draw within the priors. Density normalization can change amplitudes.

| Parameter | Best fit | Saved data input | Prior minimum | Prior maximum |
| --- | --- | --- | --- | --- |
| M_0_LRG | 13.43645 | 12.97646 | 12.5 | 13.5 |
| M_1_LRG | 13.43365 | 13.62722 | 13 | 14.5 |
| alpha_LRG | 0.99552 | 1.325596 | 0.5 | 1.5 |
| f_sigv_LRG | 1.127846 | 1.19387 | 0.5 | 1.5 |
| log_Mcent_LRG | 12.8159 | 12.66915 | 12.4 | 13.5 |
| sigma_M_LRG | 0.4397039 | 0.1719817 | 0.05 | 1 |
| Ac_LRG | 0.7940909 | 0.3491289 | 0 | 1 |
| nu_LRG | 0.9430462 | 1.673389 | 0.5 | 4 |
| f_vcen_LRG | 0.1567233 | 0.09930514 | 0 | 1 |
| ab_c_cen_LRG | 0.4294762 | 0.3605874 | -1 | 1 |
| ab_c_sat_LRG | 0.8338784 | -0.006133712 | -1 | 1 |
| ab_env_cen_LRG | -0.1121149 | 0.1667338 | -1 | 1 |
| ab_env_sat_LRG | 0.09310843 | 0.7366165 | -1 | 1 |
| ab_shear_cen_LRG | -0.4544537 | -0.910796 | -1 | 1 |
| ab_shear_sat_LRG | -0.9321036 | -0.1875149 | -1 | 1 |
| Ac_ELG | 0.421193 | 0.2264542 | 0 | 0.5 |
| M_0_ELG | 12.43781 | 12.17689 | 11.5 | 12.5 |
| M_1_ELG | 12.79224 | 12.82383 | 12.5 | 14 |
| alpha_ELG | 1.312402 | 0.8503974 | 0.5 | 1.5 |
| gamma_ELG | 3.819652 | 4.056111 | 0.5 | 10 |
| f_sigv_ELG | 0.6243351 | 1.326974 | 0.5 | 1.5 |
| log_Mcent_ELG | 11.82966 | 11.92708 | 11.2 | 12.5 |
| sigma_M_ELG | 0.04135843 | 0.1304524 | 0 | 1 |
| nu_ELG | 2.655036 | 2.813184 | 0.5 | 4 |
| f_vcen_ELG | 0.6317817 | 0.3669561 | 0 | 1 |
| exp_frac_ELG | 0.4406038 | 0.4651047 | 0 | 1 |
| exp_scale_ELG | 5.23261 | 6.041057 | 0 | 8 |
| nfw_rescale_ELG | 1.772308 | 1.212031 | 0 | 2 |
| ab_c_cen_ELG | -0.2390358 | 0.1517711 | -1 | 1 |
| ab_c_sat_ELG | -0.2591729 | -0.2212353 | -1 | 1 |
| ab_env_cen_ELG | -0.3108003 | 0.4095396 | -1 | 1 |
| ab_env_sat_ELG | 0.2742924 | 0.4309918 | -1 | 1 |
| ab_shear_cen_ELG | -0.1204465 | -0.5418937 | -1 | 1 |
| ab_shear_sat_ELG | 0.3017376 | -0.2041099 | -1 | 1 |

### Artifacts

Output directory: `/global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/tmp/mock_fits/60083108`.

Archived run metadata: [e9d2c978567443c387d03371353ef6b9.json](tmp/mock_fits/60083108/run_history/e9d2c978567443c387d03371353ef6b9.json).

Current run files: [configuration.json](tmp/mock_fits/60083108/configuration.json), [fit_inputs.npz](tmp/mock_fits/60083108/fit_inputs.npz), [best_fit.npy](tmp/mock_fits/60083108/best_fit.npy), [best_fit_statistics.npz](tmp/mock_fits/60083108/best_fit_statistics.npz).

![F2: mock data and best fit](tmp/mock_fits/60083108/run_history/e9d2c978567443c387d03371353ef6b9.png)

<a id="f3"></a>

## F3: LRG / wp+xi0+xi2+delta_sigma

Run ID: `bdeb82e6f6a14df08409b3e5a83bdfdd`. State: Completed.

| Setting | Value |
| --- | --- |
| Simulation | AbacusSummit_small_c000_ph3000 |
| Snapshot redshift | 0.5 |
| Data file | /global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/test_data/test_data_minimizer.npy |
| Covariance file | /global/cfs/cdirs/desicollab/users/arocher/HODDIES_results/Abcaus_small_boxes_corr/z0.500/Mcov_small_box_z0.5_wp+xi+delta_sigma_LRG.txt |
| Source covariance pairs | ['LRG_LRG'] |
| Source covariance statistics | ['wp', 'xi0', 'xi2', 'delta_sigma'] |
| Fitted pair order | ['LRG_LRG'] |
| Fitted statistics | ['wp', 'xi0', 'xi2', 'delta_sigma'] |
| Covariance mock count | 1786 |
| Precision multiplier | 0.9467787 |
| Mock seed | 10 |
| MPI ranks | 16 |
| Threads/rank | 16 |
| Slurm job | 60083692 |
| Started (UTC) | 2026-10-21T08:25:59+00:00 |
| Finished (UTC) | 2026-10-21T08:31:25+00:00 |
| Elapsed seconds | 325.3553 |

### Models

**LRG**: `SHOD`.

| Input setting | Value |
| --- | --- |
| HOD_model | SHOD |
| Ac | 0.3491289 |
| log_Mcent | 12.66915 |
| sigma_M | 0.1719817 |
| gamma | 1 |
| pmax | 1 |
| Q | 100 |
| satellites | True |
| sat_HOD_model | Nsat_pow_law |
| As | 1 |
| M_0 | 12.97646 |
| M_1 | 13.62722 |
| alpha | 1.325596 |
| f_sigv | 1.19387 |
| f_vcen | 0.09930514 |
| vel_sat | rd_normal |
| v_infall | 0 |
| link_sat_to_central | True |
| assembly_bias | {'c': [0.3605874291029574, -0.006133712448889939], 'env': [0.16673375157378234, 0.7366165366863413], 'shear': [-0.9107959885376563, -0.1875148739088941]} |
| nu | 1.673389 |
| conformity_bias | False |
| exp_frac | 0 |
| exp_scale | 1 |
| nfw_rescale | 1 |
| density | 0.0007 |
| vsmear | 0 |
| ab_c_cen | 0.3605874 |
| ab_c_sat | -0.006133712 |
| ab_env_cen | 0.1667338 |
| ab_env_sat | 0.7366165 |
| ab_shear_cen | -0.910796 |
| ab_shear_sat | -0.1875149 |

### Optimizer

| Setting / diagnostic | Value |
| --- | --- |
| method | cmaes |
| maxiter | 200 |
| popsize | 16 |
| xtol | 1 |
| ftol | 47 |
| fun | 79.31442 |
| nit | 200 |
| nfev | 3200 |
| status | -1 |
| success | False |
| message | maximum number of iterations is reached |

### Fitted parameters

“Saved data input” is the HOD setting stored with the mock measurements. The optimizer starts from a seeded draw within the priors. Density normalization can change amplitudes.

| Parameter | Best fit | Saved data input | Prior minimum | Prior maximum |
| --- | --- | --- | --- | --- |
| M_0_LRG | 13.19268 | 12.97646 | 12.5 | 13.5 |
| M_1_LRG | 13.50144 | 13.62722 | 13 | 14.5 |
| alpha_LRG | 1.213578 | 1.325596 | 0.5 | 1.5 |
| f_sigv_LRG | 1.196522 | 1.19387 | 0.5 | 1.5 |
| log_Mcent_LRG | 12.6427 | 12.66915 | 12.4 | 13.5 |
| sigma_M_LRG | 0.108115 | 0.1719817 | 0.05 | 1 |
| Ac_LRG | 0.2686307 | 0.3491289 | 0 | 1 |
| nu_LRG | 1.5118 | 1.673389 | 0.5 | 4 |
| f_vcen_LRG | 0.2564597 | 0.09930514 | 0 | 1 |
| ab_c_cen_LRG | 0.1079405 | 0.3605874 | -1 | 1 |
| ab_c_sat_LRG | 0.2902623 | -0.006133712 | -1 | 1 |
| ab_env_cen_LRG | -0.4828621 | 0.1667338 | -1 | 1 |
| ab_env_sat_LRG | -0.226065 | 0.7366165 | -1 | 1 |
| ab_shear_cen_LRG | -0.3484473 | -0.910796 | -1 | 1 |
| ab_shear_sat_LRG | 0.08702211 | -0.1875149 | -1 | 1 |

### Artifacts

Output directory: `/global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/tmp/mock_fits/60083692`.

Archived run metadata: [bdeb82e6f6a14df08409b3e5a83bdfdd.json](tmp/mock_fits/60083692/run_history/bdeb82e6f6a14df08409b3e5a83bdfdd.json).

Current run files: [configuration.json](tmp/mock_fits/60083692/configuration.json), [fit_inputs.npz](tmp/mock_fits/60083692/fit_inputs.npz), [best_fit.npy](tmp/mock_fits/60083692/best_fit.npy), [best_fit_statistics.npz](tmp/mock_fits/60083692/best_fit_statistics.npz).

![F3: mock data and best fit](tmp/mock_fits/60083692/run_history/bdeb82e6f6a14df08409b3e5a83bdfdd.png)

<a id="f4"></a>

## F4: ELG / wp+xi0+xi2+delta_sigma

Run ID: `7a795f07c54d4985aeb32ff11251c17b`. State: Completed.

| Setting | Value |
| --- | --- |
| Simulation | AbacusSummit_small_c000_ph3000 |
| Snapshot redshift | 0.5 |
| Data file | /global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/test_data/test_data_minimizer.npy |
| Covariance file | /global/cfs/cdirs/desicollab/users/arocher/HODDIES_results/Abcaus_small_boxes_corr/z0.500/Mcov_small_box_z0.5_wp+xi+delta_sigma_ELG.txt |
| Source covariance pairs | ['ELG_ELG'] |
| Source covariance statistics | ['wp', 'xi0', 'xi2', 'delta_sigma'] |
| Fitted pair order | ['ELG_ELG'] |
| Fitted statistics | ['wp', 'xi0', 'xi2', 'delta_sigma'] |
| Covariance mock count | 1786 |
| Precision multiplier | 0.9467787 |
| Mock seed | 10 |
| MPI ranks | 16 |
| Threads/rank | 16 |
| Slurm job | 60083689 |
| Started (UTC) | 2026-10-21T08:26:00+00:00 |
| Finished (UTC) | 2026-10-21T08:32:33+00:00 |
| Elapsed seconds | 393.2854 |

### Models

**ELG**: `mHMQ`.

| Input setting | Value |
| --- | --- |
| HOD_model | mHMQ |
| Ac | 0.2264542 |
| log_Mcent | 11.92708 |
| sigma_M | 0.1304524 |
| gamma | 4.056111 |
| pmax | 1 |
| Q | 100 |
| satellites | True |
| sat_HOD_model | Nsat_pow_law |
| As | 0.11 |
| M_0 | 12.17689 |
| M_1 | 12.82383 |
| alpha | 0.8503974 |
| f_sigv | 1.326974 |
| f_vcen | 0.3669561 |
| vel_sat | rd_normal |
| v_infall | 0 |
| link_sat_to_central | False |
| assembly_bias | {'c': [0.1517711324782447, -0.22123530620496], 'env': [0.4095395762108023, 0.4309918411409557], 'shear': [-0.5418936905119249, -0.2041099364426886]} |
| nu | 2.813184 |
| conformity_bias | False |
| exp_frac | 0.4651047 |
| exp_scale | 6.041057 |
| nfw_rescale | 1.212031 |
| density | 0.001 |
| vsmear | 0 |
| ab_c_cen | 0.1517711 |
| ab_c_sat | -0.2212353 |
| ab_env_cen | 0.4095396 |
| ab_env_sat | 0.4309918 |
| ab_shear_cen | -0.5418937 |
| ab_shear_sat | -0.2041099 |

### Optimizer

| Setting / diagnostic | Value |
| --- | --- |
| method | cmaes |
| maxiter | 200 |
| popsize | 16 |
| xtol | 1 |
| ftol | 47 |
| fun | 132.6977 |
| nit | 200 |
| nfev | 3200 |
| status | -1 |
| success | False |
| message | maximum number of iterations is reached |

### Fitted parameters

“Saved data input” is the HOD setting stored with the mock measurements. The optimizer starts from a seeded draw within the priors. Density normalization can change amplitudes.

| Parameter | Best fit | Saved data input | Prior minimum | Prior maximum |
| --- | --- | --- | --- | --- |
| Ac_ELG | 0.2581021 | 0.2264542 | 0 | 0.5 |
| M_0_ELG | 12.45973 | 12.17689 | 11.5 | 12.5 |
| M_1_ELG | 12.80112 | 12.82383 | 12.5 | 14 |
| alpha_ELG | 0.8886195 | 0.8503974 | 0.5 | 1.5 |
| gamma_ELG | 7.084022 | 4.056111 | 0.5 | 10 |
| f_sigv_ELG | 1.233991 | 1.326974 | 0.5 | 1.5 |
| log_Mcent_ELG | 11.68237 | 11.92708 | 11.2 | 12.5 |
| sigma_M_ELG | 0.7022078 | 0.1304524 | 0 | 1 |
| nu_ELG | 3.479612 | 2.813184 | 0.5 | 4 |
| f_vcen_ELG | 0.07762088 | 0.3669561 | 0 | 1 |
| exp_frac_ELG | 0.5525711 | 0.4651047 | 0 | 1 |
| exp_scale_ELG | 5.524864 | 6.041057 | 0 | 8 |
| nfw_rescale_ELG | 0.5900491 | 1.212031 | 0 | 2 |
| ab_c_cen_ELG | 0.4415017 | 0.1517711 | -1 | 1 |
| ab_c_sat_ELG | 0.2361409 | -0.2212353 | -1 | 1 |
| ab_env_cen_ELG | 0.1423691 | 0.4095396 | -1 | 1 |
| ab_env_sat_ELG | 0.3949187 | 0.4309918 | -1 | 1 |
| ab_shear_cen_ELG | -0.2947338 | -0.5418937 | -1 | 1 |
| ab_shear_sat_ELG | -0.1247405 | -0.2041099 | -1 | 1 |

### Artifacts

Output directory: `/global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/tmp/mock_fits/60083689`.

Archived run metadata: [7a795f07c54d4985aeb32ff11251c17b.json](tmp/mock_fits/60083689/run_history/7a795f07c54d4985aeb32ff11251c17b.json).

Current run files: [configuration.json](tmp/mock_fits/60083689/configuration.json), [fit_inputs.npz](tmp/mock_fits/60083689/fit_inputs.npz), [best_fit.npy](tmp/mock_fits/60083689/best_fit.npy), [best_fit_statistics.npz](tmp/mock_fits/60083689/best_fit_statistics.npz).

![F4: mock data and best fit](tmp/mock_fits/60083689/run_history/7a795f07c54d4985aeb32ff11251c17b.png)

<a id="f5"></a>

## F5: LRG+ELG / wp+xi0+xi2+delta_sigma

Run ID: `bd52ea67e180487a9236e857f9e5ad08`. State: Completed.

| Setting | Value |
| --- | --- |
| Simulation | AbacusSummit_small_c000_ph3000 |
| Snapshot redshift | 0.5 |
| Data file | /global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/test_data/test_data_minimizer.npy |
| Covariance file | /global/cfs/cdirs/desicollab/users/arocher/HODDIES_results/Abcaus_small_boxes_corr/z0.500/Mcov_small_box_z0.5_wp+xi+delta_sigma_ELG_LRG.txt |
| Source covariance pairs | ['ELG_ELG', 'ELG_LRG', 'LRG_LRG'] |
| Source covariance statistics | ['wp', 'xi0', 'xi2', 'delta_sigma'] |
| Fitted pair order | ['ELG_ELG', 'ELG_LRG', 'LRG_LRG'] |
| Fitted statistics | ['wp', 'xi0', 'xi2', 'delta_sigma'] |
| Covariance mock count | 1786 |
| Precision multiplier | 0.8554622 |
| Mock seed | 10 |
| MPI ranks | 16 |
| Threads/rank | 16 |
| Slurm job | 60084816 |
| Started (UTC) | 2026-10-21T08:49:38+00:00 |
| Finished (UTC) | 2026-10-21T09:05:42+00:00 |
| Elapsed seconds | 964.89 |

### Models

**LRG**: `SHOD`.

| Input setting | Value |
| --- | --- |
| HOD_model | SHOD |
| Ac | 0.3491289 |
| log_Mcent | 12.66915 |
| sigma_M | 0.1719817 |
| gamma | 1 |
| pmax | 1 |
| Q | 100 |
| satellites | True |
| sat_HOD_model | Nsat_pow_law |
| As | 1 |
| M_0 | 12.97646 |
| M_1 | 13.62722 |
| alpha | 1.325596 |
| f_sigv | 1.19387 |
| f_vcen | 0.09930514 |
| vel_sat | rd_normal |
| v_infall | 0 |
| link_sat_to_central | True |
| assembly_bias | {'c': [0.3605874291029574, -0.006133712448889939], 'env': [0.16673375157378234, 0.7366165366863413], 'shear': [-0.9107959885376563, -0.1875148739088941]} |
| nu | 1.673389 |
| conformity_bias | False |
| exp_frac | 0 |
| exp_scale | 1 |
| nfw_rescale | 1 |
| density | 0.0007 |
| vsmear | 0 |
| ab_c_cen | 0.3605874 |
| ab_c_sat | -0.006133712 |
| ab_env_cen | 0.1667338 |
| ab_env_sat | 0.7366165 |
| ab_shear_cen | -0.910796 |
| ab_shear_sat | -0.1875149 |

**ELG**: `mHMQ`.

| Input setting | Value |
| --- | --- |
| HOD_model | mHMQ |
| Ac | 0.2264542 |
| log_Mcent | 11.92708 |
| sigma_M | 0.1304524 |
| gamma | 4.056111 |
| pmax | 1 |
| Q | 100 |
| satellites | True |
| sat_HOD_model | Nsat_pow_law |
| As | 0.11 |
| M_0 | 12.17689 |
| M_1 | 12.82383 |
| alpha | 0.8503974 |
| f_sigv | 1.326974 |
| f_vcen | 0.3669561 |
| vel_sat | rd_normal |
| v_infall | 0 |
| link_sat_to_central | False |
| assembly_bias | {'c': [0.1517711324782447, -0.22123530620496], 'env': [0.4095395762108023, 0.4309918411409557], 'shear': [-0.5418936905119249, -0.2041099364426886]} |
| nu | 2.813184 |
| conformity_bias | False |
| exp_frac | 0.4651047 |
| exp_scale | 6.041057 |
| nfw_rescale | 1.212031 |
| density | 0.001 |
| vsmear | 0 |
| ab_c_cen | 0.1517711 |
| ab_c_sat | -0.2212353 |
| ab_env_cen | 0.4095396 |
| ab_env_sat | 0.4309918 |
| ab_shear_cen | -0.5418937 |
| ab_shear_sat | -0.2041099 |

### Optimizer

| Setting / diagnostic | Value |
| --- | --- |
| method | cmaes |
| maxiter | 200 |
| popsize | 16 |
| xtol | 1 |
| ftol | 128.5 |
| fun | 332.1401 |
| nit | 200 |
| nfev | 3200 |
| status | -1 |
| success | False |
| message | maximum number of iterations is reached |

### Fitted parameters

“Saved data input” is the HOD setting stored with the mock measurements. The optimizer starts from a seeded draw within the priors. Density normalization can change amplitudes.

| Parameter | Best fit | Saved data input | Prior minimum | Prior maximum |
| --- | --- | --- | --- | --- |
| M_0_LRG | 13.35939 | 12.97646 | 12.5 | 13.5 |
| M_1_LRG | 13.48404 | 13.62722 | 13 | 14.5 |
| alpha_LRG | 1.096772 | 1.325596 | 0.5 | 1.5 |
| f_sigv_LRG | 1.036034 | 1.19387 | 0.5 | 1.5 |
| log_Mcent_LRG | 12.79952 | 12.66915 | 12.4 | 13.5 |
| sigma_M_LRG | 0.4245562 | 0.1719817 | 0.05 | 1 |
| Ac_LRG | 0.6205689 | 0.3491289 | 0 | 1 |
| nu_LRG | 2.572085 | 1.673389 | 0.5 | 4 |
| f_vcen_LRG | 0.2504786 | 0.09930514 | 0 | 1 |
| ab_c_cen_LRG | -0.3743002 | 0.3605874 | -1 | 1 |
| ab_c_sat_LRG | 0.5085442 | -0.006133712 | -1 | 1 |
| ab_env_cen_LRG | -0.6043234 | 0.1667338 | -1 | 1 |
| ab_env_sat_LRG | -0.4368648 | 0.7366165 | -1 | 1 |
| ab_shear_cen_LRG | 0.2999136 | -0.910796 | -1 | 1 |
| ab_shear_sat_LRG | -0.8887347 | -0.1875149 | -1 | 1 |
| Ac_ELG | 0.2708125 | 0.2264542 | 0 | 0.5 |
| M_0_ELG | 11.76985 | 12.17689 | 11.5 | 12.5 |
| M_1_ELG | 12.97845 | 12.82383 | 12.5 | 14 |
| alpha_ELG | 1.10699 | 0.8503974 | 0.5 | 1.5 |
| gamma_ELG | 5.179286 | 4.056111 | 0.5 | 10 |
| f_sigv_ELG | 1.109234 | 1.326974 | 0.5 | 1.5 |
| log_Mcent_ELG | 11.63857 | 11.92708 | 11.2 | 12.5 |
| sigma_M_ELG | 0.5743173 | 0.1304524 | 0 | 1 |
| nu_ELG | 3.149661 | 2.813184 | 0.5 | 4 |
| f_vcen_ELG | 0.472 | 0.3669561 | 0 | 1 |
| exp_frac_ELG | 0.7224617 | 0.4651047 | 0 | 1 |
| exp_scale_ELG | 2.756023 | 6.041057 | 0 | 8 |
| nfw_rescale_ELG | 1.135004 | 1.212031 | 0 | 2 |
| ab_c_cen_ELG | 0.5937362 | 0.1517711 | -1 | 1 |
| ab_c_sat_ELG | -0.8081712 | -0.2212353 | -1 | 1 |
| ab_env_cen_ELG | -0.384141 | 0.4095396 | -1 | 1 |
| ab_env_sat_ELG | -0.3766805 | 0.4309918 | -1 | 1 |
| ab_shear_cen_ELG | 0.08585045 | -0.5418937 | -1 | 1 |
| ab_shear_sat_ELG | 0.4113894 | -0.2041099 | -1 | 1 |

### Artifacts

Output directory: `/global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/tmp/mock_fits/60084816`.

Archived run metadata: [bd52ea67e180487a9236e857f9e5ad08.json](tmp/mock_fits/60084816/run_history/bd52ea67e180487a9236e857f9e5ad08.json).

Current run files: [configuration.json](tmp/mock_fits/60084816/configuration.json), [fit_inputs.npz](tmp/mock_fits/60084816/fit_inputs.npz), [best_fit.npy](tmp/mock_fits/60084816/best_fit.npy), [best_fit_statistics.npz](tmp/mock_fits/60084816/best_fit_statistics.npz).

![F5: mock data and best fit](tmp/mock_fits/60084816/run_history/bd52ea67e180487a9236e857f9e5ad08.png)
