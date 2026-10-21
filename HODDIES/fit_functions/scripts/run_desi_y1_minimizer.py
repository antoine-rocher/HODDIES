#!/usr/bin/env python3
"""Fit DESI Y1 clustering, optionally with LRG lensing at z=0.5.

Examples, from the repository root and a HODDIES/cosmodesi Python environment::

    python HODDIES/fit_functions/scripts/run_desi_y1_minimizer.py --tracers ELG --check-data
    python HODDIES/fit_functions/scripts/run_desi_y1_minimizer.py --tracers ELG
    python HODDIES/fit_functions/scripts/run_desi_y1_minimizer.py --tracers LRG
    python HODDIES/fit_functions/scripts/run_desi_y1_minimizer.py --tracers LRG ELG
    python HODDIES/fit_functions/scripts/run_desi_y1_minimizer.py --tracers LRG --with-delta-sigma --check-data

For MPI, add --mpi to the command launched with srun/mpiexec. Each rank loads
its own halo catalogue; choose ranks/node to fit the full highbase catalogue in
memory. --maxiter 2 --popsize 4 gives a short optimizer check, not a science fit.
--check-data reads and validates real measurements/covariance without halos.

The default remains wp + xi0 + xi2 at 0.8 < z < 1.1 with z=0.95 halos.
--with-delta-sigma selects LRG only, 0.4 < z < 0.6 clustering, and z=0.5 halos.
It reads rp, ds_obs_lrg1, ds_err_lrg1 from --delta-sigma-file, excludes missing
signal/error rows at the ends, and sets edges_rp=np.logspace(-1, 1.6, 13).
The model is evaluated at the CSV rp centres using the estimator’s internal
fine grid, so all 13 valid measurements remain aligned with the fit vector.
DeltaSigma uses the CSV units unchanged (h Msun/pc^2, matching HOD output).
Its covariance is diag(ds_err_lrg1**2), with zero clustering-lensing covariance.
The finite-mock precision correction applies to clustering only. Field particles
are loaded only when lensing is enabled. --z-simu 0.5 also supports LRG clustering
alone; other low-redshift tracers and lensing bins are unavailable here.

The data paths, cuts, density targets, and ELG ten-parameter profile follow
HOD/Dev/run_fit13-16.py, run_fit.py, and Y1_param_files/*z0.95*.yaml. In this bin,
use z=0.95 Abacus halos. The hybrid covariance is C_ij=R_small,ij*sigma_JK,i*sigma_JK,j.
The Y1 README says to use default_angular_bitwise rather than pip_* weights.

Vector order is pair, then [wp, xi0, xi2], then separation bin. Joint pairs are
[ELG_ELG, ELG_LRG, LRG_LRG], matching the legacy joint correlation matrix.
The pipeline receives the Hartlap deficit D=(ndata+1)/(nmocks-1) and applies
(1-D) once to the precision matrix, reproducing the old scripts. For a hybrid
covariance this is the legacy convention, not a new uncertainty calibration.

ELG: mHMQ, conformity, exponential + rescaled NFW satellites, fitted f_sigv,
no assembly bias. LRG: SHOD, concentration assembly bias for centrals/satellites,
fitted f_sigv and f_vcen, link_sat_to_central=True. The latter links the mean
satellite occupation to mean Ncen; conformity additionally conditions on the
actual central draw. LRG conformity is False. Joint draws populate LRG first;
ELG concentration-bias coefficients remain fixed at zero. Density targets are
legacy examples, exposed below as CLI options. No redshift-error smearing or
extra model-noise covariance is added. The current f_vcen prescription is
v_c,axis = v_h,axis + f_vcen*Vrms, as implemented by HOD.make_mock_cat.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

# Permit direct execution without importing a different installed checkout.
REPO_ROOT = Path(__file__).resolve().parents[3]
if (REPO_ROOT / 'HODDIES' / 'hod.py').is_file():
    sys.path.insert(0, str(REPO_ROOT))

DATA_ROOT = Path('/global/cfs/cdirs/desi/users/arocher/Y1/Y1_data_v2')
CORRELATION_ROOT = Path('/global/cfs/cdirs/desi/users/arocher/Y1/2PCF_for_corr/Abcaus_small_boxes')
ABACUS_ROOT = Path('/global/cfs/cdirs/desi/cosmosim/Abacus')
FIT_STATS = ['wp', 'xi_ells']
Z_SIM = 0.95
DELTA_SIGMA_FILE = REPO_ROOT.parent / 'tmp/bgs2_bgs3_lrg1_lrg2_best_fit_hsc.csv'


def available_cpu_count():
    """CPUs accessible to this rank, capped by its Slurm task allocation."""
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    try:
        allocated = int(os.environ.get('SLURM_CPUS_PER_TASK', '0'))
    except ValueError:
        allocated = 0
    if allocated > 0:
        count = min(count, allocated)
    return max(1, count)


@dataclass
class FitData:
    observations: dict
    errors: dict
    pairs: list[str]
    vector: np.ndarray
    sigma: np.ndarray
    correlation: np.ndarray
    covariance: np.ndarray
    clustering: dict
    files: dict
    correlation_file: Path
    hartlap_deficit: float
    redshift_bin: tuple = (0.8, 1.1)
    z_simu: float = Z_SIM
    lensing: dict | None = None
    precision: np.ndarray | None = None

    @property
    def statistics(self):
        return list(self.observations)


def tracer_order(tracers):
    """LRGs get central priority in joint HOD draws; fitting pairs sort separately."""
    requested = set(tracers)
    if not requested or requested - {'ELG', 'LRG'} or len(tracers) != len(requested):
        raise ValueError('Select ELG, LRG, or LRG ELG, without duplicates.')
    return [tracer for tracer in ('LRG', 'ELG') if tracer in requested]


def measurement_path(data_dir, mode, pair, region, weights, redshift_bin=(0.8, 1.1)):
    mapped = ['ELG_LOPnotqso' if tracer == 'ELG' else tracer for tracer in pair.split('_')]
    names = [mapped[0]] if mapped[0] == mapped[1] else ['_'.join(mapped), '_'.join(mapped[::-1])]
    candidates = [Path(data_dir) / mode /
                  f'allcounts_{name}_{region}_{redshift_bin[0]:g}_{redshift_bin[1]:g}_{weights}_log_njack128_nran4_split20.npy'
                  for name in names]
    for filename in candidates:
        if filename.is_file():
            return filename
    raise FileNotFoundError('Missing Y1 measurement; tried:\n' + '\n'.join(map(str, candidates)))


def load_delta_sigma(filename):
    """Read the LRG1 measurements using the configured fixed radial edges."""
    filename = Path(filename)
    saved = np.genfromtxt(filename, delimiter=',', names=True, ndmin=1)
    required = {'rp', 'ds_obs_lrg1', 'ds_err_lrg1'}
    if not required <= set(saved.dtype.names or ()):
        raise ValueError(f'{filename}: required columns are {sorted(required)}.')
    rp, signal, error = [np.asarray(saved[name], dtype=float)
                         for name in ('rp', 'ds_obs_lrg1', 'ds_err_lrg1')]
    if rp.size < 2 or not np.isfinite(rp).all() or np.any(rp <= 0) or np.any(np.diff(rp) <= 0):
        raise ValueError('Lensing rp must contain at least two finite, positive, increasing centres.')
    valid = np.isfinite(signal) & np.isfinite(error)
    selected = np.flatnonzero(valid)
    if not selected.size or np.any(error[valid] <= 0):
        raise ValueError('Lensing requires finite measurements with strictly positive errors.')
    if np.any(np.diff(selected) != 1):
        raise ValueError('Missing lensing rows inside the radial range are not supported.')
    edges = np.logspace(-1, 1.6, 13)
    if np.any(rp[valid] < edges[0]) or np.any(rp[valid] > edges[-1]):
        raise ValueError('Lensing measurement radii must lie within the configured radial edges.')
    metadata = dict(file=str(filename.resolve()), columns=['rp', 'ds_obs_lrg1', 'ds_err_lrg1'],
                    covariance='diagonal CSV errors squared', cross_covariance='zero with clustering',
                    units='h Msun/pc^2', rp_units='Mpc/h', excluded_rows=np.flatnonzero(~valid).tolist(),
                    edges_rp=edges.tolist(), bin_edges='np.logspace(-1, 1.6, 13)',
                    rp_eval=rp[valid].tolist(), evaluation='CSV rp centres on the internal fine model grid')
    return rp[valid], signal[valid], error[valid], edges, metadata


def load_fit_data(tracers, data_dir=DATA_ROOT, correlation_dir=CORRELATION_ROOT,
                  correlation_file=None, region='GCcomb', weights='default_angular_bitwise',
                  nmocks=1883, z_simu=None, with_delta_sigma=False, delta_sigma_file=DELTA_SIGMA_FILE):
    """Load the original Y1 cuts and align data, JK errors, and all covariance blocks."""
    from pycorr import TwoPointCorrelationFunction
    from HODDIES import HOD
    from HODDIES.fit_functions import flatten_stats

    tracers = tracer_order(tracers)
    z_simu = (0.5 if with_delta_sigma else Z_SIM) if z_simu is None else z_simu
    if z_simu not in (0.5, Z_SIM):
        raise ValueError('Supported simulation redshifts are 0.5 and 0.95.')
    if with_delta_sigma and (tracers != ['LRG'] or z_simu != 0.5):
        raise ValueError('delta_sigma is available only for LRG at z=0.5 (0.4 < z < 0.6).')
    if z_simu == 0.5 and tracers != ['LRG']:
        raise ValueError('Only LRG is supported in the 0.4 < z < 0.6 bin.')
    redshift_bin = (0.4, 0.6) if z_simu == 0.5 else (0.8, 1.1)
    pairs = ['_'.join(pair) for pair in HOD.get_comb_tr_list(tracers)]
    observations = {stat: {} for stat in FIT_STATS}
    errors = {stat: {} for stat in FIT_STATS}
    files, reference_edges = {}, {}
    for pair in pairs:
        files[pair] = {}
        for mode, statistic in [('rppi', 'wp'), ('smu', 'xi_ells')]:
            filename = measurement_path(data_dir, mode, pair, region, weights, redshift_bin)
            files[pair][mode] = str(filename)
            estimator = TwoPointCorrelationFunction.load(str(filename))
            estimator = estimator[::2][4:-3] if mode == 'rppi' else estimator[16:-6]
            options = {'pimax': 40} if mode == 'rppi' else {'ells': [0, 2]}
            separation, values, sigma = estimator(return_sep=True, return_std=True, **options)
            values, sigma = np.asarray(values), np.asarray(sigma)
            if values.shape != sigma.shape or not np.isfinite(values).all():
                raise ValueError(f'Invalid measurement values/shapes in {filename}')
            if not np.isfinite(sigma).all() or np.any(sigma <= 0):
                raise ValueError(f'Invalid jackknife errors in {filename}')
            observations[statistic][pair] = (np.asarray(separation), values)
            errors[statistic][pair] = sigma
            edges = tuple(np.array(edge, copy=True) for edge in estimator.edges)
            if mode in reference_edges:
                for expected, actual in zip(reference_edges[mode], edges):
                    if expected.shape != actual.shape or not np.allclose(expected, actual):
                        raise ValueError(f'Tracer pairs have incompatible {mode} bin edges.')
            else:
                reference_edges[mode] = edges

    vector = flatten_stats(observations, FIT_STATS, pairs)
    sigma = flatten_stats(errors, FIT_STATS, pairs)  # Bare arrays: no coordinate/error ambiguity.
    if correlation_file is None:
        label = '_'.join(sorted(tracers))
        correlation_file = Path(correlation_dir) / f'z{z_simu:.3f}' / f'Mcorr_small_box_z{z_simu:g}_wp+xi_{label}.txt'
    correlation_file = Path(correlation_file)
    correlation = np.loadtxt(correlation_file)
    expected_shape = (vector.size, vector.size)
    if correlation.shape != expected_shape:
        raise ValueError(f'{correlation_file}: shape {correlation.shape}, expected {expected_shape}. '
                         f'Required order: {pairs}, each with wp, xi0, xi2. No matrix cropping is applied.')
    if (not np.isfinite(correlation).all() or
            not np.allclose(correlation, correlation.T, atol=1e-10) or
            not np.allclose(np.diag(correlation), 1., atol=1e-8)):
        raise ValueError('The small-box correlation matrix must be finite, symmetric, with unit diagonal.')
    np.linalg.cholesky(correlation)
    covariance = correlation * np.outer(sigma, sigma)
    if nmocks <= vector.size + 2:
        raise ValueError('nmocks must exceed ndata + 2 for the legacy precision correction.')
    deficit = (vector.size + 1.) / (nmocks - 1.)
    clustering = dict(
        rsd=True, los='z',
        xi_rppi=dict(pimax=40, edges_rppi=reference_edges['rppi']),
        xi_smu=dict(multipole_index=[0, 2], edges_smu=reference_edges['smu']),
    )
    lensing, precision = None, None
    if with_delta_sigma:
        rp, ds, ds_error, edges, lensing = load_delta_sigma(delta_sigma_file)
        observations['delta_sigma'] = {'LRG_LRG': (rp, ds)}
        errors['delta_sigma'] = {'LRG_LRG': ds_error}
        nclustering = vector.size
        vector = flatten_stats(observations, list(observations), pairs)
        sigma = np.concatenate([sigma, ds_error])
        combined = np.eye(vector.size)
        combined[:nclustering, :nclustering] = correlation
        correlation = combined
        covariance = correlation * np.outer(sigma, sigma)
        precision = np.linalg.inv(covariance)
        precision[:nclustering, :nclustering] *= 1. - deficit
        lensing.update(clustering_ndata=int(nclustering), precision_correction='clustering block only')
        clustering['delta_sigma'] = dict(edges_rp=edges, rp_eval=rp.copy(), pimax=40)
        files['LRG_LRG']['delta_sigma'] = str(Path(delta_sigma_file).resolve())
    return FitData(observations, errors, pairs, vector, sigma, correlation,
                   covariance, clustering, files, correlation_file, deficit,
                   redshift_bin, z_simu, lensing, precision)


def model_configuration(tracers, output, nthreads=4, seed=10,
                        elg_density=0.001, lrg_density=0.0007):
    """Legacy ELG profile plus SHOD LRG velocity/concentration-bias example."""
    tracers = tracer_order(tracers)
    models = {
        'ELG': dict(HOD_model='mHMQ', Ac=0.1, log_Mcent=11.64, sigma_M=0.30, gamma=5.57,
                    satellites=True, sat_HOD_model='Nsat_pow_law', As=0.41,
                    M_0=11.20, M_1=13., alpha=0.81, nu=1., density=elg_density,
                    conformity_bias=True, link_sat_to_central=False,
                    vel_sat='rd_normal', f_sigv=1.63, f_vcen=0., vsmear=None,
                    exp_frac=0.55, exp_scale=6.14, nfw_rescale=0.67,
                    assembly_bias={'c': [0., 0.]}),
        'LRG': dict(HOD_model='SHOD', Ac=1., log_Mcent=12.67, sigma_M=0.08,
                    satellites=True, sat_HOD_model='Nsat_pow_law', As=1.,
                    M_0=12.61, M_1=13.78, alpha=1.03, nu=1., density=lrg_density,
                    conformity_bias=False, link_sat_to_central=True,
                    vel_sat='rd_normal', f_sigv=1., f_vcen=0.1, vsmear=None,
                    exp_frac=0., exp_scale=1., nfw_rescale=1.,
                    assembly_bias={'c': [0., 0.]}),
    }
    priors = {
        'ELG': dict(M_0=[11., 11.8], As=[0.01, 0.7], log_Mcent=[11.4, 12.2],
                    alpha=[0.3, 1.5], f_sigv=[0.8, 2.2], sigma_M=[0.05, 1.],
                    gamma=[1., 8.], exp_scale=[0.9, 10.], exp_frac=[0., 1.],
                    nfw_rescale=[0.1, 2.]),
        'LRG': dict(M_0=[12.4, 13.5], M_1=[13., 14.5], log_Mcent=[12.4, 13.2],
                    alpha=[0.5, 1.5], f_sigv=[0.5, 1.5], sigma_M=[0.02, 0.5],
                    f_vcen=[0., 1.], assembly_bias={'c': [[-1., 1.], [-1., 1.]]}),
    }
    return dict(
        tracers=tracers, **{tr: models[tr] for tr in tracers},
        use_assembly_bias='LRG' in tracers, use_particles=False,
        nthreads=nthreads, seed=seed, cosmo={'engine': 'class'},
        fit_param=dict(priors={tr: priors[tr] for tr in tracers},
                       fit_statistics=FIT_STATS.copy(), add_poisson_noise=False,
                       dir_output_fit=str(output), minimizer={'save_fn': 'best_fit.npy'}),
    )


def build_hod(simulation, configuration, inputs):
    from HODDIES import HOD

    configuration = deepcopy(configuration)
    priors = deepcopy(configuration['fit_param']['priors'])
    configuration['clustering_settings'] = deepcopy(inputs.clustering)
    configuration['fit_param']['fit_statistics'] = inputs.statistics.copy()
    hod = HOD(simulation, **configuration)
    # HOD merges dictionaries recursively: remove default LRG priors in ELG-only runs.
    hod.args['fit_param']['priors'] = priors
    if 'LRG' in configuration['tracers'] and 'ab_c' not in hod.hcat.columns():
        raise ValueError('Requested LRG concentration assembly bias was not initialized.')
    if inputs.precision is None:
        hod.initialize_fit(inputs.vector, inputs.covariance, hartlap_factor=inputs.hartlap_deficit)
    else:
        # Encode the block-specific correction in the effective covariance;
        # initialize_fit must not apply a second, global Hartlap multiplier.
        hod.initialize_fit(inputs.vector, np.linalg.inv(inputs.precision), hartlap_factor=0.)
    if hod._comb_tr_list != inputs.pairs:
        raise ValueError('Model pair ordering does not match the covariance ordering.')
    return hod


def initial_parameters(hod):
    values = []
    for name in hod.name_params:
        parameter, tracer = name.rsplit('_', 1)
        if parameter.startswith('ab_'):
            proxy, component = parameter[3:].rsplit('_', 1)
            value = hod.args[tracer]['assembly_bias'][proxy][0 if component == 'cen' else 1]
        else:
            value = hod.args[tracer][parameter]
        values.append(value)
    values = np.asarray(values, dtype=float)
    bounds = np.asarray(hod.priors)
    if not np.all((values >= bounds[:, 0]) & (values <= bounds[:, 1])):
        raise ValueError('Initial HOD parameters fall outside their priors.')
    return values


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def save_inputs(inputs, configuration, output, simulation_parameters, nmocks, run_metadata=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    arrays = dict(data_vector=inputs.vector, sigma=inputs.sigma,
                  sigma_jk=inputs.sigma[:inputs.lensing['clustering_ndata']] if inputs.lensing else inputs.sigma,
                  correlation=inputs.correlation, covariance=inputs.covariance,
                  pair_order=np.asarray(inputs.pairs), hartlap_deficit=inputs.hartlap_deficit,
                  statistic_order=np.asarray(inputs.statistics))
    if inputs.precision is not None:
        arrays['precision_matrix'] = inputs.precision
    for statistic in inputs.statistics:
        for pair, (x, y) in inputs.observations[statistic].items():
            arrays.update({f'{pair}_{statistic}_x': x, f'{pair}_{statistic}_y': y,
                           f'{pair}_{statistic}_sigma': inputs.errors[statistic][pair]})
    np.savez(output / 'fit_inputs.npz', **arrays)
    manifest = dict(configuration=configuration, simulation=simulation_parameters,
                    data_files=inputs.files, correlation_file=inputs.correlation_file,
                    pair_order=inputs.pairs, statistic_order=[name for statistic in inputs.statistics
                        for name in (['xi0', 'xi2'] if statistic == 'xi_ells' else [statistic])],
                    redshift_bin=list(inputs.redshift_bin), nmocks=nmocks,
                    precision_multiplier=1. - inputs.hartlap_deficit,
                    cuts={'rppi': '[::2][4:-3]', 'smu': '[16:-6]', 'pimax': 40})
    if inputs.lensing is not None:
        manifest['lensing'] = inputs.lensing
    if run_metadata is not None:
        manifest['run'] = run_metadata
    (output / 'configuration.json').write_text(json.dumps(manifest, indent=2, default=_json_default) + '\n')


def save_best_fit(hod, result, inputs, output, seed, plot=True):
    """Restore the optimizer's best point, rather than its last evaluated point."""
    from HODDIES.fit_functions import flatten_stats

    point = np.asarray([result['x']], dtype=float).copy()
    point.dtype = [(name, 'f8') for name in hod.name_params]
    hod.update_new_param(point, hod.name_params)
    mock = hod.make_mock_cat(fix_seed=seed, verbose=False)
    model = hod.compute_stats(mock, stat=inputs.statistics, tracers=hod._tracer_to_fit)
    vector = flatten_stats(model, inputs.statistics, inputs.pairs)
    residual = inputs.vector - vector
    chi2 = float(residual @ hod.inv_cov2 @ residual)
    arrays = dict(model_vector=vector, parameters=result['x'],
                  parameter_names=np.asarray(hod.name_params), chi2=chi2)
    for statistic in inputs.statistics:
        for pair in inputs.observations[statistic]:
            x, y = model[statistic][pair]
            arrays.update({f'{pair}_{statistic}_x': x, f'{pair}_{statistic}_y': y})
    np.savez(Path(output) / 'best_fit_statistics.npz', **arrays)
    print(f'Best-point reevaluation: chi2={chi2:.6g}; optimizer value={result["fun"]:.6g}', flush=True)
    if not plot:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = {statistic: {pair: (*inputs.observations[statistic][pair], inputs.errors[statistic][pair])
                        for pair in inputs.observations[statistic]} for statistic in inputs.statistics}
    fig = hod.plot_stats(mock, stats=inputs.statistics, tracers=list(hod._tracer_to_fit),
                         data=data, residuals=True, cross=True, by_tracer='rows',
                         show=False, save_fn=str(Path(output) / 'best_fit.png'))
    plt.close(fig)


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument('--tracers', nargs='+', choices=['ELG', 'LRG'], default=['ELG'])
    result.add_argument('--with-delta-sigma', '--delta-sigma', action='store_true',
                        help='Include CSV LRG1 lensing, using LRG only at z=0.5 and 0.4 < z < 0.6.')
    result.add_argument('--delta-sigma-file', type=Path, default=DELTA_SIGMA_FILE,
                        help='CSV containing rp, ds_obs_lrg1, ds_err_lrg1; errors define a diagonal covariance.')
    result.add_argument('--z-simu', type=float, choices=[0.5, Z_SIM],
                        help='Default: 0.5 with lensing, otherwise 0.95. Determines the clustering redshift bin.')
    result.add_argument('--data-dir', type=Path, default=DATA_ROOT)
    result.add_argument('--correlation-dir', type=Path, default=CORRELATION_ROOT)
    result.add_argument('--correlation-file', type=Path, help='Override matrix in the documented pair/bin order.')
    result.add_argument('--region', default='GCcomb')
    result.add_argument('--weights', default='default_angular_bitwise')
    result.add_argument('--nmocks', type=int, default=1883)
    result.add_argument('--abacus-root', type=Path, default=ABACUS_ROOT)
    result.add_argument('--sim-name', default='AbacusSummit_highbase_c000_ph100')
    result.add_argument('--mass-cut', type=float, default=None)
    result.add_argument('--elg-density', type=float, default=0.001)

    result.add_argument('--lrg-density', type=float, default=0.0007)
    result.add_argument('--nthreads', type=int, default=None,
                        help='Threads per MPI rank (default: CPU affinity, capped by SLURM_CPUS_PER_TASK).')
    result.add_argument('--seed', type=int, default=10)
    result.add_argument('--method', choices=['cmaes', 'de', 'pso', 'cpso', 'na', 'vdcma'], default='cmaes')
    result.add_argument('--maxiter', type=int, default=300)
    result.add_argument('--popsize', type=int, default=None,
                        help='Population size (default: MPI rank count with --mpi, minimum 2; otherwise 5).')
    result.add_argument('--xtol', type=float, default=1)
    result.add_argument('--ftol', type=float, default=None,
                        help='Objective target for convergence (chi2 for CMA-ES; default: number of fitted data points / 2).')
    result.add_argument('--output', type=Path)
    result.add_argument('--check-data', action='store_true')
    result.add_argument('--mpi', action='store_true')
    result.add_argument('--no-plot', action='store_true')
    result.add_argument('--report', type=Path, default=REPO_ROOT / 'DESI_Y1_FIT_RESULTS.md',
                        help='Markdown report to update after saving results.')
    result.add_argument('--no-report', action='store_true', help='Skip automatic Markdown report updates.')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
    tracers = tracer_order(args.tracers)
    args.z_simu = (0.5 if args.with_delta_sigma else Z_SIM) if args.z_simu is None else args.z_simu
    if args.with_delta_sigma and (tracers != ['LRG'] or args.z_simu != 0.5):
        raise ValueError('delta_sigma is available only for LRG at z=0.5 (use --tracers LRG).')
    if args.z_simu == 0.5 and tracers != ['LRG']:
        raise ValueError('Only LRG is supported in the 0.4 < z < 0.6 bin.')
    if args.elg_density <= 0 or args.lrg_density <= 0:
        raise ValueError('Density targets must be positive.')
    comm = None
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank() if comm is not None else 0
    if args.nthreads is None:
        args.nthreads = available_cpu_count()
    if args.popsize is None:
        args.popsize = max(2, comm.Get_size()) if comm is not None else 5
    if args.nthreads < 1 or args.maxiter < 1 or args.popsize < 2 or args.xtol <= 0:
        raise ValueError('Require nthreads/maxiter >= 1, popsize >= 2, xtol > 0.')
    if args.ftol is not None and (not np.isfinite(args.ftol) or args.ftol < 0):
        raise ValueError('Require ftol >= 0 and finite.')
    try:
        from HODDIES.sim_loader import AbacusSummitSim, setup_logging
        setup_logging(stream=sys.stdout)
        inputs = (load_fit_data(tracers, args.data_dir, args.correlation_dir,
                                args.correlation_file, args.region, args.weights, args.nmocks,
                                z_simu=args.z_simu, with_delta_sigma=args.with_delta_sigma,
                                delta_sigma_file=args.delta_sigma_file)
                  if rank == 0 else None)
        if comm is not None:
            inputs = comm.bcast(inputs, root=0)
        if args.ftol is None:
            args.ftol = inputs.vector.size / 2.0
        label = '_'.join(sorted(tracers))
        redshift_label = f'{inputs.redshift_bin[0]:g}_{inputs.redshift_bin[1]:g}'
        suffix = '_delta_sigma' if args.with_delta_sigma else ''
        output = args.output or Path('tmp') / f'desi_y1_{label}_z{redshift_label}{suffix}'
        configuration = model_configuration(tracers, output, args.nthreads, args.seed,
                                              args.elg_density, args.lrg_density)
        configuration['fit_param']['fit_statistics'] = inputs.statistics.copy()
        configuration['fit_param']['minimizer'].update(
            method=args.method, maxiter=args.maxiter, popsize=args.popsize,
            xtol=args.xtol, ftol=args.ftol)
        simulation_parameters = dict(path_to_sim=str(args.abacus_root), sim_name=args.sim_name,
                                     z_simu=args.z_simu, mass_cut=args.mass_cut, nthreads=args.nthreads,
                                     load_particles=False, load_field_particles=args.with_delta_sigma)
        if rank == 0:
            run_metadata = dict(id=uuid.uuid4().hex, started_at=started_at,
                                state='Data checked; no optimization' if args.check_data else
                                      'Started; completion not recorded',
                                mpi_ranks=comm.Get_size() if comm is not None else 1,
                                slurm_job_id=os.environ.get('SLURM_JOB_ID'))
            save_inputs(inputs, configuration, output, simulation_parameters, args.nmocks, run_metadata)
            print(f'Pairs: {inputs.pairs}; ndata={inputs.vector.size}; '
                  f'precision multiplier={1-inputs.hartlap_deficit:.8f}', flush=True)
            print(f'Hybrid covariance correlation source: {inputs.correlation_file}', flush=True)
        if args.check_data:
            return 0
        if rank == 0:
            print(f'Optimizer: {args.method}; popsize={args.popsize}; '
                  f'MPI ranks={comm.Get_size() if comm is not None else 1}; '
                  f'threads/rank={args.nthreads}; ftol={args.ftol:g}', flush=True)
        simulation = AbacusSummitSim(**simulation_parameters)
        hod = build_hod(simulation, configuration, inputs)
        initial = initial_parameters(hod)
        np.random.seed(args.seed)
        result = hod.run_minimizer(init_params=initial, seed=args.seed, mpi_comm=comm,
                                   return_log10chi2=False, method=args.method,
                                   maxiter=args.maxiter, popsize=args.popsize, xtol=args.xtol,
                                   ftol=args.ftol)
        if rank == 0:
            save_best_fit(hod, result, inputs, output, args.seed, plot=not args.no_plot)
            from HODDIES.fit_functions.scripts.summarize_desi_y1_fits import save_run_summary, update_report
            save_run_summary(output, result, hod.priors, time.monotonic() - started, plot=not args.no_plot)
            if not args.no_report:
                try:
                    update_report(args.report, extra_roots=[output])
                    print(f'Updated fit report: {args.report}', flush=True)
                except Exception as error:
                    # Report I/O must not discard a completed fit or abort other ranks.
                    print(f'Fit saved, but report update failed: {error}', file=sys.stderr, flush=True)
            print(result, flush=True)
        if comm is not None:
            comm.Barrier()
        return 0
    except Exception:
        if comm is not None and comm.Get_size() > 1:
            import traceback
            traceback.print_exc()
            comm.Abort(1)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
