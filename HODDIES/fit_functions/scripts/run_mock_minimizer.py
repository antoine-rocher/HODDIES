#!/usr/bin/env python3
"""Fit the saved mock-test measurements with selectable tracers and statistics.

Run from the repository root in the cosmodesi environment::

    python HODDIES/fit_functions/scripts/run_mock_minimizer.py --tracers LRG --statistics wp xi_ells --check-data
    python HODDIES/fit_functions/scripts/run_mock_minimizer.py --tracers LRG --statistics wp
    python HODDIES/fit_functions/scripts/run_mock_minimizer.py --tracers LRG ELG --statistics wp xi2 --covariance-file /path/to/joint.txt

The default data file is the trusted, pickled test_data_minimizer.npy bundled
with this checkout. Model settings, priors, and bin edges come from that file.
xi_ells expands to the saved multipoles (xi0 and xi2 in the bundled data).
Covariance order is pair, statistic, multipole, bin; delta_sigma is auto-only.
The source covariance filename follows the selected statistic families: wp,
xi (all saved multipoles), and delta_sigma. For example wp + xi_ells uses
Mcov_small_box_z0.5_wp+xi_<tracers>.txt, without a lensing block. Describe
other layouts with --covariance-statistics and --covariance-tracers. Subsets
retain their full covariance, including cross blocks. No SV3 measurement files are needed. Field particles load only for
lensing fits. The simulation defaults to ph3000 at z=0.5, as in the old test.

MPI is opt-in: launch with srun/mpiexec and --mpi. Each rank loads a catalogue.
Rank 0 saves results and updates MOCK_FIT_RESULTS.md. Each invocation gets a new
output directory unless --output is supplied; repeated directories keep JSON
and plot history, while their latest arrays are replaced. --check-data loads
no halos and performs no optimization. No parameter uncertainties are estimated.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations_with_replacement
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
DATA_FILE = REPO_ROOT / 'HODDIES/fit_functions/test_data/test_data_minimizer.npy'
COVARIANCE_DIR = Path('/global/cfs/cdirs/desi/users/arocher/HODDIES_results/Abcaus_small_boxes_corr/z0.500')
ABACUS_ROOT = Path('/global/cfs/cdirs/desi/cosmosim/Abacus')
WORKFLOW = 'mock_minimizer_v1'


def reporting():
    # Importing the adjacent script also works without importing HODDIES.__init__.
    import importlib.util
    name = '_hoddies_mock_fit_reporting'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name('summarize_mock_fits.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def tracer_order(tracers):
    if not tracers or len(set(tracers)) != len(tracers) or set(tracers) - {'LRG', 'ELG'}:
        raise ValueError('Choose LRG, ELG, or LRG ELG without duplicates.')
    return [tracer for tracer in ('LRG', 'ELG') if tracer in tracers]


def pair_order(tracers):
    return ['_'.join(pair) for pair in combinations_with_replacement(sorted(tracers), 2)]


def expand_statistics(statistics, parameters):
    multipoles = parameters['clustering_settings']['xi_smu']['multipole_index']
    expanded = [name for stat in statistics for name in
                ([f'xi{ell}' for ell in multipoles] if stat == 'xi_ells' else [stat])]
    if not expanded or len(set(expanded)) != len(expanded):
        raise ValueError('Choose nonempty statistics without duplicates or overlapping xi_ells/xiN.')
    if set(expanded) - {'wp', 'delta_sigma', *(f'xi{ell}' for ell in multipoles)}:
        raise ValueError('Requested statistic is absent from the saved multipole configuration.')
    return expanded


def observation(data, parameters, statistic, pair):
    if statistic.startswith('xi') and statistic[2:].isdigit():
        x, y = data['xi_ells'][pair]
        multipoles = parameters['clustering_settings']['xi_smu']['multipole_index']
        index = multipoles.index(int(statistic[2:]))
        y = np.asarray(y)
        if y.ndim != 2 or y.shape[0] != len(multipoles):
            raise ValueError(f'Invalid saved multipoles for {pair}: {y.shape}.')
        x = np.asarray(x)
        x = x[index] if x.ndim > 1 else x
        y = y[index]
    else:
        x, y = data[statistic][pair]
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or not y.size or not np.isfinite([x, y]).all():
        raise ValueError(f'Invalid saved {statistic}/{pair} measurements.')
    return x.copy(), y.copy()


@dataclass
class FitInputs:
    parameters: dict
    statistics: list
    pairs: list
    observations: dict
    errors: dict
    vector: np.ndarray
    covariance: np.ndarray
    indices: np.ndarray
    covariance_file: Path
    covariance_statistics: list
    covariance_pairs: list
    hartlap_deficit: float


def load_fit_data(data_file, tracers, statistics, covariance_file=None,
                  covariance_tracers=None, covariance_statistics=None,
                  nmocks=1786, no_hartlap=False):
    """Select by named source blocks, never by truncating the covariance matrix."""
    saved = np.load(data_file, allow_pickle=True).item()
    parameters, data = saved['param'], saved['data']
    tracers = tracer_order(tracers)
    stats = expand_statistics(statistics, parameters)
    if covariance_statistics is None:
        # Matrix files use canonical family order; xi contains all saved multipoles.
        covariance_statistics = [name for name, selected in
                                 [('wp', 'wp' in stats),
                                  ('xi_ells', any(stat.startswith('xi') for stat in stats)),
                                  ('delta_sigma', 'delta_sigma' in stats)] if selected]
    source_stats = expand_statistics(covariance_statistics, parameters)
    source_tracers = tracer_order(covariance_tracers or tracers)
    if not set(tracers) <= set(source_tracers) or not set(stats) <= set(source_stats):
        raise ValueError('Covariance layout must contain every fitted tracer and statistic.')
    pairs, source_pairs = pair_order(tracers), pair_order(source_tracers)
    if covariance_file is None:
        label = '+'.join('xi' if stat == 'xi_ells' else stat for stat in covariance_statistics)
        covariance_file = COVARIANCE_DIR / f'Mcov_small_box_z0.5_{label}_{"_".join(sorted(source_tracers))}.txt'
    covariance_file = Path(covariance_file).resolve()
    if not covariance_file.is_file():
        raise FileNotFoundError(f'Missing covariance: {covariance_file}. Supply --covariance-file and '
                                'describe its ordering with --covariance-tracers/--covariance-statistics.')
    blocks, offset = {}, 0
    for pair in source_pairs:
        for statistic in source_stats:
            if statistic == 'delta_sigma' and pair.split('_')[0] != pair.split('_')[1]:
                continue
            _, y = observation(data, parameters, statistic, pair)
            blocks[pair, statistic] = np.arange(offset, offset + y.size)
            offset += y.size
    matrix = np.loadtxt(covariance_file)
    if matrix.shape != (offset, offset):
        raise ValueError(f'Covariance shape {matrix.shape}, expected {(offset, offset)} for '
                         f'{source_pairs}, each with {source_stats} (auto-only delta_sigma).')
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError('Covariance must be finite and symmetric.')
    observations = {stat: {} for stat in stats}
    selected, values = [], []
    for pair in pairs:
        for statistic in stats:
            if statistic == 'delta_sigma' and pair.split('_')[0] != pair.split('_')[1]:
                continue
            x, y = observation(data, parameters, statistic, pair)
            observations[statistic][pair] = (x, y)
            selected.extend(blocks[pair, statistic])
            values.extend(y)
    indices = np.asarray(selected, dtype=int)
    covariance = matrix[np.ix_(indices, indices)]
    np.linalg.cholesky(covariance)
    vector = np.asarray(values)
    if not no_hartlap and nmocks <= vector.size + 2:
        raise ValueError('nmocks must exceed ndata + 2 for the finite-mock precision correction.')
    deficit = 0. if no_hartlap else (vector.size + 1.) / (nmocks - 1.)
    errors, start = {stat: {} for stat in stats}, 0
    sigma = np.sqrt(np.diag(covariance))
    for pair in pairs:
        for statistic in stats:
            if pair in observations[statistic]:
                size = observations[statistic][pair][1].size
                errors[statistic][pair] = sigma[start:start + size]
                start += size
    return FitInputs(parameters, stats, pairs, observations, errors, vector, covariance,
                     indices, covariance_file, source_stats, source_pairs, deficit)


def model_configuration(inputs, tracers, output, nthreads, seed):
    config = deepcopy(inputs.parameters)
    for tracer in config['tracers']:
        if tracer not in tracers:
            config.pop(tracer, None)
    config.update(tracers=tracer_order(tracers), nthreads=nthreads, seed=seed, setup_logger=True)
    # Use the selected simulation settings, rather than the data-producing halo catalogue.
    config.pop('hcat', None)
    fit = config['fit_param']
    fit['priors'] = {tracer: fit['priors'][tracer] for tracer in config['tracers']}
    fit.update(fit_statistics=inputs.statistics.copy(), dir_output_fit=str(output),
               add_poisson_noise=False, hartlap_factor=inputs.hartlap_deficit)
    fit['minimizer'] = dict(save_fn='best_fit.npy')
    return config


def build_hod(simulation, configuration, inputs):
    from HODDIES import HOD
    hod = HOD(simulation, **deepcopy(configuration))
    # Recursive default merges must not reintroduce priors for an unselected tracer.
    hod.args['fit_param']['priors'] = deepcopy(configuration['fit_param']['priors'])
    hod.initialize_fit(inputs.vector, inputs.covariance, hartlap_factor=inputs.hartlap_deficit)
    if hod._comb_tr_list != inputs.pairs:
        raise ValueError('Model pair ordering differs from saved input/covariance ordering.')
    return hod


def save_inputs(inputs, manifest, output):
    output.mkdir(parents=True, exist_ok=True)
    arrays = dict(data_vector=inputs.vector, covariance=inputs.covariance,
                  covariance_indices=inputs.indices, pair_order=inputs.pairs,
                  statistic_order=inputs.statistics, hartlap_deficit=inputs.hartlap_deficit)
    for stat in inputs.statistics:
        for pair, (x, y) in inputs.observations[stat].items():
            arrays.update({f'{pair}_{stat}_x': x, f'{pair}_{stat}_y': y,
                           f'{pair}_{stat}_sigma': inputs.errors[stat][pair]})
    np.savez(output / 'fit_inputs.npz', **arrays)
    reporting().write_json(output / 'configuration.json', manifest)


def save_best_fit(hod, result, inputs, output, seed, plot=True):
    from HODDIES.fit_functions import flatten_stats
    point = np.asarray(result['x'], dtype=float)
    if point.shape != (len(hod.name_params),) or not np.isfinite(point).all():
        raise ValueError('Optimizer returned invalid parameters.')
    # update_new_param expects one row containing a structured record array.
    structured = point.reshape(1, -1).copy()
    structured.dtype = [(name, 'f8') for name in hod.name_params]
    hod.update_new_param(structured, hod.name_params)
    mock = hod.make_mock_cat(fix_seed=seed, verbose=False)
    model = hod.compute_stats(mock, stat=inputs.statistics, tracers=hod._tracer_to_fit)
    vector = flatten_stats(model, inputs.statistics, inputs.pairs)
    if vector.shape != inputs.vector.shape or not np.isfinite(vector).all():
        raise ValueError('Best-fit model has invalid shape or non-finite values.')
    residual = inputs.vector - vector
    chi2 = float(residual @ hod.inv_cov2 @ residual)
    arrays = dict(model_vector=vector, parameters=point, parameter_names=hod.name_params, chi2=chi2)
    for stat in inputs.statistics:
        for pair in inputs.observations[stat]:
            x, y = model[stat][pair]
            arrays.update({f'{pair}_{stat}_x': x, f'{pair}_{stat}_y': y})
    np.savez(output / 'best_fit_statistics.npz', **arrays)
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        data = {stat: {pair: (*entry, inputs.errors[stat][pair]) for pair, entry in pairs.items()}
                for stat, pairs in inputs.observations.items()}
        fig = hod.plot_stats(mock, stats=inputs.statistics, tracers=hod._tracer_to_fit,
                             data=data, residuals=True, cross=True, by_tracer='rows',
                             show=False, save_fn=str(output / 'best_fit.png'))
        plt.close(fig)
    return chi2


def available_cpu_count():
    count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    try:
        allocated = int(os.environ.get('SLURM_CPUS_PER_TASK', '0'))
    except ValueError:
        allocated = 0
    return min(count, allocated) if allocated > 0 else count


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument('--tracers', '--tracer', nargs='+', choices=['LRG', 'ELG'], default=['LRG'])
    result.add_argument('--statistics', '--stats', nargs='+', choices=['wp', 'xi_ells', 'xi0', 'xi2', 'delta_sigma'], default=['wp', 'xi_ells'])
    result.add_argument('--data-file', type=Path, default=DATA_FILE)
    result.add_argument('--covariance-file', type=Path)
    result.add_argument('--covariance-tracers', nargs='+', choices=['LRG', 'ELG'], help='Source matrix tracers (default: fitted tracers).')
    result.add_argument('--covariance-statistics', nargs='+', choices=['wp', 'xi_ells', 'xi0', 'xi2', 'delta_sigma'], default=None, help='Source matrix statistics in order, within each pair (default: selected families in wp, xi, delta_sigma order; xi includes all saved multipoles).')
    result.add_argument('--nmocks', type=int, default=1786)
    result.add_argument('--no-hartlap', action='store_true', help='Disable the finite-mock precision multiplier; otherwise (nmocks-ndata-2)/(nmocks-1).')
    result.add_argument('--abacus-root', type=Path, default=ABACUS_ROOT)
    result.add_argument('--sim-name', default='AbacusSummit_small_c000_ph3000')
    result.add_argument('--z-simu', type=float, default=.5)
    result.add_argument('--mass-cut', type=float)
    result.add_argument('--nthreads', type=int)
    result.add_argument('--seed', type=int, default=10)
    result.add_argument('--method', choices=['cmaes', 'de', 'pso', 'cpso', 'na', 'vdcma'], default='cmaes')
    result.add_argument('--maxiter', type=int, default=100)
    result.add_argument('--popsize', type=int, help='Default: MPI rank count (minimum 2), or 5 in serial.')
    result.add_argument('--xtol', type=float, default=1.)
    result.add_argument('--ftol', type=float, help='Absolute objective target for CMA-ES; default: Ndata / 2.')
    result.add_argument('--output', type=Path, help='Default: a new directory with a unique run ID under tmp/mock_fits/.')
    result.add_argument('--check-data', action='store_true')
    result.add_argument('--mpi', action='store_true')
    result.add_argument('--no-plot', action='store_true')
    result.add_argument('--report', type=Path, default=REPO_ROOT / 'MOCK_FIT_RESULTS.md')
    result.add_argument('--no-report', action='store_true')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    tracers = tracer_order(args.tracers)
    comm = None
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    rank, ranks = (comm.Get_rank(), comm.Get_size()) if comm is not None else (0, 1)
    started = time.monotonic()
    try:
        args.nthreads = available_cpu_count() if args.nthreads is None else args.nthreads
        args.popsize = max(2, ranks) if args.popsize is None and comm is not None else (args.popsize if args.popsize is not None else 5)
        if args.nthreads < 1 or args.maxiter < 1 or args.popsize < 2 or not np.isfinite(args.xtol) or args.xtol <= 0:
            raise ValueError('Require nthreads/maxiter >= 1, popsize >= 2, and finite xtol > 0.')
        if args.ftol is not None and (not np.isfinite(args.ftol) or args.ftol < 0):
            raise ValueError('Require finite ftol >= 0.')
        inputs = (load_fit_data(args.data_file, tracers, args.statistics, args.covariance_file,
                               args.covariance_tracers, args.covariance_statistics,
                               args.nmocks, args.no_hartlap) if rank == 0 else None)
        if comm is not None:
            inputs = comm.bcast(inputs, root=0)
        args.ftol = inputs.vector.size / 2. if args.ftol is None else args.ftol
        run_id = uuid.uuid4().hex if rank == 0 else None
        if comm is not None:
            run_id = comm.bcast(run_id, root=0)
        output = (args.output or REPO_ROOT / 'tmp/mock_fits' / f'{"_".join(tracers)}_{"+".join(inputs.statistics)}_{run_id}').resolve()
        configuration = model_configuration(inputs, tracers, output, args.nthreads, args.seed)
        options = dict(method=args.method, maxiter=args.maxiter, popsize=args.popsize, xtol=args.xtol, ftol=args.ftol)
        configuration['fit_param']['minimizer'].update(options)
        simulation_parameters = dict(path_to_sim=str(args.abacus_root), sim_name=args.sim_name,
                                     z_simu=args.z_simu, mass_cut=args.mass_cut, nthreads=args.nthreads,
                                     load_particles=bool(configuration.get('use_particles', False)),
                                     load_field_particles='delta_sigma' in inputs.statistics)
        configuration['hcat'] = simulation_parameters
        metadata = dict(workflow=WORKFLOW, configuration=configuration, simulation=simulation_parameters,
                        data_file=str(args.data_file.resolve()), covariance_file=str(inputs.covariance_file),
                        covariance_statistics=inputs.covariance_statistics, covariance_pairs=inputs.covariance_pairs,
                        statistic_order=inputs.statistics, pair_order=inputs.pairs, ndata=inputs.vector.size,
                        nmocks=args.nmocks, precision_multiplier=1. - inputs.hartlap_deficit,
                        run=dict(id=run_id, started_at=reporting().utc_now(), mpi_ranks=ranks,
                                 slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                                 state='Data checked; no optimization' if args.check_data else 'Started; completion not recorded'))
        if rank == 0:
            save_inputs(inputs, metadata, output)
            print(f'Pairs: {inputs.pairs}; statistics: {inputs.statistics}; Ndata={inputs.vector.size}; '
                  f'precision multiplier={1. - inputs.hartlap_deficit:.8g}\nOutput: {output}', flush=True)
        if not args.check_data:
            from HODDIES.sim_loader import AbacusSummitSim, setup_logging
            setup_logging(stream=sys.stdout)
            simulation = AbacusSummitSim(**simulation_parameters)
            hod = build_hod(simulation, configuration, inputs)
            # Start from a reproducible draw within the priors, rather than the saved truth.
            initial = hod.get_init_params(seed=args.seed)
            np.random.seed(args.seed)
            result = hod.run_minimizer(init_params=initial, seed=args.seed, mpi_comm=comm,
                                       return_log10chi2=False, **options)
            if rank == 0:
                chi2 = save_best_fit(hod, result, inputs, output, args.seed, plot=not args.no_plot)
                reporting().save_run_summary(output, result, hod.priors, time.monotonic() - started, plot=not args.no_plot)
                print(f'Best-point chi2={chi2:.7g}; optimizer success={result.get("success")}', flush=True)
        if rank == 0 and not args.no_report:
            try:
                reporting().update_report(args.report, extra_roots=[output])
                print(f'Updated report: {args.report}', flush=True)
            except Exception as error:
                print(f'Artifacts saved, but report update failed: {error}', file=sys.stderr, flush=True)
        if comm is not None:
            comm.Barrier()
        return 0
    except Exception:
        if comm is not None and ranks > 1:
            import traceback
            traceback.print_exc()
            comm.Abort(1)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
