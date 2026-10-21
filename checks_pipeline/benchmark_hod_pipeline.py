#!/usr/bin/env python3
"""Benchmark SHOD mock generation on Abacus base/highbase and Uchuu.

Use --check-inputs before submitting run_hod_benchmark.sh. Catalogue loading,
HOD construction, diagnostics and warm-ups are recorded outside the repeated
make_mock_cat timings. Each box/profile runs in a fresh process, loading once.
--smoke-test uses small synthetic catalogues and never labels them real results.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time
import traceback
from types import MethodType
from urllib.parse import quote


def current_cpu_affinity():
    try:
        return set(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return None


# Capture the task mask before NumPy/Numba/OpenMP can pin the primary thread.
INITIAL_CPU_AFFINITY = current_cpu_affinity()


@contextmanager
def initial_cpu_affinity():
    """Give child processes the original task mask, then restore OpenMP binding."""
    previous = current_cpu_affinity()
    changed = INITIAL_CPU_AFFINITY is not None and previous is not None and previous != INITIAL_CPU_AFFINITY
    if changed:
        os.sched_setaffinity(0, INITIAL_CPU_AFFINITY)
    try:
        yield
    finally:
        if changed:
            os.sched_setaffinity(0, previous)


REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
THREADS = [4, 8, 16, 32, 64, 128, 256]
DENSITIES = [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
BOXES = {
    'abacus_base': dict(label='Abacus base 2 Gpc/h', length=2000., sim_name='AbacusSummit_base_c000_ph000'),
    'abacus_highbase': dict(label='Abacus highbase 1 Gpc/h', length=1000., sim_name='AbacusSummit_highbase_c000_ph100'),
    'uchuu': dict(label='Uchuu 2 Gpc/h', length=2000., sim_name='Uchuu2Gpc'),
}
# One SHOD shape for both scans and all boxes. A low cutoff supplies sufficient
# capacity for n=1e-2; this is an LRG-style functional form, not a DESI fit.
MODEL = dict(HOD_model='SHOD', Ac=1., log_Mcent=11., sigma_M=.4,
             satellites=True, sat_HOD_model='Nsat_pow_law', As=1., M_0=11., M_1=13., alpha=1.,
             link_sat_to_central=False, conformity_bias=False, nu=1.,
             f_sigv=1., f_vcen=0., vel_sat='rd_normal', v_infall=0.,
             exp_frac=0., exp_scale=1., nfw_rescale=1., vsmear=None,
             assembly_bias={'c': [0., 0.]})


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        temporary = Path(stream.name)
    temporary.replace(path)


def cpus_available():
    """Initial task CPU budget, independent of OpenMP primary-thread binding."""
    count = len(INITIAL_CPU_AFFINITY) if INITIAL_CPU_AFFINITY is not None else (os.cpu_count() or 1)
    allocation = os.environ.get('SLURM_CPUS_PER_TASK')
    if allocation and allocation.isdigit() and int(allocation) > 0:
        count = min(count, int(allocation))
    return max(1, count)


def machine_info():
    model = platform.processor()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.is_file():
        model = next((line.split(':', 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                      if line.startswith('model name')), model)
    info = dict(host=platform.node(), cpu_model=model, available_logical_cpus=cpus_available(),
                initial_cpu_affinity=sorted(INITIAL_CPU_AFFINITY) if INITIAL_CPU_AFFINITY is not None else None,
                main_thread_affinity=sorted(current_cpu_affinity()) if current_cpu_affinity() is not None else None,
                python=platform.python_version(), platform=platform.platform(),
                slurm_job=os.environ.get('SLURM_JOB_ID'),
                slurm_cpus_per_task=os.environ.get('SLURM_CPUS_PER_TASK'))
    for key in ['OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OMP_DYNAMIC', 'NUMBA_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        info[key] = os.environ.get(key)
    try:
        info['git_revision'] = subprocess.check_output(
            ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        info['git_dirty'] = bool(subprocess.check_output(
            ['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return info


def grid(config, box):
    densities = []
    if config['suite'] != 'density':
        densities.append(config['fixed_density'])
    if config['suite'] != 'fixed' and box == 'abacus_base':
        densities += config['densities']
    return [(threads, density) for threads in config['threads'] for density in sorted(set(densities))]


def input_path(config, box):
    if box == 'uchuu':
        return Path(config['uchuu_root']) / 'Uchuu_halo_catalogs' / f'halodir_{config["uchuu_snapshot"]:03d}'
    return Path(config['abacus_root']) / config['sim_names'][box] / 'halos' / f'z{config["abacus_z"]:.3f}'


def synthetic_catalogue(box, particles, nthreads, config):
    """Exercise the actual HOD and both native satellite samplers on small data."""
    import numpy as np
    from HODDIES.sim_loader import Base_catalogue, AbacusSummitSim

    class SyntheticCatalogue(Base_catalogue):
        def cosmo(self, **kwargs):
            return None

    rng = np.random.default_rng(config['seed'])
    n, length, per_host = config['synthetic_halos'], 100., 8
    data = {axis: rng.uniform(0., length, n).astype('f4') for axis in 'xyz'}
    data.update({f'v{axis}': rng.normal(0., 150., n).astype('f4') for axis in 'xyz'})
    data.update(Mh=10**rng.uniform(10.8, 14., n), Rh=np.full(n, 350., dtype='f4'),
                c=np.full(n, 5., dtype='f4'), Vrms=np.full(n, 200., dtype='f4'),
                halo_id=np.arange(n, dtype='i8'))
    parts = None
    if particles:
        parts = {axis: (np.repeat(data[axis], per_host) + rng.normal(0., .1, n * per_host)).astype('f4')
                 for axis in 'xyz'}
        parts.update({f'v{axis}': (np.repeat(data[f'v{axis}'], per_host)
                                  + rng.normal(0., 100., n * per_host)).astype('f4') for axis in 'xyz'})
        parts['halo_id'] = np.repeat(data['halo_id'], per_host)
    sim = SyntheticCatalogue(data, particle_data=parts, boxsize=length, z_simu=.5,
                             nthreads=nthreads, setup_logger=False)
    if particles and box != 'uchuu':
        sim.hcat['npoutA'] = np.full(n, per_host, dtype='i8')
        sim.hcat['npstartA'] = np.arange(n, dtype='i8') * per_host
        sim.part_subsamples = dict(pos=np.column_stack([parts[a] for a in 'xyz']),
                                  vel=np.column_stack([parts[f'v{a}'] for a in 'xyz']))
        sim.assign_sat_to_part = MethodType(AbacusSummitSim.assign_sat_to_part, sim)
    return sim


def load_catalogue(config, box, profile, threads):
    from HODDIES.sim_loader import AbacusSummitSim, UchuuSim
    particles = profile == 'particles'
    if config['smoke_test']:
        return synthetic_catalogue(box, particles, threads, config)
    if box == 'uchuu':
        # The distributed files use HDF5 compression filters.
        import hdf5plugin  # noqa: F401
        from HODDIES.sim_loader.Uchuu_io import get_Uchuu_snapshot_file
        if get_Uchuu_snapshot_file(float(config['uchuu_z'])) != config['uchuu_snapshot']:
            raise ValueError('uchuu-snapshot must match the native snapshot index for uchuu-z.')
        sim = UchuuSim(path_to_sim=config['uchuu_root'], sim_name='Uchuu2Gpc',
                       z_simu=config['uchuu_z'], mass_cut=None, load_particles=particles, nthreads=threads)
        if particles:
            prepare_uchuu_subhalos(sim)
        # Apply an optional HOST mass cut after loading so smaller subhalos are
        # not accidentally discarded by the generic reader's mass_cut option.
        if config['mass_cut'] is not None:
            sim.hcat = sim.hcat[sim.hcat['log10_Mh'] > config['mass_cut']]
        sim.check_hcat_cols()
        sim.check_hcat_args()
    else:
        sim = AbacusSummitSim(path_to_sim=config['abacus_root'], sim_name=config['sim_names'][box],
                              z_simu=config['abacus_z'], mass_cut=config['mass_cut'],
                              load_particles=particles, load_field_particles=False, nthreads=threads)
    if not math.isclose(float(sim.boxsize), BOXES[box]['length']):
        raise ValueError(f'{box}: expected box size {BOXES[box]["length"]}, got {sim.boxsize}')
    if particles and sim.part_subsamples is None:
        raise ValueError('Requested particle/subhalo sampling, but the loader returned no auxiliary catalogue.')
    return sim


def prepare_uchuu_subhalos(sim):
    """The preprocessed halo_id is the subhalo's own ID; the sampler needs upid."""
    if sim.part_subsamples is None or 'upid' not in sim.part_subsamples.columns():
        raise ValueError('Uchuu subhalos require the ultimate host-ID column upid.')
    sim.part_subsamples['subhalo_id'] = sim.part_subsamples['halo_id']
    sim.part_subsamples['halo_id'] = sim.part_subsamples['upid']
    if hasattr(sim, '_order_part_index'):
        del sim._order_part_index


def density_capacity(hod, target, unscaled_ngal=None):
    """The current normalizer refuses Ac * scale > 1; never time that failure as success."""
    count = float(hod.ngal('LRG')[0]) if unscaled_ngal is None else float(unscaled_ngal)
    if count <= 0:
        raise ValueError('The configured SHOD has zero expected galaxies.')
    scale = float(target * hod.boxsize**3 / count)
    central_amplitude = float(hod.args['LRG']['Ac'] * scale)
    if central_amplitude > 1.:
        raise ValueError(f'Target density {target:g} is infeasible: scaled Ac={central_amplitude:g} > 1. '
                         'Lower log_Mcent/M_0, increase satellite capacity, or lower the halo mass cut.')
    return dict(unscaled_density=count / hod.boxsize**3, density_scale=scale,
                scaled_Ac=central_amplitude)


def mock_counts(mock, length):
    import numpy as np
    total = int(mock.size)
    satellites = int(np.count_nonzero(mock['Central'] == 0))
    return dict(ngal=total, nsat=satellites, realized_density=total / length**3,
                satellite_fraction=satellites / total if total else 0.)


def warm_up(hod, sim, config, seed):
    """Measure fallback diagnostics outside the timed repetitions."""
    import numpy as np
    diagnostics = dict(requested_satellites=0, fallback_satellites=0)
    original = sim.assign_sat_to_part

    def inspected(*args, **kwargs):
        values = original(*args, **kwargs)
        diagnostics['requested_satellites'] += int(len(values[-1]))
        diagnostics['fallback_satellites'] += int(np.count_nonzero(values[-1]))
        return values

    if hod.args['use_particles']:
        sim.assign_sat_to_part = inspected
    start = time.perf_counter()
    try:
        for index in range(config['warmups']):
            mock = hod.make_mock_cat(fix_seed=seed + index, verbose=False)
            if config['include_clustering']:
                hod.compute_stats(mock, stat=['wp', 'xi_ells'], tracers=['LRG'], verbose=False)
            del mock
    finally:
        sim.assign_sat_to_part = original
    diagnostics['seconds'] = time.perf_counter() - start
    requested = diagnostics['requested_satellites']
    diagnostics['fallback_fraction'] = diagnostics['fallback_satellites'] / requested if requested else None
    if hod.args['use_particles'] and (not requested or diagnostics['fallback_satellites'] == requested):
        raise ValueError('No satellites were actually placed on particles/subhalos during warm-up.')
    return diagnostics


def worker(manifest_path, case_id):
    import logging
    import numpy as np
    import numba
    sys.path.insert(0, str(REPO))
    from HODDIES import HOD
    from HODDIES.sim_loader import setup_logging
    setup_logging(level=logging.ERROR)
    manifest = json.loads(Path(manifest_path).read_text())
    config, output = manifest['configuration'], Path(manifest_path).parent
    box, profile = case_id.split('__')
    destination = output / 'cases' / f'{case_id}.json'
    record = (json.loads(destination.read_text()) if destination.is_file() else
              dict(case_id=case_id, box=box, profile=profile, points=[]))
    if record.get('error'):
        record.setdefault('previous_failures', []).append(
            dict(error=record.pop('error'), traceback=record.pop('traceback', None),
                 machine=record.get('machine')))
    record.update(state='started; completion not recorded', machine=machine_info(), numpy=np.__version__, numba=numba.__version__)
    atomic_json(destination, record)
    supported = [n for n in config['threads'] if n <= cpus_available() and n <= numba.config.NUMBA_NUM_THREADS]
    pending = [(n, d) for n, d in grid(config, box)
               if not any(p['threads'] == n and p['density'] == d and p['status'] == 'ok' for p in record['points'])]
    print(f'{case_id}: initial CPU budget={cpus_available()}, '
          f'Numba capacity={numba.config.NUMBA_NUM_THREADS}, supported threads={supported}', flush=True)
    if not pending:
        record['state'] = 'completed'
        atomic_json(destination, record)
        return 0
    try:
        if not supported:
            raise ValueError(f'No requested count in {config["threads"]} fits the initial CPU budget '
                             f'({cpus_available()}) and Numba capacity ({numba.config.NUMBA_NUM_THREADS}). '
                             'Use srun --nodes=1 --ntasks=1 --cpus-per-task=256 --cpu-bind=cores.')
        load_threads = min(config['load_threads'], max(supported))
        numba.set_num_threads(load_threads)
        before = time.perf_counter()
        # Uchuu's multiprocessing readers need the full task mask as well.
        with initial_cpu_affinity():
            sim = load_catalogue(config, box, profile, load_threads)
        record['load_seconds'] = time.perf_counter() - before
        record['halo_count'] = int(sim.hcat.size)
        record['boxsize'] = float(sim.boxsize)
        record['snapshot_z'] = float(sim.z_simu)
        if sim.part_subsamples is not None:
            first = 'pos' if isinstance(sim.part_subsamples, dict) and 'pos' in sim.part_subsamples else 'x'
            record['auxiliary_count'] = int(len(sim.part_subsamples[first]))
        else:
            record['auxiliary_count'] = 0
        atomic_json(destination, record)
        for threads, density in pending:
            # Replace previous skipped/failed attempts, retaining completed points.
            record['points'] = [p for p in record['points'] if (p['threads'], p['density']) != (threads, density)]
            point = dict(threads=threads, density=density, samples=[], status='pending',
                         hardware={key: record['machine'][key] for key in
                                   ['host', 'cpu_model', 'available_logical_cpus']})
            record['points'].append(point)
            if threads not in supported:
                point.update(status='skipped', reason='Insufficient allocated CPUs or Numba thread capacity')
                atomic_json(destination, record)
                continue
            try:
                numba.set_num_threads(threads)
                sim.nthreads = threads
                params = deepcopy(config['model'])
                params['density'] = float(density)
                before = time.perf_counter()
                hod = HOD(sim, tracers=['LRG'], LRG=params, use_particles=profile == 'particles',
                          use_assembly_bias=False, nthreads=threads, seed=config['seed'],
                          setup_logger=False, clustering_settings={'rsd': not config['smoke_test']})
                point['initialization_seconds'] = time.perf_counter() - before
                if hod.nthreads != threads or numba.get_num_threads() != threads:
                    raise ValueError(f'Requested {threads} threads, but actual HOD/Numba thread count differs')
                point['actual_threads'] = int(hod.nthreads)
                before = time.perf_counter()
                point.update(density_capacity(hod, density))
                point['capacity_check_seconds'] = time.perf_counter() - before
                point['warmup'] = warm_up(hod, sim, config, config['seed'])
                for repeat in range(config['repeats']):
                    gc.collect()
                    seed = config['seed'] + config['warmups'] + repeat
                    start = time.perf_counter()
                    mock = hod.make_mock_cat(fix_seed=seed, verbose=False)
                    elapsed = time.perf_counter() - start
                    sample = dict(repeat=repeat, seed=seed, seconds=elapsed,
                                  **mock_counts(mock, sim.boxsize))
                    if config['include_clustering']:
                        start = time.perf_counter()
                        stats = hod.compute_stats(mock, stat=['wp', 'xi_ells'], tracers=['LRG'], verbose=False)
                        sample['clustering_seconds'] = time.perf_counter() - start
                        del stats
                    # Count validation and catalogue destruction are outside the clock.
                    tolerance = max(config['density_tolerance'], 5. / math.sqrt(density * sim.boxsize**3))
                    sample['density_relative_error'] = abs(sample['realized_density'] / density - 1.)
                    sample['density_valid'] = sample['density_relative_error'] <= tolerance
                    point['samples'].append(sample)
                    del mock
                    atomic_json(destination, record)
                if not all(s['density_valid'] for s in point['samples']):
                    raise ValueError('Realized number density exceeds the configured/5-sigma sampling tolerance')
                point['status'] = 'ok'
                print(f'{case_id}: threads={threads}, density={density:g}, '
                      f'median={np.median([s["seconds"] for s in point["samples"]]):.4g}s', flush=True)
                del hod
            except Exception as error:
                point.update(status='failed', reason=str(error), traceback=traceback.format_exc())
                print(point['traceback'], file=sys.stderr, flush=True)
            point['worker_peak_rss_gib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
            atomic_json(destination, record)
        record['state'] = 'completed' if all(p['status'] == 'ok' for p in record['points']) else 'partial'
    except Exception as error:
        record.update(state='failed', error=str(error), traceback=traceback.format_exc())
        print(record['traceback'], file=sys.stderr, flush=True)
    atomic_json(destination, record)
    return 0 if record['state'] == 'completed' else 1


def summaries(cases):
    import numpy as np
    rows, raw = [], []
    for case in cases:
        for point in case.get('points', []):
            base = dict(box=case['box'], profile=case['profile'], threads=point['threads'],
                        density=point['density'], status=point['status'],
                        host=point.get('hardware', {}).get('host'), cpu_model=point.get('hardware', {}).get('cpu_model'))
            raw += [dict(base, **sample) for sample in point['samples']]
            row = dict(base, reason=point.get('reason', ''), actual_threads=point.get('actual_threads'),
                       fallback_fraction=point.get('warmup', {}).get('fallback_fraction'))
            if point['status'] == 'ok':
                times = [sample['seconds'] for sample in point['samples']]
                row.update(median_seconds=float(np.median(times)), q25_seconds=float(np.percentile(times, 25)),
                           q75_seconds=float(np.percentile(times, 75)), repeats=len(times),
                           mean_realized_density=float(np.mean([s['realized_density'] for s in point['samples']])),
                           mean_ngal=float(np.mean([s['ngal'] for s in point['samples']])),
                           satellite_fraction=float(np.mean([s['satellite_fraction'] for s in point['samples']])))
                if 'clustering_seconds' in point['samples'][0]:
                    row['clustering_median_seconds'] = float(np.median([s['clustering_seconds'] for s in point['samples']]))
            rows.append(row)
    return rows, raw


def write_csv(path, rows):
    columns = list(dict.fromkeys(['box', 'profile', 'threads', 'density', 'status']
                                + [key for row in rows for key in row]))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plots(output, config, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    good = [row for row in rows if row['status'] == 'ok']
    colors = dict(abacus_base='tab:blue', abacus_highbase='tab:red', uchuu='tab:green')
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    for box in config['boxes']:
        for profile in config['profiles']:
            data = sorted([r for r in good if r['box'] == box and r['profile'] == profile
                           and math.isclose(r['density'], config['fixed_density'])], key=lambda r: r['threads'])
            if not data:
                continue
            median = np.array([r['median_seconds'] for r in data])
            err = [median - [r['q25_seconds'] for r in data], [r['q75_seconds'] for r in data] - median]
            label = BOXES[box]['label'] + ('; NFW' if profile == 'nfw' else '; ' + ('subhalos' if box == 'uchuu' else 'particles'))
            ax.errorbar([r['threads'] for r in data], median, yerr=err, color=colors[box],
                        marker='d' if profile == 'nfw' else 'o', ls='--', capsize=3, label=label)
    if min(config['threads']) <= 32 <= max(config['threads']):
        ax.axvline(32, color='k', ls='--', lw=.8, alpha=.5)
    ax.set(xscale='log', yscale='log', xlabel='Number of threads (logical CPUs)',
           ylabel='Mock generation time [s]', title=f'SHOD; target density = {config["fixed_density"]:g} (h/Mpc)³')
    ax.set_xticks(config['threads'], [str(n) for n in config['threads']])
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=8)
    else:
        ax.text(.5, .5, 'No completed timings yet', transform=ax.transAxes, ha='center')
    if config['smoke_test']:
        fig.suptitle('SYNTHETIC SMOKE TEST: not simulation performance')
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / 'thread_scaling.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(config['profiles']), figsize=(7 * len(config['profiles']), 5), squeeze=False)
    for ax, profile in zip(axes[0], config['profiles']):
        for i, threads in enumerate(config['threads']):
            data = sorted([r for r in good if r['box'] == 'abacus_base' and r['profile'] == profile
                           and r['threads'] == threads and r['density'] in config['densities']], key=lambda r: r['density'])
            if not data:
                continue
            median = np.array([r['median_seconds'] for r in data])
            err = [median - [r['q25_seconds'] for r in data], [r['q75_seconds'] for r in data] - median]
            ax.errorbar([r['density'] for r in data], median, yerr=err, capsize=3, marker='d', ls='--',
                        color=plt.get_cmap('viridis')(.05 + .8 * i / max(1, len(config['threads']) - 1)), label=f'{threads} threads')
        ax.set(xscale='log', yscale='log', xlabel='Target galaxy density [(h/Mpc)³]',
               ylabel='Mock generation time [s]', title=f'Abacus base; {profile}')
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        else:
            ax.text(.5, .5, 'No completed timings yet', transform=ax.transAxes, ha='center')
        ax.grid(alpha=.2)
    if config['smoke_test']:
        fig.suptitle('SYNTHETIC SMOKE TEST: not simulation performance')
    fig.tight_layout()
    fig.savefig(output / 'density_scaling.png', dpi=170)
    plt.close(fig)


def table(headers, rows):
    def cell(value):
        if value is None:
            return '—'
        if isinstance(value, float):
            return f'{value:.6g}'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(cell, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows]])


def report(output, destination):
    output, destination = Path(output).resolve(), Path(destination).resolve()
    manifest = json.loads((output / 'manifest.json').read_text())
    config = manifest['configuration']
    cases = [json.loads(path.read_text()) for path in sorted((output / 'cases').glob('*.json'))]
    rows, raw = summaries(cases)
    write_csv(output / 'summary.csv', rows)
    write_csv(output / 'timings.csv', raw)
    plots(output, config, rows)
    def relative(path):
        return quote(os.path.relpath(path, destination.parent), safe='/._-')
    done = sum(row['status'] == 'ok' for row in rows)
    planned = sum(len(grid(config, box)) * len(config['profiles']) for box in config['boxes'])
    lines = ['# HOD pipeline benchmark', '',
             '**SYNTHETIC SMOKE TEST. These numbers are not real simulation timings.**' if config['smoke_test']
             else ('**Measured catalogue generation timings; unmeasured cases are explicitly pending.**' if done
                   else '**No successful real-simulation timings are available yet.**'), '',
             f'Completed points: **{done}/{planned}**. Output directory: `{output}`.', '',
             '## Measurement definition', '',
             '- Timed operation: `HOD.make_mock_cat(fix_seed=..., verbose=False)`, including density normalization, '
             'central/satellite draws, positions, velocities and catalogue construction.',
             '- Catalogue loading, HOD construction, capacity checks, warm-ups/JIT, mock counting, garbage collection '
             'and catalogue deletion are outside the repeated generation timings. Warm-ups also initialize the '
             'Uchuu subhalo index. Per-call subhalo mapping inside the sampler remains timed.',
             f'- {config["warmups"]} warm-up call(s) per point; {config["repeats"]} timed draws. '
             'Plots show medians with the 25th–75th percentile range. Seeds repeat across comparable points; '
             'different thread counts need not give identical mock catalogues.',
             '- Thread counts are logical CPUs. Both the Numba mask and HOD/loader thread counts are set and checked. '
             'Points exceeding the job CPU allocation or Numba capacity are skipped, never silently clamped.',
             '- CPU eligibility is captured before importing threaded libraries. OpenMP can pin its primary '
             'thread to one core; that later mask is not treated as the job CPU budget. Reader/benchmark '
             'child processes inherit the original task mask.',
             '- One SHOD shape is used across boxes, profiles and densities. Its low mass cutoff is chosen '
             'to support 1e-2; capacity is validated per catalogue. This is an LRG-style SHOD performance model, '
             'not a calibrated DESI LRG HOD. '
             'Only occupation amplitudes are density-normalized. Assembly bias and conformity are off.',
             '- A point fails if the scaled central amplitude exceeds one or its realized density fails the '
             f'max({config["density_tolerance"]:.1%}, 5/sqrt(expected Ngal)) tolerance. '
             'Particle/subhalo modes must place at least some satellites on actual auxiliary objects. '
             'The warm-up NFW fallback fraction is reported; it is not assumed to be zero.',
             '- Uchuu uses the native loader. Its subhalos are matched to hosts using `upid`, retaining their '
             'original IDs as `subhalo_id`; this mapping is outside the timed calls. Compressed HDF5 filters '
             'are registered with hdf5plugin before loading.',
             '- Separate fresh processes load each box/profile once, so catalogues do not coexist in memory. '
             'Worker logs and partial JSON records survive interruptions.',
             '- Abacus and Uchuu differ in halo finder, resolution and cosmology. Snapshots are explicitly recorded; '
             'the default Abacus z=0.5 and Uchuu z=0.49 are close but not identical.',
             '- Clustering ' + ('is additionally timed and stored separately; it is excluded from the two mock-generation plots.'
                                if config['include_clustering'] else 'is not part of this benchmark.'), '',
             '## Plots', '', f'![Time versus threads]({relative(output / "thread_scaling.png")})', '',
             f'![Time versus target density, Abacus base]({relative(output / "density_scaling.png")})', '',
             '## Saved numbers and provenance', '',
             f'- [Individual timed draws]({relative(output / "timings.csv")})',
             f'- [Aggregated timings and failed/skipped points]({relative(output / "summary.csv")})',
             f'- [Configuration and hardware]({relative(output / "manifest.json")})', '',
             'Case JSON files include loading/initialization/warm-up times, realized density, satellite fraction, '
             'fallback diagnostics, per-point hardware and worker peak RSS (Linux ru_maxrss, including loading and JIT).', '',
             'Driver environment at manifest creation is listed below; execution hardware for each measured '
             'point is stored in CSV and case JSON files.', '',
             table(['Hardware / software setting', 'Value'], manifest['machine'].items()), '',
             '## Catalogue setup', '',
             table(['Box', 'Path', 'Simulation', 'Snapshot z', 'Expected size [Mpc/h]'],
                   [[BOXES[box]['label'], input_path(config, box), config['sim_names'][box],
                     config['uchuu_z'] if box == 'uchuu' else config['abacus_z'], BOXES[box]['length']]
                    for box in config['boxes']]), '',
             '## SHOD input parameters', '', table(['Parameter', 'Value'], config['model'].items()), '',
             f'Optional host mass cut: `{config["mass_cut"]}` in log10(Msun/h). '
             'Uchuu subhalos are not removed by this host selection.', '',
             '## Fixed density: timings', '',
             table(['Box', 'Profile', 'Threads', 'Median [s]', 'Q25 [s]', 'Q75 [s]', 'Mean density', 'Warm-up NFW fallback', 'State'],
                   [[r['box'], r['profile'], r['threads'], r.get('median_seconds'), r.get('q25_seconds'),
                     r.get('q75_seconds'), r.get('mean_realized_density'), r.get('fallback_fraction'), r['status']]
                    for r in rows if math.isclose(r['density'], config['fixed_density'])]), '']
    for profile in config['profiles']:
        lines += [f'## Abacus base density scan: {profile}', '',
                  table(['Density [(h/Mpc)³]'] + [f'{n} threads [s]' for n in config['threads']],
                        [[density] + [next((r.get('median_seconds', r['status']) for r in rows
                          if r['box'] == 'abacus_base' and r['profile'] == profile
                          and r['threads'] == threads and r['density'] == density), 'pending')
                          for threads in config['threads']] for density in config['densities']]), '']
    lines += ['## Cases, preparation and issues', '']
    for case in cases:
        lines += [f'### {case["case_id"]}', '',
                  table(['Setting', 'Value'], [[key, case.get(key)] for key in
                        ['state', 'boxsize', 'snapshot_z', 'halo_count', 'auxiliary_count', 'load_seconds']]), '',
                  f'[Worker log]({relative(output / "logs" / (case["case_id"] + ".log"))}); '
                  f'[Case measurements]({relative(output / "cases" / (case["case_id"] + ".json"))}).', '']
        if case.get('error'):
            lines += [f'Case error: {case["error"]}', '']
        for point in case['points']:
            if point['status'] != 'ok':
                lines += [f'- {point["threads"]} threads, density={point["density"]:g}: '
                          f'{point["status"]}: {point.get("reason", "interrupted before completion")}']
        lines += ['']
    lines += ['## Run / refresh', '', 'From the repository root in the cosmodesi environment:', '',
              '```bash', 'python checks_pipeline/benchmark_hod_pipeline.py --check-inputs',
              'sbatch checks_pipeline/run_hod_benchmark.sh',
              f'python checks_pipeline/benchmark_hod_pipeline.py --report-only --output {output}',
              f'python checks_pipeline/benchmark_hod_pipeline.py --resume --output {output}', '```', '',
              'A smoke test can be run with `--smoke-test --boxes abacus_base --threads 4 8 --repeats 2`. '
              'Its report is saved in its own output directory. Use a compute allocation for real catalogues.', '']
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(lines))
    return done, planned


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--boxes', nargs='+', choices=list(BOXES), default=list(BOXES))
    p.add_argument('--profiles', nargs='+', choices=['nfw', 'particles'], default=['nfw', 'particles'])
    p.add_argument('--threads', nargs='+', type=int,
                   help='Default: 4 8 16 32 64 128 256; smoke-test default: 4 8.')
    p.add_argument('--fixed-density', type=float, default=7e-4)
    p.add_argument('--densities', nargs='+', type=float, default=DENSITIES)
    p.add_argument('--suite', choices=['all', 'fixed', 'density'], default='all')
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--warmups', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--density-tolerance', type=float, default=.05)
    p.add_argument('--include-clustering', action='store_true', help='Time wp+xi0+xi2 separately after each mock.')
    p.add_argument('--abacus-root', default='/global/cfs/cdirs/desi/cosmosim/Abacus')
    p.add_argument('--base-sim', default=BOXES['abacus_base']['sim_name'])
    p.add_argument('--highbase-sim', default=BOXES['abacus_highbase']['sim_name'])
    p.add_argument('--abacus-z', type=float, default=.5)
    p.add_argument('--uchuu-root', default='/global/homes/a/arocher/users_arocher/HODDIES_data/Uchuu')
    p.add_argument('--uchuu-z', type=float, default=.49)
    p.add_argument('--uchuu-snapshot', type=int, default=40, help='Preflight directory only; must match native redshift mapping.')
    p.add_argument('--mass-cut', type=float, default=None)
    p.add_argument('--model', type=Path, help='JSON overrides for the shared SHOD parameters.')
    p.add_argument('--load-threads', type=int, default=32)
    p.add_argument('--output', type=Path)
    p.add_argument('--report', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--check-inputs', action='store_true', help='Check catalogue directories without loading them.')
    p.add_argument('--plan-only', action='store_true', help='Write a manifest/report with pending points; do not load catalogues.')
    p.add_argument('--report-only', action='store_true')
    p.add_argument('--smoke-test', action='store_true')
    p.add_argument('--synthetic-halos', type=int, default=12000)
    p.add_argument('--worker', help=argparse.SUPPRESS)
    p.add_argument('--manifest', type=Path, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.worker:
        return worker(args.manifest, args.worker)
    if args.report_only:
        if args.output is None:
            p.error('--report-only requires --output')
        manifest = json.loads((args.output / 'manifest.json').read_text())
        report(args.output, args.report or manifest['report'])
        return 0
    if args.threads is None:
        args.threads = [4, 8] if args.smoke_test else THREADS.copy()
    config = {key: value for key, value in vars(args).items() if key not in
              ['output', 'report', 'resume', 'check_inputs', 'plan_only', 'report_only', 'worker', 'manifest', 'model', 'base_sim', 'highbase_sim']}
    if (not args.threads or min(args.threads) < 1 or args.repeats < 1 or args.warmups < 1
            or args.load_threads < 1 or args.synthetic_halos < 1000):
        p.error('Require positive thread/repeat/warm-up counts and synthetic-halos >= 1000.')
    if not all(math.isfinite(d) and d > 0 for d in [args.fixed_density, *args.densities, args.density_tolerance]):
        p.error('Densities and density tolerance must be positive and finite.')
    if args.suite == 'density':
        config['boxes'] = ['abacus_base']
    config['boxes'] = list(dict.fromkeys(config['boxes']))
    config['profiles'] = list(dict.fromkeys(config['profiles']))
    config['threads'] = sorted(set(args.threads))
    config['densities'] = sorted(set(args.densities))
    config['sim_names'] = dict(abacus_base=args.base_sim, abacus_highbase=args.highbase_sim, uchuu='Uchuu2Gpc')
    config['model'] = deepcopy(MODEL)
    if args.model:
        config['model'].update(json.loads(args.model.read_text()))
    if (config['model']['HOD_model'] != 'SHOD' or config['model']['link_sat_to_central']
            or config['model']['conformity_bias']):
        p.error('Use SHOD with link_sat_to_central=False and conformity_bias=False for linear density normalization.')
    if args.check_inputs:
        missing = False
        for box in config['boxes']:
            path = input_path(config, box)
            available = bool(list(path.glob('*.h5'))) if box == 'uchuu' else bool(list((path / 'halo_info').glob('halo_info_*.asdf')))
            print(f'{box}: {"OK" if available else "MISSING"}: {path}')
            missing |= not available
        return int(missing)
    if not args.plan_only and not args.smoke_test and not os.environ.get('SLURM_JOB_ID'):
        p.error('Run real catalogues in a Slurm compute allocation. Use --plan-only or --smoke-test on a login node.')
    if not args.plan_only and not any(n <= cpus_available() for n in config['threads']):
        p.error(f'Requested threads {config["threads"]} exceed the initial CPU budget ({cpus_available()}). '
                'Use srun --ntasks=1 --cpus-per-task=256 --cpu-bind=cores; '
                'setting a shell variable alone does not allocate CPUs.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = (args.output or REPO / 'checks_pipeline/results/hod_benchmark' / (stamp + ('_synthetic' if args.smoke_test else ''))).resolve()
    destination = (args.report or (output / 'REPORT.md' if args.smoke_test else REPO / 'checks_pipeline/HOD_PIPELINE_BENCHMARK.md')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for folder in ['cases', 'logs']:
        (output / folder).mkdir(exist_ok=True)
    filename = output / 'manifest.json'
    if filename.is_file():
        old = json.loads(filename.read_text())
        if not args.resume or old['configuration'] != config:
            p.error('Output already exists: use --resume with the same configuration, or choose a new --output.')
    else:
        atomic_json(filename, dict(configuration=config, machine=machine_info(), report=str(destination),
                                   created_at=datetime.now(timezone.utc).isoformat(),
                                   configuration_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()))
    report(output, destination)
    if args.plan_only:
        print(f'Plan written to {destination}; no catalogues loaded.')
        return 0
    failures = 0
    for box in config['boxes']:
        for profile in config['profiles']:
            case_id = f'{box}__{profile}'
            print(f'Running {case_id}; log: {output / "logs" / (case_id + ".log")}', flush=True)
            env = dict(os.environ, NUMBA_NUM_THREADS=str(min(max(config['threads']), cpus_available())),
                       OMP_NUM_THREADS=str(min(max(config['threads']), cpus_available())),
                       OMP_THREAD_LIMIT=str(min(max(config['threads']), cpus_available())), OMP_DYNAMIC='FALSE',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
            with (output / 'logs' / f'{case_id}.log').open('a' if args.resume else 'w') as log:
                # Reporting can pin the controller's primary thread through
                # OpenMP. Fresh workers must inherit the original task mask.
                with initial_cpu_affinity():
                    completed = subprocess.run([sys.executable, str(SCRIPT), '--worker', case_id, '--manifest', str(filename)],
                                               env=env, stdout=log, stderr=subprocess.STDOUT)
            failures += completed.returncode != 0
            if completed.returncode:
                case_path = output / 'cases' / f'{case_id}.json'
                case = (json.loads(case_path.read_text()) if case_path.is_file() else
                        dict(case_id=case_id, box=box, profile=profile, points=[]))
                if case.get('state') not in ['partial', 'failed']:
                    case.update(state='interrupted' if case['points'] else 'failed',
                                error=f'Worker exited with code {completed.returncode}; see worker log.')
                atomic_json(case_path, case)
            report(output, destination)
    done, planned = report(output, destination)
    print(f'Finished {done}/{planned} points. Report: {destination}', flush=True)
    return int(failures > 0)


if __name__ == '__main__':
    raise SystemExit(main())
