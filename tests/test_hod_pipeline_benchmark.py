"""Benchmark validity, density limits, real samplers, and saved measurements."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'checks_pipeline/benchmark_hod_pipeline.py'
spec = importlib.util.spec_from_file_location('hod_pipeline_benchmark', SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.fixture
def config():
    values = vars(benchmark.parser().parse_args([]))
    values.update(model=deepcopy(benchmark.MODEL), threads=benchmark.THREADS.copy(),
                  sim_names={name: box['sim_name'] for name, box in benchmark.BOXES.items()})
    return values


def test_requested_grid_has_all_boxes_modes_threads_and_densities(config):
    assert config['threads'] == [4, 8, 16, 32, 64, 128, 256]
    assert len(benchmark.grid(config, 'abacus_base')) == 56
    assert len(benchmark.grid(config, 'abacus_highbase')) == 7
    assert len(benchmark.grid(config, 'uchuu')) == 7
    assert sum(len(benchmark.grid(config, box)) * 2 for box in config['boxes']) == 140
    assert min(config['densities']) == 1e-4 and max(config['densities']) == 1e-2


def test_cpu_budget_keeps_initial_task_mask_when_openmp_pins_main_thread(monkeypatch):
    monkeypatch.setattr(benchmark, 'INITIAL_CPU_AFFINITY', set(range(8)))
    monkeypatch.setattr(benchmark.os, 'sched_getaffinity', lambda pid: {0})
    monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
    assert benchmark.cpus_available() == 8
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '4')
    assert benchmark.cpus_available() == 4


def test_openmp_bound_controller_and_workers_keep_the_full_task_budget(tmp_path):
    allowed = sorted(os.sched_getaffinity(0))[:8]
    if len(allowed) < 4:
        pytest.skip('Needs at least four available logical CPUs')
    # A fresh interpreter is essential: the allocation must be captured before
    # NumPy/OpenMP import changes the primary thread's affinity.
    bootstrap = 'import os,runpy,sys; os.sched_setaffinity(0,' + repr(set(allowed)) + '); '
    bootstrap += 'sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name="__main__")'
    env = dict(os.environ, OMP_PLACES='cores', OMP_PROC_BIND='spread', OMP_NUM_THREADS=str(len(allowed)),
               OMP_THREAD_LIMIT=str(len(allowed)), OMP_DYNAMIC='FALSE', NUMBA_NUM_THREADS=str(len(allowed)),
               OPENBLAS_NUM_THREADS='1', SLURM_CPUS_PER_TASK=str(len(allowed)))
    threads = [2, 4] if len(allowed) < 8 else [4, 8]
    run = subprocess.run([sys.executable, '-c', bootstrap, str(SCRIPT), '--smoke-test',
                          '--boxes', 'abacus_base', '--profiles', 'nfw', '--suite', 'fixed',
                          '--threads', *map(str, threads), '--repeats', '1', '--output', str(tmp_path)],
                         env=env, capture_output=True, text=True, timeout=180)
    assert run.returncode == 0, run.stdout + run.stderr + '\n' + (
        (tmp_path / 'logs/abacus_base__nfw.log').read_text() if (tmp_path / 'logs/abacus_base__nfw.log').exists() else '')
    case = json.loads((tmp_path / 'cases/abacus_base__nfw.json').read_text())
    assert case['machine']['available_logical_cpus'] == len(allowed)
    assert len(case['machine']['initial_cpu_affinity']) == len(allowed)
    assert case['state'] == 'completed'
    assert {point['actual_threads'] for point in case['points']} == set(threads)


def test_impossible_density_is_not_accepted_as_a_benchmark_point():
    hod = SimpleNamespace(ngal=lambda tracer: (20000., .1), boxsize=100., args={'LRG': {'Ac': 1.}})
    assert benchmark.density_capacity(hod, 1e-2)['scaled_Ac'] == .5
    with pytest.raises(ValueError, match='infeasible'):
        benchmark.density_capacity(hod, .03)


def test_uchuu_subhalo_ids_are_mapped_to_ultimate_host_ids():
    class Columns(dict):
        def columns(self):
            return list(self)
    sim = SimpleNamespace(part_subsamples=Columns(halo_id=np.array([1001, 1002]), upid=np.array([31, 31])),
                          _order_part_index=np.array([0, 1]))
    benchmark.prepare_uchuu_subhalos(sim)
    np.testing.assert_array_equal(sim.part_subsamples['halo_id'], [31, 31])
    np.testing.assert_array_equal(sim.part_subsamples['subhalo_id'], [1001, 1002])
    assert not hasattr(sim, '_order_part_index')


def test_summary_excludes_failed_points_from_successful_timings():
    point = dict(threads=4, density=7e-4, status='ok', actual_threads=4,
                 samples=[dict(seconds=time, realized_density=7e-4, ngal=700, satellite_fraction=.1)
                          for time in [1., 3., 5.]])
    failed = dict(threads=8, density=7e-4, status='failed', samples=[], reason='density mismatch')
    rows, raw = benchmark.summaries([dict(box='abacus_base', profile='nfw', points=[point, failed])])
    assert rows[0]['median_seconds'] == 3.
    assert rows[0]['q25_seconds'] == 2. and rows[0]['q75_seconds'] == 4.
    assert 'median_seconds' not in rows[1]
    assert len(raw) == 3


@pytest.mark.parametrize('box,profile', [('abacus_base', 'nfw'), ('abacus_base', 'particles'), ('uchuu', 'particles')])
def test_real_hod_and_native_satellite_samplers_on_small_catalogues(tmp_path, monkeypatch, config, box, profile):
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '4')
    config.update(smoke_test=True, synthetic_halos=12000, threads=[2, 4],
                  repeats=2, warmups=1, densities=[1e-4, 1e-2], boxes=[box], profiles=[profile])
    manifest = tmp_path / 'manifest.json'
    benchmark.atomic_json(manifest, dict(configuration=config, machine=benchmark.machine_info()))
    case_id = f'{box}__{profile}'
    assert benchmark.worker(manifest, case_id) == 0
    record = json.loads((tmp_path / 'cases' / f'{case_id}.json').read_text())
    assert record['boxsize'] == 100.  # Explicitly synthetic; no real simulation timings.
    assert record['state'] == 'completed'
    for point in record['points']:
        assert point['status'] == 'ok' and point['actual_threads'] == point['threads']
        assert point['scaled_Ac'] <= 1.
        assert len(point['samples']) == 2
        assert all(s['density_valid'] and s['seconds'] > 0 for s in point['samples'])
        if profile == 'particles':
            assert point['warmup']['requested_satellites'] > point['warmup']['fallback_satellites']
    # Render real measured numbers, CSV files and both plots without sim reload.
    done, planned = benchmark.report(tmp_path, tmp_path / 'REPORT.md')
    assert done == planned == len(benchmark.grid(config, box))
    assert (tmp_path / 'thread_scaling.png').read_bytes().startswith(b'\x89PNG')
    assert (tmp_path / 'density_scaling.png').read_bytes().startswith(b'\x89PNG')
    assert 'SYNTHETIC SMOKE TEST' in (tmp_path / 'REPORT.md').read_text()
    assert 'median_seconds' in (tmp_path / 'summary.csv').read_text()


def test_threads_above_job_allocation_are_skipped(tmp_path, monkeypatch, config):
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '4')
    config.update(smoke_test=True, threads=[2, 64], boxes=['abacus_base'], profiles=['nfw'],
                  suite='fixed', repeats=1)
    manifest = tmp_path / 'manifest.json'
    benchmark.atomic_json(manifest, dict(configuration=config))
    assert benchmark.worker(manifest, 'abacus_base__nfw') == 1
    record = json.loads((tmp_path / 'cases/abacus_base__nfw.json').read_text())
    skipped = next(p for p in record['points'] if p['threads'] == 64)
    assert skipped['status'] == 'skipped' and not skipped['samples']
    assert 'actual_threads' not in skipped


def test_early_worker_failure_is_visible_in_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, 'machine_info', lambda: {'host': 'test'})
    monkeypatch.setattr(benchmark.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=137))
    assert benchmark.main(['--smoke-test', '--boxes', 'abacus_base', '--profiles', 'nfw',
                           '--threads', '2', '--output', str(tmp_path)]) == 1
    case = json.loads((tmp_path / 'cases/abacus_base__nfw.json').read_text())
    assert case['state'] == 'failed'
    assert '137' in (tmp_path / 'REPORT.md').read_text()
    assert (tmp_path / 'summary.csv').read_text().startswith('box,profile,threads,density,status')


def test_resuming_failed_case_does_not_report_old_error_as_current(tmp_path, monkeypatch, config):
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '4')
    config.update(smoke_test=True, threads=[2], suite='fixed', repeats=1)
    manifest = tmp_path / 'manifest.json'
    benchmark.atomic_json(manifest, dict(configuration=config))
    destination = tmp_path / 'cases/abacus_base__nfw.json'
    benchmark.atomic_json(destination, dict(case_id='abacus_base__nfw', box='abacus_base', profile='nfw',
                          state='failed', points=[], error='Previous CPU-affinity error', traceback='old traceback',
                          machine={'available_logical_cpus': 1}))
    assert benchmark.worker(manifest, 'abacus_base__nfw') == 0
    case = json.loads(destination.read_text())
    assert case['state'] == 'completed' and 'error' not in case and 'traceback' not in case
    assert case['previous_failures'][0]['error'] == 'Previous CPU-affinity error'
