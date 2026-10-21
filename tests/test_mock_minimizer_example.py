"""Covariance subsets, current-pipeline fits, and independent mock-run history."""
import json
from pathlib import Path

import numpy as np
import pytest

from HODDIES.fit_functions.scripts import run_mock_minimizer as example
from HODDIES.fit_functions.scripts import summarize_mock_fits as reporting


@pytest.fixture(autouse=True)
def isolate_reporting(monkeypatch):
    monkeypatch.setattr(example, 'reporting', lambda: reporting)
    monkeypatch.setattr(reporting, 'DEFAULT_ROOTS', [])


@pytest.fixture
def mock_inputs(tmp_path):
    rp_edges = np.geomspace(.3, 8., 4)
    s_edges = np.geomspace(.3, 8., 5)
    rp = np.sqrt(rp_edges[:-1] * rp_edges[1:])
    s = np.sqrt(s_edges[:-1] * s_edges[1:])
    params = dict(tracers=['LRG', 'ELG'], use_assembly_bias=False, use_particles=False,
                  clustering_settings=dict(rsd=False, los='z',
                      xi_rppi=dict(pimax=4, edges_rppi=[rp_edges, np.arange(-4., 5.)]),
                      xi_smu=dict(multipole_index=[0, 2], edges_smu=[s_edges, np.linspace(-1., 1., 21)])),
                  fit_param=dict(priors={tr: dict(M_0=[12., 13.]) for tr in ['LRG', 'ELG']}))
    for tr, model in [('LRG', 'SHOD'), ('ELG', 'mHMQ')]:
        params[tr] = dict(HOD_model=model, M_0=12.5, M_1=13., log_Mcent=12.5,
                          sigma_M=.3, alpha=1., Ac=.5, As=.5, f_sigv=1., gamma=3.,
                          density=.001, satellites=True, sat_HOD_model='Nsat_pow_law',
                          assembly_bias={'c': [0., 0.]})
    data = {st: {} for st in ['wp', 'xi_ells', 'delta_sigma']}
    offset = 0
    for pair in example.pair_order(['LRG', 'ELG']):
        for st, x, shape in [('wp', rp, (3,)), ('xi_ells', s, (2, 4)), ('delta_sigma', rp, (3,))]:
            if st == 'delta_sigma' and pair == 'ELG_LRG':
                continue
            n = int(np.prod(shape))
            data[st][pair] = (x, np.arange(offset, offset + n, dtype=float).reshape(shape))
            offset += n
    path = tmp_path / 'data.npy'
    np.save(path, dict(param=params, data=data))
    # Nonzero off-diagonal elements expose accidental diagonal-only selection.
    rng = np.random.default_rng(5)
    a = rng.normal(size=(offset, offset))
    covariance = a @ a.T + np.eye(offset)
    cov_file = tmp_path / 'covariance.txt'
    np.savetxt(cov_file, covariance)
    return path, cov_file, covariance


def load(mock_inputs, tracers, stats, **kwargs):
    path, cov_file, _ = mock_inputs
    kwargs.setdefault('covariance_statistics', ['wp', 'xi_ells', 'delta_sigma'])
    return example.load_fit_data(path, tracers, stats, cov_file,
                                 covariance_tracers=['LRG', 'ELG'], **kwargs)


@pytest.mark.parametrize('tracers, stats, indices', [
    (['ELG'], ['wp'], list(range(3))),
    (['LRG'], ['xi2', 'wp'], list(range(32, 36)) + list(range(25, 28))),
    (['LRG', 'ELG'], ['delta_sigma'], list(range(11, 14)) + list(range(36, 39))),
    (['ELG', 'LRG'], ['wp', 'xi_ells'], list(range(11)) + list(range(14, 25)) + list(range(25, 36))),
])
def test_subset_preserves_all_covariance_blocks(mock_inputs, tracers, stats, indices):
    inputs = load(mock_inputs, tracers, stats)
    np.testing.assert_array_equal(inputs.indices, indices)
    np.testing.assert_array_equal(inputs.vector, indices)
    np.testing.assert_allclose(inputs.covariance, mock_inputs[2][np.ix_(indices, indices)])
    assert inputs.hartlap_deficit == (len(indices) + 1) / 1785
    if 'delta_sigma' in inputs.statistics:
        assert 'ELG_LRG' not in inputs.observations['delta_sigma']


def test_invalid_layouts_and_precision_are_rejected(mock_inputs):
    with pytest.raises(ValueError, match='duplicates'):
        load(mock_inputs, ['LRG'], ['xi_ells', 'xi2'])
    with pytest.raises(ValueError, match='Covariance layout'):
        load(mock_inputs, ['LRG'], ['delta_sigma'], covariance_statistics=['wp'])
    with pytest.raises(ValueError, match='nmocks'):
        load(mock_inputs, ['LRG'], ['wp'], nmocks=4)
    assert load(mock_inputs, ['LRG'], ['wp'], nmocks=0, no_hartlap=True).hartlap_deficit == 0.
    np.savetxt(mock_inputs[1], np.eye(3))
    with pytest.raises(ValueError, match='Covariance shape'):
        load(mock_inputs, ['LRG'], ['wp'])


def test_check_data_and_report_need_no_halo_catalogue(mock_inputs, tmp_path, monkeypatch):
    from HODDIES import sim_loader
    monkeypatch.setattr(sim_loader, 'AbacusSummitSim', lambda **kwargs: pytest.fail('check-data loaded halos'))
    monkeypatch.setattr(reporting, 'DEFAULT_ROOTS', [])
    output = tmp_path / 'check'
    report = tmp_path / 'report.md'
    assert example.main(['--data-file', str(mock_inputs[0]), '--covariance-file', str(mock_inputs[1]),
                         '--covariance-tracers', 'LRG', 'ELG',
                         '--covariance-statistics', 'wp', 'xi_ells', 'delta_sigma', '--tracer', 'ELG', '--stats', 'xi2',
                         '--check-data', '--output', str(output), '--report', str(report)]) == 0
    manifest = json.loads((output / 'configuration.json').read_text())
    assert manifest['statistic_order'] == ['xi2']
    assert manifest['configuration']['fit_param']['priors'].keys() == {'ELG'}
    assert manifest['simulation']['load_field_particles'] is False
    assert 'Data checked; no optimization' in report.read_text()
    assert not (output / 'best_fit_statistics.npz').exists()


def toy_simulation():
    from HODDIES.sim_loader import Base_catalogue
    class ToyCatalogue(Base_catalogue):
        def cosmo(self, **kwargs):
            return None
    rng = np.random.default_rng(42)
    size = 12000
    data = {axis: rng.uniform(0., 100., size) for axis in 'xyz'}
    data.update({f'v{axis}': rng.normal(0., 150., size) for axis in 'xyz'})
    data.update(Mh=10**rng.uniform(11., 14.8, size), Rh=np.full(size, 400.),
                c=rng.uniform(3., 10., size), Vrms=np.full(size, 200.), halo_id=np.arange(size, dtype='i8'))
    return ToyCatalogue(data, boxsize=100., z_simu=.5, nthreads=2, setup_logger=False)


@pytest.mark.parametrize('tracers, stats, plot', [
    (['ELG'], ['xi2'], False), (['LRG'], ['wp'], True),
    (['LRG', 'ELG'], ['wp', 'xi_ells'], True),
])
def test_cli_with_real_hod_optimizer_and_plot(mock_inputs, tmp_path, monkeypatch, tracers, stats, plot):
    from HODDIES import sim_loader
    simulation = toy_simulation()
    monkeypatch.setattr(sim_loader, 'AbacusSummitSim', lambda **kwargs: simulation)
    output, report = tmp_path / 'fit', tmp_path / 'report.md'
    options = [] if plot else ['--no-plot']
    assert example.main(['--data-file', str(mock_inputs[0]), '--covariance-file', str(mock_inputs[1]),
                         '--covariance-tracers', 'LRG', 'ELG',
                         '--covariance-statistics', 'wp', 'xi_ells', 'delta_sigma', '--tracers', *tracers, '--statistics', *stats,
                         '--output', str(output), '--report', str(report), '--nthreads', '2',
                         '--maxiter', '1', '--popsize', '4', '--ftol', '0', *options]) == 0
    with np.load(output / 'best_fit_statistics.npz') as saved:
        result = np.load(output / 'best_fit.npy', allow_pickle=True).item()
        np.testing.assert_allclose(saved['chi2'], result['fun'], rtol=1e-10)
        np.testing.assert_array_equal(saved['parameters'], result['x'])
        assert saved['parameter_names'].size == len(tracers)
    record = json.loads((output / 'fit_summary.json').read_text())
    assert record['state'] == 'Completed'
    assert record['optimizer']['success'] is False  # Deliberately bounded to one iteration.
    assert (output / 'best_fit.png').is_file() == plot
    assert bool(record['plot_file']) == plot
    if plot:
        assert Path(record['plot_file']).is_file()
    assert 'Completed' in report.read_text()


def make_record(mock_inputs, output, run_id, chi2, plot=True):
    inputs = load(mock_inputs, ['ELG'], ['wp'])
    config = example.model_configuration(inputs, ['ELG'], output, 2, 10)
    manifest = dict(workflow=example.WORKFLOW, configuration=config, simulation=dict(sim_name='toy', z_simu=.5),
                    data_file=str(mock_inputs[0]), covariance_file=str(mock_inputs[1]),
                    covariance_pairs=inputs.covariance_pairs, covariance_statistics=inputs.covariance_statistics,
                    statistic_order=inputs.statistics, pair_order=inputs.pairs, ndata=3, nmocks=1786,
                    precision_multiplier=1-inputs.hartlap_deficit,
                    run=dict(id=run_id, started_at='2026-10-21T08:00:00+00:00', mpi_ranks=1, state='Started'))
    example.save_inputs(inputs, manifest, output)
    np.savez(output / 'best_fit_statistics.npz', chi2=chi2, parameter_names=['M_0_ELG'], parameters=[12.5])
    if plot:
        (output / 'best_fit.png').write_bytes(run_id.encode())
    return reporting.save_run_summary(output, dict(fun=chi2, success=False, status=-1), [[12., 13.]], .1, plot)


def test_repeated_runs_and_interruption_do_not_mix_results(mock_inputs, tmp_path):
    output = tmp_path / 'output'
    first = make_record(mock_inputs, output, 'first', 4.)
    first_json = (output / 'run_history/first.json').read_text()
    second = make_record(mock_inputs, output, 'second', 3., plot=False)
    assert second['plot_file'] is None  # A stale best_fit.png must not be attached.
    assert (output / 'run_history/first.json').read_text() == first_json
    assert Path(first['plot_file']).read_bytes() == b'first'
    manifest = json.loads((output / 'configuration.json').read_text())
    manifest['run']['id'] = 'interrupted'
    reporting.write_json(output / 'configuration.json', manifest)
    records, issues = reporting.collect_runs([tmp_path, output])
    assert not issues and len(records) == 3
    interrupted = next(record for record in records if record['run_id'] == 'interrupted')
    assert interrupted['chi2'] is None
    assert all(row['value'] is None for row in interrupted['parameters'])
    content = reporting.render_report(records, issues, tmp_path / 'report.md', [tmp_path])
    assert content.count('best_fit_statistics.npz') == 0  # Latest arrays precede this interrupted run.
    assert 'run_history/first.png' in content
    assert 'run_history/second.png' not in content


def test_report_registry_keeps_extra_roots_and_excludes_other_workflows(mock_inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, 'DEFAULT_ROOTS', [])
    make_record(mock_inputs, tmp_path / 'one', 'one', 4.)
    make_record(mock_inputs, tmp_path / 'two', 'two', 3.)
    reporting.write_json(tmp_path / 'desi/configuration.json', dict(workflow='other'))
    reporting.write_json(tmp_path / 'broken/configuration.json', dict(workflow=example.WORKFLOW))
    report = tmp_path / 'report.md'
    reporting.update_report(report, [tmp_path / 'one'])
    reporting.update_report(report, [tmp_path / 'two', tmp_path / 'desi', tmp_path / 'broken'])
    content = report.read_text()
    assert 'Run ID: `one`' in content and 'Run ID: `two`' in content
    assert 'Files requiring attention' in content
    assert content.count('State: Completed') == 2


@pytest.mark.parametrize('stats, label, source_indices, selected_indices', [
    (['wp'], 'wp', list(range(25, 28)), list(range(25, 28))),
    (['xi_ells'], 'xi', list(range(28, 36)), list(range(28, 36))),
    (['xi2', 'wp'], 'wp+xi', list(range(25, 36)), list(range(32, 36)) + list(range(25, 28))),
    (['delta_sigma'], 'delta_sigma', list(range(36, 39)), list(range(36, 39))),
    (['delta_sigma', 'wp'], 'wp+delta_sigma', list(range(25, 28)) + list(range(36, 39)), list(range(36, 39)) + list(range(25, 28))),
    (['wp', 'xi_ells', 'delta_sigma'], 'wp+xi+delta_sigma', list(range(25, 39)), list(range(25, 39))),
])
def test_default_matrix_matches_selected_families(mock_inputs, tmp_path, monkeypatch, stats, label, source_indices, selected_indices):
    monkeypatch.setattr(example, 'COVARIANCE_DIR', tmp_path)
    path = tmp_path / f'Mcov_small_box_z0.5_{label}_LRG.txt'
    np.savetxt(path, mock_inputs[2][np.ix_(source_indices, source_indices)])
    inputs = example.load_fit_data(mock_inputs[0], ['LRG'], stats)
    assert inputs.covariance_file == path
    np.testing.assert_array_equal(inputs.vector, selected_indices)
    np.testing.assert_allclose(inputs.covariance, mock_inputs[2][np.ix_(selected_indices, selected_indices)])
    assert ('delta_sigma' in inputs.covariance_statistics) == ('delta_sigma' in stats)


def test_missing_matching_family_file_does_not_fall_back(mock_inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(example, 'COVARIANCE_DIR', tmp_path)
    np.savetxt(tmp_path / 'Mcov_small_box_z0.5_wp+xi+delta_sigma_LRG.txt', mock_inputs[2][25:, 25:])
    with pytest.raises(FileNotFoundError, match='z0.5_wp_LRG.txt'):
        example.load_fit_data(mock_inputs[0], ['LRG'], ['wp'])
    assert example.parser().parse_args([]).covariance_statistics is None
