"""Reporting checks: retain previous fits and never mix incomplete reruns."""

import json

import numpy as np
import pytest

from HODDIES.fit_functions.scripts import run_desi_y1_minimizer as example
from HODDIES.fit_functions.scripts import summarize_desi_y1_fits as reporting


def make_output(output, run_id='first', tracers=('ELG',), chi2=42.):
    output.mkdir(parents=True, exist_ok=True)
    config = example.model_configuration(list(tracers), output)
    config['fit_param']['minimizer'].update(method='cmaes', maxiter=5, popsize=8, xtol=1., ftol=34.5)
    manifest = dict(configuration=config,
                    simulation=dict(sim_name='synthetic-test-only', z_simu=.95),
                    redshift_bin=[.8, 1.1], pair_order=['ELG_ELG'],
                    statistic_order=['wp', 'xi0', 'xi2'], correlation_file='test-correlation.txt',
                    data_files={}, nmocks=1883, precision_multiplier=.96,
                    run=dict(id=run_id, started_at='2026-10-19T12:00:00+00:00',
                             mpi_ranks=8, state='Started; completion not recorded'))
    reporting.write_json(output / 'configuration.json', manifest)
    rows = reporting.prior_rows(config)
    values = [(row['lower'] + row['upper']) / 2 for row in rows]
    np.savez(output / 'fit_inputs.npz', data_vector=np.zeros(69))
    np.savez(output / 'best_fit_statistics.npz', chi2=chi2,
             parameter_names=np.array([row['name'] for row in rows]), parameters=values)
    result = dict(fun=np.float64(chi2), nit=np.int64(5), nfev=np.int64(40),
                  status=np.int64(-1), success=np.bool_(False), message='Maximum iterations reached')
    bounds = [(row['lower'], row['upper']) for row in rows]
    return result, bounds


def test_repeated_output_directory_keeps_independent_results(tmp_path):
    output = tmp_path / 'ELG'
    result, bounds = make_output(output, run_id='first', chi2=42.)
    (output / 'best_fit.png').write_bytes(b'first run comparison')
    reporting.save_run_summary(output, result, bounds, .5)
    first_json = (output / 'run_history/first.json').read_text()
    result, bounds = make_output(output, run_id='second', chi2=35.)
    (output / 'best_fit.png').write_bytes(b'second run comparison')
    reporting.save_run_summary(output, result, bounds, .7)
    assert (output / 'run_history/first.json').read_text() == first_json
    records, issues = reporting.collect_runs([tmp_path, output])
    completed = [record for record in records if record['state'] == 'Completed']
    assert not issues
    assert len(completed) == 2  # Overlapping scan roots do not duplicate fits.
    assert {record['chi2'] for record in completed} == {42., 35.}
    assert (output / 'run_history/first.png').read_bytes() == b'first run comparison'
    assert (output / 'run_history/second.png').read_bytes() == b'second run comparison'
    content = reporting.render_report(records, issues, tmp_path / 'report.md', [tmp_path])
    assert '![F1: DESI Y1 data and best-fit wp, xi0, xi2](ELG/run_history/first.png)' in content
    assert '![F2: DESI Y1 data and best-fit wp, xi0, xi2](ELG/run_history/second.png)' in content
    assert completed[0]['optimizer']['success'] is False
    assert completed[0]['optimizer']['status'] == -1


def test_incomplete_rerun_cannot_reuse_old_best_point(tmp_path):
    output = tmp_path / 'ELG'
    result, bounds = make_output(output)
    reporting.save_run_summary(output, result, bounds, .5)
    manifest = json.loads((output / 'configuration.json').read_text())
    manifest['run']['id'] = 'interrupted'
    reporting.write_json(output / 'configuration.json', manifest)
    records, issues = reporting.collect_runs([tmp_path])
    assert not issues
    assert len([record for record in records if record['state'] == 'Completed']) == 1
    interrupted = next(record for record in records if record['run_id'] == 'interrupted')
    assert interrupted['chi2'] is None
    assert all(row['value'] is None for row in interrupted['parameters'])


def test_report_keeps_registered_roots_and_convergence_status(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, 'DEFAULT_ROOTS', [])
    first, second = tmp_path / 'root1/ELG', tmp_path / 'root2/LRG'
    for output, run_id, tracers in [(first, 'ELG-test', ('ELG',)), (second, 'LRG-test', ('LRG',))]:
        result, bounds = make_output(output, run_id=run_id, tracers=tracers)
        reporting.save_run_summary(output, result, bounds, .5)
    report = tmp_path / 'report.md'
    reporting.update_report(report, [first.parent])
    reporting.update_report(report, [second.parent])
    content = report.read_text()
    assert 'ELG-test' in content and 'LRG-test' in content
    assert '| status | -1 |' in content
    assert '| success | False |' in content
    assert '0.7118644' in content  # 42 / (69 - 10).
    assert '[ELG-test.json]' in content
    assert 'ab_c_cen_LRG' in content and 'ab_c_sat_LRG' in content
    assert 'Not run; model template' in content  # Joint fit has no saved result.


def test_existing_npz_results_can_be_reported_without_pickle(tmp_path):
    output = tmp_path / 'legacy'
    make_output(output)
    manifest = json.loads((output / 'configuration.json').read_text())
    manifest.pop('run')
    reporting.write_json(output / 'configuration.json', manifest)
    records, issues = reporting.collect_runs([tmp_path])
    assert not issues
    legacy = next(record for record in records if record['output_directory'] == str(output))
    assert legacy['state'].startswith('Completed (legacy')
    assert legacy['chi2'] == 42.
    assert all(row['value'] is not None for row in legacy['parameters'])


def test_nonfinite_result_is_not_archived(tmp_path):
    result, bounds = make_output(tmp_path, chi2=float('nan'))
    with pytest.raises(ValueError, match='non-finite'):
        reporting.save_run_summary(tmp_path, result, bounds, .5)
    assert not (tmp_path / 'run_history').exists()


def test_no_plot_does_not_reuse_a_previous_run_image(tmp_path):
    result, bounds = make_output(tmp_path)
    (tmp_path / 'best_fit.png').write_bytes(b'previous run image')
    record = reporting.save_run_summary(tmp_path, result, bounds, .5, plot=False)
    assert record['plot_file'] is None
    assert reporting.comparison_plot(record) is None
    assert not (tmp_path / 'run_history/first.png').exists()


def test_old_unarchived_image_is_not_assigned_to_another_run(tmp_path):
    result, bounds = make_output(tmp_path, run_id='current')
    record = reporting.save_run_summary(tmp_path, result, bounds, .5)
    record.pop('plot_file')  # Simulate records made before per-run plot archival.
    record['run_id'] = 'older'
    (tmp_path / 'best_fit.png').write_bytes(b'current run image')
    assert reporting.comparison_plot(record) is None


def test_data_only_figure_uses_saved_measurements(tmp_path):
    make_output(tmp_path)
    rp, s = np.array([.5, 1.5]), np.array([1., 2., 3.])
    np.savez(tmp_path / 'fit_inputs.npz', data_vector=np.zeros(8),
             ELG_ELG_wp_x=rp, ELG_ELG_wp_y=np.ones(2), ELG_ELG_wp_sigma=np.full(2, .1),
             ELG_ELG_xi_ells_x=s, ELG_ELG_xi_ells_y=np.ones((2, 3)),
             ELG_ELG_xi_ells_sigma=np.full((2, 3), .1))
    records, issues = reporting.collect_runs([tmp_path])
    record = next(record for record in records if record['output_directory'] == str(tmp_path))
    assert record['chi2'] is None
    content = reporting.render_report([record], issues, tmp_path / 'report.md', [tmp_path])
    assert '![F1: DESI Y1 measurements of wp, xi0, xi2](data.png)' in content
    assert (tmp_path / 'data.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert 'No completed best-fit model is available yet.' in content


def test_corrupt_manifest_does_not_hide_good_runs(tmp_path):
    result, bounds = make_output(tmp_path / 'good')
    reporting.save_run_summary(tmp_path / 'good', result, bounds, .5)
    broken = tmp_path / 'broken'
    broken.mkdir()
    (broken / 'configuration.json').write_text('{')
    records, issues = reporting.collect_runs([tmp_path])
    assert any(record['state'] == 'Completed' for record in records)
    assert len(issues) == 1 and 'broken' in issues[0]
