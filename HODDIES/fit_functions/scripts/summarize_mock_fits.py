#!/usr/bin/env python3
"""Summarize run_mock_minimizer.py results without loading halos or pickle files.

    python HODDIES/fit_functions/scripts/summarize_mock_fits.py
    python HODDIES/fit_functions/scripts/summarize_mock_fits.py --root /path/to/fits --output MOCK_FIT_RESULTS.md

Only manifests tagged mock_minimizer_v1 are included. Completed runs keep
independent JSON and comparison-plot snapshots in run_history/. The latest
configuration can describe an interrupted rerun without inheriting older fits.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import quote

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT = REPO_ROOT / 'MOCK_FIT_RESULTS.md'
DEFAULT_ROOTS = [REPO_ROOT / 'tmp/mock_fits']
WORKFLOW = 'mock_minimizer_v1'


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, default=json_default, allow_nan=False) + '\n')


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parameter_rows(configuration):
    rows = []
    for tracer, priors in configuration['fit_param']['priors'].items():
        for name, bounds in priors.items():
            if name == 'assembly_bias':
                for proxy, limits in bounds.items():
                    for index, component in enumerate(('cen', 'sat')):
                        rows.append(dict(name=f'ab_{proxy}_{component}_{tracer}', lower=limits[index][0],
                                         upper=limits[index][1], input=configuration[tracer]['assembly_bias'][proxy][index], value=None))
            else:
                rows.append(dict(name=f'{name}_{tracer}', lower=bounds[0], upper=bounds[1],
                                 input=configuration[tracer].get(name), value=None))
    return rows


def save_run_summary(output, result, bounds, elapsed_seconds, plot=True):
    output = Path(output).resolve()
    manifest = json.loads((output / 'configuration.json').read_text())
    if manifest.get('workflow') != WORKFLOW:
        raise ValueError('Not a mock minimizer output directory.')
    with np.load(output / 'best_fit_statistics.npz', allow_pickle=False) as saved:
        names, values, chi2 = saved['parameter_names'].tolist(), saved['parameters'].tolist(), float(saved['chi2'])
    if not np.isfinite(chi2) or not np.isfinite(values).all():
        raise ValueError('Cannot archive non-finite fit results.')
    rows = {row['name']: row for row in parameter_rows(manifest['configuration'])}
    parameters = [dict(rows[name], value=value, lower=float(bound[0]), upper=float(bound[1]))
                  for name, value, bound in zip(names, values, bounds, strict=True)]
    optimizer = dict(manifest['configuration']['fit_param']['minimizer'])
    optimizer.update({key: result.get(key) for key in ('fun', 'nit', 'nfev', 'status', 'success', 'message')})
    run_id = manifest['run']['id']
    archived_plot = None
    if plot and (output / 'best_fit.png').is_file():
        target = output / 'run_history' / f'{run_id}.png'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output / 'best_fit.png', target)
        archived_plot = str(target)
    record = dict(workflow=WORKFLOW, run_id=run_id, state='Completed', manifest=manifest,
                  output_directory=str(output), parameters=parameters, chi2=chi2,
                  optimizer=optimizer, finished_at=utc_now(), elapsed_seconds=elapsed_seconds,
                  plot_file=archived_plot)
    write_json(output / 'run_history' / f'{run_id}.json', record)
    write_json(output / 'fit_summary.json', record)
    return record


def collect_runs(roots):
    records, issues, seen, manifests = [], [], set(), set()
    for root in roots:
        root = Path(root).resolve()
        if root.is_dir():
            manifests.update(root.rglob('configuration.json'))
    for path in sorted(manifests):
        try:
            manifest = json.loads(path.read_text())
            if manifest.get('workflow') != WORKFLOW:
                continue
            output = path.parent
            history = sorted((output / 'run_history').glob('*.json'))
            for archive in history:
                try:
                    record = json.loads(archive.read_text())
                    if record.get('workflow') != WORKFLOW:
                        continue
                    if record['run_id'] not in seen:
                        records.append(record)
                        seen.add(record['run_id'])
                except (OSError, ValueError, KeyError, TypeError) as error:
                    issues.append(f'{archive}: {error}')
            run = manifest['run']
            if run['id'] not in seen:
                records.append(dict(workflow=WORKFLOW, run_id=run['id'], state=run['state'],
                                    manifest=manifest, output_directory=str(output), chi2=None,
                                    parameters=parameter_rows(manifest['configuration']),
                                    optimizer=manifest['configuration']['fit_param']['minimizer'], plot_file=None))
                seen.add(run['id'])
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(f'{path}: {error}')
    records.sort(key=lambda record: (record['manifest']['run']['started_at'], record['run_id']))
    return records, issues


def cell(value):
    if value is None:
        return '—'
    if isinstance(value, float):
        return f'{value:.7g}'
    return str(value).replace('|', '\\|').replace('\n', ' ')


def table(headers, rows):
    return '\n'.join('| ' + ' | '.join(map(cell, row)) + ' |'
                     for row in [headers, ['---'] * len(headers), *rows])


def link(path, report, label=None):
    path = Path(path)
    return f'[{label or path.name}]({quote(os.path.relpath(path, Path(report).parent), safe="/._-")})'


def render_report(records, issues, report, roots):
    lines = ['# Mock HOD fit results', '', f'Generated {utc_now()} from saved mock-fit artifacts.', '',
             'Fits from `run_mock_minimizer.py`. Data and covariance use pair, statistic, multipole, bin order; '
             'lensing contributes auto-pairs only. The saved finite-mock precision multiplier is applied once. '
             'χ² is reevaluated at the optimizer’s best parameters with its fixed mock seed; '
             'dof = Ndata − Nfree. Optimizer success is separate from whether a result was saved. '
             'Parameter uncertainties are not estimated.', '',
             'Regenerate from the repository root in the cosmodesi environment:', '',
             '```bash', 'python HODDIES/fit_functions/scripts/summarize_mock_fits.py',
             '# Include another output directory:',
             'python HODDIES/fit_functions/scripts/summarize_mock_fits.py --root /path/to/fits', '```', '',
             'Search roots: ' + ', '.join(f'`{root}`' for root in roots) + '.', '', '## Fit overview', '']
    overview = []
    for index, record in enumerate(records, 1):
        manifest = record['manifest']
        ndata, nfree, chi2 = manifest['ndata'], len(record['parameters']), record['chi2']
        dof = ndata - nfree
        overview.append([f'[F{index}](#f{index})', '+'.join(manifest['configuration']['tracers']),
                         '+'.join(manifest['statistic_order']), record['state'], ndata, nfree, dof,
                         chi2, chi2 / dof if chi2 is not None and dof > 0 else None,
                         record['optimizer'].get('success')])
    lines += [table(['Fit', 'Tracers', 'Statistics', 'State', 'Ndata', 'Nfree', 'dof', 'χ²', 'χ²/dof', 'Optimizer success'], overview), '']
    if not records:
        lines += ['No saved mock-test fits found. Run the minimizer or use `--root` to include its output directory.', '']
    else:
        names = list(dict.fromkeys(row['name'] for record in records for row in record['parameters']))
        values = [{row['name']: row['value'] for row in record['parameters']} for record in records]
        lines += ['## Best-fit parameter comparison', '',
                  table(['Parameter', *[f'F{i}' for i in range(1, len(records) + 1)]],
                        [[name, *[result.get(name) for result in values]] for name in names]), '']
    for index, record in enumerate(records, 1):
        manifest, output = record['manifest'], Path(record['output_directory'])
        config, run = manifest['configuration'], manifest['run']
        lines += [f'<a id="f{index}"></a>', '', f'## F{index}: ' + '+'.join(config['tracers']) + ' / ' + '+'.join(manifest['statistic_order']), '',
                  f'Run ID: `{record["run_id"]}`. State: {record["state"]}.', '',
                  table(['Setting', 'Value'], [
                      ['Simulation', manifest['simulation']['sim_name']], ['Snapshot redshift', manifest['simulation']['z_simu']],
                      ['Data file', manifest['data_file']], ['Covariance file', manifest['covariance_file']],
                      ['Source covariance pairs', manifest['covariance_pairs']], ['Source covariance statistics', manifest['covariance_statistics']],
                      ['Fitted pair order', manifest['pair_order']], ['Fitted statistics', manifest['statistic_order']],
                      ['Covariance mock count', manifest['nmocks']], ['Precision multiplier', manifest['precision_multiplier']],
                      ['Mock seed', config['seed']], ['MPI ranks', run['mpi_ranks']], ['Threads/rank', config['nthreads']],
                      ['Slurm job', run.get('slurm_job_id')], ['Started (UTC)', run['started_at']],
                      ['Finished (UTC)', record.get('finished_at')], ['Elapsed seconds', record.get('elapsed_seconds')]]), '',
                  '### Models', '']
        for tracer in config['tracers']:
            lines += [f'**{tracer}**: `{config[tracer]["HOD_model"]}`.', '',
                      table(['Input setting', 'Value'], config[tracer].items()), '']
        lines += ['### Optimizer', '',
                  table(['Setting / diagnostic', 'Value'], [[key, record['optimizer'].get(key)] for key in
                        ('method', 'maxiter', 'popsize', 'xtol', 'ftol', 'fun', 'nit', 'nfev', 'status', 'success', 'message')]), '',
                  '### Fitted parameters', '',
                  '“Saved data input” is the HOD setting stored with the mock measurements. '
                  'The optimizer starts from a seeded draw within the priors. Density normalization can change amplitudes.', '',
                  table(['Parameter', 'Best fit', 'Saved data input', 'Prior minimum', 'Prior maximum'],
                        [[row['name'], row['value'], row.get('input'), row['lower'], row['upper']] for row in record['parameters']]), '',
                  '### Artifacts', '', f'Output directory: `{output}`.', '']
        archive = output / 'run_history' / f'{record["run_id"]}.json'
        if archive.is_file():
            lines += ['Archived run metadata: ' + link(archive, report) + '.', '']
        # Latest arrays belong only to the run identified by configuration.json.
        current = json.loads((output / 'configuration.json').read_text())
        if current['run']['id'] == record['run_id']:
            files = ['configuration.json', 'fit_inputs.npz']
            if record['chi2'] is not None:
                files += ['best_fit.npy', 'best_fit_statistics.npz']
            lines += ['Current run files: ' + ', '.join(link(output / name, report) for name in files if (output / name).is_file()) + '.', '']
        plot = record.get('plot_file')
        if plot and Path(plot).is_file():
            lines += ['!' + link(plot, report, f'F{index}: mock data and best fit'), '']
        else:
            lines += ['No comparison plot was saved for this run.', '']
    if issues:
        lines += ['## Files requiring attention', '', *[f'- {cell(issue)}' for issue in issues], '']
    return '\n'.join(lines)


def update_report(report=DEFAULT_REPORT, extra_roots=()):
    report = Path(report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = report.parent / 'tmp'
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / f'.{report.name}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = lock_dir / f'.{report.name}.roots.json'
        previous = json.loads(registry.read_text()) if registry.is_file() else []
        roots = list(dict.fromkeys(Path(root).resolve() for root in [*DEFAULT_ROOTS, *previous, *extra_roots]))
        records, issues = collect_runs(roots)
        atomic_write(report, render_report(records, issues, report, roots))
        write_json(registry, roots)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    records = update_report(args.output, args.root)
    print(f'Updated {args.output}: {len(records)} mock-fit entries.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
