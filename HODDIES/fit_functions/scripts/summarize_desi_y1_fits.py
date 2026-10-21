#!/usr/bin/env python3
"""Regenerate the DESI Y1 fit report from saved inputs and completed-run records.

From the repository root, in the cosmodesi environment::

    python HODDIES/fit_functions/scripts/summarize_desi_y1_fits.py
    python HODDIES/fit_functions/scripts/summarize_desi_y1_fits.py --root /path/to/other/fits

Only DESI-example outputs with configuration.json are included. No simulation or
pickled optimizer result is loaded. Repeated fits retain independent JSON records
in run_history/, even when the main output files are overwritten.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.util
from itertools import combinations_with_replacement
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.parse import quote

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT = REPO_ROOT / 'DESI_Y1_FIT_RESULTS.md'
DEFAULT_ROOTS = [REPO_ROOT.parent / 'tmp', REPO_ROOT / 'tmp']


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def write_json(path, value):
    """Replace metadata atomically so readers never see a partially written run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def prior_rows(configuration):
    rows = []
    for tracer, priors in configuration.get('fit_param', {}).get('priors', {}).items():
        for name, bounds in priors.items():
            if name == 'assembly_bias':
                for proxy, components in bounds.items():
                    for index, component in enumerate(['cen', 'sat']):
                        rows.append(dict(name=f'ab_{proxy}_{component}_{tracer}',
                                         lower=components[index][0], upper=components[index][1],
                                         initial=configuration[tracer]['assembly_bias'][proxy][index], value=None))
            else:
                rows.append(dict(name=f'{name}_{tracer}', lower=bounds[0], upper=bounds[1],
                                 initial=configuration[tracer].get(name), value=None))
    return rows


def binning_rows(inputs, pairs):
    rows = []
    for pair in pairs:
        for statistic, labels in [('wp', ['wp']), ('xi_ells', ['xi0', 'xi2']), ('delta_sigma', ['delta_sigma'])]:
            key = f'{pair}_{statistic}'
            if key + '_x' not in inputs or key + '_y' not in inputs:
                continue
            x, y = inputs[key + '_x'], inputs[key + '_y']
            for label, values in zip(labels, np.atleast_2d(y), strict=True):
                rows.append([pair, label, int(values.size), float(np.min(x)), float(np.max(x))])
    return rows


def save_run_summary(output, result, bounds, elapsed_seconds, plot=True):
    """Store a self-contained, immutable record after best-point reevaluation."""
    output = Path(output).resolve()
    manifest = json.loads((output / 'configuration.json').read_text())
    with np.load(output / 'best_fit_statistics.npz', allow_pickle=False) as saved:
        names = saved['parameter_names'].tolist()
        values = saved['parameters'].tolist()
        chi2 = float(saved['chi2'])
    if not np.isfinite(chi2) or not np.isfinite(values).all():
        raise ValueError('Cannot record a completed fit with non-finite chi2 or parameters.')
    run_id = manifest['run']['id']
    parameters = [dict(name=name, value=value, lower=float(bound[0]), upper=float(bound[1]))
                  for name, value, bound in zip(names, values, bounds, strict=True)]
    with np.load(output / 'fit_inputs.npz', allow_pickle=False) as inputs:
        ndata = int(inputs['data_vector'].size)
        binning = binning_rows(inputs, manifest['pair_order'])
    optimizer = dict(manifest['configuration']['fit_param']['minimizer'])
    for key in ['fun', 'nit', 'nfev', 'status', 'success', 'message']:
        value = result.get(key)
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            value = None
        optimizer[key] = value
    record = dict(schema_version=1, run_id=run_id, state='Completed',
                  output_directory=str(output), manifest=manifest,
                  finished_at=utc_now(), elapsed_seconds=elapsed_seconds,
                  ndata=ndata, binning=binning, chi2=chi2, parameters=parameters, optimizer=optimizer)
    record['plot_file'] = None
    if plot and (output / 'best_fit.png').is_file():
        archived_plot = output / 'run_history' / f'{run_id}.png'
        archived_plot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output / 'best_fit.png', archived_plot)
        record['plot_file'] = str(archived_plot)
    # Keep history independent of the latest configuration/NPZ files.
    write_json(output / 'run_history' / f'{run_id}.json', record)
    write_json(output / 'fit_summary.json', record)
    return record


def _load_example():
    # Import only the adjacent example, avoiding the heavy HODDIES __init__.
    name = '_desi_y1_report_example'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name('run_desi_y1_minimizer.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def collect_runs(roots):
    records, issues, seen = [], [], set()
    manifests = set()
    for root in roots:
        root = Path(root).resolve()
        if root.is_dir():
            manifests.update(root.rglob('configuration.json'))
    for filename in sorted(manifests):
        output = filename.parent
        try:
            manifest = json.loads(filename.read_text())
            if not {'redshift_bin', 'pair_order', 'correlation_file', 'configuration'} <= manifest.keys():
                continue
            history = sorted((output / 'run_history').glob('*.json'))
            if not history and (output / 'fit_summary.json').is_file():
                history = [output / 'fit_summary.json']
            current_id = manifest.get('run', {}).get('id')
            current_recorded = False
            for archive in history:
                try:
                    record = json.loads(archive.read_text())
                    key = record['run_id']
                    if key not in seen:
                        records.append(record)
                        seen.add(key)
                    current_recorded |= key == current_id
                except (OSError, ValueError, KeyError, TypeError) as error:
                    issues.append(f'{archive}: {error}')
            if current_recorded:
                continue
            configuration = manifest['configuration']
            record = dict(run_id=current_id or str(output), output_directory=str(output),
                          manifest=manifest, parameters=prior_rows(configuration),
                          optimizer=configuration.get('fit_param', {}).get('minimizer', {}),
                          state='Inputs saved; no completed result', ndata=None, chi2=None)
            with np.load(output / 'fit_inputs.npz', allow_pickle=False) as inputs:
                record['ndata'] = int(inputs['data_vector'].size)
                record['binning'] = binning_rows(inputs, manifest['pair_order'])
            # Backfill older completed outputs that predate JSON run records.
            # New runs must match an explicit run ID, to avoid mixing old best fits
            # with a newer configuration after an interrupted rerun.
            best = output / 'best_fit_statistics.npz'
            if best.is_file() and current_id is None:
                with np.load(best, allow_pickle=False) as saved:
                    names = saved['parameter_names'].tolist()
                    values = saved['parameters'].tolist()
                    record['chi2'] = float(saved['chi2'])
                rows = {row['name']: row for row in record['parameters']}
                record['parameters'] = [dict(rows.get(name, {}), name=name, value=value)
                                        for name, value in zip(names, values, strict=True)]
                record['state'] = 'Completed (legacy files; optimizer diagnostics unavailable)'
            if current_id is not None:
                record['state'] = manifest['run'].get('state', 'Started; completion not recorded')
            records.append(record)
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(f'{filename}: {error}')
    # Show each requested model family, without inventing results for unrun fits.
    example = _load_example()
    present = {tuple(sorted(item['manifest']['configuration']['tracers'])) for item in records}
    for tracers in [['ELG'], ['LRG'], ['LRG', 'ELG']]:
        if tuple(sorted(tracers)) in present:
            continue
        configuration = example.model_configuration(tracers, '<not run>')
        manifest = dict(configuration=configuration, simulation=dict(sim_name='AbacusSummit_highbase_c000_ph100',
                        z_simu=example.Z_SIM), redshift_bin=[.8, 1.1],
                        statistic_order=['wp', 'xi0', 'xi2'],
                        pair_order=['_'.join(pair) for pair in combinations_with_replacement(sorted(tracers), 2)],
                        data_files={},
                        correlation_file=None, cuts=dict(rppi='[::2][4:-3]', smu='[16:-6]', pimax=40))
        records.append(dict(run_id='template-' + '-'.join(tracers), manifest=manifest,
                            state='Not run; model template', ndata=None, chi2=None,
                            parameters=prior_rows(configuration), optimizer={}, output_directory=None))
    return records, issues


def cell(value):
    if value is None:
        return '—'
    if isinstance(value, float):
        return f'{value:.7g}'
    return str(value).replace('|', '\\|').replace('\n', ' ')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(cell, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows]])


def link(path, report, label=None):
    path = Path(path)
    target = quote(os.path.relpath(path, Path(report).parent), safe='/._-')
    return f'[{label or path.name}]({target})'


def comparison_plot(record):
    """Return only a comparison image that belongs to this completed run."""
    if record['chi2'] is None or not record['output_directory']:
        return None
    if 'plot_file' in record:
        plot = Path(record['plot_file']) if record['plot_file'] else None
        return plot if plot is not None and plot.is_file() else None
    output = Path(record['output_directory'])
    # Older run records can have a plot archived by a previous report refresh.
    if not Path(record['run_id']).is_absolute():
        archived = output / 'run_history' / f'{record["run_id"]}.png'
        if archived.is_file():
            return archived
    current = json.loads((output / 'configuration.json').read_text())
    current_id = current.get('run', {}).get('id')
    if current_id is not None and current_id != record['run_id']:
        return None
    plot = output / 'best_fit.png'
    if not plot.is_file():
        return None
    if current_id is not None:
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plot, archived)
        return archived
    return plot


def data_plot(record):
    """Plot saved measurements without loading halos or evaluating a model."""
    if not record['output_directory']:
        return None
    output = Path(record['output_directory'])
    source, plot = output / 'fit_inputs.npz', output / 'data.png'
    if not source.is_file():
        return None
    pairs = record['manifest']['pair_order']
    panels = [('wp', 0), ('xi_ells', 0), ('xi_ells', 1)]
    if 'delta_sigma' in record['manifest'].get('statistic_order', []):
        panels.append(('delta_sigma', 0))
    with np.load(source, allow_pickle=False) as saved:
        required = [f'{pair}_{stat}_{suffix}' for pair in pairs
                    for stat in dict.fromkeys(stat for stat, _ in panels)
                    if stat != 'delta_sigma' or pair == 'LRG_LRG'
                    for suffix in ['x', 'y', 'sigma']]
        if not pairs or not all(key in saved for key in required):
            return None
        if plot.is_file() and plot.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            return plot
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(pairs), len(panels), figsize=(4.3 * len(panels), 3.5 * len(pairs)), squeeze=False)
        try:
            for row, pair in enumerate(pairs):
                for column, (stat, component) in enumerate(panels):
                    prefix = f'{pair}_{stat}'
                    if prefix + '_y' not in saved:
                        axes[row, column].set_visible(False)
                        continue
                    x = saved[prefix + '_x']
                    if x.ndim > 1:
                        x = x[component]
                    y = np.atleast_2d(saved[prefix + '_y'])[component]
                    sigma = np.atleast_2d(saved[prefix + '_sigma'])[component]
                    weight = x if stat == 'wp' else (1. if stat == 'delta_sigma' else x**2)
                    ax = axes[row, column]
                    ax.errorbar(x, weight * y, yerr=weight * sigma, fmt='o', ms=3, label='DESI Y1 data')
                    ax.set(xscale='log', xlabel='Separation [Mpc/h]',
                           title=f'{pair}: ' + ['rp wp', 's² xi0', 's² xi2', 'DeltaSigma [h Msun/pc²]'][column])
                    ax.grid(alpha=.2)
                    ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(plot, dpi=150)
        finally:
            plt.close(fig)
    return plot


def render_report(records, issues, report, roots):
    lines = ['# DESI Y1 HOD fit results', '',
             f'Generated {utc_now()} from saved fit artifacts. This file is regenerated automatically.', '',
             '## Updating this report', '',
             'New DESI example fits update this report on MPI rank 0 after best-point reevaluation. '
             'Completed runs keep JSON snapshots in `run_history/`, including repeated fits in the same directory. '
             'Their data-versus-best-fit plots are archived there and embedded below. '
             'Use a new output directory to retain each run’s full arrays and plot. '
             'Only outputs of `run_desi_y1_minimizer.py` are included; unrelated optimizer examples are excluded.', '',
             'From the repository root, with the cosmodesi environment active:', '',
             '```bash', 'python HODDIES/fit_functions/scripts/summarize_desi_y1_fits.py',
             '# Add another results directory:',
             'python HODDIES/fit_functions/scripts/summarize_desi_y1_fits.py --root /path/to/fits', '```', '',
             'Search roots: ' + ', '.join(f'`{Path(root).resolve()}`' for root in roots) + '.', '',
             '## Fit overview', '',
             'Clustering fits use wp + xi0 + xi2 in the redshift bin listed for each run. '
             'The hybrid covariance is C = R_small × outer(σ_JK, σ_JK), with the saved '
             'finite-mock precision multiplier applied once to the clustering block. '
             'Optional LRG lensing at z=0.5 uses diagonal CSV errors with zero clustering-lensing covariance; '
             'the lensing precision is unscaled. '
             'χ² is the saved reevaluation using the fit’s fixed mock seed; '
             'dof = Ndata − Nfree. Parameter uncertainties are not estimated by this optimizer.', '']
    overview = []
    for index, record in enumerate(records, 1):
        config = record['manifest']['configuration']
        nfree = len(record['parameters'])
        dof = record['ndata'] - nfree if record['ndata'] is not None else None
        reduced = record['chi2'] / dof if record['chi2'] is not None and dof > 0 else None
        overview.append([f'[F{index}](#f{index})', '+'.join(config['tracers']),
                         record['manifest'].get('redshift_bin'),
                         '+'.join(record['manifest'].get('statistic_order', [])), record['state'],
                         record['ndata'], nfree, dof, record['chi2'], reduced])
    lines += [table(['Fit', 'Tracers', 'Redshift bin', 'Statistics', 'Run state', 'Ndata', 'Nfree', 'dof', 'χ²', 'χ²/dof'], overview), '',
              '“Completed” means a result was saved; optimizer convergence is reported separately below. '
              'Blank best-fit cells mean that no completed parameter estimate is available.', '',
              '## Best-fit parameter comparison', '']
    names = list(dict.fromkeys(row['name'] for record in records for row in record['parameters']))
    values = [{row['name']: row.get('value') for row in record['parameters']} for record in records]
    lines += [table(['Parameter'] + [f'F{i}' for i in range(1, len(records) + 1)],
                    [[name] + [result.get(name) for result in values] for name in names]), '']
    for index, record in enumerate(records, 1):
        manifest = record['manifest']
        config = manifest['configuration']
        simulation = manifest.get('simulation', {})
        runtime = manifest.get('run', {})
        output = record['output_directory']
        lines += [f'<a id="f{index}"></a>', '', f'## F{index}: ' + '+'.join(config['tracers']), '',
                  f'Run ID: `{record["run_id"]}`. State: {record["state"]}.', '',
                  table(['Setting', 'Value'], [
                      ['Simulation', simulation.get('sim_name')], ['Snapshot redshift', simulation.get('z_simu')],
                      ['Halo root', simulation.get('path_to_sim')], ['Halo mass cut', simulation.get('mass_cut')],
                      ['Data redshift bin', manifest.get('redshift_bin')], ['Statistics', manifest.get('statistic_order')],
                      ['Pair order', manifest.get('pair_order')], ['Cuts', manifest.get('cuts')],
                      ['Hybrid correlation matrix', manifest.get('correlation_file')],
                      ['Covariance mock count', manifest.get('nmocks')],
                      ['Precision multiplier', manifest.get('precision_multiplier')],
                      ['Mock seed', config.get('seed')], ['MPI ranks', runtime.get('mpi_ranks')],
                      ['Threads/rank', config.get('nthreads')], ['Slurm job', runtime.get('slurm_job_id')],
                      ['Started (UTC)', runtime.get('started_at')], ['Finished (UTC)', record.get('finished_at')],
                      ['Elapsed seconds', record.get('elapsed_seconds')]]), '']
        if record.get('binning'):
            lines += [table(['Pair', 'Statistic', 'Bins', 'First bin centre [Mpc/h]', 'Last bin centre [Mpc/h]'],
                            record['binning']), '']
        if manifest.get('lensing'):
            lensing = manifest['lensing']
            lines += ['### Lensing measurement', '',
                      table(['Setting', 'Value'], [
                          ['CSV file', lensing['file']], ['Columns', lensing['columns']],
                          ['Covariance', lensing['covariance']], ['Cross covariance', lensing['cross_covariance']],
                          ['DeltaSigma units', lensing['units']], ['Excluded CSV rows (zero-based)', lensing['excluded_rows']],
                          ['Bin edges', lensing['bin_edges']],
                          ['Model evaluation', lensing.get('evaluation', 'geometric bin centres')],
                          ['Precision correction', lensing['precision_correction']]]), '']
        lines += ['### Models', '']
        for tracer in config['tracers']:
            model = config[tracer]
            lines += [f'**{tracer}**: `{model["HOD_model"]}`; satellite model `{model.get("sat_HOD_model")}`. '
                      f'Conformity: {model.get("conformity_bias")}; '
                      f'link_sat_to_central: {model.get("link_sat_to_central")}; '
                      f'global assembly bias: {config.get("use_assembly_bias")}; '
                      f'configured tracer coefficients: `{model.get("assembly_bias")}`.', '',
                      'Configured values below are the input settings; fitted values are in the parameter table. '
                      'Density normalization can change occupation amplitudes during mock generation.', '',
                      table(['Input setting', 'Value'], [[key, value] for key, value in model.items()
                                                        if key not in ['HOD_model', 'assembly_bias']]), '']
        lines += ['### Optimizer', '',
                  table(['Setting / diagnostic', 'Value'], [[key, record.get('optimizer', {}).get(key)]
                        for key in ['method', 'maxiter', 'popsize', 'xtol', 'ftol', 'fun', 'nit', 'nfev',
                                    'status', 'success', 'message']]), '',
                  'For CMA-ES, ftol is an absolute objective target. The DESI example defaults to '
                  'ftol = Ndata / 2 and, with MPI, popsize = max(2, number of ranks).', '',
                  '### Fitted parameters', '',
                  table(['Parameter', 'Best fit', 'Prior minimum', 'Prior maximum'],
                        [[row['name'], row.get('value'), row.get('lower'), row.get('upper')]
                         for row in record['parameters']]), '', '### Data and artifacts', '']
        if output:
            lines += [f'Output directory: `{output}`.']
            archive = Path(output) / 'run_history' / f'{record["run_id"]}.json'
            if archive.is_file():
                lines += ['Archived run metadata: ' + link(archive, report) + '.']
            lines += ['Latest directory artifacts: ' + ', '.join(link(Path(output) / name, report)
                      for name in ['configuration.json', 'fit_inputs.npz', 'best_fit.npy',
                                   'best_fit_statistics.npz', 'best_fit.png'] if (Path(output) / name).is_file()) + '.', '']
        else:
            lines += ['No saved run found. Settings above describe the current model template.', '']
        for pair, files in manifest.get('data_files', {}).items():
            lines += [f'- {pair}, {mode}: `{filename}`' for mode, filename in files.items()]
        lines += ['']
        lines += ['### Data and best-fit plots', '']
        statistic_label = ', '.join(manifest.get('statistic_order', ['wp', 'xi0', 'xi2']))
        plot = comparison_plot(record)
        if plot is not None:
            lines += ['!' + link(plot, report, f'F{index}: DESI Y1 data and best-fit {statistic_label}'), '',
                      'Measurements with error bars and the best-fit model for this run.', '']
        elif record['chi2'] is None:
            plot = data_plot(record)
            if plot is not None:
                lines += ['!' + link(plot, report, f'F{index}: DESI Y1 measurements of {statistic_label}'), '',
                          'Measurements with saved errors (jackknife for clustering). No completed best-fit model is available yet.', '']
            else:
                lines += ['No saved measurements or best-fit plot are available for this entry.', '']
        else:
            lines += ['No comparison plot was saved for this run.', '']
    if issues:
        lines += ['## Files requiring attention', '', *[f'- {cell(issue)}' for issue in issues], '']
    return '\n'.join(lines)


def update_report(report=DEFAULT_REPORT, extra_roots=()):
    report = Path(report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    # Serialize rank-0 writers from different jobs; the lock lives in ignored tmp/.
    lock_dir = report.parent / 'tmp'
    lock_dir.mkdir(exist_ok=True)
    with (lock_dir / f'.{report.name}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = lock_dir / f'.{report.name}.roots.json'
        previous = json.loads(registry.read_text()) if registry.is_file() else []
        roots = list(dict.fromkeys(Path(root).resolve() for root in [*DEFAULT_ROOTS, *previous, *extra_roots]))
        records, issues = collect_runs(roots)
        content = render_report(records, issues, report, roots)
        with tempfile.NamedTemporaryFile(mode='w', dir=report.parent, delete=False) as stream:
            stream.write(content)
            temporary = Path(stream.name)
        temporary.replace(report)
        write_json(registry, [str(root) for root in roots])
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, action='append', default=[], help='Additional results root (repeatable).')
    parser.add_argument('--output', type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    records = update_report(args.output, args.root)
    print(f'Updated {args.output}: {len(records)} fit entries.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
