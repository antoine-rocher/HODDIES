"""DESI example ordering/covariance checks and bounded current-pipeline fits."""

from pathlib import Path
import numpy as np
import pytest

from HODDIES.fit_functions.scripts import run_desi_y1_minimizer as example


@pytest.mark.parametrize('affinity, slurm_cpus, expected', [
    (64, '32', 32), (8, '32', 8), (24, None, 24),
    (12, 'invalid', 12), (12, '0', 12),
])
def test_available_cpus_respect_rank_affinity_and_slurm(monkeypatch, affinity, slurm_cpus, expected):
    monkeypatch.setattr(example.os, 'sched_getaffinity', lambda pid: set(range(affinity)))
    if slurm_cpus is None:
        monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
    else:
        monkeypatch.setenv('SLURM_CPUS_PER_TASK', slurm_cpus)
    assert example.available_cpu_count() == expected


@pytest.mark.parametrize('cpu_count, expected', [(8, 8), (None, 1)])
def test_available_cpus_fallback_without_affinity(monkeypatch, cpu_count, expected):
    def unavailable(pid):
        raise OSError('Affinity unavailable')
    monkeypatch.setattr(example.os, 'sched_getaffinity', unavailable)
    monkeypatch.setattr(example.os, 'cpu_count', lambda: cpu_count)
    monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
    assert example.available_cpu_count() == expected


@pytest.fixture
def measured_inputs(tmp_path, monkeypatch):
    """Distinct auto/cross blocks make order and error/value mixing detectable."""
    from pycorr import TwoPointCorrelationFunction

    root = tmp_path / 'data'
    corr_root = tmp_path / 'corr' / 'z0.950'
    corr_root.mkdir(parents=True)
    loaded = []
    markers = {'ELG_LOPnotqso': 100., 'LRG_ELG_LOPnotqso': 200., 'LRG': 300.}
    for mode in ['rppi', 'smu']:
        (root / mode).mkdir(parents=True)
        for name in markers:
            (root / mode / f'allcounts_{name}_GCcomb_0.8_1.1_default_angular_bitwise_log_njack128_nran4_split20.npy').touch()

    class Estimator:
        def __init__(self, mode, marker):
            self.mode, self.marker = mode, marker
            self.edges = (np.arange(3.), np.linspace(-1., 1., 5))

        def __getitem__(self, selection):
            loaded.append((self.mode, selection))
            return self

        def __call__(self, **kwargs):
            x = np.array([0.5, 1.5])
            if self.mode == 'rppi':
                assert kwargs['pimax'] == 40
                y, error = self.marker + np.array([1., 2.]), np.array([2., 3.])
            else:
                assert kwargs['ells'] == [0, 2]
                y = self.marker + np.array([[3., 4.], [5., 6.]])
                error = np.array([[4., 5.], [6., 7.]])
            return x, y, error

    def load(filename):
        filename = Path(filename)
        name = filename.name.removeprefix('allcounts_').split('_GCcomb_')[0]
        return Estimator(filename.parent.name, markers[name])

    monkeypatch.setattr(TwoPointCorrelationFunction, 'load', load)
    for label, ndata in [('ELG', 6), ('LRG', 6), ('ELG_LRG', 18)]:
        correlation = np.eye(ndata) * 0.8 + np.ones((ndata, ndata)) * 0.2
        np.savetxt(corr_root / f'Mcorr_small_box_z0.95_wp+xi_{label}.txt', correlation)
    return root, corr_root.parent, loaded


@pytest.mark.parametrize('tracers, markers, pairs', [
    (['ELG'], [100.], ['ELG_ELG']),
    (['LRG'], [300.], ['LRG_LRG']),
    (['LRG', 'ELG'], [100., 200., 300.], ['ELG_ELG', 'ELG_LRG', 'LRG_LRG']),
])
def test_hybrid_covariance_and_pair_statistic_order(measured_inputs, tracers, markers, pairs):
    data_dir, correlation_dir, cuts = measured_inputs
    inputs = example.load_fit_data(tracers, data_dir, correlation_dir)
    expected = np.concatenate([marker + np.arange(1., 7.) for marker in markers])
    sigma = np.tile(np.arange(2., 8.), len(markers))
    assert inputs.pairs == pairs
    np.testing.assert_array_equal(inputs.vector, expected)
    np.testing.assert_array_equal(inputs.sigma, sigma)
    np.testing.assert_allclose(inputs.covariance, inputs.correlation * np.outer(sigma, sigma))
    np.testing.assert_allclose(np.diag(inputs.covariance), sigma**2)
    assert inputs.hartlap_deficit == pytest.approx((len(expected) + 1) / 1882)
    for offset in range(0, len(cuts), 3):
        assert cuts[offset:offset+3] == [('rppi', slice(None, None, 2)),
                                         ('rppi', slice(4, -3)), ('smu', slice(16, -6))]


def test_covariance_dimension_mismatch_is_not_cropped(measured_inputs):
    data_dir, correlation_dir, _ = measured_inputs
    wrong = correlation_dir / 'wrong.txt'
    np.savetxt(wrong, np.eye(20))
    with pytest.raises(ValueError, match='No matrix cropping'):
        example.load_fit_data(['ELG'], data_dir, correlation_dir, correlation_file=wrong)
    with pytest.raises(ValueError, match='nmocks must exceed'):
        example.load_fit_data(['ELG'], data_dir, correlation_dir, nmocks=8)


def test_requested_model_physics_and_priors(tmp_path):
    elg = example.model_configuration(['ELG'], tmp_path)
    assert elg['use_assembly_bias'] is False
    assert elg['ELG']['HOD_model'] == 'mHMQ'
    assert elg['ELG']['conformity_bias'] is True
    assert elg['ELG']['exp_frac'] > 0
    assert elg['fit_param']['priors']['ELG']['exp_frac'] == [0., 1.]
    joint = example.model_configuration(['ELG', 'LRG'], tmp_path)
    assert joint['tracers'] == ['LRG', 'ELG']
    assert joint['use_assembly_bias'] is True
    assert joint['ELG']['assembly_bias'] == {'c': [0., 0.]}
    assert 'assembly_bias' not in joint['fit_param']['priors']['ELG']
    assert joint['LRG']['HOD_model'] == 'SHOD'
    assert joint['LRG']['link_sat_to_central'] is True
    assert 'f_sigv' in joint['fit_param']['priors']['LRG']
    assert 'f_vcen' in joint['fit_param']['priors']['LRG']
    assert joint['fit_param']['priors']['LRG']['assembly_bias']['c'] == [[-1., 1.], [-1., 1.]]


def toy_inputs(tracers):
    from HODDIES import HOD
    pairs = ['_'.join(pair) for pair in HOD.get_comb_tr_list(tracers)]
    rp_edges, s_edges = np.geomspace(.3, 8., 4), np.geomspace(.3, 8., 5)
    rp, s = np.sqrt(rp_edges[:-1] * rp_edges[1:]), np.sqrt(s_edges[:-1] * s_edges[1:])
    observations = {'wp': {}, 'xi_ells': {}}
    errors = {'wp': {}, 'xi_ells': {}}
    for pair in pairs:
        observations['wp'][pair] = (rp, np.zeros(3))
        observations['xi_ells'][pair] = (s, np.zeros((2, 4)))
        errors['wp'][pair] = np.ones(3)
        errors['xi_ells'][pair] = np.ones((2, 4))
    ndata = 11 * len(pairs)
    clustering = dict(rsd=False, los='z',
                      xi_rppi=dict(pimax=4, edges_rppi=(rp_edges, np.arange(-4., 5.))),
                      xi_smu=dict(multipole_index=[0, 2], edges_smu=(s_edges, np.linspace(-1., 1., 21))))
    return example.FitData(observations, errors, pairs, np.zeros(ndata), np.ones(ndata),
                           np.eye(ndata), np.eye(ndata), clustering, {}, Path('synthetic.txt'), .05)


def toy_simulation():
    from HODDIES.sim_loader import Base_catalogue
    class ToyCatalogue(Base_catalogue):
        def cosmo(self, **kwargs):
            return None
    rng = np.random.default_rng(42)
    size = 12_000
    masses = rng.uniform(11., 14.8, size)
    data = {axis: rng.uniform(0., 100., size) for axis in 'xyz'}
    data.update({f'v{axis}': rng.normal(0., 150., size) for axis in 'xyz'})
    data.update(Mh=10**masses, Rh=np.full(size, 400.), c=rng.uniform(3., 10., size),
                Vrms=np.full(size, 200.), halo_id=np.arange(size, dtype='i8'))
    return ToyCatalogue(data, boxsize=100., z_simu=.95, nthreads=2, setup_logger=False)


@pytest.mark.parametrize('tracers', [['ELG'], ['LRG'], ['LRG', 'ELG']])
def test_current_optimizer_and_best_point_outputs(tmp_path, monkeypatch, tracers):
    from HODDIES import HOD
    from HODDIES import sim_loader
    from HODDIES.fit_functions import flatten_stats
    simulation = toy_simulation()
    inputs = toy_inputs(tracers)
    configuration = example.model_configuration(tracers, tmp_path, nthreads=2)
    hod = example.build_hod(simulation, configuration, inputs)
    assert set(hod.args['fit_param']['priors']) == set(tracers)
    np.testing.assert_allclose(hod.inv_cov2, np.eye(inputs.vector.size) * .95)
    if 'LRG' in tracers:
        assert 'ab_c' in simulation.columns
        assert hod.args['LRG']['assembly_bias'].keys() == {'c'}
    if 'ELG' in tracers:
        assert hod.args['ELG']['assembly_bias'] == {'c': [0., 0.]}
    initial = example.initial_parameters(hod)
    mock = hod.make_mock_cat(fix_seed=10, verbose=False)
    model = hod.compute_stats(mock, stat=example.FIT_STATS, tracers=tracers)
    assert flatten_stats(model, example.FIT_STATS, inputs.pairs).shape == inputs.vector.shape
    assert np.isfinite(flatten_stats(model, example.FIT_STATS, inputs.pairs)).all()

    # The real CLI path runs the real HOD/Stochopy optimizer; only inputs are small.
    monkeypatch.setattr(sim_loader, 'AbacusSummitSim', lambda **kwargs: simulation)
    monkeypatch.setattr(example, 'load_fit_data', lambda *args, **kwargs: inputs)
    plot_options = ['--no-plot'] if tracers == ['LRG'] else []
    native_plot_stats = HOD.plot_stats
    plotted = []

    def record_plot_stats(self, *args, **kwargs):
        fig = native_plot_stats(self, *args, **kwargs)
        plotted.append((fig, kwargs['data']))
        return fig

    monkeypatch.setattr(HOD, 'plot_stats', record_plot_stats)
    ftol_options = ['--ftol', '1e-7'] if tracers == ['ELG'] else []
    expected_ftol = 1e-7 if ftol_options else inputs.vector.size / 2.0
    assert example.main(['--tracers', *tracers, '--output', str(tmp_path),
                         '--nthreads', '2', '--maxiter', '1', '--popsize', '4',
                         '--report', str(tmp_path / 'fit_report.md'),
                         *ftol_options, *plot_options]) == 0
    result = np.load(tmp_path / 'best_fit.npy', allow_pickle=True).item()
    assert result['param_fit']['fit_param']['minimizer']['ftol'] == expected_ftol
    saved = np.load(tmp_path / 'best_fit_statistics.npz')
    assert np.isfinite(result['fun']) and np.isfinite(saved['chi2'])
    np.testing.assert_allclose(saved['chi2'], result['fun'], rtol=1e-10)
    if not plot_options:
        assert (tmp_path / 'best_fit.png').is_file()
        assert len(plotted) == 1
        fig, plot_data = plotted[0]
        assert fig._plot_stats_stats == ['wp', 'xi0', 'xi2']
        assert len(fig._plot_stats_axes) == len(inputs.pairs)
        assert len(fig._plot_stats_drawn) == 3 * len(inputs.pairs)
        assert all(ax is not None for _, residual_axes in fig._plot_stats_axes for ax in residual_axes)
        for statistic in example.FIT_STATS:
            for pair in inputs.pairs:
                np.testing.assert_array_equal(plot_data[statistic][pair][2], inputs.errors[statistic][pair])
    else:
        assert not plotted
    np.testing.assert_array_equal(saved['parameters'], result['x'])
    assert saved['model_vector'].shape == inputs.vector.shape
    assert (tmp_path / 'fit_inputs.npz').is_file()
    assert (tmp_path / 'configuration.json').is_file()
    assert (tmp_path / 'fit_summary.json').is_file()
    assert len(list((tmp_path / 'run_history').glob('*.json'))) == 1
    assert (tmp_path / 'fit_report.md').is_file()


@pytest.fixture
def low_redshift_inputs(measured_inputs, tmp_path):
    data_dir, correlation_dir, _ = measured_inputs
    # The estimator fixture above remains active; add low-redshift file names.
    for mode in ('rppi', 'smu'):
        (data_dir / mode / 'allcounts_LRG_GCcomb_0.4_0.6_default_angular_bitwise_log_njack128_nran4_split20.npy').touch()
    corr = correlation_dir / 'z0.500'
    corr.mkdir()
    np.savetxt(corr / 'Mcorr_small_box_z0.5_wp+xi_LRG.txt', np.eye(6) * .8 + .2)
    csv = tmp_path / 'lensing.csv'
    np.savetxt(csv, [[.1, 10., 1.], [1., 5., 2.], [10., 2., 3.], [100., np.nan, np.nan]],
               delimiter=',', header='rp,ds_obs_lrg1,ds_err_lrg1', comments='')
    return data_dir, correlation_dir, csv


def test_lrg_lensing_uses_low_bin_and_block_precision(low_redshift_inputs):
    data_dir, correlation_dir, csv = low_redshift_inputs
    inputs = example.load_fit_data(['LRG'], data_dir, correlation_dir,
                                   with_delta_sigma=True, delta_sigma_file=csv)
    assert inputs.z_simu == .5 and inputs.redshift_bin == (.4, .6)
    assert inputs.statistics == ['wp', 'xi_ells', 'delta_sigma']
    assert inputs.correlation_file.name == 'Mcorr_small_box_z0.5_wp+xi_LRG.txt'
    np.testing.assert_array_equal(inputs.vector, [301, 302, 303, 304, 305, 306, 10, 5, 2])
    np.testing.assert_array_equal(inputs.covariance[6:, 6:], np.diag([1., 4., 9.]))
    np.testing.assert_array_equal(inputs.covariance[:6, 6:], 0.)
    np.testing.assert_allclose(inputs.precision[:6, :6], np.linalg.inv(inputs.covariance[:6, :6]) * (1 - 7/1882))
    np.testing.assert_allclose(inputs.precision[6:, 6:], np.diag([1., 1/4, 1/9]))
    assert inputs.hartlap_deficit == pytest.approx(7/1882)  # Only six clustering bins.
    assert inputs.lensing['excluded_rows'] == [3]
    edges = inputs.clustering['delta_sigma']['edges_rp']
    np.testing.assert_array_equal(edges, np.logspace(-1, 1.6, 13))
    np.testing.assert_array_equal(inputs.clustering['delta_sigma']['rp_eval'], [.1, 1., 10.])
    assert all('0.4_0.6' in path for mode, path in inputs.files['LRG_LRG'].items() if mode != 'delta_sigma')


@pytest.mark.parametrize('tracers, z', [(['ELG'], .5), (['LRG', 'ELG'], .5), (['LRG'], .95)])
def test_lensing_rejects_other_tracers_and_redshifts(tracers, z):
    with pytest.raises(ValueError, match='delta_sigma is available only'):
        example.load_fit_data(tracers, with_delta_sigma=True, z_simu=z)
    with pytest.raises(ValueError, match='delta_sigma is available only'):
        example.main(['--tracers', *tracers, '--with-delta-sigma', '--z-simu', str(z), '--check-data'])


@pytest.mark.parametrize('values, message', [
    ([[.1, 1., 1.], [1., 2., 0.]], 'strictly positive'),
    ([[.1, 1., 1.], [1., np.nan, np.nan], [10., 3., 1.]], 'inside the radial range'),
    ([[.1, 1., 1.], [1., 2., 1.], [40., 3., 1.]], 'configured radial edges'),
])
def test_invalid_lensing_measurements_rejected(tmp_path, values, message):
    csv = tmp_path / 'invalid.csv'
    np.savetxt(csv, values, delimiter=',', header='rp,ds_obs_lrg1,ds_err_lrg1', comments='')
    with pytest.raises(ValueError, match=message):
        example.load_delta_sigma(csv)


def test_lensing_check_data_never_loads_halos(low_redshift_inputs, tmp_path, monkeypatch):
    import json
    from HODDIES import sim_loader
    data_dir, correlation_dir, csv = low_redshift_inputs
    monkeypatch.setattr(sim_loader, 'AbacusSummitSim', lambda **kwargs: pytest.fail('check-data loaded halos'))
    output = tmp_path / 'checked'
    assert example.main(['--tracers', 'LRG', '--with-delta-sigma', '--data-dir', str(data_dir),
                         '--correlation-dir', str(correlation_dir), '--delta-sigma-file', str(csv),
                         '--output', str(output), '--nthreads', '2', '--check-data']) == 0
    manifest = json.loads((output / 'configuration.json').read_text())
    assert manifest['simulation']['z_simu'] == .5
    assert manifest['simulation']['load_field_particles'] is True
    assert manifest['redshift_bin'] == [.4, .6]
    assert manifest['statistic_order'] == ['wp', 'xi0', 'xi2', 'delta_sigma']
    assert manifest['lensing']['precision_correction'] == 'clustering block only'
    with np.load(output / 'fit_inputs.npz') as saved:
        assert saved['data_vector'].shape == (9,)
        assert saved['sigma_jk'].shape == (6,)
        np.testing.assert_array_equal(saved['LRG_LRG_delta_sigma_y'], [10., 5., 2.])
        assert saved['precision_matrix'].shape == (9, 9)
    assert not (output / 'best_fit.npy').exists()


def test_lensing_optimizer_saved_outputs_and_report(low_redshift_inputs, tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from HODDIES import hod as hod_module
    from HODDIES import sim_loader
    from HODDIES.fit_functions.scripts import summarize_desi_y1_fits as reporting
    data_dir, correlation_dir, csv = low_redshift_inputs
    simulation = toy_simulation()
    simulation.z_simu = .5
    simulation.field_particles = {'pos': np.ones((200, 3))}
    simulation.cosmo = lambda **kwargs: SimpleNamespace(rho_m=lambda z: 1.)
    simulation_options, plotted = [], []
    def catalogue(**kwargs):
        simulation_options.append(kwargs)
        return simulation
    monkeypatch.setattr(sim_loader, 'AbacusSummitSim', catalogue)
    native_load = example.load_fit_data
    def load(*args, **kwargs):
        inputs = native_load(*args, **kwargs)
        # Periodic toy catalogue: no cosmological RSD conversion required.
        inputs.clustering['rsd'] = False
        return inputs
    monkeypatch.setattr(example, 'load_fit_data', load)
    def estimate(positions, particles, rbins, rp_eval=None, **kwargs):
        np.testing.assert_array_equal(rbins, np.logspace(-1, 1.6, 13))
        x = np.asarray(rp_eval) if rp_eval is not None else np.sqrt(rbins[:-1] * rbins[1:])
        return x, np.full(x.size, np.mean(positions[:, 0]) / 10.)
    monkeypatch.setattr(hod_module, 'compute_delta_sigma', estimate)
    native_plot = hod_module.HOD.plot_stats
    def plot(self, *args, **kwargs):
        fig = native_plot(self, *args, **kwargs)
        plotted.append(fig)
        return fig
    monkeypatch.setattr(hod_module.HOD, 'plot_stats', plot)
    output, report = tmp_path / 'fit', tmp_path / 'report.md'
    assert example.main(['--tracers', 'LRG', '--with-delta-sigma', '--data-dir', str(data_dir),
                         '--correlation-dir', str(correlation_dir), '--delta-sigma-file', str(csv),
                         '--output', str(output), '--report', str(report), '--nthreads', '2',
                         '--maxiter', '1', '--popsize', '4', '--ftol', '0']) == 0
    assert simulation_options[0]['load_field_particles'] is True
    assert simulation_options[0]['z_simu'] == .5
    result = np.load(output / 'best_fit.npy', allow_pickle=True).item()
    with np.load(output / 'best_fit_statistics.npz') as saved, np.load(output / 'fit_inputs.npz') as inputs:
        np.testing.assert_allclose(saved['chi2'], result['fun'], rtol=1e-10)
        residual = inputs['data_vector'] - saved['model_vector']
        np.testing.assert_allclose(saved['chi2'], residual @ inputs['precision_matrix'] @ residual)
        assert saved['LRG_LRG_delta_sigma_y'].shape == (3,)
    assert plotted[0]._plot_stats_stats == ['wp', 'xi0', 'xi2', 'delta_sigma']
    summary = json.loads((output / 'fit_summary.json').read_text())
    assert summary['binning'][-1][:3] == ['LRG_LRG', 'delta_sigma', 3]
    content = report.read_text()
    assert 'Lensing measurement' in content and 'diagonal CSV errors squared' in content
    assert 'clustering block only' in content
    assert 'best-fit wp, xi0, xi2, delta_sigma' in content
    # Reporting an input-only lensing run draws all four measurement panels too.
    summary['chi2'] = None
    assert reporting.data_plot(summary).is_file()
