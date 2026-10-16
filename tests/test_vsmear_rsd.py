"""One sampled redshift-error realization follows a catalogue through statistics."""

from types import SimpleNamespace

import numba
import numpy as np
import pytest
from mpytools import Catalog

from HODDIES import HOD, vsmear_func
from HODDIES.utils import apply_rsd
import HODDIES.hod as hod_module


def make_catalogue(tracers=('LRG', 'ELG')):
    size = 4 * len(tracers)
    return Catalog.from_dict({
        'TRACER': np.repeat(tracers, 4),
        'x': np.concatenate([10. * i + np.arange(1., 5.) for i in range(len(tracers))]),
        'y': np.arange(size, dtype=float) + 5.,
        'z': np.arange(size, dtype=float) + 20.,
        'vx': np.arange(size, dtype=float) + 10.,
        'vy': np.arange(size, dtype=float) - 10.,
        'vz': np.arange(size, dtype=float) + 50.,
    })


def make_hod(monkeypatch, tracers=('LRG', 'ELG')):
    hod = HOD.__new__(HOD)
    hod._get_default_parameters()
    hod.args['tracers'] = list(tracers)
    for tracer in tracers:
        hod.args[tracer] = {'vsmear': {
            'model': 'gaussian' if tracer == 'LRG' else 'lorentzian', 'dv': 4.,
        }}
    hod.args['clustering_settings']['CIC']['max_count'] = 3
    hod.rng = np.random.RandomState(42)
    hod.cosmo = SimpleNamespace(efunc=lambda z: 2.)
    hod.z_simu, hod.boxsize, hod.H_0, hod.nthreads = .5, 100., 100., 1
    hod.logger = SimpleNamespace(info=lambda *args: None, warning=lambda *args: None)
    draws, positions = [], []

    for model in ('gaussian', 'lorentzian'):
        original = getattr(vsmear_func, model)

        def sampler(*, size, dv, rng, _model=model, _original=original, **params):
            draws.append((_model, size))
            return _original(size=size, dv=dv, rng=rng, **params)

        monkeypatch.setattr(vsmear_func, model, sampler)

    def record(mode, pos1, pos2=None):
        positions.append((mode, [np.array(p, copy=True) for p in pos1],
                          None if pos2 is None else [np.array(p, copy=True) for p in pos2]))

    def compute_twopoint(pos1, mode, *args, pos2=None, **kwargs):
        record(mode, pos1, pos2)

        def project(*, return_sep, ells=None, pimax=None):
            sep = np.array([1., 2., 3.])
            if ells is None and pimax is None:
                return sep, np.array([-.5, .5]), np.ones((3, 2))
            values = np.ones(3) if ells is None or np.isscalar(ells) else np.ones((len(ells), 3))
            return sep, values

        return project

    def compute_power_spectrum(*, pos1, pos2=None, **kwargs):
        record('power', pos1, pos2)
        return SimpleNamespace(poles=lambda **kwargs: (np.array([.1, .2]), np.ones((2, 2))))

    def compute_cic(x, y, z, *args, **kwargs):
        record('CIC', [x, y, z])
        return np.ones(x.size, dtype=int)

    monkeypatch.setattr(hod_module, 'compute_twopoint', compute_twopoint)
    monkeypatch.setattr(hod_module, 'compute_power_spectrum', compute_power_spectrum)
    monkeypatch.setattr(hod_module, 'compute_CIC', compute_cic)
    return hod, make_catalogue(tracers), draws, positions


def assert_positions_match_catalogue(hod, cat, calls, *, rsd=True):
    expected = {}
    for tracer in np.unique(cat['TRACER']):
        subset = cat[cat['TRACER'] == tracer]
        expected[subset['x'][0]] = (
            apply_rsd(subset, hod.z_simu, hod.boxsize, hod.cosmo, hod.H_0,
                      vsmear=subset['vsmear']) if rsd else
            [subset[axis] % hod.boxsize for axis in 'xyz'])
    for _, pos1, pos2 in calls:
        for pos in (pos1, pos2):
            if pos is not None:
                for actual, wanted in zip(pos, expected[pos[0][0]]):
                    np.testing.assert_array_equal(actual, wanted)
        if pos2 is not None and pos1[0][0] == pos2[0][0]:
            np.testing.assert_array_equal(pos1, pos2)


@pytest.mark.parametrize('los', ['x', 'y', 'z'])
def test_apply_rsd_uses_velocity_offsets_once_without_changing_catalogue(los):
    cat = make_catalogue(('LRG',))
    original = cat.deepcopy()
    errors = np.array([-5., 0., 20., 20000.])
    factor = 1.5 / (100. * 2.)
    pos = apply_rsd(cat, .5, 100., SimpleNamespace(efunc=lambda z: 2.),
                    los=los, vsmear=errors)
    for axis, actual in zip('xyz', pos):
        expected = cat[axis]
        if axis == los:
            expected = expected + (cat['v' + axis] + errors) * factor
        np.testing.assert_allclose(actual, expected % 100., rtol=1.e-14, atol=1.e-13)
    for column in cat.columns():
        np.testing.assert_array_equal(cat[column], original[column])


def test_apply_rsd_scalar_and_none_are_deterministic_offsets():
    cat, cosmo = make_catalogue(('LRG',)), SimpleNamespace(efunc=lambda z: 2.)
    np.testing.assert_array_equal(apply_rsd(cat, .5, 100., cosmo, vsmear=None),
                                  apply_rsd(cat, .5, 100., cosmo, vsmear=0.))
    np.testing.assert_array_equal(apply_rsd(cat, .5, 100., cosmo, vsmear=7.),
                                  apply_rsd(cat, .5, 100., cosmo, vsmear=np.full(cat.size, 7.)))
    with pytest.raises((TypeError, ValueError), match='vsmear|get_vsmear'):
        apply_rsd(cat, .5, 100., cosmo, vsmear={'model': 'gaussian', 'dv': 7.})


def test_all_statistics_and_auto_cross_pairs_share_the_same_errors(monkeypatch):
    hod, cat, draws, calls = make_hod(monkeypatch)
    hod.compute_stats(cat, stat=['wp', 'xi0', 'power_spectrum', 'CIC'],
                      tracers=['LRG', 'ELG'], verbose=False)
    assert draws == [('gaussian', 4), ('lorentzian', 4)]
    assert set(mode for mode, _, _ in calls) == {'rppi', 'smu', 'power', 'CIC'}
    for mode in ('rppi', 'smu', 'power'):
        # Estimators must use their autocorrelation path, including shot noise.
        assert [pos2 is None for name, _, pos2 in calls if name == mode] == [True, False, True]
    assert np.issubdtype(cat['vsmear'].dtype, np.floating)
    assert_positions_match_catalogue(hod, cat, calls)


def test_individual_statistic_methods_reuse_catalogue_errors(monkeypatch):
    hod, cat, draws, calls = make_hod(monkeypatch)
    for method in (hod.get_wp, hod.get_xiells, hod.get_PS, hod.get_CIC):
        method(cat, tracers=['LRG', 'ELG'], verbose=False)
    assert draws == [('gaussian', 4), ('lorentzian', 4)]
    assert_positions_match_catalogue(hod, cat, calls)


def test_cross_correlation_wrappers_reuse_errors_and_project_estimators(monkeypatch):
    hod, cat, draws, calls = make_hod(monkeypatch)
    for method in (hod.get_cross_wp, hod.get_cross_xiells):
        result = method(cat, tracers=['LRG', 'ELG'], verbose=False)
        assert set(result) == {'LRG_LRG', 'LRG_ELG', 'ELG_ELG'}
        for separation, statistic in result.values():
            assert separation.shape == (3,)
            assert statistic.shape[-1] == 3
    assert draws == [('gaussian', 4), ('lorentzian', 4)]
    assert_positions_match_catalogue(hod, cat, calls)


def test_config_change_refreshes_only_the_affected_tracer(monkeypatch):
    hod, cat, draws, _ = make_hod(monkeypatch)
    hod.assign_vsmear(cat, tracers=['LRG', 'ELG'])
    previous = cat['vsmear'].copy()
    hod.args['LRG']['vsmear']['dv'] = 20.
    hod.assign_vsmear(cat, tracers=['LRG', 'ELG'])
    mask = cat['TRACER'] == 'LRG'
    assert draws == [('gaussian', 4), ('lorentzian', 4), ('gaussian', 4)]
    assert not np.array_equal(cat['vsmear'][mask], previous[mask])
    np.testing.assert_array_equal(cat['vsmear'][~mask], previous[~mask])


def test_assigning_a_second_tracer_does_not_reuse_placeholder_zeros(monkeypatch):
    hod, cat, draws, _ = make_hod(monkeypatch)
    hod.get_wp(cat, tracers='LRG', verbose=False)
    hod.get_xiells(cat, tracers='ELG', verbose=False)
    assert draws == [('gaussian', 4), ('lorentzian', 4)]
    for tracer in ('LRG', 'ELG'):
        assert np.any(cat['vsmear'][cat['TRACER'] == tracer] != 0.)


def test_nested_custom_parameter_change_refreshes_cached_errors(monkeypatch):
    hod, cat, _, _ = make_hod(monkeypatch, tracers=('LRG',))
    draws = []

    def custom(*, size, dv, rng, offset):
        draws.append(offset)
        return rng.normal(offset, dv, size=size)

    monkeypatch.setattr(vsmear_func, 'custom', custom, raising=False)
    hod.args['LRG']['vsmear'] = {'model': 'custom', 'dv': 1., 'params': {'offset': 2.}}
    hod.assign_vsmear(cat)
    hod.assign_vsmear(cat)
    hod.args['LRG']['vsmear']['params']['offset'] = 3.
    hod.assign_vsmear(cat)
    assert draws == [2., 3.]


def test_explicit_rng_and_force_control_resampling(monkeypatch):
    hod, cat, draws, _ = make_hod(monkeypatch, tracers=('LRG',))
    expected_rng = np.random.RandomState(5)
    rng = np.random.RandomState(5)
    hod.assign_vsmear(cat, rng=rng)
    np.testing.assert_array_equal(cat['vsmear'], expected_rng.normal(0., 4., cat.size))
    hod.assign_vsmear(cat, rng=rng)
    assert len(draws) == 1
    hod.assign_vsmear(cat, rng=rng, force=True)
    np.testing.assert_array_equal(cat['vsmear'], expected_rng.normal(0., 4., cat.size))
    assert len(draws) == 2
    np.testing.assert_array_equal(hod.rng.random_sample(5), np.random.RandomState(42).random_sample(5))


def test_catalogue_slice_keeps_matching_errors_without_redrawing(monkeypatch):
    hod, cat, draws, calls = make_hod(monkeypatch)
    hod.assign_vsmear(cat, tracers=['LRG', 'ELG'])
    mask = np.array([True, False, True, False, False, True, False, True])
    subset = cat[mask]
    hod.get_wp(subset, tracers=['LRG', 'ELG'], verbose=False)
    assert draws == [('gaussian', 4), ('lorentzian', 4)]
    np.testing.assert_array_equal(subset['vsmear'], cat['vsmear'][mask])
    assert_positions_match_catalogue(hod, subset, calls)


def test_existing_offsets_are_used_and_explicit_disable_clears_them(monkeypatch):
    hod, cat, draws, _ = make_hod(monkeypatch)
    existing = np.arange(cat.size, dtype=float) + 1.
    cat['vsmear'] = existing.copy()
    hod.assign_vsmear(cat, tracers='LRG')
    hod.assign_vsmear(cat, tracers='ELG')
    assert not draws
    np.testing.assert_array_equal(cat['vsmear'], existing)
    hod.args['LRG']['vsmear'] = None
    hod.args['ELG']['vsmear'] = {'model': 'lorentzian', 'dv': 0.}
    hod.assign_vsmear(cat, tracers=['LRG', 'ELG'])
    assert not draws
    np.testing.assert_array_equal(cat['vsmear'], np.zeros(cat.size))


def test_missing_vsmear_configuration_disables_errors_without_using_rng(monkeypatch):
    hod, cat, draws, _ = make_hod(monkeypatch, tracers=('LRG',))
    del hod.args['LRG']['vsmear']
    hod.assign_vsmear(cat)
    assert not draws
    np.testing.assert_array_equal(cat['vsmear'], np.zeros(cat.size))
    np.testing.assert_array_equal(hod.rng.random_sample(5), np.random.RandomState(42).random_sample(5))


@pytest.mark.parametrize('rsd, has_cosmology', [(False, True), (True, False)])
def test_real_space_paths_do_not_sample_velocity_errors(monkeypatch, rsd, has_cosmology):
    hod, cat, draws, calls = make_hod(monkeypatch)
    hod.args['clustering_settings']['rsd'] = rsd
    if not has_cosmology:
        hod.cosmo = None
    for method in (hod.get_wp, hod.get_xiells, hod.get_PS, hod.get_CIC):
        method(cat, tracers=['LRG', 'ELG'], verbose=False)
    hod.compute_stats(cat, stat=['wp', 'xi0', 'power_spectrum', 'CIC'],
                      tracers=['LRG', 'ELG'], verbose=False)
    assert not draws
    assert_positions_match_catalogue(hod, cat, calls, rsd=False)


def test_fixed_mock_seed_includes_velocity_error_realization():
    original_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        index = np.arange(16)
        halos = {'halo_id': index, 'log10_Mh': np.full(16, 13.),
                 'Mh': np.full(16, 1.e13), 'Rh': np.full(16, 350.), 'c': np.full(16, 6.)}
        for component, axis in enumerate('xyz'):
            halos[axis] = index.astype(float) + 5. * component
            halos['v' + axis] = 2. * index + component
        loader = SimpleNamespace(
            hcat=Catalog.from_dict(halos), part_subsamples=None, field_particles=None,
            z_simu=.5, boxsize=100., cosmo=lambda **kwargs: None,
        )
        hod = HOD(loader, tracers=['LRG'], seed=42, nthreads=1, setup_logger=False,
                  LRG={'HOD_model': 'SHOD', 'Ac': 1., 'log_Mcent': 10., 'sigma_M': .01,
                       'density': None, 'satellites': False,
                       'vsmear': {'model': 'lorentzian', 'dv': 12.}})
        first = hod.make_mock_cat(fix_seed=23, verbose=False)
        hod.rng.random_sample(100)
        second = hod.make_mock_cat(fix_seed=23, verbose=False)
        assert first.size == 16
        assert np.any(first['vsmear'] != 0.)
        for column in first.columns():
            np.testing.assert_array_equal(first[column], second[column])
    finally:
        numba.set_num_threads(original_threads)
