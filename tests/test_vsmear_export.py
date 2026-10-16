"""Best-fit exports retain sampled velocity errors and honor export settings."""

from types import SimpleNamespace

import numba
import numpy as np
import pytest
from mpytools import Catalog

from HODDIES import HOD


@pytest.mark.parametrize('model', ['gaussian', 'lorentzian'])
@pytest.mark.parametrize('use_vsmear', [False, True])
@pytest.mark.parametrize('fit_type', ['wp', 'xi'])
def test_best_fit_export_reuses_errors_and_filters_only_written_columns(
        tmp_path, monkeypatch, model, use_vsmear, fit_type):
    hod = HOD.__new__(HOD)
    hod.args = {'tracers': ['LRG'],
                'LRG': {'vsmear': {'model': model, 'dv': 12.}},
                'fit_param': {'fit_type': fit_type, 'use_vsmear': use_vsmear}}
    hod.rng = np.random.RandomState(42)
    hod.result_fit = {'x': np.array([.5])}
    cat = Catalog.from_dict({
        'TRACER': np.full(5, 'LRG'), 'x': np.arange(5.),
        'y': np.arange(5.) + 10., 'z': np.arange(5.) + 20.,
        'vz': np.arange(5.) + 30.,
    })
    hod.assign_vsmear(cat)
    expected_errors = cat['vsmear'].copy()
    expected_rng = np.random.RandomState()
    expected_rng.set_state(hod.rng.get_state())
    monkeypatch.setattr(hod, 'get_param_and_prior', lambda: (['Ac_LRG'], [[0., 1.]]))
    monkeypatch.setattr(hod, 'update_new_param', lambda *args, **kwargs: None)
    monkeypatch.setattr(hod, 'make_mock_cat', lambda **kwargs: cat)
    used_errors = []

    def get_cross_wp(mock_cat, **kwargs):
        used_errors.append(mock_cat['vsmear'].copy())
        return {'LRG_LRG': (np.arange(3.), np.ones(3))}

    def unexpected_redraw(*args, **kwargs):
        pytest.fail('Best-fit export must reuse the realization used by statistics')

    method = 'get_cross_wp' if fit_type == 'wp' else 'get_cross_xiells'
    monkeypatch.setattr(hod, method, get_cross_wp)
    monkeypatch.setattr(hod, 'get_vsmear', unexpected_redraw)
    filename = tmp_path / 'best_fit.fits'
    result = hod.compute_bf_corr(fix_seed=17, save_bf_cat=str(filename))

    assert fit_type in result
    np.testing.assert_array_equal(used_errors, [expected_errors])
    np.testing.assert_array_equal(cat['vsmear'], expected_errors)
    np.testing.assert_array_equal(hod.rng.random_sample(10), expected_rng.random_sample(10))
    saved = Catalog.read(str(filename))
    assert ('vsmear' in saved.columns()) == use_vsmear
    if use_vsmear:
        np.testing.assert_array_equal(saved['vsmear'], expected_errors)
    for column in ['TRACER', 'x', 'y', 'z', 'vz']:
        np.testing.assert_array_equal(saved[column], cat[column])


@pytest.mark.parametrize('spec', [None, {'model': 'lorentzian', 'dv': 12.}])
def test_zero_occupation_mock_still_returns_empty_catalogue(spec):
    original_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        index = np.arange(4)
        halos = {'halo_id': index, 'log10_Mh': np.full(4, 13.),
                 'Mh': np.full(4, 1.e13), 'Rh': np.full(4, 350.), 'c': np.full(4, 6.)}
        for component, axis in enumerate('xyz'):
            halos[axis] = index.astype(float) + 5. * component
            halos['v' + axis] = 2. * index + component
        loader = SimpleNamespace(
            hcat=Catalog.from_dict(halos), part_subsamples=None, field_particles=None,
            z_simu=.5, boxsize=100., cosmo=lambda **kwargs: None,
        )
        hod = HOD(loader, tracers=['LRG'], seed=42, nthreads=1, setup_logger=False,
                  LRG={'HOD_model': 'SHOD', 'Ac': 0., 'log_Mcent': 10., 'sigma_M': .01,
                       'density': None, 'satellites': False, 'vsmear': spec})
        cat = hod.make_mock_cat(fix_seed=23, verbose=False)
        assert cat.size == 0
        assert 'vsmear' in cat.columns()
        assert cat['vsmear'].shape == (0,)
    finally:
        numba.set_num_threads(original_threads)
