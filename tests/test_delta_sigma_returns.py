"""Lensing result collection for single- and multiple-tracer fit requests."""
from types import SimpleNamespace

import numpy as np
import pytest

from HODDIES import hod as hod_module
from HODDIES.fit_functions import flatten_stats


@pytest.fixture
def lensing(monkeypatch):
    hod = hod_module.HOD.__new__(hod_module.HOD)
    hod.boxsize, hod.nthreads = 100., 1
    hod.cosmo = SimpleNamespace(rho_m=lambda z: 1.)
    hod.field_particles = {'pos': np.ones((100, 3))}
    hod.args = {'clustering_settings': {'los': 'z', 'delta_sigma': {
        'edges_rp': np.array([.5, 1.5, 2.5]), 'pimax': 40}}}
    cat = np.array([('LRG', 1., 0., 0.), ('ELG', 2., 0., 0.)],
                   dtype=[('TRACER', 'U3'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8')])
    # Replace only the particle estimator; exercise real tracer selection and returns.
    def estimate(positions, particles, **kwargs):
        return np.array([1., 2.]), np.full(2, positions[0, 0])
    monkeypatch.setattr(hod_module, 'compute_delta_sigma', estimate)
    return hod, cat


@pytest.mark.parametrize('tracers', [['LRG'], ['ELG', 'LRG']])
@pytest.mark.parametrize('return_dic', [False, True])
def test_return_modes_preserve_each_tracer(lensing, tracers, return_dic):
    hod, cat = lensing
    result = hod.get_delta_sigma(cat, tracers=tracers, return_dic=return_dic, verbose=False)
    if len(tracers) == 1 and not return_dic:
        rp, ds = result
        np.testing.assert_array_equal(rp, [1., 2.])
        np.testing.assert_array_equal(ds, [1., 1.])
    else:
        assert list(result) == [f'{tr}_{tr}' for tr in tracers]
        for tr in tracers:
            np.testing.assert_array_equal(result[f'{tr}_{tr}'][0], [1., 2.])
            np.testing.assert_array_equal(result[f'{tr}_{tr}'][1], [1., 1.] if tr == 'LRG' else [2., 2.])


@pytest.mark.parametrize('tracers, pairs, expected', [
    (['LRG'], ['LRG_LRG'], [1., 1.]),
    (['LRG', 'ELG'], ['ELG_ELG', 'ELG_LRG', 'LRG_LRG'], [2., 2., 1., 1.]),
])
def test_fit_statistics_get_lensing_auto_pairs(lensing, tracers, pairs, expected):
    hod, cat = lensing
    result = hod.compute_stats(cat, stat=['delta_sigma'], tracers=tracers, verbose=False)
    np.testing.assert_array_equal(flatten_stats(result, ['delta_sigma'], pairs), expected)
    assert 'ELG_LRG' not in result['delta_sigma']


def test_missing_particles_still_raises(lensing):
    hod, cat = lensing
    hod.field_particles = None
    with pytest.raises(ValueError, match='Particle subsample'):
        hod.get_delta_sigma(cat, tracers=['LRG'], return_dic=True, verbose=False)


def test_requested_edges_with_thirteen_measurement_radii(lensing, monkeypatch):
    hod, cat = lensing
    edges = np.logspace(-1, 1.6, 13)
    radii = np.logspace(-.9, 1.5, 13)
    hod.args['clustering_settings']['delta_sigma'].update(edges_rp=edges, rp_eval=radii)
    def estimate(positions, particles, rbins, rp_eval=None, **kwargs):
        np.testing.assert_array_equal(rbins, edges)
        np.testing.assert_array_equal(rp_eval, radii)
        return np.asarray(rp_eval), np.ones(len(rp_eval))
    monkeypatch.setattr(hod_module, 'compute_delta_sigma', estimate)
    result = hod.compute_stats(cat, stat=['delta_sigma'], tracers=['LRG'])
    np.testing.assert_array_equal(result['delta_sigma']['LRG_LRG'][0], radii)
    assert flatten_stats(result, ['delta_sigma'], ['LRG_LRG']).shape == (13,)


def test_estimator_evaluates_fine_grid_at_measurement_radii(monkeypatch):
    from HODDIES.estimators import clustering_statistics as clustering
    edges = np.logspace(-1, 1.6, 13)
    radii = np.logspace(-.9, 1.5, 13)
    def correlation(*args, edges, **kwargs):
        rp = np.sqrt(edges[0][:-1] * edges[0][1:])
        return lambda **options: 1. / rp
    monkeypatch.setattr(clustering, 'TwoPointCorrelationFunction', correlation)
    # For wp=1/r, enclosed averaging gives DeltaSigma=1/r when rho_m/1e12=1.
    rp, ds = clustering.compute_delta_sigma(np.ones((2, 3)), np.ones((2, 3)),
                                            100., edges, 1e12, rp_eval=radii, n_int=2000)
    np.testing.assert_array_equal(rp, radii)
    np.testing.assert_allclose(ds, 1. / radii, rtol=5e-5)
    # Callers without rp_eval retain the usual 12 geometric output centres.
    rp, _ = clustering.compute_delta_sigma(np.ones((2, 3)), np.ones((2, 3)),
                                           100., edges, 1e12, n_int=2000)
    np.testing.assert_array_equal(rp, np.sqrt(edges[:-1] * edges[1:]))


@pytest.mark.parametrize('radii', [[], [.01, 1.], [1., 40.], [1., .5], [1., np.nan]])
def test_estimator_rejects_invalid_evaluation_radii(radii):
    from HODDIES.estimators.clustering_statistics import compute_delta_sigma
    with pytest.raises(ValueError, match='rp_eval'):
        compute_delta_sigma(np.ones((2, 3)), np.ones((2, 3)), 100.,
                            np.logspace(-1, 1.6, 13), 1e12, rp_eval=radii)
