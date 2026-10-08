"""Recover known tracer biases with the real multipole API and scipy fits."""

from types import SimpleNamespace

import cosmoprimo
import numpy as np
import pytest
from mpytools import Catalog

from HODDIES import HOD
import HODDIES.hod as hod_module


def linear_correlation(s, *, z):
    return (np.asarray(s) / 40.) ** -2 / (1. + z) ** 2


def make_bias_model(monkeypatch, tracers=('LRG', 'ELG'), *, existing_edges=True):
    expected = {tracer: {'LRG': 2.3, 'ELG': 1.25}[tracer] for tracer in tracers}
    hod = HOD.__new__(HOD)
    hod.init_logger(setup_logger=False)
    hod.cosmo, hod.z_simu, hod.boxsize, hod.nthreads = object(), .5, 500., 1
    smu = dict(bin_logscale=True, smin=.1, smax=30., n_s_bins=25,
               mu_max=1., n_mu_bins=101, multipole_index=[0, 2])
    if existing_edges:
        smu['edges_smu'] = (np.geomspace(.1, 30., 25), np.linspace(-1., 1., 11))
    hod.args = {'tracers': list(tracers),
                'clustering_settings': {'rsd': True, 'los': 'z', 'xi_smu': smu}}
    original_edges = smu.get('edges_smu')
    original_keys = set(smu)
    calls = {'mocks': 0, 'biases': []}
    cat = Catalog.from_dict({
        'TRACER': np.repeat(tracers, 4),
        'x': np.repeat(list(expected.values()), 4),
        'y': np.arange(4 * len(tracers), dtype=float),
        'z': np.zeros(4 * len(tracers)),
    })

    def make_mock_cat():
        calls['mocks'] += 1
        return cat

    def compute_twopoint(pos, mode, edges, boxsize, los, nthreads, **kwargs):
        assert not hod.args['clustering_settings']['rsd']
        assert mode == 'smu'
        assert edges[0][0] == 40. and edges[0][-1] == 80.
        assert np.all(pos[0] == pos[0][0])  # Only one tracer per autocorrelation.
        bias = float(pos[0][0])
        calls['biases'].append(bias)
        s = (edges[0][1:] + edges[0][:-1]) / 2.
        xi = bias**2 * linear_correlation(s, z=hod.z_simu)
        xi[0] = np.nan  # Empty pair-count bins must not spoil the fit.
        s[-1] = np.nan
        return lambda *, return_sep, ells: (s, xi)

    monkeypatch.setattr(hod, 'make_mock_cat', make_mock_cat)
    monkeypatch.setattr(hod_module, 'compute_twopoint', compute_twopoint)
    fourier = SimpleNamespace(pk_interpolator=lambda: SimpleNamespace(to_xi=lambda: linear_correlation))
    monkeypatch.setattr(cosmoprimo, 'Fourier', lambda cosmo, engine: fourier)

    def assert_restored():
        assert hod.args['clustering_settings']['rsd'] is True
        assert set(smu) == original_keys
        assert smu.get('edges_smu') is original_edges

    return hod, expected, calls, assert_restored


@pytest.mark.parametrize('tracers', [('LRG',), ('LRG', 'ELG')])
@pytest.mark.parametrize('existing_edges', [False, True])
def test_recovers_each_bias_and_restores_settings(monkeypatch, tracers, existing_edges):
    hod, expected, calls, assert_restored = make_bias_model(
        monkeypatch, tracers, existing_edges=existing_edges)
    result = hod.get_lin_bias()
    assert list(result) == list(tracers)
    assert all(isinstance(value, float) for value in result.values())
    for tracer, value in expected.items():
        assert result[tracer] == pytest.approx(value, rel=1.e-7)
    assert calls['mocks'] == 1
    assert calls['biases'] == list(expected.values())
    assert_restored()


@pytest.mark.parametrize('tracers', ['ELG', ['ELG']])
def test_fits_only_selected_tracers(monkeypatch, tracers):
    hod, expected, calls, assert_restored = make_bias_model(monkeypatch)
    result = hod.get_lin_bias(tracers=tracers)
    assert result == pytest.approx({'ELG': expected['ELG']}, rel=1.e-7)
    assert calls['mocks'] == 1
    assert calls['biases'] == [expected['ELG']]
    assert_restored()


@pytest.mark.parametrize('stage', ['mock_generation', 'second_tracer', 'fit'])
def test_restores_settings_when_calculation_fails(monkeypatch, stage):
    hod, _, _, assert_restored = make_bias_model(monkeypatch, existing_edges=False)

    def fail(*args, **kwargs):
        raise RuntimeError('Injected calculation failure')

    if stage == 'mock_generation':
        monkeypatch.setattr(hod, 'make_mock_cat', fail)
    elif stage == 'fit':
        monkeypatch.setattr('scipy.optimize.curve_fit', fail)
    else:
        original = hod.get_xiells

        def get_xiells(cat, *, tracers, ells):
            if tracers == 'ELG':
                fail()
            return original(cat, tracers=tracers, ells=ells)

        monkeypatch.setattr(hod, 'get_xiells', get_xiells)
    with pytest.raises(RuntimeError, match='Injected calculation failure'):
        hod.get_lin_bias()
    assert_restored()


def test_requires_cosmology_before_creating_a_mock(monkeypatch):
    hod, _, calls, assert_restored = make_bias_model(monkeypatch)
    hod.cosmo = None
    with pytest.raises(ValueError, match='cosmology is required'):
        hod.get_lin_bias()
    assert calls['mocks'] == 0
    assert_restored()
