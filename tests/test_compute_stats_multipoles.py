"""Individual multipoles use the actual statistics and tracer-pair routing."""

import numpy as np
import pytest
from mpytools import Catalog

from HODDIES import HOD, plot_utils
import HODDIES.hod as hod_module


S = np.array([5., 10., 20.])


@pytest.fixture(autouse=True)
def restore_registry():
    original = dict(plot_utils.STATS)
    yield
    plot_utils.STATS.clear()
    plot_utils.STATS.update(original)


def make_hod(monkeypatch, orders=(4, 0, 2), tracers=('LRG', 'ELG')):
    hod = HOD.__new__(HOD)
    hod.args = {'clustering_settings': {
        'rsd': False, 'los': 'z',
        'xi_smu': dict(bin_logscale=True, smin=1., smax=30., n_s_bins=3,
                       mu_max=1., n_mu_bins=5, multipole_index=list(orders)),
        'xi_rppi': dict(bin_logscale=True, rp_min=1., rp_max=30.,
                        n_rp_bins=3, pimax=4),
    }}
    hod.cosmo, hod.boxsize, hod.nthreads = None, 100., 1
    calls = []
    cat = Catalog.from_dict({
        'TRACER': np.repeat(tracers, 4),
        'x': np.repeat(np.arange(1, len(tracers) + 1, dtype=float), 4),
        'y': np.tile(np.arange(4, dtype=float), len(tracers)),
        'z': np.zeros(4 * len(tracers)),
    })

    def compute_twopoint(pos1, mode, edges, boxsize, los, nthreads, *, pos2, **kwargs):
        pair = (int(pos1[0][0]), int(pos2[0][0]))
        calls.append((mode, pair))

        def project(*, return_sep, ells=None, pimax=None):
            assert return_sep
            if pimax is not None:
                return S, expected(pair, 0) * pimax
            if ells is None:
                return S, np.array([.25, .75]), np.ones((S.size, 2))
            if np.isscalar(ells):
                return S, expected(pair, ells)
            return S, np.vstack([expected(pair, ell) for ell in ells])

        return project

    monkeypatch.setattr(hod_module, 'compute_twopoint', compute_twopoint)
    return hod, cat, calls


def expected(pair, ell):
    return (10 * pair[0] + pair[1]) * (ell + 1) + np.arange(S.size)


def test_individual_multipoles_share_counts_and_match_grouped_output(monkeypatch):
    hod, cat, calls = make_hod(monkeypatch)
    result = hod.compute_stats(cat, stat=['xi0', 'XI2', 'xi4', 'XI_ELLS'],
                               tracers=['LRG', 'ELG'])
    pairs = {'LRG_LRG': (1, 1), 'LRG_ELG': (1, 2), 'ELG_ELG': (2, 2)}
    assert calls == [('smu', pair) for pair in pairs.values()]
    for label, pair in pairs.items():
        _, grouped = result['xi_ells'][label]
        assert grouped.shape == (3, S.size)
        for index, ell in enumerate([4, 0, 2]):
            s, xi = result[f'xi{ell}'][label]
            assert xi.ndim == 1
            np.testing.assert_array_equal(s, S)
            np.testing.assert_array_equal(xi, expected(pair, ell))
            np.testing.assert_array_equal(xi, grouped[index])
    assert 'xi_smu' in result  # Preserve the existing grouped-result contract.


def test_single_higher_multipole_uses_string_and_single_tracer(monkeypatch):
    hod, cat, calls = make_hod(monkeypatch, orders=(2, 12, 0), tracers=('LRG',))
    result = hod.compute_stats(cat, stat='XI12', tracers='LRG')
    assert set(result) == {'xi12'}
    assert set(result['xi12']) == {'LRG_LRG'}
    np.testing.assert_array_equal(result['xi12']['LRG_LRG'][1], expected((1, 1), 12))
    assert calls == [('smu', (1, 1))]


def test_wp_and_individual_multipoles_can_be_requested_together(monkeypatch):
    hod, cat, calls = make_hod(monkeypatch, tracers=('LRG',))
    result = hod.compute_stats(cat, stat=['WP', 'xi0', 'xi2'])
    assert calls == [('rppi', (1, 1)), ('smu', (1, 1))]
    assert set(result) == {'wp', 'xi_rppi', 'xi0', 'xi2'}
    np.testing.assert_array_equal(result['wp']['LRG_LRG'][1], expected((1, 1), 0) * 4)


def test_grouped_multipoles_keep_existing_format(monkeypatch):
    hod, cat, calls = make_hod(monkeypatch, orders=(2, 0), tracers=('LRG',))
    result = hod.compute_stats(cat, stat='xi_ells')
    assert set(result) == {'xi_smu', 'xi_ells'}
    np.testing.assert_array_equal(result['xi_ells']['LRG_LRG'][1],
                                  np.vstack([expected((1, 1), ell) for ell in (2, 0)]))
    assert calls == [('smu', (1, 1))]


def test_unconfigured_multipole_fails_before_counting(monkeypatch):
    previous, _, _ = make_hod(monkeypatch, orders=(0, 2, 6))
    plot_utils._xi_ells_group(previous)
    hod, cat, calls = make_hod(monkeypatch, orders=(0, 2))
    with pytest.raises(ValueError, match="multipole 'xi6' is not configured"):
        hod.compute_stats(cat, stat=['wp', 'xi6'])
    assert not calls


def test_unknown_statistic_fails_before_counting(monkeypatch):
    hod, cat, calls = make_hod(monkeypatch)
    with pytest.raises(ValueError, match='not implemented'):
        hod.compute_stats(cat, stat=['xi0', 'unknown_stat'])
    assert not calls


def test_duplicate_names_do_not_repeat_pair_counts(monkeypatch):
    hod, cat, calls = make_hod(monkeypatch, tracers=('LRG',))
    result = hod.compute_stats(cat, stat=['xi2', 'XI2'])
    assert set(result) == {'xi2'}
    assert calls == [('smu', (1, 1))]
