"""Regression checks for configured xi and power multipoles in HOD.plot_stats."""

from itertools import combinations_with_replacement
from types import MethodType
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

from HODDIES import HOD
from HODDIES import plot_utils


K = np.array([0.05, 0.1, 0.2])


@pytest.fixture(autouse=True)
def restore_registry_and_close_figures():
    original = dict(plot_utils.STATS)
    yield
    plot_utils.STATS.clear()
    plot_utils.STATS.update(original)
    plt.close('all')


def make_hod(power_ells=(0, 2), xi_ells=(0, 2), tracers=('LRG',)):
    """Use the real plotter with deterministic estimator results."""
    hod = HOD.__new__(HOD)
    hod.args = {'clustering_settings': {
        'power_spectrum': {'multipole_index': list(power_ells)},
        'xi_smu': {'multipole_index': list(xi_ells)},
    }}
    hod.stat_calls = []
    cat = {'TRACER': np.asarray(tracers)}

    def compute_stats(self, cat, stat=None, tracers=None, **kwargs):
        self.stat_calls.append((tuple(stat), tuple(tracers)))
        source = stat[0]
        result = {}
        for pair_index, (a, b) in enumerate(combinations_with_replacement(tracers, 2)):
            amplitude = 100. * (pair_index + 1)
            if source in ('power_spectrum', 'xi_ells'):
                settings = 'power_spectrum' if source == 'power_spectrum' else 'xi_smu'
                orders = self.args['clustering_settings'][settings]['multipole_index']
                values = np.vstack([(ell + 1) * amplitude + np.arange(K.size)
                                    for ell in orders])
                entry = (K, values)
            elif source in ('xi_rppi', 'xi_smu'):
                entry = (K, np.array([0.25, 0.75]),
                         np.arange(6).reshape(3, 2) + 1.)
            else:
                entry = (K, amplitude + np.arange(K.size))
            result[f'{a}_{b}'] = entry
        return {source: result}

    hod.compute_stats = MethodType(compute_stats, hod)
    return hod, cat


def expected_power(ell, pair_index=0):
    return (ell + 1) * 100. * (pair_index + 1) + np.arange(K.size)


def test_power_multipoles_are_available_in_registry():
    stats = plot_utils.get_STATS()
    for ell, component in [(0, 0), (2, 1)]:
        spec = stats[f'pk{ell}']
        assert spec.source == 'power_spectrum'
        assert spec.component == component
        np.testing.assert_array_equal(spec.scale(K, expected_power(ell)),
                                      K * expected_power(ell))


def test_group_uses_power_settings_and_preserves_configured_order():
    hod, cat = make_hod(power_ells=(4, 0, 2), xi_ells=(0, 2))
    fig = hod.plot_stats(cat, stats='POWER_SPECTRUM', show=False)
    assert fig._plot_stats_stats == ['pk4', 'pk0', 'pk2']
    for ax, ell in zip(fig._plot_stats_axes[0][0], [4, 0, 2]):
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), K)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), K * expected_power(ell))
        assert f'P_{{{ell}}}' in ax.get_ylabel()
    assert hod.stat_calls == [(('power_spectrum',), ('LRG',))]


@pytest.mark.parametrize('orders, requested', [
    ((2, 0), ['pk0', 'PK2']),
    ((4, 2, 0), ['pk2', 'pk0']),
    ((2,), ['pk2']),
])
def test_individual_multipoles_resolve_current_object_after_previous_group(orders, requested):
    previous, _ = make_hod(power_ells=(0, 4, 2))
    plot_utils._pk_ells_group(previous)
    hod, cat = make_hod(power_ells=orders)
    fig = hod.plot_stats(cat, stats=requested, show=False)
    for ax, name in zip(fig._plot_stats_axes[0][0], requested):
        ell = int(name[2:])
        np.testing.assert_allclose(ax.lines[0].get_ydata(), K * expected_power(ell))
    assert len(hod.stat_calls) == 1


def test_unconfigured_individual_multipole_has_clear_error():
    hod, cat = make_hod(power_ells=(0,))
    with pytest.raises(ValueError, match="multipole 'pk2' is not configured"):
        hod.plot_stats(cat, stats=['pk2'], show=False)
    assert not hod.stat_calls


@pytest.mark.parametrize('layout', ['rows', 'overlay'])
def test_auto_and_cross_multipoles_share_one_estimator_call(layout):
    hod, cat = make_hod(tracers=('LRG', 'ELG'))
    fig = hod.plot_stats(cat, stats=['pk0', 'pk2'], tracers=['LRG', 'ELG'],
                         by_tracer=layout, show=False)
    assert hod.stat_calls == [(('power_spectrum',), ('LRG', 'ELG'))]
    # Estimator ordering is LRG_LRG, LRG_ELG, ELG_ELG; panels show autos first.
    for block_index, pair_index in enumerate([0, 2, 1]):
        axes = fig._plot_stats_axes[block_index if layout == 'rows' else 0][0]
        for ax, ell in zip(axes, [0, 2]):
            curve = ax.lines[0 if layout == 'rows' else block_index]
            np.testing.assert_allclose(curve.get_ydata(),
                                       K * expected_power(ell, pair_index))


def test_group_data_covariance_produces_residuals_without_unknown_warning():
    hod, cat = make_hod(power_ells=(2, 0), xi_ells=(0, 2))
    sigma = np.array([[2., 3., 4.], [5., 6., 7.]])
    residual = np.array([[1., -2., 0.5], [0.25, 1.5, -1.]])
    values = np.vstack([expected_power(ell) for ell in [2, 0]]) - residual * sigma
    covariance = np.diag(sigma.ravel() ** 2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hod.plot_stats(cat, stats='power_spectrum', show=False,
                             data={'power_spectrum': {'LRG': (K, values, covariance)}})
    assert not any('not known statistics' in str(item.message) for item in caught)
    assert fig._plot_stats_stats == ['pk2', 'pk0']
    main_axes, residual_axes = fig._plot_stats_axes[0]
    for i, (main, res_ax) in enumerate(zip(main_axes, residual_axes)):
        assert res_ax is not None
        np.testing.assert_allclose(res_ax.lines[0].get_ydata(), residual[i])
        np.testing.assert_allclose(main.lines[1].get_ydata(), K * values[i])
        np.testing.assert_allclose(fig._plot_stats_data[f'pk{[2, 0][i]}']['LRG'][2], sigma[i])


@pytest.mark.parametrize('orders', [(2, 0), (2,), (4, 0, 2)])
def test_all_uses_current_power_orders_despite_stale_registered_pk4(orders):
    previous, _ = make_hod(power_ells=(4, 0, 2))
    plot_utils._pk_ells_group(previous)
    assert 'pk4' in plot_utils.STATS
    hod, cat = make_hod(power_ells=orders)
    fig = hod.plot_stats(cat, stats='ALL', show=False)
    power_names = [name for name in fig._plot_stats_stats if name.startswith('pk')]
    assert power_names == [f'pk{ell}' for ell in orders]
    assert sum(call[0] == ('power_spectrum',) for call in hod.stat_calls) == 1


def test_all_preserves_custom_power_statistics():
    plot_utils.register_stat('power_unscaled', source='power_spectrum', component=0)
    hod, cat = make_hod()
    fig = hod.plot_stats(cat, stats='ALL', show=False)
    assert 'power_unscaled' in fig._plot_stats_stats
    index = fig._plot_stats_stats.index('power_unscaled')
    ax = fig._plot_stats_axes[0][0][index]
    np.testing.assert_allclose(ax.lines[0].get_ydata(), expected_power(0))


@pytest.mark.parametrize('prefix, source', [
    ('xi', 'xi_ells'), ('pk', 'power_spectrum'),
])
def test_individual_higher_multipoles_use_independent_configured_order(prefix, source):
    orders = (6, 0, 12, 2, 4)
    other_orders = (4, 2, 0)
    hod, cat = make_hod(
        power_ells=orders if prefix == 'pk' else other_orders,
        xi_ells=orders if prefix == 'xi' else other_orders)
    requested = [f'{prefix.upper()}4', f'{prefix}6', f'{prefix}12']
    fig = hod.plot_stats(cat, stats=requested, show=False)
    assert fig._plot_stats_stats == [f'{prefix}4', f'{prefix}6', f'{prefix}12']
    for ax, ell in zip(fig._plot_stats_axes[0][0], [4, 6, 12]):
        np.testing.assert_allclose(ax.lines[0].get_ydata(), K * expected_power(ell))
    assert hod.stat_calls == [((source,), ('LRG',))]


@pytest.mark.parametrize('orders, requested', [
    ((6, 4, 0, 2), ['xi4', 'xi6']),
    ((2, 0), ['xi0', 'XI2']),
    ((6,), ['xi6']),
])
def test_individual_xi_multipoles_resolve_current_object_after_previous_group(orders, requested):
    previous, _ = make_hod(xi_ells=(0, 4, 6, 2))
    plot_utils._xi_ells_group(previous)
    hod, cat = make_hod(xi_ells=orders)
    fig = hod.plot_stats(cat, stats=requested, show=False)
    for ax, name in zip(fig._plot_stats_axes[0][0], requested):
        ell = int(name[2:])
        np.testing.assert_allclose(ax.lines[0].get_ydata(), K * expected_power(ell))
    assert hod.stat_calls == [(('xi_ells',), ('LRG',))]


@pytest.mark.parametrize('xi_orders, power_orders', [
    ((6,), (12,)),
    ((6, 0, 4), (2, 10, 0)),
])
def test_all_uses_current_xi_and_power_orders_after_previous_groups(xi_orders, power_orders):
    previous, _ = make_hod(xi_ells=(12, 6, 4, 0, 2),
                           power_ells=(10, 0, 6, 4, 2))
    plot_utils._xi_ells_group(previous)
    plot_utils._pk_ells_group(previous)
    hod, cat = make_hod(xi_ells=xi_orders, power_ells=power_orders)
    fig = hod.plot_stats(cat, stats='ALL', show=False)
    assert [name for name in fig._plot_stats_stats
            if name.startswith('xi') and name[2:].isdigit()] == [
                f'xi{ell}' for ell in xi_orders]
    assert [name for name in fig._plot_stats_stats
            if name.startswith('pk') and name[2:].isdigit()] == [
                f'pk{ell}' for ell in power_orders]
    for prefix, orders in [('xi', xi_orders), ('pk', power_orders)]:
        for ell in orders:
            panel = fig._plot_stats_stats.index(f'{prefix}{ell}')
            ax = fig._plot_stats_axes[0][0][panel]
            np.testing.assert_allclose(ax.lines[0].get_ydata(), K * expected_power(ell))
    assert sum(call[0] == ('xi_ells',) for call in hod.stat_calls) == 1
    assert sum(call[0] == ('power_spectrum',) for call in hod.stat_calls) == 1


def test_unconfigured_individual_xi_multipole_has_clear_error():
    hod, cat = make_hod(xi_ells=(0, 2, 4))
    with pytest.raises(ValueError, match="multipole 'xi6' is not configured") as caught:
        hod.plot_stats(cat, stats=['xi6'], show=False)
    assert 'xi_smu' in str(caught.value)
    assert 'multipole_index' in str(caught.value)
    assert not hod.stat_calls


def test_individual_higher_xi_group_data_covariance_preserves_component_order():
    orders = (6, 0, 4)
    hod, cat = make_hod(xi_ells=orders, power_ells=(4, 2, 0))
    sigma = np.array([[2., 3., 4.], [5., 6., 7.], [8., 9., 10.]])
    residual = np.array([[1., -2., 0.5], [0.25, 1.5, -1.], [-0.5, 2., 1.]])
    values = np.vstack([expected_power(ell) for ell in orders]) - residual * sigma
    covariance = np.diag(sigma.ravel() ** 2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig = hod.plot_stats(cat, stats=['xi4', 'XI6'], show=False,
                             data={'xi_ells': {'LRG': (K, values, covariance)}})
    assert not any('not known statistics' in str(item.message) for item in caught)
    main_axes, residual_axes = fig._plot_stats_axes[0]
    for main, res_ax, ell in zip(main_axes, residual_axes, [4, 6]):
        component = orders.index(ell)
        assert res_ax is not None
        np.testing.assert_allclose(res_ax.lines[0].get_ydata(), residual[component])
        np.testing.assert_allclose(main.lines[1].get_ydata(), K * values[component])
        np.testing.assert_allclose(fig._plot_stats_data[f'xi{ell}']['LRG'][2],
                                   sigma[component])
    assert hod.stat_calls == [(('xi_ells',), ('LRG',))]


def test_all_preserves_custom_xi_statistics():
    plot_utils.register_stat('xi_unscaled', source='xi_ells', component=0)
    hod, cat = make_hod(xi_ells=(6, 2))
    fig = hod.plot_stats(cat, stats='ALL', show=False)
    assert 'xi_unscaled' in fig._plot_stats_stats
    panel = fig._plot_stats_stats.index('xi_unscaled')
    ax = fig._plot_stats_axes[0][0][panel]
    np.testing.assert_allclose(ax.lines[0].get_ydata(), expected_power(6))
