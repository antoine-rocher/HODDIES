"""Central velocity offsets in actual single- and multi-tracer mock generation."""

from types import SimpleNamespace

import numba
import numpy as np
import pytest
from mpytools import Catalog

from HODDIES import HOD


def make_hod(tracers=('LRG',), *, satellites=False, vrms=True,
             integer_velocities=False, **tracer_overrides):
    numba.set_num_threads(2)
    index = np.arange(32)
    log_mass = np.linspace(13.0, 14.0, index.size)
    halo_data = {
        'halo_id': 7 * index[::-1] + 3,
        'log10_Mh': log_mass,
        'Mh': 10 ** log_mass,
        'Rh': np.full(index.size, 350.),
        'c': np.full(index.size, 6.),
    }
    for component, axis in enumerate('xyz'):
        halo_data[axis] = index.astype(float) + 5 * component
        halo_data['v' + axis] = 3 * index + 10 * component - 40
        if not integer_velocities:
            halo_data['v' + axis] = halo_data['v' + axis].astype(float) + .25
    if vrms:
        halo_data['Vrms'] = 100. + 3 * index
    loader = SimpleNamespace(
        hcat=Catalog.from_dict(halo_data), part_subsamples=None, field_particles=None,
        z_simu=.5, boxsize=100., cosmo=lambda **kwargs: None,
    )
    parameters = {}
    for tracer in tracers:
        parameters[tracer] = dict(
            HOD_model='SHOD', Ac=1., log_Mcent=10., sigma_M=.01,
            density=None, satellites=satellites, As=.35, M_0=10.,
            M_1=13.5, alpha=1., vel_sat='rd_normal',
        )
        parameters[tracer].update(tracer_overrides.get(tracer, {}))
    return HOD(loader, tracers=list(tracers), nthreads=2,
               setup_logger=False, **parameters)


def assert_columns_equal(first, second):
    assert first.columns() == second.columns()
    for column in first.columns():
        np.testing.assert_array_equal(first[column], second[column])


def assert_central_offsets(mock, halos, factor, tracer):
    centrals = mock[(mock['Central'] == 1) & (mock['TRACER'] == tracer)]
    assert centrals.size > 0
    indices = np.array([
        np.flatnonzero(halos['halo_id'] == halo_id)[0]
        for halo_id in centrals['halo_id']
    ])
    for axis in 'xyz':
        np.testing.assert_array_equal(centrals[axis], halos[axis][indices])
        expected = halos['v' + axis][indices] + factor * halos['Vrms'][indices]
        np.testing.assert_allclose(centrals['v' + axis], expected)


@pytest.mark.parametrize('tracer', ['LRG', 'ELG', 'QSO', 'CUSTOM'])
def test_default_bias_leaves_central_velocities_unchanged(tracer):
    hod = make_hod((tracer,))
    assert hod.args[tracer]['f_vcen'] == 0.
    mock = hod.make_mock_cat(fix_seed=42, verbose=False)
    assert mock.size == hod.hcat.size
    np.testing.assert_array_equal(mock['Central'], 1)
    assert_central_offsets(mock, hod.hcat, 0., tracer)


@pytest.mark.parametrize('factor', [0., .3, -.4])
def test_central_only_offset_uses_its_own_halo_vrms(factor):
    hod = make_hod(LRG={'f_vcen': factor})
    source = hod.hcat.deepcopy()
    mock = hod.make_mock_cat(fix_seed=42, verbose=False)
    assert mock.size == source.size
    assert_central_offsets(mock, source, factor, 'LRG')
    assert_columns_equal(hod.hcat, source)
    assert_columns_equal(mock, hod.make_mock_cat(fix_seed=42, verbose=False))


def test_offset_promotes_integer_velocities_without_truncation():
    hod = make_hod(integer_velocities=True, LRG={'f_vcen': .125})
    source = hod.hcat.deepcopy()
    mock = hod.make_mock_cat(fix_seed=42, verbose=False)
    assert_central_offsets(mock, source, .125, 'LRG')
    assert np.issubdtype(mock['vx'].dtype, np.floating)
    assert_columns_equal(hod.hcat, source)


def test_satellites_and_random_sampling_are_unchanged_by_central_bias():
    hod = make_hod(satellites=True)
    source = hod.hcat.deepcopy()
    reference = hod.make_mock_cat(fix_seed=42, verbose=False)
    hod.args['LRG']['f_vcen'] = .6
    biased = hod.make_mock_cat(fix_seed=42, verbose=False)
    np.testing.assert_array_equal(biased['halo_id'], reference['halo_id'])
    np.testing.assert_array_equal(biased['Central'], reference['Central'])
    for axis in 'xyz':
        np.testing.assert_array_equal(biased[axis], reference[axis])
    satellite_mask = reference['Central'] == 0
    assert satellite_mask.any()
    assert_columns_equal(biased[satellite_mask], reference[satellite_mask])
    assert_central_offsets(biased, source, .6, 'LRG')
    assert_columns_equal(hod.hcat, source)
    assert_columns_equal(biased, hod.make_mock_cat(fix_seed=42, verbose=False))


def test_each_tracer_uses_its_own_bias_and_unmodified_halo_velocities():
    hod = make_hod(('LRG', 'ELG'),
                   LRG={'Ac': .5}, ELG={'Ac': .5})
    source = hod.hcat.deepcopy()
    reference = hod.make_mock_cat(fix_seed=42, verbose=False)
    factors = {'LRG': .35, 'ELG': -.2}
    for tracer, factor in factors.items():
        hod.args[tracer]['f_vcen'] = factor
    biased = hod.make_mock_cat(fix_seed=42, verbose=False)
    for column in ('halo_id', 'Central', 'TRACER', 'x', 'y', 'z'):
        np.testing.assert_array_equal(biased[column], reference[column])
    for tracer, factor in factors.items():
        assert_central_offsets(biased, source, factor, tracer)
    assert_columns_equal(hod.hcat, source)
    assert_columns_equal(biased, hod.make_mock_cat(fix_seed=42, verbose=False))


def test_missing_vrms_is_supported_when_central_bias_is_zero():
    hod = make_hod(vrms=False)
    mock = hod.make_mock_cat(fix_seed=42, verbose=False)
    assert mock.size == hod.hcat.size
    for axis in 'xyz':
        np.testing.assert_array_equal(mock['v' + axis], hod.hcat['v' + axis])


def test_missing_vrms_warns_once_and_keeps_halo_velocities(monkeypatch):
    hod = make_hod(vrms=False, LRG={'f_vcen': .3})
    messages = []
    monkeypatch.setattr(hod.logger, 'warning', messages.append)
    mock = hod.make_mock_cat(fix_seed=42, verbose=False)
    assert mock.size == hod.hcat.size
    for axis in 'xyz':
        np.testing.assert_array_equal(mock['v' + axis], hod.hcat['v' + axis])
    assert_columns_equal(mock, hod.make_mock_cat(fix_seed=42, verbose=False))
    assert len(messages) == 1
    assert 'Vrms' in messages[0]
    assert 'skipping central velocity bias' in messages[0]
