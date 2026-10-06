"""Regression checks for particle satellite velocities in both simulation loaders."""

import logging

import numba
import numpy as np
import pytest

from HODDIES.sim_loader.abacus_io import AbacusSummitSim, _compute_sat_from_abacus_part
from HODDIES.sim_loader.sim_loader import Base_catalogue
from HODDIES.utils import sample_satellites_from_particles


def make_catalogue(kind, boost=0.0):
    # Deliberately unsorted hosts, with one excluded host and one empty host.
    halo_ids = np.array([30, 10, 20, 40])
    halo_vel = np.array([[100., -40., 9.], [700., 80., 2.],
                         [-50., 30., 5.], [0., 1., 2.]]) + boost
    particle_ids = np.array([30, 30, 30, 10, 20, 20])
    pos = np.arange(18, dtype=np.float64).reshape(6, 3)
    vel = 2 * pos - 17 + boost
    cls = Base_catalogue if kind == 'generic' else AbacusSummitSim
    catalogue = cls.__new__(cls)
    catalogue.nthreads = 2
    catalogue.logger = logging.getLogger(__name__)
    catalogue.hcat = dict(halo_id=halo_ids, npoutA=np.array([3, 1, 2, 0]),
                         npstartA=np.array([0, 3, 4, 6]))
    for j, axis in enumerate('xyz'):
        catalogue.hcat['v' + axis] = halo_vel[:, j]
    if kind == 'generic':
        catalogue.part_subsamples = {'halo_id': particle_ids}
        for j, axis in enumerate('xyz'):
            catalogue.part_subsamples[axis] = pos[:, j]
            catalogue.part_subsamples['v' + axis] = vel[:, j]
    else:
        catalogue.part_subsamples = {'pos': pos, 'vel': vel}
    return catalogue, pos, vel, halo_vel


@pytest.mark.parametrize('kind', ['generic', 'abacus'])
@pytest.mark.parametrize('factor', [0., 0.5, 1., 1.7])
def test_bias_host_order_and_missing_particles(kind, factor):
    numba.set_num_threads(2)
    catalogue, pos, vel, halo_vel = make_catalogue(kind)
    mask = np.array([True, False, True, True])
    counts = np.array([2, 3, 1])
    seed = np.array([123, 456], dtype=np.uint32)
    reference = catalogue.assign_sat_to_part(mask, counts, seed=seed)
    result = catalogue.assign_sat_to_part(mask, counts, seed=seed, f_sigv=factor)
    np.testing.assert_array_equal(result[-1], [False, False, False, False, True, True])
    valid = ~result[-1]
    for j in range(3):
        np.testing.assert_array_equal(result[j][valid], reference[j][valid])
    # x uniquely identifies the sampled particle; expected hosts follow output blocks.
    particle_index = (result[0][valid] / 3).astype(int)
    host_velocity = np.repeat(halo_vel[mask], counts, axis=0)[valid]
    expected = host_velocity + factor * (vel[particle_index] - host_velocity)
    np.testing.assert_allclose(np.column_stack(result[3:6])[valid], expected, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize('kind', ['generic', 'abacus'])
def test_velocity_bias_is_invariant_under_common_boost(kind):
    numba.set_num_threads(2)
    results = []
    mask = np.array([True, False, True, True])
    for boost in [0., 32.]:
        catalogue, _, _, _ = make_catalogue(kind, boost)
        results.append(catalogue.assign_sat_to_part(
            mask, np.array([2, 3, 1]), seed=np.array([123, 456]), f_sigv=0.5))
    valid = ~results[0][-1]
    np.testing.assert_allclose(np.column_stack(results[1][3:6])[valid],
                               np.column_stack(results[0][3:6])[valid] + 32.)


@pytest.mark.parametrize('kind', ['generic', 'abacus'])
def test_direct_kernel_default_and_missing_host_validation(kind):
    numba.set_num_threads(2)
    arrays = [np.arange(3, dtype=np.float64) + j for j in range(6)]
    counts = np.array([3])
    if kind == 'generic':
        kernel = sample_satellites_from_particles
        args = (*arrays, np.arange(3), np.array([0, 3]), counts)
    else:
        kernel = _compute_sat_from_abacus_part
        args = (*arrays, np.array([3]), np.array([0]), counts, np.array([0, 3]), 1)
    result = kernel(*args, seed=np.array([123]))
    indices = result[0].astype(int)
    for j in range(3, 6):
        np.testing.assert_array_equal(result[j], arrays[j][indices])
    with pytest.raises(ValueError, match='Host halo velocities'):
        kernel(*args, f_sigv=0.5)
