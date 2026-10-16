"""Analytic counts visit each occupied mass bin exactly once."""

import numba
import numpy as np
import pytest

from HODDIES.utils import compute_ngal


@numba.njit
def constant_occupation(log_mass, parameters):
    return parameters[0]


@pytest.mark.parametrize('kernel', [compute_ngal, compute_ngal.py_func],
                         ids=['compiled', 'python_bounds_checked'])
@pytest.mark.parametrize('conformity,linked',
                         [(False, False), (True, False),
                          (False, True), (True, True)])
def test_constant_central_and_satellite_counts(kernel, conformity, linked):
    masses = np.linspace(12., 14., 257)
    central_mean, satellite_mean = .25, 2.
    # The second satellite parameter is the lower mass cutoff used by ngal.
    total, fraction, mean_mass = kernel(
        masses, constant_occupation, constant_occupation,
        np.array([central_mean]), np.array([satellite_mean, 0.]),
        conformity, linked)
    expected_centrals = len(masses) * central_mean
    # compute_ngal currently applies the central factor once whenever either
    # flag is set, including when both are enabled. Preserve that API here.
    expected_satellites = len(masses) * satellite_mean
    if conformity or linked:
        expected_satellites *= central_mean
    np.testing.assert_allclose(total * (1. - fraction), expected_centrals)
    np.testing.assert_allclose(total * fraction, expected_satellites)
    np.testing.assert_allclose(total, expected_centrals + expected_satellites)
    assert 10. ** masses.min() <= mean_mass <= 10. ** masses.max()


@pytest.mark.parametrize('kernel', [compute_ngal, compute_ngal.py_func],
                         ids=['compiled', 'python_bounds_checked'])
def test_constant_central_only_count(kernel):
    masses = np.linspace(12., 14., 257)
    total, fraction, mean_mass = kernel(
        masses, constant_occupation, constant_occupation, np.array([.25]))
    np.testing.assert_allclose(total, len(masses) * .25)
    assert fraction == 0.
    assert 10. ** masses.min() <= mean_mass <= 10. ** masses.max()
