"""The exact CMP Poisson limit, including low satellite occupations."""

import numba
import numpy as np
import pytest
from scipy.stats import poisson

from HODDIES.utils import (cmp_lambda_from_mu_allscale, cmp_pmf_vec,
                          cmp_sample, get_max_x)


@pytest.mark.parametrize('mean', [0., 1.e-8, .1, .3, .4, .49, .5,
                                  np.nextafter(.5, np.inf), 1., 10., 100.])
def test_nu_one_preserves_requested_mean(mean):
    assert cmp_lambda_from_mu_allscale(mean, 1.) == mean


@pytest.mark.parametrize('mean', [.1, .4, .5, 1., 10., 100.])
def test_nu_one_pmf_matches_poisson(mean):
    rate = cmp_lambda_from_mu_allscale(mean, 1.)
    counts = np.arange(get_max_x(mean) + 1)
    actual = cmp_pmf_vec(counts, rate, 1.)
    np.testing.assert_allclose(actual, poisson.pmf(counts, mean),
                               rtol=2.e-12, atol=1.e-14)
    np.testing.assert_allclose(np.sum(counts * actual), mean, rtol=1.e-11)
    np.testing.assert_allclose(np.sum(counts * (counts - 1) * actual),
                               mean ** 2, rtol=1.e-11)


@numba.njit
def cmp_draw_and_next_uniform(mean, seed):
    np.random.seed(seed)
    count = cmp_sample(mean, 1.)
    return count, np.random.rand()


@pytest.mark.parametrize('mean', [.1, .4, .5, 1., 5., 15., 100.])
def test_poisson_limit_retains_one_uniform_inverse_cdf_sampling(mean):
    # Preserve common uniform draws when varying nu continuously in the CMP path.
    seed = 123
    random = np.random.RandomState(seed)
    uniform = random.rand()
    expected_next = random.rand()
    expected_count = np.searchsorted(poisson.cdf(np.arange(get_max_x(mean) + 1), mean),
                                    uniform, side='left')
    count, following = cmp_draw_and_next_uniform(mean, seed)
    assert count == expected_count
    assert following == expected_next
