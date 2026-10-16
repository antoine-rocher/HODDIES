"""DESI table sampling preserves the original Y3 inverse-CDF prescription."""

import numpy as np
import pytest
from scipy.interpolate import interp1d

from HODDIES import HOD, vsmear_func


def make_hod(spec, seed=42):
    hod = HOD.__new__(HOD)
    hod.args = {'LRG': {'vsmear': spec}}
    hod.rng = np.random.RandomState(seed)
    return hod


@pytest.fixture
def table(tmp_path):
    # Include a plateau and a non-unit normalization, as in prepared DESI CDFs.
    vbin = np.array([-1., -.6, 0., .4, .9, 1.4])
    cdf = np.array([.08, .08, .4, .9, 1.5, 2.])
    np.savez(tmp_path / 'LRG_z0.4-0.6_CDF.npz', vbin=vbin, cdf=cdf)
    return dict(tracer='LRG', zmin=.4, zmax=.6, data_dir=tmp_path), vbin, cdf


def original_y3(size, rng, vbin, cdf_raw):
    """Original algorithm, using a local RNG instead of resetting NumPy globally."""
    cdf, ind = np.unique(cdf_raw, return_index=True)
    x_grid = vbin[ind]
    inv_cdf = interp1d(cdf / cdf[-1], x_grid, bounds_error=False,
                      fill_value=(x_grid[0], x_grid[-1]), kind='cubic')
    exponent = inv_cdf(rng.uniform(0, 1, int(np.ceil(size / 2))))
    dv = np.append(10 ** exponent, -10 ** exponent)
    rng.shuffle(dv)
    if size % 2:
        dv = dv[1:]
    return dv.astype(np.float32)


@pytest.mark.parametrize('size', [0, 1, 7, 100])
@pytest.mark.parametrize('width', ['missing', None])
def test_desi_getter_matches_y3_for_all_sizes(table, size, width):
    params, vbin, cdf = table
    spec = {'model': 'desi', 'params': params}
    if width is None:
        spec['dv'] = None
    first, second = make_hod(spec), make_hod(spec)
    reference_rng = np.random.RandomState(42)
    expected = original_y3(size, reference_rng, vbin, cdf)
    actual = first.get_vsmear('LRG', size, verbose=False)
    assert actual.shape == (size,)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, second.get_vsmear('LRG', size, verbose=False))
    np.testing.assert_array_equal(first.rng.random_sample(5), reference_rng.random_sample(5))


def test_desi_pairing_dtype_and_global_rng(table):
    params, _, _ = table
    before = np.random.get_state()
    offsets = vsmear_func.desi(size=100, dv=None, rng=np.random.RandomState(42), **params)
    assert offsets.dtype == np.float32
    np.testing.assert_array_equal(np.sort(offsets), np.sort(-offsets))
    assert offsets.astype(np.float64).sum() == 0.
    for original, current in zip(before, np.random.get_state()):
        np.testing.assert_array_equal(original, current)


@pytest.mark.parametrize('tracer,dvmode,filename', [
    ('LRG', 'model', 'LRG_z0.4-0.6_kernel0.3_CDF.npz'),
    ('LRG_subsample', 'obs', 'LRG_z0.4-0.6_CDF.npz'),
    ('ELG_slitless', 'obs', 'ELG_slitless_z0.4-0.6_CDF.npz'),
])
def test_desi_table_names(table, tracer, dvmode, filename):
    params, vbin, cdf = table
    np.savez(params['data_dir'] / filename, vbin=vbin, cdf=cdf)
    params.update(tracer=tracer, dvmode=dvmode)
    np.testing.assert_array_equal(
        vsmear_func.desi(11, None, np.random.RandomState(5), **params),
        original_y3(11, np.random.RandomState(5), vbin, cdf))


def test_desi_zero_width_disables_without_advancing_rng(table):
    hod = make_hod({'model': 'desi', 'dv': 0., 'params': table[0]})
    assert hod.get_vsmear('LRG', 5, verbose=False) == 0
    np.testing.assert_array_equal(hod.rng.random_sample(5),
                                  np.random.RandomState(42).random_sample(5))


def test_desi_rejects_positive_width(table):
    hod = make_hod({'model': 'desi', 'dv': 40., 'params': table[0]})
    with pytest.raises(ValueError, match='(?i)(omit|none|null)'):
        hod.get_vsmear('LRG', 5, verbose=False)


@pytest.mark.parametrize('override', [
    {'tracer': 'invalid'}, {'dvmode': 'invalid'},
    {'zmin': .8, 'zmax': .4}, {'zmin': np.nan}, {'zmax': np.inf},
])
def test_desi_invalid_table_selection(table, override):
    params = dict(table[0], **override)
    with pytest.raises(ValueError):
        vsmear_func.desi(5, None, np.random.RandomState(42), **params)


def test_desi_missing_table_reports_filename(table):
    params = dict(table[0], zmin=.6, zmax=.8)
    with pytest.raises((ValueError, FileNotFoundError), match='LRG_z0.6-0.8_CDF.npz'):
        vsmear_func.desi(5, None, np.random.RandomState(42), **params)


@pytest.mark.parametrize('cdf', [
    [0., .4, .2, .6, .8, 1.],  # A CDF must not decrease.
    [0., 0., .5, .5, 1., 1.],  # Cubic interpolation needs four distinct values.
])
def test_desi_rejects_unusable_cdf(table, cdf):
    params, vbin, _ = table
    np.savez(params['data_dir'] / 'LRG_z0.4-0.6_CDF.npz', vbin=vbin, cdf=cdf)
    with pytest.raises(ValueError):
        vsmear_func.desi(5, None, np.random.RandomState(42), **params)


def test_gaussian_dictionary_keeps_existing_random_stream():
    hod = make_hod({'model': 'gaussian', 'dv': 40.})
    np.testing.assert_array_equal(hod.get_vsmear('LRG', 11, verbose=False),
                                  np.random.RandomState(42).normal(0., 40., 11))
