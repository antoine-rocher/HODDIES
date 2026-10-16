"""Velocity-error models, custom prescriptions, and legacy configuration."""

from types import SimpleNamespace

import numpy as np
import pytest

from HODDIES import HOD, vsmear_func


def make_hod(spec, seed=42):
    hod = HOD.__new__(HOD)
    hod.args = {'LRG': {'vsmear': spec}}
    hod.rng = np.random.RandomState(seed)
    return hod


def sample(spec, size=1000):
    return make_hod(spec).get_vsmear('LRG', size, verbose=False)


def test_gaussian_width_is_standard_deviation():
    errors = sample({'model': 'gaussian', 'dv': 40.}, 200_000)
    assert abs(errors.mean()) < .4
    assert errors.std() == pytest.approx(40., rel=.01)


def test_lorentzian_width_is_half_width_at_half_maximum():
    errors = sample({'model': 'lorentzian', 'dv': 40.}, 200_000)
    # A Cauchy distribution has no variance; its quartiles are +/- dv.
    np.testing.assert_allclose(np.quantile(errors, [.25, .5, .75]),
                               [-40., 0., 40.], atol=1.)


@pytest.mark.parametrize('spec', [{'model': 'gaussian', 'dv': 40.}, {'dv': 40.}])
def test_gaussian_dictionary_preserves_legacy_random_stream(spec):
    legacy, configured = make_hod(40.), make_hod(spec)
    for size in (11, 0, 73):
        np.testing.assert_array_equal(
            legacy.get_vsmear('LRG', size, verbose=False),
            configured.get_vsmear('LRG', size, verbose=False))
    np.testing.assert_array_equal(legacy.rng.random_sample(10),
                                  configured.rng.random_sample(10))


@pytest.mark.parametrize('spec', [None, 0, {'dv': 0.},
                                 {'model': 'lorentzian', 'dv': 0.}])
def test_disabled_smearing_does_not_consume_random_numbers(spec):
    hod = make_hod(spec)
    assert hod.get_vsmear('LRG', 100, verbose=False) == 0
    np.testing.assert_array_equal(hod.rng.random_sample(10),
                                  np.random.RandomState(42).random_sample(10))


def test_custom_model_receives_parameters_and_hod_random_generator(monkeypatch):
    hod = make_hod({'model': 'custom', 'dv': 12., 'params': {'offset': 3.}})

    def custom(size, dv, rng, *, offset):
        assert rng is hod.rng
        return rng.normal(offset, dv, size=size)

    monkeypatch.setattr(vsmear_func, 'custom', custom, raising=False)
    errors = hod.get_vsmear('LRG', 13, verbose=False)
    np.testing.assert_array_equal(errors, np.random.RandomState(42).normal(3., 12., 13))


@pytest.mark.parametrize('spec', [
    'gaussian', [0.8, 1.1], {}, {'model': 'gaussian'},
    {'model': 'missing', 'dv': 1.}, {'model': 'missing', 'dv': 0.},
    {'model': None, 'dv': 1.}, {'model': 12, 'dv': 1.},
    {'dv': -1.}, {'dv': np.inf}, {'dv': np.nan}, {'dv': [1.]},
    {'dv': '40'}, {'dv': 1., 'params': []},
    {'dv': 1., 'params': {'size': 3}},
    {'dv': 1., 'params': {'dv': 3}},
    {'dv': 1., 'params': {'rng': None}},
    -1., np.inf, np.nan,
])
def test_invalid_configuration_raises_value_error(spec):
    with pytest.raises(ValueError):
        sample(spec)


def test_named_model_must_be_callable(monkeypatch):
    monkeypatch.setattr(vsmear_func, 'invalid', 12., raising=False)
    with pytest.raises(ValueError):
        sample({'model': 'invalid', 'dv': 1.})


@pytest.mark.parametrize('output', [0., np.zeros((5, 1)),
                                    np.zeros(4), np.full(5, np.nan),
                                    np.full(5, np.inf)])
def test_custom_output_must_be_finite_and_match_catalogue_size(monkeypatch, output):
    monkeypatch.setattr(vsmear_func, 'custom', lambda **kwargs: output, raising=False)
    with pytest.raises(ValueError):
        sample({'model': 'custom', 'dv': 1.}, size=5)


@pytest.mark.parametrize('override_legacy_scalar', [False, True])
def test_dictionary_is_loaded_from_yaml_or_overrides_legacy_scalar(tmp_path, override_legacy_scalar):
    param_file = tmp_path / 'hod.yaml'
    spec = {'model': 'lorentzian', 'dv': 25.}
    config = ('  vsmear: 0\n' if override_legacy_scalar else
              '  vsmear:\n    model: lorentzian\n    dv: 25.0\n')
    param_file.write_text('tracers: [LRG]\nLRG:\n  satellites: false\n' + config)
    loader = SimpleNamespace(z_simu=.5, boxsize=100., hcat={},
                             part_subsamples=None, field_particles=None,
                             cosmo=lambda **kwargs: None)
    overrides = {'LRG': {'vsmear': spec}} if override_legacy_scalar else {}
    hod = HOD(loader, param_file=str(param_file), setup_logger=False, seed=42,
              nthreads=1, **overrides)
    np.testing.assert_array_equal(
        hod.get_vsmear('LRG', 8, verbose=False),
        25. * np.random.RandomState(42).standard_cauchy(size=8))
