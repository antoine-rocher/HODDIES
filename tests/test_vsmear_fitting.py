"""Fitting and training preserve velocity-error models while varying dv."""

from copy import deepcopy
import logging

import numpy as np
import pytest

from HODDIES import HOD, vsmear_func


def make_hod(spec, fit_width=True):
    hod = HOD.__new__(HOD)
    priors = {'vsmear': [0., 100.]} if fit_width else {}
    hod.args = {
        'tracers': ['LRG'],
        'LRG': {'vsmear': deepcopy(spec), 'HOD_model': 'SHOD',
                'conformity_bias': False},
        'fit_param': {'priors': {'LRG': priors}, 'fit_statistics': ['xi0'],
                      'emulator': {'sampling_type': 'lhs'}},
    }
    hod._tracer_to_fit = ['LRG']
    hod.name_params = ['vsmear_LRG'] if fit_width else []
    hod.logger = logging.getLogger(__name__)
    return hod


@pytest.mark.parametrize('spec', [None, 0., 40.])
def test_fit_width_normalizes_legacy_configuration(spec):
    hod = make_hod(spec)
    params = np.array([(25.,)], dtype=[('vsmear_LRG', 'f8')])
    hod.update_new_param([params], params.dtype.names)
    assert hod.args['LRG']['vsmear'] == {'model': 'gaussian', 'dv': 25.}


@pytest.mark.parametrize('model', ['gaussian', 'lorentzian', 'custom'])
def test_fit_width_preserves_model_and_custom_parameters(monkeypatch, model):
    monkeypatch.setattr(vsmear_func, 'custom', lambda **kwargs: None, raising=False)
    spec = {'model': model, 'dv': 40., 'params': {'tail_fraction': .02}}
    hod = make_hod(spec)
    previous = hod.args['LRG']['vsmear']
    params = np.array([(25.,)], dtype=[('vsmear_LRG', 'f8')])
    hod.update_new_param([params], params.dtype.names)
    assert hod.args['LRG']['vsmear'] == dict(spec, dv=25.)
    assert previous == spec  # Previously saved parameter snapshots stay valid.
    assert hod.get_param_and_prior() == (['vsmear_LRG'], [[0., 100.]])


@pytest.mark.parametrize('dv', [-1., np.inf, np.nan])
def test_invalid_fit_width_is_rejected_without_overwriting_model(dv):
    spec = {'model': 'lorentzian', 'dv': 40.}
    hod = make_hod(spec)
    params = np.array([(dv,)], dtype=[('vsmear_LRG', 'f8')])
    with pytest.raises(ValueError, match='finite, nonnegative'):
        hod.update_new_param([params], params.dtype.names)
    assert hod.args['LRG']['vsmear'] == spec


@pytest.mark.parametrize('model', ['desi', 'custom_calibration'])
def test_calibrated_models_reject_width_priors_and_parameter_updates(monkeypatch, model):
    def custom_calibration(**kwargs):
        return None

    custom_calibration.requires_dv = False
    monkeypatch.setattr(vsmear_func, 'custom_calibration', custom_calibration, raising=False)
    spec = {'model': model, 'params': {'tracer': 'LRG'}}
    hod = make_hod(spec)
    with pytest.raises(ValueError, match='does not use a fitted dv'):
        hod.get_param_and_prior()
    params = np.array([(25.,)], dtype=[('vsmear_LRG', 'f8')])
    with pytest.raises(ValueError, match='does not use a fitted dv'):
        hod.update_new_param([params], params.dtype.names)
    assert hod.args['LRG']['vsmear'] == spec


def test_training_varies_width_without_replacing_lorentzian_model(tmp_path, monkeypatch):
    spec = {'model': 'lorentzian', 'dv': 40., 'params': {'offset': 2.}}
    hod = make_hod(spec)
    monkeypatch.setattr(hod, '_initialize_fit_params', lambda **kwargs: None)
    mocked_specs = []

    def make_mock_cat(tracers, fix_seed, verbose):
        assert tracers == ['LRG']
        assert fix_seed == 42
        mocked_specs.append(deepcopy(hod.args['LRG']['vsmear']))
        return {}

    monkeypatch.setattr(hod, 'make_mock_cat', make_mock_cat)
    monkeypatch.setattr(hod, 'compute_stats', lambda *args, **kwargs: {'xi0': {}})
    params = np.array([(25.,), (50.,)], dtype=[('vsmear_LRG', 'f8')])
    hod.compute_training(training_points=params, path_to_save_point=str(tmp_path), seed=42)

    assert mocked_specs == [dict(spec, dv=25.), dict(spec, dv=50.)]
    for index, dv in enumerate([25., 50.]):
        result = np.load(tmp_path / f'train_hod_{index}.npy', allow_pickle=True).item()
        assert result['LRG']['vsmear'] == dict(spec, dv=dv)
        assert result['hod_fit_param']['vsmear_LRG'] == dv


@pytest.mark.parametrize('spec, model', [
    (None, None), (0., None), (40., 'gaussian'),
    ({'dv': 0.}, None), ({'dv': 40.}, 'gaussian'),
    ({'model': 'lorentzian', 'dv': 0.}, None),
    ({'model': 'lorentzian', 'dv': 40.}, 'lorentzian'),
    ({'model': 'desi'}, 'desi'), ({'model': 'desi', 'dv': None}, 'desi'),
    ({'model': 'desi', 'dv': 0.}, None),
])
def test_fit_names_identify_enabled_velocity_error_models(spec, model):
    name = make_hod(spec, fit_width=False)._get_fit_model_name()
    if model is None:
        assert 'with_zerr' not in name
    else:
        assert f'_with_zerr_{model}' in name


@pytest.mark.parametrize('spec, model', [
    (None, 'gaussian'), (0., 'gaussian'),
    ({'model': 'lorentzian', 'dv': 0.}, 'lorentzian'),
])
def test_fitted_width_names_identify_model_even_when_initially_disabled(spec, model):
    name = make_hod(spec)._get_fit_model_name()
    assert f'+vsmear_{model}' in name
    assert 'with_zerr' not in name
