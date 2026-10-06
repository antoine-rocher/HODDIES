"""Compute assembly-bias fields with canonical and older notebook imports."""

import importlib
from pathlib import Path

import numba
import numpy as np
import pytest
from mpytools import Catalog

from HODDIES import HOD
from HODDIES.sim_loader import AbacusSummitSim


@pytest.mark.parametrize('legacy_import', [False, True])
@pytest.mark.parametrize('existing_cache_directory', [False, True])
def test_assembly_bias_computes_and_reloads_environment(
        tmp_path, monkeypatch, legacy_import, existing_cache_directory):
    numba.set_num_threads(2)
    loader_class = AbacusSummitSim
    if legacy_import:
        # Reproduce a notebook that created `test` before fixing its imports.
        monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'HODDIES'))
        loader_class = importlib.import_module('sim_loader.abacus_io').AbacusSummitSim
        assert loader_class.__module__ == 'sim_loader.abacus_io'
    loader = loader_class.__new__(loader_class)
    loader.init_logger(setup_logger=False)
    loader.z_simu, loader.boxsize, loader.nthreads = .5, 64., 2
    loader.sim_name = 'AbacusSummit_small_c000_ph3000'
    loader._root_dir_sim = str(tmp_path / 'simulation')
    snapshot = Path(loader._root_dir_sim) / loader.sim_name / 'halos' / 'z0.500'
    snapshot.mkdir(parents=True)
    positions = np.random.default_rng(42).uniform(0., loader.boxsize, (1024, 3)).astype('f4')
    loader.field_particles = {'pos': positions}
    loader.part_subsamples = None
    loader.cosmo = lambda **kwargs: None
    size = 32
    loader.hcat = Catalog.from_dict({
        'x': positions[:size, 0], 'y': positions[:size, 1], 'z': positions[:size, 2],
        'vx': np.zeros(size), 'vy': np.zeros(size), 'vz': np.zeros(size),
        'halo_id': np.arange(size), 'Mh': np.full(size, 1.e13),
        'log10_Mh': np.full(size, 13.), 'Rh': np.full(size, 350.),
        'Vrms': np.full(size, 200.), 'c': np.linspace(3., 8., size),
    })
    cache_directory = tmp_path / 'fields'
    if existing_cache_directory:
        cache_directory.mkdir()
    hod = HOD(
        loader, tracers=['LRG'], use_assembly_bias=True, nthreads=2,
        cell_size=8., R=1.5, dir_to_save_env_mesh=str(cache_directory), setup_logger=False,
        LRG=dict(HOD_model='SHOD', Ac=.6, log_Mcent=12., sigma_M=.2,
                 satellites=False, density=None),
    )
    for name in ('density_mesh', 'shear_mesh'):
        mesh = getattr(loader, name)
        assert mesh.shape == (8, 8, 8)
        assert np.all(np.isfinite(mesh))
    for proxy in ('c', 'env', 'shear'):
        assert f'ab_{proxy}' in loader.columns
        assert np.all(np.isfinite(loader.hcat[f'ab_{proxy}']))
    mock = hod.make_mock_cat(fix_seed=42, verbose=False)
    assert mock.size > 0
    saved_density = loader.density_mesh.copy()
    saved_shear = loader.shear_mesh.copy()
    cache = cache_directory / f'env_shear_map_{loader.sim_name}_z0.500.h5'
    assert cache.is_file()

    # A cached run must work without loading or computing particle fields.
    loader.density_mesh = loader.shear_mesh = loader.field_particles = None
    def unexpected_particle_load(**kwargs):
        pytest.fail('A cached environment should not load particles.')
    monkeypatch.setattr(loader, 'load_particle_field', unexpected_particle_load)
    loader.load_env_based_properties(cell_size=8., R=1.5,
                                     dir_to_save_env_mesh=str(cache_directory))
    np.testing.assert_array_equal(loader.density_mesh, saved_density)
    np.testing.assert_array_equal(loader.shear_mesh, saved_shear)
