"""Assembly-bias ranking when constructing HODs or enabling it in a notebook."""

import numba
import numpy as np
import pytest
from mpytools import Catalog

from HODDIES import HOD
from HODDIES.sim_loader import Base_catalogue


@pytest.mark.parametrize('enable_at_creation', [False, True])
@pytest.mark.parametrize('bin_edges', [None, [12., 14.]])
def test_assembly_bias_uses_mass_bins_and_generates_mock(enable_at_creation, bin_edges):
    numba.set_num_threads(2)
    size = 32
    index = np.arange(size, dtype=float)
    loader = Base_catalogue.__new__(Base_catalogue)
    loader.init_logger(setup_logger=False)
    loader.hcat = Catalog.from_dict({
        'x': index, 'y': index + 2., 'z': index + 4.,
        'vx': index, 'vy': index + 1., 'vz': index + 2.,
        'halo_id': np.arange(size), 'Mh': np.full(size, 1.e13),
        'log10_Mh': np.full(size, 13.), 'Rh': np.full(size, 350.),
        'Vrms': np.full(size, 200.), 'c': index + 3.,
        'env': index[::-1].copy(), 'shear': (index + 1.) ** 2,
    })
    loader.part_subsamples = loader.field_particles = None
    loader.z_simu, loader.boxsize = .5, 100.
    loader.cosmo = lambda **kwargs: None
    options = {} if bin_edges is None else {'bins': np.asarray(bin_edges)}
    hod = HOD(
        loader, tracers=['LRG'], nthreads=2, setup_logger=False,
        use_assembly_bias=enable_at_creation, **options,
        LRG=dict(HOD_model='SHOD', Ac=.6, log_Mcent=12., sigma_M=.2,
                 satellites=False, density=None,
                 assembly_bias={'c': [.4, 0.], 'env': [0., 0.], 'shear': [0., 0.]}),
    )
    if not enable_at_creation:
        assert all(f'ab_{proxy}' not in loader.columns for proxy in ('c', 'env', 'shear'))
        hod.args['use_assembly_bias'] = True
    mock = hod.make_mock_cat(fix_seed=42, verbose=False)
    assert mock.size > 0
    np.testing.assert_array_equal(mock['TRACER'], 'LRG')
    np.testing.assert_array_equal(mock['Central'], 1)
    for proxy in ('c', 'env', 'shear'):
        ranks = loader.hcat[f'ab_{proxy}']
        assert ranks.shape == (size,)
        assert np.all(np.isfinite(ranks))
        np.testing.assert_allclose([ranks.min(), ranks.max()], [-.5, .5])
    # Cached ranks must support another seeded call, too.
    repeated = hod.make_mock_cat(fix_seed=42, verbose=False)
    np.testing.assert_array_equal(repeated['halo_id'], mock['halo_id'])
