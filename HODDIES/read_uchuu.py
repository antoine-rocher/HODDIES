""" Io tools to load Uchuu simualtions"""

from functools import partial
import hdf5plugin
import h5py
import glob 
import numpy as np 
import multiprocessing
import time 
from mpytools import Catalog
import os 
from cosmoprimo import Cosmology


def read_chuncks_2gpc(chunks, filename, group='main', mass_cut=None):
    start, end = chunks
    with h5py.File(filename, 'r') as f:
        dtype = [(col, f[group][col].dtype) for col in f[group].keys()]
        data = np.zeros(end - start, dtype=dtype)
        for col in f[group].keys():
            data[col] = f[group][col][start:end]
        if mass_cut is not None:
            return data[data['log10_Mh'] > mass_cut]
    return data


def read_chunk_DDE(chunk, filename, mass_cut=10.8, load_subhalos=False):
    start, end = chunk
    
    # Standardized output column names
    col_output = [
        'x', 'y', 'z', 'vx', 'vy', 'vz',
        'Rs', 'Rh', 'c', 'Mh', 'log10_Mh',
        'Vrms', 'halo_id'
    ]
    
    # Expected input names (but case-insensitive except Mvir, Rvir)
    col_input_names = [
        'X', 'Y', 'Z', 'VX', 'VY', 'VZ',
        'Rs', 'Mvir', 'Rvir', 'Vrms',
        'ID', 'PID'
    ]
    # if load_subhalos:
    #     load_subhalos=False
    #     print('Can not use subhalos with Uchuu DDE simulations', flush=True)
    # if load_subhalos:
    #     col_input_names += ['UPID']
    
    # Open once to inspect keys
    with h5py.File(filename, 'r') as f:
        # Build lowercase lookup: lowercase -> actual key
        key_map = {k.lower(): k for k in f.keys()}
        
        # Resolve actual names (case-insensitive)
        resolved = {}
        for name in col_input_names:
            if name.lower() in key_map:
                resolved[name] = key_map[name.lower()]
            else:
                raise KeyError(f"Column {name} not found in file keys {list(f.keys())}")
        
        # Special names for convenience
        pid_name = resolved['PID']
    
    # Define dtypes: float32 for all, except halo_id as int64
    dtype = []
    for name in col_output:
        if name == 'halo_id':
            dtype.append((name, np.int64))
        else:
            dtype.append((name, np.float32))
    
    # Initialize data array
    data = np.zeros(end - start, dtype=dtype)
    
    # Read the actual data
    with h5py.File(filename, 'r') as f:
        
        data['x']  = f[resolved['X']][start:end]
        data['y']  = f[resolved['Y']][start:end]
        data['z']  = f[resolved['Z']][start:end]
        data['vx'] = f[resolved['VX']][start:end]
        data['vy'] = f[resolved['VY']][start:end]
        data['vz'] = f[resolved['VZ']][start:end]
        data['Rs'] = f[resolved['Rs']][start:end]
        data['Mh'] = f[resolved['Mvir']][start:end]
        data['Rh'] = f[resolved['Rvir']][start:end]
        data['Vrms'] = f[resolved['Vrms']][start:end]
        data['halo_id'] = f[resolved['ID']][start:end]

        if load_subhalos: upid_col = f[resolved['UPID']][start:end]
        # Masks
        mask_mass = np.ones(data['Mh'].size, dtype=bool) if mass_cut is None else (data['Mh'] > 10**mass_cut)
        mask_pid = f[pid_name][start:end] == -1
        mask_halo = mask_pid & mask_mass
    
    # Derived columns
    data['log10_Mh'] = np.log10(data['Mh'])
    data['c'] = data['Rh'] / data['Rs']
    if load_subhalos:
        mask_subhalo = (~mask_pid) & (mask_mass)
        data_sub = np.zeros(mask_subhalo.sum(), dtype=dtype+[('upid', np.int64)])
        
        for col in data.dtype.names:
            data_sub[col] = data[col][mask_subhalo]
        data_sub['upid'] = upid_col[mask_subhalo]
        return data[mask_halo], data_sub
    return data[mask_halo], None



def get_DDE_snapshot_file(z):
    """
    Return the Rockstar snapshot file for an exact redshift z.
    Raise ValueError if z is not in the available list.
    """
    snapshots = {
        2.03: "out_1.rockstar.h5",
        1.77: "out_2.rockstar.h5",
        1.54: "out_3.rockstar.h5",
        1.30: "out_4.rockstar.h5",
        1.03: "out_5.rockstar.h5",
        0.78: "out_6.rockstar.h5",
        0.49: "out_7.rockstar.h5",
        0.30: "out_8.rockstar.h5",
        0.19: "out_9.rockstar.h5",
        0.00: "out_10.rockstar.h5",
    }
    
    if z in snapshots:
        return snapshots[z]
    else:
        valid = ", ".join(map(str, snapshots.keys()))
        raise ValueError(f"Redshift {z} not available. Valid redshifts are: {valid}")

def get_Uchuu_snapshot_file(z):
    Uchuu_snapshot_redshifts = {
        "13.960": 1,
        "12.690": 2,
        "11.510": 3,
        "10.440": 4,
        "9.470": 5,
        "8.580": 6,
        "7.760": 7,
        "7.020": 8,
        "6.340": 9,
        "5.730": 10,
        "5.160": 11,
        "4.630": 12,
        "4.270": 13,
        "3.930": 14,
        "3.610": 15,
        "3.310": 16,
        "3.130": 17,
        "2.950": 18,
        "2.780": 19,
        "2.610": 20,
        "2.460": 21,
        "2.300": 22,
        "2.160": 23,
        "2.030": 24,
        "1.900": 25,
        "1.770": 26,
        "1.650": 27,
        "1.540": 28,
        "1.430": 29,
        "1.320": 30,
        "1.220": 31,
        "1.120": 32,
        "1.030": 33,
        "0.940": 34,
        "0.860": 35,
        "0.780": 36,
        "0.700": 37,
        "0.630": 38,
        "0.560": 39,
        "0.490": 40,
        "0.430": 41,
        "0.360": 42,
        "0.300": 43,
        "0.250": 44,
        "0.190": 45,
        "0.140": 46,
        "0.093": 47,
        "0.045": 48,
        "0.022": 49,
        "0.000": 50
    }
    if z in Uchuu_snapshot_redshifts:
        return f'halodir_{str(Uchuu_snapshot_redshifts[z]).zfill(3)}'
    else:
        valid = ", ".join(map(str, Uchuu_snapshot_redshifts.keys()))
        raise ValueError(f"Redshift {z} not available. Valid redshifts are: {valid}")

def read_Uchuu(sim_name, z_snapshot, path_to_sim='/pscratch/sd/a/arocher/Uchuu/', nchuncks=32, load_subhalos=False, mass_cut=10.8):
    opt_ch = 128 if sim_name=='Uchuu2Gpc' else 32
    nchuncks = min(nchuncks, opt_ch)
    DDE_sims = ['Planck18', 'Planck18_DDE', 'DESIY1_DDE', 'Uchuu2Gpc']
    if sim_name not in DDE_sims:
        raise ValueError(f"Uchuu DDE simulation names {sim_name} not available. Valid names are: {DDE_sims}")
    dirname_hcat = os.path.join(path_to_sim, 'Uchuu_halo_catalogs', get_Uchuu_snapshot_file('{:.3f}'.format(z_snapshot)), '*') if sim_name=='Uchuu2Gpc' else os.path.join(path_to_sim,'DDE', sim_name, get_DDE_snapshot_file(z_snapshot))
    
    if mass_cut is not None:
        print(f'Apply mass cut at 10^{mass_cut} M_sol/h', flush=True)

    if load_subhalos & (sim_name!='Uchuu2Gpc'):
        load_subhalos=False
        print('Can not use subhalos with Uchuu DDE simulations', flush=True)
    subh = 'with subhalos' if load_subhalos else ''
    print(f'Load Uchuu {sim_name} simulation at z={z_snapshot:.3f} {subh}', flush=True)
    filenames = glob.glob(dirname_hcat)
    print(f'Reading {filenames}', flush=True)
    st = time.time()
    with multiprocessing.Pool(nchuncks) as p:
        uchuu_halo, uchuu_subhalo = [], []
        for fn in filenames:
            print('Reading file', fn, flush=True)
            if sim_name=='Uchuu2Gpc':
                cat = Catalog.read(fn, group='main')
                chunk = np.linspace(0, cat.csize, nchuncks, dtype=int)
                chunks = list(zip(chunk[:-1], chunk[1:]))
                res = p.map(partial(read_chuncks_2gpc, filename=fn, group='main', mass_cut=mass_cut), chunks)
                uchuu_halo += [np.concatenate(res)]
                if load_subhalos:
                    cat = Catalog.read(fn, group='sub')
                    chunk = np.linspace(0, cat.csize, nchuncks, dtype=int)
                    chunks = list(zip(chunk[:-1], chunk[1:]))
                    res_sub = p.map(partial(read_chuncks_2gpc, filename=fn, group='sub', mass_cut=mass_cut), chunks)
                    uchuu_subhalo += [np.concatenate(res_sub)]
            else:
                cat = h5py.File(fn, 'r')    
                chunk = np.linspace(0, cat[list(cat.keys())[0]].size, nchuncks, dtype=int)
                chunks = list(zip(chunk[:-1], chunk[1:]))
                cat.close()
                res = p.map(partial(read_chunk_DDE, filename=fn, mass_cut=mass_cut, load_subhalos=load_subhalos), chunks)
        p.close()
    # hcat_tmp.dtype.names = cols
    # Catalog.from_array(hcat_tmp).save(os.path.join(path_to_sim, f'halo_Mvir1e113_slab{str(ii).zfill(2)}.npy'))
    # size_cat += [hcat_tmp.size]
    # print('Reading done', time.time()-st, flush=True)
    if sim_name=='Uchuu2Gpc':
        uchuu_halo = Catalog.from_array(np.concatenate(uchuu_halo))
        if load_subhalos:
            uchuu_subhalo = Catalog.from_array(np.concatenate(uchuu_subhalo))
        else:
            uchuu_subhalo = None
        # print('Reading done', time.time()-st, flush=True)
    else:
        uchuu_halo = Catalog.from_array(np.hstack([res[i][0] for i in range(len(res))]))
        print('Done', time.time()-st, flush=True)
        if load_subhalos:
            uchuu_subhalo = Catalog.from_array(np.hstack([res[i][1] for i in range(len(res))]))
        else:
            uchuu_subhalo=None
        # print('Reading done', time.time()-st, flush=True)
    return uchuu_halo, uchuu_subhalo
    


def UchuuPlanck2018(engine=None, extra_params=None, **params):
    """
    Initialize :class:`Cosmology` based on Table 4 Planck2015 TT,TE,EE+lowP+lensing.
    
    Parameters
    ----------
    engine : string, default=None
        Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_no
wiggle', 'bbks'].
        If ``None``, returns current :attr:`Cosmology.engine`.

    extra_params : dict, default=None
        Extra engine parameters, typically precision parameters.

    params : dict
        Cosmological and calculation parameters which take priority over the default ones.

    Returns
    -------
    cosmology : Cosmology
    """
    from cosmoprimo import constants
    default_params = dict(h=0.6766, Omega_b=0.048975, Omega_k=0., sigma8=0.8102, k_pivot=0.05, n_s=0.9665, 
                          m_ncdm=[0.06], neutrino_hierarchy=None, T_ncdm_over_cmb=constants.TNCDM_OVER_CMB, N_eff=constants.NEFF, 
                          tau_reio=0.063, A_L=1.0, w0_fld=-1., wa_fld=0.)
    return Cosmology(engine=engine, extra_params=extra_params, **default_params).clone(**params)


def UchuuPlanck2018DDE(engine=None, extra_params=None, **params):
    """
    Initialize :class:`Cosmology` based on Table 4 Planck2015 TT,TE,EE+lowP+lensing.
    
    Parameters
    ----------
    engine : string, default=None
        Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_no
wiggle', 'bbks'].
        If ``None``, returns current :attr:`Cosmology.engine`.

    extra_params : dict, default=None
        Extra engine parameters, typically precision parameters.

    params : dict
        Cosmological and calculation parameters which take priority over the default ones.

    Returns
    -------
    cosmology : Cosmology
    """
    from cosmoprimo import constants
    default_params = dict(h=0.6766, Omega_b=0.048975, Omega_k=0., sigma8=0.8102, k_pivot=0.05, n_s=0.9665,
                          m_ncdm=[0.06], neutrino_hierarchy=None, T_ncdm_over_cmb=constants.TNCDM_OVER_CMB, N_eff=constants.NEFF, 
                          tau_reio=0.063, A_L=1.0, w0_fld=-0.45, wa_fld=-1.79)
    return Cosmology(engine=engine, extra_params=extra_params, **default_params).clone(**params)

def UchuuDESIY1DDE(engine=None, extra_params=None, **params):
    """
    Initialize :class:`Cosmology` based on Table 4 Planck2015 TT,TE,EE+lowP+lensing.
    
    Parameters
    ----------
    engine : string, default=None
        Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_no
wiggle', 'bbks'].
        If ``None``, returns current :attr:`Cosmology.engine`.

    extra_params : dict, default=None
        Extra engine parameters, typically precision parameters.

    params : dict
        Cosmological and calculation parameters which take priority over the default ones.

    Returns
    -------
    cosmology : Cosmology
    """
    from cosmoprimo import constants
    default_params = dict(h=0.6470, Omega_m=0.3440, Omega_b=0.048975 , Omega_k=0., sigma8=0.8102, k_pivot=0.05, n_s=0.9665, 
                          m_ncdm=[0.06], neutrino_hierarchy=None, T_ncdm_over_cmb=constants.TNCDM_OVER_CMB, N_eff=constants.NEFF, 
                          tau_reio=0.063, A_L=1.0, w0_fld=-0.45, wa_fld=-1.79)
    return Cosmology(engine=engine, extra_params=extra_params, **default_params).clone(**params)
