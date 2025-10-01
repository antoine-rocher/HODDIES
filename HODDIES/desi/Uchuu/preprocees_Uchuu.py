from functools import partial
import hdf5plugin
import h5py    
import glob 
import numpy as np 
import multiprocessing
import time 
from mpytools import Catalog, setup_logging
import os 
import argparse

def get_z_from_snapshot(halodir):
    Uchuu_snapshot_redshifts = {
        '001': '13.960',
        '002': '12.690',
        '003': '11.510',
        '004': '10.440',
        '005': '9.470',
        '006': '8.580',
        '007': '7.760',
        '008': '7.020',
        '009': '6.340',
        '010': '5.730',
        '011': '5.160',
        '012': '4.630',
        '013': '4.270',
        '014': '3.930',
        '015': '3.610',
        '016': '3.310',
        '017': '3.130',
        '018': '2.950',
        '019': '2.780',
        '020': '2.610',
        '021': '2.460',
        '022': '2.300',
        '023': '2.160',
        '024': '2.030',
        '025': '1.900',
        '026': '1.770',
        '027': '1.650',
        '028': '1.540',
        '029': '1.430',
        '030': '1.320',
        '031': '1.220',
        '032': '1.120',
        '033': '1.030',
        '034': '0.940',
        '035': '0.860',
        '036': '0.780',
        '037': '0.700',
        '038': '0.630',
        '039': '0.560',
        '040': '0.490',
        '041': '0.430',
        '042': '0.360',
        '043': '0.300',
        '044': '0.250',
        '045': '0.190',
        '046': '0.140',
        '047': '0.093',
        '048': '0.045',
        '049': '0.022',
        '050': '0.000'
    }
    return Uchuu_snapshot_redshifts[halodir]


def save_catalogue(filename, main, subs):
    """
    Save main halos and subhalos in one HDF5 file using Blosc compression.
    
    Parameters
    ----------
    filename : str
        Output HDF5 file name.
    main : np.ndarray
        Structured array of main halos.
    subs : np.ndarray
        Structured array of subhalos.
    """
    with h5py.File(filename, "w") as f:
        # Main halos
        g_main = f.create_group("main")
        for name in main.dtype.names:
            g_main.create_dataset(
                name,
                data=main[name],
                **hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
            )
        
        # Subhalos
        g_sub = f.create_group("sub")
        for name in subs.dtype.names:
            g_sub.create_dataset(
                name,
                data=subs[name],
                **hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
            )

def read_chunk(chunk, filename, mass_cut=10.8):
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
        'ID', 'PID', 'UPID'
    ]
    
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

        upid_col = f[resolved['UPID']][start:end]
        # Masks
        mask_mass = (data['Mh'] > 10**mass_cut)
        mask_pid = f[pid_name][start:end] == -1
        mask_halo = mask_pid & mask_mass
    
    # Derived columns
    data['log10_Mh'] = np.log10(data['Mh'])
    data['c'] = data['Rh'] / data['Rs']
    
    mask_subhalo = (~mask_pid) & (mask_mass)
    data_sub = np.zeros(mask_subhalo.sum(), dtype=dtype+[('upid', np.int64)])
    
    for col in data.dtype.names:
        data_sub[col] = data[col][mask_subhalo]
    data_sub['upid'] = upid_col[mask_subhalo]
    return data[mask_halo], data_sub



def read_halos_subhalos(filename, read_subs=True):
    """
    Read halos saved with save_halos_grouped() into structured NumPy arrays.
    
    Parameters
    ----------
    filename : str
        Input HDF5 file.
    read_subs : bool, optional
        If True (default), read both main halos and subhalos.
        If False, read only main halos.
    
    Returns
    -------
    main : np.ndarray
        Structured array of main halos.
    subs : np.ndarray or None
        Structured array of subhalos if read_subs=True, else None.
    """
    with h5py.File(filename, "r") as f:
        # --- Main halos ---
        g_main = f["main"]
        names = list(g_main.keys())
        n = g_main[names[0]].shape[0]  # number of rows
        dtype = [(name, g_main[name].dtype) for name in names]
        main = np.zeros(n, dtype=dtype)
        for name in names:
            main[name] = g_main[name][:]
        
        # --- Subhalos (optional) ---
        subs = None
        if read_subs and "sub" in f:
            g_sub = f["sub"]
            names = list(g_sub.keys())
            n = g_sub[names[0]].shape[0]
            dtype = [(name, g_sub[name].dtype) for name in names]
            subs = np.zeros(n, dtype=dtype)
            for name in names:
                subs[name] = g_sub[name][:]
    
    return main, subs

if __name__ == "__main__":  

    parser = argparse.ArgumentParser()
    parser.add_argument('--halodir', help='halodir name of Uchuu sim i.e. 033', type=str)
    parser.add_argument('--dir_name', help='root directory of Uchuu sim i.e. /dvs_ro/cfs/cdirs/desi/mocks/cai/Uchuu-SHAM/Uchuu-halo-catalogs (default)', default='/dvs_ro/cfs/cdirs/desi/mocks/cai/Uchuu-SHAM/Uchuu-halo-catalogs/', type=str)
    parser.add_argument('--save_dir', help='root directory of Uchuu sim i.e. /pscratch/sd/a/arocher/Uchuu/Uchuu_halo_catalogs/ (default)', default='/pscratch/sd/a/arocher/Uchuu/Uchuu_halo_catalogs/', type=str)
    parser.add_argument('--batch_num', help='batch number 0 to 10, save one catalog each 10 halolist', type=int)

    args = parser.parse_args()

    
    nchuncks = 32
    dir_name = args.dir_name
    halodir = args.halodir.zfill(3)
    
    save_name = os.path.join(args.save_dir, f'halodir_{halodir}', f'Uchuu_halodir_{halodir}_halos_subhalos_z{get_z_from_snapshot(halodir)}_{str(args.batch_num).zfill(2)}.h5')
    # if os.path.exists(save_name): exit()
    filenames = glob.glob(os.path.join(dir_name, f'halodir_{halodir}', '*.h5'))
    filenames.sort()
    mass_cut=10.8
    st = time.time()
    uchuu_halo_tmp, uchuu_subhalo_tmp = [], []
    with multiprocessing.Pool(nchuncks) as p:
        print('pool done', time.time() - st)
        for ii, fn in enumerate(filenames[10*args.batch_num:10*args.batch_num+10]):
            st1 = time.time()
            print('Reading file', fn)
            cat = h5py.File(fn, 'r')    
            chunk = np.linspace(0, cat[list(cat.keys())[0]].size, nchuncks, dtype=int)
            chunks = list(zip(chunk[:-1], chunk[1:]))
            cat.close()
            res = p.map(partial(read_chunk, filename=fn, mass_cut=mass_cut), chunks)
            uchuu_halo_tmp += [np.hstack([res[i][0] for i in range(len(res))])]
            uchuu_subhalo_tmp += [np.hstack([res[i][1] for i in range(len(res))])]
            print('Done', time.time() - st1)
        p.close()
    # hcat_tmp.dtype.names = cols
    # Catalog.from_array(hcat_tmp).save(os.path.join(path_to_sim, f'halo_Mvir1e113_slab{str(ii).zfill(2)}.npy'))
    # size_cat += [hcat_tmp.size]
    print(time.time()-st, flush=True)
    st = time.time()
    uchuu_halo_tmp =np.concatenate(uchuu_halo_tmp)
    uchuu_subhalo_tmp =np.concatenate(uchuu_subhalo_tmp)
    print('Concat done', time.time()-st, flush=True)
    # uchuu_halo = np.hstack([res[i][0] for i in range(len(res))])
    # uchuu_subhalo = np.hstack([res[i][1] for i in range(len(res))])
    
    os.makedirs(os. path. dirname(save_name), exist_ok=True)
    print(f'Write {save_name}')
    st = time.time()
    save_catalogue(save_name, uchuu_halo_tmp, uchuu_subhalo_tmp)
    print(f'Done ', time.time()-st)
