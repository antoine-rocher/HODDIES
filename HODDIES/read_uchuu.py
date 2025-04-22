from functools import partial
import h5py    
import glob 
import numpy as np 
import multiprocessing
import time 
from mpytools import Catalog, setup_logging
import os 
from cosmoprimo import Cosmology

setup_logging()




def uchuu_cosmo(engine='class'):
    
    """
    Initialize :class:`Cosmology` based on Pinnocchio .param file 

    Parameters
    ----------
    engine: engine for cosmoprimo: class or camb
    
    Returns
    -------
    cosmology : Cosmology
    """
    

    return Cosmology(**dict(Omega0_m=0.3089,
                                    Omega_b=0.0486, 
                                    Omega0_L = 0.6911,
                                    h=0.6774, sigma8=0.8159, n_s=0.9667, w0_fdl=-1., wa_fdl=0),engine=engine)
                        



def read_chunk(chunk, filename, mass_cut=10.8):
    start, end = chunk
    # Reopen the file in the worker process:
    cols = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'log10_Mh', 'Mvir', 'Rvir', 'rs', 'c', 'vrms', 'id']
    dtype = [np.float32]*12 + [np.int64]
    dty = list(zip(cols,dtype))
    data = np.zeros(end-start, dtype=dty)
    with h5py.File(filename, 'r') as f:
        for col in ['x', 'y', 'z', 'vx', 'vy', 'vz', 'Mvir', 'Rvir', 'rs', 'vrms', 'id']:
            data[col] = f[col][start:end]
        # data[-1] = data[cols.index('Rvir')]/data[cols.index('Rs')]
        mask = data['Mvir'] > 10**mass_cut
        mask &= f['pid'][start:end] == -1
        data = data[mask]
        data['log10_Mh'] = np.log10(data['Mvir'])
        data['c'] = data['Rvir']/data['rs']
    return data


def preprocess_uchuu(dir_snap='/global/cfs/cdirs/desi/mocks/cai/Uchuu-SHAM/Uchuu-halo-catalogs/Uchuu_halo_catalogs/halodir_034/', 
                     path_to_sim = '/global/homes/a/arocher/pscratch/Uchuu/Uchuu_halo_catalogs/halodir_034', nchuncks = 256, mass_cut=10.8):

    pool = multiprocessing.Pool(nchuncks)
    print('pool done')
    files=glob.glob(os.path.join(dir_snap,'*.h5'))
    files.sort()
    cols = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'log10_Mh', 'Mh', 'Rh', 'Rs', 'c', 'Vrms', 'halo_id']
    size_cat = []
    for ii, file in enumerate(files):
        cat = h5py.File(file, 'r')    
        chunk = np.linspace(0, cat['Mvir'].size, nchuncks, dtype=int)
        chunks = list(zip(chunk[:-1], chunk[1:]))
        cat.close()
        st = time.time()
        hcat_tmp = np.hstack(pool.map(partial(read_chunk, filename=file, mass_cut=mass_cut), chunks))
        hcat_tmp.dtype.names = cols
        Catalog.from_array(hcat_tmp).save(os.path.join(path_to_sim, f'halo_Mvir1e108_slab{str(ii).zfill(2)}.npy'))
        size_cat += [hcat_tmp.size]
        print(ii, time.time()-st, flush=True)

    np.savetxt(os.path.join(path_to_sim, 'Catalog_row_numbers_list.txt'), size_cat, header='# Nb of rows in each catalogs')





def read_uchuu_cat(path_to_sim ='/global/homes/a/arocher/pscratch/Uchuu/Uchuu_halo_catalogs/halodir_034'):

    files = glob.glob(os.path.join(path_to_sim, 'halo_Mvir1e108_slab*.npy'))
    files.sort()
    nb_row = np.loadtxt(os.path.join(path_to_sim, 'Catalog_row_numbers_list.txt'))
    hcat = None
    tmp_size = 0
    for i, file in enumerate(files):
        hcat_tmp = Catalog.load(file)
        if hcat is None: 
            hcat = Catalog.from_array(np.zeros(int(nb_row.sum()), dtype=hcat_tmp.to_array().dtype))
        hcat[tmp_size:tmp_size+hcat_tmp.size] = hcat_tmp
        tmp_size += hcat_tmp.size
    return hcat
    
