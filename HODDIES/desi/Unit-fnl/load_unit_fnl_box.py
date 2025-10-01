from functools import partial
import h5py    
import numpy as np 
import multiprocessing
import time 
from mpytools import Catalog
import pandas as pd

def read_chunk(chunk, filename, col_names, mass_cut=10.8):
    start, end = chunk
    # Reopen the file in the worker process:
    data = pd.read_hdf(filename,key='rockstar_catalog',columns=col_names, start=start, stop=end)
    pid = pd.read_hdf(filename,key='rockstar_catalog',columns=['PID'], start=start, stop=end)
    mask_mass = np.ones(data['Mvir'].size, dtype=bool) if mass_cut is None else (data['Mvir'].to_numpy() > 10**mass_cut)
    mask_pid = (pid.to_numpy() == -1).flatten()
    mask_halo = mask_pid & mask_mass
    data = data[mask_halo]
    data['log10_Mh'] = np.log10(data['Mvir'])
    data['c'] = data['Rvir']/data['Rs']

    cols_outputs = [
        'x', 'y', 'z', 'vx', 'vy', 'vz', 'Mh', 'Rh', 'Rs',
        'Vrms', 'halo_id', 'log10_Mh', 'c']
    data.rename(columns=dict(zip(data.columns,cols_outputs)), inplace=True)
    dtype={}
    for name in data.columns:
        if name == 'halo_id':
            dtype[name] = np.int64
        else:
            dtype[name] = np.float32
    return data.astype(dtype).to_records(index=False)



nchuncks = 256
filename = '/global/cfs/projectdirs/desi/mocks/UNIT/PNG-UNITsim-XL/fnl20/PNG_UNITsim_XL_N6144_L3000_fnl20_z1_12.h5'
cat = h5py.File(filename, 'r')    
chunk = np.linspace(0, cat['rockstar_catalog']['table'].size, nchuncks, dtype=int)
chunks = list(zip(chunk[:-1], chunk[1:]))
cat.close()


mass_cut=11.5
st = time.time() 
cols = ['X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'Mvir','Rvir', 'Rs', 'Vrms','#ID']
st = time.time()
with multiprocessing.Pool(nchuncks) as p:
    print('pool done', time.time()-st)
    st = time.time()
    res = p.map(partial(read_chunk, filename=filename, col_names=cols, mass_cut=mass_cut), chunks)
    p.close()

    
print('Done', time.time()-st)
st = time.time()

res = np.hstack(res)
print('stack done', time.time()-st)
print(Catalog.from_array(res))


# hcat_tmp.dtype.names = cols
# Catalog.from_array(hcat_tmp).save(os.path.join(path_to_sim, f'halo_Mvir1e113_slab{str(ii).zfill(2)}.npy'))
# size_cat += [hcat_tmp.size]
# print(time.time()-st, flush=True)

# uchuu_halo = np.hstack([res[i] for i in range(len(res))])
