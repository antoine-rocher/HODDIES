import numpy as np
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import time 
from numba import njit, numba
import os 
import socket
import fitsio
import sys
from mpytools import Catalog


def read_Abacus_hcat(args, use_L2=True, halo_lc=False):
    if os.getenv('NERSC_HOST') == 'perlmutter':
        dir_sim = '/global/cfs/cdirs/desi/cosmosim/Abacus' 
    elif socket.gethostname() == 'antoine-ThinkPad-P1-Gen-6':
        dir_sim = '/home/antoine/Bureau/Transfert/postdoc/Abacus_sims'
    if use_L2:
        Lsuff = 'L2'
    else : 
        Lsuff = ''
    
    usecols=['id', f'x_{Lsuff}com', f'v_{Lsuff}com', 'N', f'r25_{Lsuff}com', f'r98_{Lsuff}com', f'sigmav3d_{Lsuff}com'] if not halo_lc else ['index_halo', f'pos_interp', f'vel_interp', 'N_interp', 'redshift_interp']
    str_z = format(args['hcat']["z_simu"], ".3f")
    if 'small' in args['hcat']['sim_name']:
        path_to_sim = os.path.join(dir_sim, 'small',
                                    args['hcat']['sim_name'], "halos", "z{:.3f}".format(args['hcat']["z_simu"]))
    elif halo_lc:
        if 'z_simu_lc' not in args['hcat'].keys():
            args['hcat']["z_simu_lc"] = [args['hcat']["z_simu"]]
        if not isinstance(args['hcat']["z_simu_lc"], list):
            args['hcat']["z_simu_lc"] = [args['hcat']["z_simu_lc"]]
        path_to_sim = [os.path.join(dir_sim, 'halo_light_cones',
                                    args['hcat']['sim_name'], "z{:.3f}".format(z_lc)) for z_lc in args['hcat']["z_simu_lc"]]
    else:
        path_to_sim = os.path.join(dir_sim, 
                                    args['hcat']['sim_name'], "halos",  "z{:.3f}".format(args['hcat']["z_simu"]))
    tt = time.time()

    hcat, part_subsamples, boxsize, origin = load_hcat_from_Abacus(path_to_sim, usecols, Lsuff, halo_lc,
                                                  Nthread=args['nthreads'],  mass_cut=args['hcat']['mass_cut'], use_particles=args['hcat']['load_particles'])
    print(f'{args["hcat"]["sim_name"]} at {args["hcat"]["z_simu"]} loaded, took {time.strftime("%H:%M:%S",time.gmtime(time.time() - tt))}', flush=True)
    return hcat, part_subsamples, boxsize, origin



def load_CompaSO(path_to_sim, usecols, mass_cut = None, halo_lc=False, use_particles=None):
    """
    --- Function to load AbacusSummit halo catalogs
    """

    if use_particles:
        load_part = True
        usecols += ['npstartA', 'npoutA']
    else:
        load_part = False
        
    if halo_lc: 
        hcats = [CompaSOHaloCatalog(path, fields=usecols,  subsamples=dict(A=load_part), cleaned=True) for path in path_to_sim]
        hcat_i = np.concatenate([cat.halos for cat in hcats])
        header = hcats[0].header
    else:
        hcat = CompaSOHaloCatalog(f"{path_to_sim}", fields=usecols,  subsamples=dict(A=load_part), cleaned=True)
        hcat_i, header = hcat.halos, hcat.header
    
    part_subsamples = hcat.subsamples if use_particles else None
    n_p = 'N' if not halo_lc else 'N_interp'
    N = mass_cut/header['ParticleMassHMsun'] if mass_cut is not None else 0

    return hcat_i[hcat_i[n_p] > N], header, part_subsamples




@njit(parallel=True, fastmath=True)
def compute_col_from_Abacus(N, pos, vel, ParticleMassHMsun, 
                            x, y, z, vx, vy, vz, 
                            Mvir, log10_Mh, Rs, Rvir, c,
                            r25, r98, Nthread):
    """
    --- Function to prepare colunms for the halo catalog 
    """
    
    numba.set_num_threads(Nthread)
    # starting index of each thread
    hstart = np.rint(np.linspace(0, len(N), Nthread + 1))
    # figuring out the number of halos kept for each thread

    for tid in numba.prange(Nthread):
        for i in range(int(hstart[tid]), int(hstart[tid + 1])):
            Mvir[i] = (N[i]*ParticleMassHMsun)
            log10_Mh[i] = np.log10(Mvir[i])
            x[i], y[i], z[i] = pos[i]
            vx[i], vy[i], vz[i] = vel[i]
            Rs[i] = r25[i]*1000
            Rvir[i] = r98[i]*1000
            c[i] = r98[i]/r25[i]        

       
def load_hcat_from_Abacus(path_to_sim, usecols, Lsuff, halo_lc=False, Nthread=64, mass_cut=None, verbose=True, use_particles=None):
    """
    --- Function which returns Abacus halo catalog for HOD studies
    """
    if verbose :
        start = time.time()
        ld_part = 'with particles' if use_particles else ''
        print(f"Load Compaso cat from {path_to_sim} {ld_part}...")

    hcat, header, part_subsamples = load_CompaSO(path_to_sim, mass_cut=mass_cut, usecols=usecols, halo_lc=halo_lc, use_particles=use_particles)
    if verbose :
        print(f"Done took", time.strftime("%H:%M:%S",time.gmtime(time.time() - start)), flush=True)
        start = time.time()
        print("Compute columns...")
    
    n_p, pos, vel, index = ('N', f'x_{Lsuff}com', f'v_{Lsuff}com', 'id') if not halo_lc else ('N_interp', 'pos_interp', 'vel_interp', 'index_halo')
    
    dic = dict((col, np.empty(hcat[n_p].size, dtype='float32')) 
                   for col in ['x', 'y','z','vx','vy','vz', 'Rs','Rh', 'c', 'Mh', 'log10_Mh'])
    dic['Vrms'] = np.array(hcat[f'sigmav3d_{Lsuff}com'])
    if use_particles:
        dic['npstartA'] = np.array(hcat['npstartA'], dtype='int64')
        dic['npoutA'] = np.array(hcat['npoutA'], dtype='int64')
    

    compute_col_from_Abacus(hcat[n_p], hcat[pos], hcat[vel], 
                            np.float32(header["ParticleMassHMsun"]),
                            dic['x'], dic['y'], dic['z'],
                            dic['vx'], dic['vy'], dic['vz'],
                            dic['Mh'], dic['log10_Mh'],
                            dic['Rs'], dic['Rh'], dic['c'],
                            hcat[f'r25_{Lsuff}com'], hcat[f'r98_{Lsuff}com'], Nthread)
    
    dic['row_id'] = np.array(hcat[index], dtype=np.int64)
    if halo_lc: 
        dic['redshift_interp'] = np.array(hcat['redshift_interp'], dtype=np.float32)
    if verbose : 
        print(f"Done took ", time.strftime("%H:%M:%S",time.gmtime(time.time() - start)))
    origin = None if not halo_lc else np.array(header['LightConeOrigins']).reshape(-1, 3)
    
    return Catalog.from_dict(dic), part_subsamples, header["BoxSizeHMpc"], origin

