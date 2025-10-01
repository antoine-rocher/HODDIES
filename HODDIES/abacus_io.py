""" Io tools to load AbacusSummit simualtions"""

import numpy as np
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import time 
from numba import njit, numba
import os 
from mpytools import Catalog


def read_Abacus_hcat(args, dir_sim, use_L2=True):
    """
    Load an Abacus halo catalog using provided configuration parameters.

    Parameters
    ----------
    args : dict
        Dictionary from the parameter file including simulation name, redshift, etc.
    dir_sim : str
        Path to the root simulation directory.
    use_L2 : bool, optional
        Whether to use L2com statistic from Abacus simulation or not. Default is True.
    Returns
    -------
    hcat : Catalog
        Halo catalog with derived physical quantities.
    part_subsamples : dict or None
        Dictionary of particle subsamples if particles are loaded, otherwise None.
    boxsize : float
        Size of the simulation box in Mpc/h.
    origin : ndarray or None
        Origin(s) of light cone if halo_lc is True, otherwise None.
    """

    if use_L2:
        Lsuff = 'L2'
    else : 
        Lsuff = ''
    
    usecols=['id', f'x_{Lsuff}com', f'v_{Lsuff}com', 'N', f'r25_{Lsuff}com', f'r98_{Lsuff}com', f'sigmav3d_{Lsuff}com'] 
    if 'small' in args['hcat']['sim_name']:
        path_to_sim = os.path.join(dir_sim, 'small',
                                    args['hcat']['sim_name'], "halos", "z{:.3f}".format(args['hcat']["z_simu"]))
    elif args['hcat']['halo_lc']:
        usecols =['index_halo', f'pos_interp', f'vel_interp', 'N_interp', 'redshift_interp', f'r25_{Lsuff}com', f'r98_{Lsuff}com', f'sigmav3d_{Lsuff}com'] 
        if not isinstance(args['hcat']["z_simu"], list):
            args['hcat']["z_simu"] = [args['hcat']["z_simu"]]
        path_to_sim = [os.path.join(dir_sim, 'halo_light_cones',
                                    args['hcat']['sim_name'], "z{:.3f}".format(z_lc)) for z_lc in args['hcat']["z_simu"]]
    else:
        path_to_sim = os.path.join(dir_sim, 
                                    args['hcat']['sim_name'], "halos",  "z{:.3f}".format(args['hcat']["z_simu"]))
    tt = time.time()

    hcat, part_subsamples, boxsize, origin = load_hcat_from_Abacus(path_to_sim, usecols, Lsuff, args['hcat']['halo_lc'], Nthread=args['nthreads'],  
                                                                   mass_cut=args['hcat']['mass_cut'], use_particles=args['hcat']['load_particles'])
    print(f'{args["hcat"]["sim_name"]} at {args["hcat"]["z_simu"]} loaded, took {time.strftime("%H:%M:%S",time.gmtime(time.time() - tt))}', flush=True)

    return hcat, part_subsamples, boxsize, origin



def load_CompaSO(path_to_sim, usecols, mass_cut=None, halo_lc=False, use_particles=None):
    """
    Load the CompaSO halo catalog from AbacusSummit simulations.

    Parameters
    ----------
    path_to_sim : str or list
        Path(s) to the simulation directory.
    usecols : list of str
        Fields to load from the catalog.
    mass_cut : float, optional
        Minimum halo mass threshold in log10(Msun/h). Halos with smaller mass are excluded.
    halo_lc : bool, optional
        If True, load halo light cone catalogs from multiple redshifts.
    use_particles : bool, optional
        Whether to load particle data or not.

    Returns
    -------
    hcat_i : np.ndarray
        Halo catalog array after applying mass cut.
    header : dict
        Metadata and simulation header.
    part_subsamples : dict or None
        Dictionary of particle subsamples if applicable.
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
    N = 10**mass_cut/header['ParticleMassHMsun'] if mass_cut is not None else 0

    return hcat_i[hcat_i[n_p] > N], header, part_subsamples


def load_CompaSO_particles(path_to_sim, ds_fac=1):
    """
    Load the CompaSO halo catalog from AbacusSummit simulations.

    Parameters
    ----------
    path_to_sim : str or list
        Path(s) to the simulation directory.
    usecols : list of str
        Fields to load from the catalog.
    mass_cut : float, optional
        Minimum halo mass threshold in log10(Msun/h). Halos with smaller mass are excluded.
    halo_lc : bool, optional
        If True, load halo light cone catalogs from multiple redshifts.
    use_particles : bool, optional
        Whether to load particle data or not.

    Returns
    -------
    hcat_i : np.ndarray
        Halo catalog array after applying mass cut.
    header : dict
        Metadata and simulation header.
    part_subsamples : dict or None
        Dictionary of particle subsamples if applicable.
    """
       

    hcat = CompaSOHaloCatalog(path_to_sim, fields=None, subsamples=dict(A=True), cleaned=True)    
    part_subsamples = hcat.subsamples if use_particles else None
    n_p = 'N' if not halo_lc else 'N_interp'
    N = 10**mass_cut/header['ParticleMassHMsun'] if mass_cut is not None else 0

    return hcat_i[hcat_i[n_p] > N], header, part_subsamples



@njit(parallel=True, fastmath=True)
def compute_col_from_Abacus(N, pos, vel, ParticleMassHMsun, 
                            x, y, z, vx, vy, vz, 
                            Mvir, log10_Mh, Rs, Rvir, c,
                            r25, r98, Nthread):
    """
    Compute derived physical columns for halos (mass, positions, velocities, radii, concentration).

    Parameters
    ----------
    N : array
        Number of particles per halo.
    pos, vel : arrays
        Position and velocity vectors.
    ParticleMassHMsun : float
        Mass of one simulation particle in Msun/h.
    x, y, z : arrays
        Output arrays for halo positions.
    vx, vy, vz : arrays
        Output arrays for halo velocities.
    Mvir, log10_Mh : arrays
        Output arrays for virial mass and log10(Mvir).
    Rs, Rvir : arrays
        Output arrays for scale radius and virial radius in kpc/h.
    c : array
        Output array for halo concentration.
    r25, r98 : arrays
        Input radius including 25 and 98 % of the halo particles 
    Nthread : int
        Number of threads for parallel execution.
    """

    numba.set_num_threads(Nthread)
    # starting index of each thread
    # figuring out the number of halos kept for each thread

    for i in numba.prange(len(N)):
            Mvir[i] = (N[i]*ParticleMassHMsun)
            log10_Mh[i] = np.log10(Mvir[i])
            x[i], y[i], z[i] = pos[i]
            vx[i], vy[i], vz[i] = vel[i]
            Rs[i] = r25[i]*1000
            Rvir[i] = r98[i]*1000
            c[i] = r98[i]/r25[i]        


def load_hcat_from_Abacus(path_to_sim, usecols, Lsuff, halo_lc=False, Nthread=64, mass_cut=None, verbose=True, use_particles=None):

    """
    Wrapper to load and process Abacus halo catalogs, used for HOD modeling.

    Parameters
    ----------
    path_to_sim : str or list
        Path(s) to the halo catalog directory.
    usecols : list
        Fields to read from the catalog.
    Lsuff : str
        Suffix (e.g., 'L2') used to determine which position/velocity columns to load.
    halo_lc : bool, optional
        Whether to load from halo light cone catalogs. Default is False.
    Nthread : int, optional
        Number of threads for computation. Default is 64.
    mass_cut : float, optional
        Minimum mass threshold to include halos. Default is None.
    verbose : bool, optional
        Whether to print progress messages. Default is True.
    use_particles : bool, optional
        Whether to include particle-level data in the returned catalog.

    Returns
    -------
    Catalog
        Processed catalog of halo data with derived fields.
    part_subsamples : dict or None
        Particle subsamples if available.
    boxsize : float
        Simulation box size in comoving Mpc/h.
    origin : ndarray or None
        Light cone origin(s) if applicable.
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
    
    dic = dict((col, np.empty(hcat[n_p].size, dtype='float32')) for col in ['x', 'y','z','vx','vy','vz', 'Rs','Rh', 'c', 'Mh', 'log10_Mh'])
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
    
    dic['halo_id'] = np.array(hcat[index], dtype=np.int64)
    if halo_lc: 
        dic['redshift_interp'] = np.array(hcat['redshift_interp'], dtype=np.float32)
    if verbose : 
        print(f"Done took ", time.strftime("%H:%M:%S",time.gmtime(time.time() - start)))
    origin = None if not halo_lc else np.array(header['LightConeOrigins']).reshape(-1, 3)
    
    return Catalog.from_dict(dic), part_subsamples, header["BoxSizeHMpc"], origin
    
def get_boxsize_from_simname(sim_name_or_path: str) -> float:
    """
    Return the box size (in Mpc/h) given a simulation path or name.

    Parameters
    ----------
    sim_name_or_path : str
        The simulation name or path containing one of the keys.

    Returns
    -------
    float
        Box size in Mpc/h.
    """
    sim_name_or_path = sim_name_or_path.lower()

    mapping = {
        "high": 1000.0,        # 1 Gpc/h
        "huge": 7500.0,     # ⚠ placeholder, confirm exact value
        "hugebase": 2.0,    # 2 Gpc/h
        "fixedbase": 1180.0,  # 1.18 Gpc/h
        "small": 500.0,       # 0.5 Gpc/h
        "pngbase": 2000.0,      # 2 Gpc/h
        "base": 2000.0,        # standard size (2 Gpc/h)
    }

    # Match key in string
    for key, size in mapping.items():
        if key in sim_name_or_path:
            return size

    raise ValueError(f"Unknown simulation type in: {sim_name_or_path}")