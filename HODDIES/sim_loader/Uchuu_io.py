""" Io tools to load Uchuu simualtions"""

from functools import partial
import h5py
import glob 
import numpy as np 
import multiprocessing
import time 
from mpytools import Catalog
import os 
from .sim_loader import Base_catalogue


def _read_chunk_2gpc(chunks, filename, group='main', mass_cut=None):
    """Read a preprocessed chunk with explicit, process-local configuration."""
    start, end = chunks
    with h5py.File(filename, 'r') as handle:
        datasets = handle[group]
        dtype = [(name, datasets[name].dtype) for name in datasets]
        data = np.empty(end - start, dtype=dtype)
        for name in datasets:
            data[name] = datasets[name][start:end]
        if mass_cut is not None:
            data = data[data['log10_Mh'] > mass_cut]
    return data


def _read_chunk_dde(chunk, filename, mass_cut=None):
    """Read main hosts, allocating output only for rows passing the selection.

    Slice columns contiguously before filtering in NumPy. Preserve the original
    float32 mass selection, derived properties, row order, and int64 IDs.
    DDE subhalo loading remains unsupported.
    """
    start, end = chunk
    names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'Rs', 'Rh', 'c', 'Mh',
             'log10_Mh', 'Vrms', 'halo_id']
    dtype = [(name, np.int64 if name == 'halo_id' else np.float32) for name in names]
    mapping = {'x': 'X', 'y': 'Y', 'z': 'Z', 'vx': 'VX', 'vy': 'VY', 'vz': 'VZ',
               'Rs': 'Rs', 'Rh': 'Rvir', 'Vrms': 'Vrms', 'halo_id': 'ID'}
    with h5py.File(filename, 'r') as handle:
        keys = {name.lower(): name for name in handle}
        required = [*mapping.values(), 'Mvir', 'PID']
        resolved = {}
        for name in required:
            if name.lower() not in keys:
                raise KeyError(f'Column {name} not found in file keys {list(handle.keys())}')
            resolved[name] = keys[name.lower()]

        # The old reader converted Mvir to float32 before applying mass_cut.
        mass = handle[resolved['Mvir']][start:end].astype(np.float32)
        keep = handle[resolved['PID']][start:end] == -1
        if mass_cut is not None:
            keep &= mass > 10**mass_cut
        data = np.empty(np.count_nonzero(keep), dtype=dtype)
        if not len(data):
            return data, None
        data['Mh'] = mass[keep]
        for output, source in mapping.items():
            data[output] = handle[resolved[source]][start:end][keep]

    data['log10_Mh'] = np.log10(data['Mh'])
    data['c'] = data['Rh'] / data['Rs']
    return data, None


class UchuuSim(Base_catalogue):
    
    def __init__(self, **kwargs):
        self.init_logger(kwargs.get('setup_logger', True))
        self.init_params(**kwargs)
        self.init_uchuu_args(**kwargs)     
        if self.z_simu is None:
            raise ValueError('Redshift is not provided, please provide z_simu.')
        self.read_Uchuu()

    def init_uchuu_args(self, **kwargs):
        """
        Check if all parameter are correctly defined for.
        """
        self._root_dir_sim = kwargs.get('path_to_sim', None)
        if not self._root_dir_sim:
            raise ValueError("Directory path for simulation data is not provided.")
        
        self.sim_name = kwargs.get('sim_name', None)
        if self.sim_name is None:
            raise ValueError('Simulation name not provided, please provide sim_name.')

        self.mass_cut = kwargs.get('mass_cut', None)
        self.use_particles = kwargs.get('load_particles', False)

    def cosmo(self, **cosmo_params):
        """
        Return the cosmology object associated with the Uchuu simulation.

        Parameters
        ----------
        cosmo_params : dict
            Parameters from cosmoprimo to customize the cosmology initialization.
        Returns
        -------
        cosmo : Cosmology object
            Cosmology corresponding to the Uchuu simulation.
        """

        try: 
            engine = cosmo_params.get('engine', 'class')
            cosmo_name= 'Planck2018' if self.sim_name == 'Planck18' else 'Planck2018DDE' if self.sim_name == 'Planck18_DDE' else 'DESIY1DDE' if self.sim_name == 'DESIY1_DDE' else 'Planck2015'
            self.logger.info('Initialize Uchuu {} cosmology'.format(cosmo_name))
            return Uchuu_cosmo(cosmo_name, engine=engine)

        except ImportError:
            import warnings
            warnings.warn('Could not import cosmoprimo. Install cosmoprimo with "python -m pip install git+https://github.com/cosmodesi/cosmoprimo[class,camb,extras]".\n'\
                  'Cosmology needed to apply RSD when computing correlations. No cosmology set.')
            return None


    def read_chuncks_2gpc(self, chunks, filename, group='main'):
        """Read a preprocessed chunk using this simulation's mass cut."""
        return _read_chunk_2gpc(chunks, filename, group=group, mass_cut=self.mass_cut)

    def read_chunk_DDE(self, chunk, filename):
        """Read main hosts from a Rockstar chunk using this simulation's mass cut."""
        return _read_chunk_dde(chunk, filename, mass_cut=self.mass_cut)


    def read_Uchuu(self, nchuncks=32):
        """
        Load Uchuu simulation data (halos and optionally subhalos) from the specified directory and snapshot. The method reads the data in parallel using multiprocessing, applies a mass cut if specified, and stores the resulting halo and subhalo catalogs as attributes of the class.

        Parameters
        ----------
        nchuncks : int, optional    
        Number of chunks to split each file into. Default is 32. The number
        of reader processes is limited by both this value and ``nthreads``.

        Raises
        ------
        ValueError
            If the specified simulation name is not available in the Uchuu DDE simulations.
        
        Notes
        -----
        The method expects the halo catalogs to be organized in a specific directory structure based on the simulation name and snapshot redshift. It also checks for the presence of subhalos if the `use_particles` flag is set to True, and will disable subhalo loading for simulations that do not support it. 
        """

        opt_ch = 128 if self.sim_name=='Uchuu2Gpc' else 32
        if nchuncks < 1:
            raise ValueError('nchuncks must be positive.')
        nchuncks = min(nchuncks, opt_ch)
        nreaders = max(1, min(self.nthreads, nchuncks))
        self.boxsize = boxsize_Uchuu(self.sim_name)
        DDE_sims = ['Planck18', 'Planck18_DDE', 'DESIY1_DDE', 'Uchuu2Gpc']
        if self.sim_name not in DDE_sims:
            raise ValueError(f"Uchuu DDE simulation names {self.sim_name} not available. Valid names are: {DDE_sims}")
        dirname_hcat = os.path.join(self._root_dir_sim, 'Uchuu_halo_catalogs', f'halodir_{get_Uchuu_snapshot_file(self.z_simu):03d}', '*') if self.sim_name=='Uchuu2Gpc' else os.path.join(self._root_dir_sim,'DDE', self.sim_name, get_DDE_snapshot_file(self.z_simu))
        
        if self.mass_cut is not None:
            self.logger.info(f'Apply mass cut at 10^{self.mass_cut} M_sol/h')

        if self.use_particles & (self.sim_name!='Uchuu2Gpc'):
            self.use_particles=False
            self.logger.warning('Can not use subhalos with Uchuu DDE simulations')
        subh = 'with subhalos' if self.use_particles else ''
        self.logger.info(f'Load Uchuu {self.sim_name} simulation at z={self.z_simu:.3f} {subh}')
        filenames = glob.glob(dirname_hcat)
        self.logger.info(f'Reading {filenames}')
        if not filenames:
            raise FileNotFoundError(f'No Uchuu catalogues match {dirname_hcat}')
        self.logger.info(f'Use {nreaders} reader processes and {nchuncks} chunks per file')
        st = time.time()
        uchuu_halo, uchuu_subhalo = [], []
        with multiprocessing.Pool(nreaders) as pool:
            for fn in filenames:
                if self.sim_name == 'Uchuu2Gpc':
                    groups = ['main', 'sub'] if self.use_particles else ['main']
                    for group in groups:
                        # Only inspect the dataset shape; do not construct a catalogue.
                        with h5py.File(fn, 'r') as handle:
                            size = next(iter(handle[group].values())).shape[0]
                        edges = np.linspace(0, size, nchuncks + 1, dtype=int)
                        chunks = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]
                        if not chunks:
                            chunks = [(0, 0)]
                        reader = partial(_read_chunk_2gpc, filename=fn, group=group,
                                         mass_cut=self.mass_cut)
                        result = pool.map(reader, chunks)
                        (uchuu_halo if group == 'main' else uchuu_subhalo).extend(result)
                else:
                    with h5py.File(fn, 'r') as handle:
                        size = handle[next(iter(handle))].size
                    edges = np.linspace(0, size, nchuncks + 1, dtype=int)
                    chunks = [(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]
                    if not chunks:
                        chunks = [(0, 0)]
                    reader = partial(_read_chunk_dde, filename=fn, mass_cut=self.mass_cut)
                    result = pool.map(reader, chunks)
                    uchuu_halo.extend(halos for halos, _ in result)

        # Concatenate once after reading all files, preserving input row order.
        self.hcat = Catalog.from_array(np.concatenate(uchuu_halo))
        self.part_subsamples = (Catalog.from_array(np.concatenate(uchuu_subhalo))
                                if self.use_particles else None)
        self.logger.info(f'Uchuu catalogue read in {time.time()-st:.2f} seconds')


    def load_env_based_properties(self, dir_particle_file=None, dir_to_save_env_mesh='tmp/', cell_size=5, R=1.5, subset=0.05, **kwargs):
        """
        Load or compute environment-based properties (density and shear) from Uchuu simulation data to compute assembly bias. This method checks for precomputed density and shear meshes, and if not found, computes them from particle outputs.
        """                    

        from .Uchuu.read_particles import read_gadget2_multi
        from HODDIES.environment_func import compute_env_shear_from_particles

        self.dir_to_save_env_mesh = kwargs.get('dir_to_save_env_mesh', dir_to_save_env_mesh)
        if self.sim_name == 'Uchuu2Gpc':   
            if path and os.path.isdir(dir_to_save_env_mesh):
                snapshot_num = get_Uchuu_snapshot_file('{:.3f}'.format(self.z_simu))
                path = os.path.join(
                    path,
                    f'env_shear_map_Uchuu_snapdir_{snapshot_num:03d}.h5'
                )
                
            if path and os.path.exists(path):
                self.logger.info(f'Load precomputed density and shear mesh for Uchuu env/shear mesh at {path}...')
                env_prop = Catalog.read(path)
                self.density_mesh, self.shear_mesh = env_prop['density'], env_prop['shear']
            
            else:
                self.logger.info('Precomputed density and shear mesh for Uchuu not found. Computing them from particle outputs...')
                dir_particle_file = kwargs.get('dir_particle_file', dir_particle_file)
                if dir_particle_file is None:
                    self.logger.info(f"Particle outputs not provided.  Assembly bias properties cannot be computed for env and shear properties. Please provide dir_particle_file to compute environment and shear properties.")
                    self.density_mesh, self.shear_mesh = None, None
                    return 0
                
                if 'snapdir' not in dir_particle_file:
                    self.logger.info(f"dir_particle_file: {dir_particle_file} should contain 'snapdir' in its path. Assembly bias properties cannot be computed for env and shear properties. Please provide correct dir_particle_file to compute environment and shear properties.")

                    self.density_mesh, self.shear_mesh = None, None
                    return 0
                    
                for dir_particle_file in os.listdir(dir_particle_file):
                    snapdir = dir_particle_file[dir_particle_file.find('snapdir'):dir_particle_file.find('snapdir')+11]
                    path_to_save = os.path.join(dir_to_save_env_mesh, f'env_shear_map_Uchuu_{snapdir}.h5')
                    if not os.path.exists(dir_to_save_env_mesh):
                        self.logger.info(f'Create directory {os.path.join(dir_to_save_env_mesh)} to save density and shear mesh')
                        os.makedirs(dir_to_save_env_mesh)
                    else:
                        self.logger.info(f"Saving environment and shear map to {path_to_save}")
                    
                    data, header = read_gadget2_multi(dir_particle_file+'/', n_threads=self.nthreads, subset_fraction=subset)
                    self.logger.info("\n=== Snapshot Summary ===")
                    self.logger.info(f"Total particles read: {len(data['id']):,}")
                    self.logger.info(f"Particles Mass: {header['Massarr'][1]} 10^10 Msun/h")
                    self.logger.info(f"BoxSize: {header['BoxSize']} Mpc/h")

                    self.density_mesh, self.shear_mesh = compute_env_shear_from_particles(data['pos'], header['BoxSize'], cell_size=cell_size, R=R, path_to_save=path_to_save)

        else:
            self.logger.info(f"Particle outputs are not available for Uchuu {self.sim_name} simulation. Assembly bias properties cannot be computed for env and shear properties.")
            self.density_mesh, self.shear_mesh = None, None



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
    if isinstance(z, float):
        z = f'{z:.3f}'
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
        return Uchuu_snapshot_redshifts[z]
    else:
        valid = ", ".join(map(str, Uchuu_snapshot_redshifts.keys()))
        raise ValueError(f"Redshift {z} not available. Valid redshifts are: {valid}")

def boxsize_Uchuu(sim_name):
    if sim_name == 'Uchuu2Gpc':
        return 2000.0
    elif sim_name in ['Planck18', 'Planck18_DDE', 'DESIY1_DDE']:
        return 1000.0
    else:
        None
    


def Uchuu_cosmo(name='Planck2015', engine='class', extra_params=None, **params):
    """
    Initialize :class:`Cosmology` for Uchuu simulations.

    Parameters
    ----------
    name : string, default='2015'
        One of 'Planck2015', 'Planck2018', 'Planck2018DDE', 'DESIY1DDE'.
    engine : string, default=None
        Engine name, one of ['class', 'camb', 'eisenstein_hu', 'eisenstein_hu_nowiggle', 'bbks'].
        If ``None``, returns current :attr:`Cosmology.engine`.

    extra_params : dict, default=None
        Extra engine parameters, typically precision parameters.

    params : dict
        Cosmological and calculation parameters which take priority over the default ones.

    Returns
    -------
    cosmology : Cosmology
    """
    try: 
        from cosmoprimo import constants, Cosmology
    except ImportError:
        import warnings
        warnings.warn('Could not import cosmoprimo. Install cosmoprimo with "python -m pip install git+https://github.com/cosmodesi/cosmoprimo[class,camb,extras]".\n'\
                'Cosmology needed to apply RSD when computing correlations. No cosmology set.')
        return None

    common = dict(Omega_k=0., m_ncdm=[0.06], neutrino_hierarchy=None, T_ncdm_over_cmb=constants.TNCDM_OVER_CMB, N_eff=constants.NEFF, A_L=1.0, k_pivot=0.05)
    if name == 'Planck2015':
        # Reference: https://www.skiesanduniverses.org/Simulations/Uchuu/
        default_params = dict(h=0.6774, Omega_m=0.3089, Omega_b=0.0486, sigma8=0.8159, n_s=0.9667, tau_reio=0.063, **common)
    elif name == 'Planck2018':
        # Reference: Table I of https://arxiv.org/pdf/2503.19352
        default_params = dict(h=0.6766, Omega_m=0.3111, Omega_b=0.048975, sigma8=0.8102, n_s=0.9665, tau_reio=0.063, **common)
    elif name == 'Planck2018DDE':
        default_params = dict(h=0.6766, Omega_m=0.3111, Omega_b=0.048975, sigma8=0.8102, n_s=0.9665, tau_reio=0.063, w0_fld=-0.45, wa_fld=-1.79, **common)
    elif name == 'DESIY1DDE':
        default_params = dict(h=0.6470, Omega_m=0.3440, Omega_b=0.048975, sigma8=0.8102, n_s=0.9665, tau_reio=0.063, w0_fld=-0.45, wa_fld=-1.79, **common)
    else:
        raise NotImplementedError('Uchuu cosmology {} not implemented; available cosmologies are Planck2015, Planck2018, Planck2018DDE, DESIY1DDE')
    return Cosmology(engine=engine, extra_params=extra_params, **default_params).clone(**params)