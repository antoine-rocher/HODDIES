""" Io tools to load AbacusSummit simualtions"""

import numpy as np
from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
import time 
from numba import njit, numba
import os 
from mpytools import Catalog
from . import Base_catalogue
import glob 
from pathlib import Path
from abacusnbody.data.read_abacus import read_asdf

class AbacusSummitSim(Base_catalogue):
    """
    Class to load Abacus summit simulation
    """
    def __init__(self, **kwargs):
        
        self.init_logger(kwargs.get('setup_logger', True))
        self.init_params(**kwargs)
        if self.z_simu is None:
            raise ValueError('Redshift is not provided, please provide z_simu.')
        self.init_abacus_args(**kwargs)
        self.read_Abacus_hcat()
        if kwargs.get('load_field_particles', False):
            self.field_particles = self.load_particle_field()


    def init_abacus_args(self, **kwargs):
        """
        Check if all parameter are correctly defined for.
        """
        self.use_particles = kwargs.get('load_particles', False)
        self.halo_lc = kwargs.get('halo_lc', False)
        self._root_dir_sim = kwargs.get('path_to_sim', None)
        if not self._root_dir_sim:
            raise ValueError("Directory path for simulation data is not provided.")
        self.sim_name = kwargs.get('sim_name', None)
        if self.sim_name is None:
            raise ValueError('Simulation name not provided, please provide sim_name.')
        self._is_sim_abacus=True
        
        
    def cosmo(self, **cosmo_args):
        """
        Return the cosmology object associated with the Abacus simulation.

        Returns
        -------
        cosmo : Cosmology object
            Cosmology corresponding to the Abacus simulation.
        """
        try: 
            engine = cosmo_args.get('engine', 'class')
            from cosmoprimo.fiducial import AbacusSummit
            self.logger.info('Initialize Abacus c{} cosmology'.format(self.sim_name.split('_c')[-1][:3]))
            return AbacusSummit(self.sim_name.split('_c')[-1][:3], engine=engine)

        except ImportError:
            import warnings
            warnings.warn('Could not import cosmoprimo. Install cosmoprimo with "python -m pip install git+https://github.com/cosmodesi/cosmoprimo[class,camb,extras]".\n'\
                  'Cosmology needed to apply RSD when computing correlations. No cosmology set.')
            return None


    def read_Abacus_hcat(self, use_L2=True):
        """
        Load an Abacus halo catalog using provided configuration parameters.

        Parameters
        ----------
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
            self.__Lsuff = 'L2'
        else : 
            self.__Lsuff = ''
        
        self.usecols = ['id', f'x_{self.__Lsuff}com', f'v_{self.__Lsuff}com', 'N', f'r25_{self.__Lsuff}com', f'r98_{self.__Lsuff}com', f'sigmav3d_{self.__Lsuff}com'] 
        if 'small' in self.sim_name:
            self._root_dir_sim = os.path.join(self._root_dir_sim, 'small')
        
        if self.halo_lc:
            self.usecols =['index_halo', f'pos_interp', f'vel_interp', 'N_interp', 'redshift_interp', f'r25_{self.__Lsuff}com', f'r98_{self.__Lsuff}com', f'sigmav3d_{self.__Lsuff}com'] 
            if not isinstance(self.z_simu, list):
                self.z_simu = [self.z_simu]
            self.__path_to_sim = [os.path.join(self._root_dir_sim, 'halo_light_cones',
                                        self.sim_name, "z{:.3f}".format(z_lc)) for z_lc in self.z_simu]
        else:
            self.__path_to_sim = os.path.join(self._root_dir_sim, 
                                        self.sim_name, "halos",  "z{:.3f}".format(self.z_simu))

        self.load_hcat_from_Abacus()
        

    def load_CompaSO(self):
        """
        Load the CompaSO halo catalog from AbacusSummit simulations.

        Parameters
        ----------
        self.__path_to_sim : str or list
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

        start = time.time()
        ld_part = 'with particles' if self.use_particles else ''
        self.logger.info(f"Load Compaso catalogue from {self.__path_to_sim} {ld_part}...")
        if self.use_particles:
            load_part = True
            self.usecols += ['npstartA', 'npoutA']
        else:
            load_part = False
            
        if self.halo_lc: 
            hcats = [CompaSOHaloCatalog(path, fields=self.usecols,  subsamples=dict(A=load_part), cleaned=True) for path in self.__path_to_sim]
            hcat_i = np.concatenate([cat.halos for cat in hcats])
            self.__header = hcats[0].header
        else:
            hcat = CompaSOHaloCatalog(f"{self.__path_to_sim}", fields=self.usecols,  subsamples=dict(A=load_part), cleaned=True)
            hcat_i, self.__header = hcat.halos, hcat.header
        
        self.part_subsamples = hcat.subsamples if self.use_particles else None
        n_p = 'N' if not self.halo_lc else 'N_interp'
        N = 10**self.mass_cut/self.__header['ParticleMassHMsun'] if self.mass_cut is not None else 0

        hcat = hcat_i[hcat_i[n_p] > N]
        self.logger.info(f"Done took {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
        return hcat

    def load_particle_field(
        self,
        subsample="A",
        fraction=1 / 10,
        seed=42,
        load=("pos",),
        verbose=True,
    ):
        """
        Randomly subsample Abacus snapshot particles, one file at a time.

        Parameters
        ----------
        sim_dir : str or Path
            Snapshot directory, e.g. ".../halos/z0.500".
        subsample : {"A", "B", "AB", "both"}
            Particle subsample(s) to read.
        fraction : float
            Independent retention probability for each available particle.
            Relative to the selected A/B particles, not the full simulation.
        seed : int or None
            Random seed.
        load : tuple of str
            Columns to return: ("pos",), ("vel",), or ("pos", "vel").
        populations : tuple of str
            Default ("halo", "field") includes the full matter distribution.
            Use ("field",) for only particles in the raw field files.
        verbose : bool
            Print progress.

        Returns
        -------
        particles : dict
            NumPy arrays with shape (N_selected, 3), keyed by "pos"/"vel".
        info : dict
            Particle counts, selection settings, and box size.

        Notes
        -----
        Each file is fully read and decoded before subsampling.
        Memory holds one decoded file plus the accumulated selected particles.
        """

        from pathlib import Path
        from abacusnbody.data.read_abacus import read_asdf

        start = time.time()
        sim_dir = Path(self.__path_to_sim)
        selection = str(subsample).upper()
        if selection == "BOTH":
            selection = "AB"
        if 'small' in str(sim_dir):
            selection = "A"
        if selection not in ("A", "B", "AB"):
            raise ValueError("subsample must be 'A', 'B', 'AB', or 'both'.")

        fraction = float(fraction)
        if not np.isfinite(fraction) or not 0 < fraction <= 1:
            raise ValueError("fraction must satisfy 0 < fraction <= 1.")

        load = (load,) if isinstance(load, str) else tuple(load)
        if not load or len(set(load)) != len(load) or not set(load) <= {"pos", "vel"}:
            raise ValueError("load must contain 'pos', 'vel', or both, without duplicates.")

        populations=("halo", "field")

        self.logger.info(f"Load Abacus particles using {fraction:.3%} of the subsample {subsample}...")
        # Validate all requested directories before starting expensive reads.
        files = []
        n_all = 0 
        for sample in selection:
            for population in populations:
                directory = sim_dir / f"{population}_rv_{sample}"
                matches = sorted(
                    directory.glob(f"{population}_rv_{sample}_*.asdf")
                )
                if not matches:
                    raise FileNotFoundError(
                        f"No particle RV files found in {directory}. "
                        "Check availability for this simulation and redshift."
                    )
                files.extend(matches)

        rng = np.random.default_rng(seed)
        table_all = []
        for i, filename in enumerate(files):
            table_tmp = read_asdf(
                str(filename),
                load=list(load),
                dtype=np.float32,
                verbose=False,
            )


            # A binomial count followed by uniform selection without replacement
            # is equivalent to independently retaining each particle with
            # probability `fraction`, without allocating N random floats.
            n = len(table_tmp)
            nkeep = int(np.round(n * fraction))
            indices = rng.choice(n, size=nkeep, replace=False, shuffle=False)

            if verbose:
                self.logger.info(
                    f"[{i}/{len(files)}] {filename.name}: "
                    f"{nkeep:,} / {n:,} particles retained"
                )
            n_all += nkeep

            table_all.append(table_tmp[indices])
            del table_tmp

        self.logger.info(f"Compiled {n_all} particles, took time: {time.time() - start}")
        return np.concatenate(table_all, axis=0)


    @staticmethod
    @njit(parallel=True, fastmath=True, cache=True)
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


    def load_hcat_from_Abacus(self):

        """
        Wrapper to load and process Abacus halo catalogs, used for HOD modeling.

        Catalog
            Processed catalog of halo data with derived fields.
        part_subsamples : dict or None
            Particle subsamples if available.
        boxsize : float
            Simulation box size in comoving Mpc/h.
        origin : ndarray or None
            Light cone origin(s) if applicable.
        """

        hcat = self.load_CompaSO()
        
        start = time.time()
        self.logger.info("Compute required columns...")
        n_p, pos, vel, index = ('N', f'x_{self.__Lsuff}com', f'v_{self.__Lsuff}com', 'id') if not self.halo_lc else ('N_interp', 'pos_interp', 'vel_interp', 'index_halo')
        
        dic = dict((col, np.empty(hcat[n_p].size, dtype='float32')) for col in ['x', 'y','z','vx','vy','vz', 'Rs','Rh', 'c', 'Mh', 'log10_Mh'])
        dic['Vrms'] = np.array(hcat[f'sigmav3d_{self.__Lsuff}com'])
        if self.use_particles:
            dic['npstartA'] = np.array(hcat['npstartA'], dtype='int64')
            dic['npoutA'] = np.array(hcat['npoutA'], dtype='int64')
        

        self.compute_col_from_Abacus(hcat[n_p], hcat[pos], hcat[vel], 
                                np.float32(self.__header["ParticleMassHMsun"]),
                                dic['x'], dic['y'], dic['z'],
                                dic['vx'], dic['vy'], dic['vz'],
                                dic['Mh'], dic['log10_Mh'],
                                dic['Rs'], dic['Rh'], dic['c'],
                                hcat[f'r25_{self.__Lsuff}com'], hcat[f'r98_{self.__Lsuff}com'], self.nthreads)
        
        dic['halo_id'] = np.array(hcat[index], dtype=np.int64)
        if self.halo_lc: 
            dic['redshift_interp'] = np.array(hcat['redshift_interp'], dtype=np.float32)
       
        self.logger.info("Done took {0}".format(time.strftime("%H:%M:%S",time.gmtime(time.time() - start))))
        self.origin = None if not self.halo_lc else np.array(self.__header['LightConeOrigins']).reshape(-1, 3)
        
        self.boxsize = self.__header["BoxSizeHMpc"]
        self.hcat = Catalog.from_dict(dic)

    
    def get_boxsize(self, sim_name_or_path: str) -> float:
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
                self.boxsize = size

        raise ValueError(f"Unknown simulation type in: {sim_name_or_path}")

    @property
    def __particles_snap_available(self):
        """
        Return the list of available redshifts for which particle snapshots are available in AbacusSummit simulations.
        """
        
        available_part_redshifts = [3.0, 2.5, 2.0, 1.7, 1.4, 1.1, 0.8, 0.5, 0.4, 0.3, 0.2, 0.1]
        
        return available_part_redshifts
       

    def check_redshift_available(self):
        """
            Check if the requested redshift is available for AbacusSummit snapshots.
        """
        available_z = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.575, 0.65, 0.725, 0.8, 0.875, 0.95, 1.025, 1.1, 1.175, 1.25, 1.325, 1.4, 1.475, 1.55, 1.625, 1.7, 1.85, 2.0, 2.25, 2.5, 2.75, 3.0, 5.0, 8.0]
        if self.z_simu in available_z:
            pass
        else:  
            raise ValueError(f"Redshift z={self.z_simu} not available for AbacusSummit. Available redshifts are: {available_z}")


    def load_env_based_properties(self, **kwargs):
        """
        Load or compute environment-based properties (density and shear) from Abacus simulation to compute assembly bias. This method checks for precomputed density and shear meshes, and if not found, computes them from particle outputs.
        """ 
        
        from abacusnbody.data.read_abacus import read_asdf
        import hdf5plugin
        import h5py    
        from HODDIES.environment_func import calc_env, calc_shear_from_dsmo

        cell_size = kwargs.get('cell_size', 5)
        R = kwargs.get('R', 1.5)
        dir_to_save_env_mesh = kwargs.get('dir_to_save_env_mesh', '/tmp')
        path_to_save = os.path.join(
            dir_to_save_env_mesh,
            f'env_shear_map_{self.sim_name}_z{self.z_simu:.3f}.h5'
        )

        if dir_to_save_env_mesh and os.path.isdir(dir_to_save_env_mesh):
            if os.path.exists(path_to_save):
                self.logger.info(f'Load precomputed density and shear mesh for {self.sim_name} at {path_to_save}...')
                env_prop = Catalog.read(path_to_save)
                self.density_mesh, self.shear_mesh = env_prop['density'], env_prop['shear']
                return
        else:
            self.logger.info(f'Precomputed density and shear mesh not found. Create directory {os.path.join(dir_to_save_env_mesh)} to save density and shear mesh')
            os.makedirs(dir_to_save_env_mesh, exist_ok=True)
        
        if self.z_simu not in self.__particles_snap_available:
            import warnings
            warnings.warn(
                f"Redshift snapshot {self.z_simu:.3f} does not have particle outputs. Assembly bias properties cannot be computed for env and shear properties. Available redshifts with particles are: {self.__particles_snap_available}")
            self.density_mesh = None
            self.shear_mesh = None
            return

        path = os.path.join(self._root_dir_sim, self.sim_name, 'halos', f'z{self.z_simu:.3f}')
        if not os.path.exists(path):
            raise NameError(f'Wrong simulation path: {path}')
        if self.z_simu not in self.__particles_snap_available:
            import warnings
            warnings.warn(
            f"Particles are not available at redshift z={self.z_simu}. Assembly bias properties cannot be computed for env and shear properties. Available redshifts with particles are: {self.__particles_snap_available}")
            self.density_mesh = None
            self.shear_mesh = None 
            return
        if getattr(self, "field_particles", None) is None:
            self.logger.info('Particles are not loaded. Load Abacus particles to compute density and shear mesh')
            self.field_particles = self.load_particle_field(subsample="A", fraction=1/10, seed=42, load=("pos",))

        dsmo = calc_env(self.field_particles['pos'], self.boxsize, cell_size=cell_size, R=R) 
        shear = calc_shear_from_dsmo(dsmo, self.boxsize, cell_size=cell_size, R=R, workers=-1) 
        
        # self.logger.info(f'Save to {path_to_save}')
        with h5py.File(path_to_save, "w") as f:
            f.create_dataset(
                'density',
                data=dsmo,
                **hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE))
                
            f.create_dataset(
                'shear',
                data=shear,
                **hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE))
        self.density_mesh = dsmo
        self.shear_mesh = shear


    
    def assign_sat_to_part(self, mask_sat, list_nsat, seed=None, f_sigv=1.0):
        """
        Assign satellite galaxies to particles in the Abacus simulation.

        Parameters
        ----------        
        list_nsat : array-like, optional
            List of number of satellites per halo. 
        mask_sat : array-like, optional
            Boolean mask indicating which halo include satellites to assign.
        seed : int, optional
            Random seed for reproducibility.
        f_sigv : float, optional
            Apply v_sat = v_halo + f_sigv * (v_part - v_halo) inside the
            Numba loop. One preserves the particle velocities.

        Returns
        -------
        x_sat, y_sat, z_sat : array
            The assigned positions of the satellite galaxies in the x, y, and z coordinates.
        vx_sat, vy_sat, vz_sat : array
            The assigned velocities of the satellite galaxies in the vx, vy, and vz components.
        mask_nfw : array
            Boolean mask indicating which satellites were assigned using NFW profile due to insufficient particles.
        
        Notes
        -----
        - Satellite catalog (`sat_cat`) is modified in place to assign positions and velocities from particles.

        """
    
        if self.part_subsamples is None:
            raise ValueError("Particle subsamples are not loaded. Cannot assign satellites to particles.")
        self.logger.info("Assign satellites to particles...")

        x_sat, y_sat, z_sat, vx_sat, vy_sat, vz_sat, mask_nfw = _compute_sat_from_abacus_part(self.part_subsamples['pos'].T[0], self.part_subsamples['pos'].T[1], self.part_subsamples['pos'].T[2],
                                                  self.part_subsamples['vel'].T[0], self.part_subsamples['vel'].T[1], self.part_subsamples['vel'].T[2],
                                                  self.hcat['npoutA'][mask_sat], self.hcat['npstartA'][mask_sat], list_nsat,  np.insert(np.cumsum(list_nsat), 0, 0), self.nthreads, seed=seed,
                                                  vx_h=self.hcat['vx'][mask_sat], vy_h=self.hcat['vy'][mask_sat], vz_h=self.hcat['vz'][mask_sat], f_sigv=f_sigv)
        return x_sat, y_sat, z_sat, vx_sat, vy_sat, vz_sat, mask_nfw




@njit(parallel=True, fastmath=True, cache=True)
def _compute_sat_from_abacus_part(xp, yp, zp, vxp, vyp, vzp, npout, npstart, nb_sat, 
                                cum_sum_sat, Nthread, seed=None, vx_h=None,
                                vy_h=None, vz_h=None, f_sigv=1.0):

    """
    Sample satellite galaxy positions and velocities from Abacus particle subsamples.

    Parameters
    ----------
    xp, yp, zp : arrays
        Particle positions in each axis.
    vxp, vyp, vzp : arrays
        Particle velocities in each axis.
    npout : array
        Number of available particles for each halo.
    npstart : array
        Starting index in the global particle array for each halo.
    nb_sat : array
        Number of satellites to assign per halo.
    cum_sum_sat : array
        Cumulative sum array to locate output indices.
    Nthread : int
        Number of threads to use in parallel loop.
    seed : array, optional
        Array of seeds for reproducible random sampling across threads.
    vx_h, vy_h, vz_h : arrays, optional
        Host halo velocities in the same order as npout and nb_sat.
        Required for f_sigv other than one.
    f_sigv : float, optional
        Apply v_sat = v_halo + f_sigv * (v_part - v_halo) in each component.
        One preserves the particle velocities.

    Returns
    -------
    x_sat, y_sat, z_sat : arrays
        Satellite positions in each axis from particle subsamples.
    vx_sat, vy_sat, vz_sat : arrays
        Satellite velocities in each axis from particle subsamples.
    mask_nfw : array
        Boolean mask identifying entries with no enough particles (to be filled using NFW).
    """

    if f_sigv != 1.0 and (vx_h is None or vy_h is None or vz_h is None):
        raise ValueError('Host halo velocities are required for particle velocity bias.')

    mask_nfw = np.zeros(nb_sat.sum(), dtype='bool')
    x_sat = np.zeros(nb_sat.sum(), dtype='float32')
    y_sat = np.zeros(nb_sat.sum(), dtype='float32') 
    z_sat = np.zeros(nb_sat.sum(), dtype='float32') 
    vx_sat = np.zeros(nb_sat.sum(), dtype='float32')
    vy_sat = np.zeros(nb_sat.sum(), dtype='float32')
    vz_sat = np.zeros(nb_sat.sum(), dtype='float32')
    hstart = np.rint(np.linspace(0, npout.size, Nthread + 1))
    for tid in numba.prange(Nthread):
        if seed is not None:
            np.random.seed(seed[tid])

        for i in range(int(hstart[tid]), int(hstart[tid + 1])):
            if nb_sat[i] < npout[i]:
                tt = np.random.choice(npout[i], nb_sat[i], replace=False) + npstart[i]
            else:
                tt = np.arange(npout[i]) + npstart[i]
                mask_nfw[cum_sum_sat[i]+npout[i]: cum_sum_sat[i+1]] = True
            for j in range(tt.size):
                particle_index = tt[j]
                satellite_index = cum_sum_sat[i] + j
                x_sat[satellite_index] = xp[particle_index]
                y_sat[satellite_index] = yp[particle_index]
                z_sat[satellite_index] = zp[particle_index]
                if vx_h is not None and vy_h is not None and vz_h is not None:
                    vx_sat[satellite_index] = vx_h[i] + f_sigv * (vxp[particle_index] - vx_h[i])
                    vy_sat[satellite_index] = vy_h[i] + f_sigv * (vyp[particle_index] - vy_h[i])
                    vz_sat[satellite_index] = vz_h[i] + f_sigv * (vzp[particle_index] - vz_h[i])
                else:
                    vx_sat[satellite_index] = vxp[particle_index]
                    vy_sat[satellite_index] = vyp[particle_index]
                    vz_sat[satellite_index] = vzp[particle_index]
    return x_sat, y_sat, z_sat, vx_sat, vy_sat, vz_sat, mask_nfw
