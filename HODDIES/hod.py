""" Base HOD class """


from numba import njit, jit, numba
import time 
import os
from .utils import *
from .HOD_models import _SHOD, _GHOD, _SFHOD, _SHOD, _LNHOD, _HMQ, _mHMQ, _Nsat_pow_law
import yaml 
import glob
from mpytools import Catalog
import collections.abc

class HOD:

    """Class with tools to generate HOD mock catalogs and plotting functions"""

    def __init__(self, param_file=None, args=None, hcat_file=None, path_to_abacus_sim=None, read_pinnochio=None, subsample=None, **kwargs):
        
        """
        Initialize :class:`HOD`. 

        Parameters
        ----------
        param_file : str, default=None
            Input parameter file to initialize the HOD class. If None, the default parameter file 'default_HOD_parameters.yaml' is used.

        args : dict, default=None
            Optional
            Input dictonary to initialize  the HOD class. Carefull ``args`` is prefered against ``param_file``. 
        hcat : dict, ndarray, Catalog, default=None
            Optional
            Input halo catalog. The halo catalog must have at least these columns names: ['x', 'y', 'z', 'vx', 'vy', 'vz','Mh', 'Rh', 'Rs', 'c', 'Vrms', 'halo_id']. 
            'x', 'y', 'z', and 'vx', 'vy', 'vz' are halo positions and velocities. 
            Mh and Rh are halo mass and radius (most of the time consider as Mvir and Rvir). 
            'c' is the halo concentration, 'Vrms' is the velocity dispersion of the halo particles (used for NFW satellites). 
            'halo_id' is a unique integer to identify each halo.
        boxsize : int, default = None
            Simulation box size, to be set if hcat is provide. Prefered if boxsize value is also provided in the parameter file.

        path_to_abacus_sim : str, default=None
            Optional,
            Path to Abacus simulation directory. In this case, it automatically load the Abacus box/LC at the corresponding redshift snapshots and initialze boxsize and cosmology.

        read_pinnochio : bool, default=None
            Optional, load Pinnochio simulation catalog. Need to provide the path in the input parameter file.

        subsample : dict, ndarray, Catalog, default=None
            Optional, Not yet ready!
            Input of particles or subhalo catalog. The subsample catalog must have at least these columns names: ['x', 'y', 'z', 'vx', 'vy', 'vz', 'halo_id']. 
        kwargs : dict
            Optional arguments that can be added that will replace the one provided in the parameter file.

        """
        
        
        self.args = yaml.load(open(os.path.join(os.path.dirname(__file__), 'default_HOD_parameters.yaml')), Loader=yaml.FullLoader)  
        self.cosmo = None
        self.H_0 = 100 # H_0 is always set to 100 km/s/Mpc

        def update_dic(d, u):
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k] = update_dic(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        new_args = yaml.load(open(param_file), Loader=yaml.FullLoader) if param_file is not None else args if args is not None else self.args
        update_dic(self.args, new_args)
        update_dic(self.args, kwargs)
        self.boxsize = self.args['hcat']['boxsize']

        self.args['nthreads'] = min(numba.get_num_threads(), self.args['nthreads'])
        print('Set number of threads to {}'.format(self.args['nthreads']), flush=True)
        self.part_subsamples = subsample

        if path_to_abacus_sim  is not None:
            from .abacus_io import read_Abacus_hcat
            self.args['hcat'] = self.args['hcat'] | self.args['hcat']['Abacus']
            self.hcat, self.part_subsamples, self.boxsize, self.origin = read_Abacus_hcat(self.args, path_to_abacus_sim)
                            
        elif read_pinnochio  is not None:
            from .pinnochio_io import read_pinnochio_hcat
            print('Read Pinnochio', flush=True)
            start = time.time()
            self.hcat, self.boxsize, self.cosmo = read_pinnochio_hcat(self.args)
            print('Done {:.2f}'.format(time.time()-start), flush=True)  
        
        elif hcat_file is not None:
            init_cols = ['x', 'y', 'z', 'vx', 'vy', 'vz','Mh', 'Rh', 'Rs', 'c', 'Vrms', 'halo_id']

            if isinstance(hcat_file, str):
                self.hcat = Catalog.read(hcat_file)

            elif isinstance(hcat_file, Catalog):
                self.hcat = hcat_file

            elif isinstance(hcat_file, dict):
                self.hcat = Catalog.from_dict(hcat_file)

            elif isinstance(hcat_file, np.ndarray):
                if hcat_file.dtype.names is None:
                    raise TypeError(f'Halo catalog must be a structured ndarray with field {init_cols}')
                self.hcat = Catalog.from_array(hcat_file)

        else: 
            if self.args['hcat']['path_to_sim'] is None:
                raise FileNotFoundError('Provide a halo catalog or a filename to read it')
            if not os.path.exists(self.args['hcat']['path_to_sim']): 
                raise FileNotFoundError('{} not found'.format(self.args['hcat']['path_to_sim']))
            self.hcat = Catalog.read(self.args['hcat'][['path_to_sim']]) 

        if self.boxsize is None: raise ValueError('Boxsize not provided')

        # init cosmology 
        if self.cosmo is None:
            self.init_cosmology()

        if 'log10_Mh' not in self.hcat.columns(): 
            self.hcat['log10_Mh'] = np.log10(self.hcat['Mh'])
        

        if 'c' not in self.hcat.columns():
            print('Concentration column "c" is not provided. The concentration is computed from colossus package using mass-concentration relation of {} with {} as mass definition'.format(self.args['cm_relation'], self.args['mass_def']), flush=True)
            from pinnochio_io import get_concentration
            self.hcat['c'] = get_concentration(self.hcat['Mvir'], cosmo=self.cosmo, mdef=self.args['mass_def'], cmrelation=self.args['cm_relation'])
        
        try :
            self._fun_cHOD, self._fun_sHOD = {}, {}
            for tr in self.args['tracers']:
                self._fun_cHOD[tr] = globals()['_'+self.args[tr]['HOD_model']]
                self._fun_sHOD[tr] = globals()['_'+self.args[tr]['sat_HOD_model']]
        except :
            from . import HOD_models 
            help(HOD_models)
            raise ValueError('{} not implemented in HOD models'.format(self.args['HOD_param']['HOD_model']))
                
        if self.args['assembly_bias']:
            self._compute_assembly_bias_columns()

    def init_cosmology(self):
        """
        Initialize the cosmology model using Cosmoprimo.

        This method attempts to set the `self.cosmo` attribute using cosmological parameters
        defined in the input configuration (`self.args['cosmo']`). It supports loading:
        
        - AbacusSummit cosmologies if the simulation name (`sim_name`) is present
        - Custom cosmology from provided parameters if no Abacus name is specified

        It uses the `cosmoprimo` package to construct the cosmology and raises a warning if
        the library is not available.

        Returns
        -------
        None
            Sets the `self.cosmo` attribute to an instance of `cosmoprimo.Cosmology`
            or leaves it as `None` if the initialization fails.

        Raises
        ------
        ImportWarning
            If `cosmoprimo` is not installed, a warning is issued and no cosmology is set.

        Notes
        -----
        - Required fields in `self.args['cosmo']` are passed as keyword arguments to
        `cosmoprimo.fiducial.Cosmology`.
        - For AbacusSummit cosmologies, the cosmology identifier is extracted from
        the `sim_name` string.
        - Cosmology is used later for computing RSD and distance-based statistics.
        """

        try: 
            from cosmoprimo.fiducial import Cosmology, AbacusSummit
            if "sim_name" in self.args["hcat"].keys():
                print('Initialize Abacus c{} cosmology'.format(self.args['hcat']['sim_name'].split('_c')[-1][:3]))
                self.cosmo = AbacusSummit(self.args['hcat']['sim_name'].split('_c')[-1][:3]).get_background(engine=self.args['cosmo']['engine'], flush=True)   
            else:
                print('Initialize custom cosmology from the "cosmo" parameters', flush=True)
                self.cosmo = Cosmology(**{k: v for k, v in self.args['cosmo'].items() if v is not None})
        except ImportError:
            import warnings
            warnings.warn('Could not import cosmoprimo. Install cosmoprimo with "python -m pip install python -m pip install git+https://github.com/cosmodesi/cosmoprimo".\n'\
                  'Cosmology needed to apply RSD when computing correlations. No cosmology set.')
            self.cosmo = None

    def __init_hod_param(self, tracer):

        """
        Initialize the HOD (Halo Occupation Distribution) parameters for mock galaxy creation based on the selected HOD model.

        Parameters
        ----------
        tracer : str
            The key identifying the specific tracer (e.g., galaxy type) for which the HOD parameters will be initialized. Need to be in self.tracers
        
        Returns
        -------
        tuple
            A tuple containing:
            - np.float64 array of central galaxy parameters (hod_list_param_cen).
            - np.array of satellite galaxy parameters if applicable (hod_list_param_sat).
            - np.array of assembly bias parameters if applicable (hod_param_ab).

        Raises
        ------
        ValueError
            If the provided HOD model is not recognized or supported.
        
        Notes
        -----
        The function handles the following HOD models:
            - 'HMQ': Central galaxy parameters with specific components.
            - 'GHOD', 'LNHOD', 'SHOD': Central galaxy parameters with different components.
            - 'SFHOD', 'mHMQ': Central galaxy parameters with a 'gamma' component.
        
        Satellite galaxy parameters are initialized if 'satellites' is set to True, 
        and assembly bias parameters are initialized if 'assembly_bias' is provided.

        Example
        -------
        For a given 'tracer' (e.g., 'galaxy'), the function might initialize:
        - Central parameters: [Ac, log_Mcent, sigma_M, gamma, Q, pmax]
        - Satellite parameters: [As, M_0, M_1, alpha]
        - Assembly bias parameters: List derived from the 'assembly_bias' dictionary.
        """

        hod_list_param_sat, hod_param_ab = None, None
        if self.args[tracer]['HOD_model'] == 'HMQ':
            hod_list_param_cen = [self.args[tracer]['Ac'], self.args[tracer]['log_Mcent'], self.args[tracer]['sigma_M'], self.args[tracer]['gamma'], self.args[tracer]['Q'], self.args[tracer]['pmax']]               
        elif ('GHOD' in self.args[tracer]['HOD_model']) | ('LNHOD' in self.args[tracer]['HOD_model']) | ('SHOD' in self.args[tracer]['HOD_model']):
            hod_list_param_cen = [self.args[tracer]['Ac'], self.args[tracer]['log_Mcent'], self.args[tracer]['sigma_M']]
        elif ('SFHOD' in self.args[tracer]['HOD_model']) | ('mHMQ' in self.args[tracer]['HOD_model']):
            hod_list_param_cen = [self.args[tracer]['Ac'], self.args[tracer]['log_Mcent'], self.args[tracer]['sigma_M'], self.args[tracer]['gamma']]
        else: 
            raise ValueError('{} not implemented in HOD models'.format(self.args['HOD_param']['HOD_model']))
        
        if self.args[tracer]['satellites']:
            hod_list_param_sat = np.array([self.args[tracer]['As'], self.args[tracer]['M_0'], self.args[tracer]['M_1'], self.args[tracer]['alpha']], dtype='float64')

        if self.args[tracer]['assembly_bias']:
            hod_param_ab = np.array(list(self.args[tracer]['assembly_bias'].values()), dtype='float64').T

        return np.float64(hod_list_param_cen), hod_list_param_sat, hod_param_ab


    def get_ds_fac(self, tracer, verbose=False):
        """
        Calculate the density scaling factor based on the specified tracer and density value.

        Parameters
        ----------
        tracer : str
            The key identifying the specific tracer (e.g., galaxy type) for which the density scaling factor is calculated.
        verbose : bool, optional
            If True, prints information about the density and scaling factor. Default is False.

        Returns
        -------
        float
            The density scaling factor. If no density is set, returns 1.

        Notes
        -----
        This function uses the density specified in `self.args[tracer]['density']` to calculate the density scaling factor.
        If the density is specified as a float, it multiplies it by the cube of the box size and divides by the number of galaxies
        (obtained using the `ngal` method). If no density is specified, the function defaults to returning 1.

        Example
        -------
        If `self.args[tracer]['density']` is set to `0.01` and `self.boxsize = 100`:
        - The density scaling factor will be calculated as `0.01 * 100^3 / self.ngal(tracer)[0]`.
        If no density is specified, the function simply returns 1.

        If `verbose` is set to True, the following message will be printed:
        - "Set density to 0.01 gal/Mpc/h".
        """
        
        if isinstance(self.args[tracer]['density'], float):
            if verbose:
                print('Set density to {} gal/Mpc/h'.format(self.args[tracer]['density']))
            return self.args[tracer]['density']*self.boxsize**3 /self.ngal(tracer)[0]
        else:
            if verbose: print('No density set')
            return 1 
    
    def ngal(self, tracer, verbose=False):
        """
        Return the number of galaxy and the satelitte fraction 
        
        Parameters
        ----------
            tracer: str
                Name of the galaxy tracer in self.tracers 
            verbose (bool, optional): Defaults to False.

        Returns
        -------
        ngal: float
            Total number of galaxies expected
        
        fsat: float
            Expected fraction of satellite galaxy (n_sat/ngal)
        """
        start = time.time()
        hod_list_param_cen, hod_list_param_sat, _ = self.__init_hod_param(tracer)
        ngal, fsat = compute_ngal(self.hcat['log10_Mh'], self._fun_cHOD[tracer], self._fun_sHOD[tracer], self.args['nthreads'], 
                                hod_list_param_cen, hod_list_param_sat, self.args[tracer]['conformity_bias'])
        if verbose:
            print(time.time()-start)
        return ngal, fsat   


    def calc_env_factor(self, cellsize=5, resampler='cic'):

        """
        Compute density around each halos on a mesh. Based on pypower CatalogMesh https://pypower.readthedocs.io/en/latest/api/api.html#pypower.mesh.CatalogMesh

        Parameters
        ----------
        cellsize : array, float, default=5
            Physical size of mesh cells.
        
        resampler : string, ResampleWindow, default='tsc'
            Resampler used to assign particles to the mesh.
            Choices are ['ngp', 'cic', 'tcs', 'pcs'].

        Returns
        -------
        None
            Add column named ``env`` in the halo catalog self.hcat 
        """
        from pypower import CatalogMesh

        print(f'Compute environment in cellsize {cellsize}...', flush=True)
        import warnings
        warnings.warn('Enviroment factor computed on halo not particles, need to be updated. Assume boxcenter to 0')
        positions = [self.hcat['x'], self.hcat['y'], self.hcat['z']]
        mesh = CatalogMesh(data_positions=positions, cellsize=cellsize, boxsize=self.boxsize,
                            boxcenter=0., resampler=resampler, interlacing=0)
        painted_mesh = mesh.to_mesh(field='data')
        self.hcat['env'] = painted_mesh.readout(
            np.array(positions).T, resampler=resampler)
        print('Done !', flush=True)


    def _compute_assembly_bias_columns(self):
        """
        Initialize assembly bias columns for all tracers.

        This method creates the assembly bias columns for each tracer by iterating over the unique assembly bias
        column names defined in the `assembly_bias` parameter file for each tracer. For each column, the
        `set_assembly_bias_values` method is called to assign the values to the halo catalog.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This function relies on the existence of the `self.args['tracers']` and `self.args[tracer]['assembly_bias']`
        configurations. The assembly bias columns are created using the column names defined in the `assembly_bias`
        keys for each tracer.

        Example
        -------
        If there are multiple tracers defined in `self.args['tracers']` and each tracer has specific assembly bias
        columns defined in `self.args[tracer]['assembly_bias']`, this method will iterate over all of them and create
        the corresponding assembly bias columns in the halo catalog.
        """

        ab_proxy = np.unique([[l for l in ll] for ll in [list(self.args[tr]['assembly_bias'].keys()) for tr in self.args['tracers']]])
        for ab in ab_proxy:
            self.set_assembly_bias_values(ab)

    def set_assembly_bias_values(self, col, bins=50):

        """
        Assign ranked values for assembly bias computation based on a specific column.

        This method assigns ranked values linearly between -0.5 and 0.5 for the assembly bias computation. The
        values are based on a histogram of halo masses (`log10_Mh`), and the `col` parameter specifies the column
        in the halo catalog that will be used for the ranking. The method first checks if the required column 
        (`ab_{col}`) already exists; if not, it computes the values based on the input column and adds them to the
        halo catalog.

        Parameters
        ----------
        col : str
            The column name in the halo catalog used to compute assembly bias values. 
            This column is expected to be present in the halo catalog.
        bins : int, optional
            The number of bins used for mass binning in the histogram of halo masses. Default is 50.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the specified column is not present in the halo catalog, a `ValueError` is raised.

        Notes
        -----
        The method uses `log10_Mh` values to bin the halos and ranks the halos in each bin based on the specified column.
        The assembly bias values are assigned linearly between -0.5 and 0.5 within each bin.
        If the column `env` is requested but not present, the `calc_env_factor()` method is called to compute it first.
        Additionally, the `ab_{col}` column is only added if it does not already exist in the halo catalog.

        Example
        -------
        If `col = 'env'`, the method checks if an assembly bias column for `env` exists. If not, it computes and adds it.
        The halos are binned based on their mass, and each halo is ranked within its mass bin. A value between -0.5 and 0.5
        is assigned to each halo in the catalog based on the ranking of the `env` column.
        """

        if f'ab_{col}' in self.hcat.columns():
            return 0
        
        if (col == 'env') & ('env' not in self.hcat.columns()):
            self.calc_env_factor()

        if col not in self.hcat.columns():
            raise ValueError(f'{col} not in halo catalog columns')
        
        print(f'Set value for assembly bias according {col}...', flush=True)
        
        nb, mbins = np.histogram(self.hcat['log10_Mh'], bins=bins)
        self.hcat[f'ab_{col}'] = np.zeros_like(self.hcat['log10_Mh'])
        mask_bin_hcat = [(self.hcat['log10_Mh'] > b_inf-1e-6) &  (self.hcat['log10_Mh'] <= b_sup) for b_inf, b_sup in zip(mbins[:-1], mbins[1:])]
        idx_sort_mbins = []
        i = 1
        for mask in mask_bin_hcat:
            hcat_bin = self.hcat[mask]
            f_c = np.zeros_like(hcat_bin['log10_Mh'])
            idx_sort_mbins += [np.argsort(hcat_bin[col])[::-1]]
            np.put(f_c, idx_sort_mbins[-1], np.linspace(-0.5, 0.5, len(hcat_bin[col])))
            self.hcat[f'ab_{col}'][mask] = f_c
            if i % 10 == 0:
                print(f'{i*2}% done...', flush=True)
            i += 1

    def make_mock_cat(self, tracers=None, fix_seed=None, verbose=True):
        """
        Generate HOD mock catalogs.

        This method creates mock galaxy catalogs based on the Halo Occupation Distribution (HOD) model for each 
        specified tracer. It computes the central and satellite galaxies, assigns them to halos, and optionally 
        includes assembly bias effects, conformity bias, and uses particle-level data for satellite galaxy 
        positions. It handles multiple tracers and provides options for reproducibility and verbose output.

        Parameters
        ----------
        tracers : list or str, optional
            Name(s) of the galaxy tracers (e.g., 'LRG', 'ELG') to include in the mock catalog. If None, all 
            tracers defined in `self.tracers` are considered. Defaults to None.
        fix_seed : int, optional
            Fix the seed for reproducibility. This is useful for ensuring consistent results when using a fixed 
            number of threads (`nthreads`). Defaults to None.
        verbose : bool, optional
            If True, the function prints progress messages during execution. Defaults to True.

        Returns
        -------
        final_cat : dict
            A dictionary containing mock catalogs for each tracer specified. Each catalog is represented by a `Catalog`
            object, which contains the generated galaxy data (centrals and satellites) for the corresponding tracer.

        Notes
        -----
        - The method relies on HOD models defined for each tracer in `self.args[tracer]['HOD_model']` and `self.args[tracer]['sat_HOD_model']`.
        - If `self.args['assembly_bias']` is enabled, assembly bias columns will be computed and included in the mock catalog.
        - The `fix_seed` parameter ensures that the mock catalogs are generated in a reproducible manner, but it requires consistent 
        thread configurations (`self.args['nthreads']`).
        - When `tracers` includes both 'ELG' and 'LRG', the method handles the case where both tracers share the same halo by placing
        one LRG at the center and positioning other galaxies (like ELGs) based on the NFW profile.

        Example
        -------
        To generate mock catalogs for both 'LRG' and 'ELG' tracers with fixed seed for reproducibility:

        final_cat = mock_catalog.make_mock_cat(tracers=['LRG', 'ELG'], fix_seed=42)
        """

        rng = np.random.RandomState(seed=fix_seed)
        timeall = time.time()

        start = time.time()
        
        if tracers is None: 
            tracers = self.args['tracers']
        else:
            tracers = tracers if isinstance(tracers, list) else [tracers]
        if verbose:
            print('Create mock catalog for {}'.format(tracers), flush=True)         
        
        if self.args['assembly_bias']:
            self._compute_assembly_bias_columns()

        final_cat = {}
        mask_id = np.ones(self.hcat.size, dtype=bool)
        count_gal = {}
        for tracer in tracers:
            self._fun_cHOD[tracer] = globals()['_'+self.args[tracer]['HOD_model']]
            self._fun_sHOD[tracer] = globals()['_'+self.args[tracer]['sat_HOD_model']]
            if verbose:
                print('Run HOD for {}'.format(tracer), flush=True)         
            hod_list_param_cen, hod_list_param_sat, hod_list_ab_param = self.__init_hod_param(tracer)
            
            if self.args[tracer]['satellites']:
                ds = self.get_ds_fac(tracer, verbose=verbose)
                if (hod_list_param_cen[0]*ds > 1) or (hod_list_param_cen[0]*ds > 1):
                    import warnings
                    warnings.warn(f'Ac={hod_list_param_cen[0]} or As={hod_list_param_cen[0]} is > 1, the density is not fixed to {self.args[tracer]["density"]}')
                else : 
                    hod_list_param_cen[0] *= ds
                    hod_list_param_sat[0] *= ds

            if fix_seed is not None:
                seed = rng.randint(0, 4294967295, self.args['nthreads'])
            else:
                seed = None
            
            if self.args['assembly_bias'] & (hod_list_ab_param is not None):
                cols_ab =  ['ab_'+col for col in self.args[tracer]['assembly_bias'].keys()]
                if np.all([col in self.hcat.columns() for col in cols_ab]):
                    ab_arr =  np.vstack([self.hcat[col] for col in cols_ab]).T
                else:
                    import warnings
                    warnings.warn('Precomputed columns for assembly bias have not been found {}. Continue without assembly bias.'.format(cols_ab))
                    hod_list_ab_param=None
            else:
                hod_list_ab_param, ab_arr = None, None

            cent, sat, cond_cent, proba_sat = compute_N(self.hcat['log10_Mh'], self._fun_cHOD[tracer], self._fun_sHOD[tracer], hod_list_param_cen, hod_list_param_sat, hod_list_ab_param,
                                                            self.args['nthreads'], ab_arr, self.args[tracer]['conformity_bias'], seed)

            mask_cent = cond_cent == 1
            Nb_sat = proba_sat.sum()
    
            cent_cat = self.hcat[mask_cent]
            cent_cat['Central'] = np.ones(cent_cat['x'].size,dtype='int')
            if (cent > 1).any():
                import warnings
                warnings.warn(f'WARNING Ncent>1 {(cent > 1).sum()} times')
            if verbose:
                print("HOD Computed", time.time() - start, flush=True)         
            
            mask_sat = proba_sat > 0
            if (not self.args[tracer]['satellites']) | (Nb_sat == 0):
                Nb_sat=0
                final_cat[tracer] = cent_cat

            else:
                start = time.time()
                if verbose:
                    print("Start satellite assignement", flush=True)
                list_nsat = proba_sat[mask_sat]
                sat_cat = Catalog.from_array(np.repeat(self.hcat[mask_sat].to_array(), list_nsat))

                if fix_seed is not None:
                    seed = rng.randint(0, 4294967295, self.args['nthreads'])
                else:
                    seed = None
                
                if self.args['use_particles'] & (self.part_subsamples is not None):
                    if verbose: print('Using particles', flush=True)
                    if 'Abacus' in self.args['hcat']['sim_name']:
                        mask_nfw = compute_sat_from_abacus_part(self.part_subsamples['pos'].T[0],self.part_subsamples['pos'].T[1],self.part_subsamples['pos'].T[2],
                            self.part_subsamples['vel'].T[0], self.part_subsamples['vel'].T[1],self.part_subsamples['vel'].T[2],
                            sat_cat['x'], sat_cat['y'], sat_cat['z'], sat_cat['vx'], sat_cat['vy'], sat_cat['vz'],
                            self.hcat['npoutA'][mask_sat], self.hcat['npstartA'][mask_sat], list_nsat, np.insert(np.cumsum(list_nsat), 0, 0), self.args['nthreads'], seed=seed)
                        if verbose: print(f'{mask_nfw.sum()} satellites will be positioned using NFW', flush=True)

                    elif self.part_subsamples is not None:
                        if fix_seed is not None:
                            seed = rng.randint(0, 4294967295, sat_cat.size)
                        else:
                            seed = None
                        mask_nfw = compute_sat_from_part(self.part_subsamples['halo_id'], sat_cat['halo_id'], list_nsat, self.args['nthreads'], seed=seed)
                        if verbose: print(f'{mask_nfw.sum()} satellites will be positioned using NFW', flush=True)
                else:
                    mask_nfw = np.ones(Nb_sat, dtype=bool)
                
                if mask_nfw.sum() > 0:
                    
                    if fix_seed is not None:
                        seed1 = rng.randint(0, 4294967295, self.args['nthreads'])
                    else:
                        seed1 = None
                    rd_pos = getPointsOnSphere_jit(Nb_sat, np.minimum(Nb_sat, self.args['nthreads']), seed1)
                    if self.args[tracer]['vel_sat'] == 'NFW':
                        seed2 = rng.randint(0, 4294967295, self.args['nthreads'])
                        rd_v = getPointsOnSphere_jit(Nb_sat, np.minimum(Nb_sat, self.args['nthreads']), seed2)
                        ut = np.cross(rd_pos,rd_v, axis=1) 
                        rd_vel = (ut.T / np.linalg.norm(ut, axis=1).flatten()).T
                    else:
                        rd_vel = np.ones_like(rd_pos)
                    
                    vrms_h = sat_cat['Vrms'][mask_nfw] if 'Vrms' in sat_cat.columns() else np.zeros_like(sat_cat['vx'][mask_nfw])

                    sat_cat['x'][mask_nfw], sat_cat['y'][mask_nfw], sat_cat['z'][mask_nfw], \
                    sat_cat['vx'][mask_nfw], sat_cat['vy'][mask_nfw], sat_cat['vz'][mask_nfw] = compute_fast_NFW(sat_cat['x'][mask_nfw], sat_cat['y'][mask_nfw], sat_cat['z'][mask_nfw],
                                                                                                                            sat_cat['vx'][mask_nfw], sat_cat['vy'][mask_nfw], sat_cat['vz'][mask_nfw],
                                                                                                                            sat_cat['c'][mask_nfw], sat_cat['Mh'][mask_nfw], sat_cat['Rh'][mask_nfw], 
                                                                                                                            rd_pos, rd_vel, exp_frac=self.args[tracer]['exp_frac'], 
                                                                                                                            exp_scale=self.args[tracer]['exp_scale'], nfw_rescale=self.args[tracer]['nfw_rescale'],
                                                                                                                            vrms_h=vrms_h, f_sigv=self.args[tracer]['f_sigv'], v_infall=self.args[tracer]['v_infall'], 
                                                                                                                            vel_sat=self.args[tracer]['vel_sat'], Nthread=self.args['nthreads'], seed=seed)

                sat_cat['Central'] = sat_cat.zeros()

                if verbose:
                    print("Satellite assignement done", time.time() - start, flush=True)

                
                final_cat[tracer] = Catalog.concatenate((cent_cat,sat_cat))
                final_cat[tracer]['TRACER'] = [tracer]*final_cat[tracer].size
                if verbose:
                    print('{} mock catalogue done'.format(tracer), time.time()-start, flush=True)
                    print("{} central galaxies, {} satellites, fraction of satellite {:.2f} ".format(mask_cent.sum(),Nb_sat, Nb_sat/final_cat[tracer].size), flush=True)
                
            mask_id &= (mask_cent | mask_sat)
            count_gal[tracer] = np.int64(proba_sat + cond_cent)

            if verbose:
                print("Done overall time ", tracer, time.time() - timeall, flush=True)

        # When LRG and ELG are in the same halo, put 1 LRG at the center and all other galaxies are position following NFW profile
        if ((tracers == ['ELG', 'LRG']) | (tracers == ['LRG', 'ELG'])) & (mask_id.sum() >0):
            mask_elg = np.in1d(final_cat['ELG']['halo_id'], final_cat['LRG']['halo_id'])
            mask_lrg = np.in1d(final_cat['LRG']['halo_id'], final_cat['ELG']['halo_id'])
            
            cen_LRG = self.hcat[mask_id]
            cen_LRG['Central'] = cen_LRG.ones()

            if ((count_gal['LRG'] > 1) & mask_id).sum() > 0:
                sat_LRG = Catalog.from_array(np.repeat(self.hcat[(count_gal['LRG'] > 1) & mask_id].to_array(), count_gal['LRG'][(count_gal['LRG'] > 1) & mask_id]-1))
                sat_LRG = self.init_elg_sat_for_lrg('LRG', sat_LRG, fix_seed=fix_seed)
                final_cat['LRG'][mask_lrg] = Catalog.concatenate(cen_LRG, sat_LRG)
            else:
                final_cat['LRG'][mask_lrg] = cen_LRG
            
            sat_ELG = Catalog.from_array(np.repeat(self.hcat[mask_id].to_array(), count_gal['ELG'][mask_id]))
            sat_ELG = self.init_elg_sat_for_lrg('ELG', sat_ELG, fix_seed=fix_seed)
            final_cat['ELG'][mask_elg] = sat_ELG

        final_cat = Catalog.concatenate(list(final_cat.values())) 
        return final_cat



    def init_elg_sat_for_lrg(self, tracer, sat_cat, fix_seed=None):

        """
        Draft function to assign ELG satellites around LRG central galaxies in a NFW profile 
        if they are in the same halo.

        This function generates the positions and velocities of Emission Line Galaxies (ELGs) 
        around Luminous Red Galaxies (LRGs) using a Navarro–Frenk–White (NFW) profile. It 
        ensures that the ELG satellites are placed according to the halo's mass distribution, 
        and their velocities are assigned based on the chosen model (NFW or random normal).

        Parameters
        ----------
        tracer : str
            The name of the tracer (e.g., 'LRG') used to determine the velocity model and 
            other simulation parameters for satellite galaxies.
        sat_cat : Catalog
            A catalog of the satellite galaxies, which includes properties such as 
            position, velocity, concentration, mass, and radius. The function will modify 
            the positions and velocities of these satellites.
        fix_seed : int, optional
            A seed for random number generation, ensuring reproducibility of the mock catalog. 
            Defaults to None, meaning that a random seed will be generated internally.

        Returns
        -------
        Catalog
            The input satellite catalog (`sat_cat`) is updated with new positions, velocities, 
            and a `Central` flag (set to 0 for satellites).
        
        Notes
        -----
        - The position of satellites is computed on a spherical shell using `getPointsOnSphere_jit`.
        - The velocity of the satellites can be generated in two ways:
            - 'NFW': Satellites are assigned velocities based on the NFW profile.
            - 'rd_normal': Satellites are assigned random velocities.
        - If the `Vrms` column is available in `sat_cat`, it is used to adjust the velocity distribution.
        - This function directly modifies the `sat_cat` and returns the updated catalog.
        """

        rng = np.random.RandomState(seed=fix_seed)

        if fix_seed is not None:
            seed = rng.randint(0, 4294967295, self.args['nthreads'])
        else:
            seed = None

        Nb_sat = sat_cat.size

        if fix_seed is not None:
            seed1 = rng.randint(0, 4294967295, self.args['nthreads'])
        else:
            seed1 = None
        rd_pos = getPointsOnSphere_jit(Nb_sat, np.minimum(Nb_sat, self.args['nthreads']), seed1)
        if self.args[tracer]['vel_sat'] == 'NFW':
            seed2 = rng.randint(0, 4294967295, self.args['nthreads'])
            rd_v = getPointsOnSphere_jit(Nb_sat, np.minimum(Nb_sat, self.args['nthreads']), seed2)
            ut = np.cross(rd_pos,rd_v, axis=1) 
            rd_vel = (ut.T / np.linalg.norm(ut, axis=1).flatten()).T
        elif self.args[tracer]['vel_sat'] == 'rd_normal':
            rd_vel = np.ones_like(rd_pos)
        vrms_h = sat_cat['Vrms'] if 'Vrms' in sat_cat.columns() else np.zeros_like(sat_cat['vx'])

        sat_cat['x'], sat_cat['y'], sat_cat['z'], sat_cat['vx'], sat_cat['vy'], sat_cat['vz'] = compute_fast_NFW(sat_cat['x'], sat_cat['y'], sat_cat['z'],
                                                                                                                    sat_cat['vx'], sat_cat['vy'], sat_cat['vz'],
                                                                                                                    sat_cat['c'], sat_cat['Mh'], sat_cat['Rh'], 
                                                                                                                    rd_pos, rd_vel, exp_frac=0, 
                                                                                                                    exp_scale=self.args[tracer]['exp_scale'], nfw_rescale=self.args[tracer]['nfw_rescale'],
                                                                                                                    vrms_h=vrms_h, f_sigv=self.args[tracer]['f_sigv'], v_infall=self.args[tracer]['v_infall'], 
                                                                                                                    vel_sat=self.args[tracer]['vel_sat'], Nthread=self.args['nthreads'], seed=seed)
        
        sat_cat['Central'] = sat_cat.zeros()
        return sat_cat


    def get_2PCF(self, cats, tracers=None, ells=None, R1R2=None, verbose=True):
        """
        Compute the two-point correlation function (2PCF) for a given mock catalog in a cubic box.

        This function calculates the two-point correlation function (2PCF) for specified galaxy tracers 
        in a mock catalog. It computes the correlation for multiple tracers if provided, and returns 
        the 2PCF and its separation distance for each tracer.

        Parameters
        ----------
        cats : dict
            A dictionary of mock catalogs where the keys are the names of the tracers 
            and the values are the corresponding catalogs (e.g., 'LRG', 'ELG').
        tracers : list or str, optional
            A list of tracer names (keys in `cats`) for which the 2PCF should be computed. 
            If None, all tracers in `self.args['tracers']` are considered. Defaults to None.
        ells : tuple of int, optional
            Multipoles to project onto. If None, ells in `self.args['2PCF_settings']['multipole_index']` are considered. Default to None.
        R1R2 : tuple or None, optional
            A tuple defining a range for R1 and R2 for the 2PCF computation. If None, 
            the default values will be used. Defaults to None.
        verbose : bool, optional
            If True, prints progress and computation time for each tracer. Defaults to True.

        Returns
        -------
        s_all : list
            A list of separation distances corresponding to the computed 2PCF for each tracer.
        xi_all : list
            A list of the two-point correlation function (2PCF) values for each tracer.

        Notes
        -----
        - The function uses `apply_rsd` to account for redshift space distortions (RSD) if enabled and Cosmology set.
        - The separation distances `s` and the correlation values `xi` are calculated using the 
        `compute_2PCF` function, and the results are stored for each tracer.
        - The results are returned as lists (`s_all` and `xi_all`) when multiple tracers are provided.
        - The function supports a log-scale binning option for radial bins if `bin_logscale` is True.
        - The output is either the 2PCF for a single tracer (if only one tracer is given) or for 
        all tracers provided in the list.

        Example
        -------
        s, xi = get_2PCF(cats, tracers=['LRG', 'ELG'])
        """

        if tracers is None: 
            tracers = self.args['tracers'] 
        tracers = tracers if isinstance(tracers, list) else [tracers]
        

        if self.args['2PCF_settings']['edges_smu'] is None:
            if self.args['2PCF_settings']['bin_logscale']:
                r_bins = np.geomspace(self.args['2PCF_settings']['rmin'], self.args['2PCF_settings']['rmax'], self.args['2PCF_settings']['n_r_bins'])
            else:
                r_bins = np.linspace(self.args['2PCF_settings']['rmin'], self.args['2PCF_settings']['rmax'], self.args['2PCF_settings']['n_r_bins'])
            
            self.args['2PCF_settings']['edges_smu'] = (r_bins, np.linspace(-self.args['2PCF_settings']['mu_max'], self.args['2PCF_settings']['mu_max'], self.args['2PCF_settings']['n_mu_bins']))
        ells = self.args['2PCF_settings']['multipole_index'] if ells is None else ells
        s_all, xi_all = [],[]
        for tr in tracers:
            mock_cat = cats[cats['TRACER'] == tr]
            if verbose:
                print('#Compute xi(s,mu) using l={} for {}...'.format(ells, tr), flush=True)
                time1 = time.time()

            if self.cosmo is not None:
                if self.args['2PCF_settings']['rsd']:
                    pos = apply_rsd (mock_cat, self.args['hcat']['z_simu'], self.boxsize, self.cosmo, self.H_0, self.args['2PCF_settings']['los'], self.args[tr]['vsmear'])
            else:
                if self.args['2PCF_settings']['rsd']:
                    print('Cosmology not set, does not apply rsd', flush=True)
                pos = mock_cat['x']%self.boxsize, mock_cat['y']%self.boxsize, mock_cat['z']%self.boxsize
            
            s, xi = compute_2PCF(pos, self.args['2PCF_settings']['edges_smu'], self.boxsize, ells, self.args['2PCF_settings']['los'], self.args['nthreads'], R1R2=R1R2)
            if verbose:
                print('#Done in {:.3f} s'.format(time.time()-time1), flush=True)
            if len(tracers) > 1:
                s_all += [s]
                xi_all += [xi]
            else: 
                return s, xi
        return s_all, xi_all

    def get_wp(self, cats, tracers=None, R1R2=None, verbose=True):
        """
        Compute the projected two-point correlation function (wp) for a given mock catalog in a cubic box.

        This function calculates the projected two-point correlation function (wp) for specified galaxy tracers 
        in a mock catalog. It can compute wp for multiple tracers, returning the results for each.

        Parameters
        ----------
        cats : dict
            A dictionary of mock catalogs where the keys are the names of the tracers 
            and the values are the corresponding catalogs (e.g., 'LRG', 'ELG').
        tracers : list or str, optional
            A list of tracer names (keys in `cats`) for which the wp should be computed. 
            If None, all tracers in `self.args['tracers']` are considered. Defaults to None.
        R1R2 : tuple or None, optional
            A tuple defining a range for R1 and R2 for the wp computation. If None, 
            the default values will be used. Defaults to None.
        verbose : bool, optional
            If True, prints progress and computation time for each tracer. Defaults to True.

        Returns
        -------
        rp_all : list
            A list of projected separation distances corresponding to the computed wp for each tracer.
        wp_all : list
            A list of the projected two-point correlation function (wp) values for each tracer.

        Notes
        -----
        - The function uses `apply_rsd` to account for redshift space distortions (RSD) if enabled and Cosmology set.
        - The projected separation distances `rp` and the correlation values `wp` are calculated using the 
        `compute_wp` function, and the results are stored for each tracer.
        - The results are returned as lists (`rp_all` and `wp_all`) when multiple tracers are provided.
        - The function supports a log-scale binning option for radial bins if `bin_logscale` is True.
        - The output is either the wp for a single tracer (if only one tracer is given) or for 
        all tracers provided in the list.

        Example
        -------
        rp, wp = get_wp(cats, tracers=['LRG', 'ELG'])
        """

        if tracers is None: 
            tracers = self.args['tracers'] 
        tracers = tracers if isinstance(tracers, list) else [tracers]

        if self.args['2PCF_settings']['edges_rppi'] is None:
            if self.args['2PCF_settings']['bin_logscale']:
                r_bins = np.geomspace(self.args['2PCF_settings']['rp_min'], self.args['2PCF_settings']['rp_max'], self.args['2PCF_settings']['n_rp_bins']+1, endpoint=(True))
            else:
                r_bins = np.linspace(self.args['2PCF_settings']['rp_min'], self.args['2PCF_settings']['rp_max'], self.args['2PCF_settings']['n_rp_bins']+1)
            self.args['2PCF_settings']['edges_rppi'] = (r_bins, np.linspace(-self.args['2PCF_settings']['pimax'], self.args['2PCF_settings']['pimax'], 2*self.args['2PCF_settings']['pimax']+1))

        rp_all, wp_all = [],[]
        for tr in tracers:
            mock_cat = cats[cats['TRACER'] == tr]
            if verbose:
                print('#Compute wp for {}...'.format(tr), flush=True)
                time1 = time.time()

            if self.cosmo is not None:
                if self.args['2PCF_settings']['rsd']:
                    pos = apply_rsd (mock_cat, self.args['hcat']['z_simu'], self.boxsize, self.cosmo, self.H_0, self.args['2PCF_settings']['los'], self.args[tr]['vsmear'])
            else:
                if self.args['2PCF_settings']['rsd']:
                    print('Cosmology not set, does not apply rsd', flush=True)
                pos = mock_cat['x']%self.boxsize, mock_cat['y']%self.boxsize, mock_cat['z']%self.boxsize

            rp, wp = compute_wp(pos, self.args['2PCF_settings']['edges_rppi'], self.boxsize, self.args['2PCF_settings']['pimax'], self.args['2PCF_settings']['los'],  self.args['nthreads'], R1R2=R1R2)
            if verbose:
                print('Done in {:.3f} s'.format(time.time()-time1), flush=True)
            if len(tracers) > 1:
                rp_all += [rp]
                wp_all += [wp]
            else: 
                return rp, wp
        return rp_all, wp_all
    

    def get_crosswp(self, cats, tracers, R1R2=None, verbose=True):
        """
        Compute the projected correlation and cross-correlation functions (wp) for a given mock catalog in a cubic box.

        This function computes the projected two-point correlation and cross-correlation function (wp) for pairs of tracers in the 
        mock catalogs. It calculates wp for all combinations of tracers provided, handling redshift space distortions 
        (RSD) if enabled.

        Parameters
        ----------
        cats : dict
            A dictionary of mock catalogs where each key is a tracer and its corresponding catalog is the value 
            (e.g., 'LRG', 'ELG').
        tracers : list of str
            A list of tracer names (keys in `cats`) for which the cross wp should be computed. The function 
            computes the wp for all pairs of tracers in the list.
        R1R2 : tuple or None, optional
            A tuple defining a range for R1 and R2 for the wp computation. If None, the default values will be used.
            Defaults to None.
        verbose : bool, optional
            If True, prints progress and computation time for each pair of tracers. Defaults to True.

        Returns
        -------
        res_dict : dict
            A dictionary where the keys are the concatenated names of tracer pairs (e.g., 'LRG_ELG') and the 
            values are the corresponding projected two-point correlation functions (wp) for each pair.

        Notes
        -----
        - The function computes the cross-correlation wp for all unique pairs of tracers from the input list.
        - If redshift space distortions (RSD) are enabled, the positions of galaxies in the catalogs are adjusted accordingly.
        - The results are stored in `res_dict` with keys in the format 'tracer1_tracer2', where each value is the wp 
        corresponding to the pair of tracers.
        - The function uses `compute_wp` to calculate the wp for each tracer pair.

        Example
        -------
        res = get_crosswp(cats, tracers=['LRG', 'ELG', 'QSO'])
        """
        
        if self.args['2PCF_settings']['edges_rppi'] is None:
            if self.args['2PCF_settings']['bin_logscale']:
                r_bins = np.geomspace(self.args['2PCF_settings']['rp_min'], self.args['2PCF_settings']['rp_max'], self.args['2PCF_settings']['n_rp_bins']+1, endpoint=(True))
            else:
                r_bins = np.linspace(self.args['2PCF_settings']['rp_min'], self.args['2PCF_settings']['rp_max'], self.args['2PCF_settings']['n_rp_bins']+1)
            self.args['2PCF_settings']['edges_rppi'] = (r_bins, np.linspace(-self.args['2PCF_settings']['pimax'], self.args['2PCF_settings']['pimax'], 2*self.args['2PCF_settings']['pimax']+1))


        res_dict = {}
        com_tr =np.vstack([np.array(np.meshgrid(tracers,tracers)).T.reshape(-1, len(tracers)).flatten().reshape(len(tracers),len(tracers),2)[i,i:] for i in range(len(tracers))])
        mask_tr = dict(zip(tracers, [cats['TRACER'] == tr for tr in tracers]))
        for tr in com_tr:
            if verbose:
                print('#Compute wp for {}...'.format(tr), flush=True)
                time1 = time.time()

            if self.cosmo is not None:
                if self.args['2PCF_settings']['rsd']:
                    pos1 = apply_rsd(cats[mask_tr[tr[0]]], self.args['hcat']['z_simu'], self.boxsize, self.cosmo, self.H_0, self.args['2PCF_settings']['los'], self.args[tr[0]]['vsmear'])
                    pos2 = apply_rsd(cats[mask_tr[tr[1]]], self.args['hcat']['z_simu'], self.boxsize, self.cosmo, self.H_0, self.args['2PCF_settings']['los'], self.args[tr[1]]['vsmear'])
            else:
                if self.args['2PCF_settings']['rsd']:
                    print('Cosmology not set, does not apply rsd', flush=True)
                pos1 = cats[mask_tr[tr[0]]]['x']%self.boxsize, cats[mask_tr[tr[0]]]['y']%self.boxsize, cats[mask_tr[tr[0]]]['z']%self.boxsize
                pos2 = cats[mask_tr[tr[1]]]['x']%self.boxsize, cats[mask_tr[tr[1]]]['y']%self.boxsize, cats[mask_tr[tr[1]]]['z']%self.boxsize
            
            res_dict[f'{tr[0]}_{tr[1]}'] = compute_wp(pos1, self.args['2PCF_settings']['edges_rppi'], self.boxsize, self.args['2PCF_settings']['pimax'], self.args['2PCF_settings']['los'],  self.args['nthreads'], R1R2=R1R2, pos2=pos2)
            if verbose:
                print('#Done in {:.3f} s'.format(time.time()-time1), flush=True)

        return res_dict
    

    def get_cross2PCF(self, cats, tracers, ells=None, R1R2=None, verbose=True):
        """
        Compute the two-point correlation function (2PCF) multipoles for a given mock catalog in a cubic box.

        This function computes the two-point correlation function (2PCF) and cross-correlations multipoles for pairs of tracers in the mock catalogs. 
        It calculates the 2PCF for all combinations of tracers provided, handling redshift space distortions (RSD) if enabled.

        Parameters
        ----------
        cats : dict
            A dictionary of mock catalogs where each key is a tracer and its corresponding catalog is the value 
            (e.g., 'LRG', 'ELG').
        tracers : list of str
            A list of tracer names (keys in `cats`) for which the cross 2PCF should be computed. The function 
            computes the 2PCF for all pairs of tracers in the list.
        ells : tuple of int, optional
            Multipoles to project onto. If None, ells in `self.args['2PCF_settings']['multipole_index']` are considered. Default to None.
        R1R2 : tuple or None, optional
            A tuple defining a range for R1 and R2 for the 2PCF computation. If None, the default values will be used.
            Defaults to None.
        verbose : bool, optional
            If True, prints progress and computation time for each pair of tracers. Defaults to True.

        Returns
        -------
        res_dict : dict
            A dictionary where the keys are the concatenated names of tracer pairs (e.g., 'LRG_ELG') and the 
            values are the average separations and the corresponding two-point correlation functions (2PCF) for each pair.

        Notes
        -----
        - The function computes the cross-correlation 2PCF for all unique pairs of tracers from the input list.
        - If redshift space distortions (RSD) are enabled, the positions of galaxies in the catalogs are adjusted accordingly.
        - The results are stored in `res_dict` with keys in the format 'tracer1_tracer2', where each value is the 2PCF 
        corresponding to the pair of tracers.
        - The function uses `compute_2PCF` to calculate the 2PCF for each tracer pair.
    

        Example
        -------
        res = get_cross2PCF(cats, tracers=['LRG', 'ELG', 'QSO'])
        """

        
        if self.args['2PCF_settings']['edges_smu'] is None:
            if self.args['2PCF_settings']['bin_logscale']:
                r_bins = np.geomspace(self.args['2PCF_settings']['rmin'], self.args['2PCF_settings']['rmax'], self.args['2PCF_settings']['n_r_bins'])
            else:
                r_bins = np.linspace(self.args['2PCF_settings']['rmin'], self.args['2PCF_settings']['rmax'], self.args['2PCF_settings']['n_r_bins'])
            
            self.args['2PCF_settings']['edges_smu'] = (r_bins, np.linspace(-self.args['2PCF_settings']['mu_max'], self.args['2PCF_settings']['mu_max'], self.args['2PCF_settings']['n_mu_bins']))
        ells = self.args['2PCF_settings']['multipole_index'] if ells is None else ells
        res_dict = {}
        com_tr =np.vstack([np.array(np.meshgrid(tracers,tracers)).T.reshape(-1, len(tracers)).flatten().reshape(len(tracers),len(tracers),2)[i,i:] for i in range(len(tracers))])
        mask_tr = dict(zip(tracers, [cats['TRACER'] == tr for tr in tracers]))
        for tr in com_tr:
            if verbose:
                print('#Compute xi(s,mu) using l={} for {}...'.format(ells, tr), flush=True)
                time1 = time.time()

            if self.cosmo is not None:
                if self.args['2PCF_settings']['rsd']:
                    pos1 = apply_rsd(cats[mask_tr[tr[0]]], self.args['hcat']['z_simu'], self.boxsize, self.cosmo, self.H_0, self.args['2PCF_settings']['los'], self.args[tr[0]]['vsmear'])
                    pos2 = apply_rsd(cats[mask_tr[tr[1]]], self.args['hcat']['z_simu'], self.boxsize, self.cosmo, self.H_0, self.args['2PCF_settings']['los'], self.args[tr[1]]['vsmear'])
            else:
                if self.args['2PCF_settings']['rsd']:
                    print('Cosmology not set, does not apply rsd', flush=True)
                pos1 = cats[mask_tr[tr[0]]]['x']%self.boxsize, cats[mask_tr[tr[0]]]['y']%self.boxsize, cats[mask_tr[tr[0]]]['z']%self.boxsize
                pos2 = cats[mask_tr[tr[1]]]['x']%self.boxsize, cats[mask_tr[tr[1]]]['y']%self.boxsize, cats[mask_tr[tr[1]]]['z']%self.boxsize

            res_dict[f'{tr[0]}_{tr[1]}'] = compute_2PCF(pos1, self.args['2PCF_settings']['edges_smu'], self.boxsize, self.args['2PCF_settings']['multipole_index'],  self.args['2PCF_settings']['los'], self.args['nthreads'], pos2=pos2, R1R2=R1R2)
            if verbose:
                print('#Done in {:.3f} s'.format(time.time()-time1), flush=True)
        return res_dict
    
    
    def HOD_plot(self, tracer=None, fig=None):

        """
        Plot the HOD (Halo Occupation Distribution) for a given tracer or set of tracers.

        This function generates a plot showing the Halo Occupation Distribution (HOD) for specified tracers.
        It uses different colors for each tracer, and can handle multiple tracers at once. If no tracer is
        specified, the function uses the default tracers defined in the arguments.

        Parameters
        ----------
        tracer : str or list of str, optional
            The name(s) of the tracer(s) for which to plot the HOD. If None, it uses the tracers defined in `self.args['tracers']`.
            If a single tracer name is provided, it will be converted into a list.
        fig : matplotlib.figure.Figure, optional
            An existing `matplotlib` figure object to which the plot will be added. If None, a new figure will be created.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The `matplotlib` figure object containing the plotted HOD.

        Notes
        -----
        - The function uses predefined colors for each tracer: 'ELG' (deepskyblue), 'QSO' (seagreen), and 'LRG' (red).
        - The function calls `__init_hod_param` to initialize the HOD parameters and then uses `plot_HOD` to generate the plots.
        - If no `fig` is provided, a new figure is created and returned.
        - The plot is not displayed until `plt.show()` is called, which happens automatically after all tracers are plotted.

        Example
        -------
        HOD_plot(tracer='ELG')   # Plot HOD for the 'ELG' tracer.
        HOD_plot(tracer=['ELG', 'QSO'])  # Plot HOD for both 'ELG' and 'QSO' tracers.
        """
        import matplotlib.pyplot as plt

    
        if tracer is None:
            tracer=self.args['tracers']
        else:
            tracer = tracer if isinstance(tracer, list) else [tracer]
        colors = {'ELG': 'deepskyblue', 'QSO': 'seagreen', 'LRG': 'red'}
        for tr in tracer:
            pc, ps, pab =self.__init_hod_param(tr)
            fig = plot_HOD(pc, ps, self._fun_cHOD[tr], self._fun_sHOD[tr], label=tr, fig=fig, show=False, color=colors[tr] if tr in colors.keys() else None)
        plt.show()


    def plot_HMF(self, cats, show_sat=False, range=(10.8, 15), tracer=None, inital_HMF=None):
        """
        Plot the Halo Mass Function (HMF) for a given mock catalog.

        This function generates a plot of the Halo Mass Function (HMF) using the halo mass values (`log10_Mh`) 
        from the provided catalog(s). The plot can optionally include histograms for central and satellite galaxies,
        and can display an initial HMF for comparison.

        Parameters
        ----------
        cats : dict
            A dictionary of catalog data for different tracers. Each catalog should contain a `log10_Mh` array representing 
            the halo mass in base-10 logarithmic form, and a `Central` array indicating whether the galaxy is central (1) or satellite (0).
        
        show_sat : bool, optional
            Whether to show the histogram for satellite galaxies separately. Default is False.
        
        range : tuple, optional
            The range for the satellite galaxy histogram. Default is (10.8, 15).
        
        tracer : str or list of str, optional
            The tracer(s) for which to plot the HMF. If None, it uses the default tracers defined in `self.args['tracers']`. 
            If a single tracer is provided, it will be converted into a list.
        
        inital_HMF : bool, optional
            Whether to plot the initial Halo Mass Function (HMF) for comparison. Default is None (does not plot the initial HMF).
        
        Returns
        -------
        None
            The function generates and displays the plot but does not return any value.
        
        Notes
        -----
        - The function uses different colors for each tracer: 'ELG' (deepskyblue), 'QSO' (seagreen), and 'LRG' (red).
        - For satellite galaxies, the histograms are plotted with different line styles (`--` for centrals, `:` for satellites).
        - The initial HMF (if provided) is plotted using a gray color.
        - The y-axis is displayed on a logarithmic scale, and the x-axis represents the logarithm of the halo mass in solar masses.

        Example
        -------
        plot_HMF(cats, show_sat=True, range=(10.8, 15), tracer='ELG')  # Plot HMF for the 'ELG' tracer with satellite galaxies.
        plot_HMF(cats, inital_HMF=True)  # Plot HMF with the initial HMF included.
        """
        import matplotlib.lines as mlines
        import matplotlib.pyplot as plt

        colors = {'ELG': 'deepskyblue', 'QSO': 'seagreen', 'LRG': 'red'}
        handles=[]
        
        if tracer is None:
            tracer=self.args['tracers']
        else:
            if not isinstance(tracer, list):
                tracer = [tracer]

        for i, tr in enumerate(tracer):
            mask = cats['TRACER'] == tr
            plt.hist(cats['log10_Mh'][mask], histtype='step', bins=100, color=colors[tr] if tr in colors.keys() else f'C{i}')
            if show_sat:
                plt.hist(cats['log10_Mh'][mask & (cats['Central']==1)], histtype='step', bins=100, range=range, color=colors[tr] if tr in colors.keys() else f'C{i}', ls='--')
                plt.hist(cats['log10_Mh'][mask & (cats['Central']==0)], histtype='step', bins=100, range=range, color=colors[tr] if tr in colors.keys() else f'C{i}', ls=':')
            handles +=[mlines.Line2D([], [], color=colors[tr] if tr in colors.keys() else f'C{i}', label=tr, ls='-')]

        if show_sat:
            handles +=[mlines.Line2D([], [], color='k', label='Centrals', ls='--')]
            handles +=[mlines.Line2D([], [], color='k', label='Satellites', ls=':')]
        if inital_HMF:
            plt.hist(self.hcat['log10_Mh'], histtype='step', bins=100, color='gray')
            handles +=[mlines.Line2D([], [], color='gray', label='inital HMF', ls='-')]

        plt.yscale('log')
        plt.ylabel('$N_{gal}$')
        plt.xlabel('$\log(M_h\ [M_{\odot}])$')
        plt.legend(handles=handles, loc='upper right')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def downsample_mock_cat(cat, ds_fac=0.1, mask=None):
        """
        Downsample a mock catalog by randomly selecting a subset of galaxies based on the given downsampling factor.

        This method randomly selects a subset of galaxies from the input catalog based on the downsampling factor `ds_fac`.
        If a mask is provided, it is used to select galaxies; otherwise, a random selection is made using `ds_fac`.
        
        Parameters
        ----------
        cat : dict
            The mock catalog to downsample. The catalog should be a dictionary containing the galaxy data, 
            such as galaxy positions (`x`, `y`, `z`), and other associated properties.
        
        ds_fac : float, optional
            The downsampling factor, representing the fraction of galaxies to retain in the downsampled catalog.
            A value between 0 and 1. Default is 0.1 (10% of the galaxies will be selected).
        
        mask : array-like, optional
            A boolean mask array that can be used to specify which galaxies to retain in the downsampled catalog.
            If not provided, a random mask is generated based on the `ds_fac` downsampling factor.
        
        Returns
        -------
        dict
            A downsampled catalog containing a subset of the galaxies from the original catalog, based on the mask.
        
        Notes
        -----
        - If `mask` is provided, it should have the same length as the catalog (the number of galaxies).
        - The `ds_fac` parameter defines the probability for each galaxy to be selected; for example, `ds_fac=0.1` means each galaxy has a 10% chance of being selected.
        - The downsampling is done independently for each galaxy.

        Example
        -------
        # Downsample a catalog to 10% of its original size
        downsampled_cat = downsample_mock_cat(cat, ds_fac=0.1)

        # Downsample using a custom mask
        mask = np.array([True, False, True, True, False])
        downsampled_cat = downsample_mock_cat(cat, mask=mask)
        """

        if mask is None:
            mask = np.random.uniform(size=len(cat['x'])) < ds_fac
        return cat[mask]

        
    def compute_training(self, nreal=20, training_points=None, start_point=0, verbose=False):
        
        """
        Generate and save training data for HOD model fitting by sampling parameter sets 
        (training points), generating mock catalogs, and computing clustering statistics.

        This method automates the process of training data generation for halo occupation distribution (HOD) 
        modeling. It evaluates HOD parameter samples, generates mock galaxy catalogs, computes desired clustering 
        statistics (e.g., wp, xi), and stores the results for each training point on disk.

        Parameters
        ----------
        nreal : int, optional
            Number of mock realizations to generate for each training point. Default is 20.
        training_points : structured array or None, optional
            Array of training points with named fields corresponding to HOD parameters. If None, 
            training points will be generated using `genereate_training_points`.
        start_point : int, optional
            Starting index for training point numbering (useful when continuing interrupted runs). Default is 0.
        verbose : bool, optional
            Whether to print progress messages during execution. Default is False.

        Raises
        ------
        ValueError
            If the defined tracers in `self.args` do not match those defined in the priors, 
            or if the number of training parameters doesn't match expectations.

        Notes
        -----
        - The method checks consistency between tracers and prior parameter definitions.
        - For each training point, the HOD parameters are injected into the model and multiple 
        mock realizations are generated.
        - The 2PCF (xi) and/or wp (projected correlation function) are computed per realization, depending 
        on `fit_type`.
        - Each result is saved as a .npy file with a name format based on sampling type and training point index.

        Files Saved
        -----------
        - One `.npy` file per training point is saved to `path_to_training_point` with structure:
            {
                tracer_1: <updated HOD params dict>,
                tracer_2: ...,
                'wp': [...],
                'xi': [...],
                'hod_fit_param': <parameter values used>
            }

        Example
        -------
        self.compute_training(nreal=10, verbose=True)
        
        """

        from .fits_functions import genereate_training_points

        if not set(self.args['tracers']) == set(self.args['fit_param']['priors'].keys()):
            raise ValueError('The defined tracers ({}) does not correspond to tracers defined in the priors({})'.format(self.args['tracers'], self.args['fit_param']['priors'].keys()))
        
        name_param, priors_array = self._get_param_and_prior()

        if len(name_param) != len(training_points.dtype.names):
            raise ValueError('The training sample shape ({}) does not correspond to the number of parameters ({})'.format(len(self.args['fit_param']['priors'][tr]), len(training_points.dtype.names)))
                
        if training_points is None:
            training_points = genereate_training_points(self.args['fit_param']['n_calls'], self.args['fit_param']['priors'], sampling_type='lhs', path_to_save_training_point=self.args['fit_param']['path_to_training_point'], rand_seed=None)
            
        tracers = self.args['tracers']
                
        if verbose:
            print(f'Run training sample', flush=True)

        name_param_tr = {}
        for tr in tracers:
            name_param_tr[tr] = [x.split(f'_{tr}')[0] for x in name_param if tr in x]

        for nb_point, param in enumerate(training_points):
            if not os.path.exists(os.path.join(self.args['fit_param']["path_to_training_point"], '{}_{}.npy'.format(self.args['fit_param']['sampling_type'], nb_point+start_point))):
                start = time.time()     
                result = {}
                for tr in tracers:
                    # var_name_tr = np.array(training_points.dtype.names)[np.array([tr in var for var in training_points.dtype.names])].tolist()
                    idx = np.where([tr in vv for vv in name_param])[0].tolist()
                    tr_par = list(name_param[i] for i in idx)
                    self.args[tr].update(dict(zip(name_param_tr[tr], param[tr_par])))
                    if 'assembly_bias' in self.args['fit_param']['priors'][tr].keys():
                        for var in self.args['fit_param']['priors'][tr]['assembly_bias'].keys():
                            self.args[tr]['assembly_bias'][var] = [param[f'ab_{var}_cen_{tr}'], param[f'ab_{var}_sat_{tr}']]

                    result[tr] = self.args[tr].copy()
                print('Compute HOD:\n',  '\n'.join(['{}:{}'.format(tt,ttt) for tt, ttt in zip(name_param, param)]), flush=True)
                cats = [self.make_mock_cat(tracers, verbose=verbose) for i in range(nreal)]
                
                if 'wp' in self.args['fit_param']["fit_type"]:
                    result['wp']= [self.get_crosswp(cats[i], tracers=tracers, verbose=verbose) for i in range(nreal)]
                if 'xi' in self.args['fit_param']["fit_type"]:
                    result['xi'] = [self.get_cross2PCF(cats[i], tracers=tracers, verbose=verbose) for i in range(nreal)]
                result['hod_fit_param'] = param
                np.save(os.path.join(self.args['fit_param']["path_to_training_point"], '{}_{}.npy'.format(self.args['fit_param']['sampling_type'], nb_point+start_point)), result)
                print('Point {} done {:.2f}'.format(nb_point+start_point, time.time()-start), flush=True)        


    def read_training(self, data, inv_cov2):
        
        """
        Loads and processes HOD training samples, computes chi² statistics for each sample 
        against a target dataset, and returns a structured array for Gaussian Process training.

        Parameters
        ----------
        data : array_like
            Observed data vector (e.g., wp or xi measurements) to compare against model predictions.

        inv_cov2 : ndarray
            Inverse of the covariance matrix used in chi² computation.

        Returns
        -------
        trainning_set : structured ndarray
            Structured array where each row corresponds to a training point, including:
            - HOD parameters
            - Mean chi² value for the realizations
            - Uncertainty on chi² (standard deviation / sqrt(N_real))
        
        Notes
        -----
        - Reads all training `.npy` files from `self.args['fit_param']['path_to_training_point']` with the given sampling type.
        - Applies covariance matrix adjustments if requested.
        - Supports either 'wp', 'xi', or both statistics depending on `self.args['fit_param']['fit_type']`.
        - Combines model realizations by flattening tracer combinations and statistics into a single vector.
        - Computes chi² using the `compute_chi2()` utility, which is assumed to match the data/model shape.

        Example
        -------
        >>> train_set = model.read_training(observed_data, inv_cov2)
        """

        from fits_functions import compute_chi2

        print('Read training sample...', flush=True)
        files = glob.glob(os.path.join(self.args['fit_param']["path_to_training_point"], '{}_*.npy'.format(self.args['fit_param']['sampling_type'])))
        files.sort()
        for ii,file in enumerate(files):
            res_param =  np.load(file, allow_pickle=True)[()]
            if ii == 0:
                name_arr = list(res_param['hod_fit_param'].dtype.names) + ['chi2', 'chi2_err']
                trainning_set = np.zeros((len(files),len(name_arr)))
            stats = ['wp', 'xi'] if ('wp' in self.args['fit_param']["fit_type"]) & ('xi' in self.args['fit_param']["fit_type"]) else ['wp'] if ('wp' in self.args['fit_param']["fit_type"]) else ['xi']
            res = {}
            comb_trs = res_param[stats[0]][0].keys() 
            nreal = len(res_param[stats[0]])
            res = [np.hstack([np.hstack([np.hstack(res_param[stat][i][comb_tr][1])for stat in stats]) for comb_tr in comb_trs]) for i in range(nreal)]

            chi2 = np.mean([compute_chi2(model_arr, data, inv_Cov2=inv_cov2) for model_arr in res])
            chi2_err = np.std([compute_chi2(model_arr, data, inv_Cov2=inv_cov2) for model_arr in res])/np.sqrt(nreal)
            trainning_set[ii] = np.hstack((res_param['hod_fit_param'].tolist(),chi2,chi2_err))

        trainning_set.dtype=[(name, dt) for name, dt in zip(name_arr, ['float64']*len(name_arr))]
        return trainning_set
    
        
    def run_gp_mcmc(self, training_set, niter, logchi2=True,
                        nb_points=1, remove_edges=0.9,
                        random_state=None, verbose=True):
        
        """
        Performs Gaussian Process Regression (GPR) on a training set, runs MCMC sampling over 
        the GPR-predicted posterior, and returns the next suggested parameter point(s) for exploration.

        This function enables Bayesian optimization for halo model fitting by building a GP emulator
        on existing training data, sampling from the GP posterior using MCMC, and identifying 
        the most promising regions in parameter space.
        Detail of the method in arxiv:2302.07056

        Parameters
        ----------
        training_set : structured array
            Training data containing parameters and corresponding chi² values (and uncertainties).

        niter : int
            Current iteration index (used for file naming and logging).

        logchi2 : bool, optional
            If True, the GP models log(chi²). Default is True.

        nb_points : int, optional
            Number of new points to return from the GP+MCMC sampling. Default is 1.

        remove_edges : float, optional
            Factor to shrink prior boundaries when enforcing parameter limits. Values egal to 1 
            keep the prior boundaries. Default is 0.9.

        random_state : int or None, optional
            Seed for reproducibility. Default is None.

        verbose : bool, optional
            If True, print progress and diagnostics. Default is True.

        Returns
        -------
        new_points : ndarray
            Array of shape (nb_points, n_parameters) with newly suggested parameter values.
        

        Notes
        -----
        - Trains a GP model using scikit-learn's `GaussianProcessRegressor`.
        - Runs MCMC sampling using `emcee` or `zeus`. Default sampler is emcee.
        - Logs GPR and MCMC diagnostics to `output_GP_*.txt`.
        - Saves the full sampled chain with GP predictions to `chains/chain_*.txt`.
        - The GP kernel is configured based on `self.args['fit_param']['kernel_gp']`. Default kernel is Matern 5/2.
        - Trained GP model and MCMC output are saved for post-analysis and reproducibility.
        - During MCMC, parameter boundaries are enforced via a likelihood mask.
        - GPR score, prediction at the prior mean, and best predicted chi² are logged.

        Raises
        ------
        ValueError
            If an unsupported GP kernel or sampler is specified.

        Example
        -------
        >>> new_pts = model.run_gp_mcmc(training_data, niter=5, nb_points=3, logchi2=True)

        """

        priors = self.args['fit_param']['priors']
        priors_array = np.vstack([list(priors[tr].values()) for tr in self.args['tracers']])
        nvar = len(priors_array)
        name_param = training_set.dtype.names[:-2]
        ranges = np.hstack((priors_array, np.mean(priors_array, axis=1).reshape(nvar,-1), np.diff(priors_array, axis=1)))

        dir_output_file = self.args['fit_param']['dir_output_fit']
        fit_name = self.args['fit_param']['fit_name']

        arr_training = np.concatenate(training_set.tolist(), axis=0).T
        
        os.makedirs(dir_output_file, exist_ok=True)
        if logchi2:
            X_train, Y_train, Y_err = arr_training[:nvar].T, np.log(arr_training[-2]), arr_training[-1]/arr_training[-2]  
        else:
            X_train, Y_train, Y_err = arr_training[:nvar].T, arr_training[-2], arr_training[-1]

        length_scale = np.ones(nvar)
        if self.args['fit_param']['length_scale_bounds'] == "fix":
            length_scale = length_scale

        if  self.args['fit_param']['kernel_gp'] == 'RBF':
            kernel = 1.0 * skg.kernels.RBF(length_scale=length_scale,
                                            length_scale_bounds=self.args['fit_param']['length_scale_bounds'])
        elif self.args['fit_param']['kernel_gp'] == 'Matern_52':
            kernel = 1.0 * skg.kernels.Matern(length_scale=length_scale,
                                                length_scale_bounds=self.args['fit_param']['length_scale_bounds'], nu=5/2)
            
        else:
            raise ValueError('Only RBF or Matern_52 Kernel are available not {}'.format(self.args['fit_param']['kernel_gp']))
            
        if verbose:
            print(f"Running GPR iteration {niter}...", flush=True)
            start = time.time()

        gp = skg.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
                                            alpha=Y_err**2, random_state=random_state).fit(X_train, Y_train)
        if verbose:
            print(
                f"GPR computed took {time.strftime('%H:%M:%S',time.gmtime(time.time() - start))}")
            print('#score=', gp.score(X_train, Y_train), flush=True)
            print("#", gp.kernel_.get_params(), flush=True)

        def likelihood(x):
            if logchi2:
                L = -np.exp(gp.predict(x.reshape(-1, nvar)))/2  
            else:
                L = -gp.predict(x.reshape(-1, nvar))/2
            # print(L,x)
            cond = np.abs(x-ranges[:, 2]) < (ranges[:, 3]/2)*remove_edges
            # print(cond)
            if cond.all():
                return L
            else:
                return -np.inf

        p0 = np.random.uniform(0, 1, (self.args['fit_param']['nwalkers'], nvar))
        for k in range(nvar):
            p0[:, k] = (p0[:, k]-0.5)*ranges[k, 3] * 0.8 + ranges[k, 2]  # 0.8 edges
        
        if verbose:
            print(f"Run MCMC for iteration {niter}...", flush=True)
            start = time.time()
        if self.args['fit_param']['sampler'] == "zeus":
            sampler_mcmc = zeus.EnsembleSampler(
                self.args['fit_param']['nwalkers'], nvar, likelihood)  # , args=[nvar])
            sampler_mcmc.run_mcmc(p0, self.args['fit_param']['n_iter'])  # per walker
            chain = sampler_mcmc.get_chain(flat=True)
        elif self.args['fit_param']['sampler'] == "emcee":
            sampler_mcmc = emcee.EnsembleSampler(
                self.args['fit_param']['nwalkers'], nvar, likelihood)  # , args=[nvar])
            sampler_mcmc.run_mcmc(p0, self.args['fit_param']['n_iter'])  # per walker
            '''with Pool() as pool:
                sampler_mcmc = emcee.EnsembleSampler(self.args['fit_param']['nwalkers'], nvar, likelihood, pool=pool)
                sampler_mcmc.run_mcmc(p0, self.args['fit_param']['n_iter'])'''
            chain = sampler_mcmc.flatchain
        else: 
            raise ValueError('Only emcee or zeus sampler are available not {}'.format(self.args['fit_param']['sampler']))
        
        trimmed = chain[int(len(chain)/4):]
        if verbose:
            print(f'MCMC computed took {time.strftime("%H:%M:%S",time.gmtime(time.time() - start))}', flush=True)
        
        new_points = trimmed[:, :nvar][np.random.randint(len(trimmed), size=nb_points)]


        Gp_pred = gp.predict(trimmed, return_std=True)
        ind = np.where(Gp_pred[0] == Gp_pred[0].min())[0][0]
        pred = gp.predict(ranges[:, 2].reshape(-1, nvar),
                            return_std=True)  # Best fit pred
        if verbose:
            print("#Pred at fix point (mean values of each params):",
                    ranges[:, 2], pred, flush=True)
            print("#best GP prediction:",
                    trimmed[ind], Gp_pred[0][ind], Gp_pred[1][ind], flush=True)

        multi_GR = multivariate_gelman_rubin(sampler_mcmc.get_chain().transpose([1, 0, 2])[:, 2500:, :])
        if verbose:
            print("#multivariate_gelman_rubin: ", multi_GR, flush=True)

        res = np.hstack((trimmed, np.exp(Gp_pred[0]).reshape(len(Gp_pred[0]),1) if logchi2 else Gp_pred[0].reshape(len(Gp_pred[0]),1), (np.exp(Gp_pred[0])*Gp_pred[1]).reshape(len(Gp_pred[0]),1) if logchi2 else Gp_pred[1].reshape(len(Gp_pred[0]),1)))
        
        os.makedirs(os.path.join(dir_output_file, 'chains'), exist_ok=True)
        np.savetxt(os.path.join(dir_output_file, 'chains', f'chain_{nvar}p_{fit_name}_{niter}.txt'), res)

        if niter == 0:
            list_lenghtscale = []
            for i in range(nvar):
                list_lenghtscale.append("ls%d" % i)
            
            f = open(os.path.join(dir_output_file,
                                    f'output_GP_{nvar}p_{fit_name}.txt'), "w")
            
            f.write("N_iter GPscore GP_predfix Er_predfix BestPred Er_BestPred multivariate_gelman_rubin "
                    + ' '.join(map(str, name_param))+" "
                    + ' '.join(map(str, list_lenghtscale))+"\n")
            f.write(str(niter)+" "+str(gp.score(X_train, Y_train))+" "
                    + str(pred[0][0])+" "+str(pred[1][0])+" "
                    + str(Gp_pred[0][ind])+" "
                    + str(Gp_pred[1][ind])+" " + str(multi_GR)+" "
                    + ' '.join(map(str, trimmed[ind]))+" "
                    + ' '.join(map(str, gp.kernel_.get_params(False)["k2"].length_scale))+"\n")
            f.close()
        else:
            f = open(os.path.join(dir_output_file,
                                    f'output_GP_{nvar}p_{fit_name}.txt'), "a")
            
            f.write(str(niter)+" "+str(gp.score(X_train, Y_train))+" "
                    + str(pred[0][0])+" "+str(pred[1][0])+" "
                    + str(Gp_pred[0][ind])+" "
                    + str(Gp_pred[1][ind])+" " + str(multi_GR)+" "
                    + ' '.join(map(str, trimmed[ind]))+" "
                    + ' '.join(map(str, gp.kernel_.get_params(False)["k2"].length_scale))+"\n")
            f.close()

        return new_points
    

    def run_fit(self, data_arr, inv_Cov2, training_point,
                resume_fit=False, verbose=True):
        
        """
        Execute the Gaussian Process MCMC fitting routine for HOD parameter inference.

        This method performs iterative Gaussian Process-driven MCMC sampling to explore the
        Halo Occupation Distribution (HOD) parameter space, fitting mock catalog outputs to observed
        clustering statistics such as the 2-point correlation function.

        For methodological details, see: https://arxiv.org/abs/2302.07056

        Parameters
        ----------
        data_arr : array_like
            Observed data vector used in chi-squared comparisons (e.g., wp, xi).

        inv_Cov2 : ndarray
            Inverse of the covariance matrix used in the chi-squared computation.
            Must match the dimensionality of `data_arr`.

        training_point : structured ndarray
            Existing training sample including HOD parameters and chi-squared values,
            used to condition the GP model.

        resume_fit : bool, optional
            If True, resumes from a previously saved fit by loading logs and chains.
            Default is False.

        verbose : bool, optional
            If True, displays detailed iteration-level logs. Default is True.

        Returns
        -------
        None
            All fitting results are saved to disk. No return value.

        Notes
        -----
        - Creates and updates files under `dir_output_fit`, including:
            - `*.txt` logs of sampled parameter values and chi² results
            - Chains of samples in `chains/` directory
            - Diagnostic metrics such as KL divergence
        - Calls the following key internal methods:
            - `make_mock_cat()`: to generate mock catalogs
            - `get_crosswp()`, `get_cross2PCF()`: for 2PCF computation
            - `compute_chi2()`: to evaluate model-data fit
            - `run_gp_mcmc()`: for parameter sampling via GP-MCMC
        - Convergence is optionally monitored via KL divergence, but the stopping criterion is commented out.
        - Handles both projected (wp) and full-space (xi) correlation functions depending on `fit_type`.
        - Assumes the availability of `emcee` or `zeus` samplers for MCMC.
        - Results are appended to an evolving training set across iterations.

        Example
        -------
        >>> model.run_fit(data_arr, inv_cov2, training_set, resume_fit=True)
        """
        import pandas as pd
        from .fits_functions import compute_chi2
        if self.args['fit_param']['sampler'] == "zeus":
            import zeus
        elif self.args['fit_param']['sampler'] == "emcee":
            import emcee 
        else: 
            raise ValueError('Only emcee or zeus sampler are available not {}'.format(self.args['fit_param']['sampler']))
        
        import sklearn.gaussian_process as skg

        nmock = self.args['fit_param']['nb_real']
        dir_output_file= self.args['fit_param']['dir_output_fit']
        fit_name = self.args['fit_param']['fit_name']
        priors = self.args['fit_param']['priors']
        priors_array = np.vstack([list(priors[tr].values()) for tr in self.args['tracers']])
        nvar = len(priors_array)  
        arr_dtype = training_point.dtype

        iter = 0
        if resume_fit & os.path.exists(os.path.join(dir_output_file, f"{nvar}p_{fit_name}.txt")):
            output_point = pd.read_csv(os.path.join(
                dir_output_file, f"{nvar}p_{fit_name}.txt"), sep=" ", comment="#")
            
            training_point = np.concatenate((np.array(training_point.tolist()).reshape(len(training_point), -1), output_point[list(training_point.dtype.names)].values))
            training_point.dtype = arr_dtype

            iter = output_point["N_iter"].loc[len(output_point)-1]+1
            p = np.loadtxt(os.path.join(dir_output_file, 'chains',
                                        f'chain_{nvar}p_{fit_name}_{iter-1}.txt'))[:, :nvar]
            D_kl = 10
            if verbose:
                print("#resume fit at iteration ", iter, "len param point ", len(training_point), flush=True)
                
        print("Run gpmcmc...", flush=True)
        for j in range(iter, self.args['fit_param']['n_calls']):
            if verbose:
                print(f'Iteration {j}...', flush=True)
                time_compute_mcmc = time.time()

            new_params = self.run_gp_mcmc(training_point, j, logchi2=self.args['fit_param']['logchi2'],
                        nb_points=1, remove_edges=0.9,
                        random_state=None, verbose=True)


            if verbose:
                print("#time_compute_gpmcmc =", time.time() - time_compute_mcmc, flush=True)

            ### Test de Kullback Leibler
            D_kl1 = 10
            if j > 0:
                if j == 1:
                    q = np.loadtxt(os.path.join(dir_output_file, 'chains', f'chain_{nvar}p_{fit_name}_{0}.txt'))[:, :nvar]
                    p = np.loadtxt(os.path.join(dir_output_file, 'chains', f'chain_{nvar}p_{fit_name}_{1}.txt'))[:, :nvar]
                    D_kl = np.array([])
                else:
                    q = p
                    p = np.loadtxt(os.path.join(dir_output_file, 'chains', f'chain_{nvar}p_{fit_name}_{j}.txt'))[:, :nvar]
                n_dim = nvar
                cov_q = np.cov(q.T)
                cov_p = np.cov(p.T)
                inv_cov_q = np.linalg.inv(cov_q)
                mean_q = np.mean(q, axis=0)
                mean_p = np.mean(p, axis=0)
                D_kl1 = 0.5 * (np.log10(np.linalg.det(cov_q) / np.linalg.det(cov_p)) - n_dim + np.trace(np.matmul(
                    inv_cov_q, cov_p)) + np.matmul((mean_q - mean_p).T, np.matmul(inv_cov_q, (mean_q - mean_p))))
                D_kl = np.append(D_kl1, D_kl)
                # print (j, D_kl)
                # if len(D_kl) > 5:
                #     if (D_kl[-5:] < 0.1).all():
                #         sys.exit("Procedure converged at iteration %d!" % j)

            #Compute chi2
            new_train_point = np.zeros((len(new_params), nvar+2))

            new_params.dtype = [(name, dt) for name, dt in zip(training_point.dtype.names, ['float64']*nvar)]

            if verbose:
                print("#run old parralel chi2 points", new_params, len(new_params))

            if j == 0:
                f = open(os.path.join(dir_output_file, f"{nvar}p_{fit_name}.txt"), "w")
                f.write("N_iter "+' '.join(map(str, training_point.dtype.names)) + " D_kl1\n")
                f.close()

            for i, new_p in enumerate(new_params):
                for tr in self.args['tracers']:
                    for var in self.args['fit_param']['priors'][tr].keys():
                        self.args[tr][var] = new_p['{}_{}'.format(var, tr)][0]
                    if verbose:
                        print(f"# {tr} {[(var, self.args[tr][var]) for var in self.args['fit_param']['priors'][tr].keys()]}", flush=True)
                
                time_function_compute_parralel_chi2 = time.time()
                print(f'Run {nmock} galaxy catalog for iteration {j}', flush=True)
                cats = [self.make_mock_cat(self.args['tracers'], verbose=False) for jj in range(nmock)]

                print(f'Time to compute {nmock} cats : {time.strftime("%H:%M:%S",time.gmtime(time.time() - time_function_compute_parralel_chi2))}', flush=True)


                time_function_compute_parralel_chi2 = time.time()
                print('Run 2PCF...', flush=True)
                result = {}
                if 'wp' in self.args['fit_param']["fit_type"]:
                    result['wp']= [self.get_crosswp(cats[i], tracers=self.args['tracers'], verbose=False) for i in range(nmock)]
                if 'xi' in self.args['fit_param']["fit_type"]:
                    result['xi'] = [self.get_cross2PCF(cats[i], tracers=self.args['tracers'], verbose=False) for i in range(nmock)]
                if verbose:
                    print("#Time to compute 2PCFs =", time.strftime("%H:%M:%S", time.gmtime(time.time()-time_function_compute_parralel_chi2)), flush=True)
                    

                stats = ['wp', 'xi'] if ('wp' in self.args['fit_param']["fit_type"]) & ('xi' in self.args['fit_param']["fit_type"]) else ['wp'] if ('wp' in self.args['fit_param']["fit_type"]) else ['xi']
                res = {}
                comb_trs = result[stats[0]][0].keys() 
                res = [np.hstack([np.hstack([np.hstack(result[stat][i][comb_tr][1])for stat in stats]) for comb_tr in comb_trs]) for i in range(nmock)]
                #res_std = np.std(res, axis=0)            
                #iCov2 = inv_Cov2*res_std*res_std[:, None]
                

                chi2 = np.mean([compute_chi2(model_arr, data_arr, inv_Cov2=inv_Cov2) for model_arr in res])
                chi2_err = np.std([compute_chi2(model_arr, data_arr, inv_Cov2=inv_Cov2) for model_arr in res])/np.sqrt(nmock)

                new_train_point[i] = np.hstack((new_params[i].tolist()[0], chi2, chi2_err))
                if verbose:
                    print('#### NEW ADDED POINT:', ' '.join(map(str, new_params[i].tolist()[0])), chi2, chi2_err, D_kl1, flush=True)

                f = open(os.path.join(dir_output_file, f"{nvar}p_{fit_name}.txt"), "a")
                f.write(str(str(j)+" "+' '.join(map(str, new_params[i].tolist()[0])))+" "+str(chi2)+" "+str(chi2_err)+" "+str(D_kl1)+"\n")
                f.close()

            new_train_point.dtype = arr_dtype
            training_point = np.vstack((training_point, new_train_point))
            if verbose:
                print(f'Iteration {j} done, took {time.strftime("%H:%M:%S",time.gmtime(time.time()-time_compute_mcmc))}', flush=True)
