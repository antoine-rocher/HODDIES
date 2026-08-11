import numpy as np 
import numba
import logging
import sys
import time
            

def setup_logging(level=logging.INFO, stream=sys.stdout, filename=None, filemode='w'):
    """
    Activate logging with an elapsed-time and date prefix, e.g.::

        [000000.05]  05-02 06:57 AbacusSummitSim              INFO     message

    The leading ``[...]`` is the number of seconds elapsed since this call, and
    ``05-02 06:57`` is the wall-clock date/time (``%m-%d %H:%M``). Call this once
    (typically at the top of a script or notebook) so the loggers created in
    :meth:`Base_catalogue.init_logger` actually emit their messages.

    Parameters
    ----------
    level : int or str, default=logging.INFO
        Logging level, either a ``logging`` constant or one of
        ``('info', 'debug', 'warning', 'error')``.
    stream : file-like, default=sys.stdout
        Stream to write log records to (ignored when ``filename`` is given).    
    filename : str, default=None
        If provided, write log records to this file instead of ``stream``.
    filemode : str, default='w'
        Mode used to open ``filename``.
    """
    levels = {'info': logging.INFO, 'debug': logging.DEBUG,
              'warning': logging.WARNING, 'error': logging.ERROR}
    if isinstance(level, str):
        level = levels[level.lower()]

    # Remove existing root handlers so repeated calls do not duplicate output.
    for handler in list(logging.root.handlers):
        logging.root.removeHandler(handler)

    t0 = time.time()

    class _ElapsedFormatter(logging.Formatter):
        def format(self, record):
            self._style._fmt = ('[%09.2f] ' % (time.time() - t0)
                                 + ' %(asctime)s %(name)-28s %(levelname)-8s %(message)s')
            return super().format(record)

    fmt = _ElapsedFormatter(datefmt='%m-%d %H:%M')

    if filename is not None:
        handler = logging.FileHandler(filename, mode=filemode)
    else:
        handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(fmt)

    logging.root.addHandler(handler)
    logging.root.setLevel(level)

    # Third-party libraries are extremely chatty below INFO (numba dumps the bytecode
    # of every jitted function), which would bury the HODDIES records. Keep them at
    # INFO so that level='debug' only adds HODDIES messages.
    if level < logging.INFO:
        for name in ['numba', 'matplotlib', 'PIL', 'h5py', 'asyncio', 'fsspec', 'jax']:
            logging.getLogger(name).setLevel(logging.INFO)


class BaseLogger:
    """
    Mixin providing a :mod:`logging` logger named after the concrete class.

    Any class calling ``self.init_logger()`` in its ``__init__`` gets a ``self.logger``.
    """

    def init_logger(self, setup_logger=True, name=None):
        """
        Initialize the logger for the class.

        Logging is activated with :func:`setup_logging` if it has not been configured
        yet, so messages are visible by default. Call :func:`setup_logging` explicitly
        (e.g. at the top of a script or notebook) to choose the level, or to redirect
        records to a file.
        """
        if not logging.root.handlers and setup_logger:
            setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__ if name is None else name)
        self.logger.info(f'Initializing {self.__class__.__name__}.')


class Base_catalogue(BaseLogger):
    """
        Initialize the catalogue class for the simulation data. This class is designed to handle the loading and management of N-body simulations. It provides methods to initialize the catalogue with the necessary parameters, check for required columns, and access properties such as box size and columns of the simulation data.
    """

    def __init__(self, halo_data, particle_data=None, **kwargs):
        #TBD
        # self.boxsize = kwargs.get('boxsize', None)
        self.init_logger(kwargs.get('setup_logger', True))
        self.init_params(**kwargs)
        
        self.init_cat(halo_data, particle_data, **kwargs)
        self.check_hcat_cols()
        self.check_hcat_args()

    def init_params(self, **kwargs):
        """
        Initialize the parameters for the simulation catalogue.
        This method sets the box size, mass cut, and number of threads for the simulation catalogue 
        based on the provided keyword arguments. If the parameters are not provided, default values are used.
        """
        
        self.boxsize = kwargs.get('boxsize', None)
        self.mass_cut = kwargs.get('mass_cut', None)
        self.nthreads = min(numba.get_num_threads()-1, kwargs.get('nthreads', 32))
        numba.set_num_threads(self.nthreads)
        self._init_halo_cols=['x', 'y', 'z', 'vx', 'vy', 'vz','Mh', 'Rh', 'c', 'Vrms', 'halo_id']
        self._init_part_cols = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'halo_id']
        self.z_simu = kwargs.get('z_simu', None)
        

    
    def init_cat(self, halo_data, particle_data=None, mapping_halo_cols : dict = None, mapping_part_cols : dict = None):
        """
        Initialize the halo and particle catalogues with the provided data.
        
        Parameters
        ----------
        halo_data : dict or structured array
            The halo catalog data, which can be in the form of a dictionary or a structured array. The keys or field names should correspond to the halo properties (e.g., 'x', 'y', 'z', 'vx', 'vy', 'vz', 'Mh', 'Rh', 'c', 'Vrms', 'halo_id').
        particle_data : dict or structured array, optional
            The particle catalog data, which can be in the form of a dictionary or a structured array. The keys or field names should correspond to the particle properties (e.g., 'x', 'y', 'z', 'vx', 'vy', 'vz', 'halo_id'). If not provided, the particle catalog will be set to None.
        mapping_halo_cols : dict, optional
            A dictionary mapping the column names in the halo_data to the expected column names in the catalogue. If not provided, the default mapping will be used (i.e., the keys in halo_data should match the expected column names).
        mapping_part_cols : dict, optional
            A dictionary mapping the column names in the particle_data to the expected column names in the catalogue. If not provided, the default mapping will be used (i.e., the keys in particle_data should match the expected column names).
        Raises
        ------
        TypeError
            If halo_data or particle_data is not in a format that supports __getitem__ (e.g., dict, structured ndarray, astropy table, pandas DataFrame).
        ValueError
            If mapping_halo_cols or mapping_part_cols is missing required columns or if the provided column names do not match the expected column names in the catalogue.
        
        """
        from mpytools import Catalog
        if mapping_halo_cols is None:
            mapping_halo_cols = {col: col for col in self._init_halo_cols}
        else:
            if not isinstance(mapping_halo_cols, dict):
                raise TypeError(f'mapping_halo_cols must be a dictionary, got {type(mapping_halo_cols)}')
            if not all(col in mapping_halo_cols for col in self._init_halo_cols):
                missing = [col for col in self._init_halo_cols if col not in mapping_halo_cols]
                raise ValueError(f'mapping_halo_cols is missing required columns: {missing}. Required columns are: {self._init_halo_cols}. Current mapping_halo_cols keys are: {list(mapping_halo_cols.keys())}')
        if mapping_part_cols is None:
            mapping_part_cols = {col: col for col in self._init_part_cols}
        else:
            if not isinstance(mapping_part_cols, dict):
                raise TypeError(f'mapping_part_cols must be a dictionary, got {type(mapping_part_cols)}')
            if not all(col in mapping_part_cols for col in self._init_part_cols):
                missing = [col for col in self._init_halo_cols if col not in mapping_halo_cols]
                raise ValueError(f'mapping_halo_cols is missing required columns: {missing}. Required columns are: {self._init_halo_cols}. Current mapping_halo_cols keys are: {list(mapping_halo_cols.keys())}')

        if hasattr(halo_data, '__getitem__'):
            all_cols = np.unique(list(mapping_halo_cols.keys())+ self._init_halo_cols)
            self.hcat = Catalog()
            self.hcat.data = {col: Catalog.from_dict(halo_data)[mapping_halo_cols[col]] for col in all_cols}
        else:
            raise TypeError(f'Invalid type for halo catalog: {type(halo_data)}. halo_data should be in a format that supports __getitem__ (e.g., dict, structured ndarray, astropy table, pandas DataFrame).')

        
        if particle_data is None:
            self.part_subsamples = None
        else:
            if hasattr(particle_data, '__getitem__'):
                self.part_subsamples = Catalog()
                self.part_subsamples.data = {col: Catalog.from_dict(particle_data)[mapping_part_cols[col]] for col in self._init_part_cols}
            else:
                raise TypeError(f'Invalid type for particle catalog: {type(particle_data)}. particle_data should be in a format that supports __getitem__ (e.g., dict, structured ndarray, astropy table, pandas DataFrame).')
            
        

    def cosmo(self, **cosmo_params):
        """
        Return the cosmology object associated with the simulation.
        Parameters
        ----------
        cosmo_params : dict
            Parameters from cosmoprimo to customize the cosmology initialization.
        Returns
        -------
        cosmo : Cosmology object from cosmprimo
        """
        if not all(cosmo_params.values()):
            self.logger.info('Cosmology parameters not provided. Cosmology needed to apply RSD when computing correlations. No cosmology set.')
            return None
        try: 
            from cosmoprimo.fiducial import Cosmology
            self.logger.info('Initialize custom cosmology from the "cosmo" parameters')
            
            return Cosmology(**{k: v for k, v in cosmo_params.items() if v is not None})
            
        except ImportError:
            import warnings
            warnings.warn('Could not import cosmoprimo. Install cosmoprimo with "python -m pip install git+https://github.com/cosmodesi/cosmoprimo[class,camb,extras]".\n'\
                  'Cosmology needed to apply RSD when computing correlations. No cosmology set.')
        
            return None
    
    @property
    def columns(self):
        """
        Return the columns of the simulation.
        """
        return self.hcat.columns()


    def check_hcat_cols(self):
        """
        Check if the catalogue has the required columns.
        """

        if not np.all([col in self.columns for col in self._init_halo_cols]):
            missing_cols = [col for col in self._init_halo_cols if col not in self.columns]
            raise ValueError('Missing columns {} in the halo catalogue. Required columns are: {}'.format(missing_cols, self._init_halo_cols), 'Current columns are: {}'.format(self.columns))
        
        if self.part_subsamples is not None:
            if not np.all([col in self.part_subsamples.columns() for col in self._init_part_cols]):
                missing_cols = [col for col in self._init_part_cols if col not in self.part_subsamples.columns()]
                raise ValueError('Missing columns {} in the particle catalogue. Current columns are: {}. Required columns are: {}'.format(missing_cols, self.part_subsamples.columns(), self._init_part_cols))
        if 'log10_Mh' not in self.columns: 
            self.hcat['log10_Mh'] = np.log10(self.hcat['Mh'])
    
    def check_hcat_args(self):
        """
        Check if the catalogue has the required arguments.
        """

        if self.boxsize is None:
            raise ValueError('Box size is not provided.')


    def load_env_based_properties(self, **kwargs):
        """
        Placeholder method to load environment-based properties (density and shear) from simulation data. This method should be implemented in subclasses for specific simulations (e.g., Abacus, Uchuu). It is intended to compute or load precomputed density and shear meshes based on particle distributions.
        """        
        self.logger.info('Function to load environment-based properties (env and shear) is not defined. Please implement this method in the subclass for specific simulations.')
        self.density_mesh = kwargs.get('density_mesh', None)
        self.shear_mesh = kwargs.get('shear_mesh', None)
        
        

    def get_env_col(self):
        """
        Compute the environment and shear columns for the halo catalog based on the density and shear meshes. This method uses interpolation to assign values from the density and shear meshes to each halo in the catalog based on their positions. The positions are converted to grid index space, and periodic wrapping is applied to ensure that halos outside the box boundaries are correctly handled.
        """

        from scipy.interpolate import interpn

        N_dim = self.density_mesh.shape[0]
        cell_size = self.boxsize / N_dim

        # Halo positions in grid index space (apply periodic wrapping)
        GroupPos = (
            np.array([self.hcat['x'], self.hcat['y'], self.hcat['z']]).T / cell_size
        ).astype(int) % N_dim

        grid_axes = (np.arange(N_dim), np.arange(N_dim), np.arange(N_dim))

        for name, mesh in [('env', self.density_mesh), ('shear', self.shear_mesh)]:
            
            self.logger.info(f"Initializing {name} column...")
            self.hcat[name] = interpn(grid_axes, mesh, GroupPos)
            self.logger.info("Done!")

    def set_assembly_bias_values(self, columns, bins=50):

        """
        Assign ranked values for assembly bias computation based on a specific column.

        This method assigns ranked values linearly between -0.5 and 0.5 for the assembly bias computation. The
        values are based on a histogram of halo masses (`log10_Mh`), and the `columns` parameter specifies the columns
        in the halo catalog that will be used for the ranking. The method first checks if the required column 
        (`ab_{col}`) already exists; if not, it computes the values based on the input column and adds them to the
        halo catalog.

        Parameters
        ----------
        columns : list of str
            The column names in the halo catalog used to compute assembly bias values. 
            These columns are expected to be present in the halo catalog.
        bins : int, optional
            The number of bins used for mass binning in the histogram of halo masses. Default is 50.

        Returns
        -------
        list of str
            The column names that were not available in the halo catalog.

        Warnings
        ------
        ValueError
            If the specified column is not present in the halo catalog, a `ValueError` warning is raised.

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
        from HODDIES.utils import initialize_assembly_bias_value

        col_to_remove = []
        for col in columns:
            if f'ab_{col}' in self.columns:
                continue  # Skip if assembly bias column already exists
            
            if ((col == 'env') & ('env' not in self.columns)) | ((col == 'shear') & ('shear' not in self.columns)):
                self.load_env_based_properties()
                if self.density_mesh is None or self.shear_mesh is None:
                    self.logger.info(f"Continue without assembly bias for {col}.")
                    col_to_remove.append(col)   
                    continue
                self.get_env_col()
        
            if col not in self.columns:
                import warnings
                warnings.warn(ValueError(f'Column {col} is not in halo catalog columns. Cannot compute assembly bias for this column.'))
                col_to_remove.append(col)   
                continue

            self.logger.info(f'Set value for assembly bias according to {col}...')
            self.hcat[f'ab_{col}'] = initialize_assembly_bias_value(self.hcat['log10_Mh'], self.hcat[col], bins=bins)
            self.logger.info(f'Done !')
        return col_to_remove


    def assign_sat_to_part(self, mask_sat, list_nsat, seed=None):
        """
        Default method to assign satellites to particles.
        This method assigns satellite galaxies to particles based on the provided satellite catalog (`sat_cat`) and the number of satellites per halo (`list_nsat`). It uses the `halo_to_particle_indices` and `sample_satellites_from_particles` functions to perform the assignment. The method returns a mask indicating which satellites will be positioned using the NFW profile.

        Parameters
        ----------
        sat_cat : array-like
            The satellite catalog containing the properties of satellite galaxies, including their positions and velocities.
        list_nsat : array-like
            The number of satellites per halo, used to determine how many satellites to assign to each halo.
        seed : list[int], optional
            A list of random seeds for reproducibility. If provided, it should have the same length as `sat_cat`. If not provided, a default seed is used.
        
        Returns
        -------
        x_sat, y_sat, z_sat : array-like
            The assigned positions of the satellite galaxies in the x, y, and z coordinates.
        vx_sat, vy_sat, vz_sat : array-like
            The assigned velocities of the satellite galaxies in the vx, vy, and vz components.
        mask_nfw : array-like
            A boolean mask indicating which satellites will be positioned using the NFW profile. True values correspond to satellites that will be assigned to particles, while False values correspond to satellites that will not be assigned to particles.
        Notes
        -----
        - Satellite catalog (`sat_cat`) is modified in place to assign positions and velocities from particles.
        - The method assumes that the satellite catalog (`sat_cat`) contains a column named `'halo_id'` that identifies the host halo for each satellite.
        - The method also assumes that the particle catalog (`self.part_subsamples`) contains a column named `'halo_id'` that identifies the host halo for each particle.
        - The method uses the `halo_to_particle_indices` function to map halo IDs to particle indices, and the `sample_satellites_from_particles` function to sample satellites from the particles based on their positions and velocities.
        - The method sorts the satellite catalog by halo ID before performing the assignment to ensure that satellites are assigned to the correct host halos.
        - The method uses the `self.nthreads` attribute to determine the number of threads to use for parallel processing when sampling satellites from particles.
        - The method returns a boolean mask (`mask_nfw`) indicating which satellites will be positioned using the NFW profile, allowing for further analysis or processing of the assigned satellites.

        """

        from HODDIES.utils import halo_to_particle_indices, sample_satellites_from_particles        
    
        uniq_sat_id = self.hcat['halo_id'][mask_sat]
        sort_index = np.argsort(uniq_sat_id)
        unsort_index = np.argsort(sort_index)
        if not hasattr(self, '_order_part_index'):
            self._order_part_index = np.argsort(self.part_subsamples['halo_id'])
        flat, offsets = halo_to_particle_indices(self.part_subsamples['halo_id'][self._order_part_index], self._order_part_index, uniq_sat_id, self.nthreads)
        # Sample satellites from particles, outputs are sorted according to the halo_id
        x_sat, y_sat, z_sat, vx_sat, vy_sat, vz_sat, mask_nfw = sample_satellites_from_particles(self.part_subsamples['x'], self.part_subsamples['y'], self.part_subsamples['z'],
                                            self.part_subsamples['vx'], self.part_subsamples['vy'], self.part_subsamples['vz'], flat, offsets, list_nsat, self.nthreads, seed=seed)             
        return x_sat[unsort_index], y_sat[unsort_index], z_sat[unsort_index], vx_sat[unsort_index], vy_sat[unsort_index], vz_sat[unsort_index], mask_nfw[unsort_index]
