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
        if not setup_logger:
            self.logger = logging.getLogger(self.__class__.__name__ if name is None else name)
            self.logger.addHandler(logging.NullHandler())
            self.logger.propagate = False
            return

        if not logging.root.handlers:
            setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__ if name is None else name)
        self.logger.info(f'Initializing {self.__class__.__name__}.')

import numpy as np
import numba
import logging
import sys
import time

PACKAGE_LOGGER = 'hoddies'   # root of this package's logger namespace


def setup_logging(level=logging.INFO, stream=sys.stdout, filename=None, filemode='w'):
    """
    Activate logging with an elapsed-time and date prefix, e.g.::

        [000000.05]  05-02 06:57 AbacusSummitSim              INFO     message

    The leading ``[...]`` is the number of seconds elapsed since this call, and
    ``05-02 06:57`` is the wall-clock date/time (``%m-%d %H:%M``). Call this once
    (typically at the top of a script or notebook) so the loggers created in
    :meth:`Base_catalogue.init_logger` actually emit their messages.

    The handler is attached to the ``hoddies`` logger rather than the root logger,
    so third-party libraries (pycorr, numba, matplotlib, ...) are not made verbose
    as a side effect. To see their records as well, configure the root logger
    separately, e.g. ``logging.basicConfig(level=logging.INFO)``.

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

    logger = logging.getLogger(PACKAGE_LOGGER)

    # Remove existing handlers so repeated calls do not duplicate output.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    t0 = time.time()

    class _ElapsedFormatter(logging.Formatter):
        def format(self, record):
            # Strip the package prefix so records read 'AbacusSummitSim',
            # not 'hoddies.AbacusSummitSim'.
            record.shortname = record.name.split('.', 1)[-1]
            self._style._fmt = ('[%09.2f] ' % (time.time() - t0)
                                + ' %(asctime)s %(shortname)-28s %(levelname)-8s %(message)s')
            return super().format(record)

    fmt = _ElapsedFormatter(datefmt='%m-%d %H:%M')

    if filename is not None:
        handler = logging.FileHandler(filename, mode=filemode)
    else:
        handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(fmt)

    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False   # do not hand records to the root logger


class BaseLogger:
    """
    Mixin providing a :mod:`logging` logger named after the concrete class.

    Any class calling ``self.init_logger()`` in its ``__init__`` gets a ``self.logger``.
    """

    def init_logger(self, setup_logger=True, name=None):
        """
        Initialize the logger for the class.

        The logger is named ``hoddies.<ClassName>``, so configuring the ``hoddies``
        logger controls this package without affecting libraries such as pycorr.
        Logging is activated with :func:`setup_logging` if this package's logger has
        not been configured yet, so messages are visible by default. Call
        :func:`setup_logging` explicitly (e.g. at the top of a script or notebook)
        to choose the level, or to redirect records to a file.

        Parameters
        ----------
        setup_logger : bool, default=True
            If ``False``, ``self.logger`` is created but emits nothing.
        name : str, default=None
            Logger name suffix; defaults to the concrete class name.
        """
        package = logging.getLogger(PACKAGE_LOGGER)

        if setup_logger and not package.handlers:
            setup_logging()

        self.logger = logging.getLogger(
            f'{PACKAGE_LOGGER}.{self.__class__.__name__ if name is None else name}')
        self.logger.propagate = bool(setup_logger)

        if setup_logger:
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
        
        self.init_cat(halo_data, particle_data)
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
        self.field_particles = getattr(self, 'part_subsamples', None)
        # self.field_particles = self.load_particle_field() # if self.part_subsamples is not similar to the particle field (ie only particles in halos)

        
    
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
                missing = [col for col in self._init_part_cols if col not in mapping_part_cols]
                raise ValueError(f'mapping_part_cols is missing required columns: {missing}. Required columns are: {self._init_part_cols}. Current mapping_part_cols keys are: {list(mapping_part_cols.keys())}')

        
        
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

    def set_assembly_bias_values(self, columns, bins=50, **kwargs):

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
                self.load_env_based_properties(**kwargs)
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
            print(self.hcat['log10_Mh'], self.hcat[col])
            self.hcat[f'ab_{col}'] = initialize_assembly_bias_value(self.hcat['log10_Mh'], self.hcat[col], bins=bins)
            self.logger.info(f'Done !')
        return col_to_remove


    def assign_sat_to_part(self, mask_sat, list_nsat, seed=None, f_sigv=1.0):
        """
        Default method to assign satellites to particles.
        Assign particles to the selected host halos using the requested number
        of satellites per halo. Return positions, biased velocities, and a mask
        identifying satellites that require analytic placement.

        Parameters
        ----------
        mask_sat : array-like
            Boolean mask selecting host halos, in halo-catalogue order.
        list_nsat : array-like
            The number of satellites per halo, used to determine how many satellites to assign to each halo.
        seed : int or array-like, optional
            A random seed or one seed per host halo. Other seed-array lengths
            are used to initialise a reproducible stream of per-halo seeds.
        f_sigv : float, optional
            Particle velocity bias: v_sat = v_halo + f_sigv * (v_part - v_halo).
            One preserves the particle velocities.
        
        Returns
        -------
        x_sat, y_sat, z_sat : array-like
            The assigned positions of the satellite galaxies in the x, y, and z coordinates.
        vx_sat, vy_sat, vz_sat : array-like
            The assigned velocities of the satellite galaxies in the vx, vy, and vz components.
        mask_nfw : array-like
            True for satellites requiring analytic placement because their host
            has too few particles.
        Notes
        -----
        Particle indices are grouped by host ID, preserving the order of the
        selected halos. Each halo's output block has exactly list_nsat[i] rows.
        The velocity bias is applied inside the Numba sampling loop.

        """

        from HODDIES.utils import halo_to_particle_indices, sample_satellites_from_particles        
    
        uniq_sat_id = self.hcat['halo_id'][mask_sat]
        if not hasattr(self, '_order_part_index'):
            self._order_part_index = np.argsort(self.part_subsamples['halo_id'])
        flat, offsets = halo_to_particle_indices(self.part_subsamples['halo_id'][self._order_part_index], self._order_part_index, uniq_sat_id, self.nthreads)
        if seed is not None:
            seed_array = np.asarray(seed, dtype=np.uint32)
            if seed_array.ndim == 0 or seed_array.size != len(list_nsat):
                seed_input = int(seed_array) if seed_array.ndim == 0 else seed_array
                seed = np.random.RandomState(seed_input).randint(
                    0, 4294967295, size=len(list_nsat), dtype=np.uint32)
            else:
                seed = seed_array
        # The CSR mapping and outputs preserve selected halo order.
        x_sat, y_sat, z_sat, vx_sat, vy_sat, vz_sat, mask_nfw = sample_satellites_from_particles(self.part_subsamples['x'], self.part_subsamples['y'], self.part_subsamples['z'],
                                            self.part_subsamples['vx'], self.part_subsamples['vy'], self.part_subsamples['vz'], flat, offsets, list_nsat, seed=seed,
                                            vx_h=self.hcat['vx'][mask_sat], vy_h=self.hcat['vy'][mask_sat], vz_h=self.hcat['vz'][mask_sat], f_sigv=f_sigv)
        return x_sat, y_sat, z_sat, vx_sat, vy_sat, vz_sat, mask_nfw


    def plot_HMF(self, save_fn=None, show=False):
        """
        Plot the inital Halo Mass Function (HMF) from the halo catalog.

        This function generates a plot of the Halo Mass Function (HMF) using the halo mass values (`log10_Mh`) 
        from the provided catalog(s). The plot can optionally include histograms for central and satellite galaxies,
        and can display an initial HMF for comparison.

        Parameters
        ----------        
        range : tuple, optional
            The range for the satellite galaxy histogram. Default is (10.8, 15).
        
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
        plot_HMF()  # Plot HMF for the 'ELG' tracer with satellite galaxies.
        """
        import matplotlib.lines as mlines
        import matplotlib.pyplot as plt

        handles=[]
        
        plt.hist(self.hcat['log10_Mh'], histtype='step', bins=100, color='gray')
        handles +=[mlines.Line2D([], [], color='gray', label='inital HMF', ls='-')]

        plt.yscale('log')
        plt.ylabel('$N_{h}$')
        plt.xlabel(r'$\log(M_h\ [M_{\odot}])$')
        plt.legend(handles=handles, loc='upper right')
        plt.tight_layout()
        if save_fn is not None: 
            plt.savefig(save_fn)
        if show: 
            plt.show()

    def load_particle_field(self):
        """
        Load a subsample of particles from the simulation to compute density and shear meshes.
        This method is intended to be implemented in subclasses for specific simulations (e.g., Abacus, Uchuu).
        It should load a representative subsample of particles from the simulation data, which can then be used
        to compute the density and shear fields for environment-based properties.

        Returns
        -------
        field_particles : array-like
            A subsample of particles from the simulation, which can be used to compute density and shear meshes.
        
        Notes
        -----
        The implementation of this method should ensure that the loaded particle subsample is representative
        of the overall particle distribution in the simulation. The subsample can be used to compute density
        and shear fields for environment-based properties in the halo catalog.
        """

        self.field_particles = None
