""" Base HOD class """

import numba
import time 
import os
from .utils import *
from .estimators.clustering_statistics import compute_twopoint, compute_delta_sigma, get_list_stat, compute_CIC, compute_power_spectrum
from . import HOD_models
import yaml 
import glob
from mpytools import Catalog
import collections.abc
from .fit_functions.fits_functions import compute_chi2
import numbers
import numpy as np
from .sim_loader import BaseLogger

class HOD(BaseLogger):

    """Class with tools to generate HOD mock catalogs and plotting functions"""

    def __init__(self, base_catalog, param_file=None, **kwargs):
        
        """
        Initialize :class:`HOD`. 

        Parameters
        ----------
        base_catalog : Base_catalogue
            An instance of the :class:`Base_catalogue` class, which contains the halo catalog and associated data.
        param_file : str, default=None
            Input parameter file to initialize the HOD class. If None, the default parameter file 'default_HOD_parameters.yaml' is used.
        kwargs : dict
            Optional arguments that can be added that will replace the one provided in the parameter file.

        """
        setup_logger=kwargs.get('setup_logger', True)
        print('setup_logger', setup_logger)
        self.init_logger(setup_logger=kwargs.get('setup_logger', True))

        # self.args = yaml.load(open(os.path.join(os.path.dirname(__file__), 'default_HOD_parameters.yaml')), Loader=yaml.FullLoader)
        self._get_default_parameters()
        self.H_0 = 100 # H_0 is always set to 100 km/s/Mpc
        self.base_catalog = base_catalog
        self.nthreads = min(numba.get_num_threads(), kwargs.get('nthreads', 32))

        
        new_args = yaml.load(open(param_file), Loader=yaml.FullLoader) if param_file is not None else self.args
        # if new_args['fit_param'].get('priors') is not None:
        #     self.args['fit_param'].pop('priors')
        # if kwargs.get('fit_param') is not None:
        #     if kwargs['fit_param'].get('priors') is not None:
        #         self.args['fit_param'].pop('priors')
        
        update_dic(self.args, new_args)
        update_dic(self.args, kwargs)
        self._initialize_tracer_params(self.args['tracers'])
        # if 'tracers' not in self.args:
        #     raise(ValueError('need to provide tracer names'))

        
        if self.base_catalog.z_simu is None:
            self.logger.warning('Redshift of the simulation is not provided cannot compute statistics with RSD, please provide z_simu of the snapshot.')
            self.args['clustering_settings']['rsd'] = False
        # self.use_particles = kwargs.get('use_particles', None)

        self.z_simu = self.base_catalog.z_simu
        self.boxsize = self.base_catalog.boxsize
        self.hcat = self.base_catalog.hcat
        self.part_subsamples = self.base_catalog.part_subsamples
        self.field_particles = self.base_catalog.field_particles

        
        self.logger.info(f'Set number of threads to {self.nthreads}')

        # init cosmology 
        self.cosmo = self.base_catalog.cosmo(**self.args['cosmo'])
        
        try :
            self._fun_cHOD, self._fun_sHOD = {}, {}
            for tr in self._tracers():
                self._fun_cHOD[tr] = getattr(HOD_models, '_'+self.args[tr]['HOD_model'])
                self._fun_sHOD[tr] = getattr(HOD_models, '_'+self.args[tr]['sat_HOD_model'])
        except AttributeError:
            help(HOD_models)
            raise ValueError('HOD model or satellite HOD model not implemented in HOD_models '
                             '(available functions listed above). Requested: {}'.format(
                                 {tr: (self.args[tr].get('HOD_model'), self.args[tr].get('sat_HOD_model'))
                                  for tr in self._tracers()}))
                
        if self.args['use_assembly_bias']:
                self._compute_assembly_bias_columns()
        self.rng = np.random.RandomState(seed=self.args['seed'])
    
    
    def _warn_once(self, msg):
        """
        Log `msg` at WARNING level only the first time it is seen.

        Mimics the de-duplication of :func:`warnings.warn`, which these messages used
        previously, so configuration warnings raised inside :meth:`make_mock_cat` do
        not repeat on every call during a fit.
        """
        if not hasattr(self, '_warned_msgs'):
            self._warned_msgs = set()
        if msg not in self._warned_msgs:
            self._warned_msgs.add(msg)
            self.logger.warning(msg)


    def _initialize_tracer_params(self, tracer):
        """
        Initializes HOD parameters for one or more tracers in-place,
        using the default LRG parameters as a template.

        This function sets the default HOD parameters for the given tracer(s)
        by updating the main `self.args` dictionary. If a tracer's configuration
        does not exist in `self.args`, it will be created.

        Parameters
        ----------
        tracer : str or list of str
            The name(s) of the tracer(s) to initialize.

        Raises
        ------
        TypeError
            If the tracer is not a string or a list of strings, or if elements
            in the list are not strings.
        """
        
        tracer_templates = {
            'LRG': {
                'HOD_model': 'SHOD', 'Ac': 1, 'log_Mcent': 12.75, 'sigma_M': 0.5, 'gamma': 1, 'pmax': 1, 'Q': 100,
                'satellites': True, 'sat_HOD_model': 'Nsat_pow_law', 'As': 1, 'M_0': 13, 'M_1': 13.5, 'alpha': 1,
                'f_sigv': 1, 'f_vcen': 0.0, 'vel_sat': 'rd_normal', 'v_infall': 0, 'link_sat_to_central':False,
                'assembly_bias':{'c': [0, 0], 'env': [0, 0], 'shear': [0, 0]}, 'nu': 1,
                'conformity_bias': False, 'exp_frac': 0, 'exp_scale': 1, 'nfw_rescale': 1,
                'density': 0.0007, 'vsmear': 0
            },
            'ELG': {
                'HOD_model': 'mHMQ', 'Ac': 0.05, 'log_Mcent': 11.63, 'sigma_M': 0.12, 'gamma':2, 'pmax': 1, 'Q': 100,
                'satellites': True, 'sat_HOD_model': 'Nsat_pow_law', 'As': 0.11, 'M_0': 11.63, 'M_1': 11.7, 'alpha': 0.6,
                'f_sigv': 1, 'f_vcen': 0.0, 'vel_sat': 'rd_normal', 'v_infall': 0, 'link_sat_to_central':False,
                'assembly_bias': {'c': [0, 0], 'env': [0, 0], 'shear': [0, 0]}, 'nu': 1,
                'conformity_bias': False, 'exp_frac': 0, 'exp_scale': 1, 'nfw_rescale': 1,
                'density': 0.001, 'vsmear': 0
            },
            'QSO': {
                'HOD_model': 'SHOD', 'Ac': 1, 'log_Mcent': 13.25, 'sigma_M': 0.6, 'gamma': 1, 'pmax': 1, 'Q': 100,
                'satellites': True, 'sat_HOD_model': 'Nsat_pow_law', 'As': 1, 'M_0': 13.25, 'M_1': 14.25, 'alpha': 1.3,
                'f_sigv': 1, 'f_vcen': 0.0, 'vel_sat': 'rd_normal', 'v_infall': 0, 'link_sat_to_central':False,
                'assembly_bias': {'c': [0, 0], 'env': [0, 0], 'shear': [0, 0]}, 'nu': 1,
                'conformity_bias': False, 'exp_frac': 0, 'exp_scale': 1, 'nfw_rescale': 1,
                'density': 0.0001, 'vsmear': 100
            }
        }
        
        tracers = tracer
        if isinstance(tracer, str):
            tracers = [tracer]

        if not isinstance(tracers, list) or not all(isinstance(t, str) for t in tracers):
            raise TypeError(f"Tracer must be a string or a list of strings, but got {type(tracer)}")

        for t in tracers:
            if t not in self.args:
                self.args[t] = {}
            
            template = tracer_templates.get(t, tracer_templates['LRG'])
            
            # Create a copy to avoid modifying the template dictionary
            params_to_set = template.copy()
            
            # Update with existing values to preserve them
            params_to_set.update(self.args[t])
            
            # Set the merged parameters
            self.args[t] = params_to_set

    # def _initialize_fit_params(self, tracer):
    #     """
    #     Initializes HOD fit parameters for one or more tracers in-place.
    #     """
    #     fit_param_template = {
    #         'nb_real': 20, 'fit_name': 'myhodfit', 'path_to_training_point': None, 'dir_output_fit': 'path_to_save_fit_outputs',
    #         'fit_type': 'wp+xi', 'generate_training_sample': True, 'sampling_type': 'Hammersley',
    #         'N_training_points': 800, 'seed_training': 18, 'n_calls': 800, 'logchi2': True,
    #         'sampler': 'emcee', 'n_iter': 10000, 'nwalkers': 20, 'func_aq': 'EI',
    #         'length_scale_bounds': [0.001, 10], 'length_scale': False, 'kernel_gp': 'Matern_52',
    #         'save_fn': 'results_fit.npy', 'use_desi_data': True, 'zmin': 0.8, 'zmax': 1.1,
    #         'dir_data': '/global/homes/a/arocher/users_arocher/Y3/loa-v1/v1.1/PIP', 'region': 'GCcomb',
    #         'weights_type': 'pip_angular_bitwise', 'njack': 128, 'nran': 4, 'bin_type': 'log',
    #         'load_cov_jk': False, 'corr_dir': '/dvs_ro/cfs/cdirs/desi/users/arocher/Y1/2PCF_for_corr/Abcaus_small_boxes/',
    #         'nb_mocks': 1883,
    #         'priors': {}
    #     }

    #     priors_templates = {
    #         'LRG': {
    #             'M_0': [12.5, 13.5], 'M_1': [13, 14.5], 'alpha': [0.5, 1.5], 'f_sigv': [0.5, 1.5],
    #             'log_Mcent': [12.4, 13.5], 'sigma_M': [0.05, 1]
    #         },
    #         'ELG': {
    #             'M_0': [11.0, 12.5], 'M_1': [11.0, 12.5], 'alpha': [0.3, 1.2], 'f_sigv': [0.5, 1.5],
    #             'log_Mcent': [11.0, 12.5], 'sigma_M': [0.1, 1]
    #         },
    #         'QSO': {
    #             'M_0': [12.5, 14.0], 'M_1': [13.5, 15.0], 'alpha': [0.8, 1.8], 'f_sigv': [0.5, 1.5],
    #             'log_Mcent': [12.5, 14.0], 'sigma_M': [0.1, 1.0]
    #         }
    #     }
        

    #     if 'fit_param' not in self.args:
    #         self.args['fit_param'] = {}
        
    #     params_to_set = fit_param_template.copy()
    #     params_to_set.update(self.args['fit_param'])
    #     self.args['fit_param'] = params_to_set
        
    #     tracers = tracer
    #     if isinstance(tracer, str):
    #         tracers = [tracer]
        
    #     for t in tracers:
    #         if t not in self.args['fit_param']['priors']:
    #             # a default prior is assigned if the tracer is not defined
    #             template = priors_templates.get(t, priors_templates['LRG'])
    #             self.args['fit_param']['priors'][t] = template.copy()

    def _get_default_parameters(self):
        """
        Returns a dictionary with the default HOD and analysis parameters.
        """
        default_params = {
            'tracers': 'LRG',
            'cosmo': {
                'fiducial_cosmo': None, 'engine': 'class', 'h': None, 'Omega_m': None, 'Omega_cdm': None,
                'Omega_L': None, 'Omega_b': None, 'sigma_8': None, 'n_s': None, 'w0_fdl': None, 'wa_fdl': None
            },

            'seed': None, 'nthreads': 32, 'use_assembly_bias': False, 'use_particles': False, 'dir_to_save_env_mesh': 'tmp/',
        }
        
        default_params['clustering_settings'] = self.__get_default_clustering_parameters()
        default_params['fit_param'] = self.__get_default_fit_parameters()
        self.args = default_params


    @staticmethod
    def __get_default_clustering_parameters():
        """
        Returns a dictionary with the default clustering analysis parameters.
        """
        return yaml.load(open(os.path.join(os.path.dirname(__file__), 'estimators/default_clustering_parameters.yaml')), Loader=yaml.FullLoader)


    @staticmethod
    def __get_default_fit_parameters():
        """
        Returns a dictionary with the default fit parameters.
        """
        return yaml.load(open(os.path.join(os.path.dirname(__file__), 'fit_functions/default_HOD_fit_parameters.yaml')), Loader=yaml.FullLoader)


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
        hod_list_param_cen = [self.args[tracer][name] for name in self._cen_param_names(self.args[tracer]['HOD_model'])]

        if self.args[tracer]['satellites']:
            hod_list_param_sat = np.array([self.args[tracer][name] for name in self._sat_param_names(self.args[tracer]['sat_HOD_model'])], dtype='float64')

        if self.args['use_assembly_bias']:
            hod_param_ab = np.array(list(self.args[tracer]['assembly_bias'].values()), dtype='float64').T

        return np.float64(hod_list_param_cen), hod_list_param_sat, hod_param_ab


    def _hod_param_names(self, model):
        """
        Return the ordered list of tracer-dictionary parameter names declared on
        the ``_<model>`` HOD function in HOD_models (its ``hod_param_names``
        attribute, set with the ``hod_model`` decorator there).

        This keeps the code model-agnostic: adding a new HOD model only requires
        defining and tagging its ``_<model>`` function in HOD_models.py -- nothing
        here (parameter loading in :meth:`__init_hod_param`, validation in
        :meth:`check_HOD_params`) needs to change.

        Raises
        ------
        ValueError
            If ``model`` is not implemented, or is implemented but not tagged with
            its parameter names.
        """
        try:
            func = getattr(HOD_models, '_' + model)
        except AttributeError:
            raise ValueError('{} not implemented in HOD models'.format(model))
        try:
            return func.hod_param_names
        except AttributeError:
            raise ValueError("HOD model '{0}' has no declared parameters: tag its "
                             "`_{0}` function in HOD_models.py with @hod_model([...]).".format(model))


    def _cen_param_names(self, HOD_model):
        """Return the ordered central-HOD parameter names required by ``HOD_model`` (see :meth:`_hod_param_names`)."""
        return self._hod_param_names(HOD_model)


    def _sat_param_names(self, sat_HOD_model):
        """Return the ordered satellite-HOD parameter names required by ``sat_HOD_model`` (see :meth:`_hod_param_names`)."""
        return self._hod_param_names(sat_HOD_model)


    def check_HOD_params(self, tracers=None, raise_error=True):
        """
        Check that every parameter required by each tracer's HOD model is defined
        in that tracer's parameter dictionary ``self.args[tracer]``.

        For each tracer, the required parameters are:

        - ``HOD_model`` itself, and the central-HOD parameters it needs
          (see :meth:`_cen_param_names`);
        - when ``self.args[tracer]['satellites']`` is True, ``sat_HOD_model`` and the
          satellite-HOD parameters (see :meth:`_sat_param_names`).

        Parameters
        ----------
        tracers : str or list of str, optional
            Tracer(s) to check. If None, all tracers in ``self._tracers()`` are checked.
        raise_error : bool, default=True
            If True, raise a ``ValueError`` listing the missing parameters. If False,
            only return the dictionary of missing parameters.

        Returns
        -------
        missing : dict
            ``{tracer: [missing_param_names]}`` for tracers with missing parameters
            (empty dict if every required parameter is defined).

        Raises
        ------
        ValueError
            If ``raise_error`` is True and at least one required parameter is missing,
            or if a tracer uses an HOD model that is not implemented.
        """
        if tracers is None:
            tracers = self._tracers()
        elif isinstance(tracers, str):
            tracers = [tracers]

        missing = {}
        for tracer in tracers:
            if tracer not in self.args:
                raise ValueError(f'{tracer} HOD parameters are not defined in the arguments.')
            tr_args = self.args[tracer]

            required = ['HOD_model']
            if 'HOD_model' in tr_args:
                required += self._cen_param_names(tr_args['HOD_model'])
            if tr_args.get('satellites', False):
                required += ['sat_HOD_model']
                if 'sat_HOD_model' in tr_args:
                    required += self._sat_param_names(tr_args['sat_HOD_model'])
                if (tr_args['nu'] >= 4) | (tr_args['nu'] <= 0.5):
                    self.logger.warning(f"nu must satisfy {0.5} <= nu <= {4}; got {tr_args['nu']}. Outside that range, fix it to {np.clip(tr_args['nu'], 0.5, 4)}")
                tr_args['nu'] = np.clip(tr_args['nu'], 0.5, 4)

                if (tr_args['vel_sat'] != 'NFW') and (tr_args['vel_sat'] != 'rd_normal') and (tr_args['vel_sat'] != 'infall'):
                    raise ValueError(f'vel_sat={tr_args["vel_sat"]} is not a valid argument for satellite velocities, only "rd_normal", "infall", "NFW" are available.')

            miss = [name for name in required if name not in tr_args]
            if miss:
                missing[tracer] = miss

        if missing and raise_error:
            details = '; '.join('{} (HOD_model={}): missing {}'.format(tr, self.args[tr].get('HOD_model', '?'), names)
                                for tr, names in missing.items())
            raise ValueError('Undefined HOD parameters -> ' + details)
        return missing

    def check_clustering_settings(self, stat):
        """
        Check that the clustering settings are valid.

        Raises
        ------
        ValueError
            If any of the clustering settings are invalid.
        """

        if self.args['clustering_settings'].get(stat, None) is None:
            raise ValueError(f"Clustering settings for {stat} are implemented.")
        

    def check_cat_tracers(self, cat, tracers=None):
        uniq_tr = list(np.unique(cat['TRACER']))
        if tracers is None: 
            tracers = uniq_tr
        else:
            tracers = tracers if isinstance(tracers, list) else [tracers]

        missing_tr = [tr for tr in tracers if tr not in uniq_tr]
        if missing_tr:
            raise ValueError(f'{missing_tr} not found in the catalog. Available: {sorted(uniq_tr)}')
        return tracers

    def _tracers(self):
        """
        Helper function to return the list of tracers defined in the parameter file
        """

        if isinstance(self.args['tracers'], str):
            self.args['tracers'] = [self.args['tracers']]
        return self.args['tracers']



    def get_ds_fac(self, tracer, verbose=False):
        """
        Calculate the density scaling factor based on the specified tracer and density value.

        Parameters
        ----------
        tracer : str
            The key identifying the specific tracer (e.g., galaxy type) for which the density scaling factor is calculated.
        verbose : bool, optionalAB_c_cen
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

        If `verbose` is set to True, the following message will be logged:
        - "Set density to 0.01 gal/Mpc/h".
        """

        if isinstance(self.args[tracer]['density'], float):
            if verbose:
                self.logger.debug(f"Set density to {self.args[tracer]['density']} gal/Mpc/h")
            return self.args[tracer]['density']*self.boxsize**3 /self.ngal(tracer)[0]
        else:
            if verbose:
                self.logger.debug('No density set')
            return 1
    
    def ngal(self, tracer, return_mass=False, verbose=False):
        """
        Return the number of galaxy and the satelitte fraction 
        
        Parameters
        ----------
            tracer: str
                Name of the galaxy tracer in self.tracers 
            return_mass (bool, optional): Defaults to False.
            verbose (bool, optional): Defaults to False.

        Returns
        -------
        ngal: float
            Total number of galaxies expected
        
        fsat: float
            Expected fraction of satellite galaxy (n_sat/ngal)
        
        mean_hmass: float, optional
            Mean halo mass of the galaxies in log10(Msun/h) (only returned if return_mass is True)
        
        """
        start = time.time()
        hod_list_param_cen, hod_list_param_sat, _ = self.__init_hod_param(tracer)
        ngal, fsat, mean_hmass = compute_ngal(self.hcat['log10_Mh'], self._fun_cHOD[tracer], self._fun_sHOD[tracer],
                                hod_list_param_cen, hod_list_param_sat, self.args[tracer]['conformity_bias'], self.args[tracer]['link_sat_to_central'])
        if verbose:
            self.logger.debug(f'ngal computed in {time.time()-start:.2f} sec')
        if return_mass:
            return ngal, fsat, np.log10(mean_hmass)   
        else:
            return ngal, fsat


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

        ab_proxy = []
        for tr in self._tracers():
            if self.args[tr].get('assembly_bias'):
                ab_proxy += [list(self.args[tr]['assembly_bias'].keys())]
        ab_proxy = list(set().union(*ab_proxy))
        abproxy_to_remove = self.base_catalog.set_assembly_bias_values(ab_proxy, **self.args)
        self._remove_env_bias(abproxy_to_remove)


    def _remove_env_bias(self, ab_proxy_to_remove):
            """
            Remove assembly bias parameters from a list of columns for all tracers.
            Parameters
            ----------
            ab_proxy_to_remove : list
                List of assembly bias column names to remove from the tracers' parameters.

            Notes
            -----
            This function iterates over all tracers and removes the specified assembly bias parameters from both the
            tracer-specific assembly bias dictionary and the fit parameter priors dictionary. If a specified column
            does not exist in the dictionaries, it is ignored (no error is raised).
            """
            
            for tr in self._tracers():
                for c in ab_proxy_to_remove:
                    self.args.get(tr, {}).get('assembly_bias', {}).pop(c, None)
                    self.args.get('fit_param', {}).get('priors', {}).get(tr, {}).get('assembly_bias', {}).pop(c, None)



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
        - The tracer parameter `f_vcen` adds `f_vcen * Vrms` to each Cartesian
        central velocity component: `v_c = v_h + f_vcen * Vrms`. It defaults
        to zero and applies a deterministic offset without additional random draws.
        - If `self.args['assembly_bias']` is enabled, assembly bias columns will be computed and included in the mock catalog.
        - The `fix_seed` parameter ensures that the mock catalogs are generated in a reproducible manner, but it requires consistent 
        thread configurations (`self.nthreads`).
        - When `tracers` includes both 'ELG' and 'LRG', the method handles the case where both tracers share the same halo by placing
        one LRG at the center and positioning other galaxies (like ELGs) based on the NFW profile.

        Example
        -------
        To generate mock catalogs for both 'LRG' and 'ELG' tracers with fixed seed for reproducibility:

        final_cat = mock_catalog.make_mock_cat(tracers=['LRG', 'ELG'], fix_seed=42)
        """

        rng = np.random.RandomState(seed=fix_seed)
        start_all = time.time()
        
        if tracers is None: 
            tracers = self._tracers()
        else:
            tracers = tracers if isinstance(tracers, list) else [tracers]
            for tr in tracers:
                if tr not in self._tracers():
                    raise ValueError(f'{tr} not in defined tracers {self._tracers()}')
        # Fail early with a clear message if any required HOD parameter is missing.
        self.check_HOD_params(tracers)
        if verbose:
            self.logger.info(f'Create mock catalog for {tracers}')


        if self.args['use_assembly_bias']:
            self._compute_assembly_bias_columns()

        final_cat = {}

        
        weight = np.ones(self.hcat['log10_Mh'].size, dtype=np.float64)
        if len(tracers) > 1:
            cc, bins = np.histogram(self.hcat['log10_Mh'], bins=100)
        
        for tracer in tracers:
            start = time.time()
            self._fun_cHOD[tracer] = getattr(HOD_models, '_'+self.args[tracer]['HOD_model'])
            self._fun_sHOD[tracer] = getattr(HOD_models, '_'+self.args[tracer]['sat_HOD_model'])
            if verbose:
                self.logger.info(f'Run HOD for {tracer}')
            hod_list_param_cen, hod_list_param_sat, hod_list_ab_param = self.__init_hod_param(tracer)

            ds = self.get_ds_fac(tracer, verbose=verbose)
            if (hod_list_param_cen[0]*ds > 1):
                self._warn_once(f'Ac={hod_list_param_cen[0]*ds} is > 1, the density is not fixed to {self.args[tracer]["density"]}')
            else : 
                hod_list_param_cen[0] *= ds
                if hod_list_param_sat is not None:
                    hod_list_param_sat[0] *= 1 if self.args[tracer]['link_sat_to_central'] else ds

            if fix_seed is not None:
                seed = rng.randint(0, 4294967295, self.nthreads)
            else:
                seed = None
            
            if self.args['use_assembly_bias'] & (hod_list_ab_param is not None):
                cols_ab =  ['ab_'+col for col in self.args[tracer]['assembly_bias'].keys()]
                if np.all([col in self.hcat.columns() for col in cols_ab]):
                    ab_arr =  np.vstack([self.hcat[col] for col in cols_ab]).T
                else:
                    self._warn_once(f'Precomputed columns for assembly bias have not been found {cols_ab}. Continue without assembly bias.')
                    hod_list_ab_param=None
            else:
                hod_list_ab_param, ab_arr = None, None
            if verbose:
                self.logger.debug(f'Initialisation in {time.time()-start:.2f} sec')
                st = time.time()
            cond_cent, proba_sat, Nb_sat = compute_N(self.hcat['log10_Mh'], self._fun_cHOD[tracer], self._fun_sHOD[tracer], hod_list_param_cen, 
                                                   hod_list_param_sat, self.args[tracer]['nu'], hod_list_ab_param, self.nthreads, ab_arr, weight,
                                                   self.args[tracer]['conformity_bias'], self.args[tracer]['link_sat_to_central'], seed)
            if len(tracers) > 1:
                # Update the weight using the halo mass function to ensure the total number of galaxies is correct when multiple tracers are used. The weight is set to zero if the halo is already populated by a central galaxy.
                mm = weight > 0
                cc_1, bins = np.histogram(self.hcat['log10_Mh'][~cond_cent & mm], bins=bins)
                idx = np.digitize(self.hcat['log10_Mh'][~cond_cent & mm], bins) - 1    # -1 because digitize is 1-based
                idx = np.clip(idx, 0, len(cc) - 1)  # guard the rightmost edge
                weight[cond_cent] = 0
                weight[weight > 0] = cc[idx] / cc_1[idx]                    # one value per element of `masses`
            
            if verbose:
                self.logger.debug(f'Ncent, Nsat computed in {time.time()-st:.2f} sec')
                st = time.time()
            
            cent_cat = self.hcat[cond_cent]
            cent_cat['Central'] = np.ones(cent_cat['x'].size,dtype='int')
            f_vcen = self.args[tracer].get('f_vcen', 0.0)
            if f_vcen != 0.0:
                if 'Vrms' not in cent_cat.columns():
                    self._warn_once(
                        f"Vrms is unavailable for tracer '{tracer}'; skipping "
                        "central velocity bias and keeping halo velocities.")
                else:
                    velocity_offset = f_vcen * cent_cat['Vrms']
                    for velocity in ('vx', 'vy', 'vz'):
                        cent_cat[velocity] = cent_cat[velocity] + velocity_offset

            if verbose:
                self.logger.debug(f'Central catalog in {time.time()-st:.2f} sec')
                self.logger.debug(f'HOD computed for {tracer} in {time.time()-start:.2f} sec')
            
            if (not self.args[tracer]['satellites']) | (Nb_sat == 0):
                Nb_sat=0
                final_cat[tracer] = cent_cat
                final_cat[tracer]['TRACER'] = [tracer]*final_cat[tracer].size
                
            else:
                start_sat = time.time()
                if verbose:
                    self.logger.debug('Start satellite assignement')
                mask_sat = proba_sat > 0
                list_nsat = proba_sat[mask_sat]
                # sat_cat = Catalog.from_array(np.repeat(self.hcat[mask_sat].to_array(), list_nsat))
                sat_cat = Catalog()
                sat_cat.data = {col: np.repeat(self.hcat[col][mask_sat], list_nsat) for col in self.hcat.columns()}
                if self.args['use_particles'] & (self.part_subsamples is not None):
                    if verbose:
                        start_part = time.time()
                        self.logger.info("Assign satellites to particles...")
                    if fix_seed is not None:
                        seed = rng.randint(0, 4294967295, self.nthreads)
                    else:
                        seed = None

                    sat_cat['x'], sat_cat['y'], sat_cat['z'], sat_cat['vx'], sat_cat['vy'], sat_cat['vz'], mask_nfw = self.base_catalog.assign_sat_to_part(
                        mask_sat, list_nsat, seed=seed, f_sigv=self.args[tracer]['f_sigv'])
                    if verbose:
                        self.logger.debug(f'Sample satellites from particles done in {time.time() - start_part:.2f} sec')
                        self.logger.debug(f'{mask_nfw.sum()} satellites will be positioned using NFW')
                else:
                    if self.args['use_particles']:
                        self._warn_once('use_particles is set to True but no particle subsample found, continue with NFW profile for satellites')
                    if self.part_subsamples is not None:
                        self._warn_once('Particle subsample loaded but use_particles is set to False, continue with NFW profile for satellites. To use particles, set use_particles to True.')
                    mask_nfw = np.ones(Nb_sat, dtype=bool)
                    
                
                if mask_nfw.sum() > 0:
                    start_nfw = time.time()
                    if fix_seed is not None:
                        seed1 = rng.randint(0, 4294967295, self.nthreads)
                    else:
                        seed1 = None
                    rd_pos = getPointsOnSphere_jit(Nb_sat, np.minimum(Nb_sat, self.nthreads), seed1)
                    if self.args[tracer]['vel_sat'] == 'NFW':
                        seed2 = rng.randint(0, 4294967295, self.nthreads)
                        rd_v = getPointsOnSphere_jit(Nb_sat, np.minimum(Nb_sat, self.nthreads), seed2)
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
                    vel_sat=self.args[tracer]['vel_sat'], Nthread=self.nthreads, seed=seed)

                    if verbose:
                        self.logger.debug(f'NFW satellite assignment done in {time.time() - start_nfw:.2f} sec')
                sat_cat['Central'] = sat_cat.zeros()

                if verbose:
                    self.logger.debug(f'Satellite assignement done in {time.time() - start_sat:.2f} sec')

                if verbose:
                    self.logger.debug('Make final catalog')
                final_cat[tracer] = Catalog.concatenate((cent_cat,sat_cat))
                # final_cat[tracer]['TRACER'] = [tracer]*final_cat[tracer].size
                final_cat[tracer]['TRACER'] = np.full(final_cat[tracer].size, tracer)
            if verbose:
                self.logger.info(f'{tracer} mock catalogue done in {time.time()-start:.2f} sec')


        final_cat = Catalog.concatenate(list(final_cat.values()))

        if verbose:
            self.logger.info(f'Total time to create the mock catalog: {time.time() - start_all:.2f} seconds')
            for tr in tracers:
                mask_tr = final_cat['TRACER'] == tr
                Ncen = np.count_nonzero(final_cat['Central'][mask_tr])
                Nsa = mask_tr.sum()-Ncen
                self.logger.info(f'Total number of {tr}: {mask_tr.sum()} with {Ncen} central, {Nsa} satellite and a satellites fraction: {Nsa / mask_tr.sum():.2f}')
            Nb_sat = np.count_nonzero(final_cat['Central'] == 0)
            self.logger.info(f'Total number of galaxies in the catalog: {final_cat.size}')
            self.logger.info(f'Total number of centrals: {final_cat.size-Nb_sat}')
            self.logger.info(f'Total number of satellites: {Nb_sat}')
            
        return final_cat


    

    def get_vsmear(self, tracer, cat_size, verbose=True):

        """
        Generate a random velocity smear for the specified tracer
        """

        if isinstance(self.args[tracer]['vsmear'], numbers.Number) and (self.args[tracer]['vsmear'] != 0):
            if self.args[tracer]['vsmear'] < 0:
                raise ValueError('vsmear must be positive')
            if verbose:
                self.logger.info(f"Generate gaussian vsmear for {tracer} of {self.args[tracer]['vsmear']} km/s...")
            vsmear = self.rng.normal(0, self.args[tracer]['vsmear'], cat_size)

        elif isinstance(self.args[tracer]['vsmear'], list):
            from HODDIES.desi.Y3_redshift_systematics import vsmear as gen_vsmear
            if verbose:
                self.logger.info(f"Generate vsmear for {tracer} at z {self.args[tracer]['vsmear'][0]:.2f}-{self.args[tracer]['vsmear'][1]:.2f}...")
            vsmear = gen_vsmear(tracer, self.args[tracer]['vsmear'][0], self.args[tracer]['vsmear'][1], cat_size, dvmode='obs',seed=42,verbose=verbose)
        else:
            vsmear = 0
        return vsmear
    

    def get_xiells(self, cats, tracers=None, ells=None, R1R2=None, verbose=True):
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
            Multipoles to project onto. If None, ells in `self.args['clustering_settings']['smu']['multipole_index']` are considered. Default to None.
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
        `compute_twopoint` function, and the results are stored for each tracer.
        - The results are returned as lists (`s_all` and `xi_all`) when multiple tracers are provided.
        - The function supports a log-scale binning option for radial bins if `bin_logscale` is True.
        - The output is either the 2PCF for a single tracer (if only one tracer is given) or for 
        all tracers provided in the list.

        Example
        -------
        s, xi = get_xiells(cats, tracers=['LRG', 'ELG'])
        """

        tracers = self.check_cat_tracers(cats, tracers)

        self.check_clustering_settings('xi_smu')
        smu_settings = self.args['clustering_settings'].get('xi_smu')
            
        if smu_settings['bin_logscale']:
            r_bins = np.geomspace(smu_settings['smin'], smu_settings['smax'], smu_settings['n_s_bins'])
        else:
            r_bins = np.linspace(smu_settings['smin'], smu_settings['smax'], smu_settings['n_s_bins'])

        if smu_settings.get('edges_smu', None) is None:
            smu_settings['edges_smu'] = (r_bins, np.linspace(-smu_settings['mu_max'], smu_settings['mu_max'], smu_settings['n_mu_bins']))
        ells = smu_settings['multipole_index'] if ells is None else ells
        s_all, xi_all = [],[]
        for tr in tracers:
            mock_cat = cats[cats['TRACER'] == tr]
            if verbose:
                self.logger.info(f'Compute xi(s,mu) using l={ells} for {tr}...')
                time1 = time.time() 

            if (self.cosmo is not None) & (self.args['clustering_settings']['rsd']):
                vsmear = self.get_vsmear(tr, mock_cat.size, verbose=verbose)
                pos = apply_rsd(mock_cat, self.z_simu, self.boxsize, self.cosmo, self.H_0, self.args['clustering_settings']['los'], vsmear)
            else:
                if self.args['clustering_settings']['rsd']:
                    self.logger.warning('Cosmology not set, does not apply rsd')
                pos = mock_cat['x']%self.boxsize, mock_cat['y']%self.boxsize, mock_cat['z']%self.boxsize
            
            result = compute_twopoint(pos, 'smu', smu_settings['edges_smu'], self.boxsize, self.args['clustering_settings']['los'], self.nthreads, R1R2=R1R2)
            s,xi = result(return_sep=True, ells=ells)

            if verbose:
                self.logger.info(f'Done in {time.time()-time1:.3f} s')
            if len(tracers) > 1:
                s_all += [s]
                xi_all += [xi]
            else: 
                return s, xi
        return s_all, xi_all


    def get_wp(self, cats, tracers=None, pimax=None, R1R2=None, verbose=True):
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
        pimax : float, optional
            The maximum projection distance for the wp computation. If None, the default value will be used. Defaults to None.
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
        `compute_twopoint` function, and the results are stored for each tracer.
        - The results are returned as lists (`rp_all` and `wp_all`) when multiple tracers are provided.
        - The function supports a log-scale binning option for radial bins if `bin_logscale` is True.
        - The output is either the wp for a single tracer (if only one tracer is given) or for 
        all tracers provided in the list.

        Example
        -------
        rp, wp = get_wp(cats, tracers=['LRG', 'ELG'])
        """

        tracers = self.check_cat_tracers(cats, tracers)

        self.check_clustering_settings('xi_rppi')
        rppi_settings = self.args['clustering_settings'].get('xi_rppi', None)

        pimax = rppi_settings['pimax'] if pimax is None else pimax
        if rppi_settings['bin_logscale']:
            r_bins = np.geomspace(rppi_settings['rp_min'], rppi_settings['rp_max'], rppi_settings['n_rp_bins']+1, endpoint=(True))
        else:
            r_bins = np.linspace(rppi_settings['rp_min'], rppi_settings['rp_max'], rppi_settings['n_rp_bins']+1)
            
        if rppi_settings.get('edges_rppi', None) is None:
            rppi_settings['edges_rppi'] = (r_bins, np.linspace(-pimax, pimax, 2*pimax+1))

        rp_all, wp_all = [],[]
        for tr in tracers:
            mock_cat = cats[cats['TRACER'] == tr]
            if verbose:
                self.logger.info(f'Compute wp for {tr}...')
                time1 = time.time()
            
            if (self.cosmo is not None) & (self.args['clustering_settings']['rsd']):
                vsmear = self.get_vsmear(tr, mock_cat.size, verbose=verbose)
                pos = apply_rsd(mock_cat, self.z_simu, self.boxsize, self.cosmo, self.H_0, self.args['clustering_settings']['los'], vsmear)
            else:
                if self.args['clustering_settings']['rsd']:
                    self.logger.warning('Cosmology not set, does not apply rsd')
                pos = mock_cat['x']%self.boxsize, mock_cat['y']%self.boxsize, mock_cat['z']%self.boxsize
                
            result = compute_twopoint(pos, 'rppi', rppi_settings['edges_rppi'], self.boxsize, self.args['clustering_settings']['los'],  self.nthreads, R1R2=R1R2)
            rp, wp = result(return_sep=True, pimax=pimax)
            if verbose:
                self.logger.info(f'Done in {time.time()-time1:.3f} s')
            if len(tracers) > 1:
                rp_all += [rp]
                wp_all += [wp]
            else: 
                return rp, wp
        return rp_all, wp_all


    def get_PS(self, cat, tracers=None, ells=None, verbose=True):
        """
        Compute the power spectrum (PS) for a given mock catalog in a cubic box.

        This function calculates the power spectrum (PS) for specified galaxy tracers 
        in a mock catalog. It can compute PS for multiple tracers, returning the results for each.

        Parameters
        ----------
        cats : dict
            A dictionary of mock catalogs where the keys are the names of the tracers 
            and the values are the corresponding catalogs (e.g., 'LRG', 'ELG').
        tracers : list or str, optional
            A list of tracer names (keys in `cats`) for which the PS should be computed. 
            If None, all tracers in `self.args['tracers']` are considered. Defaults to None.
        ells : tuple of int, optional
            The multipole indices for which to compute the power spectrum. Defaults to the values in `ps_settings['multipole_index']`.
        verbose : bool, optional
            If True, prints progress and computation time for each tracer. Defaults to True.

        Returns
        -------
        k_all : list
            A list of wavenumbers corresponding to the computed PS for each tracer.
        Pk_all : list
            A list of the power spectrum (PS) values for each tracer.

        Notes
        -----
        - The function uses `apply_rsd` to account for redshift space distortions (RSD) if enabled and Cosmology set.
        - The wavenumbers `k` and the power spectrum values `Pk` are calculated using the 
        `compute_power_spectrum` function, and the results are stored for each tracer.
        - The results are returned as lists (`k_all` and `Pk_all`) when multiple tracers are provided.
        - The output is either the PS for a single tracer (if only one tracer is given) or for 
        all tracers provided in the list.

        Example
        -------
        k, Pk = get_PS(cats, tracers=['LRG', 'ELG'])
        """

        tracers = self.check_cat_tracers(cat, tracers)

        self.check_clustering_settings('power_spectrum')
        ps_settings = self.args['clustering_settings'].get('power_spectrum', None)

        if ps_settings.get('k_edges', None) is None:
            if ps_settings['bin_logscale']:
                ps_settings['k_edges'] = np.geomspace(ps_settings['kmin'], ps_settings['kmax'], ps_settings['n_k_bins']+1, endpoint=(True))
            else:
                ps_settings['k_edges'] = np.linspace(ps_settings['kmin'], ps_settings['kmax'], ps_settings['n_k_bins']+1)

        k_all, Pk_all = [],[]
        ells = ps_settings['multipole_index'] if ells is None else ells
        for tr in tracers:
            mock_cat = cat[cat['TRACER'] == tr]
            if verbose:
                self.logger.info(f'Compute PS for {tr}...')
                time1 = time.time()
            
            if (self.cosmo is not None) & (self.args['clustering_settings']['rsd']):
                vsmear = self.get_vsmear(tr, mock_cat.size, verbose=verbose)
                pos = apply_rsd(mock_cat, self.z_simu, self.boxsize, self.cosmo, self.H_0, self.args['clustering_settings']['los'], vsmear)
            else:
                if self.args['clustering_settings']['rsd']:
                    self.logger.warning('Cosmology not set, does not apply rsd')
                pos = mock_cat['x']%self.boxsize, mock_cat['y']%self.boxsize, mock_cat['z']%self.boxsize
                
            k, Pk = compute_power_spectrum(pos1=pos, nmesh=ps_settings['nmesh'], boxsize=self.boxsize, kedges=ps_settings['k_edges'], ells=ells, los=self.args['clustering_settings']['los'], resampler=ps_settings['resampler'], interlacing=ps_settings['interlacing']).poles(ell=ells, return_k=True, complex=False)
            if verbose:
                self.logger.info(f'Done in {time.time()-time1:.3f} s')
            if len(tracers) > 1:
                k_all += [k]
                Pk_all += [Pk]
            else: 
                return k, Pk
        return k_all, Pk_all


    def get_CIC(self, cat, tracers=None, return_dic=False, return_counts=False, max_count=None, density=True, verbose=True):
        """
        Compute counts-in-cylinder (CIC) for a given tracer.

        Parameters
        ----------
        cat : structured array
            Catalog containing the tracer.
        tracers : list or str
            Name of the tracer(s) for which to compute CIC.
        return_dic : bool, optional
            If True, returns a dictionary with the counts-in-cylinder for each tracer. Defaults to False.
        return_counts : bool, optional
            If True, returns the counts for each object in the catalog. Defaults to False.
        max_count : int, optional
            Maximum count to consider for the histogram. If None, it will be determined from the data. Defaults to None.
        density : bool, optional
            If True, returns the histogram as a density (normalized). Defaults to True.
        verbose : bool, optional
            If True, prints progress and computation time. Defaults to True.

        Returns
        -------
        cens : array
            Bin centers for the counts-in-cylinder histogram.
        hist : array
            Histogram of counts-in-cylinder for the specified tracer(s).
        counts : array
            Array of counts-in-cylinder for each object in the catalog.
        
        Notes
        -----
        - The function uses `apply_rsd` to account for redshift space distortions (RSD) if enabled and Cosmology set.
        - The counts-in-cylinder are computed using the `compute_CIC` function, which calculates the number of neighboring galaxies within a specified cylinder around each galaxy.
        - The results can be returned as a dictionary if `return_dic` is True, with keys corresponding to each tracer and values containing the bin centers and histogram.
        - The `max_count` parameter allows for controlling the range of counts considered in the histogram, which can be useful for focusing on specific ranges of interest.
        - The `density` parameter allows for returning the histogram as a normalized density, which can be useful for comparing distributions across different tracers or datasets.
        - The function supports multiple tracers, and the results are computed and returned for each tracer specified in the `tracers` parameter. If only one tracer is provided, the results are returned directly without being wrapped in a list or dictionary.
        - The `verbose` parameter allows for controlling the verbosity of the output, providing information on the progress and timing of the computations for each tracer. 
        """

        tracers = self.check_cat_tracers(cat, tracers)

        self.check_clustering_settings('CIC')
        cic_settings = self.args['clustering_settings'].get('CIC', None)
        max_count = cic_settings.get('max_count', None) if max_count is None else max_count
        
        counts = []
        hists = []

        for tr in tracers:
            mock_cat = cat[cat['TRACER'] == tr]
            if self.args['clustering_settings']['rsd']:
                vsmear = self.get_vsmear(tr, mock_cat.size, verbose=verbose)
                pos = apply_rsd(mock_cat, self.z_simu, self.boxsize, self.cosmo, self.H_0, self.args['clustering_settings']['los'], vsmear)
            else:
                pos = mock_cat['x']%self.boxsize, mock_cat['y']%self.boxsize, mock_cat['z']%self.boxsize
            counts += [compute_CIC(pos[0], pos[1], pos[2], self.boxsize, cic_settings['R_max'], cic_settings['L_max'])]
            if max_count is None:
                max_count = int(np.nanmax(counts))
            bins = np.arange(-0.5, max_count + 1.5)
            hist, edges = np.histogram(counts, bins=bins)
            cens = 0.5 * (edges[:-1] + edges[1:])
            if density and hist.sum() > 0:
                hist = hist / hist.sum()
            hists += [hist]
        if return_dic:
            return {f'{tr}_{tr}': [cens, hh] for tr, hh in zip(tracers, hists)}
        if return_counts:
            return cens, hist, counts
        return cens, hist
        
    

    def get_delta_sigma(self, cats, tracers=None, verbose=True, return_dic=False):
        """
        Compute ΔΣ(R) in units of 1e12[Msun/h / (Mpc/h)^2] for one or several lens tracers using particle.
    
        Parameters
        ----------
        cats : dict-like or structured array
            Catalog containing tracers (same logic as get_wp).
        tracers : list or str, optional
            Which tracers to compute ΔΣ for. If None, use all tracers.
        verbose : bool, optional
    
        Returns
        -------
        rp_all : list or array
            rp midpoints for ΔΣ(R).
        ds_all : list or array
            ΔΣ(R) in units of 1e12[Msun/h / (Mpc/h)^2]  for each tracer.
        """


        if self.field_particles is None:
            raise ValueError('Particle subsample is required to compute ΔΣ(R).')
        
        tracers = self.check_cat_tracers(cats, tracers)

        self.check_clustering_settings('delta_sigma')
        ds_settings = self.args['clustering_settings'].get('delta_sigma', None)
        if ds_settings.get('edges_rp', None) is None:
            if ds_settings['bin_logscale']:
                ds_settings['edges_rp'] = np.geomspace(ds_settings['rp_min'], ds_settings['rp_max'], ds_settings['n_rp_bins']+1, endpoint=(True))
            else:
                ds_settings['edges_rp'] = np.linspace(ds_settings['rp_min'], ds_settings['rp_max'], ds_settings['n_rp_bins']+1)
    
        if return_dic:
            res_dict = {}
        for tr in tracers:
    
            if verbose:
                self.logger.info(f'Compute ΔΣ for {tr}...')
                t0 = time.time()
    
            # Select tracer catalog
            mock_cat = cats[cats['TRACER'] == tr]
    
            # Lens positions (periodic)
            pos_lens = np.vstack(
                [mock_cat['x'],
                 mock_cat['y'],
                 mock_cat['z']]
            ).T % self.boxsize
    
            rp, ds = compute_delta_sigma(
                pos_lens,
                self.field_particles['pos'][::100]%self.boxsize,
                rbins=ds_settings['edges_rp'],
                boxsize=self.boxsize,
                rho_m=self.cosmo.rho_m(0.5) * 1e10,
                los=self.args['clustering_settings'].get('los', 'z'),
                pimax=ds_settings['pimax'],
                nthreads=self.nthreads,
            )
            if return_dic:
                res_dict[f'{tr}_{tr}'] = rp, ds
            if verbose:
                self.logger.info(f'Done in {time.time() - t0:.3f} s')

        if (len(tracers) == 1) & ~return_dic:
            return rp, ds
        return res_dict
    
    
    # def get_cross_wp(self, cats, tracers, return_rppi=None, R1R2=None, verbose=True):
    #     """
    #     Compute the projected correlation and cross-correlation functions (wp) for a given mock catalog in a cubic box.

    #     This function computes the projected two-point correlation and cross-correlation function (wp) for pairs of tracers in the 
    #     mock catalogs. It calculates wp for all combinations of tracers provided, handling redshift space distortions 
    #     (RSD) if enabled.

    #     Parameters
    #     ----------
    #     cats : dict
    #         A dictionary of mock catalogs where each key is a tracer and its corresponding catalog is the value 
    #         (e.g., 'LRG', 'ELG').
    #     tracers : list of str
    #         A list of tracer names (keys in `cats`) for which the cross wp should be computed. The function 
    #         computes the wp for all pairs of tracers in the list.
    #     R1R2 : tuple or None, optional  
    #         A tuple defining a range for R1 and R2 for the wp computation. If None, the default values will be used.
    #         Defaults to None.
    #     verbose : bool, optional
    #         If True, prints progress and computation time for each pair of tracers. Defaults to True.

    #     Returns
    #     -------
    #     res_dict : dict
    #         A dictionary where the keys are the concatenated names of tracer pairs (e.g., 'LRG_ELG') and the 
    #         values are the corresponding projected two-point correlation functions (wp) for each pair.

    #     Notes
    #     -----
    #     - The function computes the cross-correlation wp for all unique pairs of tracers from the input list.
    #     - If redshift space distortions (RSD) are enabled, the positions of galaxies in the catalogs are adjusted accordingly.
    #     - The results are stored in `res_dict` with keys in the format 'tracer1_tracer2', where each value is the wp 
    #     corresponding to the pair of tracers.
    #     - The function uses `compute_twopoint` to calculate the wp for each tracer pair.

    #     Example
    #     -------
    #     res = get_cross_wp(cats, tracers=['LRG', 'ELG', 'QSO'])
    #     """
    #     tracers = self.check_cat_tracers(cats, tracers)
        
    #     self.check_clustering_settings('xi_rppi')
    #     setting_rppi = self.args['clustering_settings'].get('xi_rppi', None)

    #     if setting_rppi.get('edges_rppi', None) is None:
    #         if setting_rppi['bin_logscale']:
    #             r_bins = np.geomspace(setting_rppi['rp_min'], setting_rppi['rp_max'], setting_rppi['n_rp_bins']+1, endpoint=(True))
    #         else:
    #             r_bins = np.linspace(setting_rppi['rp_min'], setting_rppi['rp_max'], setting_rppi['n_rp_bins']+1)
    #         setting_rppi['edges_rppi'] = (r_bins, np.linspace(-setting_rppi['pimax'], setting_rppi['pimax'], 2*setting_rppi   ['pimax']+1))


    #     res_dict = {}
    #     com_tr = self.get_comb_tr_list(tracers)
    #     mask_tr = dict(zip(tracers, [cats['TRACER'] == tr for tr in tracers]))
        
    #     for tr in com_tr:
    #         if verbose:
    #             self.logger.info(f'Compute wp for {tr}...')
    #             time1 = time.time()

    #         if (self.cosmo is not None) & (self.args['clustering_settings']['rsd']):
    #             vsmear_0, vsmear_1 = self.get_vsmear(tr[0], mask_tr[tr[0]].sum(), verbose=verbose), self.get_vsmear(tr[1], mask_tr[tr[1]].sum(), verbose=verbose)
    #             pos1 = apply_rsd(cats[mask_tr[tr[0]]], self.z_simu, self.boxsize, self.cosmo, self.H_0, self.args['clustering_settings']['los'], vsmear_0)
    #             pos2 = apply_rsd(cats[mask_tr[tr[1]]], self.z_simu, self.boxsize, self.cosmo, self.H_0, self.args['clustering_settings']['los'], vsmear_1)
    #         else:
    #             if self.args['clustering_settings']['rsd']:
    #                 self.logger.warning('Cosmology not set, does not apply rsd')
    #             pos1 = cats[mask_tr[tr[0]]]['x']%self.boxsize, cats[mask_tr[tr[0]]]['y']%self.boxsize, cats[mask_tr[tr[0]]]['z']%self.boxsize
    #             pos2 = cats[mask_tr[tr[1]]]['x']%self.boxsize, cats[mask_tr[tr[1]]]['y']%self.boxsize, cats[mask_tr[tr[1]]]['z']%self.boxsize
            
    #         result = compute_twopoint(pos1, 'rppi', setting_rppi['edges_rppi'], self.boxsize, self.args['clustering_settings']['los'],  self.nthreads, R1R2=R1R2, pos2=pos2)
    #         if return_rppi:
    #             res_dict[f'{tr[0]}_{tr[1]}'] = result
    #         else:
    #             res_dict[f'{tr[0]}_{tr[1]}'] = result(return_sep=True, pimax=setting_rppi['pimax'])
    #         if verbose:
    #             self.logger.info(f'Done in {time.time()-time1:.3f} s')

    #     return res_dict

    def get_cross_wp(self, cats, tracers=None, pimax=None, R1R2=None, verbose=True):
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
        - The function uses `compute_twopoint` to calculate the wp for each tracer pair.

        Example
        -------
        res = get_cross_wp(cats, tracers=['LRG', 'ELG', 'QSO'])
        """

        results = self.get_cross_twopoint(cats, 'rppi', tracers, R1R2=R1R2, verbose=verbose, pimax=pimax)
        pimax = self.args['clustering_settings']['xi_rppi']['pimax'] if pimax is None else pimax
        for trs, res in (zip(results.keys(), results.values())):
            results[trs] = res(return_sep=True, pimax=pimax)
            
        return results        


    def get_cross_xiells(self, cats, tracers=None, ells=None, R1R2=None, verbose=True):
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
            Multipoles to project onto. If None, ells in `self.args['clustering_settings']['smu']['multipole_index']` are considered. Default to None.
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
        - The function uses `compute_twopoint` to calculate the 2PCF for each tracer pair.
    

        Example
        -------
        res = get_cross_xiells(cats, tracers=['LRG', 'ELG', 'QSO'])
        """
        
        results = self.get_cross_twopoint(cats, 'smu', tracers, R1R2=R1R2, verbose=verbose, ells=ells)
        ells = self.args['clustering_settings']['xi_smu']['multipole_index'] if ells is None else ells
        for trs, res in (zip(results.keys(), results.values())):
            results[trs] = res(return_sep=True, ells=ells)
            
        return results


    def get_cross_twopoint(self, cats, mode, tracers=None, R1R2=None, verbose=True, return_pycorr_obj=False, **kwargs):
        """
        Compute the two-point correlation function (2PCF) multipoles for a given mock catalog in a cubic box.

        This function computes the two-point correlation function (2PCF) and cross-correlations multipoles for pairs of tracers in the mock catalogs. 
        It calculates the 2PCF for all combinations of tracers provided, handling redshift space distortions (RSD) if enabled.

        Parameters
        ----------
        cats : dict
            A dictionary of mock catalogs where each key is a tracer and its corresponding catalog is the value 
            (e.g., 'LRG', 'ELG').
        mode : str
            The mode of the two-point correlation function to compute (e.g., 'smu', 'rppi').
        tracers : list of str
            A list of tracer names (keys in `cats`) for which the cross 2PCF should be computed. The function 
            computes the 2PCF for all pairs of tracers in the list.
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
        - The function uses `compute_twopoint` to calculate the 2PCF for each tracer pair.
    

        Example
        -------
        res = get_cross_twopoint(cats, mode='smu', tracers=['LRG', 'ELG', 'QSO'])
        """
        
        tracers = self.check_cat_tracers(cats, tracers)
        if (mode == 'wp') | ('rppi' in mode):
            mode = 'rppi'
            self.check_clustering_settings('xi_rppi')
            settings = self.args['clustering_settings'].get('xi_rppi', None)
            settings['pimax'] = kwargs.get('pimax',  settings['pimax'])
            if settings['bin_logscale']:
                r_bins = np.geomspace(settings['rp_min'], settings['rp_max'], settings['n_rp_bins']+1, endpoint=True)
            else:
                r_bins = np.linspace(settings['rp_min'], settings['rp_max'], settings['n_rp_bins']+1, endpoint=True)
            if settings.get('edges_rppi', None) is None:
                settings['edges_rppi'] = (r_bins, np.linspace(-settings['pimax'], settings['pimax'], 2*settings['pimax']+1))
            edges = settings['edges_rppi']
            
        elif (mode == 'xi_ells') | ('smu' in mode):
            mode = 'smu'
            self.check_clustering_settings('xi_smu')
            settings = self.args['clustering_settings']['xi_smu']
            if settings['bin_logscale']:
                r_bins = np.geomspace(settings['smin'], settings['smax'], settings['n_s_bins']+1, endpoint=True)
            else:
                r_bins = np.linspace(settings['smin'], settings['smax'], settings['n_s_bins']+1, endpoint=True)
            if settings.get('edges_smu', None) is None:
                settings['edges_smu'] = (r_bins, np.linspace(-settings['mu_max'], settings['mu_max'], settings['n_mu_bins']))
            edges = settings['edges_smu']
            ells = settings.get('multipole_index', [0,2])

        else:
            raise ValueError('Twopoint mode must be wp, xi_ells, xi_smu or xi_rppi')                
                
        res_dict = {}
        com_tr = self.get_comb_tr_list(tracers)
        mask_tr = dict(zip(tracers, [cats['TRACER'] == tr for tr in tracers]))
        for tr in com_tr:
            if verbose: 
                self.logger.info(f'Compute xi_{mode} for {tr}...')
                time1 = time.time()
            
            if (self.cosmo is not None) & (self.args['clustering_settings']['rsd']):
                vsmear_0, vsmear_1 = self.get_vsmear(tr[0], mask_tr[tr[0]].sum(), verbose=verbose), self.get_vsmear(tr[1], mask_tr[tr[1]].sum(), verbose=verbose)
                pos1 = apply_rsd(cats[mask_tr[tr[0]]], self.z_simu, self.boxsize, self.cosmo, self.H_0, self.args['clustering_settings']['los'], vsmear_0)
                pos2 = apply_rsd(cats[mask_tr[tr[1]]], self.z_simu, self.boxsize, self.cosmo, self.H_0, self.args['clustering_settings']['los'], vsmear_1)
            else:
                if self.args['clustering_settings']['rsd']:
                    self.logger.warning('Cosmology not set, does not apply rsd')
                pos1 = cats[mask_tr[tr[0]]]['x']%self.boxsize, cats[mask_tr[tr[0]]]['y']%self.boxsize, cats[mask_tr[tr[0]]]['z']%self.boxsize
                pos2 = cats[mask_tr[tr[1]]]['x']%self.boxsize, cats[mask_tr[tr[1]]]['y']%self.boxsize, cats[mask_tr[tr[1]]]['z']%self.boxsize

            res = compute_twopoint(pos1, mode, edges, self.boxsize, self.args['clustering_settings']['los'], self.nthreads, pos2=pos2, R1R2=R1R2)
            res_dict[f'{tr[0]}_{tr[1]}'] = res if return_pycorr_obj else res_dict[f'{tr[0]}_{tr[1]}'](return_sep=True)

            if mode == 'xi_ells':
                res_dict[f'{tr[0]}_{tr[1]}'] = res(return_sep=True, ells=ells) 
            if mode == 'wp':
                res_dict[f'{tr[0]}_{tr[1]}'] = res(return_sep=True, pimax=settings['pimax'])
            if verbose:
                self.logger.info(f'Done in {time.time()-time1:.3f} s')
        return res_dict


    def get_cross_PS(self, cats, tracers=None, verbose=True):
        """
        Compute the two-point correlation function (2PCF) multipoles for a given mock catalog in a cubic box.

        This function computes the two-point correlation function (2PCF) and cross-correlations multipoles for pairs of tracers in the mock catalogs. 
        It calculates the 2PCF for all combinations of tracers provided, handling redshift space distortions (RSD) if enabled.

        Parameters
        ----------
        cats : dict
            A dictionary of mock catalogs where each key is a tracer and its corresponding catalog is the value 
            (e.g., 'LRG', 'ELG').
        mode : str
            The mode of the two-point correlation function to compute (e.g., 'smu', 'rppi').
        tracers : list of str
            A list of tracer names (keys in `cats`) for which the cross 2PCF should be computed. The function 
            computes the 2PCF for all pairs of tracers in the list.
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
        - The function uses `compute_twopoint` to calculate the 2PCF for each tracer pair.
    

        Example
        -------
        res = get_cross_PS(cats, tracers=['LRG', 'ELG', 'QSO'])
        """
        
        tracers = self.check_cat_tracers(cats, tracers)
        
        self.check_clustering_settings('power_spectrum')
        ps_settings = self.args['clustering_settings'].get('power_spectrum', None)

        if ps_settings.get('k_edges', None) is None:
            if ps_settings['bin_logscale']:
                ps_settings['k_edges'] = np.geomspace(ps_settings['kmin'], ps_settings['kmax'], ps_settings['n_k_bins']+1, endpoint=(True))
            else:
                ps_settings['k_edges'] = np.linspace(ps_settings['kmin'], ps_settings['kmax'], ps_settings['n_k_bins']+1)
        ells = ps_settings['multipole_index']
                
        res_dict = {}
        com_tr = self.get_comb_tr_list(tracers)
        mask_tr = dict(zip(tracers, [cats['TRACER'] == tr for tr in tracers]))
        for tr in com_tr:
            if verbose: 
                self.logger.info(f'Compute PS for {tr}...')
                time1 = time.time()
            
            if (self.cosmo is not None) & (self.args['clustering_settings']['rsd']):
                vsmear_0, vsmear_1 = self.get_vsmear(tr[0], mask_tr[tr[0]].sum(), verbose=verbose), self.get_vsmear(tr[1], mask_tr[tr[1]].sum(), verbose=verbose)
                pos1 = apply_rsd(cats[mask_tr[tr[0]]], self.z_simu, self.boxsize, self.cosmo, self.H_0, self.args['clustering_settings']['los'], vsmear_0)
                pos2 = apply_rsd(cats[mask_tr[tr[1]]], self.z_simu, self.boxsize, self.cosmo, self.H_0, self.args['clustering_settings']['los'], vsmear_1)
            else:
                if self.args['clustering_settings']['rsd']:
                    self.logger.warning('Cosmology not set, does not apply rsd')
                pos1 = cats[mask_tr[tr[0]]]['x']%self.boxsize, cats[mask_tr[tr[0]]]['y']%self.boxsize, cats[mask_tr[tr[0]]]['z']%self.boxsize
                pos2 = cats[mask_tr[tr[1]]]['x']%self.boxsize, cats[mask_tr[tr[1]]]['y']%self.boxsize, cats[mask_tr[tr[1]]]['z']%self.boxsize

            k, Pk = compute_power_spectrum(pos1=pos1, pos2=pos2, nmesh=ps_settings['nmesh'], boxsize=self.boxsize, kedges=ps_settings['k_edges'], ells=ells, los=self.args['clustering_settings']['los'], resampler=ps_settings['resampler'], interlacing=ps_settings['interlacing']).poles(ell=ells, return_k=True, complex=False)

            res_dict[f'{tr[0]}_{tr[1]}'] = k, Pk

            if verbose:
                self.logger.info(f'Done in {time.time()-time1:.3f} s')
        return res_dict
    

    # def get_cross_twopoint(self, cats, mode, tracers, ells=None, R1R2=None, verbose=True):
    #     """
    #     Compute the two-point correlation function (2PCF) multipoles for a given mock catalog in a cubic box.

    #     This function computes the two-point correlation function (2PCF) and cross-correlations multipoles for pairs of tracers in the mock catalogs. 
    #     It calculates the 2PCF for all combinations of tracers provided, handling redshift space distortions (RSD) if enabled.

    #     Parameters
    #     ----------
    #     cats : dict
    #         A dictionary of mock catalogs where each key is a tracer and its corresponding catalog is the value 
    #         (e.g., 'LRG', 'ELG').
    #     mode : str
    #         The mode of the two-point correlation function to compute (e.g., 'smu', 'rppi').
    #     tracers : list of str
    #         A list of tracer names (keys in `cats`) for which the cross 2PCF should be computed. The function 
    #         computes the 2PCF for all pairs of tracers in the list.
    #     R1R2 : tuple or None, optional
    #         A tuple defining a range for R1 and R2 for the 2PCF computation. If None, the default values will be used.
    #         Defaults to None.
    #     verbose : bool, optional
    #         If True, prints progress and computation time for each pair of tracers. Defaults to True.

    #     Returns
    #     -------
    #     res_dict : dict
    #         A dictionary where the keys are the concatenated names of tracer pairs (e.g., 'LRG_ELG') and the 
    #         values are the average separations and the corresponding two-point correlation functions (2PCF) for each pair.

    #     Notes
    #     -----
    #     - The function computes the cross-correlation 2PCF for all unique pairs of tracers from the input list.
    #     - If redshift space distortions (RSD) are enabled, the positions of galaxies in the catalogs are adjusted accordingly.
    #     - The results are stored in `res_dict` with keys in the format 'tracer1_tracer2', where each value is the 2PCF 
    #     corresponding to the pair of tracers.
    #     - The function uses `compute_twopoint` to calculate the 2PCF for each tracer pair.
    

    #     Example
    #     -------
    #     res = get_cross_twopoint(cats, mode='smu', tracers=['LRG', 'ELG', 'QSO'])
    #     """
        
    #     if (mode == 'wp') | ('rppi' in mode): 
    #         return_rppi = False if mode == 'wp' else True     
    #         res_dict = self.get_cross_wp(cats, tracers, R1R2=R1R2, return_rppi=return_rppi, verbose=verbose)

    #     elif (mode == 'xi_ells') | ('smu' in mode):
    #         mode = 'smu'
    #         res_dict = self.get_cross2PCF(cats, tracers, ells=ells, R1R2=R1R2, verbose=verbose)

    #     if mode not in ['smu', 'rppi']:
    #         raise ValueError('Twopoint mode must be wp, xi_ells, smu or rppi')
        
    #     return res_dict



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
            tracer=self._tracers()
        else:
            tracer = tracer if isinstance(tracer, list) else [tracer]
        colors = {'ELG': 'deepskyblue', 'QSO': 'seagreen', 'LRG': 'red'}
        for tr in tracer:
            pc, ps, pab =self.__init_hod_param(tr)
            fig = plot_HOD(pc, ps, self._fun_cHOD[tr], self._fun_sHOD[tr], label=tr, fig=fig, show=False, color=colors[tr] if tr in colors.keys() else None)
        plt.show()


    def plot_HMF(self, cats, show_sat=False, range=(10.5, 15.2), tracer=None, inital_HMF=None):
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
            tracer=self._tracers()
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
        plt.ylabel(r'd$N$/d$M_{h}$ [$h$/Mpc$^3$]')
        plt.xlabel(r'$\log(M_h\ [M_{\odot}])$')
        plt.xlim(range[0]-0.1, range[1]+0.1)
        plt.legend(handles=handles, loc='upper right')
        plt.tight_layout()
        plt.show()

    def plot_initial_HMF(self, save_fn=None, show=False):
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
    

    def update_new_param(self, new_params, name_param, verbose=False):
        """
        Helper function to update the parameters in the argument dictionary
        """
        name_param_tr = {}

        if not set(self._tracers()) == set(self.args['fit_param']['priors'].keys()):
            raise ValueError('The defined tracers ({}) does not correspond to tracers defined in the priors({})'.format(self._tracers(), self.args['fit_param']['priors'].keys()))

        for tr in self._tracers():
            name_param_tr[tr] = [x.split(f'_{tr}')[0] for x in name_param if tr in x]
            
        for i, new_p in enumerate(new_params):
            for tr in self._tracers():
                idx = np.where([tr in vv for vv in new_p.dtype.names])[0].tolist()
                tr_par = list(name_param[i] for i in idx)

                self.args[tr].update(dict(zip(name_param_tr[tr], new_p[tr_par][0])))
                if 'assembly_bias' in self.args['fit_param']['priors'][tr].keys():
                    for var in self.args['fit_param']['priors'][tr]['assembly_bias'].keys():
                        self.args[tr]['assembly_bias'][var] = [new_p[f'ab_{var}_cen_{tr}'][0], new_p[f'ab_{var}_sat_{tr}'][0]]
                if verbose:
                    self.logger.debug(f"{tr} {[(var, self.args[tr][var]) for var in self.args['fit_param']['priors'][tr].keys()]}")


    def get_param_and_prior(self):
        """
        Helper function to return the list of parameters and their priors
        """
        priors = {}
        priors_array = []
        if not hasattr(self, '_tracer_to_fit'):
            self._initialize_fit_params()
        for tr in self._tracer_to_fit:
            priors[tr] = self.args['fit_param']['priors'][tr].copy()

            if 'assembly_bias' in priors[tr].keys():
                for var in priors[tr]['assembly_bias'].keys():
                    priors[tr][f'ab_{var}_cen'] =  priors[tr]['assembly_bias'][var][0]
                    priors[tr][f'ab_{var}_sat'] = priors[tr]['assembly_bias'][var][1]
                priors[tr].pop('assembly_bias')
            priors_array += list(priors[tr].values())
        name_param = [f'{var}_{tr}' for tr in priors.keys() for var in priors[tr].keys()]

        mask = [isinstance(x, list) and len(x) == 2 for x in priors_array]
        if not np.all(mask):
            raise ValueError("Priors should be 1D arrays of shape (2,) for lower and upper bounds. Please check the priors for parameter {}.".format(np.array(name_param)[~np.array(mask)]))
        return name_param, priors_array

    
    @staticmethod
    def get_comb_tr_list(tracers):
        """
        Helper function to return the list of tracer combinations
        """
        if isinstance(tracers, str):
            tracers = [tracers]
        return np.vstack([np.array(np.meshgrid(tracers,tracers)).T.reshape(-1, len(tracers)).flatten().reshape(len(tracers),len(tracers),2)[i,i:] for i in range(len(tracers))])
    

    def compute_stats(self, cat, stat=None, tracers=None, verbose=False):
        """
        Compute clustering statistics for a given mock catalog.
        Parameters
        ----------
        cat : dict
            A dictionary of mock catalogs where each key is a tracer and its corresponding catalog is the value 
            (e.g., 'LRG', 'ELG').
        stat : str or list of str, optional
            A list of clustering statistics to compute. If None, all available statistics will be computed.
            Available statistics include 'xi_rppi', 'wp', 'xi_smu', 'xi_ells',
            'delta_sigma', 'CIC', and 'power_spectrum'. Individual correlation
            multipoles can be selected with names such as 'xi0', 'xi2', or
            'xi4', as in plot_stats. The order must be configured in
            clustering_settings['xi_smu']['multipole_index']. Names are
            case-insensitive.
        tracers : list of str, optional
            A list of tracer names (keys in `cat`) for which the clustering statistics should be computed. If None, all tracers in `cat` will be used.
        verbose : bool, optional
            Whether to print progress messages during execution. Default is False.
        
        Returns
        -------
        result : dict
            A dictionary containing the computed clustering statistics for the specified tracers. The keys are the names of the statistics, and the values are the corresponding computed results.
            Individual multipoles have entries such as
            result['xi0']['LRG_LRG'] = (s, xi0), with one-dimensional values.
        """

        from .plot_utils import _expand_stats, _resolve_name

        if stat is None:
            stat = get_list_stat()
        elif isinstance(stat, str):
            stat = [stat]
        available = get_list_stat()
        resolved, xi_stats, miss_stat = [], [], []
        for name in stat:
            key = _resolve_name(name, available)
            if key is None and name.lower().startswith('xi') and name[2:].isdigit():
                key = _expand_stats(self, [name])[0]
                xi_stats.append(key)
            if key is None:
                miss_stat.append(name)
            else:
                resolved.append(key)
        if miss_stat:
            raise ValueError(f'Statistics {miss_stat} not implemented. Choose among {available} '
                             'or configured xi multipoles such as xi0 and xi2.')
        stat = resolved
        
        result = {}
        if ('xi_rppi' in stat) | ('wp' in stat):
            result['xi_rppi'] = {}
            res = self.get_cross_twopoint(cat, 'rppi', tracers=tracers, verbose=verbose, return_pycorr_obj=True)
            for tr in res.keys():
                result['xi_rppi'][tr] = res[tr](return_sep=True)
            if 'wp' in stat:
                result['wp'] = {}
                for tr in result['xi_rppi'].keys():
                    result['wp'][tr] = res[tr](return_sep=True, pimax=self.args['clustering_settings']['xi_rppi']['pimax'])
        
        if 'xi_smu' in stat or 'xi_ells' in stat or xi_stats:
            res = self.get_cross_twopoint(cat, 'smu', tracers=tracers, verbose=verbose, return_pycorr_obj=True)
            if 'xi_smu' in stat or 'xi_ells' in stat:
                result['xi_smu'] = {}
                for tr in res.keys():
                    result['xi_smu'][tr] = res[tr](return_sep=True)
            if 'xi_ells' in stat:
                result['xi_ells'] = {}
                for tr in res.keys():
                    result['xi_ells'][tr] = res[tr](return_sep=True, ells=self.args['clustering_settings']['xi_smu']['multipole_index'])
            for name in dict.fromkeys(xi_stats):
                result[name] = {tr: estimator(return_sep=True, ells=int(name[2:]))
                                for tr, estimator in res.items()}
        
        if 'delta_sigma' in stat:
            if self.field_particles is None:
                self.logger.info('Field particles not set, cannot compute ΔΣ(R)')
            else:
                result['delta_sigma'] = self.get_delta_sigma(cat, tracers=tracers, verbose=verbose, return_dic=True)

        if 'CIC' in stat:
            result['CIC'] = self.get_CIC(cat, tracers=tracers, verbose=verbose, return_dic=True)
            
        if 'power_spectrum' in stat:
            result['power_spectrum'] = self.get_cross_PS(cat, tracers=tracers, verbose=verbose)
            
        return result




    def compute_training(self, training_points=None, path_to_save_point=None, start_point=0, seed=None, verbose=False, overwrite=False, **kwargs):
        
        """
        Generate and save training data for HOD model fitting by sampling parameter sets 
        (training points), generating mock catalogs, and computing clustering statistics.

        This method automates the process of training data generation for halo occupation distribution (HOD) 
        modeling. It evaluates HOD parameter samples, generates mock galaxy catalogs, computes desired clustering 
        statistics (e.g., wp, xi), and stores the results for each training point on disk.

        Parameters
        ----------
        training_points : structured array or None, optional
            Array of training points with named fields corresponding to HOD parameters. If None, 
            training points will be generated using `genereate_training_points`.
        path_to_save_point : str, optional
            Path to the directory where the training points will be saved. If None, the default path will be used.
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
        self.compute_training(verbose=True)
        
        """

        from .fit_functions import genereate_training_points
        self._initialize_fit_params(**kwargs)
        emu_settings = self.args['fit_param']['emulator']
        path_to_save_point = emu_settings['path_to_training_point'] if path_to_save_point is None else path_to_save_point
        sampling_type = emu_settings['sampling_type']
        if not set(self._tracers()) == set(self.args['fit_param']['priors'].keys()):
            raise ValueError('The defined tracers ({}) does not correspond to tracers defined in the priors({})'.format(self._tracers(), self.args['fit_param']['priors'].keys()))
        
        
        if training_points is None:
            
            # if test_set:
            #     self.logger.info(f"Creating {sampling_type} sample for test set...")
            #     path_to_save_point = path_to_test_set if path_to_test_set is not None else os.path.join(emu_settings['path_to_training_point'], 'test_set')
            #     sampling_type = emu_settings.get('test_sampling_type', 'lhs')
            #     training_points = genereate_training_points(emu_settings.get('N_test_points', 10), self.name_params, self.priors, sampling_type=sampling_type, path_to_save_training_point=path_to_save_point, rand_seed=None)
            # else:
            self.logger.info(f"Creating {sampling_type} sample for training set...")
            training_points = genereate_training_points(emu_settings['N_training_points'], self.name_params, self.priors, sampling_type=sampling_type, path_to_save_training_point=path_to_save_point, rand_seed=None)
        tracers = self._tracer_to_fit
        
        if len(self.name_params) != len(training_points.dtype.names):
            raise ValueError('The training sample shape ({}) does not correspond to the number of parameters ({})'.format(len(self.name_params), len(training_points.dtype.names)))
                
        if verbose:
            self.logger.info('Run training sample')

        name_param_tr = {}
        for tr in tracers:
            name_param_tr[tr] = [x.split(f'_{tr}')[0] for x in self.name_params if tr in x]

        # stats = 
        for nb_point, param in enumerate(training_points):
            if (not os.path.exists(os.path.join(path_to_save_point, 'train_hod_{}.npy'.format(nb_point+start_point)))) | overwrite:
                start = time.time()     
                result = {}
                for tr in tracers:
                    # var_name_tr = np.array(training_points.dtype.names)[np.array([tr in var for var in training_points.dtype.names])].tolist()
                    idx = np.where([tr in vv for vv in self.name_params])[0].tolist()
                    tr_par = list(self.name_params[i] for i in idx)
                    self.args[tr].update(dict(zip(name_param_tr[tr], param[tr_par])))
                    if 'assembly_bias' in self.args['fit_param']['priors'][tr].keys():
                        for var in self.args['fit_param']['priors'][tr]['assembly_bias'].keys():
                            self.args[tr]['assembly_bias'][var] = [param[f'ab_{var}_cen_{tr}'], param[f'ab_{var}_sat_{tr}']]

                    result[tr] = self.args[tr].copy()
                self.logger.info('Compute HOD:' + ', '.join('{}: {:.3f}'.format(tt, ttt) for tt, ttt in zip(self.name_params, param)))
                cat = self.make_mock_cat(tracers, fix_seed=seed, verbose=verbose)
                
                result.update(self.compute_stats(cat, stat=self.args['fit_param']['fit_statistics'], tracers=self._tracers(), verbose=False))
                result['comb_trs'] = self.get_comb_tr_list(self._tracers())
                result['hod_fit_param'] = param
                result['param_file'] = self.args
                np.save(os.path.join(path_to_save_point, 'train_hod_{}.npy'.format(nb_point+start_point)), result)
                self.logger.info('Point {} done {:.2f}'.format(nb_point+start_point, time.time()-start))        


    # def read_training(self, data, inv_cov2):
        
    #     """
    #     Loads and processes HOD training samples, computes chi² statistics for each sample 
    #     against a target dataset, and returns a structured array for Gaussian Process training.

    #     Parameters
    #     ----------
    #     data : array_like
    #         Observed data vector (e.g., wp or xi measurements) to compare against model predictions.

    #     inv_cov2 : ndarray
    #         Inverse of the covariance matrix used in chi² computation.

    #     Returns
    #     -------
    #     training_set : structured ndarray
    #         Structured array where each row corresponds to a training point, including:
    #         - HOD parameters
    #         - Mean chi² value for the realizations
    #         - Uncertainty on chi² (standard deviation / sqrt(N_real))
        
    #     Notes
    #     -----
    #     - Reads all training `.npy` files from `self.args['fit_param']['path_to_training_point']` with the given sampling type.
    #     - Applies covariance matrix adjustments if requested.
    #     - Supports either 'wp', 'xi', or both statistics depending on `self.args['fit_param']['fit_type']`.
    #     - Combines model realizations by flattening tracer combinations and statistics into a single vector.
    #     - Computes chi² using the `compute_chi2()` utility, which is assumed to match the data/model shape.

    #     Example
    #     -------
    #     >>> train_set = model.read_training(observed_data, inv_cov2)
    #     """

    #     from HODDIES.fits_functions_old import compute_chi2

    #     print('Read training sample...', flush=True)
    #     files = glob.glob(os.path.join(self.args['fit_param']["path_to_training_point"], '{}_*.npy'.format(self.args['fit_param']['sampling_type'])))
    #     files.sort()
    #     for ii,file in enumerate(files):
    #         res_param =  np.load(file, allow_pickle=True)[()]
    #         if ii == 0:
    #             name_arr = list(res_param['hod_fit_param'].dtype.names) + ['chi2', 'chi2_err']
    #             training_set = np.zeros((len(files),len(name_arr)))
    #         stats = ['wp', 'xi'] if ('wp' in self.args['fit_param']["fit_type"]) & ('xi' in self.args['fit_param']["fit_type"]) else ['wp'] if ('wp' in self.args['fit_param']["fit_type"]) else ['xi']
    #         res = {}
    #         comb_trs = res_param[stats[0]][0].keys() 
    #         nreal = len(res_param[stats[0]])
    #         res = [np.hstack([np.hstack([np.hstack(res_param[stat][i][comb_tr][1])for stat in stats]) for comb_tr in comb_trs]) for i in range(nreal)]

    #         chi2 = np.mean([compute_chi2(model_arr, data, inv_Cov2=inv_cov2) for model_arr in res])
    #         chi2_err = np.std([compute_chi2(model_arr, data, inv_Cov2=inv_cov2) for model_arr in res])/np.sqrt(nreal)
    #         training_set[ii] = np.hstack((res_param['hod_fit_param'].tolist(),chi2,chi2_err))

    #     training_set.dtype=[(name, dt) for name, dt in zip(name_arr, ['float64']*len(name_arr))]
    #     return training_set
    
        
    # def run_gp_mcmc(self, training_set, niter, logchi2=True,
    #                     nb_points=1, remove_edges=0.9,
    #                     random_state=None, verbose=True):
        
    #     """
    #     Performs Gaussian Process Regression (GPR) on a training set, runs MCMC sampling over 
    #     the GPR-predicted posterior, and returns the next suggested parameter point(s) for exploration.

    #     This function enables Bayesian optimization for halo model fitting by building a GP emulator
    #     on existing training data, sampling from the GP posterior using MCMC, and identifying 
    #     the most promising regions in parameter space.
    #     Detail of the method in arxiv:2302.07056

    #     Parameters
    #     ----------
    #     training_set : structured array
    #         Training data containing parameters and corresponding chi² values (and uncertainties).

    #     niter : int
    #         Current iteration index (used for file naming and logging).

    #     logchi2 : bool, optional
    #         If True, the GP models log(chi²). Default is True.

    #     nb_points : int, optional
    #         Number of new points to return from the GP+MCMC sampling. Default is 1.

    #     remove_edges : float, optional
    #         Factor to shrink prior boundaries when enforcing parameter limits. Values egal to 1 
    #         keep the prior boundaries. Default is 0.9.

    #     random_state : int or None, optional
    #         Seed for reproducibility. Default is None.

    #     verbose : bool, optional
    #         If True, print progress and diagnostics. Default is True.

    #     Returns
    #     -------
    #     new_points : ndarray
    #         Array of shape (nb_points, n_parameters) with newly suggested parameter values.
        

    #     Notes
    #     -----
    #     - Trains a GP model using scikit-learn's `GaussianProcessRegressor`.
    #     - Runs MCMC sampling using `emcee` or `zeus`. Default sampler is emcee.
    #     - Logs GPR and MCMC diagnostics to `output_GP_*.txt`.
    #     - Saves the full sampled chain with GP predictions to `chains/chain_*.txt`.
    #     - The GP kernel is configured based on `self.args['fit_param']['kernel_gp']`. Default kernel is Matern 5/2.
    #     - Trained GP model and MCMC output are saved for post-analysis and reproducibility.
    #     - During MCMC, parameter boundaries are enforced via a likelihood mask.
    #     - GPR score, prediction at the prior mean, and best predicted chi² are logged.

    #     Raises
    #     ------
    #     ValueError
    #         If an unsupported GP kernel or sampler is specified.

    #     Example
    #     -------
    #     >>> new_pts = model.run_gp_mcmc(training_data, niter=5, nb_points=3, logchi2=True)

    #     """

    #     priors = self.args['fit_param']['priors']
    #     priors_array = np.vstack([list(priors[tr].values()) for tr in self._tracers()])
    #     nvar = len(priors_array)
    #     name_param = training_set.dtype.names[:-2]
    #     ranges = np.hstack((priors_array, np.mean(priors_array, axis=1).reshape(nvar,-1), np.diff(priors_array, axis=1)))

    #     dir_output_file = self.args['fit_param']['dir_output_fit']
    #     fit_name = self.args['fit_param']['fit_name']

    #     arr_training = np.concatenate(training_set.tolist(), axis=0).T
        
    #     os.makedirs(dir_output_file, exist_ok=True)
    #     if logchi2:
    #         X_train, Y_train, Y_err = arr_training[:nvar].T, np.log(arr_training[-2]), arr_training[-1]/arr_training[-2]  
    #     else:
    #         X_train, Y_train, Y_err = arr_training[:nvar].T, arr_training[-2], arr_training[-1]

    #     length_scale = np.ones(nvar)
    #     if self.args['fit_param']['length_scale_bounds'] == "fix":
    #         length_scale = length_scale

    #     if  self.args['fit_param']['kernel_gp'] == 'RBF':
    #         kernel = 1.0 * skg.kernels.RBF(length_scale=length_scale,
    #                                         length_scale_bounds=self.args['fit_param']['length_scale_bounds'])
    #     elif self.args['fit_param']['kernel_gp'] == 'Matern_52':
    #         kernel = 1.0 * skg.kernels.Matern(length_scale=length_scale,
    #                                             length_scale_bounds=self.args['fit_param']['length_scale_bounds'], nu=5/2)
            
    #     else:
    #         raise ValueError('Only RBF or Matern_52 Kernel are available not {}'.format(self.args['fit_param']['kernel_gp']))
            
    #     if verbose:
    #         print(f"Running GPR iteration {niter}...", flush=True)
    #         start = time.time()

    #     gp = skg.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10,
    #                                         alpha=Y_err**2, random_state=random_state).fit(X_train, Y_train)
    #     if verbose:
    #         print(
    #             f"GPR computed took {time.strftime('%H:%M:%S',time.gmtime(time.time() - start))}")
    #         print('#score=', gp.score(X_train, Y_train), flush=True)
    #         print("#", gp.kernel_.get_params(), flush=True)

    #     def likelihood(x):
    #         if logchi2:
    #             L = -np.exp(gp.predict(x.reshape(-1, nvar)))/2  
    #         else:
    #             L = -gp.predict(x.reshape(-1, nvar))/2
    #         # print(L,x)
    #         cond = np.abs(x-ranges[:, 2]) < (ranges[:, 3]/2)*remove_edges
    #         # print(cond)
    #         if cond.all():
    #             return L
    #         else:
    #             return -np.inf

    #     p0 = np.random.uniform(0, 1, (self.args['fit_param']['nwalkers'], nvar))
    #     for k in range(nvar):
    #         p0[:, k] = (p0[:, k]-0.5)*ranges[k, 3] * 0.8 + ranges[k, 2]  # 0.8 edges
        
    #     if verbose:
    #         print(f"Run MCMC for iteration {niter}...", flush=True)
    #         start = time.time()
    #     if self.args['fit_param']['sampler'] == "zeus":
    #         sampler_mcmc = zeus.EnsembleSampler(
    #             self.args['fit_param']['nwalkers'], nvar, likelihood)  # , args=[nvar])
    #         sampler_mcmc.run_mcmc(p0, self.args['fit_param']['n_iter'])  # per walker
    #         chain = sampler_mcmc.get_chain(flat=True)
    #     elif self.args['fit_param']['sampler'] == "emcee":
    #         sampler_mcmc = emcee.EnsembleSampler(
    #             self.args['fit_param']['nwalkers'], nvar, likelihood)  # , args=[nvar])
    #         sampler_mcmc.run_mcmc(p0, self.args['fit_param']['n_iter'])  # per walker
    #         '''with Pool() as pool:
    #             sampler_mcmc = emcee.EnsembleSampler(self.args['fit_param']['nwalkers'], nvar, likelihood, pool=pool)
    #             sampler_mcmc.run_mcmc(p0, self.args['fit_param']['n_iter'])'''
    #         chain = sampler_mcmc.flatchain
    #     else: 
    #         raise ValueError('Only emcee or zeus sampler are available not {}'.format(self.args['fit_param']['sampler']))
        
    #     trimmed = chain[int(len(chain)/4):]
    #     if verbose:
    #         print(f'MCMC computed took {time.strftime("%H:%M:%S",time.gmtime(time.time() - start))}', flush=True)
        
    #     new_points = trimmed[:, :nvar][np.random.randint(len(trimmed), size=nb_points)]


    #     Gp_pred = gp.predict(trimmed, return_std=True)
    #     ind = np.where(Gp_pred[0] == Gp_pred[0].min())[0][0]
    #     pred = gp.predict(ranges[:, 2].reshape(-1, nvar),
    #                         return_std=True)  # Best fit pred
    #     if verbose:
    #         print("#Pred at fix point (mean values of each params):",
    #                 ranges[:, 2], pred, flush=True)
    #         print("#best GP prediction:",
    #                 trimmed[ind], Gp_pred[0][ind], Gp_pred[1][ind], flush=True)

    #     multi_GR = multivariate_gelman_rubin(sampler_mcmc.get_chain().transpose([1, 0, 2])[:, 2500:, :])
    #     if verbose:
    #         print("#multivariate_gelman_rubin: ", multi_GR, flush=True)

    #     res = np.hstack((trimmed, np.exp(Gp_pred[0]).reshape(len(Gp_pred[0]),1) if logchi2 else Gp_pred[0].reshape(len(Gp_pred[0]),1), (np.exp(Gp_pred[0])*Gp_pred[1]).reshape(len(Gp_pred[0]),1) if logchi2 else Gp_pred[1].reshape(len(Gp_pred[0]),1)))
        
    #     os.makedirs(os.path.join(dir_output_file, 'chains'), exist_ok=True)
    #     np.savetxt(os.path.join(dir_output_file, 'chains', f'chain_{nvar}p_{fit_name}_{niter}.txt'), res)

    #     if niter == 0:
    #         list_lenghtscale = []
    #         for i in range(nvar):
    #             list_lenghtscale.append("ls%d" % i)
            
    #         f = open(os.path.join(dir_output_file,
    #                                 f'output_GP_{nvar}p_{fit_name}.txt'), "w")
            
    #         f.write("N_iter GPscore GP_predfix Er_predfix BestPred Er_BestPred multivariate_gelman_rubin "
    #                 + ' '.join(map(str, name_param))+" "
    #                 + ' '.join(map(str, list_lenghtscale))+"\n")
    #         f.write(str(niter)+" "+str(gp.score(X_train, Y_train))+" "
    #                 + str(pred[0][0])+" "+str(pred[1][0])+" "
    #                 + str(Gp_pred[0][ind])+" "
    #                 + str(Gp_pred[1][ind])+" " + str(multi_GR)+" "
    #                 + ' '.join(map(str, trimmed[ind]))+" "
    #                 + ' '.join(map(str, gp.kernel_.get_params(False)["k2"].length_scale))+"\n")
    #         f.close()
    #     else:
    #         f = open(os.path.join(dir_output_file,
    #                                 f'output_GP_{nvar}p_{fit_name}.txt'), "a")
            
    #         f.write(str(niter)+" "+str(gp.score(X_train, Y_train))+" "
    #                 + str(pred[0][0])+" "+str(pred[1][0])+" "
    #                 + str(Gp_pred[0][ind])+" "
    #                 + str(Gp_pred[1][ind])+" " + str(multi_GR)+" "
    #                 + ' '.join(map(str, trimmed[ind]))+" "
    #                 + ' '.join(map(str, gp.kernel_.get_params(False)["k2"].length_scale))+"\n")
    #         f.close()

    #     return new_points
    

    # def run_fit(self, data_arr, inv_Cov2, training_point,
    #             resume_fit=False, verbose=True):
        
    #     """
    #     Execute the Gaussian Process MCMC fitting routine for HOD parameter inference.

    #     This method performs iterative Gaussian Process-driven MCMC sampling to explore the
    #     Halo Occupation Distribution (HOD) parameter space, fitting mock catalog outputs to observed
    #     clustering statistics such as the 2-point correlation function.

    #     For methodological details, see: https://arxiv.org/abs/2302.07056

    #     Parameters
    #     ----------
    #     data_arr : array_like
    #         Observed data vector used in chi-squared comparisons (e.g., wp, xi).

    #     inv_Cov2 : ndarray
    #         Inverse of the covariance matrix used in the chi-squared computation.
    #         Must match the dimensionality of `data_arr`.

    #     training_point : structured ndarray
    #         Existing training sample including HOD parameters and chi-squared values,
    #         used to condition the GP model.

    #     resume_fit : bool, optional
    #         If True, resumes from a previously saved fit by loading logs and chains.
    #         Default is False.

    #     verbose : bool, optional
    #         If True, displays detailed iteration-level logs. Default is True.

    #     Returns
    #     -------
    #     None
    #         All fitting results are saved to disk. No return value.

    #     Notes
    #     -----
    #     - Creates and updates files under `dir_output_fit`, including:
    #         - `*.txt` logs of sampled parameter values and chi² results
    #         - Chains of samples in `chains/` directory
    #         - Diagnostic metrics such as KL divergence
    #     - Calls the following key internal methods:
    #         - `make_mock_cat()`: to generate mock catalogs
    #         - `get_cross_wp()`, `get_cross2PCF()`: for 2PCF computation
    #         - `compute_chi2()`: to evaluate model-data fit
    #         - `run_gp_mcmc()`: for parameter sampling via GP-MCMC
    #     - Convergence is optionally monitored via KL divergence, but the stopping criterion is commented out.
    #     - Handles both projected (wp) and full-space (xi) correlation functions depending on `fit_type`.
    #     - Assumes the availability of `emcee` or `zeus` samplers for MCMC.
    #     - Results are appended to an evolving training set across iterations.

    #     Example
    #     -------
    #     >>> model.run_fit(data_arr, inv_cov2, training_set, resume_fit=True)
    #     """
    #     import pandas as pd
    #     from .fits_functions_old import compute_chi2
    #     if self.args['fit_param']['sampler'] == "zeus":
    #         import zeus
    #     elif self.args['fit_param']['sampler'] == "emcee":
    #         import emcee 
    #     else: 
    #         raise ValueError('Only emcee or zeus sampler are available not {}'.format(self.args['fit_param']['sampler']))
        
    #     import sklearn.gaussian_process as skg

    #     nmock = self.args['fit_param']['nb_real']
    #     dir_output_file= self.args['fit_param']['dir_output_fit']
    #     fit_name = self.args['fit_param']['fit_name']
    #     priors = self.args['fit_param']['priors']
    #     priors_array = np.vstack([list(priors[tr].values()) for tr in self._tracers()])
    #     nvar = len(priors_array)  
    #     arr_dtype = training_point.dtype

    #     iter = 0
    #     if resume_fit & os.path.exists(os.path.join(dir_output_file, f"{nvar}p_{fit_name}.txt")):
    #         output_point = pd.read_csv(os.path.join(
    #             dir_output_file, f"{nvar}p_{fit_name}.txt"), sep=" ", comment="#")
            
    #         training_point = np.concatenate((np.array(training_point.tolist()).reshape(len(training_point), -1), output_point[list(training_point.dtype.names)].values))
    #         training_point.dtype = arr_dtype

    #         iter = output_point["N_iter"].loc[len(output_point)-1]+1
    #         p = np.loadtxt(os.path.join(dir_output_file, 'chains',
    #                                     f'chain_{nvar}p_{fit_name}_{iter-1}.txt'))[:, :nvar]
    #         D_kl = 10
    #         if verbose:
    #             print("#resume fit at iteration ", iter, "len param point ", len(training_point), flush=True)
                
    #     print("Run gpmcmc...", flush=True)
    #     for j in range(iter, self.args['fit_param']['n_calls']):
    #         if verbose:
    #             print(f'Iteration {j}...', flush=True)
    #             time_compute_mcmc = time.time()

    #         new_params = self.run_gp_mcmc(training_point, j, logchi2=self.args['fit_param']['logchi2'],
    #                     nb_points=1, remove_edges=0.9,
    #                     random_state=None, verbose=True)


    #         if verbose:
    #             print("#time_compute_gpmcmc =", time.time() - time_compute_mcmc, flush=True)

    #         ### Test de Kullback Leibler
    #         D_kl1 = 10
    #         if j > 0:
    #             if j == 1:
    #                 q = np.loadtxt(os.path.join(dir_output_file, 'chains', f'chain_{nvar}p_{fit_name}_{0}.txt'))[:, :nvar]
    #                 p = np.loadtxt(os.path.join(dir_output_file, 'chains', f'chain_{nvar}p_{fit_name}_{1}.txt'))[:, :nvar]
    #                 D_kl = np.array([])
    #             else:
    #                 q = p
    #                 p = np.loadtxt(os.path.join(dir_output_file, 'chains', f'chain_{nvar}p_{fit_name}_{j}.txt'))[:, :nvar]
    #             n_dim = nvar
    #             cov_q = np.cov(q.T)
    #             cov_p = np.cov(p.T)
    #             inv_cov_q = np.linalg.inv(cov_q)
    #             mean_q = np.mean(q, axis=0)
    #             mean_p = np.mean(p, axis=0)
    #             D_kl1 = 0.5 * (np.log10(np.linalg.det(cov_q) / np.linalg.det(cov_p)) - n_dim + np.trace(np.matmul(
    #                 inv_cov_q, cov_p)) + np.matmul((mean_q - mean_p).T, np.matmul(inv_cov_q, (mean_q - mean_p))))
    #             D_kl = np.append(D_kl1, D_kl)
    #             # print (j, D_kl)
    #             # if len(D_kl) > 5:
    #             #     if (D_kl[-5:] < 0.1).all():
    #             #         sys.exit("Procedure converged at iteration %d!" % j)

    #         #Compute chi2
    #         new_train_point = np.zeros((len(new_params), nvar+2))

    #         new_params.dtype = [(name, dt) for name, dt in zip(training_point.dtype.names, ['float64']*nvar)]

    #         if verbose:
    #             print("#run old parralel chi2 points", new_params, len(new_params))

    #         if j == 0:
    #             f = open(os.path.join(dir_output_file, f"{nvar}p_{fit_name}.txt"), "w")
    #             f.write("N_iter "+' '.join(map(str, training_point.dtype.names)) + " D_kl1\n")
    #             f.close()

    #         for i, new_p in enumerate(new_params):
    #             for tr in self._tracers():
    #                 for var in self.args['fit_param']['priors'][tr].keys():
    #                     self.args[tr][var] = new_p['{}_{}'.format(var, tr)][0]
    #                 if verbose:
    #                     self.logger.debug(f"{tr} {[(var, self.args[tr][var]) for var in self.args['fit_param']['priors'][tr].keys()]}")
                
    #             time_function_compute_parralel_chi2 = time.time()
    #             print(f'Run {nmock} galaxy catalog for iteration {j}', flush=True)
    #             cats = [self.make_mock_cat(self._tracers(), verbose=False) for jj in range(nmock)]

    #             print(f'Time to compute {nmock} cats : {time.strftime("%H:%M:%S",time.gmtime(time.time() - time_function_compute_parralel_chi2))}', flush=True)


    #             time_function_compute_parralel_chi2 = time.time()
    #             print('Run 2PCF...', flush=True)
    #             result = {}
    #             if 'wp' in self.args['fit_param']["fit_type"]:
    #                 result['wp']= [self.get_cross_wp(cats[i], tracers=self._tracers(), verbose=False) for i in range(nmock)]
    #             if 'xi' in self.args['fit_param']["fit_type"]:
    #                 result['xi'] = [self.get_cross2PCF(cats[i], tracers=self._tracers(), verbose=False) for i in range(nmock)]
    #             if verbose:
    #                 print("#Time to compute 2PCFs =", time.strftime("%H:%M:%S", time.gmtime(time.time()-time_function_compute_parralel_chi2)), flush=True)
                    

    #             stats = ['wp', 'xi'] if ('wp' in self.args['fit_param']["fit_type"]) & ('xi' in self.args['fit_param']["fit_type"]) else ['wp'] if ('wp' in self.args['fit_param']["fit_type"]) else ['xi']
    #             res = {}
    #             comb_trs = result[stats[0]][0].keys() 
    #             res = [np.hstack([np.hstack([np.hstack(result[stat][i][comb_tr][1])for stat in stats]) for comb_tr in comb_trs]) for i in range(nmock)]
    #             #res_std = np.std(res, axis=0)            
    #             #iCov2 = inv_Cov2*res_std*res_std[:, None]
                

    #             chi2 = np.mean([compute_chi2(model_arr, data_arr, inv_Cov2=inv_Cov2) for model_arr in res])
    #             chi2_err = np.std([compute_chi2(model_arr, data_arr, inv_Cov2=inv_Cov2) for model_arr in res])/np.sqrt(nmock)

    #             new_train_point[i] = np.hstack((new_params[i].tolist()[0], chi2, chi2_err))
    #             if verbose:
    #                 print('#### NEW ADDED POINT:', ' '.join(map(str, new_params[i].tolist()[0])), chi2, chi2_err, D_kl1, flush=True)

    #             f = open(os.path.join(dir_output_file, f"{nvar}p_{fit_name}.txt"), "a")
    #             f.write(str(str(j)+" "+' '.join(map(str, new_params[i].tolist()[0])))+" "+str(chi2)+" "+str(chi2_err)+" "+str(D_kl1)+"\n")
    #             f.close()

    #         new_train_point.dtype = arr_dtype
    #         training_point = np.vstack((training_point, new_train_point))
    #         if verbose:
    #             print(f'Iteration {j} done, took {time.strftime("%H:%M:%S",time.gmtime(time.time()-time_compute_mcmc))}', flush=True)

    
    # def initialize_fit(self, data_vec=None, inv_cov2=None, diag_err=None, add_poisson_noise=True, nmocks_std=20, **kwargs):

    #     from HODDIES.fits_functions_old import load_desi_data, get_corr_small_boxes
    #     from pycorr import utils

    #     if not set(self._tracers()) == set(self.args['fit_param']['priors'].keys()):
    #         raise ValueError('The defined tracers ({}) does not correspond to tracers defined in the priors({})'.format(self._tracers(), self.args['fit_param']['priors'].keys()))

    #     # self.args['fit_param']['pimax'] = self.args['2PCF_settings']['pimax']
    #     # self.args['fit_param']['multipole_index'] = self.args['2PCF_settings']['multipole_index']
    #     # self.args['fit_param']['z_simu'] = self.z_simu
    #     # self.args['fit_param'].update(kwargs)
        
    #     if self.args['fit_param']['use_desi_data']:
    #         data_dic = load_desi_data(self.args['fit_param'], self._tracers(), load_cov_jk=self.args['fit_param']['load_cov_jk'])

    #         mm = [list(data_dic.keys())[i].endswith(tuple(self._tracers())) for i in range(len(data_dic.keys()))]
    #         comb_trs = [list(data_dic.keys())[i] for i in np.arange(len(data_dic.keys()))[mm].tolist()]
    #         data_vec = np.hstack([np.hstack([np.hstack(data_dic[comb_tr][stat][1]) for stat in data_dic.get(comb_tr).keys()]) for comb_tr in comb_trs])
    #         diag_err = np.hstack([np.hstack([np.hstack(data_dic[comb_tr][stat][2]) for stat in data_dic.get(comb_tr).keys()]) for comb_tr in comb_trs])

    #         if 'wp' in  data_dic['edges'].keys():
    #             self.args['2PCF_settings']['edges_rppi'] = data_dic['edges']['wp']
    #         if 'xi' in  data_dic['edges'].keys():
    #             self.args['2PCF_settings']['edges_smu'] = data_dic['edges']['xi']
    #         if self.args['fit_param']['use_vsmear']:
    #             for tr in self._tracers():
    #                 print('Apply vsmear for {} at z{}-{}'.format(tr, self.args['fit_param']['zmin'], self.args['fit_param']['zmax']), flush=True)
    #                 self.args[tr]['vsmear'] = [self.args['fit_param']['zmin'], self.args['fit_param']['zmax']]

    #         if add_poisson_noise:
    #             print('Compute poisson noise...', flush=True)

    #             cats = [self.make_mock_cat(self._tracers(), verbose=False) for jj in range(nmocks_std)]

    #             result = {}
    #             if 'wp' in self.args['fit_param']["fit_type"]:
    #                 result['wp']= [self.get_cross_wp(cats[i], tracers=self._tracers(), verbose=False) for i in range(nmocks_std)]
    #             if 'xi' in self.args['fit_param']["fit_type"]:
    #                 result['xi'] = [self.get_cross2PCF(cats[i], tracers=self._tracers(), verbose=False) for i in range(nmocks_std)]

    #             stats = ['wp', 'xi'] if ('wp' in self.args['fit_param']["fit_type"]) & ('xi' in self.args['fit_param']["fit_type"]) else ['wp'] if ('wp' in self.args['fit_param']["fit_type"]) else ['xi']

    #             comb_trs = result[stats[0]][0].keys() 
    #             std_poisson = np.std([np.hstack([np.hstack([np.hstack(result[stat][i][comb_tr][1])for stat in stats]) for comb_tr in comb_trs]) for i in range(nmocks_std)], axis=0)
    #             print('Done', flush=True)
    #         else: 
    #             std_poisson = np.zeros_like(diag_err)

    #         if self.args['fit_param']['load_cov_jk']:
    #             std_2 = np.sqrt(np.diag(data_dic['cov_jk']) + std_poisson**2)
    #             corr_jk = utils.cov_to_corrcoef(data_dic['cov_jk'])
    #             inv_cov2 = np.linalg.inv(corr_jk*std_2*std_2[:,None])
    #         else:
    #             corr = get_corr_small_boxes(self.args['fit_param'], self._tracers())
    #             sig_all = np.sqrt(diag_err**2 + std_poisson**2)
    #             cov = corr*sig_all*sig_all[:,None]
    #             hartlap_fac = (len(cov)+1)/(self.args['fit_param']['nb_mocks']-1)
    #             inv_cov2 = np.linalg.inv(cov/(1-hartlap_fac))

    #             if np.isnan(diag_err).any():
    #                 mask = np.isnan(cov)
    #                 cov = np.nan_to_num(cov, nan=1)
    #                 inv_cov2 = np.linalg.inv(cov/(1-hartlap_fac))
    #                 for i, mm in enumerate(mask):
    #                     inv_cov2[i, mm] = 0
    #                 data_vec = np.nan_to_num(data_vec, nan=0)
    #                 diag_err = np.nan_to_num(diag_err, nan=0)
    #         self.data = data_vec
    #         self.inv_cov2 = inv_cov2
    #         self.sig = diag_err
    #         self.sig_model = std_poisson
    #         self.name_params, self.priors = self.get_param_and_prior()
    #         return 0

    #     elif add_poisson_noise:
    #         print('Compute poisson noise...', flush=True)

    #         cats = [self.make_mock_cat(self._tracers(), verbose=False) for jj in range(nmocks_std)]

    #         result = {}
    #         if 'wp' in self.args['fit_param']["fit_type"]:
    #             result['wp']= [self.get_cross_wp(cats[i], tracers=self._tracers(), verbose=False) for i in range(nmocks_std)]
    #         if 'xi' in self.args['fit_param']["fit_type"]:
    #             result['xi'] = [self.get_cross2PCF(cats[i], tracers=self._tracers(), verbose=False) for i in range(nmocks_std)]

    #         stats = ['wp', 'xi'] if ('wp' in self.args['fit_param']["fit_type"]) & ('xi' in self.args['fit_param']["fit_type"]) else ['wp'] if ('wp' in self.args['fit_param']["fit_type"]) else ['xi']

    #         comb_trs = result[stats[0]][0].keys() 
    #         std_poisson = np.std([np.hstack([np.hstack([np.hstack(result[stat][i][comb_tr][1])for stat in stats]) for comb_tr in comb_trs]) for i in range(nmocks_std)], axis=0)
            
    #         print('Done', flush=True)
            
    #     else:
    #         std_poisson = 0

    #     if inv_cov2 is not None:
    #         mask = inv_cov2.diagonal() == 0
    #         if mask.sum() != 0:
    #             idx = mask.sum()
    #             cov_re = np.linalg.inv(inv_cov2[idx:,idx:])
    #             sig2 = np.zeros_like(inv_cov2.diagonal())
    #             sig2[idx:] = cov_re.diagonal()
    #             diag_err = np.sqrt(sig2)
    #             sig_all = np.sqrt(diag_err**2 + std_poisson**2)
    #             # Manque add poisson noise to cov +hartlap
    #         else:
    #             sig_all = np.sqrt((np.linalg.inv(inv_cov2).diagonal()))

    #         if data_vec.size != inv_cov2.diagonal().size:
    #             raise ValueError('The lenght of the data vector ({}) does not correspond to the shape of the covariance matrix ({})'.format(data_vec.size, inv_cov2.shape))
        
    #     else:
    #         sig_all = np.sqrt(diag_err**2 + std_poisson**2)
    #         if data_vec.size != sig_all.size:
    #             raise ValueError('The lenght of the data vector ({}) does not correspond to the shape of the covariance matrix ({})'.format(data_vec.size, sig_all.size))

    #     self.data = data_vec
    #     self.inv_cov2 = inv_cov2
    #     self.sig = sig_all
    #     self.sig_model = std_poisson
    #     self.name_params, self.priors = self.get_param_and_prior()

    def _get_fit_model_name(self):
        fit_model_name = []

        for tr in self._tracer_to_fit:
            ext = '+conf' if self.args[tr]['conformity_bias'] else ''
            ext += '+exp' if ('exp_frac' in self.args['fit_param']['priors'][tr].keys()) else ''
            ext += '+' + '+'.join([f'ab_{var}' for var in self.args['fit_param']['priors'][tr]['assembly_bias'].keys()]) if 'assembly_bias' in self.args['fit_param']['priors'][tr].keys() else ''
            ext += '+nu' if ('nu' in self.args['fit_param']['priors'][tr].keys()) else ''
            ext += '+vsmear' if ('vsmear' in self.args['fit_param']['priors'][tr].keys()) else ''
            ext += '_with_zerr' if  (self.args[tr]['vsmear'] != 0)  else ''
            fit_model_name += ['{}_{}_{}p'.format(tr, self.args[tr]['HOD_model'] + ext, len(self.name_params))]
        return '_'.join(fit_model_name)

    def _initialize_fit_params(self, **kwargs):
        """
        Initialize the fitting parameters.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments of the fitting parameters to update the default settings.

        Returns
        -------
        None
        """
        self.init_logger('Fit HOD')        
        update_dic(self.args['fit_param'], kwargs)
        self._tracer_to_fit = list(self.args['fit_param']['priors'].keys())
        self.name_params, self.priors = self.get_param_and_prior()
        self._stats_to_fit = self.args['fit_param']["fit_statistics"] if isinstance(self.args['fit_param']["fit_statistics"], list) else [self.args['fit_param']["fit_statistics"]]
        self._comb_tr_list = ['_'.join(comb_tr) for comb_tr in self.get_comb_tr_list(self._tracer_to_fit)]
        self.fit_model_name = self._get_fit_model_name()

    def initialize_fit(self, data_vec, err, **kwargs):
        """
        Initialize the fitting process by setting up the data vector, error vector or covariance matrix, and preparing the fitting parameters.
        Parameters
        ----------
        data_vec : array_like
            The data vector.
        err : array_like
            The error vector or covariance matrix.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        None
        """

        fit_params = self.args['fit_param']
        update_dic(fit_params, kwargs)
        self._initialize_fit_params(**fit_params)
        
        hartlap_fac = fit_params.get('hartlap_factor', 0)
        self.data = np.asarray(data_vec)
        cov = np.asarray(err)
        if cov.ndim == 1:
            # error vector, shape (N,)
            cov = np.diag(cov**2)          # convert to covariance matrix
        elif cov.ndim == 2:
            # covariance matrix, shape (N, N)
            pass
        else:
            raise ValueError('The error provided must be either a 1D error vector or a 2D covariance matrix.')
        
        if cov.shape[0] != data_vec.size:
            raise ValueError('The lenght of the data vector ({}) does not correspond to the shape of the covariance matrix ({})'.format(data_vec.size, cov.shape))

        if fit_params['add_poisson_noise']:
            nmocks_std = fit_params.get('neval_poisson_noise', 50)
            self.logger.info(f'Compute poisson noise using {nmocks_std} mocks.')            
            # Generate mock catalogs
            cats = [self.make_mock_cat(self._tracer_to_fit, verbose=False) for jj in range(nmocks_std)]
            result = [self.compute_stats(cats[i], stat=self._stats_to_fit, tracers=self._tracer_to_fit, verbose=False) for i in range(nmocks_std)]
            std_poisson = np.std([np.hstack([np.hstack([np.hstack(result[i][stat][comb_tr][-1])for stat in self._stats_to_fit]) for comb_tr in self._comb_tr_list]) for i in range(nmocks_std)], axis=0)    
            # Can also try with cov_poisson = np.cov([np.hstack([np.hstack([np.hstack(result[i][stat][comb_tr][1])for stat in self._stats_to_fit]) for comb_tr in self._comb_tr_list]) for i in range(nmocks_std)], rowvar=False)        
            cov_poisson = np.diag(std_poisson**2)
        else:
            cov_poisson = 0

        cov = cov + cov_poisson
        self.inv_cov2 = np.linalg.inv(cov/(1-hartlap_fac))
        if np.isnan(self.inv_cov2).any():
            raise ValueError('The covariance matrix contains NaN values, please check the input covariance matrix.')

        
    def get_init_params(self, seed=None):
        """
        Get initial parameter values for the minimization.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility. Default is None.

        Returns
        -------
        init_values : array_like
            Initial parameter values.
        """

        if seed is not None:
            np.random.seed(seed)
        low, high = np.vstack(self.priors).T
        u = np.random.beta(2, 2, size=(len(low)))
        init_values = low + u * (high - low)
        return init_values
    
    
    def run_minimizer(self, init_params=None, seed=10, mpi_comm=None, **kwargs):
        """
        Run the minimization procedure to find the best-fit parameters.

        Parameters
        ----------
        init_params : array_like, optional
            Initial parameter values for the minimization. If None, random values within the prior bounds are generated. Default is None.
        seed : int, optional
            Random seed for reproducibility when generating initial parameters. Default is 10.
        mpi_comm : MPI communicator, optional
            MPI communicator for parallel execution. If None, runs in serial. Default is None.
        kwargs : dict
            Additional keyword arguments to update the minimizer settings.
        """

        from stochopy.optimize import minimize
        from .fit_functions import func_stochopy
                
        self.args['fit_param']['minimizer'].update(kwargs)
        self.logger.info('Run minimizer')
        self.logger.info('Priors: '+(', ').join([f'{nn}: {val}' for nn, val in zip(self.name_params, self.priors)]))
        #Create directory to store the results
        if self.args['fit_param']['minimizer']['save_fn']:
            filename = self.args['fit_param']['minimizer']['save_fn']
        else:
            filename = 'best_fit_result_{}.npy'.format(self.fit_model_name)
            
        os.makedirs(self.args['fit_param']['dir_output_fit'], exist_ok=True)
        path_to_save_result = os.path.join(self.args['fit_param']['dir_output_fit'], filename)

        if init_params is None:
            init_params = self.get_init_params()
        self.logger.info('First point: '+(', ').join([f'{nn}: {val}' for nn, val in zip(self.name_params, init_params)]))

        options = {}
        for key in ['maxiter', 'popsize', 'xtol']:
            options[key] = self.args['fit_param']['minimizer'][key]
        if mpi_comm is None:
            mpi_rank = 0
        else:
            mpi_rank = mpi_comm.Get_rank()
            options['workers'] = mpi_comm.Get_size()
            options['backend']= 'mpi'

        
        res = minimize(func_stochopy, args=(self, seed),
                    bounds=self.priors, x0=init_params,
                    method=self.args['fit_param']['minimizer']['method'], options=options)
        
        self.result_fit = res
        res['param_fit'] = self.args.copy()
        
        if mpi_rank==0:
            self.logger.info('Save fit result to: {}'.format(path_to_save_result))
            np.save(path_to_save_result, res)           
        return res

    

    def compute_bf_corr(self, bf_file=None, verbose=False, fix_seed=None, save_bf_cat=None, **kwargs):

        from HODDIES.fits_functions_old import compute_chi2
        name_param, priors_array = self.get_param_and_prior()
        if hasattr(self, 'result_fit'):
            self.result_fit = self.result_fit if bf_file is None else np.load(bf_file, allow_pickle=True).item()
        else:
            if bf_file is None:
                raise ValueError('No best fit file provided and no previous fit result found.')
            self.result_fit = np.load(bf_file, allow_pickle=True).item()

        new_params= np.array([self.result_fit['x']])
        
        new_params.dtype = [(name, dt) for name, dt in zip(name_param, ['float64']*len(name_param))]
        self.update_new_param(new_params, name_param)
        print('Best fit point:', *zip(name_param, self.result_fit['x']), flush=True)
        cat = self.make_mock_cat(fix_seed=fix_seed)
        result = {}
        if 'wp' in self.args['fit_param']["fit_type"]:
            result['wp'] = self.get_cross_wp(cat, tracers=self._tracers(), verbose=verbose)
        if 'xi' in self.args['fit_param']["fit_type"]:
            result['xi'] = self.get_cross2PCF(cat, tracers=self._tracers(), verbose=verbose)

        
        if hasattr(self, 'data'):
            stats = ['wp', 'xi'] if ('wp' in self.args['fit_param']["fit_type"]) & ('xi' in self.args['fit_param']["fit_type"]) else ['wp'] if ('wp' in self.args['fit_param']["fit_type"]) else ['xi']
            res = {}
            comb_trs = result[stats[0]].keys() 
            res = np.hstack([np.hstack([np.hstack(result[stat][comb_tr][1])for stat in stats]) for comb_tr in comb_trs])
            result['chi2'] = compute_chi2(res, self.data, inv_Cov2=self.inv_cov2)
        
        if save_bf_cat is not None:
            if self.args['fit_param']['use_vsmear']:
                cat[f'vsmear'] = np.zeros(cat.size, dtype=np.float32)
                for tr in self._tracers():
                    mm = cat['TRACER'] == tr
                    cat[f'vsmear'][mm] = self.get_vsmear(tr, mm.sum(), verbose=verbose)
            cat.write(save_bf_cat)
            print(f'Save best fit catalog to {save_bf_cat}', flush=True)

        return result


    def plot_bf_data(self, figsize=None, pow_sep=1, suptitle=None, suptitle_fontsize=12, fontsize=8, save_fn=None, fig=None, show=False, shift=0, max_sig = 5, fix_seed=None, add_no_vsmear=False, save_bf_cat=None, **kwargs):

        from HODDIES.fits_functions_old import load_desi_data
        from matplotlib.gridspec import GridSpec
        import matplotlib.pyplot as plt
        
        
        data_dic = load_desi_data(self.args['fit_param'], self._tracers(), multipole_index=self.args['clustering_settings']['xi_smu']['multipole_index'], pimax=self.args['clustering_settings']['xi_smu']['pimax'],**kwargs)
        self.args['clustering_settings']['xi_smu']['edges_rppi'] = data_dic['edges']['wp'] if 'wp' in self.args['fit_param']["fit_type"] else None
        self.args['clustering_settings']['xi_smu']['edges_smu'] = data_dic['edges']['xi'] if 'xi' in self.args['fit_param']["fit_type"] else None
        result_bf = self.compute_bf_corr(fix_seed=fix_seed, save_bf_cat=save_bf_cat, **kwargs)
        stats = ['wp', 'xi'] if ('wp' in self.args['fit_param']["fit_type"]) & ('xi' in self.args['fit_param']["fit_type"]) else ['wp'] if ('wp' in self.args['fit_param']["fit_type"]) else ['xi']
        comb_trs = list(result_bf[stats[0]].keys())

        if add_no_vsmear and self.args['fit_param']['use_vsmear']:
            tmp_vsmear = []
            for tr in self._tracers():
                tmp_vsmear += [self.args[tr]['vsmear']]
                self.args[tr]['vsmear'] = 0
            result_bf_no_vsmear = self.compute_bf_corr(fix_seed=fix_seed, **kwargs)
            for ii,tr in enumerate(self._tracers()):
                self.args[tr]['vsmear'] = tmp_vsmear[ii]
        else:
            result_bf_no_vsmear = None
        nb_tracers = len(comb_trs)
        ncols = len(data_dic[comb_trs[0]].keys())            

        if 'xi' in data_dic[comb_trs[0]].keys():
            ncols += 1
            ells = self.args['clustering_settings']['xi_smu']['multipole_index']
        else:
            ells = None

        default_color = {'BGS_BGS': 'yellowgreen', 'ELG_ELG': 'steelblue', 'LRG_LRG': 'orangered', 'QSO_QSO': 'seagreen', 'ELG_LRG': 'firebrick', 
                        'LRG_ELG': 'firebrick', 'QSO_ELG': 'skyblue', 'ELG_QSO': 'skyblue', 'LRG_QSO': 'peru', 'QSO_LRG': 'peru'}
        
        if fig is None:
            new_fig = True
            if figsize is None: 
                size = 9 if nb_tracers == 1 else 18 if nb_tracers == 3 else 27
                figsize=(9, size/ncols)
            fig = plt.figure(figsize=figsize)
            fig.suptitle(suptitle, fontsize=suptitle_fontsize)
            gs = GridSpec(len(comb_trs)*2, ncols, height_ratios=[ncols, len(comb_trs)]*len(comb_trs), hspace=0.0)
            
        else:   
            new_fig=False
            axes = fig.axes
        i_ax = 0
        # Setup figure and grid    

        # for lax, trs in zip(axes, comb_trs):
        for ii, trs in enumerate(comb_trs):
            ii = ii * 2
            # for col in range(len(stats)):
                
            color = default_color[trs] if kwargs.get('color') is None else kwargs['color']
            col = 0
            for corr in data_dic[trs].keys():
                
                sep_data, res_data, sig_data = data_dic[trs][corr]
                sep_m, res_model = result_bf[corr][trs]
                if result_bf_no_vsmear is not None:
                    sep_nov, res_model_nov = result_bf_no_vsmear[corr][trs]
                    
                xlabel = '$s$ [Mpc/h]' if corr == 'xi' else '$r_p$ [Mpc/h]' if corr == 'wp' else None
                ylabel = r'$s \cdot \xi_{{{:d}}}(s)$ [$\mathrm{{Mpc}}/h$]' if corr == 'xi' else r'$r_p \cdot w_p(r_p)$ [$\mathrm{{Mpc}}/h$]' if corr == 'wp' else None

                if len(res_data.shape) == 1:
                    res_data = [res_data]
                    sig_data = [sig_data]
                    res_model = [res_model]
                    if result_bf_no_vsmear is not None:
                        res_model_nov= [res_model_nov]
                
                panel_titles = ['Monopole', 'Quadrupole', 'Hexadecaople'] if corr == 'xi' else ['Projected clustering'] if corr == 'wp' else None

                for (ill, panel_title), res_m, res, sig in zip(enumerate(panel_titles), res_model, res_data, sig_data):
                    ax_main = fig.add_subplot(gs[ii, col]) if new_fig  else axes[i_ax]
                    ax_main.errorbar(sep_data, sep_data**pow_sep*res+shift ,yerr= sep_data**pow_sep*sig,fmt='.',
                                    markerfacecolor='w', zorder=0, label=f'{trs} z{self.args["fit_param"]["zmin"]}-{self.args["fit_param"]["zmax"]}', color=color)
                    ax_main.plot(sep_m, sep_m**pow_sep*res_m, color=color, alpha=0.8, lw=1.2)
                    if result_bf_no_vsmear is not None:
                        print('With-WO vsmear result ', res, res_model_nov[ill])
                        ax_main.plot(sep_nov, sep_nov**pow_sep*res_model_nov[ill], color=color, alpha=0.8, lw=1.2, ls='--', label='No vsmear')

                    if corr == 'xi':
                        ax_main.set_ylabel(ylabel.format(ells[ill]), fontsize=fontsize)
                    else: 
                        ax_main.set_ylabel(ylabel, fontsize=fontsize)

                    ax_main.grid(True)
                    ax_main.set_xscale('log')
                    if ii == 0:  ax_main.set_title(panel_title,fontsize=fontsize)
                    # ax_main.set_xlabel(xlabel, fontsize=fontsize)
                    if col == 0:
                        if corr == 'xi':
                            ax_main.set_ylabel(ylabel.format(ells[ill]), fontsize=fontsize)
                        else: 
                            ax_main.set_ylabel(ylabel, fontsize=fontsize)

                    if col == len(stats):
                        ax_main.legend(fontsize=fontsize)
                    # ax_main.tick_params(labelbottom=False)

                    # Residual plot

                    ax_res = fig.add_subplot(gs[ii + 1, col], sharex=ax_main) if new_fig else axes[i_ax+1]
                    residual = (res_m - res) / sig
                    if result_bf_no_vsmear is not None:
                        residual_nov = (res_model_nov[ill] - res) / sig
                        ax_res.plot(sep_nov, residual_nov, color=color, ls='--')
                    ax_res.axhspan(-2, 2, color='gray', alpha=0.2)
                    ax_res.axhline(0, color='black', linestyle='--')
                    ax_res.plot(sep_data, residual, color=color)
                    ax_res.set_xscale('log')
                    ax_res.set_ylim(-max_sig, max_sig)
                    ax_res.set_xlabel(xlabel, fontsize=fontsize)
                
                    if col == 0:
                        ax_res.set_ylabel(r"$\Delta/\sigma$")

                    if (ii == 0) &  (col == 0) & ('chi2' in result_bf.keys()):
                        props = dict(boxstyle='round', facecolor='w', alpha=0.5)

                        # place a text box in upper left in axes coords
                        ax_main.text(0.05, 0.95, r'$\chi^2 = {:.2f}$'.format(result_bf['chi2']), transform=ax_main.transAxes, fontsize=fontsize,
                                verticalalignment='top', bbox=props)
                    col +=1     
                    i_ax += 2
        fig.tight_layout()                  
        if save_fn: 
            fig.savefig(save_fn, facecolor='w',  bbox_inches='tight', pad_inches=0.1)
        if show:
            plt.show()
        return fig

        
    def get_lin_bias(self, tracers=None, verbose=False):
        """
        Fit the linear bias of each requested tracer over 40--80 Mpc/h.

        A single mock catalogue is generated, and each tracer's real-space
        monopole is fitted to ``b**2 * xi_linear(s, z=self.z_simu)`` using
        scipy's ``curve_fit``. The original RSD and separation-bin settings
        are restored after the calculation, including when a fit fails.

        Parameters
        ----------
        tracers : str or list of str, optional
            Tracers to fit. By default, all configured tracers are used.

        Returns
        -------
        dict
            Tracer names mapped to their fitted, nonnegative bias as floats.
            A single-tracer model also returns a dictionary.
        """

        from scipy.optimize import curve_fit
        from cosmoprimo import Fourier

        if self.cosmo is None:
            raise ValueError('A cosmology is required to compute the linear bias.')
        if tracers is None:
            tracers = self._tracers()
        if isinstance(tracers, str):
            tracers = [tracers]
        xi_lin = Fourier(self.cosmo, engine='class').pk_interpolator().to_xi()

        settings = self.args['clustering_settings']
        smu_settings = settings['xi_smu']
        rsd_tmp = settings['rsd']
        had_edges = 'edges_smu' in smu_settings
        edges_smu_tmp = smu_settings.get('edges_smu')

        def func(s, b):
            return b**2 * xi_lin(s, z=self.z_simu)

        biases = {}
        try:
            settings['rsd'] = False
            smu_settings['edges_smu'] = (np.linspace(40, 80, 41), np.linspace(-1, 1, 201))
            cat = self.make_mock_cat(tracers=tracers, verbose=verbose)
            for tracer in tracers:
                s, xi = self.get_xiells(cat, tracers=tracer, ells=0)
                valid = np.isfinite(s) & np.isfinite(xi)
                if np.count_nonzero(valid) < 2:
                    raise ValueError(f'Not enough finite correlation bins to fit the linear bias of {tracer}.')
                bias, _ = curve_fit(func, s[valid], xi[valid], p0=[2.], bounds=(0., np.inf))
                biases[tracer] = float(bias[0])
        finally:
            settings['rsd'] = rsd_tmp
            if had_edges:
                smu_settings['edges_smu'] = edges_smu_tmp
            else:
                smu_settings.pop('edges_smu', None)
        return biases

    def plot_wp_xi(self, cat, tracers=None, fig=None, show=True, figsize=(12, 3), fontsize=11, **kwargs):
        import matplotlib.pyplot as plt
        colors = {'ELG': 'deepskyblue', 'QSO': 'seagreen', 'LRG': 'red'}
        if isinstance(tracers, str):
            tracers = [tracers]
        elif tracers is None:
            tracers = np.unique(cat['TRACER'])
        
        if fig is None:
            fig,ax = plt.subplots(1,1+len(self.args['clustering_settings']['xi_smu']['multipole_index']), figsize=figsize)
        else:
            ax = fig.axes
        for tr in tracers:
            if 'color' not in kwargs.keys(): 
                kwargs['color'] = colors[tr] if tr in colors.keys() else None

            if 'label' in kwargs.keys(): 
                label = kwargs['label']
                kwargs.pop('label')
            else:
                label = tr

            rp, wp = self.get_wp(cat, tracers=tr)
            s, xi = self.get_xiells(cat, tracers=tr)
            

            ax[0].semilogx(rp,rp*wp, **kwargs)
            ax[1].semilogx(s,s*xi[0], **kwargs)
            ax[2].semilogx(s,s*xi[1], label=label, **kwargs)
            
            ax[0].set_xlabel('$r_p$ [Mpc/h]', fontsize=fontsize)
            ax[1].set_xlabel('$s$ [Mpc/h]', fontsize=fontsize)
            ax[2].set_xlabel('$s$ [Mpc/h]', fontsize=fontsize)
            ax[0].set_ylabel(r'$r_p \cdot w_p(r_p)$ [$\mathrm{{Mpc}}/h$]', fontsize=fontsize)
            ax[1].set_ylabel(r'$s \cdot \xi_0(s)$ [$\mathrm{{Mpc}}/h$]', fontsize=fontsize)
            ax[2].set_ylabel(r'$s \cdot \xi_2(s)$ [$\mathrm{{Mpc}}/h$]', fontsize=fontsize)
            ax[2].legend(fontsize=fontsize)
        if show: 
            fig.show()
        if save_fn:
            fig.savefig(save_fn, facecolor='w',  bbox_inches='tight', pad_inches=0.1)
        return fig

    
    def plot_stats(self, cat, stats=('wp', 'xi_ells'), tracers=None,
                    data=None, residuals=True, fig=None, show=True,
                    figsize=None, fontsize=11, colors=None, by_tracer='rows',
                    residual_band=2.0, height_ratios=(3, 1), cmap='gist_heat_r',
                    norm2d=None, block_hspace=0.55, wspace=0.32, save_fn=None,
                    data_color=None, cross=True, max_cols=4, **kwargs):
        """ 
        Plot one statistic per panel, with optional data and residuals.
        Generated by AI Claude Opus 5 model.

        Parameters
        ----------
        cat : catalogue
            Passed straight through to the getters.
        stats : sequence of str
            Statistics to plot, one panel each. Either keys of :data:`STATS`
            (see :func:`register_stat` to add your own) or group names from
            :data:`STAT_GROUPS` that expand into several panels --
            ``'xi_ells'`` expands to one panel per configured multipole,
            read from
            ``self.args['clustering_settings']['xi_smu']['multipole_index']``.
            ``'power_spectrum'`` similarly expands using
            ``clustering_settings['power_spectrum']['multipole_index']``;
            names such as ``'xi4'``, ``'xi6'``, ``'pk4'`` and ``'pk6'``
            select individual configured orders, with no fixed upper limit.
            ``'ALL'`` includes the configured orders of both families.
        tracers : str, sequence of str, or None
            Tracers to overplot. ``None`` uses ``np.unique(cat['TRACER'])``.
        data : dict or None
            Measured data to compare against, keyed by statistic name and
            then by tracer::

                data = {'wp': {'LRG': (rp, wp, err), 'QSO': (rp, wq, eq)},
                        'xi0': {'LRG': {'x': s, 'y': xi0, 'cov': C}}}

            A single leaf entry may be given in place of the tracer level,
            in which case it is assigned to the first tracer block::

                data = {'wp': (rp, wp, sigma_wp)}

            ``err`` may be a 1D vector or a full covariance matrix.
        residuals : bool
            Draw residual sub-panels where data are available.
        cross : bool
            When several tracers are given, also show the cross-correlation
            of every pair, labelled ``'LRG_QSO'`` and obtained by calling the
            getters with ``tracers=['LRG', 'QSO']``. Set ``False`` for autos
            only.
        by_tracer : {'overlay', 'rows', 'figures'}
            How to lay out several tracers.

            ``'overlay'``  all tracers drawn on the same panels;
            ``'rows'``     one row of panels per tracer in a single figure,
                        each row titled with the tracer name (default);
            ``'figures'``  one separate figure per tracer -- a list of
                        figures is returned instead of a single figure.
        residual_band : float
            Half-width of the shaded band in the residual panels, in sigma.
        max_cols : int
            Maximum number of panel columns; statistics beyond this wrap onto
            further rows within each tracer block. Default 4.

        Returns
        -------
        fig : matplotlib Figure
        """
        from .plot_utils import (_expand_stats, _unpack_data, get_STATS,
                                 _fetch, _mirror_quadrants, _resolve_name,
                                 _split_group_entry, STAT_GROUPS)
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from matplotlib.colors import LogNorm
        import warnings
        from collections.abc import Mapping
        
        default_colors = {'ELG': 'deepskyblue', 'QSO': 'seagreen', 'LRG': 'red',
                        'BGS': 'goldenrod', 'LRG_ELG': 'firebrick', 'ELG_LRG': 'firebrick',
                        'QSO_ELG': 'skyblue', 'ELG_QSO': 'skyblue', 'LRG_QSO': 'peru', 'QSO_LRG': 'peru'}
        colors = {**default_colors, **(colors or {})}
        stats = [stats] if isinstance(stats, str) else list(stats)
        if stats[0].upper() == 'ALL':
            # Resolve both multipole families from this object's configuration,
            # excluding stale orders registered while plotting another object.
            prefixes = {'xi_ells': 'xi', 'power_spectrum': 'pk'}
            all_stats = []
            for name, spec in get_STATS().items():
                prefix = prefixes.get(spec.source)
                if (prefix is not None and name.lower().startswith(prefix)
                        and name[len(prefix):].isdigit()):
                    name = spec.source
                if name not in all_stats:
                    all_stats.append(name)
            stats = all_stats
        stats = _expand_stats(self, stats)          # 'xi_ells' -> xi0, xi2, ...
        STATS = get_STATS()                         # after expansion
        unknown = [s for s in stats if s not in STATS]
        if unknown:
            raise ValueError(f'unknown statistic(s) {unknown}; available: '
                            f'{sorted(STATS)}')
        

        tracers = self.check_cat_tracers(cat, tracers)


        # --- build the list of (label, tracer_arg, data_keys, stat_keys) ---
        # autos are single tracers; crosses are pairs. Two sets of candidate
        # keys are kept, because `compute_stats` and the user-supplied `data`
        # dict may name blocks differently: compute_stats typically uses the
        # doubled form 'LRG_LRG' for autos, while data are often keyed 'LRG'.
        # Either ordering of a cross pair is accepted. Keys are built from the
        # tracer names, never by splitting the label, since tracer names may
        # themselves contain '_' (e.g. 'ELG_LOPnotqso').
        from itertools import combinations
        blocks = [(t, t, [t, f'{t}_{t}'], [f'{t}_{t}', t]) for t in tracers]
        if cross and len(tracers) > 1:
            blocks += [(f'{a}_{b}', [a, b],
                        [f'{a}_{b}', f'{b}_{a}'],
                        [f'{a}_{b}', f'{b}_{a}'])
                    for a, b in combinations(tracers, 2)]
        labels = [lbl for lbl, _, _, _ in blocks]
        all_keys = [k for _, _, ks, _ in blocks for k in ks]

        # --- normalise the data dict to data[stat][tracer] ----------------
        # A statistic's value is either a tracer-keyed mapping, or a single
        # leaf entry -- (x, y[, err]) or {'x':..., 'y':...} -- which is taken
        # to belong to the first block.
        def _is_leaf(v):
            return isinstance(v, (tuple, list)) or (
                isinstance(v, Mapping) and 'x' in v and 'y' in v)

        data = {s: (v if not _is_leaf(v) else {labels[0]: v})
                for s, v in (data or {}).items()}

        stray = [s for s in data if s not in STATS
                 and _resolve_name(s, STAT_GROUPS) is None]
        if stray:
            warnings.warn(
                f'data keys {stray} are not known statistics and will be '
                'ignored; `data` is keyed data[stat][tracer], e.g. '
                f"{{'wp': {{'{labels[0]}': (x, y, err)}}}}")


        # --- expand group keys in `data`: 'xi_ells' -> xi0, xi2, ... ------
        # Group resolvers are consulted directly rather than reusing the
        # expansion of `stats`, so data may be keyed by the group even when
        # the panels were requested individually.
        for key in [k for k in data if _resolve_name(k, STAT_GROUPS)]:
            names = STAT_GROUPS[_resolve_name(key, STAT_GROUPS)](self)
            block = data.pop(key)
            if not isinstance(block, Mapping):
                raise ValueError(f"data['{key}'] must be a mapping")
            # already split by the caller: {'xi_ells': {'xi0': ..., ...}}
            if block and all(k in names for k in block):
                for n, v in block.items():
                    data.setdefault(n, {}).update(
                        {labels[0]: v} if _is_leaf(v) else v)
                continue
            for tracer, entry in block.items():
                for n, leaf in zip(names,
                                   _split_group_entry(entry, names, key)):
                    data.setdefault(n, {})[tracer] = leaf

        # When overplotting onto an existing figure, fall back to the data
        # that were supplied when that figure was built, so residuals can be
        # computed for the new catalogue without repeating the data.
        if fig is not None:
            cached = getattr(fig, '_plot_stats_data', None)
            if cached:
                merged = {t: dict(v) for t, v in cached.items()}
                for t, v in data.items():
                    merged.setdefault(t, {}).update(v)
                data = merged

        def _data_for(keys, stat):
            """First matching tracer key wins; `keys` may be a key or a list."""

            if isinstance(keys, str):
                keys = [keys]
            block = data.get(stat)
            if not isinstance(block, Mapping):
                return None
            for k in keys:
                entry = block.get(k)
                if entry is not None:
                    return _unpack_data(entry)
            return None

        # a residual row is needed if any tracer has data with errors for that stat
        def _needs_resid(stat):
            spec = STATS[stat]
            for _, _, ks, _ in blocks:
                d = _data_for(ks, stat)
                if d is None:
                    continue
                # ratio residuals only need the data values, not their errors
                if spec.residual == 'ratio' or d[2] is not None:
                    return True
            return False

        has_resid = {s: residuals and STATS[s].ndim == 1 and _needs_resid(s)
                    for s in stats}
        any_resid = any(has_resid.values())

        # --- figure / axes ------------------------------------------------
        # `by_tracer` controls how tracers are laid out:
        #   'overlay' : all tracers on the same panels
        #   'rows'    : one row of panels per tracer (default)
        #   'figures' : one separate figure per tracer
        if by_tracer not in ('overlay', 'rows', 'figures'):
            raise ValueError("by_tracer must be 'overlay', 'rows' or 'figures'")

        # 'figures' == recurse once per tracer, returning a list of figures
        if by_tracer == 'figures' and len(blocks) > 1:
            figs = []
            for lbl, tr, ks, sks in blocks:
                sub = {}
                for s, per_tracer in data.items():
                    if not isinstance(per_tracer, Mapping):
                        continue
                    for k in ks:
                        if k in per_tracer:
                            sub[s] = {lbl: per_tracer[k]}
                            break
                figs.append(self.plot_stats(
                    self, cat, stats=stats,
                    tracers=tr if isinstance(tr, list) else [tr],
                    data=sub or None,
                    cross=False,
                    residuals=residuals, fig=None, show=show, figsize=figsize,
                    fontsize=fontsize, colors=colors, by_tracer='overlay',
                    residual_band=residual_band, height_ratios=height_ratios,
                    cmap=cmap, norm2d=norm2d, block_hspace=block_hspace,
                    wspace=wspace, data_color=data_color, max_cols=max_cols,
                    **kwargs))
            return figs

        # number of tracer-blocks stacked vertically, and the panel grid
        # inside each block (wrapped at `max_cols` columns)
        nblock = len(blocks) if by_tracer == 'rows' else 1
        ncol = max(1, min(max_cols, len(stats)))
        nsub = int(np.ceil(len(stats) / ncol))      # panel rows per block

        if figsize is None:
            panel_h = 4.4 if any_resid else 3.4
            rows_total = nsub * nblock
            figsize = (4 * ncol,
                    panel_h * rows_total
                    + (0.9 * (rows_total - 1) if rows_total > 1 else 0))

        if fig is None:
            fig = plt.figure(figsize=figsize)
            # outer grid: one cell per panel, wrapped at `ncol` columns and
            # repeated for each tracer block. `block_hspace` separates the
            # rows; within a cell the main and residual axes are glued
            # together with hspace=0.
            outer = GridSpec(nsub * nblock, ncol, figure=fig,
                            hspace=block_hspace, wspace=wspace,
                            left=0.09, right=0.98, top=0.94, bottom=0.12)
            axes_grid = []
            for b in range(nblock):
                main_ax, res_ax = [], []
                for i, s in enumerate(stats):
                    r = b * nsub + i // ncol
                    c = i % ncol
                    if any_resid and has_resid[s]:
                        inner = outer[r, c].subgridspec(
                            2, 1, height_ratios=height_ratios, hspace=0.0)
                        a = fig.add_subplot(inner[0])
                        rr = fig.add_subplot(inner[1], sharex=a)
                        a.tick_params(labelbottom=False)
                    else:
                        a = fig.add_subplot(outer[r, c])
                        rr = None
                    main_ax.append(a)
                    res_ax.append(rr)
                axes_grid.append((main_ax, res_ax))
            # remember the layout so the figure can be safely reused
            fig._plot_stats_axes = axes_grid
            fig._plot_stats_stats = list(stats)
            fig._plot_stats_data = {t: dict(v) for t, v in data.items()}
            fig._plot_stats_drawn = set()
        else:                                   # reuse an existing figure
            axes_grid = getattr(fig, '_plot_stats_axes', None)
            if axes_grid is None:
                # figure not built by plot_stats: fall back to positional
                # mapping, assuming main axes come first
                axes = fig.axes
                n = len(stats)
                if len(axes) < n:
                    raise ValueError(
                        f'the supplied figure has {len(axes)} axes but '
                        f'{n} statistic(s) were requested')
                axes_grid = [(list(axes[:n]),
                            (list(axes[n:]) + [None] * n)[:n])]
            else:
                prev = getattr(fig, '_plot_stats_stats', None)
                if prev is not None and prev != list(stats):
                    raise ValueError(
                        f'figure was built for stats {prev}, cannot overplot '
                        f'{list(stats)}; pass the same stats or fig=None')
            nblock = len(axes_grid)

        # --- draw ---------------------------------------------------------
        # compute_stats results are cached per source for this call
        _cache = {}
        for it, (blabel, tr, bkeys, skeys) in enumerate(blocks):
            # pick the axes block for this tracer / tracer pair
            blk = it if by_tracer == 'rows' else 0
            if blk >= len(axes_grid):
                raise ValueError(
                    f'no axes block for {blabel!r}: the figure has '
                    f'{len(axes_grid)} block(s) but {len(blocks)} were '
                    f"requested with by_tracer='{by_tracer}'. Pass fig=None to "
                    'build a new figure.')
            main_ax, res_ax = axes_grid[blk]
            if by_tracer == 'rows':
                main_ax[0].set_title(blabel, fontsize=fontsize + 1, loc='left')
            kw = dict(kwargs)
            # cross blocks fall back to the first tracer's colour unless the
            # combined label has its own entry in `colors`
            kw.setdefault('color', colors.get(
                blabel, colors.get(tr[0] if isinstance(tr, list) else tr)))
            label = kw.pop('label', blabel)

            for i, name in enumerate(stats):
                spec = STATS[name]
                ax, rax = main_ax[i], res_ax[i]

                # ---- 2D statistics: map, no residual panel ---------------
                if spec.ndim == 2:
                    got = _fetch(self, spec, cat, skeys, _cache, tracers)
                    if got is None:      # statistic undefined for this block
                        ax.set_visible(False)
                        if rax is not None:
                            rax.set_visible(False)
                        continue
                    x, y, z = got

                    if spec.mirror:
                        xp, yp, zp = _mirror_quadrants(x, y, z)
                    else:
                        xp, yp, zp = x, y, z.T

                    # NaNs / infs are common in 2D estimators (empty bins,
                    # zero pair counts). The coordinate arrays must be fully
                    # finite for pcolormesh, so drop any bad bins together
                    # with the matching rows/columns of the map; remaining
                    # bad values in the map itself are masked so they render
                    # as blank cells rather than poisoning the colour scale.
                    xp, yp = np.asarray(xp, float), np.asarray(yp, float)
                    zp = np.asarray(zp, float)
                    gx, gy = np.isfinite(xp), np.isfinite(yp)
                    if not (gx.all() and gy.all()):
                        # zp is laid out (len(yp), len(xp))
                        zp = zp[np.ix_(gy, gx)]
                        xp, yp = xp[gx], yp[gy]

                    finite = np.isfinite(zp)
                    if not finite.all():
                        zp = np.ma.masked_invalid(zp)

                    if xp.size < 2 or yp.size < 2 or not finite.any():
                        ax.text(0.5, 0.5, 'no finite data', ha='center',
                                va='center', transform=ax.transAxes,
                                fontsize=fontsize, color='0.5')
                        ax.set_xlabel(spec.xlabel, fontsize=fontsize)
                        ax.set_ylabel(spec.ylabel, fontsize=fontsize)
                        continue

                    # colour scale from the finite, positive values only
                    nrm = kwargs.get('norm', norm2d)
                    if nrm is None:
                        with np.errstate(invalid='ignore'):
                            pos = np.asarray(zp)[finite & (np.asarray(zp) > 0)]
                        nrm = (LogNorm(vmin=pos.min(), vmax=pos.max())
                            if pos.size else None)

                    with np.errstate(invalid='ignore', divide='ignore'):
                        mesh = ax.pcolormesh(xp, yp, zp, cmap=cmap, norm=nrm,
                                            shading=spec.shading)
                    fig.colorbar(mesh, ax=ax)

                    # contours need plain floats; skip silently if the map is
                    # entirely masked or has no finite spread
                    if spec.levels is not None and finite.any():
                        zc = np.where(finite, np.asarray(zp), np.nan)
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            with np.errstate(invalid='ignore', divide='ignore'):
                                try:
                                    ax.contour(xp, yp, zc,
                                            levels=list(spec.levels),
                                            colors='k', linewidths=0.8,
                                            alpha=0.8)
                                except (ValueError, TypeError):
                                    pass    # degenerate map: no contours

                    if spec.linthresh is not None:
                        ax.set_xscale('symlog', linthresh=spec.linthresh)
                    ax.set_xlabel(spec.xlabel, fontsize=fontsize)
                    ax.set_ylabel(spec.ylabel, fontsize=fontsize)
                    if by_tracer == 'overlay' and len(blocks) > 1:
                        ax.set_title(blabel, fontsize=fontsize)
                    continue

                # ---- model ------------------------------------------------
                got = _fetch(self, spec, cat, skeys, _cache, tracers)
                if got is None:          # statistic undefined for this block
                    ax.set_visible(False)
                    if rax is not None:
                        rax.set_visible(False)
                    continue
                x, y = got
                y_plot = spec.scale(x, y) if spec.scale else y
                # label once per block: on the last panel (overlay) or on
                # every block's last panel (rows)
                ax.plot(x, y_plot,
                        label=label if i == len(stats) - 1 else None, **kw)

                # ---- data -------------------------------------------------
                d = _data_for(bkeys, name)
                if d is not None:
                    dx, dy, dsig = d
                    drawn = getattr(fig, '_plot_stats_drawn', None)
                    if drawn is None:
                        drawn = fig._plot_stats_drawn = set()

                    # points are drawn once per (tracer, statistic); a second
                    # catalogue overplotted on the same figure reuses them
                    if (blabel, name) not in drawn:
                        dy_plot = spec.scale(dx, dy) if spec.scale else dy
                        dsig_plot = None
                        if dsig is not None:
                            dsig_plot = (spec.scale(dx, dsig) if spec.scale
                                        else dsig)
                        ax.errorbar(dx, dy_plot, yerr=dsig_plot, fmt='o', ms=4,
                                    mfc='none', capsize=0, lw=1.2, zorder=5,
                                    color=data_color or kw.get('color'),
                                    label=f'{label} data')
                        drawn.add((blabel, name))

                    # ---- residuals: always for the current model ---------
                    if rax is not None and (dsig is not None
                                            or spec.residual == 'ratio'):
                        ymod = np.interp(dx, x, y)      # model at the data x
                        if spec.residual == 'ratio':
                            with np.errstate(divide='ignore', invalid='ignore'):
                                res = np.where(dy != 0, ymod / dy - 1.0, np.nan)
                        else:
                            res = (ymod - dy) / dsig
                        rax.plot(dx, res, lw=1.4, color=kw.get('color'))

                ax.set_xscale(spec.xscale)
                ax.set_yscale(spec.yscale)
                ax.set_ylabel(spec.ylabel, fontsize=fontsize)
                ax.grid(alpha=0.25)

                if rax is None:
                    ax.set_xlabel(spec.xlabel, fontsize=fontsize)
                else:
                    rax.set_xlabel(spec.xlabel, fontsize=fontsize)
                    if spec.residual == 'ratio':
                        rax.set_ylabel(r'model$/$data$-1$', fontsize=fontsize)
                    else:
                        rax.set_ylabel(r'$\Delta/\sigma$', fontsize=fontsize)
                        rax.axhspan(-residual_band, residual_band,
                                    color='grey', alpha=0.25)
                    rax.axhline(0, ls='--', color='k', lw=1)
                    if spec.residual_ylim is not None:
                        rax.set_ylim(*spec.residual_ylim)
                    rax.set_xscale(spec.xscale)
                    rax.grid(alpha=0.25)

        # legend on the last 1D panel of each tracer block
        for blk_main, _ in axes_grid:
            for i in range(len(stats) - 1, -1, -1):
                if STATS[stats[i]].ndim != 1 or not blk_main[i].get_visible():
                    continue
                h, l = blk_main[i].get_legend_handles_labels()
                if h:
                    blk_main[i].legend(fontsize=fontsize)
                break

        if not hasattr(fig, '_plot_stats_axes'):
            fig.tight_layout()
        if show:
            fig.show()
        if save_fn:
            fig.savefig(save_fn, facecolor='w',  bbox_inches='tight', pad_inches=0.1)
        return fig
