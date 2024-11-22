import matplotlib.pyplot as plt 
import matplotlib.lines as mlines
from numba import njit, jit, numba
import time 
import fitsio
import os
import sys
from cosmoprimo.fiducial import AbacusSummit, Planck2018FullFlatLCDM, Cosmology
from utils import *
from HOD_models import _SHOD, _GHOD, _SFHOD, _SHOD, _LNHOD, _HMQ, _mHMQ, _Nsat_pow_law
from abacus_func import *
import yaml 
import socket
from pinnochio_io import *
from pypower import CatalogMesh
from mpi4py import MPI
import glob
from fits_functions import *
import pandas as pd
import emcee 
#import zeus
import sklearn.gaussian_process as skg
from scipy.stats import norm
from multiprocessing import Pool
from mpytools import Catalog
import collections.abc

class HOD:
    """
    --- HOD code 
    """

    def __init__(self, param_file=None, args=None, hcat=None, usecols=None, read_Abacus=False, read_pinnochio=False, read_Abacus_mpi=False):
        """
        ---
        """
        self.mpicomm = MPI.COMM_WORLD
        self.args = yaml.load(open('parameters_HODdefaults.yaml'), Loader=yaml.FullLoader)      

        def update_dic(d, u):
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k] = update_dic(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
            
        new_args = yaml.load(open(param_file), Loader=yaml.FullLoader) if param_file is not None else args if args is not None else None
        update_dic(self.args, new_args)
        
        #if args is not None:
        #    self.args.update(args)
        self.args['nthreads'] = min(numba.get_num_threads(), self.args['nthreads'])
        print('Set number of threads to {}'.format(self.args['nthreads']))

        if 'halo_lc' not in self.args['hcat'].keys():      
            self.args['hcat']['halo_lc'] = False

        if hcat is not None:
            if isinstance(hcat, Catalog):
                self.hcat = hcat
            elif isinstance(hcat, dict):
                self.hcat = Catalog.from_dict(hcat)
            else:
                raise ValueError ('halo catalog is not a dictionary')
            self.cosmo = Cosmology(**{k: v for k, v in self.args['cosmo'].items() if v is not None})
            
        else:
            if read_Abacus:
                self.hcat, self.part_subsamples, self.boxsize, self.origin = read_Abacus_hcat(self.args, halo_lc=self.args['hcat']['halo_lc'])
                self.cosmo = AbacusSummit(self.args['hcat']['sim_name'].split('_c')[-1][:3]).get_background(engine=self.args['cosmo']['engine'])               

            elif read_Abacus_mpi:
                print('Attention MODIF')
                #self.hcat, self.boxsize = read_Abacus_hcat(self.args)
                self.hcat = Catalog.load('/pscratch/sd/a/arocher/test_abacus_cat_mpi.fits', mpicomm=self.mpicomm)
                self.boxsize=1000
                self.cosmo = AbacusSummit(self.args['hcat']['sim_name'].split('_c')[-1][:3]).get_background(engine=self.args['cosmo']['engine']) 
                
            elif read_pinnochio:
                print('Read Pinnochio', flush=True)
                start = time.time()
                self.hcat, self.boxsize, self.cosmo = read_pinnochio_hcat(self.args)
                print('Done {:.2f}'.format(time.time()-start), flush=True)
            else: 
                self.hcat = Catalog.read(args['hcat'][['path_to_sim']])            
        self.H_0 = 100
        try:
            self.NFW_draw = np.load(self.args['precompute_NFW'])
        except FileNotFoundError:
            print("Generate random point in a NFW for satellite drawing")
            self.NFW_draw = rd_draw_NFW(15000000, self.args['precompute_NFW'])

        if 'c' not in self.hcat.columns():
            print('Concentration column "c" is not provided. The concentration is computed from mass-concentration relation of {} using {} as mass definition'.format(self.args['cm_relation'], self.args['mass_def']))
            self.hcat['c'] = get_concentration(self.hcat['Mvir'], cosmo=self.cosmo, mdef=self.args['mass_def'], cmrelation=self.args['cm_relation'])
        
        try :
            self._fun_cHOD, self._fun_sHOD = {}, {}
            for tr in self.args['tracers']:
                self._fun_cHOD[tr] = globals()['_'+self.args[tr]['HOD_model']]
                self._fun_sHOD[tr] = globals()['_'+self.args[tr]['sat_HOD_model']]
        except :
            import HOD_models 
            help(HOD_models)
            raise ValueError('{} not implemented in HOD models'.format(self.args['HOD_param']['HOD_model']))
        
        self.rng = np.random.RandomState(seed=self.args['seed'])
        
        if self.args['assembly_bias']:
            self._compute_assembly_bias_columns()
    
    
    def __init_hod_param(self, tracer):
            '''
            --- Init hod list parameters
            '''

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
        if isinstance(self.args[tracer]['density'], float):
            if verbose:
                print('Set density to {} gal/Mpc/h'.format(self.args[tracer]['density']))
            return self.args[tracer]['density']*self.boxsize**3 /self.ngal(tracer)[0]
        else:
            print('No density set')
            return 1 
    
    def ngal(self, tracer, verbose=False):
        '''
        --- Return the number of galaxy and the satelitte fraction form HOD parameters
        '''
        start = time.time()
        hod_list_param_cen, hod_list_param_sat, _ = self.__init_hod_param(tracer)
        ngal, fsat = compute_ngal(self.hcat['log10_Mh'], self._fun_cHOD[tracer], self._fun_sHOD[tracer], self.args['nthreads'], 
                                hod_list_param_cen, hod_list_param_sat, self.args[tracer]['conformity_bias'])
        if verbose:
            print(time.time()-start)
        return ngal, fsat   


    def calc_env_factor(self, cellsize=5, resampler='cic'):

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
        ab_proxy = np.unique([[l for l in ll] for ll in [list(self.args[tr]['assembly_bias'].keys()) for tr in self.args['tracers']]])
        for ab in ab_proxy:
            self.set_assembly_bias_values(ab)

    def set_assembly_bias_values(self, col, bins=50):
        """
        --- Set by mass bin a linear fonction (-0.5, 0.5) according to col value
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
        Generate mock catalogs from HOD model.

        Parameters
        ----------
        self

        fix_seed : Fix the seed for reproductibility. Caveat : Only works for a same number of threads in args['nthreads']

        Output
        ------
        mock_cat : dict
            dict of mock galaxies properties
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
                
                if self.args['use_particles']:
                    if verbose: print('Using particles', flush=True)
                    if self.part_subsamples is not None:
                        mask_nfw = compute_sat_from_part(self.part_subsamples['pos'].T[0],self.part_subsamples['pos'].T[1],self.part_subsamples['pos'].T[2],
                            self.part_subsamples['vel'].T[0], self.part_subsamples['vel'].T[1],self.part_subsamples['vel'].T[2],
                            sat_cat['x'], sat_cat['y'], sat_cat['z'], sat_cat['vx'], sat_cat['vy'], sat_cat['vz'],
                            self.hcat['npoutA'][mask_sat], self.hcat['npstartA'][mask_sat], list_nsat, np.insert(np.cumsum(list_nsat), 0, 0), self.args['nthreads'], seed=seed)
                        if verbose: print(f'{mask_nfw.sum()} satellites will be positioned using NFW', flush=True)
                    else:
                        print('No particles found continue with NFW', flush=True)
                else:
                    mask_nfw = np.ones(Nb_sat, dtype=bool)
                
                if mask_nfw.sum() > 0:
                    NFW = self.NFW_draw[self.NFW_draw < sat_cat['c'][mask_nfw].max()]
                    if len(NFW) > Nb_sat:
                        rng.shuffle(NFW)
                    else:
                        NFW = NFW[rng.randint(0, len(NFW), Nb_sat)]

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
                    sat_cat['vx'][mask_nfw], sat_cat['vy'][mask_nfw], sat_cat['vz'][mask_nfw] = compute_fast_NFW(NFW, sat_cat['x'][mask_nfw], sat_cat['y'][mask_nfw], sat_cat['z'][mask_nfw],
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
                if verbose:
                    print('{} mock catalogue done'.format(tracer), time.time()-start, flush=True)
                    print("{} central galaxies, {} satellites, fraction of satellite {:.2f} ".format(mask_cent.sum(),Nb_sat, Nb_sat/final_cat[tracer].size), flush=True)
                
            mask_id &= (mask_cent | mask_sat)
            count_gal[tracer] = np.int64(proba_sat + cond_cent)

            if verbose:
                print("Done overall time ", tracer, time.time() - timeall, flush=True)

        # When LRG and ELG are in the same halo, put 1 LRG at the center and all other galaxies are position following NFW profile
        if ((tracers == ['ELG', 'LRG']) | (tracers == ['LRG', 'ELG'])) & (mask_id.sum() >0) & ~self.args['hcat']['halo_lc']:
            mask_elg = np.in1d(final_cat['ELG']['row_id'], final_cat['LRG']['row_id'])
            mask_lrg = np.in1d(final_cat['LRG']['row_id'], final_cat['ELG']['row_id'])
            
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

        return final_cat



    def init_elg_sat_for_lrg(self, tracer, sat_cat, fix_seed=None):

        rng = np.random.RandomState(seed=fix_seed)

        if fix_seed is not None:
            seed = rng.randint(0, 4294967295, self.args['nthreads'])
        else:
            seed = None

        Nb_sat = sat_cat.size
        NFW = self.NFW_draw[self.NFW_draw < sat_cat['c'].max()]
        if len(NFW) > Nb_sat:
            rng.shuffle(NFW)
        else:
            NFW = NFW[rng.randint(0, len(NFW), Nb_sat)]

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

        sat_cat['x'], sat_cat['y'], sat_cat['z'], sat_cat['vx'], sat_cat['vy'], sat_cat['vz'] = compute_fast_NFW(NFW, sat_cat['x'], sat_cat['y'], sat_cat['z'],
                                                                                                                    sat_cat['vx'], sat_cat['vy'], sat_cat['vz'],
                                                                                                                    sat_cat['c'], sat_cat['Mh'], sat_cat['Rh'], 
                                                                                                                    rd_pos, rd_vel, exp_frac=0, 
                                                                                                                    exp_scale=self.args[tracer]['exp_scale'], nfw_rescale=self.args[tracer]['nfw_rescale'],
                                                                                                                    vrms_h=vrms_h, f_sigv=self.args[tracer]['f_sigv'], v_infall=self.args[tracer]['v_infall'], 
                                                                                                                    vel_sat=self.args[tracer]['vel_sat'], Nthread=self.args['nthreads'], seed=seed)
        
        sat_cat['Central'] = sat_cat.zeros()
        return sat_cat


                                         
    def get_2PCF(self, cats, tracers=None, R1R2=None, verbose=True):
        """
        --- Return the 2PCF for a given mock catalog in a cubic box
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

        s_all, xi_all = [],[]
        for tr in tracers:
            mock_cat = cats[tr]
            if verbose:
                print('#Computing 2PCF for {}...'.format(tr), flush=True)
                time1 = time.time()
            if self.args['2PCF_settings']['rsd']:
                pos = apply_rsd (mock_cat, self.args['hcat']['z_simu'], self.boxsize, self.H_0, self.args['2PCF_settings']['los'], self.args[tr]['vsmear'], self.cosmo)
            else:
                pos = mock_cat['x']%self.boxsize, mock_cat['y']%self.boxsize, mock_cat['z']%self.boxsize
            
            s, xi = compute_2PCF(pos, self.args['2PCF_settings']['edges_smu'], self.args['2PCF_settings']['multipole_index'], self.boxsize,  self.args['2PCF_settings']['los'], self.args['nthreads'], R1R2=R1R2)
            if verbose:
                print('#2PCF for {} computed !time = {:.3f} s'.format(tr, time.time()-time1), flush=True)
            if len(tracers) > 1:
                s_all += [s]
                xi_all += [xi]
            else: 
                return s, xi
        return s_all, xi_all

    def get_wp(self, cats, tracers=None, R1R2=None, verbose=True):
        """
        --- Return wp (projected correlation function) for a given mock catalog in a cubic box
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
            mock_cat = cats[tr]
            if verbose:
                print('#Computing wp for {}...'.format(tr), flush=True)
                time1 = time.time()
            if self.args['2PCF_settings']['rsd']:
                pos = apply_rsd (mock_cat, self.args['hcat']['z_simu'], self.boxsize, self.H_0, self.args['2PCF_settings']['los'], self.args[tr]['vsmear'], self.cosmo)
            else:
                pos = mock_cat['x']%self.boxsize, mock_cat['y']%self.boxsize, mock_cat['z']%self.boxsize
            rp, wp = compute_wp(pos, self.args['2PCF_settings']['edges_rppi'], self.args['2PCF_settings']['pimax'], self.boxsize, self.args['2PCF_settings']['los'],  self.args['nthreads'], R1R2=R1R2)
            if verbose:
                print('#wp for {} computed !time = {:.3f} s'.format(tr, time.time()-time1), flush=True)
            if len(tracers) > 1:
                rp_all += [rp]
                wp_all += [wp]
            else: 
                return rp, wp
        return rp_all, wp_all
    

    def get_crosswp(self, cats, tracers, R1R2=None, verbose=True):
        """
        --- Return wp (projected correlation function) for a given mock catalog in a cubic box
        """
        
        if self.args['2PCF_settings']['edges_rppi'] is None:
            if self.args['2PCF_settings']['bin_logscale']:
                r_bins = np.geomspace(self.args['2PCF_settings']['rp_min'], self.args['2PCF_settings']['rp_max'], self.args['2PCF_settings']['n_rp_bins']+1, endpoint=(True))
            else:
                r_bins = np.linspace(self.args['2PCF_settings']['rp_min'], self.args['2PCF_settings']['rp_max'], self.args['2PCF_settings']['n_rp_bins']+1)
            self.args['2PCF_settings']['edges_rppi'] = (r_bins, np.linspace(-self.args['2PCF_settings']['pimax'], self.args['2PCF_settings']['pimax'], 2*self.args['2PCF_settings']['pimax']+1))


        res_dict = {}
        com_tr =np.vstack([np.array(np.meshgrid(tracers,tracers)).T.reshape(-1, len(tracers)).flatten().reshape(len(tracers),len(tracers),2)[i,i:] for i in range(len(tracers))])
        for tr in com_tr:
            if verbose:
                print('#Computing wp for {}...'.format(tr), flush=True)
                time1 = time.time()
            if self.args['2PCF_settings']['rsd']:
                pos1 = apply_rsd (cats[tr[0]], self.args['hcat']['z_simu'], self.boxsize, self.H_0, self.args['2PCF_settings']['los'], self.args[tr[0]]['vsmear'], self.cosmo)
                pos2 = apply_rsd (cats[tr[1]], self.args['hcat']['z_simu'], self.boxsize, self.H_0, self.args['2PCF_settings']['los'], self.args[tr[1]]['vsmear'], self.cosmo)
            else:
                pos1 = cats[tr[0]]['x']%self.boxsize, cats[tr[0]]['y']%self.boxsize, cats[tr[0]]['z']%self.boxsize
                pos2 = cats[tr[1]]['x']%self.boxsize, cats[tr[1]]['y']%self.boxsize, cats[tr[1]]['z']%self.boxsize

            res_dict[f'{tr[0]}_{tr[1]}'] = compute_wp(pos1, self.args['2PCF_settings']['edges_rppi'], self.args['2PCF_settings']['pimax'], self.boxsize, self.args['2PCF_settings']['los'],  self.args['nthreads'], R1R2=R1R2, pos2=pos2)
            if verbose:
                print('#wp for {} computed !time = {:.3f} s'.format(tr, time.time()-time1), flush=True)

        return res_dict
    

    def get_cross2PCF(self, cats, tracers, R1R2=None, verbose=True):
        """
        --- Return the 2PCF for a given mock catalog in a cubic box
        """
        
        if self.args['2PCF_settings']['edges_smu'] is None:
            if self.args['2PCF_settings']['bin_logscale']:
                r_bins = np.geomspace(self.args['2PCF_settings']['rmin'], self.args['2PCF_settings']['rmax'], self.args['2PCF_settings']['n_r_bins'])
            else:
                r_bins = np.linspace(self.args['2PCF_settings']['rmin'], self.args['2PCF_settings']['rmax'], self.args['2PCF_settings']['n_r_bins'])
            
            self.args['2PCF_settings']['edges_smu'] = (r_bins, np.linspace(-self.args['2PCF_settings']['mu_max'], self.args['2PCF_settings']['mu_max'], self.args['2PCF_settings']['n_mu_bins']))

        res_dict = {}
        com_tr =np.vstack([np.array(np.meshgrid(tracers,tracers)).T.reshape(-1, len(tracers)).flatten().reshape(len(tracers),len(tracers),2)[i,i:] for i in range(len(tracers))])
        for tr in com_tr:
            if verbose:
                print('#Computing 2PCF for {}...'.format(tr), flush=True)
                time1 = time.time()
            if self.args['2PCF_settings']['rsd']:
                pos1 = apply_rsd (cats[tr[0]], self.args['hcat']['z_simu'], self.boxsize, self.H_0, self.args['2PCF_settings']['los'], self.args[tr[0]]['vsmear'], self.cosmo)
                pos2 = apply_rsd (cats[tr[1]], self.args['hcat']['z_simu'], self.boxsize, self.H_0, self.args['2PCF_settings']['los'], self.args[tr[1]]['vsmear'], self.cosmo)
            else:
                pos1 = cats[tr[0]]['x']%self.boxsize, cats[tr[0]]['y']%self.boxsize, cats[tr[0]]['z']%self.boxsize
                pos2 = cats[tr[1]]['x']%self.boxsize, cats[tr[1]]['y']%self.boxsize, cats[tr[1]]['z']%self.boxsize

            res_dict[f'{tr[0]}_{tr[1]}'] = compute_2PCF(pos1, self.args['2PCF_settings']['edges_smu'], self.args['2PCF_settings']['multipole_index'], self.boxsize,  self.args['2PCF_settings']['los'], self.args['nthreads'], pos2=pos2, R1R2=R1R2)
            if verbose:
                print('#2PCF for {} computed !time = {:.3f} s'.format(tr, time.time()-time1), flush=True)
        return res_dict
    
    
    def HOD_plot(self, tracer=None, fig=None):
    
        if tracer is None:
            tracer=self.args['tracers']
        else:
            tracer = tracer if isinstance(tracer, list) else [tracer]
        colors = {'ELG': 'deepskyblue', 'QSO': 'seagreen', 'LRG': 'red'}
        for tr in tracer:
            pc, ps, pab =self.__init_hod_param(tr)
            fig = plot_HOD(pc, ps, self._fun_cHOD[tr], self._fun_sHOD[tr], label=tr, fig=fig, show=False, color=colors[tr] if tr in colors.keys() else None)
        plt.show()


    def plot_HMF(self, cats, show_sat=False, range=(10.8,15), tracer=None, inital_HMF=None):

        colors = {'ELG': 'deepskyblue', 'QSO': 'seagreen', 'LRG': 'red'}
        handles=[]
        
        if tracer is None:
            tracer=self.args['tracers']
        else:
            if not isinstance(tracer, list):
                tracer = [tracer]

        for i, tr in enumerate(tracer):
            plt.hist(cats[tr]['log10_Mh'], histtype='step', bins=100, color=colors[tr] if tr in colors.keys() else f'C{i}')
            if show_sat:
                plt.hist(cats[tr]['log10_Mh'][cats[tr]['Central']==1], histtype='step', bins=100, range=range, color=colors[tr] if tr in colors.keys() else f'C{i}', ls='--')
                plt.hist(cats[tr]['log10_Mh'][cats[tr]['Central']==0], histtype='step', bins=100, range=range, color=colors[tr] if tr in colors.keys() else f'C{i}', ls=':')
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
    def downsample_mock_cat (cat, ds_fac=0.1, mask=None):
        if mask is None:
            mask = np.random.uniform(size=len(cat['x'])) < ds_fac
        return cat[mask]
    

    def compute_training(self, tracers, nreal=20, training_points=None, start_point=0, verbose=False):

        tracers = tracers if isinstance(tracers, list) else [tracers]
        if training_points is None:
            training_points = genereate_training_points(self.args['fit_param']['N_trainning_points'], 
                                                        self.args['fit_param']['priors'], 
                                                        sampling_type=self.args['fit_param']['sampling_type'], 
                                                        rand_seed=self.args['fit_param']['seed_trainning'])
        
        os.makedirs(self.args['fit_param']["path_to_training_point"], exist_ok=True)
        if np.sum([len(self.args['fit_param']['priors'][tr]) for tr in tracers]) != len(training_points.dtype.names):
            raise ValueError('The training sample shape ({}) does not correspond to the number of parameters ({})'.format(len(self.args['fit_param']['priors'][tr]), len(training_points.dtype.names)))
        if verbose:
            print(f'Run training sample computations', flush=True)

        for nb_point, param in enumerate(training_points):
            if not os.path.exists(os.path.join(self.args['fit_param']["path_to_training_point"], '{}_{}.npy'.format(self.args['fit_param']['sampling_type'], nb_point+start_point))):
                start = time.time() 
                result = {}
                for tr in tracers:
                    var_name_tr = np.array(training_points.dtype.names)[np.array([tr in var for var in training_points.dtype.names])].tolist()
                    var_name = [v.split(f'_{tr}')[0] for v in var_name_tr]
                    self.args[tr].update(dict(zip(var_name, param[var_name_tr])))
                    result[tr] = self.args[tr].copy()
                cats = [self.make_mock_cat(tracers, verbose=verbose) for i in range(nreal)]
                
                if 'wp' in self.args['fit_param']["fit_type"]:
                    result['wp']= [self.get_crosswp(cats[i], tracers=tracers, verbose=verbose) for i in range(nreal)]
                if 'xi' in self.args['fit_param']["fit_type"]:
                    result['xi'] = [self.get_cross2PCF(cats[i], tracers=tracers, verbose=verbose) for i in range(nreal)]
                result['hod_fit_param'] = param
                np.save(os.path.join(self.args['fit_param']["path_to_training_point"], '{}_{}.npy'.format(self.args['fit_param']['sampling_type'], nb_point+start_point)), result)
                print('Point {} done {:.2f}'.format(nb_point+start_point, start-time.time()))        


    def read_training(self, data, inv_cov2, sig=None, add_sig2_cosmic=False):
        
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
            #res_std = np.std(res, axis=0)             # ONLY FOR CORR MODEL !!!
            #inv_Cov2 = inv_cov2*(res_std*res_std[:, None])
            if add_sig2_cosmic:
                inv_Cov2 *= np.sqrt(self.args['sig2_cosmic']*self.args['sig2_cosmic'][:,None])

            chi2 = np.mean([compute_chi2(model_arr, data, inv_Cov2=inv_cov2, sig=sig) for model_arr in res])
            chi2_err = np.std([compute_chi2(model_arr, data, inv_Cov2=inv_cov2, sig=sig) for model_arr in res])/np.sqrt(nreal)
            trainning_set[ii] = np.hstack((res_param['hod_fit_param'].tolist(),chi2,chi2_err))

            '''for comb_tr in res_param[stats[0]][0].keys():
                res[comb_tr], res_std[comb_tr] = {}, {}
                for stat in stats:
                    res[comb_tr][f'mean_{stat}'] = np.mean([np.hstack(res_param[stat][i][comb_tr][1]) for i in range(2)], axis=0)
                    res[comb_tr][f'std_{stat}'] = np.std([np.hstack(res_param[stat][i][comb_tr][1]) for i in range(2)], axis=0)

            mean = np.hstack([np.hstack([res[comb_tr][f'mean_{stat}'] for stat in stats]) for comb_tr in comb_trs])
            std = np.hstack([np.hstack([res[comb_tr][f'std_{stat}'] for stat in stats]) for comb_tr in comb_trs])'''

        trainning_set.dtype=[(name, dt) for name, dt in zip(name_arr, ['float64']*len(name_arr))]
        return trainning_set
    
        
    def run_gp_mcmc(self, training_set, niter, logchi2=True,
                        nb_points=1, remove_edges=0.9,
                        random_state=None, verbose=True):
        
        """
        --- Function which computes Gaussian process prediction from a given training sample, then compute a MCMC over the GP prediction, and returns the next point(s) using the input aquisition function for the iterative procedure
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
                add_sig2_cosmic=False, reprise=False, verbose=True):
        
        """
        --- Run the iterative procedure.
        """
        nmock = self.args['fit_param']['nb_real']
        dir_output_file= self.args['fit_param']['dir_output_fit']
        fit_name = self.args['fit_param']['fit_name']
        priors = self.args['fit_param']['priors']
        priors_array = np.vstack([list(priors[tr].values()) for tr in self.args['tracers']])
        nvar = len(priors_array)  
        arr_dtype = training_point.dtype

        iter = 0
        if reprise & os.path.exists(os.path.join(dir_output_file, f"{nvar}p_{fit_name}.txt")):
            output_point = pd.read_csv(os.path.join(
                dir_output_file, f"{nvar}p_{fit_name}.txt"), sep=" ", comment="#")
            
            training_point = np.concatenate((np.array(training_point.tolist()).reshape(len(training_point), -1), output_point[list(training_point.dtype.names)].values))
            training_point.dtype = arr_dtype

            iter = output_point["N_iter"].loc[len(output_point)-1]+1
            p = np.loadtxt(os.path.join(dir_output_file, 'chains',
                                        f'chain_{nvar}p_{fit_name}_{iter-1}.txt'))[:, :nvar]
            D_kl = 10
            if verbose:
                print("#reprise ", iter, "len param point ",
                      len(training_point), flush=True)
                
        print("Run gpmcmc...", flush=True)
        for j in range(iter, self.args['fit_param']['n_calls']):
            if verbose:
                print(f'Iteration {j}...', flush=True)
                time_compute_mcmc = time.time()

            new_params = self.run_gp_mcmc(training_point, j, logchi2=self.args['fit_param']['logchi2'],
                        nb_points=1, remove_edges=0.9,
                        random_state=None, verbose=True)


            if verbose:
                print("#time_compute_gpmcmc =", time.time()
                      - time_compute_mcmc, flush=True)

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
                if add_sig2_cosmic:
                    iCov2 *= np.sqrt(self.args['sig2_cosmic']*self.args['sig2_cosmic'][:,None])

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
