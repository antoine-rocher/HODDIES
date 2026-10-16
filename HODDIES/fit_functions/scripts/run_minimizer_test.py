from HODDIES import HOD
from HODDIES.sim_loader import AbacusSummitSim
from HODDIES.fit_functions import flatten_stats
import numpy as np
import os 
from mpi4py import MPI
import argparse


data_file = np.load('/global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/test_data/test_data_minimizer.npy', allow_pickle=True)[()]
data_file['param']['fit_param']['fit_statistics'] = ['wp', 'xi_ells', 'delta_sigma']

default_abacus_param = {'boxsize': None,  # Simulation box size (to be set if path_to_sim is provided)
    'path_to_sim': '/dvs_ro/cfs/cdirs/desi/cosmosim/Abacus',  # Path to halo catalogs
    'mass_cut': None,  # Minimum halo mass cut in log10(Msun/h) e.g. 12 to cut M < 10**12
    'z_simu': 0.5,  # Redshift of simulation snapshot (list of redshifts from Abacus light cone shells.)
                    # Redshift available for Uchuu snapshot [0.43, 0.49, 0.56, 0.63, 0.70, 0.78, 0.86, 0.94, 1.03, 1.12, 1.22, 1.32, 1.43, 1.65]
    'sim_name': 'AbacusSummit_small_c000_ph3000',  # Specific to Abacus simulation
    'load_particles': False,  # Whether to load particle/subhalos data
    'nthreads': 32,  # Number of threads to use for loading data
    'load_field_particles': True,  # Whether to load field particles data
}

sim_abacus=AbacusSummitSim(**default_abacus_param)

HOD_obj = HOD(base_catalog=sim_abacus, **data_file['param'])


stats_to_fit = data_file['param']['fit_param']['fit_statistics']
comb_tr_list = HOD_obj.get_comb_tr_list(['LRG', 'ELG'])

data_vec = flatten_stats(data_file['data'], stats=['wp', 'xi_ells', 'delta_sigma'])
mcov= np.loadtxt('/global/cfs/cdirs/desi/users/arocher/HODDIES_results/Abcaus_small_boxes_corr/z0.500/Mcov_small_box_z0.5_wp+xi+delta_sigma_ELG_LRG.txt')
num_sim = 1786 # number of Abacus small boxes used to compute the covariance matrix

HOD_obj.initialize_fit(data_vec, mcov)  

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()

minimizer_options = {"maxiter":32, "popsize":10000, 'xtol':1e-2, 'workers':mpi_comm.Get_size(),  'backend':'mpi'}

# init_params = np.load(save_fn, allow_pickle=True).item()['x']
from pycorr import TwoPointCorrelationFunction
result = TwoPointCorrelationFunction.load('/global/cfs/cdirs/desi/survey/catalogs/edav1/xi/sv3/rppi/allcounts_ELG_NScomb_0.8_1.1_default_angular_bitwise_FKP_log_njack128_nran18_split20.npy')[::2][4:-3]
result_smu = TwoPointCorrelationFunction.load('/global/cfs/cdirs/desi/survey/catalogs/edav1/xi/sv3/smu/allcounts_ELG_NScomb_0.8_1.1_default_angular_bitwise_FKP_log_njack128_nran18_split20.npy')[16:-6]
HOD_obj.args['clustering_settings']['xi_rppi']['edges_rppi'] = result.edges
HOD_obj.args['clustering_settings']['xi_smu']['edges_smu'] = result_smu.edges



res = HOD_obj.run_minimizer(mpi_comm=mpi_comm, return_log10chi2=True, **minimizer_options)
print(f"Rank {mpi_rank} finished minimization with result: {res}", flush=True)

if mpi_rank==0:
    new_params = np.array([ HOD_obj.result_fit['x']])
    new_params.dtype = [(name, dt) for name, dt in zip(HOD_obj.name_params, ['float64']*len(HOD_obj.name_params))]
    HOD_obj.update_new_param(new_params, HOD_obj.name_params)
    cat = HOD_obj.make_mock_cat(fix_seed=10)
    dir_to_save = HOD_obj.args['fit_param']['dir_output_fit']
    save_fn_fig = os.path.join(dir_to_save, 'best_fit.png')
    err = np.sqrt(np.diag(mcov))
    idx = 0 
    for tr in data_file['data']['wp'].keys():
        for st in ['wp', 'xi_ells', 'delta_sigma']:
            if (st == 'delta_sigma') and (tr == 'ELG_LRG'):
                continue
            idx_st = idx+data_file['data'][st][tr][1].size
            data_file['data'][st][tr] = list(data_file['data'][st][tr])
            data_file['data'][st][tr] += [err[idx:idx_st].reshape(2,-1) if st=='xi_ells' else err[idx:idx_st]]
            idx += data_file['data'][st][tr][1].size
    HOD_obj.plot_stats(cat, stats=HOD_obj._stats_to_fit,
                        data=data_file['data'], residuals=True, show=False, save_fn=save_fn_fig)


    print(res, MPI.Wtime()/60, flush=True)

    # data = {'wp': {'x': rp, 'y': wp, 'err': sigma_wp}}
