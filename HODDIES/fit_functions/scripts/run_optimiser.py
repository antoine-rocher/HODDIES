from HODDIES import HOD
from HODDIES.sim_loader import AbacusSummitSim
import numpy as np
import os 
from mpi4py import MPI
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--param_file', help='path to param file', type=str)
args = parser.parse_args()
# parser.add_argument('--path_to_save_result', help='path to save fit results', type=str, default=None)

args = parser.parse_args()

import yaml


param = yaml.safe_load(open(args.param_file, 'r'))

sim_abacus=AbacusSummitSim(**param['hcat'])

HOD_obj = HOD(base_catalog=sim_abacus, setup_logger=False, **param)


data_vec = np.loadtxt('/global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/scripts/results_optimizer/data_test_optimizer.txt')
mcov= np.loadtxt('/global/cfs/cdirs/desi/users/arocher/HODDIES_results/Abcaus_small_boxes_corr/z0.500/Mcov_small_box_z0.500_wp_xi_ells_LRG_SHOD_6p.txt')
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

res = HOD_obj.run_minimizer(mpi_comm=mpi_comm, **minimizer_options)
print(f"Rank {mpi_rank} finished minimization with result: {res}", flush=True)

if mpi_rank==0:
    new_params = np.array([ HOD_obj.result_fit['x']])
    new_params.dtype = [(name, dt) for name, dt in zip(HOD_obj.name_params, ['float64']*len(HOD_obj.name_params))]
    HOD_obj.update_new_param(new_params, HOD_obj.name_params)
    cat = HOD_obj.make_mock_cat(fix_seed=10)
    dir_to_save = '/global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/scripts/results_optimizer'
    save_fn_fig = os.path.join(dir_to_save, 'best_fit.png')
    data_dict = np.load('/global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/scripts/results_optimizer/data_test_optimizer.npy', allow_pickle=True)[()]
    std_data = np.diag(mcov)**0.5 
    idx_rp = len(data_dict['wp']['LRG_LRG'][0])
    data_dict['wp']['LRG_LRG'] += [std_data[:idx_rp]]
    data_dict['xi_ells']['LRG_LRG'] += [std_data[idx_rp:].reshape(2,-1)]
    HOD_obj.plot_stats(cat, stats=HOD_obj._stats_to_fit,
                        data=data_dict, residuals=True, show=False, save_fn=save_fn_fig)

    print(res, MPI.Wtime()/60, flush=True)

    # data = {'wp': {'x': rp, 'y': wp, 'err': sigma_wp}}
