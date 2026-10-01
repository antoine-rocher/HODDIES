from mpi4py import MPI
from pycorr import TwoPointCorrelationFunction
import numpy as np
import argparse
import os
import sys
import yaml
from HODDIES.sim_loader import AbacusSummitSim
from HODDIES import HOD
# from sim_loader import setup_logging
# setup_logging()

# --- MPI setup ---
mpicomm = MPI.COMM_WORLD
rank = mpicomm.Get_rank()
size = mpicomm.Get_size()
# --- Argument parsing ---

parser = argparse.ArgumentParser()
parser.add_argument('--param_file', help='parameter file', type=str)
parser.add_argument('--seed_training', help='seed to use for generating training points', type=int, default=1234567489)
parser.add_argument('--run_test_set', help='run test set instead of training set', action='store_true', default=False)
parser.add_argument('--overwrite', help='overwrite the training sample if it already exist', action='store_true', default=False)
args = parser.parse_args()

print(args.overwrite)
param = yaml.safe_load(open(args.param_file, 'r'))

sim_abacus=AbacusSummitSim(**param['hcat'])

HOD_obj = HOD(base_catalog=sim_abacus, setup_logger=False, **param)

HOD_obj._initialize_fit_params()


if args.run_test_set:
    HOD_obj.args['fit_param']['emulator']['N_training_points'] = 400
    HOD_obj.args['fit_param']['emulator']['sampling_type'] = 'lhs'
    path_to_save_point = os.path.join(HOD_obj.args['fit_param']['emulator']["path_to_training_point"], HOD_obj.args['hcat']['sim_name'], 'z{:.3f}'.format(HOD_obj.args['hcat']['z_simu']), 'test_set', HOD_obj.fit_model_name, HOD_obj.args['fit_param']['emulator']["sampling_type"])
    os.makedirs(path_to_save_point, exist_ok=True) 
    if rank == 0:
        HOD_obj.logger.info(f"Running test set with {HOD_obj.args['fit_param']['emulator']['N_training_points']} points and {HOD_obj.args['fit_param']['emulator']['sampling_type']} sampling.")
        HOD_obj.logger.info(f"Test set will be saved in {path_to_save_point}.")   
else:
    path_to_save_point = os.path.join(HOD_obj.args['fit_param']['emulator']["path_to_training_point"], HOD_obj.args['hcat']['sim_name'], 'z{:.3f}'.format(HOD_obj.args['hcat']['z_simu']), 'training_set', HOD_obj.fit_model_name, HOD_obj.args['fit_param']['emulator']["sampling_type"])
    os.makedirs(path_to_save_point, exist_ok=True) 
    if rank == 0:        
        HOD_obj.logger.info(f"Running training set with {HOD_obj.args['fit_param']['emulator']['N_training_points']} points and {HOD_obj.args['fit_param']['emulator']['sampling_type']} sampling.")
        HOD_obj.logger.info(f"Training set will be saved in {path_to_save_point}.")

# Generate training points on root
if rank == 0:
    from HODDIES.fit_functions import genereate_training_points
    training_points = genereate_training_points(HOD_obj.args['fit_param']['emulator']['N_training_points'], HOD_obj.name_params, HOD_obj.priors, sampling_type=HOD_obj.args['fit_param']['emulator']['sampling_type'], path_to_save_training_point=HOD_obj.args['fit_param']['emulator']["path_to_training_point"] if not args.run_test_set else path_to_save_point, rand_seed=args.seed_training)
    index = np.array_split(np.arange(len(training_points)), size)
    HOD_obj.logger.info(f"Distributing {len(training_points)} training points over {size} MPI ranks...")
else:
    training_points = None
    index = None
training_points = mpicomm.bcast(training_points, root=0)
index_tmp = mpicomm.scatter(index, root=0)
HOD_obj.logger.info(f"[Rank {rank}/{size}] Received {len(index_tmp)} training points.")

if rank == 0:
    HOD_obj.logger.info('Using DESI default edges')
result = TwoPointCorrelationFunction.load('/global/cfs/cdirs/desi/survey/catalogs/edav1/xi/sv3/rppi/allcounts_ELG_NScomb_0.8_1.1_default_angular_bitwise_FKP_log_njack128_nran18_split20.npy')[::2][4:-3]
result_smu = TwoPointCorrelationFunction.load('/global/cfs/cdirs/desi/survey/catalogs/edav1/xi/sv3/smu/allcounts_ELG_NScomb_0.8_1.1_default_angular_bitwise_FKP_log_njack128_nran18_split20.npy')[16:-6]
HOD_obj.args['clustering_settings']['xi_rppi']['edges_rppi'] = result.edges
HOD_obj.args['clustering_settings']['xi_smu']['edges_smu'] = result_smu.edges



# --- Compute training points ---
HOD_obj.logger.info(f"[Rank {rank}/{size}] Starting computation for {len(index_tmp)} training points...")
HOD_obj.compute_training_v2(training_points=training_points[index_tmp], start_point=index_tmp[0], path_to_save_point=path_to_save_point, overwrite=args.overwrite)
HOD_obj.logger.info(f"[Rank {rank}/{size}] Finished computation.")

mpicomm.Barrier()
if rank == 0:
    HOD_obj.logger.info("✅ All MPI ranks have finished successfully.")

