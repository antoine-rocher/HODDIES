from HODDIES import HOD
import numpy as np
import os 
from mpi4py import MPI
import argparse


def get_HOD_model_name(args):
    fit_model_name=[]
    for tr in args['tracers']:
        n_param = len(args['fit_param']['priors'][tr])
        if 'assembly_bias' in args['fit_param']['priors'][tr].keys():
            n_param+= len(args['fit_param']['priors'][tr]['assembly_bias']) -1
        
        ext = '+conf' if args[tr]['conformity_bias'] else ''
        ext += '+exp' if ('exp_frac' in HOD_obj.args['fit_param']['priors'][tr].keys()) | (args[tr]['exp_frac'] !=0) else ''
        ext += '+'+'+'.join([f'ab_{var}' for var in args['fit_param']['priors'][tr]['assembly_bias'].keys()]) if 'assembly_bias' in args['fit_param']['priors'][tr].keys() else ''
        fit_model_name += ['{}_{}_{}p'.format(tr, args[tr]['HOD_model']+ext, n_param)]
    
    fit_model_name = '_'.join(fit_model_name)
    return fit_model_name


parser = argparse.ArgumentParser()
parser.add_argument('--dir_param_file', help='path to param file', type=str)

args = parser.parse_args()

HOD_obj= HOD(param_file=args.dir_param_file, path_to_abacus_sim='/global/cfs/cdirs/desi/cosmosim/Abacus')

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()


model_name = get_HOD_model_name(HOD_obj.args)
save_fn= f"fit_result_{model_name}_{HOD_obj.args['hcat']['sim_name']}_z{HOD_obj.args['hcat']['z_simu']}_{HOD_obj.args['fit_param']['fit_type']}.npy"

minimizer_options = {"maxiter":100, "popsize": 40, 'xtol':1e-6, 'workers':mpi_comm.Get_size(),  'backend':'mpi'}
init_params = [12.82663346, 13.82364144,  1.00700969,  0.87145873, 12.72177864, 0.21941988]

res = HOD_obj.run_minimizer(init_params=init_params, minimizer_options=minimizer_options, save_fn=save_fn, mpi_comm=mpi_comm)
        
if mpi_rank==0:
    print(res, MPI.Wtime()/60, flush=True)

