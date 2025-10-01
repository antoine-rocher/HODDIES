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
parser.add_argument('--region', help='regions; choice NGC, SGC or GC_comb; default:GC_comb', type=str, nargs='*', choices=['NGC', 'SGC', 'GC_comb'], default='GC_comb')
parser.add_argument('--zsim', help='zsnap', type=float)
parser.add_argument('--output_dir', help='output directory to save the results', type=str)
args = parser.parse_args()
# parser.add_argument('--path_to_save_result', help='path to save fit results', type=str, default=None)

args = parser.parse_args()

import yaml

args_hod = yaml.load(open(args.dir_param_file, 'r'), Loader=yaml.FullLoader)  

args_hod['hcat']['z_simu'] = args.zsim
args_hod['fit_param']['zmin'], args_hod['fit_param']['zmax'] = [0.8,1.1] if args.zsim == 0.95 else [1.1,1.4] if args.zsim == 1.25 else [1.4,1.7] if args.zsim == 1.55 else [1.7,2.1]

print('Run minimizer', flush=True)

HOD_obj= HOD(args=args_hod, path_to_abacus_sim='/global/cfs/cdirs/desi/cosmosim/Abacus')

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()

model_name = get_HOD_model_name(HOD_obj.args)
save_fn= os.path.join(args.output_dir, f"fit_result_{model_name}_{HOD_obj.args['hcat']['sim_name']}_z{HOD_obj.args['hcat']['z_simu']}_{HOD_obj.args['fit_param']['fit_type']}.npy")
print(save_fn, flush=True)
minimizer_options = {"maxiter":5, "popsize":1000, 'xtol':1e-2, 'workers':mpi_comm.Get_size(),  'backend':'mpi'}

# init_params = np.load(save_fn, allow_pickle=True).item()['x']
init_params = None  
res = HOD_obj.run_minimizer(init_params=init_params, minimizer_options=minimizer_options, save_fn=save_fn, mpi_comm=mpi_comm)
print(f"Rank {mpi_rank} finished minimization with result: {res}", flush=True)

if mpi_rank==0:
    from pathlib import Path
    save_fn_fig = save_fn[:-3] + 'png'
    # save_fn_fig.rename(save_fn_fig.with_suffix('.png'))
    fig = HOD_obj.plot_bf_data(save=save_fn_fig, fix_seed=10)
    print(res, MPI.Wtime()/60, flush=True)

