from HODDIES import HOD
import numpy as np
import os 
from mpi4py import MPI
import argparse
from HODDIES.fits_functions import compute_chi2

def likelihood(new_params, self, fix_seed=10, verbose=False, **kwargs):

    # print(*zip(self.name_params, new_params), flush=True)
    new_params= np.array([new_params])
    
    new_params.dtype = [(name, dt) for name, dt in zip(self.name_params, ['float64']*len(self.name_params))]
    HOD_obj.update_new_param(new_params, self.name_params)
    cats = HOD_obj.make_mock_cat(fix_seed=fix_seed, verbose=verbose) 

    result = {}
    if 'wp' in HOD_obj.args['fit_param']["fit_type"]:
        result['wp']= HOD_obj.get_crosswp(cats, tracers=HOD_obj.args['tracers'], verbose=verbose)
    if 'xi' in HOD_obj.args['fit_param']["fit_type"]:
        result['xi'] = HOD_obj.get_cross2PCF(cats, tracers=HOD_obj.args['tracers'], verbose=verbose)

    stats = ['wp', 'xi'] if ('wp' in HOD_obj.args['fit_param']["fit_type"]) & ('xi' in HOD_obj.args['fit_param']["fit_type"]) else ['wp'] if ('wp' in HOD_obj.args['fit_param']["fit_type"]) else ['xi']
    res = {}
    comb_trs = result[stats[0]].keys() 
    res = np.hstack([np.hstack([np.hstack(result[stat][comb_tr][1])for stat in stats]) for comb_tr in comb_trs])

    chi2 = compute_chi2(res, self.data, inv_Cov2=self.inv_cov2)
    # print('chi2=', chi2, flush=True)

    return chi2


def my_prior_transform(cube, priors):
    params = cube.copy()

    # transform location parameter: uniform prior
    for ii, [lo, hi] in enumerate(priors):
        params[ii] = cube[ii] * (hi - lo) + lo

    return params

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

args = parser.parse_args()
# parser.add_argument('--path_to_save_result', help='path to save fit results', type=str, default=None)

args = parser.parse_args()

import yaml

args_hod = yaml.load(open(args.dir_param_file, 'r'), Loader=yaml.FullLoader)  

args_hod['hcat']['Abacus']['z_simu'] = args.zsim
args_hod['fit_param']['zmin'], args_hod['fit_param']['zmax'] = [0.8,1.1] if args.zsim == 0.95 else [1.1,1.4] if args.zsim == 1.25 else [1.4,1.7] if args.zsim == 1.55 else [1.7,2.1]

print('Run MCMC', flush=True)

HOD_obj= HOD(args=args_hod, path_to_abacus_sim='/global/cfs/cdirs/desi/cosmosim/Abacus')

mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()


model_name = get_HOD_model_name(HOD_obj.args)
save_dir= f"/global/homes/a/arocher/Code/postdoc/HOD/Dev/HODDIES/HODDIES/desi/fit_result_{model_name}_{HOD_obj.args['hcat']['sim_name']}_z{HOD_obj.args['hcat']['z_simu']}_{HOD_obj.args['fit_param']['fit_type']}"

HOD_obj.initialize_fit()
param_names, priors = HOD_obj.get_param_and_prior()


import ultranest
from functools import partial

partial_LH = partial(likelihood, self=HOD_obj, fix_seed=10, verbose=False)
sampler = ultranest.ReactiveNestedSampler(param_names, partial_LH, partial(my_prior_transform, priors=priors), resume=True, 
                                          log_dir=save_dir)


res = sampler.run()


