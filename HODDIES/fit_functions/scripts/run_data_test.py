import time
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


# --- Argument parsing ---

parser = argparse.ArgumentParser()
parser.add_argument('--param_file', help='parameter file', type=str)
parser.add_argument('--seed', help='seed to use for generating mocks', type=int, default=1234567489)
parser.add_argument('--n_real', help='number of realisation', type=int, default=20)
parser.add_argument('--overwrite', help='overwrite the training sample if it already exist', action='store_true', default=False)
parser.add_argument('--phase', help='phase of the simulation', type=str, default='1')
parser.add_argument('--n_mock', help='mock number', type=int)

args = parser.parse_args()

path_to_save_point = '/global/cfs/cdirs/desi/users/arocher/HODDIES_results/test/AbacusSummit_base_c000_ph000/z0.500/mock_data/'
param = yaml.safe_load(open(args.param_file, 'r'))
param['hcat']['sim_name'] = 'AbacusSummit_base_c000_ph{}'.format(args.phase.zfill(3))
if (os.path.exists(os.path.join(path_to_save_point, 'mock_data_{}_{}_{}.npy'.format(param['hcat']['z_simu'], param['hcat']['sim_name'], args.n_mock))) | args.overwrite):
    exit('File already exists, use --overwrite to overwrite it.')


sim_abacus=AbacusSummitSim(**param['hcat'])

HOD_obj = HOD(base_catalog=sim_abacus, setup_logger=True, **param)

HOD_obj._initialize_fit_params()


# Generate mock data 
from HODDIES.fit_functions import genereate_training_points
training_points = genereate_training_points(240, HOD_obj.name_params, HOD_obj.priors, sampling_type='lhs', rand_seed=args.seed)

rng = np.random.default_rng(args.seed)
index = np.arange(240)
rng.shuffle(index)
index_tmp=index.reshape(24,10)[int(args.phase)-1]



result = TwoPointCorrelationFunction.load('/global/cfs/cdirs/desi/survey/catalogs/edav1/xi/sv3/rppi/allcounts_ELG_NScomb_0.8_1.1_default_angular_bitwise_FKP_log_njack128_nran18_split20.npy')[::2][4:-3]
result_smu = TwoPointCorrelationFunction.load('/global/cfs/cdirs/desi/survey/catalogs/edav1/xi/sv3/smu/allcounts_ELG_NScomb_0.8_1.1_default_angular_bitwise_FKP_log_njack128_nran18_split20.npy')[16:-6]
HOD_obj.args['clustering_settings']['xi_rppi']['edges_rppi'] = result.edges
HOD_obj.args['clustering_settings']['xi_smu']['edges_smu'] = result_smu.edges


import numpy as np


def concatenate_merged(*merged_dicts):
    """
    Concatenate several merged HOD dictionaries along the sample axis.

    Each statistic is stored as ``{tracer: [coord_0, ..., coord_m, measurement]}``.
    The separation vectors are kept once, taken from the first input; only the
    measurement (the last entry) is stacked.

    Whether the per-file measurement already carries a sample axis is decided by
    comparing its dimensionality against ``max(ncoord, 1)``, the dimensionality
    of a single sample. The ``max`` covers coordinate-free statistics such as
    CIC, which still have a bin axis but store no separation vector:

    ==================  =======  ==========  ==========================
    statistic           ncoord   per-file    output
    ==================  =======  ==========  ==========================
    CIC                 0        (n_b,)      (X, n_b)
    CIC                 0        (n_i, n_b)  (X, n_b)
    wp                  1        (n_rp,)     (X, n_rp)
    wp                  1        (n_i, n_rp) (X, n_rp)
    xi_smu / xi_rppi    2        (n_s, n_mu) (X, n_s, n_mu)
    xi_smu / xi_rppi    2        (n_i, n_s, n_mu)  (X, n_s, n_mu)
    ==================  =======  ==========  ==========================

    so a 2D statistic never gets flattened into ``(X * n_s, n_mu)``.

    Parameters
    ----------
    *merged_dicts : dict
        Two or more merged dictionaries.

    Returns
    -------
    dict
        Concatenated dictionary, with ``X = sum(n_i)`` samples.
    """
    if not merged_dicts:
        raise ValueError('nothing to concatenate')
    if len(merged_dicts) == 1:
        return merged_dicts[0]

    ref = merged_dicts[0]
    out = {}

    for key, val in ref.items():

        # plain top-level array (e.g. hod_fit_param): always has a sample axis
        if not isinstance(val, dict):
            out[key] = np.concatenate(
                [np.atleast_2d(np.asarray(m[key])) for m in merged_dicts], axis=0)
            continue

        # statistic: keep the separations once, stack the measurement
        out[key] = {}
        for tracer, entries in val.items():
            ncoord = len(entries) - 1
            base   = max(ncoord, 1)          # ndim of a single sample
            blocks = [np.asarray(m[key][tracer][-1]) for m in merged_dicts]

            # ndim == base     -> one sample per file, add the axis
            # ndim == base + 1 -> sample axis already present, concatenate
            promoted = []
            for j, b in enumerate(blocks):
                if b.ndim == base:
                    promoted.append(b[None])
                elif b.ndim == base + 1:
                    promoted.append(b)
                else:
                    raise ValueError(
                        f'input {j}, {key}/{tracer}: measurement has ndim={b.ndim}, '
                        f'expected {base} or {base + 1} for {ncoord} separation '
                        f'vector(s)')

            shapes = {b.shape[1:] for b in promoted}
            if len(shapes) != 1:
                raise ValueError(
                    f'{key}/{tracer}: measurement shapes differ across inputs: '
                    f'{[b.shape for b in promoted]}')

            out[key][tracer] = list(entries[:-1]) + [np.concatenate(promoted, axis=0)]

    return out

# --- Compute training points ---

tracers = HOD_obj._tracer_to_fit
name_param_tr = {}
for tr in tracers:
    name_param_tr[tr] = [x.split(f'_{tr}')[0] for x in HOD_obj.name_params if tr in x]
param = training_points[index_tmp][args.n_mock]
result = {}
start = time.time()
for tr in tracers:
    seed = args.seed + args.n_mock + int(args.phase)
    # var_name_tr = np.array(training_points.dtype.names)[np.array([tr in var for var in training_points.dtype.names])].tolist()
    idx = np.where([tr in vv for vv in HOD_obj.name_params])[0].tolist()
    tr_par = list(HOD_obj.name_params[i] for i in idx)
    HOD_obj.args[tr].update(dict(zip(name_param_tr[tr], param[tr_par])))
    if 'assembly_bias' in HOD_obj.args['fit_param']['priors'][tr].keys():
        for var in HOD_obj.args['fit_param']['priors'][tr]['assembly_bias'].keys():
            HOD_obj.args[tr]['assembly_bias'][var] = [param[f'ab_{var}_cen_{tr}'], param[f'ab_{var}_sat_{tr}']]

    # result[tr] = HOD_obj.args[tr].copy()
    
HOD_obj.logger.info('Compute HOD:' + ', '.join('{}: {:.3f}'.format(tt, ttt) for tt, ttt in zip(HOD_obj.name_params, param)))

for ii in range(args.n_real):
    seed = args.seed + args.n_mock + ii + int(args.phase)
    cat = HOD_obj.make_mock_cat(tracers, fix_seed=seed)
    if ii == 0:
        result.update(HOD_obj.compute_stats(cat, stat=HOD_obj.args['fit_param']['fit_statistics'], tracers=HOD_obj._tracers(), verbose=False))
    else:
        result_tmp = HOD_obj.compute_stats(cat, stat=HOD_obj.args['fit_param']['fit_statistics'], tracers=HOD_obj._tracers(), verbose=False)
        result = concatenate_merged(result, result_tmp)

result['comb_trs'] = HOD_obj.get_comb_tr_list(HOD_obj._tracers())
result['hod_fit_param'] = param
result['param_file'] = HOD_obj.args

np.save(os.path.join(path_to_save_point, 'mock_data_{}_{}_{}.npy'.format(HOD_obj.args['hcat']['z_simu'], HOD_obj.args['hcat']['sim_name'], args.n_mock)), result)
HOD_obj.logger.info('Point {} done {:.2f}'.format(args.n_mock, time.time()-start))

