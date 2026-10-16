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
parser.add_argument('--dir_to_save', help='directory to save the results', type=str, default='/global/cfs/cdirs/desi/users/arocher/HODDIES_results/Abcaus_small_boxes_corr')
parser.add_argument('--zsim', help='simulation redshift', type=float, default=0.5)

args = parser.parse_args()

def get_small_simname(parent='/dvs_ro/cfs/cdirs/desi/cosmosim/Abacus/small', zsim=0.5):
    import glob, os, re
    names = sorted(
    os.path.basename(d) 
    for d in glob.glob(os.path.join(parent, '*AbacusSummit_small_c000_ph*'))
    if os.path.isdir(os.path.join(d, 'halos', f'z{zsim:.3f}'))
    )    
    return names

# param = yaml.safe_load(open(args.param_file, 'r'))
param = np.load('/global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/test_data/test_data_minimizer.npy', allow_pickle=True)[()]['param']

# if rank == 0:
#     all_sim_names = get_small_simname(zsim=args.zsim)[::-1]
#     import glob, os, re
#     names = glob.glob(os.path.join(args.dir_to_save, f'z{args.zsim:.3f}', 'smu', '*AbacusSummit_small_c000_ph*'))
#     computed = [re.search(r'AbacusSummit_small_c000_ph\d+', f).group() for f in names]
#     mask = ~np.isin(all_sim_names, computed)
#     all_sim_names = np.array_split(np.asarray(all_sim_names)[mask], size)
#     all_sim_names = np.array_split(np.asarray(all_sim_names), size)
#     sim_names = all_sim_names[0]
#     print(f"Distributing {len(sim_names)} training points over {size} MPI ranks...")
# else:
#     sim_names = None
#     all_sim_names=None


# sim_names = mpicomm.scatter(all_sim_names, root=0)

rppi = TwoPointCorrelationFunction.load('/global/cfs/cdirs/desi/survey/catalogs/edav1/xi/sv3/rppi/allcounts_ELG_NScomb_0.8_1.1_default_angular_bitwise_FKP_log_njack128_nran18_split20.npy')
smu = TwoPointCorrelationFunction.load('/global/cfs/cdirs/desi/survey/catalogs/edav1/xi/sv3/smu/allcounts_ELG_NScomb_0.8_1.1_default_angular_bitwise_FKP_log_njack128_nran18_split20.npy')

import glob, os, re
names = glob.glob(os.path.join(args.dir_to_save, f'z{args.zsim:.3f}', 'delta_sigma', '*AbacusSummit_small_c000_ph*'))
computed = [re.search(r'AbacusSummit_small_c000_ph\d+', f).group() for f in names]

sim_names = [args.param_file]
if sim_names[0] in computed:
    print(f"Skipping simulation {sim_names[0]} at z={args.zsim:.3f} because it has already been computed.")
    sys.exit(0)
for sim_name in sim_names:
    param['hcat']['sim_name'] = sim_name
    print(f"Rank {rank} processing simulation {sim_name} at z={args.zsim:.3f}...")
    param['hcat']['z_simu'] = args.zsim
    param['hcat']['load_field_particles'] = True
    sim_abacus=AbacusSummitSim(**param['hcat'])
    param['setup_logger'] = False
    HOD_obj = HOD(base_catalog=sim_abacus, **param)

    stats= ['rppi', 'smu', 'delta_sigma']
    
    HOD_obj.args['clustering_settings']['xi_rppi']['edges_rppi'] = rppi.edges
    HOD_obj.args['clustering_settings']['xi_smu']['edges_smu'] = smu.edges

    path_to_save = os.path.join(args.dir_to_save, 'z{:.3f}'.format(args.zsim), '{}', 'allcounts_{{}}_{}.npy'.format(sim_name))

    cat = HOD_obj.make_mock_cat()
    for st in stats:
        if st == 'delta_sigma':
            res = HOD_obj.get_delta_sigma(cat, return_dic=True)
            dir_name = os.path.dirname(path_to_save).format(st)
            os.makedirs(dir_name, exist_ok=True)
            for tr in res.keys():
                np.savetxt(os.path.join(dir_name, 'delta_sigma_{}_{}.txt'.format(tr, sim_name)), res[tr])
        else:
            result = HOD_obj.get_cross_twopoint(cat, st, return_pycorr_obj=True)    
            os.makedirs(os.path.dirname(path_to_save).format(st), exist_ok=True)
            for tr in result.keys():
                result[tr].save(path_to_save.format(st, tr))
        
