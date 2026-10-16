from HODDIES.environment_func import compute_env_shear_from_particles
from HODDIES.sim_loader.abacus_io import load_abacus_particle_field, get_abacus_boxsize, check_abacus_sim_name, check_abacus_redshift_particle_available
import argparse
import os 

parser = argparse.ArgumentParser()

parser.add_argument('--dir_abacus_sim', help='path to the directory where particle file are saved ', type=str, default='/dvs_ro/cfs/cdirs/desi/cosmosim/Abacus/small')
parser.add_argument('--sim_name', help='name of the simulation, e.g. AbacusSummit_base_c000_ph000', type=str)
parser.add_argument('--z_sim', help='redshift of the simulation', type=float)
parser.add_argument('--subsample', help='Particle subsample to use: "A", "B" or "AB"', type=str, default='A')
parser.add_argument('--subset', help='fraction of particles to use', type=float, default=0.1)
parser.add_argument('--cell_size', help='size of the cells for the environmental shear map', type=float, default=5.0)
parser.add_argument('--R', help='radius for the environmental shear map', type=float, default=1.5)
parser.add_argument('--path_to_save', help='path to save the environmental shear map', type=str, default='/global/homes/a/arocher/users_arocher/HODDIES_data/environemental_quantities/Abacus')
parser.add_argument('--seed', help='random seed for reproducibility', type=int, default=42)

args = parser.parse_args()

check_abacus_sim_name(args.sim_name)
check_abacus_redshift_particle_available(args.z_sim)
dir_abacus_sim = os.path.join(args.dir_abacus_sim, args.sim_name, "halos",  "z{:.3f}".format(args.z_sim))


data = load_abacus_particle_field(dir_abacus_sim, fraction=args.subset, subsample=args.subsample, seed=args.seed, verbose=True) 
boxisze = get_abacus_boxsize(args.sim_name)

path_to_save = os.path.join(args.path_to_save, 'env_shear_map_{sim_name}_z{zsim:.3f}.h5'.format(sim_name=args.sim_name, zsim=args.z_sim))
print(f"Saving environmental shear map to: {path_to_save}", flush=True)
dsmo, shear = compute_env_shear_from_particles(data['pos'], boxisze, cell_size=args.cell_size, R=args.R, path_to_save=path_to_save)

