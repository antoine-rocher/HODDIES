import os
from read_particles import read_gadget2_multi
from HODDIES.environment_func import compute_env_shear_from_particles
import argparse
import glob

parser = argparse.ArgumentParser()

parser.add_argument('--dir_particle_file', help='path to the directory where particle file are saved ', type=str, default='/dvs_ro/cfs/cdirs/desi/mocks/cai/Uchuu-SHAM/Uchuu-dm-particles')
parser.add_argument('--nthreads', help='number of threads to use', type=int, default=64)
parser.add_argument('--subset', help='fraction of particles to use', type=float, default=0.05)
parser.add_argument('--cell_size', help='size of the cells for the environmental shear map', type=float, default=5.0)
parser.add_argument('--R', help='radius for the environmental shear map', type=float, default=1.5)
parser.add_argument('--path_to_save', help='path to save the environmental shear map', type=str, default='/global/homes/a/arocher/users_arocher/HODDIES_data/environemental_quantities/Uchuu/')
args = parser.parse_args()




dir_path = glob.glob('/*')
for fn in os.listdir(args.dir_particle_file):
    if 'snapdir' not in fn:
        raise ValueError(f"Uchuu file {fn} must contain 'snapdir' in its name. Please check the directory name.")

    snapdir = fn[fn.find('snapdir'):fn.find('snapdir')+11]
    path_to_save = args.path_to_save + 'env_shear_map_Uchuu_{snapdir}.h5'.format(snapdir=snapdir)
    print(f"Saving environment and shear map to {path_to_save}", flush=True)
    if os.path.exists(path_to_save):
        continue

    data, header = read_gadget2_multi(fn+'/', n_threads=args.nthreads, subset_fraction=args.subset)

    print("\n=== Snapshot Summary ===", flush=True)
    print(f"Total particles read: {len(data['id']):,}", flush=True)
    print(f"Particles Mass: {header['Massarr'][1]} 10^10 Msun/h", flush=True)
    print(f"BoxSize: {header['BoxSize']} Mpc/h", flush=True)
    print(f"Omega0: {header['Omega0']}, OmegaLambda: {header['OmegaLambda']}", flush=True)


    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
    dsmo, shear = compute_env_shear_from_particles(data['pos'], header['BoxSize'], cell_size=args.cell_size, R=args.R, path_to_save=path_to_save)
