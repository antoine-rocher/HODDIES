#!/bin/bash
#SBATCH --job-name=fit_HOD
#SBATCH -p debug
#SBATCH -C cpu
#SBATCH -N 4
#SBATCH --time=00:30:00
#SBATCH --account desi


source /global/common/software/desi/users/adematti/cosmodesi_environment.sh 2025_03
cd /global/homes/a/arocher/Code/postdoc/HOD/Dev/HODDIES/HODDIES/desi/Tuto_hod_minimizer
# srun -N 8 -n 64 -c 32 python /global/homes/a/arocher/Code/postdoc/HOD/Dev/HODDIES/HODDIES/desi/run_minimizer.py --dir_param_file /global/homes/a/arocher/Code/postdoc/HOD/Dev/HODDIES/HODDIES/desi/test_fit_param_QSO.yaml --zsim 1.85

tracer=LRG # ELG or QSO
zsim=0.5 # 0.5, 0,725 or 0.95 for LRG / 0.95 1.175, 1.325 / QSO 0.95 1.25 1.55 1.85

srun -N 4 -n 32 -c 32 python run_minimizer_desi_hod.py --tracer LRG --zsim $zsim --output_dir /global/homes/a/arocher/Code/postdoc/HOD/Dev/HODDIES/HODDIES/desi/Tuto_hod_minimizer/results


