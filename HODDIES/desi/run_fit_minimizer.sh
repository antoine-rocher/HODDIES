#!/bin/bash
#SBATCH --job-name=fit_HOD
#SBATCH -p debug
#SBATCH -C cpu
#SBATCH -N 8
#SBATCH --time=00:30:00
#SBATCH --account desi


source /global/common/software/desi/users/adematti/cosmodesi_environment.sh 2025_03
cd /global/homes/a/arocher/Code/postdoc/HOD/Dev/HODDIES/HODDIES


srun -n 64 -c 32 python /global/homes/a/arocher/Code/postdoc/HOD/Dev/HODDIES/HODDIES/desi/run_minimizer.py --dir_param_file /global/homes/a/arocher/Code/postdoc/HOD/Dev/HODDIES/HODDIES/desi/test_fit_param.yaml

