#!/bin/bash
#SBATCH --job-name=fit_HOD
#SBATCH -p debug
#SBATCH -C cpu
#SBATCH -N 8
#SBATCH --time=00:30:00
#SBATCH --account desi

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# srun -N 1 -o out_$i.txt -e err_$i.txt python /global/homes/a/arocher/Code/postdoc/HOD/Dev/run_trainning_v2.py --start $start --param_file /global/homes/a/arocher/Code/postdoc/HOD/Dev/Y3_param_files/param_$tr\_z$zsim\_wp_xi_c00$cosmo.yaml --nb_point $nb_point &


# srun -N 4 -n 16 -c 64 python /global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/scripts/run_trainning.py --param_file /global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/scripts/example_HOD_fit_parameter_file_v2.yaml &

srun -N 4 -n 16 -c 64 python /global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/scripts/run_trainning.py --param_file /global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/scripts/example_HOD_fit_parameter_file_v2.yaml --run_test_set &

wait