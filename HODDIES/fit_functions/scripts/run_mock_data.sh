#!/bin/bash


source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# srun -N 1 -o out_$i.txt -e err_$i.txt python /global/homes/a/arocher/Code/postdoc/HOD/Dev/run_trainning_v2.py --start $start --param_file /global/homes/a/arocher/Code/postdoc/HOD/Dev/Y3_param_files/param_$tr\_z$zsim\_wp_xi_c00$cosmo.yaml --nb_point $nb_point &



# for ph in {20..24}
# do
#     for n_mock in {0..9}
#     do
#         srun -n 1 -c 128 python /global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/scripts/run_data_test.py --param_file /global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/scripts/example_HOD_fit_parameter_file.yaml --phase $ph --n_mock $n_mock &
#     done
# done

ph=10
n_mock=3

srun -n 1 -c 63 python /global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/scripts/run_data_test.py --param_file /global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/scripts/example_HOD_fit_parameter_file.yaml --phase $ph --n_mock $n_mock &

ph=8
n_mock=8

srun -n 1 -c 63 python /global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/scripts/run_data_test.py --param_file /global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/scripts/example_HOD_fit_parameter_file.yaml --phase $ph --n_mock $n_mock &

ph=1
n_mock=8

srun -n 1 -c 63 python /global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/scripts/run_data_test.py --param_file /global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/scripts/example_HOD_fit_parameter_file.yaml --phase $ph --n_mock $n_mock &

ph=23
n_mock=4

srun -n 1 -c 63 python /global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/scripts/run_data_test.py --param_file /global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/scripts/example_HOD_fit_parameter_file.yaml --phase $ph --n_mock $n_mock &

wait