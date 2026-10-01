#!/bin/bash



source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# # GPU
# for i in {0..15}
# do

# srun -n 1 --gpus-per-task=1 python /global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/scripts/run_MCMC_fit.py $i 15 &

# done
# wait

for i in {0..239}
do

srun -N 1 -n 1 --cpus-per-task=1 python /global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/scripts/run_MCMC_fit.py $i 1 &

done
wait