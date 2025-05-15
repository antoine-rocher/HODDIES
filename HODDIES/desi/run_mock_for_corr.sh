#!/bin/bash
#SBATCH --job-name=corrr_mocks
#SBATCH -p debug
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --time=00:30:00
#SBATCH --account desi


source /global/common/software/desi/users/adematti/cosmodesi_environment.sh 2025_03

for zsim in 1.850  1.550 1.250
do
    for ph in {3000..5000}  
    do
        srun -n 1 -c 16 python /global/homes/a/arocher/Code/postdoc/HOD/Dev/HODDIES/HODDIES/desi/mock_for_corr_small_boxes.py --dir_param_file /global/homes/a/arocher/Code/postdoc/HOD/Dev/HODDIES/HODDIES/desi/parameters_QSO.yaml --phase $ph --zsim $zsim &
        
    done
done
wait

