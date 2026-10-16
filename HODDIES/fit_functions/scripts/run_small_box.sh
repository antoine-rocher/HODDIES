#!/bin/bash
#SBATCH --job-name=fit_HOD
#SBATCH -p debug
#SBATCH -C cpu
#SBATCH -N 8
#SBATCH --time=00:30:00
#SBATCH --account desi

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
base_dir="/dvs_ro/cfs/cdirs/desi/cosmosim/Abacus/small"

for sim_dir in "$base_dir"/AbacusSummit_small_c000_ph*; do
    [[ -d "$sim_dir" ]] || continue
    sim_name="${sim_dir##*/}"

    echo "$sim_name"
    srun -n 1 -c 32 python /global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/scripts/mock_for_corr_small_boxes.py --param_file "$sim_name" &
done
wait