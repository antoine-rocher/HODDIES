#!/bin/bash

# Perlmutter : 128 cœurs physiques par nœud CPU.
# 16 rangs x 8 threads = 128 -> occupation exacte, sans surallocation.
#
# Le point important : --cpus-per-task doit correspondre à HOD_NUM_THREADS.
# Si numba voit plus de cœurs que SLURM n'en réserve au rang, les rangs se
# disputent les mêmes cœurs et le run est plus lent qu'en série.

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main


export HOD_NUM_THREADS=32
export NUMBA_NUM_THREADS=32
export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
# export OMP_DISPLAY_ENV=verbose   # affiche la configuration au démarrage
# Évite que chaque rang recompile les kernels numba au démarrage
export NUMBA_CACHE_DIR=$SCRATCH/numba_cache
mkdir -p $NUMBA_CACHE_DIR

srun --cpu-bind=cores -N 1 -n 2 --cpus-per-task=32 python /global/homes/a/arocher/Code/postdoc/HODDIES_v2/HODDIES/HODDIES/fit_functions/scripts/run_hod_pocomc_mpi.py