#!/bin/bash


# From the repository root:
#   sbatch checks_pipeline/run_hod_benchmark.sh
#   sbatch checks_pipeline/run_hod_benchmark.sh --boxes abacus_base --profiles nfw
#   sbatch checks_pipeline/run_hod_benchmark.sh --resume --output /path/to/previous/output
# Inside an existing CPU allocation:
#   bash checks_pipeline/run_hod_benchmark.sh --resume
# One process uses the full CPU node. Box/profile workers run sequentially so
# large catalogues do not coexist or compete for CPU/memory. 256 means logical
# CPUs; on a standard Perlmutter CPU node the final point uses SMT.
set -eo pipefail
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Submit with sbatch.' >&2; exit 1; }
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u
SLURM_CPUS_PER_TASK=256
hoddies_benchmark_cpus=${SLURM_CPUS_PER_TASK:-256}
hoddies_repo_root=${HODDIES_REPO_ROOT:-/global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES}
cd "$hoddies_repo_root"
export OMP_NUM_THREADS=$hoddies_benchmark_cpus
export NUMBA_NUM_THREADS=$hoddies_benchmark_cpus
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/hoddies-benchmark-numba-${SLURM_JOB_ID}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/hoddies-benchmark-matplotlib-${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"

hoddies_benchmark_output="$hoddies_repo_root/checks_pipeline/results/hod_benchmark/${SLURM_JOB_ID}"
srun --nodes=1 --ntasks=1 --cpus-per-task="$hoddies_benchmark_cpus" --cpu-bind=cores --kill-on-bad-exit=1 \
    python checks_pipeline/benchmark_hod_pipeline.py \
    --output "$hoddies_benchmark_output" "$@" --resume
