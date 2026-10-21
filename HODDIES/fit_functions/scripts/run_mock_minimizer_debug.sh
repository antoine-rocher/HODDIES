#!/bin/bash
#SBATCH --job-name=mock_fit_debug
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

# Submit from the repository root:
#   sbatch HODDIES/fit_functions/scripts/run_mock_minimizer_debug.sh
#   sbatch HODDIES/fit_functions/scripts/run_mock_minimizer_debug.sh --tracers ELG --statistics wp xi_ells
#   sbatch HODDIES/fit_functions/scripts/run_mock_minimizer_debug.sh --tracers LRG ELG --statistics wp xi_ells delta_sigma
# Options after the filename override the Python defaults below:
#   sbatch HODDIES/fit_functions/scripts/run_mock_minimizer_debug.sh --maxiter 100
#   sbatch HODDIES/fit_functions/scripts/run_mock_minimizer_debug.sh --check-data
# Change MPI ranks/resources with sbatch options before the filename:
#   sbatch --nodes=2 --ntasks-per-node=4 --cpus-per-task=64 HODDIES/fit_functions/scripts/run_mock_minimizer_debug.sh
# Each rank loads a full small-box catalogue. Lensing also loads field particles.
# The 30-minute limit includes catalogue loading; five iterations are a short check.
# Covariance filenames follow --statistics; individual multipoles use the xi matrix.

set -eo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo 'Submit this script with sbatch to run on compute nodes.' >&2
    exit 1
fi

# Environment setup references optional unset variables; enable nounset afterward.
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u

# Slurm executes a spooled copy, so BASH_SOURCE cannot locate the checkout.
# Set HODDIES_REPO_ROOT to use another checkout.
hoddies_repo_root=${HODDIES_REPO_ROOT:-/global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES}
cd "$hoddies_repo_root"
hoddies_fit_script="$hoddies_repo_root/HODDIES/fit_functions/scripts/run_mock_minimizer.py"
[[ -f "$hoddies_fit_script" ]] || { echo "Missing script: $hoddies_fit_script" >&2; exit 1; }

hoddies_output="$hoddies_repo_root/tmp/mock_fits/${SLURM_JOB_ID}"

# srun inherits the allocation's task count and CPUs/task.
# --mpi defaults popsize to the rank count; ftol defaults to Ndata / 2.
# Rank 0 saves artifacts and automatically updates MOCK_FIT_RESULTS.md.
srun --kill-on-bad-exit=1 \
    python "$hoddies_fit_script" \
    --mpi --tracers ELG LRG --statistics wp xi_ells delta_sigma \
    --method cmaes --maxiter 200 --nthreads "${SLURM_CPUS_PER_TASK}" \
    --output "$hoddies_output" "$@"
