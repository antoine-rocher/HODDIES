#!/bin/bash
#SBATCH --job-name=desi_y1_debug
#SBATCH --account=desi
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=8
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

# Submit from the repository root:
#   sbatch HODDIES/fit_functions/scripts/run_desi_y1_minimizer_debug.sh
#   sbatch HODDIES/fit_functions/scripts/run_desi_y1_minimizer_debug.sh --tracers LRG
#   sbatch HODDIES/fit_functions/scripts/run_desi_y1_minimizer_debug.sh --tracers LRG ELG
#   sbatch HODDIES/fit_functions/scripts/run_desi_y1_minimizer_debug.sh --tracers LRG --with-delta-sigma
# Lensing selects z=0.5 and 0.4 < z < 0.6, and loads field particles on each rank.
# Options after the filename are passed to Python, overriding defaults below:
#   sbatch HODDIES/fit_functions/scripts/run_desi_y1_minimizer_debug.sh --maxiter 10
# Use sbatch --nodes=4 ... to change the number of ranks (and default popsize).
# Each rank loads a full catalogue, so keep one rank per node for this example.
# Debug jobs have a 30-minute limit; five iterations are a short test run.
# Loading the full highbase catalogue is included in that wall time.

set -eo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo 'Submit this script with sbatch to run on compute nodes.' >&2
    exit 1
fi

# The environment setup references optional, unset variables; enable nounset after it.
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
set -u

# Slurm executes a spooled copy, so BASH_SOURCE cannot locate the checkout.
# Override HODDIES_REPO_ROOT when using another checkout.
hoddies_repo_root=${HODDIES_REPO_ROOT:-/global/u1/a/arocher/Code/postdoc/HODDIES_v2/HODDIES}
cd "$hoddies_repo_root"
hoddies_fit_script="$hoddies_repo_root/HODDIES/fit_functions/scripts/run_desi_y1_minimizer.py"
[[ -f "$hoddies_fit_script" ]] || { echo "Missing script: $hoddies_fit_script" >&2; exit 1; }

# Perlmutter CPU nodes expose two logical CPUs per physical core.


hoddies_output="$hoddies_repo_root/tmp/desi_y1_debug/${SLURM_JOB_ID}"


# --mpi makes popsize equal to the rank count (8 by default).
# ftol defaults to ndata / 2. Explicit --popsize/--ftol arguments override these.
srun -c 32 -n 64 --kill-on-bad-exit=1 \
    python "$hoddies_fit_script" \
    --mpi --tracers LRG --method cmaes \
    --output "$hoddies_output" "$@" --with-delta-sigma
