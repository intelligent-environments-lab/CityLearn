#!/bin/bash
#SBATCH -p normal
#SBATCH -J citylearn_v2
#SBATCH -N 256
#SBATCH --tasks-per-node 1
#SBATCH -t 48:00:00
#SBATCH --mail-user=nweye@utexas.edu
#SBATCH --mail-type=all
#SBATCH -o slurm.out
#SBATCH -A CityLearnV2

# load modules
module load launcher

# activate virtual environment
VIRTUAL_ENVIRONMENT_PATH="path/to/virtual/environment"
source $VIRTUAL_ENVIRONMENT_PATH

# set launcher environment variables
export LAUNCHER_WORKDIR="path/to/workdir"
export LAUNCHER_JOB_FILE="reward_exploration.sh"

${LAUNCHER_DIR}/paramrun