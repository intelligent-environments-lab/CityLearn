#!/bin/bash
#SBATCH -p skx-dev
#SBATCH -J citylearn_reward_function_exploration
#SBATCH -N 4
#SBATCH -n 192
#SBATCH -t 2:00:00
#SBATCH --mail-user=nweye@utexas.edu
#SBATCH --mail-type=all
#SBATCH -o slurm.out
#SBATCH -A DemandAnalysis

# moad modules
module load launcher

# set environment variables
export LAUNCHER_WORKDIR=/work/07083/ken658/projects/citylearn/CityLearn
export LAUNCHER_JOB_FILE=reward_function_exploration.sh

# activate virtual environment
source /work/07083/ken658/projects/citylearn/env/bin/activate

${LAUNCHER_DIR}/paramrun