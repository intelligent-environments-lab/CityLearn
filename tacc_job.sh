#!/bin/bash
#SBATCH -p skx-normal
#SBATCH -J citylearn_reward_function_exploration
#SBATCH -N 5
#SBATCH -n 240
#SBATCH -t 48:00:00
#SBATCH --mail-user=nweye@utexas.edu
#SBATCH --mail-type=all
#SBATCH -o slurm.out
#SBATCH -A DemandAnalysis

# load modules
module load launcher

# activate virtual environment
source /work/07083/ken658/projects/citylearn/env/bin/activate

# set launcher environment variables
export LAUNCHER_WORKDIR="/work/07083/ken658/projects/citylearn/CityLearn"
export LAUNCHER_JOB_FILE="tacc_launcher_job_file"

${LAUNCHER_DIR}/paramrun