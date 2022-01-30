#!/bin/bash
#SBATCH -p normal
#SBATCH -J citylearn_grid_search_c8
#SBATCH -N 8
#SBATCH --tasks-per-node 1
#SBATCH -t 24:00:00
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
export LAUNCHER_JOB_FILE="tacc_launcher_job_c8"

${LAUNCHER_DIR}/paramrun
