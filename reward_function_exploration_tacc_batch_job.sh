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

# set user environment variables
export CLRFE_DATA_PATH="data_reward_function_exploration/Climate_Zone_"
export CLRFE_STATE_ACTION_SPACE_FILENAME="buildings_state_action_space.json"
export CLRFE_EPIOSIDE_TIMESTEPS="8760"
export CLRFE_EPISODES="10"
export CLRFE_SIMULATION_PERIOD_START="0"
export CLRFE_SIMULATION_PERIOD_END=$(($CLRFE_EPIOSIDE_TIMESTEPS - 1))
export CLRFE_DETERMINISTIC_PERIOD_START=$(($CLRFE_EPIOSIDE_TIMESTEPS * 7))
export CLRFE_REGRESSION_BUFFER_CAPACITY="30000"
export CLRFE_START_TRAINING=$(($CLRFE_EPIOSIDE_TIMESTEPS * 3))
export CLRFE_EXPLORATION_PERIOD=$(($CLRFE_START_TRAINING + 1))
export CLRFE_START_REGRESSION=$(($CLRFE_EPIOSIDE_TIMESTEPS * 1))
export CLRFE_ACTION_SCALING_COEF="0.5"
export CLRFE_REWARD_SCALING="5.0"

# set launcher environment variables
export LAUNCHER_WORKDIR="/work/07083/ken658/projects/citylearn/CityLearn"
export LAUNCHER_JOB_FILE="reward_function_exploration"

${LAUNCHER_DIR}/paramrun