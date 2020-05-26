#!/bin/bash -l
#SBATCH --job-name=DDPG_train
# speficity number of nodes 
#SBATCH -N 1
# specify the gpu queue

#SBATCH --partition=gpu
# Request 2 gpu
#SBATCH --gres=gpu:2
# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node=1

# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=myemailaddress@ucd.ie

# Load Anaconda as a Module
module load anaconda

# command to use
cd rl/CityLearn
conda activate citylearn
time mpiexec python train_DDPG.py
