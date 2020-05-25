#!/bin/bash -l
#SBATCH --job-name=hello
# speficity number of nodes 
#SBATCH -N 1
# specify the gpu queue

#SBATCH --partition=csgpu
# Request 2 gpus
#SBATCH --gres=gpu:2
# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node=35

# specify the walltime e.g 20 mins
#SBATCH -t 00:20:00

# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kacper.twardowski@ucdconnect.ie

# run CityLearn dir
cd rl/CityLearn

# command to use
time python train_DDPG.py