# Deep Multi-Agent Reinforcement Learning for Decentralised Continuous Cooperative Control

This repo contains the code that was used in the paper: 
[Deep Multi-Agent Reinforcement Learning for Decentralised Continuous Cooperative Control](https://arxiv.org/pdf/2003.06709.pdf)
and includes implementations of the following continuous cooperative multi-agent reinforcement learning (MARL) algorithms:
- COMIX
- COVDN
- IQL
- MADDPG
- FacMADDPG

Note: this codebase is built on top of the [PyMARL](https://github.com/oxwhirl/pymarl/) framework for MARL algorithms. 
Please refer to that repo for more documentation. It has also been restructured since the original paper, and the results may vary from those reported in the paper.

## Installation instructions

Build the Dockerfile using 
```
bash build.sh
```

## Environments

### Multi-Agent MuJoCo 
We benchmark our continuous MARL algorithms on a diverse variety of [Multi-Agent MuJoCo](https://github.com/schroederdewitt/multiagent_mujoco) tasks. 

Based on the popular single-agent [MuJoCo](https://github.com/openai/mujoco-py) benchmark suite from OpenAI Gym, 
we developed Multi-Agent MuJoCo that consists of a wide variety of robotic control tasks in which multiple agents within 
a single robot have to solve a task cooperatively. 
The number of agents and level of partial observability in each task can be finely configured. 

For more details about this benchmark, please check the [Multi-Agent MuJoCo](https://github.com/schroederdewitt/multiagent_mujoco) repo.

### Continuous Predator-Prey
We also tested our continuous MARL algorithms on a simple variant of the simple tag environment from [Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs).

To obtain a purely cooperative environment, we replace the prey's policy by a hard-coded heuristic, that, at any time step, 
moves the prey to the sampled position with the largest distance to the closest predator. If one of the cooperative agents
collides with the prey, a team reward of +10 is given; otherwise, no reward is given. 

For more details about the environment, please see the Appendix of the paper.

## Run an experiment

Run an ALGORITHM from the folder `src/config/algs`
in an ENVIRONMENT from the folder `src/config/envs`
on a specific GPU using some PARAMETERS:
```
bash run_gpu.sh <GPU> python3 src/main.py --config=<ALGORITHM> --env-config=<ENVIRONMENT> with <PARAMETERS>
```

As an example, to run the COMIX algorithm on our continuous Predator-Prey task for 1mil timesteps using docker:
```
bash run_gpu.sh <GPU> python3 src/main.py --config=comix --env-config=particle with env_args.scenario=simple_tag_coop t_max=1000000
```

The config files (src/config/algs/*.yaml) contain default hyper-parameters for the respective algorithms. 
These were sometimes changed when running the experiments on different tasks. 
Please see the Appendix of the paper for the exact hyper-parameters used.

## Paper citation

If you used this code in your research or found it helpful, please consider citing the following paper:

<pre>
@article{de2020deep,
  title={Deep Multi-Agent Reinforcement Learning for Decentralised Continuous Cooperative Control},
  author={de Witt, Christian Schroeder and Peng, Bei and Kamienny, Pierre-Alexandre and Torr, Philip and B{\"o}hmer, Wendelin and Whiteson, Shimon},
  journal={arXiv preprint arXiv:2003.06709},
  year={2020}
}
</pre>