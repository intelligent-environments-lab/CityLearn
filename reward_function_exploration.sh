#!/bin/bash
python -m reward_function_exploration mixed -sf data/reward_function_exploration/mixed.pkl -lf data/reward_function_exploration/mixed.log
python -m reward_function_exploration marlisa -sf data/reward_function_exploration/marlisa.pkl -lf data/reward_function_exploration/marlisa.log
python -m reward_function_exploration ramping_square -sf data/reward_function_exploration/ramping_square.pkl -lf data/reward_function_exploration/ramping_square.log
python -m reward_function_exploration exponential -sf data/reward_function_exploration/exponential.pkl -lf data/reward_function_exploration/exponential.log
python -m reward_function_exploration default -sf data/reward_function_exploration/default.pkl -lf data/reward_function_exploration/default.log
