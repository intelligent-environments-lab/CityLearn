from citylearn import  CityLearn, building_loader, auto_size
from energy_models import HeatPump, EnergyStorage, Building
from utils import create_env, get_agents, parse_arguments
from reward_function import reward_function

import matplotlib.pyplot as plt
import numpy as np

import logging
from pathlib import Path
import sys
import time

logger = logging.getLogger('spam_application')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

loss_coeff = 0.19/24
efficiency = 1.0

# TODO: Extend to multiple buildings (assuming we have the same cooling demand pattern for the buildings for a time period).
# TODO: Execute the optimal policy with the environment to assert the cost we have found.

# TODO: Ensure positive transfer irrespective of start state

def run_dp(cooling_pump, cooling_storage, building, **kwargs):

    global loss_coeff
    global efficiency

    # Functions to discretize a continuous quantity in level numbers (levels are from 0 to steps - 1).
    # 1. Get level number from value
    # 2. get value from level number
    # For example -1.0 to 1.0 with 3 steps will have levels
    # 0 -> -1.0
    # 1 -> 0.0
    # 2 -> 1.0

    # Gives level just below val (flooring)
    def get_level(min_val, max_val, val, level_cnt):
        slab_size = (max_val - min_val)/(level_cnt-1)
        return int((val - min_val)/slab_size)

    # Gives val of level
    def get_val(min_val, max_val, level, level_cnt):
        slab_size = (max_val - min_val)/(level_cnt-1)
        return slab_size*level + min_val


    end_time = kwargs["end_time"]
    start_time = kwargs["start_time"]
    action_levels = kwargs["action_levels"]
    action_min = kwargs["min_action_val"]
    action_max = kwargs["max_action_val"]

    charge_levels = kwargs["charge_levels"]
    charge_min = kwargs["min_charge_val"]
    charge_max = kwargs["max_charge_val"]

    sim_results = building.sim_results

    # Cost for time stamps start_time to end_time + 1 (the last one is just added for ease and have 0. cost
    # for all charge levels)
    cost_array = np.full((end_time - start_time + 2, charge_levels, action_levels), np.inf)
    cost_array[end_time+1-start_time] = np.zeros((charge_levels, action_levels))

    clipped_action_val = np.full((end_time - start_time + 2, charge_levels, action_levels), np.inf)

    # TODO (Readability): Create numpy array that can be indexed using time_step instead of time_step - start_time
    # cost = lambda t, c, a: cost_array[t-start_time][c][a]

    logger.debug("ES capacity {0}\n".format(cooling_storage.capacity))
    # logger.debug("Cooling demand\n{0}\n".format(sim_results['cooling_demand'][start_time:end_time+1]))
    # logger.debug("Outside temps\n{0}\n".format(sim_results['t_out'][start_time:end_time+1]))

    elec_no_es = []
    cooling_demand = []

    for t in range(start_time, end_time+1):
        cooling_pump.set_cop(t, sim_results['t_out'][t])
        e = cooling_pump.get_electric_consumption_cooling(sim_results['cooling_demand'][t])
        elec_no_es.append(e*e)

    logger.debug("Cost without ES {0}\n".format(np.sqrt(np.sum(elec_no_es))))

    # Store the optimal action sequence
    # optimal_action_sequence = np.zeros((end_time - start_time + 2))
    optimal_action_val = np.zeros((end_time - start_time + 2))

    for time_step in range(end_time, start_time-1, -1):
        for charge_level in range(charge_levels-1, -1, -1):
            # Minor optimization for start time
            if time_step == start_time and charge_level != 0:
                continue

            for action in range(action_levels-1, -1, -1):
                charge_on_es = get_val(0., 1., charge_level, charge_levels)
                charge_on_es = charge_on_es*(1-loss_coeff)
                charge_transfer = get_val(-1, 1, action, action_levels)

                logger.debug("Time {0} charge {1:.2f} action {2:.2f}".format(time_step, charge_on_es, charge_transfer))

                # If action tries to discharge more than what is available, skip it. All further actions in the loop
                # will discharge more, so break.
                if -1 * min(charge_transfer, 0) > charge_on_es:
                    break

                # Cannot charge more than capaciity, skip.
                if max(charge_transfer, 0) > 1 - charge_on_es:
                    continue
                
                # TODO: This is a hack, fix this.
                cooling_pump.time_step = time_step
                break_after_this_action = False

                # If we are discharging more than the required cooling demand it is valid, but it doesn't make sense to check higher
                # discharging actions after this action. So break after this one action.
                if charge_transfer < 0 and -1 * charge_transfer * cooling_storage.capacity * efficiency >= sim_results['cooling_demand'][time_step]:
                    break_after_this_action = True

                # Adapted from set_storage_cooling()
                cooling_power_avail = cooling_pump.get_max_cooling_power(t_source_cooling = sim_results['t_out'][time_step]) - sim_results['cooling_demand'][time_step]
                if charge_transfer >= 0:
                    maybe_cooling_energy_to_storage = min(cooling_power_avail, charge_transfer*cooling_storage.capacity/efficiency)
                else:
                    maybe_cooling_energy_to_storage = max(-sim_results['cooling_demand'][time_step], charge_transfer*cooling_storage.capacity*efficiency)

                if maybe_cooling_energy_to_storage >= 0:
                    maybe_next_charge_on_es = charge_on_es + maybe_cooling_energy_to_storage*efficiency/cooling_storage.capacity
                else:
                    maybe_next_charge_on_es = charge_on_es + (maybe_cooling_energy_to_storage/efficiency)/cooling_storage.capacity

                # Note that we are getting the closest lower charge level from next_charge value, this will result in some losses.
                next_charge_level = get_level(0., 1., maybe_next_charge_on_es, charge_levels)
                next_charge = get_val(0., 1., next_charge_level, charge_levels)

                cooling_energy_to_storage = (next_charge - charge_on_es)*cooling_storage.capacity*efficiency

                cooling_energy_drawn_from_heat_pump = cooling_energy_to_storage + sim_results['cooling_demand'][time_step]
                elec_demand_cooling = cooling_pump.get_electric_consumption_cooling(cooling_supply = cooling_energy_drawn_from_heat_pump)

                # J is used at places to denote energy instead of charge value.
                logger.debug("Cooling demand {0:.2f}; \
                    Maybe power avail {1:.2f}; \
                    To ES {2:.2f} J -> {3:.2f} J, {4:.3f} -> {5:.3f} -> {6:.3f}; \
                    From pump {7:.2f}; \
                    Elec^2 {8:.2f}; \
                    COP {9:.2f}".format(
                        sim_results['cooling_demand'][time_step],
                        cooling_power_avail,
                        maybe_cooling_energy_to_storage, cooling_energy_to_storage,
                        charge_on_es, maybe_next_charge_on_es, next_charge,
                        cooling_energy_drawn_from_heat_pump,
                        elec_demand_cooling*elec_demand_cooling,
                        cooling_pump.cop_cooling))

                clipped_action_val[time_step-start_time][charge_level][action] = next_charge - charge_on_es
                #logger.debug("Minimum elec energy in step {0}, charge {1} is {2}".format(time_step+1, next_charge_level, min(cost[time_step+1][next_charge_level])))
                cost_array[time_step-start_time][charge_level][action] = elec_demand_cooling*elec_demand_cooling + min(cost_array[time_step+1-start_time][next_charge_level])
                # logger.debug("\tMin sum of E^2 on this route {0:.2f}".format(cost_array[time_step-start_time][charge_level][action]))
                if break_after_this_action:
                    break

    logger.debug("\n\nOptimal sequence ----> ")
    charge_crwl = 0
    total_charged_val = 0

    for time_step in range(start_time, end_time+1):
        curr_charge = get_val(0., 1., charge_crwl, charge_levels)
        curr_charge_after_loss = get_val(0., 1., charge_crwl, charge_levels) * (1-loss_coeff)
        optimal_action_level = np.argmin(cost_array[time_step-start_time][charge_crwl])
        optimal_action_val[time_step-start_time] = \
            clipped_action_val[time_step-start_time][charge_crwl][optimal_action_level]
        if optimal_action_val[time_step-start_time] > 0:
            total_charged_val += optimal_action_val[time_step-start_time]

    next_charge = optimal_action_val[time_step-start_time] + curr_charge_after_loss
    next_charge_floor = get_val(0., 1., get_level(0., 1., next_charge, charge_levels), charge_levels)

    # logger.debug("Optimal action seq {0}".format(optimal_action_sequence[time_step-start_time]))
    logger.debug("{0:.2f}: {1:.2f} -> {2:.2f} -> +/- {3:.2f} -> {4:.2f} -> {5:.2f}; {6:.2f}".format(time_step,
                curr_charge, curr_charge_after_loss,
                optimal_action_val[time_step-start_time],
                next_charge, next_charge_floor,
                cost_array[time_step-start_time][charge_crwl][optimal_action_level]))
    charge_crwl = get_level(0., 1., next_charge, charge_levels)

    return optimal_action_val

def reset_all(entities):
    for entity in entities.values():
        entity.reset()

def get_cost_of_building(building_uids, **kwargs):
    '''
    Get the cost of a single building from start_time to end_time using DP and discrete action and charge levels.
    '''
    env, buildings, heat_pump, heat_tank, cooling_tank = create_env(building_uids, **kwargs)
    agents = get_agents(buildings, **kwargs)

    # Add different agents below.
    # if kwargs["agent"] == "RBC":
    #     state = env.reset()
    #     done = False
    #     while not done:
    #         action = agents.select_action(state)
    #         next_state, rewards, done, _ = env.step(action)
    #         state = next_state
    #     cost_rbc = env.cost()
    #     logger.info("{0}, {1}".format(cost_rbc, env.get_total_charges_made()))
    if kwargs["agent"] in ["Q", "RBC"]:
        k = 0
        episodes = kwargs["episodes"] if kwargs["agent"] == "Q" else 1
        cost, cum_reward = np.zeros((episodes,)), np.zeros((episodes,))

        for e in range(episodes): #A stopping criterion can be added, which is based on whether the cost has reached some specific threshold or is no longer improving
            cum_reward[e] = 0
            state = env.reset()

            done = False
            while not done:
                if (k)%500==0:
                    print('hour: '+str(k)+' of '+str(2500*episodes))
                action = agents.select_action(state)
                next_state, rewards, done, _ = env.step(action)
                reward = reward_function(rewards) #See comments in reward_function.py
                agents.add_to_batch(state, action, reward, next_state, done)
                state = next_state
                cum_reward[e] += reward[0]
                k+=1
            cost[e] = env.cost()
        print(cost)
        print(cum_reward)
        print(env.action_track[8][-100:])
        print(env.buildings[0].cooling_storage.energy_balance_list[2400:])

        plt.plot(env.action_track[8][-100:])
        plt.plot(env.buildings[0].cooling_device.cop_cooling_list[2400:2500])
        plt.legend(['RL Action','Heat Pump COP'])
        plt.show()

        plt.plot(env.buildings[0].cooling_storage.soc_list[2400:])
        plt.plot(env.buildings[0].cooling_storage.energy_balance_list[2400:])
        plt.legend(['State of Charge','Storage device energy balance'])
        plt.show()
    elif kwargs["agent"] == "DPDiscr":
        learning_start_time = time.time()
        optimal_action_val = run_dp(heat_pump[building_uids[-1]],
        cooling_tank[building_uids[-1]], buildings[-1], **kwargs)
        learning_end_time = time.time()

        env = CityLearn(demand_file, weather_file, buildings = [buildings[-1]], time_resolution = 1,
        simulation_period = (kwargs["start_time"]-1, kwargs["end_time"]))
        done = False
        time_step = 0
        while not done:
            _, rewards, done, _ = env.step([[optimal_action_val[time_step]]])
        time_step += 1
        cost_via_dp = env.cost()
        logger.info("{0}, {1}, {2}".format(cost_via_dp, env.get_total_charges_made(),
        learning_end_time - learning_start_time))

args = parse_arguments()

logger.info("Cost, Total charging done, Learning time")
get_cost_of_building(args.building_uids, start_time=args.start_time, end_time=args.end_time,
    action_levels=args.action_levels, min_action_val=args.min_action_val, max_action_val=args.max_action_val,
    charge_levels=args.action_levels, min_charge_val=args.min_action_val, max_charge_val=args.max_action_val,
    agent=args.agent, episodes=args.episodes)
