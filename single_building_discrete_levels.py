from citylearn import  CityLearn, building_loader, auto_size
from energy_models import HeatPump, EnergyStorage, Building
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

loss_coeff = 0.57 # 0.19/24
efficiency = 1.0

# TODO: Take into account losses in storage and while transferring charge.
# TODO: Extend to multiple buildings (assuming we have the same cooling demand pattern for the buildings for a time period).

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
    # print("Charge val for {0} to {1} for {2} is {3}".format(min, max, step, slab_size*step + min))
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

  # TODO (Readability): Create numpy array that can be indexed using time_step instead of time_step - start_time
  # cost = lambda t, c, a: cost_array[t-start_time][c][a]

  sim_results['t_out'][end_time] = 60.0
  print("ES capacity {0}\n".format(cooling_storage.capacity))
  print("Cooling demand\n{0}\n".format(sim_results['cooling_demand'][start_time:end_time+1]))
  print("Outside temps\n{0}\n".format(sim_results['t_out'][start_time:end_time+1]))

  elec_no_es = []
  cooling_demand = []

  for t in range(start_time, end_time+1):
    #cooling_demand.append(sim_results['cooling_demand'][t])
    # cooling_pump.time_step = t
    cooling_pump.set_cop(t, sim_results['t_out'][t])
    e = cooling_pump.get_electric_consumption_cooling(sim_results['cooling_demand'][t])
    print("Elec demand {0} - {1}".format(t, e))
    elec_no_es.append(e*e)

  #print("Demand {0}".format(cooling_demand))
  print("Sum of Electricity^2 without ES {0}\n".format(elec_no_es))

  # Store the optimal action sequence
  optimal_action_sequence = np.zeros((end_time - start_time + 2))

  for time_step in range(end_time, start_time-1, -1):
    for charge_level in range(charge_levels-1, -1, -1):
      # Minor optimization for start time
      if time_step == start_time and charge_level != 0:
        continue

      for action in range(action_levels-1, -1, -1):
        charge_on_es = get_val(0., 1., charge_level, charge_levels)
        charge_on_es = charge_on_es*(1-loss_coeff)
        charge_transfer = get_val(-1, 1, action, action_levels)

        print("Time {0} charge {1:.2f} action {2:.2f}".format(time_step, charge_on_es, charge_transfer))

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
        if charge_transfer < 0 and -1 * charge_transfer * cooling_storage.capacity * efficiency > sim_results['cooling_demand'][time_step]:
          break_after_this_action = True

        # Adapted from set_storage_cooling()
        cooling_power_avail = cooling_pump.get_max_cooling_power(t_source_cooling = sim_results['t_out'][time_step]) - sim_results['cooling_demand'][time_step]
        if charge_transfer >= 0:
          cooling_energy_to_storage = min(cooling_power_avail, charge_transfer*cooling_storage.capacity/efficiency)
        else:
          cooling_energy_to_storage = max(-sim_results['cooling_demand'][time_step], charge_transfer*cooling_storage.capacity*efficiency)

        cooling_energy_drawn_from_heat_pump = cooling_energy_to_storage + sim_results['cooling_demand'][time_step]

        elec_demand_cooling = cooling_pump.get_electric_consumption_cooling(cooling_supply = cooling_energy_drawn_from_heat_pump)
        if charge_level == 0:
          print("Elec demand {0} - {1}".format(time_step, elec_demand_cooling))

        if cooling_energy_to_storage >= 0:
          next_charge_on_es = charge_on_es + cooling_energy_to_storage*efficiency/cooling_storage.capacity
        else:
          next_charge_on_es = charge_on_es + (cooling_energy_to_storage/efficiency)/cooling_storage.capacity

        # Note that we are getting the closest lower charge level from next_charge value, this will result in some losses.
        next_charge_level = get_level(0., 1., next_charge_on_es, charge_levels)
        next_charge_floor = get_val(0., 1., next_charge_level, charge_levels)

        # J is used at places to denote energy instead of charge value.
        print("Cooling demand {0:.2f}; Maybe power avail {1:.2f}; To ES {2:.2f} J, {3:.2f} -> {4:.2f} -> {5:.2f}; From pump {6:.2f}; Elec^2 {7:.2f}; COP {8:.2f}".format(sim_results['cooling_demand'][time_step],
          cooling_power_avail, cooling_energy_to_storage, charge_on_es, next_charge_on_es, next_charge_floor, cooling_energy_drawn_from_heat_pump, elec_demand_cooling*elec_demand_cooling,
          cooling_pump.cop_cooling))

        #print("Minimum elec energy in step {0}, charge {1} is {2}".format(time_step+1, next_charge_level, min(cost[time_step+1][next_charge_level])))
        cost_array[time_step-start_time][charge_level][action] = elec_demand_cooling*elec_demand_cooling + min(cost_array[time_step+1-start_time][next_charge_level])
        print("\tMin sum of E^2 on this route {0:.2f}".format(cost_array[time_step-start_time][charge_level][action]))
        if break_after_this_action:
            break

  print("\n\nOptimal sequence ----> ")
  charge_crwl = 0

  for time_step in range(start_time, end_time+1):
    curr_charge = get_val(0., 1., charge_crwl, charge_levels)
    curr_charge_after_loss = get_val(0., 1., charge_crwl, charge_levels) * (1-loss_coeff)
    optimal_action_sequence[time_step-start_time] = np.argmin(cost_array[time_step-start_time][charge_crwl])
    next_charge = get_val(-1, 1, optimal_action_sequence[time_step-start_time], action_levels) + curr_charge_after_loss
    next_charge_floor = get_val(0., 1., get_level(0., 1., next_charge, charge_levels), charge_levels)

    print("Optimal action seq {0}".format(optimal_action_sequence[time_step-start_time]))
    print("{0:.2f}: {1:.2f} -> {2:.2f} -> {3:.2f} -> {4:.2f}; {5:.2f}".format(time_step, curr_charge, curr_charge_after_loss,
      next_charge, next_charge_floor,
      cost_array[time_step-start_time][charge_crwl][int(optimal_action_sequence[time_step-start_time])]))
    charge_crwl = get_level(0., 1., next_charge, charge_levels)

  return min(cost_array[0][0])

def get_cost_of_building(building_uid, **kwargs):
  '''
  Get the cost of a single building from start_time to end_time using DP and discrete action and charge levels.
  '''
  building_ids = [building_uid] #[i for i in range(8,77)]

  data_folder = Path("data/")

  demand_file = data_folder / "AustinResidential_TH.csv"
  weather_file = data_folder / 'Austin_Airp_TX-hour.csv'

  heat_pump, heat_tank, cooling_tank = {}, {}, {}

  #Ref: Assessment of energy efficiency in electric storage water heaters (2008 Energy and Buildings)
  buildings = []
  for uid in building_ids:
      heat_pump[uid] = HeatPump(nominal_power = 9e12, eta_tech = 0.22, t_target_heating = 45, t_target_cooling = 10)
      heat_tank[uid] = EnergyStorage(capacity = 9e12, loss_coeff = loss_coeff)
      cooling_tank[uid] = EnergyStorage(capacity = 9e12, loss_coeff = loss_coeff)
      buildings.append(Building(uid, heating_storage = heat_tank[uid], cooling_storage = cooling_tank[uid], heating_device = heat_pump[uid], cooling_device = heat_pump[uid]))
      buildings[-1].state_space(np.array([24.0, 40.0, 1.001]), np.array([1.0, 17.0, -0.001]))
      buildings[-1].action_space(np.array([0.5]), np.array([-0.5]))
      
  building_loader(demand_file, weather_file, buildings)  
  auto_size(buildings, t_target_heating = 45, t_target_cooling = 10)

  env = CityLearn(demand_file, weather_file, buildings = buildings, time_resolution = 1,
    simulation_period = (kwargs["start_time"], kwargs["end_time"]))

  return run_dp(heat_pump[buildings[-1].buildingId], cooling_tank[buildings[-1].buildingId], buildings[-1], **kwargs)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--action_levels',
  help='Select the number of action levels. Note: choose odd number if you want zero charging action',
  type=int, required=True)
parser.add_argument('--min_action_val', help='Select the min action value >= -1.', type=float, required=True)
parser.add_argument('--max_action_val', help='Select the max action value <= 1.', type=float, required=True)
parser.add_argument('--charge_levels',
  help='Select the number of charge levels. Note: choose odd number if you want zero charge value allowed',
  type=int, required=True)
parser.add_argument('--min_charge_val', help='Select the min charge value >= 0.', type=float, required=True)
parser.add_argument('--max_charge_val', help='Select the max charge value <= 1.', type=float, required=True)
parser.add_argument('--start_time',
  help='Start hour. Note: For less than 3500 hr, there seems to be no data for a building 8, check this', type=int,
  required=True)
parser.add_argument('--end_time', help='End hour', type=int, required=True)
parser.add_argument('--building_uid', help='Use 8 for now', type=int, required=True)
#parser.add_argument('--loss_coeff', help='The one given was 0.19/24', type=int, required=True)

args = parser.parse_args()

elect_consump = get_cost_of_building(args.building_uid, start_time=args.start_time, end_time=args.end_time,
  action_levels=args.action_levels, min_action_val=args.min_action_val, max_action_val=args.max_action_val,
  charge_levels=args.action_levels, min_charge_val=args.min_action_val, max_charge_val=args.max_action_val)

print("Electricity consumption {0}".format(elect_consump))
  # Total electricity consumption profile of all the buildings for the last 100 hours of the simulation
  # print("Plotted {0}...".format(building_idx))
  # plt.plot(elect_consump)

# plt.show()

