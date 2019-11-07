from citylearn import  CityLearn, building_loader, auto_size
from energy_models import HeatPump, EnergyStorage, Building
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def run_dp(cooling_pump, cooling_storage, building, start_time, end_time):
  def get_level(min, max, val, steps):
    slab_size = (max - min)/(steps-1)
    return int(val/slab_size)

  def get_val(min, max, step, steps):
    slab_size = (max - min)/(steps-1)
    # print("Charge val for {0} to {1} for {2} is {3}".format(min, max, step, slab_size*step + min))
    return slab_size*step + min


  num_actions = 3
  action_min = -1.
  action_max = 1.

  charge_levels = 3
  charge_min = 0.
  charge_max = 1.

  time_steps = 3

  sim_results = building.sim_results

  cost_array = np.full((end_time - start_time + 2, charge_levels, num_actions), np.inf)
  cost = lambda t, c, a: cost_array[t-start_time][c][a]

  print("ES capacity {0}".format(cooling_storage.capacity))
  print("Cooling demand {0}".format(sim_results['cooling_demand'][start_time:end_time+1]))
  elec_no_es = []
  cooling_demand = []
  for t in range(start_time, end_time+1):
    cooling_demand.append(sim_results['cooling_demand'][t])
    e = cooling_pump.get_electric_consumption_cooling(sim_results['cooling_demand'][t])
    elec_no_es.append(e*e)
  print("Demand {0}".format(cooling_demand))
  print("Electricity no ES {0}".format(elec_no_es))
  cost_array[end_time+1-start_time] = np.zeros((charge_levels, num_actions))

  optimal_action_sequence = np.zeros((end_time - start_time + 2))
  for time_step in range(end_time, start_time-1, -1):
    for charge_level in range(charge_levels-1, -1, -1):
      for action in range(num_actions-1, -1, -1):
        charge_on_es = get_val(0., 1., charge_level, charge_levels)
        charging_val = get_val(-1, 1, action, num_actions)
        print("Time {0} charge {1:.2f} action {2:.2f}".format(time_step, charge_on_es, charging_val))
        if -1 * min(charging_val, 0) > charge_on_es:
          break

        if max(charging_val, 0) > 1 - charge_on_es:
          continue
        
        break_after_this_action = False

        if charging_val < 0 and -1 * charging_val * cooling_storage.capacity > sim_results['cooling_demand'][time_step]:
          break_after_this_action = True

        cooling_power_avail = cooling_pump.get_max_cooling_power(t_source_cooling = sim_results['t_out'][time_step]) - sim_results['cooling_demand'][time_step]
        cooling_energy_to_storage = max(-sim_results['cooling_demand'][time_step], min(cooling_power_avail, charging_val*cooling_storage.capacity))
        cooling_energy_drawn_from_heat_pump = cooling_energy_to_storage + sim_results['cooling_demand'][time_step]

        elec_demand_cooling = cooling_pump.get_electric_consumption_cooling(cooling_supply = cooling_energy_drawn_from_heat_pump)
        #print("Charge taken from ES {0}".format(cooling_energy_to_storage/cooling_storage.capacity))
        next_charge = charge_on_es + cooling_energy_to_storage/cooling_storage.capacity
        #print("Next charge {0}".format(next_charge))
        next_charge_level = get_level(0., 1., next_charge, charge_levels)
        next_charge_floor = get_val(0., 1., next_charge_level, charge_levels)
        # print("Next charge level {0} {1}".format(next_charge, next_charge_level))
        print("Cooling demand {0:.2f}; Maybe power avail {1:.2f}; To ES {2:.2f}, {3:.2f} -> {4:.2f} -> {5:.2f}; From pump {6:.2f}; Elec {7:.2f}".format(sim_results['cooling_demand'][time_step],
          cooling_power_avail, cooling_energy_to_storage, charge_on_es, next_charge, next_charge_floor, cooling_energy_drawn_from_heat_pump, elec_demand_cooling))
        #print("Minimum elec energy in step {0}, charge {1} is {2}".format(time_step+1, next_charge_level, min(cost[time_step+1][next_charge_level])))
        cost_array[time_step-start_time][charge_level][action] = elec_demand_cooling*elec_demand_cooling + min(cost_array[time_step+1-start_time][next_charge_level])
        print("\tMin on this route {0:.2f}".format(cost_array[time_step-start_time][charge_level][action]))
        if break_after_this_action:
            break

  print("\n\nOptimal sequence ----> ")
  charge_crwl = 0
  for time_step in range(start_time, end_time+1):
    optimal_action_sequence[time_step-start_time] = np.argmin(cost_array[time_step-start_time][charge_crwl])
    next_charge = get_val(-1, 1, optimal_action_sequence[time_step-start_time], num_actions) + get_val(0., 1., charge_crwl, charge_levels)
    next_charge_floor = get_val(0., 1., get_level(0., 1., next_charge, charge_levels), charge_levels)
    print("Optimal action seq {0}".format(optimal_action_sequence[time_step-start_time]))
    print("{0:.2f}: {1:.2f} -> {2:.2f} -> {3:.2f}; {4:.2f}".format(time_step, get_val(0., 1., charge_crwl, charge_levels), next_charge, next_charge_floor,
      cost_array[time_step-start_time][charge_crwl][int(optimal_action_sequence[time_step-start_time])]))
    charge_crwl = get_level(0., 1., next_charge, charge_levels)

  return min(cost_array[0][0])

def get_cost_of_building(building_idx):
  '''
  Example of the implementation of a Rule-Based Controller
  '''
  building_ids = [building_idx] #[i for i in range(8,77)]

  #Building the RL environment with heating and cooling loads and weather files
  '''
  CityLearn
      Weather file
      Buildings
          File with heating and cooling demands
          CoolingDevices (HeatPump)
          CoolingStorages (EnergyStorage)
  '''

  data_folder = Path("data/")

  demand_file = data_folder / "AustinResidential_TH.csv"
  weather_file = data_folder / 'Austin_Airp_TX-hour.csv'

  heat_pump, heat_tank, cooling_tank = {}, {}, {}

  #Ref: Assessment of energy efficiency in electric storage water heaters (2008 Energy and Buildings)
  loss_factor = 0.19/24
  buildings = []
  for uid in building_ids:
      heat_pump[uid] = HeatPump(nominal_power = 9e12, eta_tech = 0.22, t_target_heating = 45, t_target_cooling = 10)
      heat_tank[uid] = EnergyStorage(capacity = 9e12, loss_coeff = loss_factor)
      cooling_tank[uid] = EnergyStorage(capacity = 9e12, loss_coeff = loss_factor)
      buildings.append(Building(uid, heating_storage = heat_tank[uid], cooling_storage = cooling_tank[uid], heating_device = heat_pump[uid], cooling_device = heat_pump[uid]))
      buildings[-1].state_space(np.array([24.0, 40.0, 1.001]), np.array([1.0, 17.0, -0.001]))
      buildings[-1].action_space(np.array([0.5]), np.array([-0.5]))
      
  building_loader(demand_file, weather_file, buildings)  
  auto_size(buildings, t_target_heating = 45, t_target_cooling = 10)

  env = CityLearn(demand_file, weather_file, buildings = buildings, time_resolution = 1, simulation_period = (3500,6000))

  return run_dp(heat_pump[buildings[-1].buildingId], cooling_tank[buildings[-1].buildingId], buildings[-1], start_time=3500, end_time=3501)
  #NO STORAGE
  # env.reset()
  # done = False
  # while not done:
  #     _, rewards, done, _ = env.step([[0*i] for i in range(len(building_ids))])
  # cost_no_es = env.cost()

  # return env.total_electric_consumption[:]

for building_idx in range(8, 9):#sys.argv[1:]:
  elect_consump = get_cost_of_building(int(building_idx))
  print("Electricity consumption {0}".format(elect_consump))
  #Total electricity consumption profile of all the buildings for the last 100 hours of the simulation
  # print("Plotted {0}...".format(building_idx))
  # plt.plot(elect_consump)

# plt.show()

