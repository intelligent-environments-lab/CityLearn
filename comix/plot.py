import numpy as np
import matplotlib.pyplot as plt
import json

methods = ["comix-naf","covdn","facmaddpg"]
sacreds = [1, 2, 3]

season_intervals = {
    "spring": (0, 24*30*3),
    "summer": (24*30*3, 24*30*6),
    "fall":   (24*30*6, 24*30*9),
    "winter": (24*30*9, 24*30*12),
}

results = {
    "net_electric_consumption_no_pv_no_storage": {},
    "net_electric_consumption_no_storage": {},
    "net_electric_consumption": {},
}

for method in methods:
    res = np.load(f"./saved_envs/{method}")
    ne1 = res["ne1"]
    ne2 = res["ne2"]
    ne3 = res["ne3"]
    results["net_electric_consumption_no_pv_no_storage"][method] = ne1
    results["net_electric_consumption_no_storage"][method] = ne2
    results["net_electric_consumption"][method] = ne3


seasons = ["spring", "summer", "fall", "winter"]
color_maps = {
    "comix-naf": "C0",
    "covdn": "C1",
    "facmaddpg": "C2",}

fs = 10
for season in seasons:
    fig, axs = plt.subplots(1,3, figsize=(20, 5))
    year = 4
    rg = season_intervals[season]
    interval = range((year-1)*8760 + rg[0], (year-1)*8760 + rg[1])
    x = np.arange(24)
    for method in methods:
        y = results["net_electric_consumption_no_pv_no_storage"][method][interval].reshape(-1, 24).mean(0)
        axs[0].plot(x, y, label=method, color=color_maps[method], linewidth=2)
    axs[0].legend(fontsize=fs)
    axs[0].set_ylabel("net electricity consumption", fontsize=fs)
    axs[0].set_xlabel("hour", fontsize=fs)
    axs[0].set_ylim(0, 450)
    axs[0].grid()
    axs[0].set_title("net_electric_consumption_no_pv_no_storage", fontsize=fs)
    for method in methods:
        y = results["net_electric_consumption_no_storage"][method][interval].reshape(-1, 24).mean(0)
        axs[1].plot(x, y, label=method, color=color_maps[method], linewidth=2)
    axs[1].legend(fontsize=fs)
    axs[1].set_xlabel("hour", fontsize=fs)
    axs[1].set_ylim(0, 450)
    axs[1].grid()
    axs[1].set_title(f"{season}\nnet_electric_consumption_no_storage", fontsize=fs)
    for method in methods:
        y = results["net_electric_consumption"][method][interval].reshape(-1, 24).mean(0)
        axs[2].plot(x, y, label=method, color=color_maps[method], linewidth=2)
    axs[2].legend(fontsize=fs)
    axs[2].set_xlabel("hour", fontsize=fs)
    axs[2].set_ylim(0, 450)
    axs[2].grid()
    axs[2].set_title("net_electric_consumption", fontsize=fs)
    plt.savefig(f"imgs/{season}.png", dpi=400)
    plt.close()

test_costs = {}
for method, sacred in zip(methods, sacreds):
    path= f"./results/sacred/{sacred}/info.json"                                        
    with open (path, "r") as f:                                                  
        data = json.load(f)                                                      
        mu = data["test_cost_mean"]                                            
        x = data["test_cost_mean_T"] 
    test_costs[method] = (x, mu)

plt.figure()
for method in methods:
    plt.plot(test_costs[method][0], test_costs[method][1], label=method, color=color_maps[method], linewidth=2.0)
plt.legend(fontsize=fs)
plt.xlabel("training step", fontsize=fs)
plt.ylabel("average test cost", fontsize=fs)
plt.savefig("imgs/test_costs.png", dpi=400)
