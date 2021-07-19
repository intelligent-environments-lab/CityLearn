import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json

methods = ["comix", "comix-naf", "covdn", "covdn-naf", "iql-cem", "facmaddpg", "marlisa"]
color_maps = {}
for i, m in enumerate(methods):
    color_maps[m] = f"C{i}"

sacreds = [1, 2, 3]

seasons = ["spring", "summer", "fall", "winter"]

season_intervals = {
    "spring": (24*(31+28), 24*(31+28+31+30+31)),
    "summer": (24*(31+28+31+30+31), 24*(31+28+31+30+31+30+31+31)),
    "fall":   (24*(31+28+31+30+31+30+31+31), 24*(31+28+31+30+31+30+31+31+30+31+30)),
    "winter": (24*(-31), 24*(31+28)),
}

results = {
    "net_electric_consumption_no_pv_no_storage": {},
    "net_electric_consumption_no_storage": {},
    "net_electric_consumption": {},
    "cost": {},
}

keys = None
for method in methods:
    res = np.load(f"./saved_envs/{method}", allow_pickle=True)
    ne1 = res["ne1"]
    ne2 = res["ne2"]
    ne3 = res["ne3"]
    try:
        cost = res["cost"].item()
    except:
        cost = res["cost"][1]
        tmp = {}
        for key in cost:
            tmp[key[:-8]] = cost[key]
        cost = tmp
    results["net_electric_consumption_no_pv_no_storage"][method] = ne1
    results["net_electric_consumption_no_storage"][method] = ne2
    results["net_electric_consumption"][method] = ne3
    results["cost"][method] = cost
    keys = cost.keys()

fs = 20
lw = 5

### plot the season electric consumption

for season in seasons:
    fig, ax = plt.subplots(1,1,figsize=(5, 5))
    year = 4
    rg = season_intervals[season]
    interval = range((year-1)*8760 + rg[0], (year-1)*8760 + rg[1])

    x = np.arange(24)
    y1 = results["net_electric_consumption_no_pv_no_storage"]["comix"][interval].reshape(-1,24).mean(0)
    y2 = results["net_electric_consumption_no_storage"]["comix"][interval].reshape(-1,24).mean(0)

    ax.plot(x, y2, label="no_storage", color="tab:gray", linewidth=lw)

    for method in methods:
        y = results["net_electric_consumption"][method][interval].reshape(-1, 24).mean(0)
        ax.plot(x, y, label=method, color=color_maps[method], linewidth=lw)
 
    #ax.legend(fontsize=fs)
    ax.set_xlabel("hour", fontsize=fs)
    ax.set_ylim(0, 300)
    ax.grid()
    ax.set_title(season.title(), fontsize=(fs+5))
    ax.set_ylabel("net electric consumption", fontsize=fs)
    plt.tight_layout()
    plt.savefig(f"imgs/{season}.png", dpi=400)
    plt.close()

# Create a color palette
plt.figure()
colors = [color_maps[x] for x in methods]
palette = dict(zip(["no storage"]+methods, ["tab:gray"] + colors))
# Create legend handles manually
handles = [matplotlib.patches.Patch(color=palette[x], label=x) for x in palette.keys()]
# Create legend
plt.legend(handles=handles)
# Get current axes object and turn off axis
plt.gca().set_axis_off()
plt.tight_layout()
plt.savefig("imgs/season_legend.png", dpi=400)
plt.close()


### plot the cost
n = len(keys)
Y = np.arange(n) * (len(methods)*0.1 * 2)

fig, ax = plt.subplots()

for ki, key in enumerate(keys):
    for i,method in enumerate(methods):
        plt.barh(Y[ki]+i*0.1, results["cost"][method][key], color=color_maps[method], height=0.1)


ax.set_yticks(Y+len(methods)/2*0.1)
ax.set_yticklabels(keys, fontsize=8)
ax.axvline(x=1, ymin=0, ymax=1000, linestyle='--', color='gray')
plt.tight_layout()
plt.savefig("imgs/cost.png", dpi=400)
plt.close()

"""
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
"""
