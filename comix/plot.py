import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

methods = {
    "covdn-naf": ("dr", 2),
    "covdn": ("dr", 5),
    "comix-naf": ("dr", 3),
    "facmaddpg": ("sig", 2),
    "iql-cem": ("sig", 5),
    "iql-naf": ("sig", 6),
    "maddpg": ("sig", 4),
}


sns.set(style="whitegrid")
template = "results_{}/sacred/{}/info.json"
plt.figure()
xmin = np.inf
xmax = 0
for m, v in methods.items():
    fname = template.format(v[0], v[1])
    with open (fname, "r") as f:
        data = json.load(f)
    mu = data["cost_mean"]
    x = data["cost_mean_T"]
    xmin = min(xmin, min(x))
    xmax = max(xmax, max(x))
    label = m
    plt.plot(x, mu, label=label)
plt.hlines(y=0.84, xmin=xmin, xmax=xmax, linestyle='-', label='marlisa')
plt.legend(fontsize=10)
plt.xlabel("Timesteps", fontsize=15)
plt.ylabel("Cost mean", fontsize=15)
plt.savefig("plot.png")
plt.close()
