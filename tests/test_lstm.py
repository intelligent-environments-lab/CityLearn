from citylearn.agents.base import BaselineAgent as Agent
from citylearn.citylearn import CityLearnEnv
import matplotlib.pyplot as plt

# initialize
env = CityLearnEnv('baeda_3dem', central_agent=True)
model = Agent(env)

# step through environment and apply agent actions
observations, _ = env.reset()

while not env.terminated:
    actions = model.predict(observations)
    observations, reward, info, terminated, truncated = env.step(actions)

# test
kpis = model.env.evaluate()
kpis = kpis.pivot(index='cost_function', columns='name', values='value').round(3)
kpis = kpis.dropna(how='all')

for b in env.buildings:
    y1 = b.indoor_dry_bulb_temperature
    y2 = b.energy_simulation.indoor_dry_bulb_temperature_without_control
    fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    ax.plot(y1, label='Predicted')
    ax.plot(y2, label='Actual')
    ax.set_xlim(None, 800)
    ax.set_xlabel('Time step')
    ax.set_ylabel('C')
    ax.set_title(b.name)
    ax.legend()
    plt.show()