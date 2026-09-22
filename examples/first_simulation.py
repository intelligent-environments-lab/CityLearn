from citylearn.agents.baseline import BusinessAsUsualAgent
from citylearn.citylearn import CityLearnEnv

env = CityLearnEnv(
    "citylearn_challenge_2022_phase_all_plus_evs",
    episode_time_steps=24,
    interface="flat",
    render_mode="none",
)
agent = BusinessAsUsualAgent(env)
observations, info = env.reset(seed=0)
terminated = truncated = False

while not (terminated or truncated):
    actions = agent.predict(observations)
    observations, rewards, terminated, truncated, info = env.step(actions)

kpis = env.evaluate_v2()
selected = [
    "district_cost_total_control_eur",
    "district_energy_grid_total_import_control_kwh",
    "district_ev_performance_departure_min_acceptable_feasible_ratio",
    "district_electrical_service_phase_violations_energy_total_kwh",
]
print(kpis.loc[
    (kpis["level"] == "district") & kpis["cost_function"].isin(selected),
    ["cost_function", "value"],
].to_string(index=False))
env.close()
