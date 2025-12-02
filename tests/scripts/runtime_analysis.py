import run_ev_rbc_export_end as obj
from pandas import DataFrame

timesteps = [438, 876, 1314, 1752, 2190,
            2628, 3066, 3504, 3942, 4380,
            4818, 5256, 5694, 6132, 6570,
            7008, 7446, 7884, 8322, 8760]


data = DataFrame(columns=
                ["Timesteps", "Environment Creation", "Agent Creation", "Environment Reset",
                "Render", "Buildings Observations Retrieval", "Total"
                ])


for i in timesteps:
    env_creation, agent_creation, env_reset, total_render_time, total_retrieval_time, total_time = obj.main(i)
    new_values = {
        "Timesteps": i,
        "Environment Creation": env_creation,
        "Agent Creation": agent_creation,
        "Environment Reset" : env_reset,
        "Render": total_render_time,
        "Buildings Observations Retrieval" : total_retrieval_time,
        "Total" : total_time
    }
    data.loc[len(data)] = new_values
    data.to_csv(f"{i}-runtime.csv", index=False)

data.to_csv("runtime.csv", index=False)