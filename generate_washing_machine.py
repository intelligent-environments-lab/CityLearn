import pandas as pd
import random
import json

# Load real data
slow_df = pd.read_csv("/Users/rui.pina/Documents/Thesis/rebase/SMART-PDM-Dataset/2-washing_machines/2022-07-07_10.25.00_2022-07-07_11.33.00/slow.csv")
voltage_values = slow_df['V'].values / 100
current_values = slow_df['A'].values
num_real_points = len(voltage_values)

# Parameters
target_rows = 8760
rows = []
timestep = 0
current_day_type = 7
current_hour = 24

# Track active window state
active_window = None

# Track if each day has already had a window
day_has_window = {day: False for day in range(1, 8)}

while len(rows) < target_rows:
    current_day = current_day_type
    current_hr = current_hour

    if active_window and timestep <= active_window['wm_end_time_step']:
        rows.append({
            'day_type': current_day,
            'hour': current_hr,
            'wm_start_time_step': active_window['wm_start_time_step'],
            'wm_end_time_step': active_window['wm_end_time_step'],
            'load_profile': json.dumps(active_window["load_profile"])
        })

    elif active_window and timestep > active_window['wm_end_time_step']:
        active_window = None

    if active_window is None:
        if random.random() < 0.7 and not day_has_window[current_day]:
            window_length = random.randint(2, 5)
            wm_start_time_step = timestep
            wm_end_time_step = wm_start_time_step + window_length - 1

            if wm_end_time_step >= target_rows:
                break

            num_load_points = random.randint(1, window_length - 1)
            sample_indices = random.sample(range(num_real_points), num_load_points)
            load_profile = [
                round((voltage_values[i] * current_values[i]) / 1000, 2)
                for i in sample_indices
            ]

            active_window = {
                'wm_start_time_step': wm_start_time_step,
                'wm_end_time_step': wm_end_time_step,
                'load_profile': load_profile
            }

            day_has_window[current_day] = True

            rows.append({
                'day_type': current_day,
                'hour': current_hr,
                'wm_start_time_step': wm_start_time_step,
                'wm_end_time_step': wm_end_time_step,
                'load_profile': json.dumps(load_profile)
            })

        elif active_window is None:
            rows.append({
                'day_type': current_day,
                'hour': current_hr,
                'wm_start_time_step': -1,
                'wm_end_time_step': -1,
                'load_profile': -1
            })

    timestep += 1
    current_hour += 1
    if current_hour > 24:
        current_hour = 1
        current_day_type += 1
        if current_day_type > 7:
            current_day_type = 1
        day_has_window[current_day_type] = False

# Save to CSV
df = pd.DataFrame(rows)
df.to_csv("synthetic_load_profile.csv", index=False)
print("Saved as 'synthetic_load_profile.csv'")
