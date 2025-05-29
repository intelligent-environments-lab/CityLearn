import pandas as pd
import os
import json
import random
from pathlib import Path

# Set your data directory
data_dir = Path("./data/datasets/citylearn_challenge_2022_phase_all_plus_evs")

# Load your schema file
with open(data_dir / "schema.json") as f:
    schema = json.load(f)["electric_vehicles_def"]

# Process each Electric Vehicle CSV
for file in data_dir.glob("Electric_Vehicle_*.csv"):
    ev_name = file.stem.replace(".csv", "")
    vehicle_id = ev_name
    battery_attrs = schema[vehicle_id]["battery"]["attributes"]
    capacity = battery_attrs["capacity"]

    df = pd.read_csv(file)

    # Extract charger name from non-null values in 'charger' column
    first_charger = df["charger"].dropna().unique()
    charger_name = first_charger[0] if len(first_charger) > 0 else f"unknown_charger_{vehicle_id}"

    # Compute current SOC values
    current_soc = []
    for _, row in df.iterrows():
        state = row["electric_vehicle_charger_state"]
        soc_arrival = row.get("electric_vehicle_soc_arrival")

        if pd.isna(state) or state == 3:
            current_soc.append("")  # leave blank for state 3 rows
        elif not pd.isna(soc_arrival):
            current_soc.append(round(soc_arrival * capacity / 100.0, 2))
        else:
            rand_soc = round(random.uniform(0.2, 0.8) * capacity, 2)
            current_soc.append(rand_soc)

    # Create output DataFrame
    df_new = pd.DataFrame({
        "electric_vehicle_charger_state": df["electric_vehicle_charger_state"],
        "electric_vehicle_id": vehicle_id,
        "electric_vehicle_battery_capacity_khw": capacity,
        "current_soc": current_soc,
        "electric_vehicle_departure_time": df["electric_vehicle_departure_time"],
        "electric_vehicle_required_soc_departure": df["electric_vehicle_required_soc_departure"],
        "electric_vehicle_estimated_arrival_time": df["electric_vehicle_estimated_arrival_time"],
        "electric_vehicle_estimated_soc_arrival": df["electric_vehicle_estimated_soc_arrival"]
    })

    # Convert necessary columns to object type to allow blank values
    df_new = df_new.astype({
        "electric_vehicle_id": "object",
        "electric_vehicle_battery_capacity_khw": "object",
        "current_soc": "object",
        "electric_vehicle_departure_time": "object",
        "electric_vehicle_required_soc_departure": "object",
        "electric_vehicle_estimated_arrival_time": "object",
        "electric_vehicle_estimated_soc_arrival": "object"
    })

    # Blank all columns except the charger state when state == 3
    mask = df_new["electric_vehicle_charger_state"] == 3
    cols_to_blank = [
        "electric_vehicle_id",
        "electric_vehicle_battery_capacity_khw",
        "current_soc",
        "electric_vehicle_departure_time",
        "electric_vehicle_required_soc_departure",
        "electric_vehicle_estimated_arrival_time",
        "electric_vehicle_estimated_soc_arrival"
    ]
    df_new.loc[mask, cols_to_blank] = ""

    # Save new file named after charger
    output_path = data_dir / f"{charger_name}.csv"
    df_new.to_csv(output_path, index=False)

    print(f"Converted {file.name} -> {output_path.name}")