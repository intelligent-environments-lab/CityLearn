PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS simulation (
    id INTEGER NOT NULL,
    "name" TEXT UNIQUE NOT NULL,
    start_timestamp TEXT NOT NULL,
    end_timestamp TEXT,
    successful INTEGER DEFAULT 0,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS environment (
    id INTEGER NOT NULL,
    simulation_id INTEGER NOT NULL,
    simulation_period_start INTEGER NOT NULL,
    simulation_period_end INTEGER NOT NULL,
    central_agent INTEGER NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY (simulation_id) REFERENCES simulation (id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS agent (
    id INTEGER NOT NULL,
    "name" TEXT NOT NULL,
    reward_style TEXT NOT NULL,
    hidden_dim TEXT,
    discount REAL,
    tau REAL,
    lr REAL,
    batch_size INTEGER,
    replay_buffer_capacity INTEGER,
    regression_buffer_capacity INTEGER,
    start_training INTEGER,
    exploration_period INTEGER,
    start_regression INTEGER,
    information_sharing INTEGER,
    pca_compression REAL,
    action_scaling_coef REAL,
    reward_scaling REAL,
    update_per_step INTEGER,
    iterations_as INTEGER,
    safe_exploration REAL,
    basic_rbc INTEGER,
    deterministic_period_start INTEGER NOT NULL,
    seed INTEGER,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS building (
    id INTEGER NOT NULL,
    "name" TEXT NOT NULL,
    environment_id INTEGER NOT NULL,
    agent_id INTEGER NOT NULL,
    "type" INTEGER,
    solar_power_installed REAL NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY (environment_id) REFERENCES environment (id) ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (agent_id) REFERENCES agent (id) ON UPDATE CASCADE ON DELETE CASCADE,
    UNIQUE (id, environment_id)
);

CREATE TABLE IF NOT EXISTS state_space (
    building_id INTEGER NOT NULL,
    "month" INTEGER NOT NULL,
    "day" INTEGER NOT NULL,
    hour INTEGER NOT NULL,
    daylight_savings_status INTEGER NOT NULL,
    t_out INTEGER NOT NULL,
    t_out_pred_6h INTEGER NOT NULL,
    t_out_pred_12h INTEGER NOT NULL,
    t_out_pred_24h INTEGER NOT NULL,
    rh_out INTEGER NOT NULL,
    rh_out_pred_6h INTEGER NOT NULL,
    rh_out_pred_12h INTEGER NOT NULL,
    rh_out_pred_24h INTEGER NOT NULL,
    diffuse_solar_rad INTEGER NOT NULL,
    diffuse_solar_rad_pred_6h INTEGER NOT NULL,
    diffuse_solar_rad_pred_12h INTEGER NOT NULL,
    diffuse_solar_rad_pred_24h INTEGER NOT NULL,
    direct_solar_rad INTEGER NOT NULL,
    direct_solar_rad_pred_6h INTEGER NOT NULL,
    direct_solar_rad_pred_12h INTEGER NOT NULL,
    direct_solar_rad_pred_24h INTEGER NOT NULL,
    t_in INTEGER NOT NULL,
    avg_unmet_setpoint INTEGER NOT NULL,
    rh_in INTEGER NOT NULL,
    non_shiftable_load INTEGER NOT NULL,
    solar_gen INTEGER NOT NULL,
    cooling_storage_soc INTEGER NOT NULL,
    dhw_storage_soc INTEGER NOT NULL,
    electrical_storage_soc INTEGER NOT NULL,
    net_electricity_consumption INTEGER NOT NULL,
    carbon_intensity INTEGER NOT NULL,
    PRIMARY KEY (building_id),
    FOREIGN KEY (building_id) REFERENCES building (id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS action_space (
    building_id INTEGER NOT NULL,
    cooling_storage INTEGER NOT NULL,
    dhw_storage INTEGER NOT NULL,
    electrical_storage INTEGER NOT NULL,
    PRIMARY KEY (building_id),
    FOREIGN KEY (building_id) REFERENCES building (id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS cooling_device (
    id INTEGER NOT NULL,
    building_id INTEGER NOT NULL,
    nominal_power REAL,
    eta_tech REAL,
    t_target_cooling REAL,
    t_target_heating REAL,
    PRIMARY KEY (id),
    FOREIGN KEY (building_id) REFERENCES building (id) ON UPDATE CASCADE ON DELETE CASCADE,
    UNIQUE (building_id)
);

CREATE TABLE IF NOT EXISTS dhw_heating_device (
    id INTEGER NOT NULL,
    building_id INTEGER NOT NULL,
    nominal_power REAL,
    efficiency REAL,
    PRIMARY KEY (id),
    FOREIGN KEY (building_id) REFERENCES building (id) ON UPDATE CASCADE ON DELETE CASCADE,
    UNIQUE (building_id)
);

CREATE TABLE IF NOT EXISTS cooling_storage (
    id INTEGER NOT NULL,
    building_id INTEGER NOT NULL,
    capacity REAL,
    max_power_output REAL,
    max_power_charging REAL,
    efficiency REAL,
    loss_coef REAL,
    PRIMARY KEY (id),
    FOREIGN KEY (building_id) REFERENCES building (id) ON UPDATE CASCADE ON DELETE CASCADE,
    UNIQUE (building_id)
);

CREATE TABLE IF NOT EXISTS dhw_storage (
    id INTEGER NOT NULL,
    building_id INTEGER NOT NULL,
    capacity REAL,
    max_power_output REAL,
    max_power_charging REAL,
    efficiency REAL,
    loss_coef REAL,
    PRIMARY KEY (id),
    FOREIGN KEY (building_id) REFERENCES building (id) ON UPDATE CASCADE ON DELETE CASCADE,
    UNIQUE (building_id)
);

CREATE TABLE IF NOT EXISTS electrical_storage (
    id INTEGER NOT NULL,
    building_id INTEGER NOT NULL,
    capacity REAL,
    nominal_power REAL,
    capacity_loss_coef REAL,
    power_efficiency_curve TEXT,
    capacity_power_curve TEXT,
    efficiency REAL,
    loss_coef REAL,
    PRIMARY KEY (id),
    FOREIGN KEY (building_id) REFERENCES building (id) ON UPDATE CASCADE ON DELETE CASCADE,
    UNIQUE (building_id)
);

CREATE TABLE IF NOT EXISTS timestep (
    id INTEGER NOT NULL,
    environment_id INTEGER NOT NULL,
    timestep INTEGER NOT NULL,
    episode INTEGER NOT NULL,
    "month" INTEGER NOT NULL,
    hour INTEGER NOT NULL,
    day_type INTEGER NOT NULL,
    daylight_savings_status INTEGER NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY (environment_id) REFERENCES environment (id) ON UPDATE CASCADE ON DELETE CASCADE,
    UNIQUE (environment_id, timestep, episode)
);

CREATE TABLE IF NOT EXISTS weather_timeseries (
    timestep_id INTEGER NOT NULL,
    outdoor_drybulb_temperature REAL NOT NULL,
    outdoor_relative_humidity REAL NOT NULL,
    diffuse_solar_radiation REAL NOT NULL,
    direct_solar_radiation REAL NOT NULL,
    PRIMARY KEY (timestep_id),
    FOREIGN KEY (timestep_id) REFERENCES timestep (id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS environment_timeseries (
    timestep_id INTEGER NOT NULL,
    environment_id INTEGER NOT NULL,
    carbon_emissions REAL,
    net_electric_consumption REAL,
    electric_consumption_electric_storage REAL,
    electric_consumption_dhw_storage REAL,
    electric_consumption_cooling_storage REAL,
    electric_consumption_dhw REAL,
    electric_consumption_cooling REAL,
    electric_consumption_appliances REAL,
    electric_generation REAL,
    PRIMARY KEY (timestep_id, environment_id),
    FOREIGN KEY (timestep_id) REFERENCES timestep (id) ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (environment_id) REFERENCES environment (id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS building_timeseries (
    timestep_id INTEGER NOT NULL,
    building_id INTEGER NOT NULL,
    indoor_temperature REAL,
    average_unmet_cooling_setpoint_difference REAL,
    indoor_relative_humidity REAL,
    cooling_demand_building REAL,
    dhw_demand_building REAL,
    electric_consumption_appliances REAL,
    electric_generation REAL,
    electric_consumption_cooling REAL,
    electric_consumption_cooling_storage REAL,
    electric_consumption_dhw REAL,
    electric_consumption_dhw_storage REAL,
    cooling_device_to_building REAL,
    cooling_device_to_storage REAL,
    cooling_storage_to_building REAL,
    dhw_heating_device_to_building REAL,
    dhw_heating_device_to_storage REAL,
    dhw_storage_to_building REAL,
    electrical_storage_electric_consumption REAL,
    PRIMARY KEY (timestep_id, building_id),
    FOREIGN KEY (timestep_id) REFERENCES timestep (id) ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (building_id) REFERENCES building (id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS cooling_device_timeseries (
    timestep_id INTEGER NOT NULL,
    cooling_device_id INTEGER NOT NULL,
    cop_cooling REAL,
    cop_heating REAL,
    electrical_consumption_cooling REAL,
    electrical_consumption_heating REAL,
    cooling_supply REAL,
    heat_supply REAL,
    PRIMARY KEY (timestep_id, cooling_device_id),
    FOREIGN KEY (timestep_id) REFERENCES timestep (id) ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (cooling_device_id) REFERENCES cooling_device (id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS dhw_heating_device_timeseries (
    timestep_id INTEGER NOT NULL,
    dhw_heating_device_id INTEGER NOT NULL,
    electrical_consumption_heating REAL,
    heat_supply REAL,
    PRIMARY KEY (timestep_id, dhw_heating_device_id),
    FOREIGN KEY (timestep_id) REFERENCES timestep (id) ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (dhw_heating_device_id) REFERENCES dhw_heating_device (id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS cooling_storage_timeseries (
    timestep_id INTEGER NOT NULL,
    cooling_storage_id INTEGER NOT NULL,
    soc REAL,
    energy_balance REAL,
    PRIMARY KEY (timestep_id, cooling_storage_id),
    FOREIGN KEY (timestep_id) REFERENCES timestep (id) ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (cooling_storage_id) REFERENCES cooling_storage (id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS dhw_storage_timeseries (
    timestep_id INTEGER NOT NULL,
    dhw_storage_id INTEGER NOT NULL,
    soc REAL,
    energy_balance REAL,
    PRIMARY KEY (timestep_id, dhw_storage_id),
    FOREIGN KEY (timestep_id) REFERENCES timestep (id) ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (dhw_storage_id) REFERENCES dhw_storage (id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS electrical_storage_timeseries (
    timestep_id INTEGER NOT NULL,
    electrical_storage_id INTEGER NOT NULL,
    soc REAL,
    energy_balance REAL,
    PRIMARY KEY (timestep_id, electrical_storage_id),
    FOREIGN KEY (timestep_id) REFERENCES timestep (id) ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (electrical_storage_id) REFERENCES electrical_storage (id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS action_timeseries (
    timestep_id INTEGER NOT NULL,
    building_id INTEGER NOT NULL,
    cooling_storage REAL,
    dhw_storage REAL,
    electrical_storage REAL,
    PRIMARY KEY (timestep_id, building_id),
    FOREIGN KEY (timestep_id) REFERENCES timestep (id) ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (building_id) REFERENCES building (id) ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE reward_timeseries (
    timestep_id INTEGER NOT NULL,
    building_id INTEGER NOT NULL,
    "value" REAL,
    PRIMARY KEY (timestep_id, building_id),
    FOREIGN KEY (timestep_id) REFERENCES timestep (id) ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (building_id) REFERENCES building (id) ON UPDATE CASCADE ON DELETE CASCADE
);