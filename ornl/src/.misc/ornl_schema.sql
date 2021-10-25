CREATE TABLE IF NOT EXISTS county (
    id INTEGER PRIMARY KEY,
    "name" TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS building_type (
    id INTEGER PRIMARY,
    "name" TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS building (
    id INTEGER PRIMARY KEY,
    source_id TEXT NOT NULL,
    county_id INTEGER REFERENCES county(id) NOT NULL,
    building_type_id INTEGER REFERNCES building_type(id) NOT NULL,
    UNIQUE (source_id)
);

CREATE TABLE IF NOT EXISTS "zone" (
    id INTEGER PRIMARY KEY,
    building_id INTEGER REFERENCES building(id) NOT NULL,
    "name" TEXT NOT NULL,
    multiplier INTEGER NOT NULL,
    UNIQUE (building_id, "name")
);

CREATE TABLE IF NOT EXISTS material (
    id INTEGER PRIMARY KEY,
    building_id INTEGER REFERENCES building(id) NOT NULL,
    idf_object TEXT NOT NULL,
    "name" TEXT NOT NULL,
    thickness REAL,
    conductivity REAL,
    density REAL,
    specific_heat REAL,
    thermal_resistance REAL,
    thermal_absorptance REAL,
    solar_absorptance REAL,
    visible_absorptance REAL,
    roughness REAL,
    u_factor REAL,
    solar_heat_gain_coefficient REAL,
    visible_transmittance REAL,
    solar_transmittance_at_normal_incidence REAL,
    front_side_solar_reflectance_at_normal_incidence REAL
    back_side_solar_reflectance_at_normal_incidence REAL
    visible_transmittance_at_normal_incidence REAL
    front_side_visible_reflectance_at_normal_incidence REAL
    back_side_visible_reflectance_at_normal_incidence REAL
    infrared_transmittance_at_normal_incidence REAL
    front_side_infrared_hemispherical_emissivity REAL
    back_side_infrared_hemispherical_emissivity REAL
    dirt_correction_factor_for_solar_and_visible REAL
    solar_diffusing BOOLEAN,
    UNIQUE (building_id, idf_object, "name")
);

CREATE TABLE IF NOT EXISTS construction (
    id INTEGER PRIMARY KEY,
    building_id INTEGER REFERENCES building(id) NOT NULL,
    idf_object TEXT NOT NULL,
    "name" TEXT NOT NULL,
    f_factor REAL,
    area REAL,
    exposed_perimeter REAL,
    UNIQUE (building_id, idf_object, "name")
);

CREATE TABLE IF NOT EXISTS construction_layer (
    id INTEGER PRIMARY KEY,
    construction_id INTEGER REFERENCES construction(id) NOT NULL,
    material_id INTEGER REFERENCES material(id) NOT NULL,
    position INTEGER NOT NULL,
    UNIQUE (construction_id, material_id, position)
);

CREATE TABLE IF NOT EXISTS surface (
    id INTEGER PRIMARY KEY,
    construction_id INTEGER REFERENCES construction(id) NOT NULL,
    zone_id INTEGER REFERENCES "zone"(id) NOT NULL,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    outside_boundary_condition TEXT NOT NULL,
    sun_exposure TEXT NOT NULL,
    wind_exposure TEXT NOT NULL,
    CONSTRAINT CHECK("type" IN ('floor', 'wall', 'roof')),
    UNIQUE (construction_id, zone_id, "name", "type")
);

CREATE TABLE IF NOT EXISTS sub_surface (
    id INTEGER PRIMARY KEY,
    surface_id INTEGER REFERENCES surface(id) NOT NULL,
    "name" TEXT NOT NULL,
    "type" TEXT NOT NULL,
    CONSTRAINT CHECK("type" IN ('window')),
    UNIQUE (surface_id, "name", "type")
);

CREATE TABLE IF NOT EXISTS internal_mass (
    id INTEGER PRIMARY KEY,
    building_id INTEGER REFERENCES building(id) NOT NULL,
    zone_id INTEGER REFERENCES "zone"(id)
    zone_list_id INTEGER REFERENCES zone_list(id),
    "name" TEXT NOT NULL,
    surface_area REAL NOT NULL,
    UNIQUE (building_id, zone_id, zone_list_id, "name")
)