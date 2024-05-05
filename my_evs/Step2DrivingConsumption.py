from emobpy import Consumption, HeatInsulation, BEVspecs, DataBase

# Initialize seed
from emobpy.tools import set_seed
set_seed()

if __name__ == '__main__':

    DB = DataBase('db')                                  # Instance of profiles' database whose input is the pickle files' folder
    DB.loadfiles_batch(kind="driving")                   # loading mobility pickle files to the database
    mname = list(DB.db.keys())[0]                        # getting the id of the first mobility profile

    HI = HeatInsulation(True)                            # Creating the heat insulation configuration by copying the default configuration
    BEVS = BEVspecs()                                    # Database that contains BEV models
    VW_ID3 = BEVS.model(('Volkswagen','ID.3',2020))      # Model instance that contains vehicle parameters
    c = Consumption(mname, VW_ID3)
    c.load_setting_mobility(DB)
    c.run(
        heat_insulation=HI,
        weather_country='DE',
        weather_year=2016,
        passenger_mass=75,                   # kg
        passenger_sensible_heat=70,          # W
        passenger_nr=1.5,                    # Passengers per vehicle including driver
        air_cabin_heat_transfer_coef=20,     # W/(m2K). Interior walls
        air_flow = 0.02,                     # m3/s. Ventilation
        driving_cycle_type='WLTC',           # Two options "WLTC" or "EPA"
        road_type=0,                         # For rolling resistance, Zero represents a new road.
        road_slope=0
        )
    c.save_profile('db')


