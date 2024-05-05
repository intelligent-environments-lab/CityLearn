from emobpy import Charging, DataBase

# Initialize seed
from emobpy.tools import set_seed
set_seed()

if __name__ == "__main__":

    DB = DataBase('db')                                  # Instance of profiles' database whose input is the pickle files' folder
    DB.loadfiles_batch(kind="availability")               # loading availability pickle files to the database
    aname = list(DB.db.keys())[0]                        # getting the id of the first availability profile

    strategies = ["immediate", "balanced", "from_0_to_24_at_home", "from_23_to_8_at_home"]

    for option in strategies:
        c = Charging(aname)
        c.load_scenario(DB)
        c.set_sub_scenario(option)
        c.run()
        c.save_profile('db')
