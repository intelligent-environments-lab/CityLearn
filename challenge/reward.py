from citylearn.citylearn import District

class Reward:
    @staticmethod
    def get(agent_index: int, district: District) -> float:
        # ************* BEGIN EDIT *************
        # Write agent reward equation. The placeholder equation is the district's latest net_electricity_consumption.
        # Alternatively, in a multi-agent setup where the number of agents equals the number of buildings,
        # the latest net_electricity_consumption of building i in the district
        # can be accessed via district.buildings[agent_index].net_electricity_consumption[-1].
        reward = district.net_electricity_consumption[-1]
        # ***************** END ****************

        return reward