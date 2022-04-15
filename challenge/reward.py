from citylearn.citylearn import District
from citylearn.reward import Reward

class ChallengeReward(Reward):
    def __init__(self):
        super().__init__()
        
    def get(self, index: int, district: District) -> float:
        # ************* BEGIN EDIT *************
        # Write agent reward equation. The placeholder equation is the district's latest net_electricity_consumption.
        # Alternatively, in a multi-agent setup where the number of agents equals the number of buildings,
        # the latest net_electricity_consumption of building i in the district
        # can be accessed via district.buildings[index].net_electricity_consumption[-1].
        reward = district.net_electricity_consumption[-1]
        # ***************** END ****************

        return reward