from citylearn.citylearn import District

class Reward:
    @staticmethod
    def get(agent_index: int, district: District) -> float:
        reward = district.buildings[agent_index].net_electricity_consumption[-1]
        return reward