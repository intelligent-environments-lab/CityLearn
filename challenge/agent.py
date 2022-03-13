from typing import List

class Agent:
    def __init__(self):
        '''Initialize the class and define any hyperparameters of the controller.'''
        raise NotImplementedError

    def select_actions(self, states: List[float]) -> List[float]:
        '''Action selection algorithm'''
        actions = []
        return actions

    def add_to_buffer(self, states: List[float], actions: List[float], reward: float, next_states: List[float], done: bool, coordination_vars: List[float] = None, coordination_vars_next: List[List[float]] =None):
        '''Make any updates to your policy, you don't have to use all the variables above (you can leave the coordination
        variables empty if you wish, or use them to share information among your different agents). You can add a counter
        within this function to compute the time-step of the simulation, since it will be called once per time-step.'''
        raise NotImplementedError