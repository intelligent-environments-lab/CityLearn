import random
from gym import spaces
from citylearn.controller.base import Controller
from typing import List, Mapping, Tuple

class Agent(Controller):
    def __init__(self, action_spaces: spaces.Box):
        '''Initialize the class and define any hyperparameters of the controller.'''
        super().__init__(action_spaces=action_spaces)

        # ************* BEGIN EDIT *************
        # Include any other initialization steps. Use the self.district property to access district and building
        # parameters needed to set up the agent. self.district is set to None after initialization to save memory.
        pass
        # ***************** END ****************

    def select_actions(self, states: List[float]) -> Tuple[List[float], Mapping]:
        '''Action selection algorithm'''
       
        # ************* BEGIN EDIT *************
        # Write action selection algorithm. The placeholder algorithm below selects random values betwenn -1 and 1.
        # Include in kwargs any keyword arguments that will be used in add_to_buffer function.
        actions = [random.uniform(-1,1) for _ in range(self.action_dimension)]
        kwargs = {}
        # ***************** END ****************
        
        self.actions = list(actions)
        self.next_time_step()
        return self.actions[-2], kwargs

    def add_to_buffer(self, states: List[float], actions: List[float], reward: float, next_states: List[float], done: bool):
        self.reward = reward

        # ************* BEGIN EDIT *************
        # Make policy updates.
        pass
        # ***************** END ****************

    # **************** BEGIN SUPPORTING FUNCTIONS ****************
    # Here, write functions to be called in __init__, select_actions and add_to_buffer functions.
    # *************************** END ****************************