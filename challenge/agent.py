import random
from citylearn.citylearn import District
from citylearn.controller.rlc import RLC
from typing import List, Mapping, Tuple

class Agent(RLC):
    def __init__(self, index: int, actions_dimension: int, district: District, **kwargs):
        '''Initialize the class and define any hyperparameters of the controller.'''
        super().__init__(**kwargs)
        self.__index = index
        self.__actions_dimension = actions_dimension
        self.__district = district

        # ************* BEGIN EDIT *************
        # Include any other initialization steps.
        pass
        # ***************** END ****************

    @property
    def index(self) -> int:
        return self.__index

    @property
    def actions_dimension(self) -> int:
        return self.__actions_dimension

    @property
    def district(self) -> District:
        return self.__district

    def select_actions(self, states: List[float]) -> Tuple[List[float], Mapping]:
        '''Action selection algorithm'''
       
        # ************* BEGIN EDIT *************
        # Write action selection algorithm. The placeholder algorithm below selects random values betwenn -1 and 1.
        # Include in kwargs any keyword arguments that will be used in add_to_buffer function.
        actions = [random.uniform(-1,1) for _ in range(self.actions_dimension)]
        kwargs = {}
        # ***************** END ****************
        
        self.actions = list(actions)
        self.next_time_step()
        return self.actions[-2], kwargs

    def add_to_buffer(self, states: List[float], actions: List[float], reward: float, next_states: List[float], done: bool, **kwargs):
        self.reward = reward

        # ************* BEGIN EDIT *************
        # Make policy updates.
        pass
        # ***************** END ****************

    # **************** BEGIN SUPPORTING FUNCTIONS ****************
    # Here, write functions to be called in __init__, select_actions and add_to_buffer functions.
    # *************************** END ****************************