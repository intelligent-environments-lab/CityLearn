### Feel free to edit this file at will, but make sure it runs properly when we execute the main.py or main.ipynb file that is provided. You can't change the main file, only to the submission files.

'''Import any packages here'''
import json
import torch

class Agent:
    def __init__(self, building_ids, buildings_states_actions, building_info):     
        with open(buildings_states_actions) as json_file:
            self.buildings_states_actions = json.load(json_file)
            
        '''Initialize the class and define any hyperparameters of the controller'''
        
            
    def select_action(self, states):
        
        '''Action selection algorithm'''
            
        return actions
                
        
    def add_to_buffer(self, states, actions, rewards, next_states, done, coordination_vars=None, coordination_vars_next=None):
        
        '''Make any updates to your policy, you don't have to use all the variables above (you can leave the coordination
        variables empty if you wish, or use them to share information among your different agents). You can add a counter
        within this function to compute the time-step of the simulation, since it will be called once per time-step'''
        
        