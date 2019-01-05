"""

Author : Mamadou Jean Baptiste Niassy

Permission is give to modify this code source as soon as you keep this header or

references the author of this code (Mamadou Jean Baptiste Niassy)

"""
from abc import ABC,abstractmethod

import pygame
import random

from Machine_Learning.deep_q_network import *
from Machine_Learning.q_learning import *

from common.constants import *

# This class is abstract
class Agent(ABC):

    # entity is the physical representation of the agent
    def __init__(self, entity,brain,env):

        self.entity = entity

        self.brain = brain

        self.environment = env

        print("Agent::Init ")
        #if self.entity == None:
            #print("emtiy is none")

        self.id = 0


    def updateModelParameters(self,state,action,reward,next_state,done):

        self.brain.updateParameters(state,action,reward,next_state,done)

    # Method to redefine
    @abstractmethod
    def reset(self):

        pass

    def render(self,screen):

        self.entity.render(screen)

    def setBrain(self,brain):

        self.brain = brain

    def getEntity(self):

        return self.entity

    def getEnvironment(self):

        return self.environment

    ###################### Brain interface ############################

    # Choose your learning algorithm
    # @abstractmethod
    def useBrain(self, brainType):
        brain = None

        if brainType == Brain_Type.Deep_Q_Network:

            brain = DeepQNetwork(agent=self, num_inputs=self.environment.state_dimension, num_outputs=self.environment.action_dimension,
                                 hidden_size=256)


        else:
            brain = QLearning(agent=self)


        self.brain = brain

