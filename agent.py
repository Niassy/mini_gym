"""

Author : Mamadou Jean Baptiste Niassy

Permission is give to modify this code source as soon as you keep this header or

references the author of this code (Mamadou Jean Baptiste Niassy)

"""
from abc import ABC,abstractmethod

import pygame
import random


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