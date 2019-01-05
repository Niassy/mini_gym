
from abc import ABC,abstractmethod

import pygame
import random

from common.constants import *
class Environment(ABC):

    def __init__(self,screen):

        self.screen = screen

        self.agents = []

        self.map = []

        self.main_agent =None

        self.num_episodes = 0

        self.state = None

        self.actions = []

        self.state_dimension = 0
        self.action_dimension = 0

        self.left_corner = 0
        self.right_corner = 0
        self.bottom_corner = 0
        self.top_corner = 0

        # If true prediction are done directly via model
        self.inference = False

        self.number_features = 1

        self.load_model = False

        self.save_model = False

        self.num_episodes =0

        self.prev_num_episodes = 0

    @abstractmethod
    def getReward(self):

        pass

    @abstractmethod
    def getPossibleActions(self):
        return

    @abstractmethod
    def getCurrentState(self):

        pass


    @abstractmethod
    def featurise_observation(self):

        pass

    @abstractmethod
    def getRandomAction(self):

        pass

    @abstractmethod
    def initialize(self):

        pass

    @abstractmethod
    def reset(self,num_episode = 0):

        pass

    @abstractmethod
    def step(self):

        pass

    @abstractmethod
    def render(self):

        pass

    @abstractmethod
    def tryPerformAction(self, agent, action, choosedAction=False):
        pass

    def getActionDimension(self):

        return self.action_dimension

    def getStateDimension(self):

        return self.state_dimension

    #@abstractmethod
    #def generateRandomPos(self):

        #pass

    def addAgent(self, agent):
        self.agents.append(agent)


    def setMainAgent(self, brainType):
        self.main_agent.useBrain(brainType)


    # def setStateDimension(self,state_dim):
    #
    #     self.state_dimension =state_dim
    #
    # def setActionDimenson(self,action_dim):
    #
    #     self.action_dimension = action_dim


