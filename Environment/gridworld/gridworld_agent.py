#from ... import agent

from agent import *

#from common.constants import GD_Actions

class GDAgent(Agent):

    # entity is the physical representation of the agent
    def __init__(self, entity,brain,env):

        super().__init__(entity,brain,env)
        #self.entity = entity

        #self.brain = brain

        # staring pos
        self.startPos = (-1,-1)

        self.fixedPos = (-1,-1)

        self.lastPos =(-1,-1)

        print("GDAgent:: init")

    # Method to redefine
    def reset(self,randomPos = False):

        #self.startPos = self.environment.generateRandomPosition()
        if randomPos == False:
            self.startPos = self.fixedPos

        self.entity.setPosition(self.startPos)


