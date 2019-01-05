
from abc import ABC,abstractmethod

from enum import Enum

class State_Type(Enum):

    single_integer = 0.  # state are mapped to integr
    feature_encoding = 1  # state mapped to feature reprensentation using tensors or matrix
    raw_pixel =2  # using directly image screen as input

class Brain(ABC):

    def __init__(self,agent):

        self.agent = agent

        self.lr = .8
        self.y = .95
        self.epsilon = 0.9

        self.epsilon_greedy = True

        # from arthur juliani repository
        #https: // github.com / awjuliani / DeepRL - Agents
        self.startE = 0.3  # Starting chance of random action
        self.endE = 0.1  # Final chance of random action

        self.epsilon = self.startE

        # If true update parameter (or compute loss)
        self.update = False

        try:

            self.agent.setBrain(self)

        except:
            print("Brain:: Init agent is probaly None")

        finally:

            self.state_type = State_Type.single_integer


    @abstractmethod
    def updateParameters(self,state,action,reward,next_state,done):

        pass




    @abstractmethod
    def getOptimalAction(self,state):

        pass


    @abstractmethod
    def save_model(self,filename):

        pass

    @abstractmethod
    def load_model(self,filename):

        pass

    @abstractmethod
    def trainModel(self):

        pass

    @abstractmethod
    def evalModel(self):

        pass