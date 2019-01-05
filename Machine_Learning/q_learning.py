
import  numpy as np

import random

from brain import *

import pickle
from Machine_Learning.hyper_parameters import *

class QLearning(Brain):

    def __init__(self,agent):

        super().__init__(agent)

        self.Q = np.zeros([self.agent.getEnvironment().getStateDimension() , self.agent.getEnvironment().getActionDimension()])

        # Set learning parameters
        self.lr = .8
        self.y = .95
        self.epsilon = 0.9

        self.epsilon_greedy = True

        # from arthur juliani repository
        #https: // github.com / awjuliani / DeepRL - Agents
        self.startE = 0.3  # Starting chance of random action
        self.endE = 0.1  # Final chance of random action

        self.epsilon = self.startE

        self.stepDrop = (self.startE - self.endE) / ANNELING_STEPS


    # Bellman Equation for updating
    def updateParameters(self,state,action,reward,next_state,done):

        print("QLearning:: update parameters:: s =",state," s1 ",next_state,"a ",action," r ",reward)
        if done == False:
            self.Q[state,action] = self.Q[state,action] + self.lr *( reward + self.y * np.max( self.Q[next_state,:]) - self.Q[state,action] )
        else:
            self.Q[state, action] = self.Q[state, action] + self.lr *reward



    def getOptimalAction(self,state):


        #print(" QLearning:: getOptimalAction:: state ",state," pos ",self.agent.entity.getPosition())
        action = np.argmax( self.Q[state,:] )

        random_choosed = False
        if self.epsilon_greedy:

            if random.random() < self.epsilon  and self.agent.environment.inference == False:
                #print("QLearning:: GetOptimalAction:: random action")
                #return self.agent.environment.getRandomAction()
                action = self.agent.environment.getRandomAction()

                random_choosed = True
            #self.epsilon-= 1 /self.agent.environment.num_episodes

        if self.agent.environment.tryPerformAction(agent = self.agent, action = action) == -1 :
            #print("QLearning:: GetOptimalAction:: No possiblle action")

            # best_value = -10000
            # best_action  = -1
            # while best_action==-1:
            #
            #     for a in range( self.Q.shape[1]):
            #         if self.Q[state, a] > best_value:
            #
            #             best_action = a
            #             best_value = self.Q[state, a]
            #
            # if best_action !=-1:
            #     return best_action

            return self.agent.environment.getRandomAction()

        #if random_choosed:
            #print("QLearning:: GetOptimalAction::  random choosed ")
        #else:
            #print("QLearning:: GetOptimalAction::  max action choosed ",action)

        # See Arthur juliani repositoty in Q Exploration
        if self.epsilon > self.endE :
            self.epsilon -= self.stepDrop

        #return np.argmax( self.Q[state,:] )
        return action

    def save_model(self,filename):

        model = { "Q_table":self.Q }

        outfile = open(filename, 'wb')
        #pickle.dump(self.Q_safety, outfile)
        pickle.dump(model, outfile)
        outfile.close()

        print("QLearning:: save_model :: Q table saved")
        #print("Q safety saved ",self.Q_safety)

    def load_model(self,filename):

        infile = open(filename, 'rb')
        obj_file = pickle.load(infile)

        self.Q = obj_file["Q_table"]

        ##print("Loaded Q  safety ",self.Q_safety)
        #print("Loaded Q moving ", self.Q_moving)

        infile.close()


        print("QLearning:: load_model :: Q table loaded")
        return obj_file



    def trainModel(self):

        pass

    def evalModel(self):

        pass