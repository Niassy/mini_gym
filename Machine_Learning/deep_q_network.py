

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

import numpy as np

import random

from brain import *

from common.pytorch_utils import *

import pickle
from Machine_Learning.hyper_parameters import *

from common.replay_buffer import *

class DQN_Model(nn.Module):

    def __init__(self, num_inputs, num_outputs, hidden_size=256):
        super(DQN_Model, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)

    # please provide numpy state
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = self.linear2(x)
        #print("value ",x)
        return x


class DeepQNetwork(Brain):

    def __init__(self,agent,num_inputs, num_outputs, hidden_size=256):

        super().__init__(agent)

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_outputs)

        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        self.model =  DQN_Model(num_inputs,num_outputs)
        self.target_model = DQN_Model(num_inputs,num_outputs)

        self.optimizer = optim.Adam(self.model.parameters())

        self.state_type = State_Type.feature_encoding

        self.update = True

        ################### Exploration strategy ######################
        self.startE = 0.3  # Starting chance of random action
        self.endE = 0.1  # Final chance of random action

        self.epsilon = self.startE

        self.stepDrop = (self.startE - self.endE) / ANNELING_STEPS

        ################### Exploration strategy ######################

    def updateParameters(self, state, action, reward, next_state, done):

        # Store episode Buffer
        self.replay_buffer.push(state, action, reward, next_state, done)

        # Compute the loss
        if self.update:
            compute_td_error(self.model,self.target_model,self.optimizer,BATCH_SIZE,self.replay_buffer,DEVICE)


    # Inspired by the Q Learning
    def getOptimalAction(self, state):

        #state = np.array([state])
        #print("state ",state.shape)
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        #print("state ",state.shape)
        q_value = self.model.forward(state)

        action = q_value.max(1)[1].item()
        random_choosed = False
        if self.epsilon_greedy and self.agent.environment.inference == False:

            if random.random() < self.epsilon and self.agent.environment.inference == False:
                # print("QLearning:: GetOptimalAction:: random action")
                # return self.agent.environment.getRandomAction()
                action = self.agent.environment.getRandomAction()
                random_choosed = True

        if self.epsilon > self.endE:
            self.epsilon -= self.stepDrop


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

            action = self.agent.environment.getRandomAction()


        #print("DEEPQN::getOptimalAction ",action)
        return action


    def save_model(self, filename):

        #self.agent.environment.prev_num_episodes = self.agent.environment.num_episodes

        total_episodes = self.agent.environment.num_episodes + self.agent.environment.prev_num_episodes

        torch.save({'epoch':total_episodes,
                   'model':self.model.state_dict(),
                   'target_model':self.target_model.state_dict(),
                    'optimizer':self.optimizer.state_dict(),
                    'loss':0
                    }

            ,filename
        )


        print("DQN:: load_model model is saved to ",filename," total epoch ",total_episodes)



    def load_model(self, filename):

        checkpoint = torch.load(filename)

        self.model.load_state_dict(checkpoint['model'])
        self.target_model.load_state_dict(checkpoint['target_model'] )
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        #self.agent.environment.prev_num_episodes = epoch

        self.agent.environment.prev_num_episodes = epoch

        print("DQN:: load_model model is loaded from",filename)
        print("Total Epoch = ",epoch)


    def trainModel(self):

        print("DQN:: trainModel")

        self.model.train()
        self.target_model.train()


    def evalModel(self):

        print("DQN:: evalModel")
        self.model.eval()




