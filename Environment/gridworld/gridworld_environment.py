
import pygame
from common.constants import *

#from ... import environment
from environment import *

from .gridworld_agent import *

#import Environment.gridworld.gridworld_agent

from .gridworld_agent import *

from GameObject.moving_entity import *

from common.constants import *

from brain import *

from Machine_Learning.q_learning import *

from Machine_Learning.deep_q_network import *

import sys

import math
class GD_Environment(Environment):

    def __init__(self,screen):

        super(GD_Environment,self).__init__(screen)

        self.tile_width = GD_TILE_WIDTH
        self.tile_height = GD_TILE_HEIGHT

        self.left_corner = 0
        self.right_corner = GD_X_LIMIT
        self.bottom_corner = GD_Y_LIMIT
        self.top_corner = 0

        self.agents_starting_pos = {}

        self.obstacles = []

        self.goal = None

        print("Gridworld environment created successfully ")

        self.map = np.zeros( (GD_MAP_HEIGHT,GD_MAP_WIDTH) )

        self.invalid_pos_map = {}

        self.actions  = [ 0,1,2,3]  # left,right,up,down

        self.action_dimension = len(self.actions)

        self.number_features = 4

        self.state_dimension = GD_MAP_WIDTH * GD_MAP_HEIGHT * self.number_features

        # main agent last distance
        self.last_dist = 0

        self.load_model = LOAD_MODEL

        self.save_model =SAVE_MODEL

        self.initialize()

    def initialize(self):

        print("GridWorlEnv:: initialize")

        agent = GDAgent( entity=MovingEntiy(0,0,GD_TILE_WIDTH,GD_TILE_HEIGHT),brain=None,env=self)
        self.addAgent(agent)

        # Set this agent as main agent
        self.main_agent = agent

        #get his brain
        #main_agent_brain = QLearning(self.main_agent)

        main_agent_brain = DeepQNetwork(agent=agent,num_inputs = self.state_dimension, num_outputs =self.action_dimension, hidden_size=256)


        # Load a pre existing model
        if self.load_model:
            main_agent_brain.load_model(MODEL_PATH)

        print("Agent brain ",self.main_agent.brain)

        # obstacle
        self.obstacles.append( Square(  0,160,GD_TILE_WIDTH,GD_TILE_HEIGHT,color = (255,0,0)))

        #self.obstacles.append( Square(  320,128,GD_TILE_WIDTH,GD_TILE_HEIGHT,color = (255,0,0)))


        self.goal = Square(  320,160,GD_TILE_WIDTH,GD_TILE_HEIGHT,color = (0,0,255) )

        self.invalid_pos_map.update({(0, 5): (0, 160)})

        self.invalid_pos_map.update({(0, 0): (0, 0)})

        self.invalid_pos_map.update({(10, 5): (320, 160)})

        # Adding multiple agent
        #for i in range(0,9):
            #agent = GDAgent(entity=MovingEntiy(0, i * GD_TILE_WIDTH, GD_TILE_WIDTH, GD_TILE_HEIGHT), brain=None, env=self)
            #self.addAgent(agent)

        self.generateRandomPosition()
        self.reset()

        # new 03/01/2018

        self.feedMap()


    def updateMap(self):

        for agent in self.agents:

            x,y = agent.entity.getPosition()

            #print("GDENV::update map  x,y = ",  (x,y)  )
            x, y = x // GD_TILE_WIDTH, y // GD_TILE_HEIGHT

            #print("GDENV::updateMap :: x ",x," y ",y)
            self.map[y, x] = 1


    def feedMap(self):

        for agent in self.agents:

            x,y = agent.entity.getPosition()

            x, y = x // GD_TILE_WIDTH, y // GD_TILE_HEIGHT

            self.map[y, x] = 1

        x, y = self.goal.getPosition()

        x, y = x // GD_TILE_WIDTH, y // GD_TILE_HEIGHT

        self.map[y, x] = 2

        for obs in self.obstacles:

            x,y = obs.getPosition()

            x,y  = x //GD_TILE_WIDTH,y // GD_TILE_HEIGHT

            self.map[y,x] = 3

    def getReward(self):

        #if done:

        # compute distance

        #x1,y1 = self.main_agent.entity.getPosition()
        #x2,y2 = self.goal.getPosition()

        #dist =  math.sqrt(  (x1 - x2) ** 2 + (y1 + y2) ** 2 )

        _,_,dist = self.utils_distance(self.main_agent.entity, self.goal)

        reward = -1
        if dist < self.last_dist:
            reward+=5

        else:
            reward-=2

        return reward

    # Get the current stae for the gridworld environment
    def getCurrentState(self):

        state_type = None
        if self.main_agent.brain.state_type == State_Type.single_integer:
            # get the position of agent in the map

            #print("GDENV:: GETCURRENT STATE :: MAP WIDTH ",GD_MAP_WIDTH)

            x,y = self.main_agent.getEntity().getPosition()

            #print("x = ",x," y = ",y)

            xMap,yMap = x // GD_TILE_WIDTH,y // GD_TILE_HEIGHT

            #print("xmap  ",xMap," ymao ",yMap)
            state_type = yMap * GD_MAP_WIDTH  + xMap

        elif self.main_agent.brain.state_type == State_Type.feature_encoding:

            state_type = self.featurise_observation()

        return state_type


    def featurise_observation(self):

        # We will consider the map of the environment
        # 0 is free
        # 1 is agent
        # 2 is goal
        # 3 is obstacle

        H, W = self.map.shape  # 11 *  11

        x,y = self.main_agent.entity.getPosition()

        xMAP,yMap = x // GD_TILE_WIDTH, y // GD_MAP_HEIGHT

        # feature_vector = np.array( (H * W * 4 ,1))
        feature_vector = []
        for h in range(H):

            for w in range(W):

        ################ free,obstructed ( feature 1)  #################

                f1 = 0
                if self.map[h, w] == 0:  # free
                    f1 = 0


                else :  # obstructed
                    f1 = 1

        ################# Feature 2 : Position contains the player (1, 0) ###################

                f2 = 0
                if h == yMap and w == xMAP:
                    f2 = 1

                else:
                    f2 = 0

        ##################### Feature 3 : Position contains goql (1, 0) ###################

                f3 = 0
                if self.map[h, w] == 2:  # goal
                    f3 = 1

                else:
                    f3 = 0

        ##################### Feature 3 : Position contains obstacle  (1, 0) ###################

                f4 = 0
                if self.map[h, w] == 3:  # goal
                    f4 = 1

                else:
                    f4 = 0


                vec_fea = np.array([f1, f2, f3,f4])
                feature_vector.append(vec_fea)


        feature_vector = np.array(feature_vector).flatten()  # .reshape(1,-1)   # convert matrix to vector

        return feature_vector

    # Get a random action among all possible actions
    def getRandomAction(self):

        #print("##################GDENV:: GetRandomAction ######################")
        possibles_actions = self.getPossibleActions()

        possibles_actions = np.array(possibles_actions)

        #print("possibles actions ",possibles_actions)

        #print("!!!!!!!!!!!!!!!!!!!!!!GDENV:: GetRandomAction!!!!!!!!!!!!!!!!!!!!!!!")
        return np.random.choice(possibles_actions)

    def getPossibleActions(self):

        possible_actions = []
        for action in self.actions:

            #print("GDENV:: getPossibleAction:: action ",action)
            # Try to perform the action for the agent
            if self.tryPerformAction( self.main_agent,action) != -1:

                possible_actions.append(action)

        return possible_actions

    # Check if the action is valid within the environment
    def tryPerformAction(self,agent, action,choosedAction=False):

        x, y = agent.entity.getPosition()
        dir = (0,0)
        #print(" LEFT VALUE GD_Actions.left.value ", GD_Actions.right.value)
        if action == GD_Actions.left.value[0]:
            dir = (-1,0)

        elif action == GD_Actions.right.value[0]:
            dir = (1, 0)

        elif action == GD_Actions.up.value[0]:
            dir = (0,-1)

        else:
            dir = (0,1)

        move = dir[0] * GD_TILE_WIDTH,dir[1] * GD_TILE_HEIGHT

        newX,newY =  x + move[0],y + move[1]

        #print("GDENV:: tryPerformAction xNEW = ",newX," yNew ",newY," actiom ",action," x ",x," y" ,y," dir ",dir," move ",move)
        if newX > self.right_corner or newY > self.bottom_corner\
                or newX <0 or newY <0:

            #print("GDENV:: tryPerformAction x = ",action," is not possible")

            return -1

        # If True the ganet will choose this action
        #if choosedAction:
            #agent.entity.setPosition((newX,newY))

        return action

    def performAction(self,agent,action):

        x,y = agent.entity.getPosition()

        dir = self.utils_getDirection(action,self.main_agent)

        move = dir[0] * GD_TILE_WIDTH, dir[1] * GD_TILE_HEIGHT

        newX,newY =  x + move[0],y + move[1]

        #print("GDENV::PerformAction :: (x,y )",(x,y)," (newX newY) ",(newX,newY)," dir = ",dir," a ",action)
        agent.entity.setPosition((newX, newY))

    # override
    def reset(self,num_episode = 0):

        self.num_episodes = num_episode
        for agent in self.agents:

            #agent.reset()
            agent.entity.setPosition( self.agents_starting_pos[agent.id] )

    # override
    def step(self):

        action = -1

        self.state = self.getCurrentState()
        action = self.main_agent.brain.getOptimalAction(self.state)


        # try:
        #
        #     action = self.main_agent.brain.getOptimalAction(self.state)
        #
        # except NameError:
        #     #print("GD_ENV Step:: Exception ::Main agent is not defined or ...")
        #     pass
        #
        # except:
        #     print ("GDENV::STEP:: Unexpected error: May be you pass wrong arguments in the function" ,   sys.exc_info()[0])
        #     #print("GD_ENV::Step:: Exception  :: main agent or brain is null")
        #     pass
        #
        # finally:
        #
        #     return action

        # perform the action
        #self.main_agent.
        self.performAction(self.main_agent,action)

        reward = self.getReward()

        done = self.main_agent.entity.collide(self.goal)

        ## Check for obstacle collision
        if done == False:
            for obstacle in self.obstacles:

                if self.main_agent.entity.collide(obstacle):
                    done = True
                    reward-=50  # obstacle so bad
                    break

        else:  # catch the objective
            reward+=50

        prev_state = self.state

        # update the main agent model
        next_state = self.getCurrentState()

        self.main_agent.updateModelParameters(state=prev_state, action=action, reward = reward,
                                              next_state = next_state, done =done)
        self.state = next_state

        _,_,self.last_dist = self.utils_distance(self.main_agent.entity,self.goal)

        #if done:
            #return done

        # new 03/01/2018
        self.updateMap()

        return done

    # override
    def render(self):

        self.screen.fill((255, 255, 255))

        for obs in self.obstacles:
            obs.render(self.screen)

        self.goal.render(self.screen)

        for agent in self.agents:

            agent.render(self.screen)


        pygame.display.flip()

    def generateRandomPosition(self):

        print("self ",self.map.shape)
        mapY,mapX = self.map.shape

        positions = []
        for agent in (self.agents):

            valid = False
            x,y = (-1,-1)
            while not valid:

                x = random.randint(0,mapX -1 )#,random.randint(0,mapY - 1)
                #andom.seed(1)
                y = random.randint(0,mapY -1 )

                #print("x = ",x," y ",y)
                valid = True
                if (x,y) in self.invalid_pos_map:
                    valid = False

                self.invalid_pos_map.update({  (x,y)  : (x * self.tile_width,y * self.tile_height)} )


            self.agents_starting_pos.update( {agent.id: ( x * self.tile_width ,y * self.tile_height)} )
            #agent.entity.setPosition(self,( x * self.tile_width ,y * self.tile_height))


    def utils_getDirection(self,action,agent):

        x, y = agent.entity.getPosition()
        dir = (0, 0)
        if action == GD_Actions.left.value[0]:
            dir = (-1, 0)

        elif action == GD_Actions.right.value[0]:
            dir = (1, 0)

        elif action == GD_Actions.up.value[0]:
            dir = (0, -1)

        else:
            dir = (0, 1)

        return dir

    def utils_distance(self,obj1,obj2):

        x1, y1 = obj1.getPosition()
        x2, y2 = obj2.getPosition()

        distX,distY = math.fabs(x1 - x2),math.fabs(y1-y2)

        dist = math.sqrt(distX ** 2 + distY ** 2)

        return distX,distY,dist



