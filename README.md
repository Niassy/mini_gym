# mini_gym

mini_gym is a free library for developping Reinforcement Learning algorithm. 
It is written in python and use pygame for rendering the game.
It contains s aet of environment you can use for implementing your artificial intelligence algorithm.
Ii is mainly aimed to use Reinforcement Learning algorithm but the architecture is very flexible and
you can use other articial technique like Goal Orienter Behaviour,Fuzzy logic,etc...

##  There are so many Reinforcement platforms... Why using mini_gym ?

mini_gym differs from the other platform by his simplicity and his ready to go testing.
Also many of the RL platform use tensorflow as for implementing the algorithm.
Our platform will be more focused on pytorch for deep learning algorithm.
You do not need to install many depedencies to run it. You just need to have python and pygame and a interpreter.
If you want to train your agent using Deep Q Network, install pytorch

In addition,you can create easily your environment and customize the state and action of the environment.

# Installation

## Installing numpy
    pip install numpy
        
## Installing pygame
    pip install pygame
    
## Installing pytorch
    pip install torch
    
    
# Architecture

## Environment

The environment defines the rules of game.For now,we have included one environment which is the famous grid world.
Each environment hiils a number of agent which interact with it.I fyou want too create your own environment, just inherit from
this class . 

Check environment.py for more details.

## Agent
The agent is the entity who interacts with the environment. The agent is controlled by a brain who is reponsible for his decision. This class is abstract and if you have to inherit from it when you create new environment.

## Brain
The brain is the learning algorithm you attach to your agent decision making.
Currently,the engines support two brain which are the Q Learning and DeepQNetwork.
This is the main component interface you have define if you have to implemted your learning algorithm.
See brain.py to check how it was implemented. The Q Learning and the Deep Q Network algorithms inherit from the brain interface


# Using Q Learning algorithm with the gridworld environment
You can check main.py for more details of the implementation

Below are the step.

## import your environment
We import the gridworld environment. Check Environment/gridworld/gridworld_environment.py for more details.

    env = GD_Environment(screen)


## Choose your learning algorithm (Q Learning or DeepQNetwork )
Here we choose Q Learning if you want to choose Q Learning:

    env.setMainAgent(Brain_Type.Q_Learning)
If you want to choose DeepQ Network

    env.setMainAgent(Brain_Type.Deep_Q_Network)

## Make your model in training mode
 
    env.training =True

## Define the number of episodes

See Machine_Learning/hyper_parameters.py for some defined constants like the number of episodes

    num_episodes = NUM_EPISODE
    
## Train your model

    max_frames = 200

    for i in range(num_episodes):

        done = False

        env.reset(i)

        env.main_agent.entity.x,env.main_agent.entity.y = 32 * random.randint(1,10),32 *random.randint(1,10)

        env.render()

        frame = 0

        time.sleep(0.01)
        print("Episode ",i)
        while not done:

            frame+=1
            done = env.step()

            env.render()

            if frame >=max_frames:
                done =True
            time.sleep(0.01)

