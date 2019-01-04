# mini_gym

mini_gym is a free library for developping Reinforcement Learning algorithm. 
It is written in python and use pygame for rendering the game.
It contains s aet of environment you can use for implementing your artificial intelligence algorithm.

1 - There are so many RL platforms... Why using mini_gym_ai?

Mmini_gym differs from the other platform by his simplicity and his ready to go testting.
You do not need to install many depedencies to run it. You just need to have python and pygame and a interpreter.
In addition,you can create easily your environment and cmustomize the state and action of the environment.


1 Architecture
    - main compoonents:
        Environment
        State
        Actions
        Agent
        Brain


mini_gym architecture:

Environment

The environment defines the rules of game.For now,we have included one environment which is the famous grid world.

GridWorld Environment

In this environment,the agent must navigate to the goal while avoiding the obstacles.
The agent get reward of +50 if reach the goal
The agent get reward of -50 if hits an obstacle
The agent get reward of -1 for each move
The agent et reward of +3 if he gets closer to his goal


4 - Implemeing an agent that plays GridWorld
You can check test.py for the implemtation

Below are the step.


############## Importing the gridworld environment ###################

#See test.py for this implentation

# import your environment
env = gridworld()

# define your agent
agent =Agent()

# Set the brain of your agent (Q Learning )

brain  = QLearning()

# If you want to use the Deep Q netwoerk
brain = DeepQNetwork

agent.setBrain(brain)

# Make your training
env.training =True

while not done:
   
  env.render()

  for i in range(NUM_EPISODE):
     
     env.step()
     
     env.render()
     
     
  ############# MAKING INFERENCE aka Using your model to predict
  
  # set te model to test mode. Note that it will have no effect if QLearning
  # is used as a Brain
  env.agent.brain.eval()
  
  
  # Notify the agent that we are test mode
  env.agent.brain.inference = True
  

while not done:
   
  env.render()

  for i in range(NUM_EPISODE):
     
     env.step()
     
     env.render()
  


Check ... to see how it was implented
