# The number of episode to run
NUM_EPISODE = 200

# Used for exploration strategy
# check Arthur Juliani repository on Reinforcement Learning
ANNELING_STEPS = 20000

# If true the model will be saved when finish the training
SAVE_MODEL = False

# If true you will load from a model
LOAD_MODEL = False

############# Neural Netork approach #################

# If you are using Deep Q Network, this defines your experience replay
# check https://deepmind.com/research/dqn/
REPLAY_BUFFER_SIZE = 10000

# The number of batch to use for the training. Also used in Deep Q Network
BATCH_SIZE = 32




