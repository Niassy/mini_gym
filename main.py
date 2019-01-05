import pygame

#from .Environment.gridworld.gridworld_agent import *

from Environment.gridworld.gridworld_environment import *

from Machine_Learning.hyper_parameters import *

def main():


    def init_Screen(width, height):
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        return screen


    screen = init_Screen(width=GD_SCREEN[0],height=GD_SCREEN[1])

    env = GD_Environment(screen)

    done = False

    #env.render()

    # Choose your learning algorithms
    # Here we choose Q Learning
    # if you want to choose Q Learning:
    # env.setMainAgent(Brain_Type.Deep_Q_Network)
    env.setMainAgent(Brain_Type.Q_Learning)


    # Set your model to training mode
    env.main_agent.brain.trainModel()


    # define the number of episode
    # see the file constants,py
    # NUM_EPISODE is defined there
    num_episodes = NUM_EPISODE

    # This is the mxaximum number of steo the agent run
    # before stoping the episode

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

    #####################################

    if env.save_model:
        env.main_agent.brain.save_model(MODEL_PATH)
        print("Model saved to q_table")

    env.inference = True

    env.main_agent.brain.evalModel()

    i = 0
    while True:

        i+=1
        frame = 0
        done = False

        env.reset()

        env.main_agent.entity.x,env.main_agent.entity.y = 32 * random.randint(1,10),32 *random.randint(1,10)

        env.render()

        time.sleep(0.1)
        print("Episode ",i)
        while not done:

            frame+=1
            done = env.step()
            env.render()

            if frame >=max_frames:
                done =True

            time.sleep(0.1)


if __name__ == '__main__':
    main()


