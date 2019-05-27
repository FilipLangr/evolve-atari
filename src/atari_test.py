import sys
import numpy as np
import gym
import pickle
from atari_config_file import config
from cartesian.cgp import *
import time

def test_final(f, timesteps=1000, timesleep=None):
    white_reward = 0
    black_reward = 0
    env = gym.make(config.gym_params['game_name']).env
    observation = env.reset()
    observation = observation / 255.0
    for t in range(timesteps):
        env.render()
        if timesleep:
            time.sleep(timesleep)
        values = f(*np.transpose(observation))
        values = np.array([np.mean(y) for y in values])
        # Get the index of highest value, it's our action.
        action = np.argmax(values)
        # Play the action.
        #action = np.random.randint(0, 18)
        observation, reward, done, _ = env.step(action)
        reward = int(reward)
        if reward > 0:
            white_reward += reward
        elif reward < 0:
            black_reward += reward
        observation = observation / 255.0
        if done:
            break
    env.render()
    print("White %d : %d Black" % (white_reward, -black_reward))
    input("Press any key to exit.")
    env.close()

if __name__== '__main__':
    if len(sys.argv) < 3:
        print("Please provide the path to saved OptimisationResult and number of timesteps. Optionally, specify argument for time.sleep().")
    else:
        path = sys.argv[1]
        timesteps = int(sys.argv[2])
        with open(path, 'rb') as f:
            atari_cgp = Cartesian("atari_cgp", **config.cartesian_params)
            res = pickle.load(f)
            final_func = compile(res.ind)
            if len(sys.argv) > 3:
                timesleep = float(sys.argv[3])
            else:
                timesleep=None
            test_final(final_func, timesteps=timesteps, timesleep=timesleep)
