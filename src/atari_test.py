import sys
import numpy as np
import gym
import pickle
from atari_config_file import config
from cartesian.cgp import *

def test_final(f, timesteps=1000):
    env = gym.make(config.gym_params['game_name']).env
    observation = env.reset()
    observation = observation / 255.0
    for t in range(timesteps):
        env.render()
        values = f(*np.transpose(observation))
        values = np.array([np.mean(y) for y in values])
        # Get the index of highest value, it's our action.
        action = np.argmax(values)
        # Play the action.
        observation, reward, done, _ = env.step(action)
        observation = observation / 255.0
        if done:
            break
    env.close()

if __name__== '__main__':
    if len(sys.argv) < 3:
        print("Please provide the path to saved OptimisationResult and number of timesteps.")
    else:
        path = sys.argv[1]
        timesteps = int(sys.argv[2])
        with open(path, 'rb') as f:
            atari_cgp = Cartesian("atari_cgp", **config.cartesian_params)
            res = pickle.load(f)
            final_func = compile(res.ind)
            test_final(final_func, timesteps=timesteps)