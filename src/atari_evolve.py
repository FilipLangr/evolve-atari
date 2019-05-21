import gym
from cartesian.cgp import *
from cartesian.algorithm import oneplus, optimize_constants
import numpy as np
import time
from atari_config_file import config

env = gym.make(config.gym_params['game_name']).env

def train():
    # Create CGP graph.
    atari_cgp = Cartesian("atari_cgp", **config.cartesian_params)
    # Optimise it!
    res = oneplus(optimisation_fce, cls=atari_cgp, **config.oneplus_params)
    env.close()
    return res

#@optimize_constants
#def optimisation_fce(individual, consts=()):
def optimisation_fce(individual):
    num_episodes = config.gym_params['num_episodes']
    timesteps = config.gym_params['timesteps']
    reward_sums = np.zeros(num_episodes)
    render = config.gym_params['render']

    # Get evolved function.
    #evolved_fce = individual
    evolved_fce = compile(individual)
    # Go through all episodes.
    for episode in range(num_episodes):
        observation = env.reset() / 255.0
        rewards = np.zeros(timesteps)
        # Go through all timesteps.
        for t in range(timesteps):
            if render:
                env.render()
                #time.sleep(0.05)
            # Evaluate evolved function and get 18 values for 18 actions.
            #values = evolved_fce(*np.transpose(observation), *consts)
            values = evolved_fce(*np.transpose(observation))
            # If any value is matrix, take the average of it.
            values = np.array([np.mean(y) for y in values])
            # Get the index of highest value, it's our action.
            action = np.argmax(values)
            # Play the action.
            observation, reward, done, _ = env.step(action)
            observation = observation / 255.0
            rewards[t] = reward
            if done:
                break
        reward_sums[episode] = np.sum(rewards)
    # Return mean sum of rewards per episodes (negative because we minimise).
    return -np.mean(reward_sums)

if __name__ == '__main__':
    res = train()
    print(res)
