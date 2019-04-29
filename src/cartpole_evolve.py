import gym
from cartesian.cgp import *
from cartesian.algorithm import oneplus, optimize_constants
import numpy as np
from cartpole_config_file import config

env = gym.make(config.gym_params['game_name']).env

def train():
    # Create CGP graph.
    atari_cgp = Cartesian("cartpole_cgp", **config.cartesian_params)
    # Optimise it!
    res = oneplus(optimisation_fce, cls=atari_cgp, **config.oneplus_params)
    env.close()
    return res

@optimize_constants
def optimisation_fce(individual, consts=()):
    num_episodes = config.gym_params['num_episodes']
    timesteps = config.gym_params['timesteps']
    reward_sums = np.zeros(num_episodes)
    render = config.gym_params['render']

    # Get evolved function.
    evolved_fce = individual
    # Go through all episodes.
    for episode in range(num_episodes):
        observation = env.reset()
        rewards = np.zeros(timesteps)
        # Go through all timesteps.
        for t in range(timesteps):
            if render:
                env.render()
            # Evaluate evolved function and get 2 values for 2 actions.
            values = evolved_fce(*observation, *consts)
            # If any value is matrix, take the average of it.
            values = np.array([np.mean(y) for y in values])
            # Get the index of highest value, it's our action.
            action = np.argmax(values)
            # Play the action.
            observation, reward, done, _ = env.step(action)
            rewards[t] = reward
            if done:
                break
        reward_sums[episode] = np.sum(rewards) / len(rewards)
    print(-np.mean(reward_sums))
    # Return mean sum of rewards per episodes (negative because we minimise).
    return -np.mean(reward_sums)

def test_final(f, timesteps=1000):
    env = gym.make(config.gym_params['game_name']).env
    observation = env.reset()
    for t in range(timesteps):
        env.render()
        values = f(*observation)
        # If any value is matrix, take the average of it.
        values = np.array([np.mean(y) for y in values])
        # Get the index of highest value, it's our action.
        action = np.argmax(values)
        # Play the action.
        observation, reward, done, _ = env.step(action)
        if done:
            break
    env.close()

if __name__ == '__main__':
    res = train()
    print(res)
    # Test final function.
    final_func = compile(res.ind)
    test_final(final_func, timesteps=1000)
