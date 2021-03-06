from collections import namedtuple
from cartpole_primitive_set import primitiveSet

#######################################
# Here we define a config for training.
#######################################

Config = namedtuple('Config', "cartesian_params oneplus_params gym_params")
config = Config(
    # Parameters defining the CGP graph.
    cartesian_params = {
        'primitive_set': primitiveSet,
        'n_columns': 11,
        'n_rows': 1,
        'n_back': 7, # How far can a node in the matrix look back for its inputs?
        'n_out': 2, # Number of output nodes, we have 2 actions.
    },
    # Parameters defining the optimisation process by one plus lambda.
    # https://cartesian.readthedocs.io/en/latest/_modules/cartesian/algorithm.html#oneplus
    oneplus_params = {
        'lambda_': 4, # Number of offsprings per generation.
        'n_mutations': 3, # Number of mutations per offspring.
        'mutation_method': "active",
        'maxiter': 100, # Maximum number of generations.
        'maxfev': None, # Maximum number of function evaluations, None means infinite.
        'f_tol': -0.999, # Absolute error in metric(ind) between iterations that is acceptable for convergence.
        'n_jobs': 1, # Number of parallel jobs, if we go parallel.
        'random_state': 13,
        'seed': None,
        'callback': lambda res: print("Loss of the last generation (lower is better): %.3f" % res.fun),
        #'callback': tb_callback,
    },
    # Parameters defining the openAI gym game.
    gym_params = {
        'game_name': 'CartPole-v0',
        'num_episodes': 10, # Number of box rounds.
        'timesteps': 1000, # Time steps of one box round.
        'render': False # Do we want to see the game?
    }
)