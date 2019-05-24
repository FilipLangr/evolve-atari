from collections import namedtuple
from atari_primitive_set import primitiveSet
#from tensorboardcolab import TensorBoardColab
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import pickle
import os

if not os.path.exists("tb/"):
    os.makedirs("tb/")
if not os.path.exists("pickles/"):
    os.makedirs("pickles/")

def save_result(res):
    with open('pickles/best_res_rand%d.pickle' % config.oneplus_params["random_state"], 'wb') as f:
        pickle.dump(res, f)

tb_writer = tf.summary.FileWriter('tb/')
def tb_callback(res):
    print("Loss of the last generation (lower is better): %.3f" % res.fun)
    val = summary_pb2.Summary.Value(tag="Training loss rand%d" % config.oneplus_params["random_state"], simple_value=res.fun)
    summary = summary_pb2.Summary(value=[val])
    tb_writer.add_summary(summary, tb_callback.cntr)
    tb_callback.cntr += 1
    save_result(res)

tb_callback.cntr = 0

#######################################
# Here we define a config for training.
#######################################

Config = namedtuple('Config', "cartesian_params oneplus_params gym_params")
config = Config(
    # Parameters defining the CGP graph.
    cartesian_params = {
        'primitive_set': primitiveSet,
        'n_columns': 40,
        'n_rows': 1,
        'n_back': 4, # How far can a node in the matrix look back for its inputs?
        'n_out': 18, # Number of output nodes, we have 18 actions.
    },
    # Parameters defining the optimisation process by one plus lambda.
    # https://cartesian.readthedocs.io/en/latest/_modules/cartesian/algorithm.html#oneplus
    oneplus_params = {
        'lambda_': 9, # Number of offsprings per generation.
        'n_mutations': 4, # Number of mutations per offspring.
        'mutation_method': "active",
        'maxiter': 1112, # Maximum number of generations.
        'maxfev': None, # Maximum number of function evaluations, None means infinite.
        'f_tol': -100, # Absolute error in metric(ind) between iterations that is acceptable for convergence.for
                     # ??? Stopping criterion ???
        'n_jobs': 1, # Number of parallel jobs, if we go parallel.
        'random_state': None,
        'seed': None,
        #'callback': lambda res: print("Loss of the last generation (lower is better): %.3f" % res.fun),
        'callback': tb_callback,
    },
    # Parameters defining the openAI gym game.
    gym_params = {
        'game_name': 'Boxing-v0',
        'num_episodes': 2, # Number of box rounds.
        'timesteps': 1000, # Time steps of one box round.
        'render': False # Do we want to see the game?
    }
)