# evolve-atari
Evolution of AI for playing Atari games (we start with Boxing game) using Cartesian Genetic Programming. The (python) implementation is derived from [Evolving simple programs for playing Atari games](https://arxiv.org/abs/1806.05695), originally written in Julia. Used CGP library is [cartesian](https://github.com/ohjeah/cartesian), Atari environement is provided by [OpenAI Gym](https://gym.openai.com/).

This is a source code of the project for Evolutionary Computing course at RMIT.

## Installation
Make sure you have `python3.7` installed. Then install requirements with:

`pip3.7 install -r requirements.txt`

## Running

In `src/` folder, a framework for training programs for openAI environments using Cartesian Genetic Programming is implemented. To perform training on Atari game (Boxing game by default), just run `python3.7 atari_evolve.py`. You can also specify training parameters and function pool in `atari_config_file.py`, `atari_primitive_set.py` and `cgp_functions.py`. Files with `cartpole_` prefix show our CGP framework on a simple CartPole example.

## Preliminary experiments
Jupyter notebook `preliminary_experiments.ipynb` contains our simple implementation of evolutionary algorithm with tournament selection method, crossover using the blending method and mutation. It is used to evolve a weight vector for linear model of simple CartPole environment.
