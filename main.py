import logging

from algorithms.nsga import nsgaGA
from algorithms.sarsa import sarsaRL
from algorithms.dqn import dqnDL

from utils.load_files import load_config, load_param_ranges
from simulator.sim import simulate


# Path to config files
PARAM_PATH = "./config/parameters.json"
CONFIG_PATH_NSGA = "./config/config-nsga.json"
CONFIG_PATH_SARSA = "./config/config-sarsa.json"
CONFIG_PATH_DQN = "./config/config-dqn.json"


# Load config files
try:
    parameter_ranges = load_param_ranges(PARAM_PATH)
    config_nsga = load_config(CONFIG_PATH_NSGA)
    config_sarsa = load_config(CONFIG_PATH_SARSA)
    config_dqn = load_config(CONFIG_PATH_DQN)
except Exception as e:
    logging.error(f"Error loading config or parameter files: {e}")
    exit(1)


# Run the NSGA (Genetic Algorithm)
print("\n=== Running NSGA-II Genetic Algorithm ===")
nsga = nsgaGA(config_nsga, parameter_ranges)
pareto_front, pareto_fitness = nsga.run(simulate)
print("\n=== NSGA-II Complete ===") # FALTA DEFINIR CUAL ES EL MEJOR SETUP QUE OBTIENE EL ALGORITMO


# Run the SARSA (Reinforcement Learning Algorithm)
print("\n=== Running SARSA Reinforcement Learning ===")
sarsa = sarsaRL(config_sarsa, parameter_ranges)
best_state, best_fitness = sarsa.run(sarsa.evaluate_function)
print("\n=== SARSA Complete ===")


# Run the DQN (Deep Q-Network) (Reinforcement Learning Algorithm)
print("\n=== Running DQN Reinforcement Learning ===")
dqn = dqnDL(config_dqn, parameter_ranges)
best_setup, best_fitness_dqn = dqn.run(dqn.evaluate_function)
print("\n=== DQN Complete ===")
print(f"Best Setup: {best_setup}")
print(f"Best Fitness: {best_fitness_dqn}")
