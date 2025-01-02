import random
import numpy as np

from typing import Callable, Tuple, List, Dict
from simulator.sim import simulate


class sarsaRL:
    def __init__(self, config: Dict[str, float], parameter_ranges: Dict[str, Tuple[float, float]]) -> None:
        """
        Initializes the SARSA reinforcement learning agent.

        Args:
            config (dict): Configuration dictionary containing alpha, gamma, epsilon, episodes, and steps.
            parameter_ranges (dict): Ranges of parameters for car setups.
        """
        self.config = config
        self.parameter_ranges = parameter_ranges
        self.alpha = config["alpha"]  # Learning rate
        self.gamma = config["gamma"]  # Discount factor
        self.epsilon = config["epsilon"]  # Exploration rate
        self.episodes = config["episodes"]  # Number of episodes
        self.steps = config["steps"]  # Maximum steps per episode
        self.q_table = self._initialize_q_table()


    def _initialize_q_table(self) -> np.ndarray:
        """
        Initializes the Q-table with random values for each state-action pair.

        Returns:
            np.ndarray: The Q-table with dimensions based on states and actions.
        """
        num_states = len(self.parameter_ranges)
        num_actions = len(self.parameter_ranges)  # ALOMEJOR HAY QUE AÑADIR MAS ACCIONES (>12)
        return np.random.uniform(low=-1, high=1, size=(num_states, num_actions))  # ALOMEJOR AÑADIR UN RANGO MAS GRANDE
    

    def select_action(self, state: int) -> int:
        """
        Selects an action using an epsilon-greedy strategy.

        Args:
            state (int): The current state index.

        Returns:
            int: The selected action index.
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, len(self.q_table[state]) - 1)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit
        

    @staticmethod
    def evaluate_function(state: int, action: int, parameter_ranges: Dict[str, Tuple[float, float]]) -> float:
        """
        Evaluates a fitness score for a given state and action using the simulate function.

        Args:
            state (int): Current state index.
            action (int): Action index leading to a car setup.
            parameter_ranges (dict): Dictionary of parameter ranges for the car setup.

        Returns:
            float: Composite fitness score.
        """
        # Map state and action to a specific car setup
        setup = [
            random.uniform(*parameter_ranges[param]) for param in parameter_ranges
        ]

        # Simulate the performance
        lap_time, max_speed, distance_covered, damage = simulate(setup)

        # Define a fitness function (example: prioritize lap time and minimize damage)
        fitness = -lap_time + 0.1 * max_speed + 0.01 * distance_covered - 0.5 * damage
        return fitness
    

    def run(self, evaluate_function: Callable[[int, int, Dict[str, Tuple[float, float]]], float]) -> Tuple[List, float]:
        """
        Executes the SARSA algorithm to find the optimal policy.

        Args:
            evaluate_function (callable): Function to evaluate the fitness of a setup.

        Returns:
            Tuple[List, float]: Best setup (state and action) and its fitness score.
        """
        best_setup = None
        best_fitness = -float("inf")

        for episode in range(self.episodes):
            print(f"\n--- Episode {episode + 1}/{self.episodes} ---")
            state = random.randint(0, len(self.parameter_ranges) - 1)  # Start at a random state
            action = self.select_action(state)

            for step in range(self.steps):
                # Evaluate the current state-action pair
                reward = evaluate_function(state, action, self.parameter_ranges)

                # Select next state and action
                next_state = random.randint(0, len(self.parameter_ranges) - 1)
                next_action = self.select_action(next_state)

                # Update Q-value
                self.q_table[state, action] += self.alpha * (
                    reward + self.gamma * self.q_table[next_state, next_action] - self.q_table[state, action]
                )

                # Update state and action
                state, action = next_state, next_action

                # Track the best setup
                if reward > best_fitness:
                    best_fitness = reward
                    best_setup = (state, action)

                print(f"Step {step + 1}/{self.steps}: State={state}, Action={action}, Reward={reward}")

        print("\n--- Best Setup Found ---")
        
        # Extract and print the actual parameter values from best_setup
        best_state, best_action = best_setup
        best_parameters = [
            random.uniform(*self.parameter_ranges[param]) for param in self.parameter_ranges
        ]
        
        print(f"Best Setup (State {best_state}, Action {best_action}):")
        for param, value in zip(self.parameter_ranges, best_parameters):
            print(f"  {param}: {value}")
        
        print(f"Best Fitness: {best_fitness}")

        return best_setup, best_fitness
