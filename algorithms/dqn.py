import random
import numpy as np
import tensorflow as tf

from collections import deque
from typing import Callable, Tuple, List, Dict
from simulator.sim import simulate


class dqnDL:
    def __init__(self, config: Dict[str, float], parameter_ranges: Dict[str, Tuple[float, float]]) -> None:
        """
        Initializes the DQN reinforcement learning agent.

        Args:
            config (dict): Configuration dictionary containing alpha, gamma, epsilon, episodes, steps, and replay settings.
            parameter_ranges (dict): Ranges of parameters for car setups.
        """
        self.config = config
        self.parameter_ranges = parameter_ranges
        self.alpha = config["alpha"]  # Learning rate
        self.gamma = config["gamma"]  # Discount factor
        self.epsilon = config["epsilon"]  # Exploration rate
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.epsilon_min = config.get("epsilon_min", 0.1)
        self.episodes = config["episodes"]  # Number of episodes
        self.steps = config["steps"]  # Maximum steps per episode
        self.batch_size = config.get("batch_size", 32)
        self.replay_buffer = deque(maxlen=config.get("replay_buffer_size", 10000))

        self.num_actions = len(parameter_ranges)
        self.state_dim = len(parameter_ranges)

        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.update_target_network()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)


    def _build_network(self) -> tf.keras.Model:
        """
        Builds the Q-network model.

        Returns:
            tf.keras.Model: The Q-network.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu", input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(self.num_actions)
        ])
        return model
    

    def update_target_network(self) -> None:
        """
        Updates the target network weights to match the Q-network.
        """
        self.target_network.set_weights(self.q_network.get_weights())


    def select_action(self, state: np.ndarray) -> int:
        """
        Selects an action using an epsilon-greedy strategy.

        Args:
            state (np.ndarray): The current state.

        Returns:
            int: The selected action index.
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)  # Explore
        q_values = self.q_network.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])  # Exploit
    

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
        setup = [
            random.uniform(*parameter_ranges[param]) for param in parameter_ranges
        ]

        lap_time, max_speed, distance_covered, damage = simulate(setup)
        
        fitness = -lap_time + 0.1 * max_speed + 0.01 * distance_covered - 0.5 * damage
        return fitness


    def store_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Stores an experience tuple in the replay buffer.

        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state after the action.
            done (bool): Whether the episode is done.
        """
        self.replay_buffer.append((state, action, reward, next_state, done))


    def train_step(self) -> None:
        """
        Performs a single training step using a batch of experiences.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)

        q_values_next = self.target_network.predict(next_states, verbose=0)
        targets = rewards + self.gamma * np.max(q_values_next, axis=1) * (1 - dones)

        q_values = self.q_network.predict(states, verbose=0)
        for i, action in enumerate(actions):
            q_values[i, action] = targets[i]

        self.q_network.fit(states, q_values, epochs=1, verbose=0)
        

    def run(self, evaluate_function: Callable[[int, int, Dict[str, Tuple[float, float]]], float]) -> Tuple[List, float]:
        """
        Executes the DQN algorithm to find the optimal policy.

        Args:
            evaluate_function (callable): Function to evaluate the fitness of a setup.

        Returns:
            Tuple[List, float]: Best setup and its fitness score.
        """
        best_fitness = -float("inf")
        best_setup = None

        for episode in range(self.episodes):
            print(f"\n--- Episode {episode + 1}/{self.episodes} ---")
            state = np.random.uniform(0, 1, self.state_dim)  # Initialize a random state

            for step in range(self.steps):
                action = self.select_action(state)
                reward = evaluate_function(state, action, self.parameter_ranges)

                next_state = np.random.uniform(0, 1, self.state_dim)
                done = step == self.steps - 1

                self.store_experience(state, action, reward, next_state, done)
                self.train_step()

                state = next_state

                if reward > best_fitness:
                    best_fitness = reward
                    best_setup = (state, action)

                print(f"Step {step + 1}/{self.steps}: Action={action}, Reward={reward}")

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.update_target_network()

        print("\n--- Best Setup Found ---")
        print(f"Best Fitness: {best_fitness}")
        return best_setup, best_fitness
