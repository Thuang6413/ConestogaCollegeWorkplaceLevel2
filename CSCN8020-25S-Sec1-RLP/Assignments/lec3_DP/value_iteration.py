#!/usr/bin/python3

import numpy as np
import time  # For measuring optimization time

ENV_SIZE = 5


class GridWorld():

    def __init__(self, env_size):
        self.env_size = env_size
        # Initialize the value function and set terminal state value to 0
        self.V = np.zeros((env_size, env_size))
        self.terminal_state = (4, 4)
        self.V[self.terminal_state] = 0

        # Define the transition probabilities and rewards
        self.actions = [(0, 1), (1, 0), (1, 0), (-1, 0)
                        ]  # Right, Down, Down, Up
        self.action_description = ["Right", "Down", "Down", "Up"]
        self.gamma = 1.0  # Discount factor

        # Updated reward function (Task 1)
        self.rewards = {
            (4, 4): 10,  # Terminal state
            (2, 2): -5,  # Grey states
            (3, 0): -5,
            (0, 4): -5
        }
        self.default_reward = -1  # Regular states

        self.pi_greedy = np.zeros((self.env_size, self.env_size), dtype=int)
        self.pi_str = [["" for _ in range(env_size)]
                       for _ in range(env_size)]  # Initialize pi_str

    '''@brief Returns the reward for a given state
    '''

    def get_reward(self, i, j):
        return self.rewards.get((i, j), self.default_reward)

    '''@brief Checks if the change in V is less than preset threshold
    '''

    def is_done(self, delta, theta_threshold):
        return delta <= theta_threshold  # Theta threshold for convergence

    '''@brief Returns True if the state is a terminal state
    '''

    def is_terminal_state(self, i, j):
        return (i, j) == self.terminal_state

    '''
    @brief Overwrites the current state-value function with a new one
    '''

    def update_value_function(self, V):
        self.V = np.copy(V)

    '''
    @brief Returns the full state-value function V_pi
    '''

    def get_value_function(self):
        return self.V

    '''@brief Returns the stored greedy policy
    '''

    def get_policy(self):
        return self.pi_greedy

    '''@brief Prints the policy using the action descriptions
    '''

    def print_policy(self):
        for row in self.pi_str:
            print(
                " ".join([f"{action:7}" if action else "term   " for action in row]))

    '''@brief Calculate the maximum value by following a greedy policy
    '''

    def calculate_max_value(self, i, j):
        # Find the maximum value for the current state using Bellman's equation
        max_value = float('-inf')
        best_action = None
        best_actions_str = ""
        for action_index in range(len(self.actions)):
            next_i, next_j = self.step(action_index, i, j)
            if self.is_valid_state(next_i, next_j):
                reward = self.get_reward(i, j)  # Reward based on current state
                value = reward + self.gamma * self.V[next_i, next_j]
                if value > max_value:
                    max_value = value
                    best_action = action_index
                    best_actions_str = self.action_description[action_index]
                elif value == max_value:
                    best_actions_str += "|" + \
                        self.action_description[action_index]
        return max_value, best_action, best_actions_str

    '''@brief Returns the next state given the chosen action and current state
    '''

    def step(self, action_index, i, j):
        # Deterministic transitions: P(s'|s) = 1.0 for valid moves, else stay
        action = self.actions[action_index]
        next_i, next_j = i + action[0], j + action[1]
        if not self.is_valid_state(next_i, next_j):
            next_i, next_j = i, j  # Stay in place if invalid
        return next_i, next_j

    '''@brief Checks if a state is within the acceptable bounds of the environment
    '''

    def is_valid_state(self, i, j):
        valid = 0 <= i < self.env_size and 0 <= j < self.env_size
        return valid

    def update_greedy_policy(self):
        for i in range(ENV_SIZE):
            for j in range(ENV_SIZE):
                if self.is_terminal_state(i, j):
                    self.pi_greedy[i, j] = 0  # No action in terminal state
                    self.pi_str[i][j] = ""
                    continue
                _, self.pi_greedy[i, j], action_str = self.calculate_max_value(
                    i, j)
                self.pi_str[i][j] = action_str


# Perform In-Place Value Iteration
gridworld = GridWorld(ENV_SIZE)
num_iterations = 1000
theta_threshold = 0.0001  # As a standard convergence threshold in reinforcement

start_time = time.time()  # Start timing
for iter in range(num_iterations):
    delta = 0
    for i in range(ENV_SIZE):
        for j in range(ENV_SIZE):
            if not gridworld.is_terminal_state(i, j):
                v_old = gridworld.V[i, j]
                gridworld.V[i, j], _, _ = gridworld.calculate_max_value(i, j)
                delta = max(delta, abs(v_old - gridworld.V[i, j]))
    if gridworld.is_done(delta, theta_threshold):
        break
runtime = time.time() - start_time  # End timing

# Print results
print("In-Place Value Iteration")
print(f"Optimization Time: {runtime:.6f} seconds")
print(f"Number of Iterations: {iter + 1}")
print("\nOptimal Value Function (after %d iterations):" % (iter + 1))
for i in range(ENV_SIZE):
    row = [f"{gridworld.V[i, j]:7.2f}" for j in range(ENV_SIZE)]
    print(" ".join(row))

print("\nOptimal Policy:")
gridworld.update_greedy_policy()
gridworld.print_policy()
