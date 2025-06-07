#!/usr/bin/python3

import numpy as np
import time  # For measuring optimization time

ENV_SIZE = 5


class GridWorld:
    def __init__(self, env_size):
        self.env_size = env_size
        # Initialize the value function and set terminal state value to 0
        self.V = np.zeros((env_size, env_size))
        # For importance sampling weights
        self.C = np.zeros((env_size, env_size))
        self.terminal_state = (4, 4)
        self.V[self.terminal_state] = 0  # Terminal state value

        # Define actions and rewards
        self.actions = [(0, 1), (1, 0), (1, 0), (-1, 0)
                        ]  # Right, Down, Down, Up
        self.action_description = ["Right", "Down", "Down", "Up"]
        self.gamma = 0.9  # Discount factor (Problem 4)

        self.rewards = {
            (4, 4): 10,  # Terminal state
            (2, 2): -5,  # Grey states
            (3, 0): -5,
            (0, 4): -5
        }
        self.default_reward = -1  # Regular states

        self.pi_greedy = np.zeros((self.env_size, self.env_size), dtype=int)
        self.pi_str = [["" for _ in range(env_size)] for _ in range(env_size)]

    '''@brief Returns the reward for a given state
    '''

    def get_reward(self, i, j):
        return self.rewards.get((i, j), self.default_reward)
    '''@brief Returns True if the state is a terminal state
    '''

    def is_terminal_state(self, i, j):
        return (i, j) == self.terminal_state

    '''@brief Returns the full state-value function V_pi
    '''

    def get_value_function(self):
        return self.V

    '''@brief Returns the stored greedy policy
    '''

    def print_policy(self):
        for row in self.pi_str:
            print(
                " ".join([f"{action:7}" if action else "term   " for action in row]))

    '''@brief Calculate the maximum value by following a greedy policy
    '''

    def calculate_max_value(self, i, j):
        max_value = float('-inf')
        best_action = None
        best_actions_str = ""
        best_actions = []
        for action_index in range(len(self.actions)):
            next_i, next_j = self.step(action_index, i, j)
            if self.is_valid_state(next_i, next_j):
                reward = self.get_reward(i, j)
                value = reward + self.gamma * self.V[next_i, next_j]
                if value > max_value - 1e-10:  # Handle numerical precision
                    if value > max_value + 1e-10:
                        max_value = value
                        best_actions = [action_index]
                        best_actions_str = self.action_description[action_index]
                    else:
                        best_actions.append(action_index)
                        best_actions_str += "|" + \
                            self.action_description[action_index]
        if best_actions:
            best_action = best_actions[0]  # Choose first optimal action
        return max_value, best_action, best_actions_str, best_actions

    '''@brief Returns the next state given the chosen action and current state
    '''

    def step(self, action_index, i, j):
        """Compute next state given action and current state."""
        action = self.actions[action_index]
        next_i, next_j = i + action[0], j + action[1]
        if not self.is_valid_state(next_i, next_j):
            next_i, next_j = i, j  # Stay if invalid
        return next_i, next_j

    '''@brief Checks if a state is within the acceptable bounds of the environment
    '''

    def is_valid_state(self, i, j):
        valid = 0 <= i < self.env_size and 0 <= j < self.env_size
        return valid

    '''Return probability of each action under random behavior policy.
    '''

    def behavior_policy(self):
        return 1.0 / len(self.actions)  # Uniform: 1/4 = 0.25

    '''Return π(a|s) for action at state (i,j).
    '''

    def target_policy_prob(self, action_index, i, j):
        _, _, _, best_actions = self.calculate_max_value(i, j)
        if action_index in best_actions:
            # Split probability among optimal actions
            return 1.0 / len(best_actions)
        return 0.0

    '''Generate an episode using behavior policy.
    '''

    def generate_episode(self):
        episode = []
        i, j = 0, 0  # Start at (0,0)
        while not self.is_terminal_state(i, j):
            action_index = np.random.choice(len(self.actions))  # Random action
            next_i, next_j = self.step(action_index, i, j)
            reward = self.get_reward(i, j)  # Reward for current state
            episode.append((i, j, action_index, reward))
            i, j = next_i, next_j
        # Add terminal state with its reward
        terminal_reward = self.get_reward(i, j)
        episode.append((i, j, None, terminal_reward))
        return episode

    '''Update greedy policy based on current value function.
    '''

    def update_greedy_policy(self):
        for i in range(ENV_SIZE):
            for j in range(ENV_SIZE):
                if self.is_terminal_state(i, j):
                    self.pi_greedy[i, j] = 0
                    self.pi_str[i][j] = ""
                    continue
                _, self.pi_greedy[i, j], action_str, _ = self.calculate_max_value(
                    i, j)
                self.pi_str[i][j] = action_str

    '''Run off-policy Monte Carlo with importance sampling.
    '''

    def monte_carlo_off_policy(self, num_episodes):
        for episode_idx in range(num_episodes):
            episode = self.generate_episode()
            G = 0  # Return
            W = 1.0  # Importance sampling ratio
            # Process episode backward
            for t in range(len(episode) - 1, -1, -1):  # Include terminal state
                i, j, action_index, reward = episode[t]
                G = reward + self.gamma * G  # Update return
                if t < len(episode) - 1:  # Skip ratio for terminal action
                    W *= self.target_policy_prob(action_index,
                                                 i, j) / self.behavior_policy()
                self.C[i, j] += W
                if self.C[i, j] > 0:
                    self.V[i, j] += (W / self.C[i, j]) * (G - self.V[i, j])


# Run Off-policy Monte Carlo
gridworld = GridWorld(ENV_SIZE)
num_episodes = 10000

start_time = time.time()  # Start timing
gridworld.monte_carlo_off_policy(num_episodes)
runtime = time.time() - start_time  # End timing

# Print results
print("Off-policy Monte Carlo with Importance Sampling")
print(f"Optimization Time: {runtime:.6f} seconds")
print(f"Number of Episodes: {num_episodes}")
print("\nEstimated Value Function:")
for i in range(ENV_SIZE):
    row = [f"{gridworld.V[i, j]:7.2f}" for j in range(ENV_SIZE)]
    print(" ".join(row))

print("\nGreedy Policy:")
gridworld.update_greedy_policy()
gridworld.print_policy()
