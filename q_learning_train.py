# Re-imports after kernel reset
import numpy as np
import gymnasium as gym
import torch
import pickle
from collections import defaultdict
from tqdm import trange
import custom_lunar_lander

# Discretization helper
def discretize_state(state, bins):
    idx = []
    for i in range(len(bins)):
        idx.append(np.digitize(state[i], bins[i]))
    return tuple(idx)

# Create bins for discretization (9D state)
# These limits come from the custom environnement
def create_bins(num_bins=16):
    bins = [
        np.linspace(-2.5, 2.5, num_bins),      # pos x
        np.linspace(-2.5, 2.5, num_bins),      # pos y
        np.linspace(-10, 10, num_bins),        # vel x
        np.linspace(-10, 10, num_bins),        # vel y
        np.linspace(-2 * np.pi, 2 * np.pi, num_bins),  # angle
        np.linspace(-10, 10, num_bins),        # angular vel
        np.array([0.5]),                       # leg contact left (binary)
        np.array([0.5]),                       # leg contact right (binary)
        np.linspace(0, 1, num_bins),           # fuel
    ]
    return bins


def q_learning_train():
  # Initialize environment
  env = gym.make("CustomLunarLander-v0")
  n_actions = env.action_space.n
  bins = create_bins()
  q_table = defaultdict(lambda: np.zeros(n_actions))

  # Hyperparameters
  alpha = 0.1
  gamma = 0.99
  epsilon = 1.0
  epsilon_min = 0.05
  epsilon_decay = 0.995
  n_episodes = 20000
  max_steps = 1000

  best_reward = -np.inf
  best_q_table = None

  reward_history = []

  # Training loop
  for episode in trange(n_episodes, desc="Training Q-Learning"):
      state, _ = env.reset()
      disc_state = discretize_state(state, bins)
      total_reward = 0

      for t in range(max_steps):
          if np.random.rand() < epsilon:
              action = env.action_space.sample()
          else:
              action = np.argmax(q_table[disc_state])

          next_state, reward, done, truncated, _ = env.step(action)
          next_disc_state = discretize_state(next_state, bins)

          # Q-learning update
          best_next_action = np.max(q_table[next_disc_state])
          q_table[disc_state][action] += alpha * (reward + gamma * best_next_action - q_table[disc_state][action])
          disc_state = next_disc_state
          total_reward += reward

          if done or truncated:
              break

      reward_history.append(total_reward)

      # Save best model
      if total_reward > best_reward:
          best_reward = total_reward
          best_q_table = dict(q_table)

      epsilon = max(epsilon * epsilon_decay, epsilon_min)

  # Save best Q-table to file
  model_path = "q_table_best.pkl"

  with open(model_path, "wb") as f:
      pickle.dump(best_q_table, f)
