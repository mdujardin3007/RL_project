# evaluate_q_table.py
import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import trange
from q_learning_train import discretize_state, create_bins


def evaluate_q_table():
  # Load Q-table
  with open("q_table_best.pkl", "rb") as f:
      q_table = pickle.load(f)

  # Setup
  env = gym.make("CustomLunarLander-v0")
  bins = create_bins()
  n_eval_episodes = 500
  rewards = []
  crashes = 0
  successes = 0
  out_of_pad = 0
  fuel_exhausted = 0

  for _ in trange(n_eval_episodes):
      state, _ = env.reset()
      disc_state = discretize_state(state, bins)
      done = False
      total_reward = 0

      while not done:
          action = np.argmax(q_table.get(disc_state, np.zeros(env.action_space.n)))
          state, reward, done, _, info = env.step(action)
          disc_state = discretize_state(state, bins)
          total_reward += reward

      rewards.append(total_reward)
      if reward == -100:
          crashes += 1
      elif info.get("landing_smoothness", {}).get("landed_on_pad", False):
          successes += 1
      elif info.get("fuel_ran_out", False):
          fuel_exhausted += 1
      else:
          out_of_pad += 1

  # Plot the evaluation metrics
  num_episodes = len(rewards)
  metrics = [
      ["Average Reward", f"{np.mean(rewards):.2f}"],
      ["Success Rate (on pad)", f"{100 * successes / num_episodes:.2f}%"],
      ["Crash Rate", f"{100 * crashes / num_episodes:.2f}%"],
      ["Landed Outside Pad", f"{100 * out_of_pad / num_episodes:.2f}%"],
      ["Ran Out of Fuel", f"{100 * fuel_exhausted / num_episodes:.2f}%"]
  ]

  column_labels = ["Metric", "Value"]

  fig, ax = plt.subplots(figsize=(6, 3))
  ax.axis('off')
  table = ax.table(cellText=metrics, colLabels=column_labels, loc='center', cellLoc='center')
  table.auto_set_font_size(False)
  table.set_fontsize(12)
  table.scale(1.5, 1.5)

  plt.title("Evaluation Metrics (Q-learning)", fontsize=16, weight='bold', pad=20)
  plt.tight_layout()
  plt.show()
