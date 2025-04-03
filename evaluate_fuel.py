#!/usr/bin/env python

import numpy as np
import torch
import gymnasium as gym
from tqdm import trange
import matplotlib.pyplot as plt
from ppo_components import ActorCritic, RunningNormalizer
import custom_lunar_lander

# === Load Model and Normalizer ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("best_model_.pth", map_location=device,  weights_only=False)

# To determine state_dim and action_dim, create a temporary environment
temp_env = gym.make('CustomLunarLander-v0', render_mode=None, max_fuel=300, variable_terrain=False)
state_dim = temp_env.observation_space.shape[0]
action_dim = temp_env.action_space.n
temp_env.close()

loaded_model = ActorCritic(state_dim, action_dim).to(device)
loaded_model.load_state_dict(checkpoint["model_state_dict"])
loaded_model.eval()

loaded_normalizer = RunningNormalizer(state_dim)
loaded_normalizer.mean = checkpoint["normalizer_mean"]
loaded_normalizer.var = checkpoint["normalizer_var"]
loaded_normalizer.count = checkpoint["normalizer_count"]

# === Parameters for Evaluation ===
num_eval_episodes = 100  # Increase for higher accuracy if needed
max_timesteps = 1000     # Maximum timesteps per episode to prevent infinite loops
fuel_levels = np.arange(50, 310, 50)  # Fuel levels: [50, 100, 150, 200, 250, 300]

# === Storage for Average Rewards ===
fuel_rewards = []
def evaluate_fuel():
  # === Evaluation Loop for Each Fuel Level ===
  for fuel in fuel_levels:
      print(f"\nEvaluating Fuel Level: {fuel}")
      env = gym.make('CustomLunarLander-v0', render_mode=None, max_fuel=fuel, variable_terrain=False)
      total_rewards = []
      
      for episode in trange(num_eval_episodes, desc=f"Fuel {fuel}"):
          state, _ = env.reset()
          done = False
          truncated = False
          episode_reward = 0
          timestep = 0
          
          while not (done or truncated) and timestep < max_timesteps:
              # Normalize and forward pass through the model
              loaded_normalizer.update([state])
              norm_state = loaded_normalizer.normalize(state)
              state_tensor = torch.FloatTensor(norm_state).unsqueeze(0).to(device)
              logits, _ = loaded_model(state_tensor)
              probs = torch.softmax(logits, dim=-1)
              action = torch.argmax(probs, dim=-1).item()
              
              # Step the environment
              next_state, reward, done, truncated, _ = env.step(action)
              episode_reward += reward
              state = next_state
              timestep += 1
          
          total_rewards.append(episode_reward)
      
      average_reward = np.mean(total_rewards)
      print(f"Avg Reward for Fuel {fuel}: {average_reward:.2f}")
      fuel_rewards.append(average_reward)
      env.close()

  plt.figure(figsize=(8, 6))
  plt.plot(fuel_levels, fuel_rewards, marker='o')
  plt.title("Average Reward vs Fuel Level")
  plt.xlabel("Fuel Level")
  plt.ylabel("Average Reward")
  plt.grid(True)
  plt.savefig("fuel_rewards.pdf")
  plt.show()

