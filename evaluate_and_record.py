# evaluate_and_record.py

import os
import numpy as np
import torch
import gymnasium as gym
from tqdm import trange  # Optional: for a progress bar
import config
import ppo_components
import custom_lunar_lander

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create environment
env = gym.make('CustomLunarLander-v0', render_mode="rgb_array", max_fuel=300, variable_terrain=False)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Load checkpoint
checkpoint = torch.load("best_model_.pth", map_location=device,  weights_only=False)

# Load model
loaded_model = ppo_components.ActorCritic(state_dim, action_dim).to(device)
loaded_model.load_state_dict(checkpoint["model_state_dict"])
loaded_model.eval()

loaded_normalizer = ppo_components.RunningNormalizer(state_dim)
loaded_normalizer.mean = checkpoint["normalizer_mean"]
loaded_normalizer.var = checkpoint["normalizer_var"]
loaded_normalizer.count = checkpoint["normalizer_count"]

def evaluate_and_record(num_eval_episodes=500, max_timesteps=1000):
    # Initialize environment
    env = gym.make('CustomLunarLander-v0', render_mode=None, max_fuel=config.MAX_FUEL, variable_terrain=False)
    
    # Storage for evaluation metrics and trajectories
    eval_rewards = []
    successful_landings = 0
    crashes = 0
    landed_outside_pad = 0
    ran_out_of_fuel = 0
    final_fuel = []
    crashed_flags = []
    
    trajectories_x = []
    trajectories_y = []
    
    # Evaluation loop over episodes
    for episode in trange(num_eval_episodes, desc="Evaluating"):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        info = {}
        
        episode_x = []
        episode_y = []
        timestep = 0
        
        while not (done or truncated) and timestep < max_timesteps:
            # Record x and y positions
            episode_x.append(state[0])
            episode_y.append(state[1])
            
            # Normalize state and perform a forward pass
            loaded_normalizer.update([state])
            norm_state = loaded_normalizer.normalize(state)
            state_tensor = torch.FloatTensor(norm_state).unsqueeze(0).to(device)
            logits, _ = loaded_model(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs, dim=-1).item()
            
            # Step the environment
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            state = next_state
            timestep += 1
        
        # If the episode ends early, pad the trajectory to have uniform length
        while len(episode_x) < max_timesteps:
            episode_x.append(episode_x[-1])
            episode_y.append(episode_y[-1])
        
        trajectories_x.append(episode_x)
        trajectories_y.append(episode_y)
        eval_rewards.append(total_reward)
        final_fuel.append(state[-1] * config.MAX_FUEL)  # Assuming the last element of state is fuel
        
        # Outcome classification based on info from the environment
        landing_info = info.get('landing_smoothness')
        if landing_info and landing_info['landed_on_pad']:
            successful_landings += 1
            crashed_flags.append(0)
        elif landing_info and not landing_info['landed_on_pad']:
            landed_outside_pad += 1
            crashed_flags.append(0)
        elif info.get('fuel_ran_out', False):
            ran_out_of_fuel += 1
            crashed_flags.append(1)
        elif reward == -100:
            crashes += 1
            crashed_flags.append(1)
        else:
            crashed_flags.append(1)
    
    env.close()
    
    # Convert trajectories to NumPy arrays
    trajectories_x = np.array(trajectories_x)
    trajectories_y = np.array(trajectories_y)
    
    # Return the collected data as a dictionary
    results = {
        "eval_rewards": np.array(eval_rewards),
        "trajectories_x": trajectories_x,
        "trajectories_y": trajectories_y,
        "final_fuel": np.array(final_fuel),
        "crashed_flags": np.array(crashed_flags),
        "successful_landings": successful_landings,
        "crashes": crashes,
        "landed_outside_pad": landed_outside_pad,
        "ran_out_of_fuel": ran_out_of_fuel
    }
    
    return results