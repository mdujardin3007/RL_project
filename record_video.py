# record_video.py

import os
import torch
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import ppo_components  # This file should contain ActorCritic and RunningNormalizer
import custom_lunar_lander
import config 

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create environment
env = gym.make('CustomLunarLander-v0', render_mode="rgb_array", max_fuel=config.MAX_FUEL, variable_terrain=False)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Load checkpoint
checkpoint = torch.load("best_model_.pth", map_location=device,  weights_only=False)

# Load model
loaded_model = ppo_components.ActorCritic(state_dim, action_dim).to(device)
loaded_model.load_state_dict(checkpoint["model_state_dict"])
loaded_model.eval()

# Load normalizer
loaded_normalizer = ppo_components.RunningNormalizer(state_dim)
loaded_normalizer.mean = checkpoint["normalizer_mean"]
loaded_normalizer.var = checkpoint["normalizer_var"]
loaded_normalizer.count = checkpoint["normalizer_count"]

# Set up video recording
video_folder = './video'
os.makedirs(video_folder, exist_ok=True)
env = RecordVideo(env, video_folder, episode_trigger=lambda e: True)

# Run simulation and record video
state, _ = env.reset()
for _ in range(2500):
    loaded_normalizer.update([state])
    norm_state = loaded_normalizer.normalize(state)
    state_tensor = torch.FloatTensor(norm_state).unsqueeze(0).to(device)
    logits, _ = loaded_model(state_tensor)
    probs = F.softmax(logits, dim=-1)
    action = torch.argmax(probs, dim=-1).item()
    next_state, reward, done, truncated, _ = env.step(action)
    state = next_state
    if done or truncated:
        state, _ = env.reset()

env.close()
