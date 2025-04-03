# train_ppo.py

import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import ppo_components 
import config
import custom_lunar_lander # Register custom env

# Hyperparameters
MAX_FUEL = config.MAX_FUEL
TOTAL_UPDATES = config.TOTAL_UPDATES
ROLLOUT_STEPS = config.ROLLOUT_STEPS
EPOCHS = config.EPOCHS
BATCH_SIZE = config.BATCH_SIZE
CLIP_EPS = config.CLIP_EPS
ENTROPY_COEFF = config.ENTROPY_COEFF
LR = config.LR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Create environment
env = gym.make('CustomLunarLander-v0', render_mode="rgb_array", max_fuel=MAX_FUEL, variable_terrain=False)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize model, optimizer, normalizer, and buffer
model = ppo_components.ActorCritic(state_dim, action_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
normalizer = ppo_components.RunningNormalizer(state_dim)
buffer = ppo_components.RolloutBuffer()

state, _ = env.reset()

# Logging variables
best_avg_return = -float('inf')
log_avg_return = []
log_policy_loss = []
log_value_loss = []
log_entropy = []
log_kl_divergence = []
log_explained_variance = []

for update in range(TOTAL_UPDATES):
    buffer.clear()
    episode_return = 0
    episode_returns = []

    # Rollout phase
    for _ in range(ROLLOUT_STEPS):
        # Update and normalize the current state
        normalizer.update([state])
        norm_state = normalizer.normalize(state)

        state_tensor = torch.FloatTensor(norm_state).unsqueeze(0).to(device)
        logits, value = model(state_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, done, truncated, _ = env.step(action.item())

        # Reward shaping: slight penalty and additional penalty if vertical speed > 0
        reward -= 0.05
        vertical_speed = state[3]
        if vertical_speed > 0:
            reward -= 0.1

        # Store the transition in the buffer
        buffer.states.append(norm_state)
        buffer.actions.append(action.item())
        buffer.log_probs.append(log_prob.item())
        buffer.rewards.append(reward)
        buffer.dones.append(done or truncated)
        buffer.values.append(value.item())

        state = next_state
        episode_return += reward
        if done or truncated:
            episode_returns.append(episode_return)
            episode_return = 0
            state, _ = env.reset()

    # Compute advantages using GAE
    with torch.no_grad():
        norm_state = normalizer.normalize(state)
        state_tensor = torch.FloatTensor(norm_state).unsqueeze(0).to(device)
        _, next_value = model(state_tensor)
        next_value = next_value.item()

    advantages = ppo_components.compute_gae(next_value, buffer.rewards, buffer.dones, buffer.values)
    returns = [adv + val for adv, val in zip(advantages, buffer.values)]
    buffer.returns = returns

    # Convert collected data to tensors
    states = torch.FloatTensor(buffer.states).to(device)
    actions = torch.LongTensor(buffer.actions).to(device)
    old_log_probs = torch.FloatTensor(buffer.log_probs).to(device)
    advantages = torch.FloatTensor(advantages).to(device)
    returns = torch.FloatTensor(returns).to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    total_kl = 0
    num_minibatches = 0

    dataset_size = len(states)
    # Optimization phase: iterate through data for multiple epochs
    for epoch in range(EPOCHS):
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)
        for start in range(0, dataset_size, BATCH_SIZE):
            end = start + BATCH_SIZE
            mb_idx = indices[start:end]
            mb_states = states[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns = returns[mb_idx]

            logits, values = model(mb_states)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - mb_old_log_probs)
            clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
            policy_loss = -torch.min(ratio * mb_advantages, clipped * mb_advantages).mean()
            value_loss = F.mse_loss(values.squeeze(), mb_returns)
            loss = policy_loss + 0.5 * value_loss - ENTROPY_COEFF * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_kl += (mb_old_log_probs - new_log_probs).mean().item()
            num_minibatches += 1

    avg_return = np.mean(episode_returns) if episode_returns else -np.inf

    with torch.no_grad():
        pred_values = model(states)[1].squeeze().cpu().numpy()
    returns_np = returns.cpu().numpy()
    explained_var = 1 - np.var(returns_np - pred_values) / (np.var(returns_np) + 1e-8)

    log_avg_return.append(avg_return)
    log_policy_loss.append(total_policy_loss / num_minibatches)
    log_value_loss.append(total_value_loss / num_minibatches)
    log_entropy.append(total_entropy / num_minibatches)
    log_kl_divergence.append(total_kl / num_minibatches)
    log_explained_variance.append(explained_var)

    if avg_return > best_avg_return:
        best_avg_return = avg_return
        torch.save({
            "model_state_dict": model.state_dict(),
            "normalizer_mean": normalizer.mean,
            "normalizer_var": normalizer.var,
            "normalizer_count": normalizer.count
        }, "best_model_.pth")

    print(f'Update {update+1}, Average Reward: {avg_return:.2f}')

# Save training logs
np.savez("training_logs_.npz",
         avg_return=log_avg_return,
         policy_loss=log_policy_loss,
         value_loss=log_value_loss,
         entropy=log_entropy,
         kl_divergence=log_kl_divergence,
         explained_variance=log_explained_variance)

env.close()
