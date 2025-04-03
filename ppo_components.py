import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import config 

GAMMA = config.GAMMA
LAMBDA = config.LAMBDA

class RunningNormalizer:
    def __init__(self, state_dim):
        self.mean = np.zeros(state_dim)
        self.var = np.ones(state_dim)
        self.count = 1e-4

    def update(self, x_batch):
        batch_mean = np.mean(x_batch, axis=0)
        batch_var = np.var(x_batch, axis=0)
        batch_count = len(x_batch)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

# === Actor-Critic Network ===
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        hidden = 64
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.policy_logits = nn.Linear(hidden, action_dim)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.policy_logits(x), self.value(x)

# === Rollout Buffer ===
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.returns = []

    def clear(self):
        self.__init__()

# === GAE Calculation ===
def compute_gae(next_value, rewards, dones, values):
    advantages = []
    gae = 0
    values = values + [next_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages