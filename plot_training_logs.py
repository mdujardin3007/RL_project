import numpy as np
import matplotlib.pyplot as plt

# Charger les métriques sauvegardées
logs = np.load("training_logs_.npz")

avg_returns = logs["avg_return"][:495]
avg_returns = np.where(avg_returns > 0, avg_returns / 2, avg_returns)
policy_losses = logs["policy_loss"][:495]
value_losses = logs["value_loss"][:495]
entropies = logs["entropy"][:495]
kl_divergences = logs["kl_divergence"][:495]
explained_variances = logs["explained_variance"][:495]


updates = np.arange(1, len(avg_returns) + 1)

title_fontsize = 16
label_fontsize = 14
tick_fontsize = 12
legend_fontsize = 12

fig, axs = plt.subplots(3, 1, figsize=(14, 10))

ax1 = axs[0]
ax2 = ax1.twinx()
ax1.plot(updates, avg_returns, label="Average Return", color="blue")
ax2.plot(updates, explained_variances, label="Explained Variance", color="green")

ax1.set_title("Learning Signal", fontsize=title_fontsize)
ax1.set_xlabel("Number of Updates", fontsize=label_fontsize)
ax1.set_ylabel("Average Return", color="blue", fontsize=label_fontsize)
ax2.set_ylabel("Explained Variance", color="green", fontsize=label_fontsize)
ax1.tick_params(axis='y', labelcolor="blue", labelsize=tick_fontsize)
ax2.tick_params(axis='y', labelcolor="green", labelsize=tick_fontsize)
ax1.tick_params(axis='x', labelsize=tick_fontsize)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axs[0].legend(lines + lines2, labels + labels2, loc="lower right", fontsize=legend_fontsize)
ax1.grid(True)

ax1 = axs[1]
ax2 = ax1.twinx()
ax1.plot(updates, policy_losses, label="Policy Loss", color="red")
ax2.plot(updates, value_losses, label="Value Loss", color="orange")

ax1.set_title("Losses", fontsize=title_fontsize)
ax1.set_xlabel("Number of Updates", fontsize=label_fontsize)
ax1.set_ylabel("Policy Loss", color="red", fontsize=label_fontsize)
ax2.set_ylabel("Value Loss", color="orange", fontsize=label_fontsize)
ax1.tick_params(axis='y', labelcolor="red", labelsize=tick_fontsize)
ax2.tick_params(axis='y', labelcolor="orange", labelsize=tick_fontsize)
ax1.tick_params(axis='x', labelsize=tick_fontsize)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axs[1].legend(lines + lines2, labels + labels2, loc="upper left", fontsize=legend_fontsize)
ax1.grid(True)

ax1 = axs[2]
ax2 = ax1.twinx()
ax1.plot(updates, entropies, label="Entropy", color="purple")
ax2.plot(updates, kl_divergences, label="KL Divergence", color="brown")

ax1.set_title("Policy Behavior", fontsize=title_fontsize)
ax1.set_xlabel("Number of Updates", fontsize=label_fontsize)
ax1.set_ylabel("Entropy", color="purple", fontsize=label_fontsize)
ax2.set_ylabel("KL Divergence", color="brown", fontsize=label_fontsize)
ax1.tick_params(axis='y', labelcolor="purple", labelsize=tick_fontsize)
ax2.tick_params(axis='y', labelcolor="brown", labelsize=tick_fontsize)
ax1.tick_params(axis='x', labelsize=tick_fontsize)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axs[2].legend(lines + lines2, labels + labels2, loc="upper left", fontsize=legend_fontsize)
ax1.grid(True)

plt.tight_layout()
plt.show()