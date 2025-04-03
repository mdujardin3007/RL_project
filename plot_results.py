# plot_results.py

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_results(results):
    num_episodes = len(results["eval_rewards"])

    metrics = [
        ["Average Reward", f"{results['eval_rewards'].mean():.2f}"],
        ["Success Rate (on pad)", f"{100 * results['successful_landings'] / num_episodes:.2f}%"],
        ["Crash Rate", f"{100 * results['crashes'] / num_episodes:.2f}%"],
        ["Landed Outside Pad", f"{100 * results['landed_outside_pad'] / num_episodes:.2f}%"],
        ["Ran Out of Fuel", f"{100 * results['ran_out_of_fuel'] / num_episodes:.2f}%"]
    ]

    column_labels = ["Metric", "Value"]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    table = ax.table(cellText=metrics, colLabels=column_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.5)

    plt.title("Evaluation Metrics", fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    print("\n" * 3)

    # --- Plot 1: Distribution of Total Rewards ---
    plt.figure(figsize=(8, 6))
    plt.hist(results['eval_rewards'], bins=20, edgecolor='black')
    plt.title('Distribution of Total Rewards Over Episodes')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("\n" * 3)

    # --- Plot 2: Scatter Plot of Remaining Fuel vs Total Reward ---
    final_fuel = np.array(results['final_fuel'])
    eval_rewards_scaled = np.array(results['eval_rewards'])
    crashed_flags = np.array(results['crashed_flags'])

    plt.figure(figsize=(8, 6))
    plt.scatter(final_fuel[crashed_flags == 0], eval_rewards_scaled[crashed_flags == 0],
                color='green', label='Landed Safely')
    plt.scatter(final_fuel[crashed_flags == 1], eval_rewards_scaled[crashed_flags == 1],
                color='red', label='Crashed')
    plt.xlabel("Remaining Fuel")
    plt.ylabel("Total Reward")
    plt.title("Remaining Fuel vs Total Reward (Crash vs Safe Landing)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("\n" * 3)

    # --- Plot 3: Mean Trajectory with Variability ---
    num_episodes = results['trajectories_x'].shape[0]
    trajectories = []
    for i in range(num_episodes):
        traj = np
