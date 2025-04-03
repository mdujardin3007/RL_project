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

    # --- Plot 1: Distribution of Total Rewards ---
    plt.figure(figsize=(8, 6))
    plt.hist(results['eval_rewards'], bins=20, edgecolor='black')
    plt.title('Distribution of Total Rewards Over Episodes')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.show()
    
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
    plt.show()
    
    # --- Plot 3: Mean Trajectory with Variability ---
    # Combine trajectories_x and trajectories_y into a list of (timesteps, 2) arrays
    num_episodes = results['trajectories_x'].shape[0]
    trajectories = []
    for i in range(num_episodes):
        traj = np.stack((results['trajectories_x'][i], results['trajectories_y'][i]), axis=1)
        trajectories.append(traj)
    
    # Determine the maximum number of timesteps (should be consistent if already padded)
    max_timesteps = max(len(traj) for traj in trajectories)
    
    # Pad shorter trajectories with NaNs for proper averaging
    padded_trajectories = np.full((num_episodes, max_timesteps, 2), np.nan)
    for i, traj in enumerate(trajectories):
        padded_trajectories[i, :len(traj), :] = traj
    
    # Compute mean and standard deviation ignoring NaNs
    mean_trajectory = np.nanmean(padded_trajectories, axis=0)
    std_deviation = np.nanstd(padded_trajectories, axis=0)
    
    plt.figure(figsize=(10, 6))
    # Plot individual trajectories in gray
    for traj in trajectories:
        plt.plot(traj[:, 0], traj[:, 1], color='gray', alpha=0.5, linewidth=0.5)
    
    # Plot mean trajectory in blue
    plt.plot(mean_trajectory[:, 0], mean_trajectory[:, 1], color='blue', label='Mean Trajectory', linewidth=2)
    
    # Add standard deviation shading for the x values
    plt.fill_betweenx(mean_trajectory[:, 1],
                      mean_trajectory[:, 0] - std_deviation[:, 0],
                      mean_trajectory[:, 0] + std_deviation[:, 0],
                      color='blue', alpha=0.2, label='Standard Deviation')
    
    # Mark the landing pad at (0, 0)
    plt.scatter(0, 0, color='red', marker='x', s=100, label='Landing Pad')
    
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Lunar Lander Mean Trajectory with Variability')
    plt.legend()
    plt.grid(True)
    plt.show()

