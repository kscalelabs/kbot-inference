import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define joint groups
JOINT_GROUPS = {
    "left_leg": [0, 1, 2, 3, 4],
    "right_leg": [5, 6, 7, 8, 9],
}

def plot_joint_group(time, data, group_name, joint_indices, base_idx, ylabel, title, output_path):
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(joint_indices):
        plt.plot(time, data[f'obs_{base_idx + i}'], label=f'Joint {idx}')
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    plt.title(f'{title} - {group_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main(data_dir: str) -> None:
    # Read the CSV files
    obs_df = pd.read_csv(f'{data_dir}/observations.csv')
    actions_df = pd.read_csv(f'{data_dir}/actions.csv')

    # Create synthetic time array based on 50Hz sampling rate
    time = np.arange(len(obs_df)) / 50.0

    # Split observations into their components based on NetworkInput structure
    vel_commands = obs_df[['obs_0', 'obs_1', 'obs_2']]  # x_vel, y_vel, rot
    timestamp = obs_df['obs_3']  # t
    dof_pos = obs_df[[f'obs_{i}' for i in range(4, 14)]]  # dof_pos (10 values)
    dof_vel = obs_df[[f'obs_{i}' for i in range(14, 24)]]  # dof_vel (10 values)
    prev_actions = obs_df[[f'obs_{i}' for i in range(24, 34)]]  # prev_actions (10 values)
    projected_gravity = obs_df[[f'obs_{i}' for i in range(34, 37)]]  # projected_gravity (3 values)

    # Plot velocity commands
    plt.figure(figsize=(15, 10))
    labels = ['x', 'y', 'yaw']
    for i, label in enumerate(labels):
        plt.plot(time, vel_commands[f'obs_{i}'], label=label)
    plt.title('Velocity Commands Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity Command')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{data_dir}/obs_velocity_commands.png')
    plt.close()

    # Plot timestamp
    plt.figure(figsize=(15, 10))
    plt.plot(time, timestamp, label='timestamp')
    plt.title('Timestamp Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Timestamp (s)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{data_dir}/obs_timestamp.png')
    plt.close()

    # Plot projected gravity
    plt.figure(figsize=(15, 10))
    labels = ['x', 'y', 'z']
    for i, label in enumerate(labels):
        plt.plot(time, projected_gravity[f'obs_{i+34}'], label=label)
    plt.title('Projected Gravity Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Gravity Vector')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{data_dir}/obs_projected_gravity.png')
    plt.close()

    # Plot joint positions, velocities, and previous actions by group
    for group_name, joint_list in JOINT_GROUPS.items():
        # Joint positions (offset by 4 for x_vel, y_vel, rot, t)
        plot_joint_group(
            time, obs_df, group_name, joint_list, 4,
            'Joint Position (rad)', 'Joint Positions Over Time',
            f'{data_dir}/obs_joint_positions_{group_name}.png'
        )

        # Joint velocities (offset by 14 for previous values + 10 positions)
        plot_joint_group(
            time, obs_df, group_name, joint_list, 14,
            'Joint Velocity (rad/s)', 'Joint Velocities Over Time',
            f'{data_dir}/obs_joint_velocities_{group_name}.png'
        )

        # Previous actions (offset by 24 for previous values + 10 velocities)
        plot_joint_group(
            time, obs_df, group_name, joint_list, 24,
            'Previous Action Value', 'Previous Actions Over Time',
            f'{data_dir}/obs_previous_actions_{group_name}.png'
        )

    # Plot neural network actions
    plt.figure(figsize=(15, 10))
    for i in range(10):
        plt.plot(time[:len(actions_df)], actions_df[f'action_{i}'], label=f'action_{i}')
    plt.title('Actions over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Action Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{data_dir}/actions_plot.png')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot neural network actions and observations")
    parser.add_argument("data_dir", type=str, help="Directory containing the nn_logs")
    args = parser.parse_args()
    main(args.data_dir)
