#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Joint names in order matching the neural network inputs/outputs
JOINT_NAMES = [
    "left_hip_pitch_04",
    "left_shoulder_pitch_03",
    "right_hip_pitch_04",
    "right_shoulder_pitch_03",
    "left_hip_roll_03",
    "left_shoulder_roll_03",
    "right_hip_roll_03",
    "right_shoulder_roll_03",
    "left_hip_yaw_03",
    "left_shoulder_yaw_02",
    "right_hip_yaw_03",
    "right_shoulder_yaw_02",
    "left_knee_04",
    "left_elbow_02",
    "right_knee_04",
    "right_elbow_02",
    "left_ankle_02",
    "left_wrist_02",
    "right_ankle_02",
    "right_wrist_02",
]

# Add these constants after JOINT_NAMES
JOINT_GROUPS = {
    'left_leg': ['left_hip_pitch_04', 'left_hip_roll_03', 'left_hip_yaw_03', 'left_knee_04', 'left_ankle_02'],
    'right_leg': ['right_hip_pitch_04', 'right_hip_roll_03', 'right_hip_yaw_03', 'right_knee_04', 'right_ankle_02'],
    'left_arm': ['left_shoulder_pitch_03', 'left_shoulder_roll_03', 'left_shoulder_yaw_02', 'left_elbow_02', 'left_wrist_02'],
    'right_arm': ['right_shoulder_pitch_03', 'right_shoulder_roll_03', 'right_shoulder_yaw_02', 'right_elbow_02', 'right_wrist_02'],
}

def plot_joint_group(
    time: np.ndarray,
    data: pd.DataFrame,
    joint_group_name: str,
    joint_names: list[str],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Helper function to plot a group of joints."""
    plt.figure(figsize=(12, 6))
    for name in joint_names:
        idx = JOINT_NAMES.index(name)
        # Determine the correct column prefix based on the DataFrame columns
        if f"joint_{idx}" in data.columns:
            column = f"joint_{idx}"
        elif f"action_{idx}" in data.columns:
            column = f"action_{idx}"
        else:
            raise KeyError(f"Could not find joint or action column for {name}")
        plt.plot(time, data[column], label=name)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(f"{title} - {joint_group_name.replace('_', ' ').title()}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_nn_data(obs_df: pd.DataFrame, actions_df: pd.DataFrame, output_dir: Path, rate: float = 50.0) -> None:
    """Plot neural network input and output values from CSV files.

    Args:
        obs_df: DataFrame containing observations
        actions_df: DataFrame containing actions
        output_dir: Directory to save the plots
        rate: Sample rate in Hz (default: 50.0)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic time array based on sample rate
    time = np.arange(len(obs_df)) / rate

    # Plot settings
    fig_size = (12, 6)

    # Split observations into their components
    vel_commands = obs_df[["obs_0", "obs_1", "obs_2"]]
    # lin_acc = obs_df[[f"obs_{i}" for i in range(3, 6)]]
    # ang_vel = obs_df[[f"obs_{i}" for i in range(6, 9)]]
    projected_gravity = obs_df[[f"obs_{i}" for i in range(3, 6)]]
    joint_pos = pd.DataFrame({
        f"joint_{i}": obs_df[f"obs_{i+6}"] for i in range(20)
    })
    joint_vel = pd.DataFrame({
        f"joint_{i}": obs_df[f"obs_{i+26}"] for i in range(20)
    })
    prev_actions = pd.DataFrame({
        f"action_{i}": obs_df[f"obs_{i+46}"] for i in range(20)
    })

    # Plot velocity commands
    plt.figure(figsize=fig_size)
    labels = ["x", "y", "yaw"]
    for i, label in enumerate(labels):
        plt.plot(time, vel_commands[f"obs_{i}"], label=label)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity Command")
    plt.title("Velocity Commands Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "inputs_velocity_commands.png")
    plt.close()

    # Plot linear acceleration
    # plt.figure(figsize=fig_size)
    # labels = ["x", "y", "z"]
    # for i, label in enumerate(labels):
    #     plt.plot(time, lin_acc[f"obs_{i+3}"], label=label)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Linear Acceleration")
    # plt.title("Linear Acceleration Over Time")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(output_dir / "inputs_imu_lin_acc.png")
    # plt.close()

    # Plot angular velocity
    # plt.figure(figsize=fig_size)
    # labels = ["x", "y", "z"]
    # for i, label in enumerate(labels):
    #     plt.plot(time, ang_vel[f"obs_{i+6}"], label=label)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Angular Velocity")
    # plt.title("Angular Velocity Over Time")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(output_dir / "inputs_imu_ang_vel.png")
    # plt.close()

    # Plot projected gravity
    plt.figure(figsize=fig_size)
    labels = ["x", "y", "z"]
    for i, label in enumerate(labels):
        plt.plot(time, projected_gravity[f"obs_{i+3}"], label=label)
    plt.xlabel("Time (s)")
    plt.ylabel("Projected Gravity")
    plt.title("Projected Gravity Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "inputs_projected_gravity.png")
    plt.close()

    # Plot joint positions by group
    for group_name, joint_list in JOINT_GROUPS.items():
        plot_joint_group(
            time,
            joint_pos,
            group_name,
            joint_list,
            "Joint Position (rad)",
            "Joint Positions Over Time",
            output_dir / f"inputs_joint_positions_{group_name}.png"
        )

    # Plot joint velocities by group
    for group_name, joint_list in JOINT_GROUPS.items():
        plot_joint_group(
            time,
            joint_vel,
            group_name,
            joint_list,
            "Joint Velocity (rad/s)",
            "Joint Velocities Over Time",
            output_dir / f"inputs_joint_velocities_{group_name}.png"
        )

    # Plot previous actions by group
    for group_name, joint_list in JOINT_GROUPS.items():
        plot_joint_group(
            time,
            prev_actions,
            group_name,
            joint_list,
            "Previous Action Value",
            "Previous Actions Over Time",
            output_dir / f"inputs_previous_actions_{group_name}.png"
        )

    # Plot neural network outputs by group
    actions_time = time[:len(actions_df)]
    for group_name, joint_list in JOINT_GROUPS.items():
        plt.figure(figsize=(12, 6))
        for name in joint_list:
            idx = JOINT_NAMES.index(name)
            plt.plot(actions_time, actions_df[f"action_{idx}"], label=name)
        plt.xlabel("Time (s)")
        plt.ylabel("Action Value")
        plt.title(f"Neural Network Actions Over Time - {group_name.replace('_', ' ').title()}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / f"outputs_{group_name}.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot neural network values from CSV files"
    )
    parser.add_argument(
        "data_dir", type=Path, help="Directory containing actions.csv and observations.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots (default: same as data_dir)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=50.0,
        help="Sample rate in Hz (default: 50.0)",
    )

    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir if args.output_dir else data_dir

    # Read CSV files
    obs_file = data_dir / "observations.csv"
    actions_file = data_dir / "actions.csv"

    if not obs_file.exists():
        print(f"Error: Observations file not found at {obs_file}")
        return
    if not actions_file.exists():
        print(f"Error: Actions file not found at {actions_file}")
        return

    obs_df = pd.read_csv(obs_file)
    actions_df = pd.read_csv(actions_file)

    plot_nn_data(obs_df, actions_df, output_dir, args.rate)
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
