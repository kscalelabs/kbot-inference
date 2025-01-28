"""Plot reference values for the neural network."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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

# Add joint groups constant after JOINT_NAMES
JOINT_GROUPS = {
    'left_leg': ['left_hip_pitch_04', 'left_hip_roll_03', 'left_hip_yaw_03', 'left_knee_04', 'left_ankle_02'],
    'right_leg': ['right_hip_pitch_04', 'right_hip_roll_03', 'right_hip_yaw_03', 'right_knee_04', 'right_ankle_02'],
    'left_arm': ['left_shoulder_pitch_03', 'left_shoulder_roll_03', 'left_shoulder_yaw_02', 'left_elbow_02', 'left_wrist_02'],
    'right_arm': ['right_shoulder_pitch_03', 'right_shoulder_roll_03', 'right_shoulder_yaw_02', 'right_elbow_02', 'right_wrist_02'],
}

def plot_joint_group(time, data, joint_group_name, joint_names, ylabel, title, output_path):
    """Helper function to plot a group of joints."""
    plt.figure(figsize=(12, 6))
    for name in joint_names:
        idx = JOINT_NAMES.index(name)
        plt.plot(time, data[:, idx], label=name)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(f"{title} - {joint_group_name.replace('_', ' ').title()}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_nn_data(data_path: Path, output_dir: Path) -> None:
    """Plot neural network input and output values from a numpy file.

    Args:
        data_path: Path to the .npz file containing inputs and outputs
        output_dir: Directory to save the plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = np.load(data_path)
    inputs = data["inputs"].reshape(-1, 66)  # (N, 1, 66) -> (N, 66)
    outputs = data["outputs"].reshape(-1, 20)  # (N, 1, 20) -> (N, 20)

    # Create time array
    time = np.arange(len(inputs)) / 50  # Assuming 50Hz sampling rate

    # Plot settings
    fig_size = (48, 12)

    # Split inputs into their components
    vel_commands = inputs[:, 0:3]
    projected_gravity = inputs[:, 3:6]
    joint_pos = inputs[:, 6:26]
    joint_vel = inputs[:, 26:46]
    actions = inputs[:, 46:66]

    # Plot velocity commands
    plt.figure(figsize=fig_size)
    labels = ["x", "y", "yaw"]
    for i, label in enumerate(labels):
        plt.plot(time, vel_commands[:, i], label=label)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity Command")
    plt.title("Velocity Commands Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "velocity_commands.png")
    plt.close()

    # Plot projected gravity
    plt.figure(figsize=fig_size)
    labels = ["x", "y", "z"]
    for i, label in enumerate(labels):
        plt.plot(time, projected_gravity[:, i], label=label)
    plt.xlabel("Time (s)")
    plt.ylabel("Gravity Vector")
    plt.title("Projected Gravity Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "projected_gravity.png")
    plt.close()

    # Replace the joint positions plot with grouped plots
    for group_name, joint_list in JOINT_GROUPS.items():
        plot_joint_group(
            time,
            joint_pos,
            group_name,
            joint_list,
            "Joint Position (rad)",
            "Joint Positions Over Time",
            output_dir / f"joint_positions_{group_name}.png"
        )

    # Replace the joint velocities plot with grouped plots
    for group_name, joint_list in JOINT_GROUPS.items():
        plot_joint_group(
            time,
            joint_vel,
            group_name,
            joint_list,
            "Joint Velocity (rad/s)",
            "Joint Velocities Over Time",
            output_dir / f"joint_velocities_{group_name}.png"
        )

    # Replace the previous actions plot with grouped plots
    for group_name, joint_list in JOINT_GROUPS.items():
        plot_joint_group(
            time,
            actions,
            group_name,
            joint_list,
            "Action Value",
            "Previous Actions Over Time",
            output_dir / f"previous_actions_{group_name}.png"
        )

    # Replace the outputs plot with grouped plots
    for group_name, joint_list in JOINT_GROUPS.items():
        plot_joint_group(
            time,
            outputs,
            group_name,
            joint_list,
            "Output Value",
            "Neural Network Outputs Over Time",
            output_dir / f"nn_outputs_{group_name}.png"
        )

    # Replace the scaled outputs plot with grouped plots
    for group_name, joint_list in JOINT_GROUPS.items():
        plot_joint_group(
            time,
            outputs * 0.5,
            group_name,
            joint_list,
            "Scaled Output Value",
            "Scaled Neural Network Outputs Over Time",
            output_dir / f"scaled_nn_outputs_{group_name}.png"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot neural network input/output values"
    )
    parser.add_argument("data_path", type=Path, help="Path to the .npz data file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to save plots (default: ./plots)",
    )

    args = parser.parse_args()
    plot_nn_data(args.data_path, args.output_dir)


if __name__ == "__main__":
    main()
