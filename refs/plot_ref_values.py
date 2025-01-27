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

    # Plot joint positions
    plt.figure(figsize=fig_size)
    for i, name in enumerate(JOINT_NAMES):
        plt.plot(time, joint_pos[:, i], label=name)
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Position (rad)")
    plt.title("Joint Positions Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "joint_positions.png")
    plt.close()

    # Plot joint velocities
    plt.figure(figsize=fig_size)
    for i, name in enumerate(JOINT_NAMES):
        plt.plot(time, joint_vel[:, i], label=name)
    plt.xlabel("Time (s)")
    plt.ylabel("Joint Velocity (rad/s)")
    plt.title("Joint Velocities Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "joint_velocities.png")
    plt.close()

    # Plot previous actions
    plt.figure(figsize=fig_size)
    for i, name in enumerate(JOINT_NAMES):
        plt.plot(time, actions[:, i], label=name)
    plt.xlabel("Time (s)")
    plt.ylabel("Action Value")
    plt.title("Previous Actions Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "previous_actions.png")
    plt.close()

    # Plot outputs
    plt.figure(figsize=fig_size)
    for i, name in enumerate(JOINT_NAMES):
        plt.plot(time, outputs[:, i], label=name)
    plt.xlabel("Time (s)")
    plt.ylabel("Output Value")
    plt.title("Neural Network Outputs Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "nn_outputs.png")
    plt.close()

    # Plot scaled outputs.
    scale = 0.5
    plt.figure(figsize=fig_size)
    for i, name in enumerate(JOINT_NAMES):
        plt.plot(time, outputs[:, i] * scale, label=name)
    plt.xlabel("Time (s)")
    plt.ylabel("Output Value")
    plt.title("Scaled Neural Network Outputs Over Time")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "scaled_nn_outputs.png")
    plt.close()


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
