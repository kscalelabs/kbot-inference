#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_nn_values(nn_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot neural network input values in meaningful groups."""
    # Based on the neural network input structure from nn.rs:
    # obs[0:3]   - targets (x_vel, y_vel, rot)
    # obs[3:6]   - gravity vector (gx, gy, gz)
    # obs[6:26]  - joint positions (20 values)
    # obs[26:46] - joint velocities (20 values)
    # obs[46:66] - previous actions (20 values)

    # Create subplots for each group of values
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot target values
    ax1.plot(nn_df["timestamp"], nn_df["obs_0"], label="X Velocity")
    ax1.plot(nn_df["timestamp"], nn_df["obs_1"], label="Y Velocity")
    ax1.plot(nn_df["timestamp"], nn_df["obs_2"], label="Rotation")
    ax1.set_title("Target Values")
    ax1.set_ylabel("Target")
    ax1.legend()
    ax1.grid(True)

    # Plot gravity vector
    ax2.plot(nn_df["timestamp"], nn_df["obs_3"], label="X")
    ax2.plot(nn_df["timestamp"], nn_df["obs_4"], label="Y")
    ax2.plot(nn_df["timestamp"], nn_df["obs_5"], label="Z")
    ax2.set_title("Gravity Vector")
    ax2.set_ylabel("Gravity")
    ax2.legend()
    ax2.grid(True)

    # Plot joint positions
    for i in range(20):
        ax3.plot(
            nn_df["timestamp"],
            nn_df[f"obs_{i+6}"],
            label=f"Joint {i}",
            alpha=0.7,
        )
    ax3.set_title("Joint Positions")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Position (rad)")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax3.grid(True)

    # Plot joint velocities
    for i in range(20):
        ax4.plot(
            nn_df["timestamp"],
            nn_df[f"obs_{i+26}"],
            label=f"Joint {i}",
            alpha=0.7,
        )
    ax4.set_title("Joint Velocities")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Velocity (rad/s)")
    ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "nn_values.png", bbox_inches="tight")
    plt.close()

    # Create a separate plot for previous actions due to space constraints
    plt.figure(figsize=(12, 6))
    for i in range(20):
        plt.plot(
            nn_df["timestamp"],
            nn_df[f"obs_{i+46}"],
            label=f"Joint {i}",
            alpha=0.7,
        )
    plt.title("Previous Actions")
    plt.xlabel("Time (s)")
    plt.ylabel("Action")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "nn_previous_actions.png", bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot neural network input values from K-Bot")
    parser.add_argument(
        "data_dir", type=str, help="Directory containing the nn_values.csv file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save plots (defaults to data directory)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the CSV file
    nn_file = data_dir / "nn_values.csv"

    if nn_file.exists():
        nn_df = pd.read_csv(nn_file)
        # Normalize timestamps to start at 0
        nn_df["timestamp"] = nn_df["timestamp"] - nn_df["timestamp"].iloc[0]
        plot_nn_values(nn_df, output_dir)
        print(f"Neural network plots saved to {output_dir}")
        print(f"  - Main plot: {output_dir / 'nn_values.png'}")
        print(f"  - Previous actions: {output_dir / 'nn_previous_actions.png'}")
    else:
        print(f"Error: Neural network data file not found at {nn_file}")


if __name__ == "__main__":
    main()
