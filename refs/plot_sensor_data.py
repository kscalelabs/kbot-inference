#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_imu_data(imu_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot IMU data in three subplots: accelerometer, gyroscope, and orientation."""
    _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

    # Plot accelerometer data
    ax1.plot(imu_df["timestamp"], imu_df["accel_x"], label="X")
    ax1.plot(imu_df["timestamp"], imu_df["accel_y"], label="Y")
    ax1.plot(imu_df["timestamp"], imu_df["accel_z"], label="Z")
    ax1.set_title("Accelerometer Data")
    ax1.set_ylabel("Acceleration (m/s²)")
    ax1.legend()
    ax1.grid(True)

    # Plot gyroscope data
    ax2.plot(imu_df["timestamp"], imu_df["gyro_x"], label="X")
    ax2.plot(imu_df["timestamp"], imu_df["gyro_y"], label="Y")
    ax2.plot(imu_df["timestamp"], imu_df["gyro_z"], label="Z")
    ax2.set_title("Gyroscope Data")
    ax2.set_ylabel("Angular Velocity (rad/s)")
    ax2.legend()
    ax2.grid(True)

    # Plot orientation data
    ax3.plot(imu_df["timestamp"], imu_df["roll"], label="Roll")
    ax3.plot(imu_df["timestamp"], imu_df["pitch"], label="Pitch")
    ax3.plot(imu_df["timestamp"], imu_df["yaw"], label="Yaw")
    ax3.set_title("Orientation")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Angle (degrees)")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "imu_data.png")
    plt.close()


def plot_actuator_data(actuator_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot actuator data grouped by actuator ID."""
    # Get unique actuator IDs
    actuator_ids = sorted(actuator_df["actuator_id"].unique())

    # Create subplots for position, velocity, torque, and temperature
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot position data
    for aid in actuator_ids:
        mask = actuator_df["actuator_id"] == aid
        ax1.plot(
            actuator_df[mask]["timestamp"],
            actuator_df[mask]["position"],
            label=f"ID {aid}",
        )
    ax1.set_title("Joint Positions")
    ax1.set_ylabel("Position (degrees)")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True)

    # Plot velocity data
    for aid in actuator_ids:
        mask = actuator_df["actuator_id"] == aid
        ax2.plot(
            actuator_df[mask]["timestamp"],
            actuator_df[mask]["velocity"],
            label=f"ID {aid}",
        )
    ax2.set_title("Joint Velocities")
    ax2.set_ylabel("Velocity (degrees/s)")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True)

    # Plot torque data
    for aid in actuator_ids:
        mask = actuator_df["actuator_id"] == aid
        ax3.plot(
            actuator_df[mask]["timestamp"],
            actuator_df[mask]["torque"],
            label=f"ID {aid}",
        )
    ax3.set_title("Joint Torques")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Torque (Nm)")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax3.grid(True)

    # Plot temperature data
    for aid in actuator_ids:
        mask = actuator_df["actuator_id"] == aid
        ax4.plot(
            actuator_df[mask]["timestamp"],
            actuator_df[mask]["temperature"],
            label=f"ID {aid}",
        )
    ax4.set_title("Joint Temperatures")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Temperature (°C)")
    ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "actuator_data.png", bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sensor data from K-Bot")
    parser.add_argument(
        "data_dir", type=str, help="Directory containing the sensor data CSV files"
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

    # Read the CSV files
    imu_file = data_dir / "imu_values.csv"
    actuator_file = data_dir / "actuator_values.csv"

    if imu_file.exists():
        imu_df = pd.read_csv(imu_file)
        # Normalize timestamps to start at 0
        imu_df["timestamp"] = imu_df["timestamp"] - imu_df["timestamp"].iloc[0]
        plot_imu_data(imu_df, output_dir)
        print(f"IMU plot saved to {output_dir / 'imu_data.png'}")
    else:
        print(f"Warning: IMU data file not found at {imu_file}")

    if actuator_file.exists():
        actuator_df = pd.read_csv(actuator_file)
        # Normalize timestamps to start at 0
        actuator_df["timestamp"] = (
            actuator_df["timestamp"] - actuator_df["timestamp"].iloc[0]
        )
        plot_actuator_data(actuator_df, output_dir)
        print(f"Actuator plot saved to {output_dir / 'actuator_data.png'}")
    else:
        print(f"Warning: Actuator data file not found at {actuator_file}")


if __name__ == "__main__":
    main()
