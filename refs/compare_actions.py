import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys

def plot_comparison(rust_log_dir: Path, python_log_dir: Path, time_window: float = 2.0):
    # Load data with headers
    rust_actions = pd.read_csv(rust_log_dir / "actions.csv")
    python_actions = pd.read_csv(python_log_dir / "actions.csv")
    
    # Calculate timestamps (50Hz)
    rust_times = np.arange(len(rust_actions)) / 50.0
    python_times = np.arange(len(python_actions)) / 50.0
    
    # Create subplots for each joint
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    fig.suptitle('Joint Position Comparison (Rust vs Python)')
    
    joint_names = [
        "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle",
        "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle"
    ]
    
    # Plot each joint
    for idx, (joint, ax) in enumerate(zip(joint_names, axes.flatten())):
        ax.plot(rust_times, rust_actions[f'action_{idx}'], 'b-', label='Rust', alpha=0.7)
        ax.plot(python_times, python_actions[f'action_{idx}'], 'r--', label='Python', alpha=0.7)
        ax.set_title(f'{joint}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (rad)')
        ax.grid(True)
        ax.legend()
        
        # Set x-axis limit to time window
        ax.set_xlim(0, time_window)
    
    plt.tight_layout()
    
    # Save plot
    comparison_dir = Path("comparisons")
    comparison_dir.mkdir(exist_ok=True)
    plt.savefig(comparison_dir / "joint_comparison.png")
    plt.close()

    # Calculate and print max differences
    print("\nMaximum absolute differences between Rust and Python:")
    for idx, joint in enumerate(joint_names):
        max_diff = np.max(np.abs(
            rust_actions[f'action_{idx}'] - python_actions[f'action_{idx}']
        ))
        print(f"{joint}: {max_diff:.6f} rad")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_outputs.py <rust_log_dir> <python_log_dir>")
        sys.exit(1)
        
    rust_log_dir = Path(sys.argv[1])
    python_log_dir = Path(sys.argv[2])
    plot_comparison(rust_log_dir, python_log_dir)
