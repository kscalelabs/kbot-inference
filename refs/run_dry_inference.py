import argparse
import asyncio
import logging
import math
import numpy as np
import onnxruntime as ort
import time
from dataclasses import dataclass
from pathlib import Path
from scipy.spatial.transform import Rotation as R

@dataclass
class Actuator:
    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float
    joint_name: str

ACTUATOR_LIST = [
    Actuator(actuator_id=31, nn_id=0, kp=300.0, kd=5.0, max_torque=70.0, joint_name="left_hip_pitch_04"),
    Actuator(actuator_id=32, nn_id=1, kp=120.0, kd=5.0, max_torque=50.0, joint_name="left_hip_roll_03"),
    Actuator(actuator_id=33, nn_id=2, kp=120.0, kd=5.0, max_torque=50.0, joint_name="left_hip_yaw_03"),
    Actuator(actuator_id=34, nn_id=3, kp=300.0, kd=5.0, max_torque=70.0, joint_name="left_knee_04"),
    Actuator(actuator_id=35, nn_id=4, kp=40.0, kd=5.0, max_torque=20.0, joint_name="left_ankle_02"),
    Actuator(actuator_id=41, nn_id=5, kp=300.0, kd=5.0, max_torque=70.0, joint_name="right_hip_pitch_04"),
    Actuator(actuator_id=42, nn_id=6, kp=120.0, kd=5.0, max_torque=50.0, joint_name="right_hip_roll_03"),
    Actuator(actuator_id=43, nn_id=7, kp=120.0, kd=5.0, max_torque=50.0, joint_name="right_hip_yaw_03"),
    Actuator(actuator_id=44, nn_id=8, kp=300.0, kd=5.0, max_torque=70.0, joint_name="right_knee_04"),
    Actuator(actuator_id=45, nn_id=9, kp=40.0, kd=5.0, max_torque=20.0, joint_name="right_ankle_02"),
]

ACTUATOR_IDS = [actuator.actuator_id for actuator in ACTUATOR_LIST]

async def dry_run_walking(
    model_path: str | Path,
    default_position: list[float],
    num_seconds: float | None = 10.0,
) -> None:
    # Create log directory with timestamp
    timestamp = int(time.time())
    log_dir = Path(f"nn_logs_{timestamp}")
    log_dir.mkdir(exist_ok=True)
    
    obs_file = open(log_dir / "observations.csv", "w")
    actions_file = open(log_dir / "actions.csv", "w")

    # Write headers
    header_count = 1 + 1 + 1 + 1 + 10 + 10 + 10 + 3  # x_vel, y_vel, rot, t, dof_pos, dof_vel, prev_actions, projected_gravity
    obs_file.write(",".join(f"obs_{i}" for i in range(header_count)) + "\n")
    actions_file.write(",".join(f"action_{i}" for i in range(10)) + "\n")

    try:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        session = ort.InferenceSession(model_path)
        output_details = [{"name": x.name, "shape": x.shape, "type": x.type} for x in session.get_outputs()]

        def policy(input_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            results = session.run(None, input_data)
            return {output_details[i]["name"]: results[i] for i in range(len(output_details))}

        default = np.array(default_position)
        target_q = np.zeros(10, dtype=np.double)
        prev_actions = np.zeros(10, dtype=np.double)
        hist_obs = np.zeros(570, dtype=np.double)

        input_data = {
            "x_vel.1": np.zeros(1).astype(np.float32),
            "y_vel.1": np.zeros(1).astype(np.float32),
            "rot.1": np.zeros(1).astype(np.float32),
            "t.1": np.zeros(1).astype(np.float32),
            "dof_pos.1": np.zeros(10).astype(np.float32),
            "dof_vel.1": np.zeros(10).astype(np.float32),
            "prev_actions.1": np.zeros(10).astype(np.float32),
            "projected_gravity.1": np.zeros(3).astype(np.float32),
            "buffer.1": np.zeros(570).astype(np.float32),
        }

        x_vel_cmd = 0.5
        y_vel_cmd = 0.0
        yaw_vel_cmd = 0.0
        frequency = 50

        start_time = time.time()
        end_time = None if num_seconds is None else start_time + num_seconds

        while end_time is None or time.time() < end_time:
            loop_start_time = time.time()

            # Zero positions and velocities for dry run
            positions = np.zeros(10)
            velocities = np.zeros(10)
            gvec = np.array([0.0, 0.0, -1.0])
            gvec[0] = -gvec[0]
            gvec[1] = -gvec[1]

            cur_pos_obs = positions - default
            cur_vel_obs = velocities

            # Update input data
            input_data["x_vel.1"] = np.array([x_vel_cmd], dtype=np.float32)
            input_data["y_vel.1"] = np.array([y_vel_cmd], dtype=np.float32)
            input_data["rot.1"] = np.array([yaw_vel_cmd], dtype=np.float32)
            input_data["t.1"] = np.array([time.time() - start_time], dtype=np.float32)
            input_data["dof_pos.1"] = cur_pos_obs.astype(np.float32)
            input_data["dof_vel.1"] = cur_vel_obs.astype(np.float32)
            input_data["prev_actions.1"] = prev_actions.astype(np.float32)
            input_data["projected_gravity.1"] = gvec.astype(np.float32)
            input_data["buffer.1"] = hist_obs.astype(np.float32)

            # Log observations
            obs_values = np.concatenate([
                input_data["x_vel.1"],
                input_data["y_vel.1"],
                input_data["rot.1"],
                input_data["t.1"],
                input_data["dof_pos.1"],
                input_data["dof_vel.1"],
                input_data["prev_actions.1"],
                input_data["projected_gravity.1"],
            ])
            obs_file.write(",".join(f"{v:.20e}" for v in obs_values) + "\n")
            obs_file.flush()

            policy_output = policy(input_data)
            positions = policy_output["actions_scaled"]
            curr_actions = policy_output["actions"]
            hist_obs = policy_output["x.3"]
            prev_actions = curr_actions

            # Log actions
            actions_file.write(",".join(f"{v:.20e}" for v in positions.flatten()) + "\n")
            actions_file.flush()

            target_q = positions + default
            print(f"Time: {time.time() - start_time:.3f}, Actions: {target_q}")

            waiting_time = 1 / frequency
            loop_end_time = time.time()
            sleep_time = max(0, waiting_time - (loop_end_time - loop_start_time))
            await asyncio.sleep(sleep_time)

    finally:
        obs_file.close()
        actions_file.close()

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seconds", type=float, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    model_path = Path(__file__).parent.parent / "simple_walking.onnx"
    default_position = [0.23, 0.0, 0.0, 0.441, -0.195, -0.23, 0.0, 0.0, -0.441, 0.195]
    await dry_run_walking(model_path, default_position, args.num_seconds)

if __name__ == "__main__":
    asyncio.run(main())
