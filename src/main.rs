use ort::Error as OrtError;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

mod actuators;
use actuators::Actuator;

fn load_onnx_model(model_path: &str) -> Result<Session, OrtError> {
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    Ok(model)
}

// +---------------------------------------------------------+
// | Active Observation Terms in Group: 'policy' (shape: (69,)) |
// +-----------+---------------------------------+-----------+
// |   Index   | Name                            |   Shape   |
// +-----------+---------------------------------+-----------+
// |     0     | kscale_imu_ang_vel              |    (3,)   |
// |     1     | velocity_commands               |    (3,)   |
// |     2     | projected_gravity               |    (3,)   |
// |     3     | joint_pos                       |   (20,)   |
// |     4     | joint_vel                       |   (20,)   |
// |     5     | actions                         |   (20,)   |
// +-----------+---------------------------------+-----------+

fn set_targets(obs: &mut ndarray::Array2<f32>) -> Result<(), Box<dyn std::error::Error>> {
    obs[[0, 0]] = 1.0;  // x_vel
    obs[[0, 1]] = 0.0;  // y_vel
    obs[[0, 2]] = 0.0;  // rot
    Ok(())
}

// Joint Names and Indices:
// ------------------------------
//   0: left_hip_pitch_04
//   1: left_shoulder_pitch_03
//   2: right_hip_pitch_04
//   3: right_shoulder_pitch_03
//   4: left_hip_roll_03
//   5: left_shoulder_roll_03
//   6: right_hip_roll_03
//   7: right_shoulder_roll_03
//   8: left_hip_yaw_03
//   9: left_shoulder_yaw_02
//  10: right_hip_yaw_03
//  11: right_shoulder_yaw_02
//  12: left_knee_04
//  13: left_elbow_02
//  14: right_knee_04
//  15: right_elbow_02
//  16: left_ankle_02
//  17: left_wrist_02
//  18: right_ankle_02
//  19: right_wrist_02
// ------------------------------

fn set_dof_pos_and_vel(obs: &mut ndarray::Array2<f32>) -> Result<(), Box<dyn std::error::Error>> {
    // Set the 20 DoF positions and velocities.
    Ok(())
}

fn set_imu(obs: &mut ndarray::Array2<f32>) -> Result<(), Box<dyn std::error::Error>> {
    // Values from 3 to 6 are the IMU angular velocities.
    obs[[0, 3]] = 0.0;
    obs[[0, 4]] = 0.0;
    obs[[0, 5]] = 0.0;
    // Values from 6 to 9 are the IMU linear accelerations.
    obs[[0, 6]] = 0.0;
    obs[[0, 7]] = 0.0;
    obs[[0, 8]] = 0.0;
    // Values from 9 to 12 are the projected gravity.
    obs[[0, 9]] = 0.0;
    obs[[0, 10]] = 0.0;
    obs[[0, 11]] = 0.0;
    Ok(())
}

fn take_action(actions: ndarray::Array2<f32>) -> Result<(), Box<dyn std::error::Error>> {
    println!("{:?}", actions);
    Ok(())
}

fn run_model(model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let model = load_onnx_model(model_path).unwrap();
    println!("Model loaded");

    let mut obs = ndarray::Array2::<f32>::zeros((1, 69));

    for i in 0..50 {
        set_targets(&mut obs)?;
        set_dof_pos_and_vel(&mut obs)?;
        set_imu(&mut obs)?;

        let outputs = model.run(ort::inputs!["obs" => obs.clone()]?)?;
        let actions = outputs[0].try_extract_tensor::<f32>().unwrap();

        // Copy actions into the next step of the observation.
        let actions_array = actions
            .into_shape_with_order(ndarray::Ix2(1, 20))
            .unwrap();
        obs.slice_mut(ndarray::s![0, 49..69])
            .assign(&actions_array.slice(ndarray::s![0, ..]));

        // Finally, scale actions by 1 / 0.5 to get the actions to take.
        let output_actions = actions_array.map(|x| x * 2.0);
        take_action(output_actions)?;
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        // Add this line
        return Err("Expected one argument".into());
    }
    let model_path = &args[1];
    run_model(model_path)?;
    Ok(())
}
