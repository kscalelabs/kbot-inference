use std::time::Duration;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::Error as OrtError;
use tracing_subscriber::FmtSubscriber;

mod actuators;
mod imu;

use actuators::Actuator;
use imu::IMU;

// Mapping from the neural network index to the actuator ID, with a flag
// indicating whether or not the actuator is oriented in the same direction
// on the real robot as it is in the URDF.
const ACTUATOR_ID_MAP: [(usize, u8, bool); 20] = [
    (0, 31, true),  // left_hip_pitch_04
    (1, 11, true),  // left_shoulder_pitch_03
    (2, 41, true),  // right_hip_pitch_04
    (3, 21, true),  // right_shoulder_pitch_03
    (4, 32, true),  // left_hip_roll_03
    (5, 12, true),  // left_shoulder_roll_03
    (6, 42, true),  // right_hip_roll_03
    (7, 22, true),  // right_shoulder_roll_03
    (8, 33, true),  // left_hip_yaw_03
    (9, 13, true),  // left_shoulder_yaw_02
    (10, 43, true), // right_hip_yaw_03
    (11, 23, true), // right_shoulder_yaw_02
    (12, 34, true), // left_knee_04
    (13, 14, true), // left_elbow_02
    (14, 44, true), // right_knee_04
    (15, 24, true), // right_elbow_02
    (16, 35, true), // left_ankle_02
    (17, 15, true), // left_wrist_02
    (18, 45, true), // right_ankle_02
    (19, 25, true), // right_wrist_02
];

fn load_onnx_model(model_path: &str) -> Result<Session, OrtError> {
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    Ok(model)
}

async fn get_targets() -> Result<[f32; 3], Box<dyn std::error::Error>> {
    Ok([0.0, 0.0, 0.0]) // x_vel, y_vel, rot
}

async fn get_dof_pos_and_vel(
    actuators: &Actuator,
    actuator_ids: &Vec<u8>,
) -> Result<[f32; 40], Box<dyn std::error::Error>> {
    let state = actuators.get_actuators_state(actuator_ids.to_vec()).await?;

    // Return array of 20 positions and 20 velocities
    let positions = state.iter().map(|s| s.position).collect::<Vec<_>>();
    let velocities = state.iter().map(|s| s.velocity).collect::<Vec<_>>();

    // Create array and copy positions and velocities into it
    let mut result = [0.0; 40];
    result[..20].copy_from_slice(
        &positions
            .iter()
            .map(|x| x.unwrap_or(0.0))
            .collect::<Vec<f64>>(),
    );
    result[20..].copy_from_slice(
        &velocities
            .iter()
            .map(|x| x.unwrap_or(0.0))
            .collect::<Vec<f64>>(),
    );

    Ok(result.map(|x| x as f32))
}

fn euler_angles_to_gravity(roll: f32, pitch: f32) -> Result<[f32; 3], Box<dyn std::error::Error>> {
    // Convert roll and pitch to radians
    let (roll, pitch) = (roll.to_radians(), pitch.to_radians());

    // Calculate trigonometric values
    let (sr, cr) = roll.sin_cos();
    let (sp, cp) = pitch.sin_cos();

    // Gravity components
    let gx = -sp; // X-axis component
    let gy = sr * cp; // Y-axis component
    let gz = -cr * cp; // Z-axis component

    // Gravity in IMU frame
    Ok([gx, gy, gz])
}

async fn get_imu_values(imu: &IMU) -> Result<[f32; 9], Box<dyn std::error::Error>> {
    let imu_values = imu.get_values().await?;
    let gravity = euler_angles_to_gravity(imu_values.roll as f32, imu_values.pitch as f32)?;

    // Return array of [ang_vel(3), linear_accel(3), projected_gravity(3)]
    Ok([
        imu_values.gyro_x as f32,
        imu_values.gyro_y as f32,
        imu_values.gyro_z as f32,
        imu_values.accel_x as f32,
        imu_values.accel_y as f32,
        imu_values.accel_z as f32,
        gravity[0],
        gravity[1],
        gravity[2],
    ])
}

fn take_action(actions: ndarray::Array2<f32>) -> Result<(), Box<dyn std::error::Error>> {
    // println!("{:?}", actions);
    Ok(())
}

async fn run_model(model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let model = load_onnx_model(model_path)?;
    tracing::info!("Model loaded");

    let mut obs = ndarray::Array2::<f32>::zeros((1, 372));

    // Gets the actuator and IMU readers.
    let kbot_actuators = Actuator::create_kbot_actuators();
    let kbot_actuator_ids = kbot_actuators.iter().map(|(id, _)| *id).collect::<Vec<_>>();
    let (imu, actuators) = tokio::try_join!(
        IMU::new(&["/dev/ttyUSB0", "/dev/ttyCH341USB0"], 9600),
        Actuator::new(
            vec!["can0", "can1", "can2", "can3", "can4"],
            Duration::from_millis(100), // Actuator timeout
            Duration::from_millis(20),  // Polling interval, for running at 50 Hz
            &kbot_actuators,
        )
    )?;

    let target_loop_rate = 50.0;
    let target_loop_interval = Duration::from_millis((1000.0 / target_loop_rate) as u64);
    let mut next_loop_time = tokio::time::Instant::now();

    let mut loop_count = 0;
    let mut loop_count_start = tokio::time::Instant::now();

    // Profiling stats.
    let mut total_sensor_time = Duration::ZERO;
    let mut total_inference_time = Duration::ZERO;
    let mut total_action_time = Duration::ZERO;

    loop {
        let sensor_start = tokio::time::Instant::now();
        let (targets, dof_values, imu_values) = tokio::join!(
            get_targets(),
            get_dof_pos_and_vel(&actuators, &kbot_actuator_ids),
            get_imu_values(&imu)
        );
        total_sensor_time += sensor_start.elapsed();

        // Populate the target values.
        let targets = targets?;
        obs.slice_mut(ndarray::s![0, 0..3])
            .assign(&ndarray::Array1::from_vec(targets.to_vec()));

        // Populates the IMU values.
        let imu_values = imu_values?;
        obs.slice_mut(ndarray::s![0, 3..12])
            .assign(&ndarray::Array1::from_vec(imu_values.to_vec()));

        // Populates the DOF values.
        let dof_values = dof_values?;
        obs.slice_mut(ndarray::s![0, 12..52])
            .assign(&ndarray::Array1::from_vec(dof_values.to_vec()));

        // Populates the action buffer.
        obs.slice_mut(ndarray::s![0, 52..72]);

        let inference_start = tokio::time::Instant::now();
        let outputs = model.run(ort::inputs!["obs" => obs.clone()]?)?;
        let actions = outputs[0].try_extract_tensor::<f32>()?;
        total_inference_time += inference_start.elapsed();

        let action_start = tokio::time::Instant::now();

        // Copy actions into the next step of the observation.
        let actions_array = actions.into_shape_with_order(ndarray::Ix2(1, 20))?;
        obs.slice_mut(ndarray::s![0, 49..69])
            .assign(&actions_array.slice(ndarray::s![0, ..]));

        // Finally, scale actions by 1 / 0.5 to get the actions to take.
        let output_actions = actions_array.map(|x| x * 2.0);
        take_action(output_actions)?;
        total_action_time += action_start.elapsed();

        // Keeps track of the loop frequency and prints profiling information
        loop_count += 1;
        if loop_count >= 100 {
            let total_elapsed = loop_count_start.elapsed();
            let loop_rate = loop_count as f32 / total_elapsed.as_secs_f32();

            // Calculate average times in milliseconds
            let avg_sensor_time = total_sensor_time.as_secs_f32() * 1000.0 / loop_count as f32;
            let avg_inference_time =
                total_inference_time.as_secs_f32() * 1000.0 / loop_count as f32;
            let avg_action_time = total_action_time.as_secs_f32() * 1000.0 / loop_count as f32;

            tracing::info!(
                    "Performance: Loop rate: {:.1} Hz, Sensor time: {:.2}ms, Inference time: {:.2}ms, Action time: {:.2}ms",
                    loop_rate,
                    avg_sensor_time,
                    avg_inference_time,
                    avg_action_time
                );

            // Reset counters
            loop_count = 0;
            loop_count_start = tokio::time::Instant::now();
            total_sensor_time = Duration::ZERO;
            total_inference_time = Duration::ZERO;
            total_action_time = Duration::ZERO;
        }

        // Replace the old sleep logic with this:
        next_loop_time += target_loop_interval;
        if let Some(sleep_duration) =
            next_loop_time.checked_duration_since(tokio::time::Instant::now())
        {
            tokio::time::sleep(sleep_duration).await;
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure logging.
    let subscriber = FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO) // Set the max log level
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Setting default subscriber failed");

    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        return Err("Expected one argument".into());
    }
    let model_path = &args[1];
    run_model(model_path).await?;
    Ok(())
}
