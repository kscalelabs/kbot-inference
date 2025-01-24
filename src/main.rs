use std::time::Duration;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::Error as OrtError;

mod actuators;
mod imu;

use actuators::Actuator;
use imu::IMU;

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

async fn get_targets() -> Result<[f32; 3], Box<dyn std::error::Error>> {
    Ok([1.0, 0.0, 0.0]) // x_vel, y_vel, rot
}

async fn get_dof_pos_and_vel(
    actuators: &Actuator,
) -> Result<[f32; 40], Box<dyn std::error::Error>> {
    let state = actuators.get_actuators_state(vec![0]).await?;

    println!("{:?}", state.iter().map(|s| s.position).collect::<Vec<_>>());

    // Return array of 20 positions and 20 velocities
    Ok([0.0; 40])
}

fn euler_angles_to_gravity(roll: f64, pitch: f64) -> Result<[f64; 3], Box<dyn std::error::Error>> {
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
    let gravity = euler_angles_to_gravity(imu_values.roll, imu_values.pitch)?;

    // Return array of [ang_vel(3), linear_accel(3), projected_gravity(3)]
    Ok([
        imu_values.gyro_x as f32,
        imu_values.gyro_y as f32,
        imu_values.gyro_z as f32,
        imu_values.accel_x as f32,
        imu_values.accel_y as f32,
        imu_values.accel_z as f32,
        gravity[0] as f32,
        gravity[1] as f32,
        gravity[2] as f32,
    ])
}

fn take_action(actions: ndarray::Array2<f32>) -> Result<(), Box<dyn std::error::Error>> {
    // println!("{:?}", actions);
    Ok(())
}

async fn run_model(model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let model = load_onnx_model(model_path)?;
    println!("Model loaded");

    let mut obs = ndarray::Array2::<f32>::zeros((1, 69));

    // Gets the IMU reader.
    let imu = IMU::new("/dev/ttyUSB0", 9600).await?;
    let actuators = Actuator::new(
        vec!["can0", "can1", "can2", "can3"],
        Duration::from_millis(100),
        Duration::from_millis(100),
        &Actuator::create_kbot_actuators(),
    )
    .await?;

    // for _ in 0..50 {
    loop {
        let (targets, dof_values, imu_values) = tokio::join!(
            get_targets(),
            get_dof_pos_and_vel(&actuators),
            get_imu_values(&imu)
        );
        let targets = targets?;
        let dof_values = dof_values?;
        let imu_values = imu_values?;

        // Populate observation with returned values
        obs.slice_mut(ndarray::s![0, 0..3])
            .assign(&ndarray::Array1::from_vec(targets.to_vec()));
        obs.slice_mut(ndarray::s![0, 3..43])
            .assign(&ndarray::Array1::from_vec(dof_values.to_vec()));
        obs.slice_mut(ndarray::s![0, 43..52])
            .assign(&ndarray::Array1::from_vec(imu_values.to_vec()));

        let outputs = model.run(ort::inputs!["obs" => obs.clone()]?)?;
        let actions = outputs[0].try_extract_tensor::<f32>()?;

        // Copy actions into the next step of the observation.
        let actions_array = actions.into_shape_with_order(ndarray::Ix2(1, 20))?;
        obs.slice_mut(ndarray::s![0, 49..69])
            .assign(&actions_array.slice(ndarray::s![0, ..]));

        // Finally, scale actions by 1 / 0.5 to get the actions to take.
        let output_actions = actions_array.map(|x| x * 2.0);
        take_action(output_actions)?;

        // Add a small delay to control the loop rate
        tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        return Err("Expected one argument".into());
    }
    let model_path = &args[1];
    run_model(model_path).await?;
    Ok(())
}
