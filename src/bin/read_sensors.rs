// This script reads sensor data from the K-Bot robot's IMU and actuators at 50Hz
// and prints the values to stdout.
//
// Run with:
//   cargo run --bin read_sensors
//
// Or using make:
//   make read-sensors

use kbot::{actuators::Actuator, imu::IMU};
use std::time::Duration;
use tracing_subscriber::FmtSubscriber; // Using crate name from Cargo.toml

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Setting default subscriber failed");

    // Initialize hardware
    let kbot_actuators = Actuator::create_kbot_actuators();
    let kbot_actuator_ids = kbot_actuators.iter().map(|(id, _)| *id).collect::<Vec<_>>();

    let (imu, actuators) = tokio::try_join!(
        IMU::new(&["/dev/ttyUSB0", "/dev/ttyCH341USB0"], 9600),
        Actuator::new(
            vec!["can0", "can1", "can2", "can3", "can4"],
            Duration::from_millis(100),
            Duration::from_millis(20),
            &kbot_actuators,
        )
    )?;

    // Set up timing for 50Hz loop
    let target_loop_rate = 50.0;
    let target_loop_interval = Duration::from_millis((1000.0 / target_loop_rate) as u64);
    let mut next_loop_time = tokio::time::Instant::now();

    // Performance tracking
    let mut loop_count = 0;
    let mut loop_count_start = tokio::time::Instant::now();

    loop {
        // Read IMU values
        if let Ok(imu_values) = imu.get_values().await {
            tracing::info!(
                "IMU - Accel (x,y,z): {:.2}, {:.2}, {:.2} | Gyro (x,y,z): {:.2}, {:.2}, {:.2} | RPY: {:.2}, {:.2}, {:.2}",
                imu_values.accel_x, imu_values.accel_y, imu_values.accel_z,
                imu_values.gyro_x, imu_values.gyro_y, imu_values.gyro_z,
                imu_values.roll, imu_values.pitch, imu_values.yaw
            );
        }

        // Read actuator states
        if let Ok(actuator_states) = actuators
            .get_actuators_state(kbot_actuator_ids.clone())
            .await
        {
            for state in actuator_states {
                if let (Some(pos), Some(vel)) = (state.position, state.velocity) {
                    tracing::info!(
                        "Actuator {} - Pos: {:.2}°, Vel: {:.2}°/s{}",
                        state.actuator_id,
                        pos,
                        vel,
                        if !state.online { " (OFFLINE)" } else { "" }
                    );
                }
            }
        }

        // Track and report loop frequency
        loop_count += 1;
        if loop_count >= 100 {
            let elapsed = loop_count_start.elapsed();
            let loop_rate = loop_count as f32 / elapsed.as_secs_f32();
            tracing::info!("Loop rate: {:.1} Hz", loop_rate);

            // Reset counters
            loop_count = 0;
            loop_count_start = tokio::time::Instant::now();
        }

        // Sleep until next iteration to maintain 50Hz
        next_loop_time += target_loop_interval;
        if let Some(sleep_duration) =
            next_loop_time.checked_duration_since(tokio::time::Instant::now())
        {
            tokio::time::sleep(sleep_duration).await;
        }
    }
}
