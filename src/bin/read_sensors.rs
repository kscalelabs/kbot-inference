// This script reads sensor data from the K-Bot robot's IMU and actuators at 50Hz
// and logs the values to CSV files.
//
// Run with:
//   cargo run --bin read_sensors [-- --duration <seconds>]
//
// Or using make:
//   make read-sensors

use clap::Parser;
use kbot::{actuators::Actuator, imu::IMU};
use std::{fs::File, io::Write, path::PathBuf, time::Duration};
use time::OffsetDateTime;
use tracing_subscriber::FmtSubscriber; // Using crate name from Cargo.toml

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Maximum duration to record data in seconds (optional)
    #[arg(long, value_name = "SECONDS")]
    duration: Option<u64>,

    /// Output directory (optional, defaults to timestamped directory)
    #[arg(long, value_name = "DIR")]
    output_dir: Option<PathBuf>,

    /// Sample rate in Hz (optional, defaults to 50)
    #[arg(long, value_name = "HZ", default_value = "50")]
    rate: f64,
}

async fn run_sensor_logging(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    // Create output directory and files
    let timestamp = OffsetDateTime::now_local()?.unix_timestamp();
    let output_dir = args
        .output_dir
        .unwrap_or_else(|| PathBuf::from(format!("sensor_logs_{}", timestamp)));
    std::fs::create_dir_all(&output_dir)?;

    let mut imu_file = File::create(output_dir.join("imu_values.csv"))?;
    let mut actuator_file = File::create(output_dir.join("actuator_values.csv"))?;

    // Write headers
    writeln!(
        imu_file,
        "timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,roll,pitch,yaw"
    )?;
    writeln!(
        actuator_file,
        "timestamp,actuator_id,position,velocity,torque,temperature,online"
    )?;

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

    // Set up timing
    let target_loop_interval = Duration::from_secs_f64(1.0 / args.rate);
    let mut next_loop_time = tokio::time::Instant::now();
    let start_time = next_loop_time;

    // Performance tracking
    let mut loop_count = 0;
    let mut loop_count_start = tokio::time::Instant::now();

    tracing::info!("Logging sensor data to {}", output_dir.display());
    if let Some(duration) = args.duration {
        tracing::info!("Recording will stop after {} seconds", duration);
    }

    loop {
        // Check if we've reached the duration limit
        if let Some(duration) = args.duration {
            if start_time.elapsed() >= Duration::from_secs(duration) {
                tracing::info!(
                    "Reached specified duration of {} seconds, stopping",
                    duration
                );
                break;
            }
        }

        let now = OffsetDateTime::now_local()?.unix_timestamp_nanos() as f64 / 1e9;

        // Read and log IMU values
        if let Ok(imu_values) = imu.get_values().await {
            writeln!(
                imu_file,
                "{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                now,
                imu_values.accel_x,
                imu_values.accel_y,
                imu_values.accel_z,
                imu_values.gyro_x,
                imu_values.gyro_y,
                imu_values.gyro_z,
                imu_values.roll,
                imu_values.pitch,
                imu_values.yaw
            )?;
        }

        // Read and log actuator states
        if let Ok(actuator_states) = actuators
            .get_actuators_state(kbot_actuator_ids.clone())
            .await
        {
            for state in actuator_states {
                writeln!(
                    actuator_file,
                    "{:.6},{},{},{},{},{},{}",
                    now,
                    state.actuator_id,
                    state.position.unwrap_or(f64::NAN),
                    state.velocity.unwrap_or(f64::NAN),
                    state.torque.unwrap_or(f64::NAN),
                    state.temperature.unwrap_or(f64::NAN),
                    state.online
                )?;
            }
        }

        // Track and report loop frequency
        loop_count += 1;
        if loop_count >= 100 {
            let elapsed = loop_count_start.elapsed();
            let loop_rate = loop_count as f32 / elapsed.as_secs_f32();
            tracing::info!("Loop rate: {:.1} Hz", loop_rate);

            // Flush files periodically
            imu_file.flush()?;
            actuator_file.flush()?;

            // Reset counters
            loop_count = 0;
            loop_count_start = tokio::time::Instant::now();
        }

        // Sleep until next iteration to maintain target rate
        next_loop_time += target_loop_interval;
        if let Some(sleep_duration) =
            next_loop_time.checked_duration_since(tokio::time::Instant::now())
        {
            tokio::time::sleep(sleep_duration).await;
        }
    }

    // Final flush of files before exit
    imu_file.flush()?;
    actuator_file.flush()?;
    tracing::info!("Sensor logging complete");

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure logging for status messages
    let subscriber = FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Setting default subscriber failed");

    let args = Args::parse();
    run_sensor_logging(args).await
}
