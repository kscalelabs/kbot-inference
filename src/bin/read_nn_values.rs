// This script reads neural network input values from the K-Bot robot at 50Hz
// and logs them to a CSV file.
//
// Run with:
//   cargo run --bin read_nn_values [-- --duration <seconds>]
//
// Or using make:
//   make read-nn-values

use clap::Parser;
use kbot::{initialize_hardware, nn::NeuralNetworkRunner};
use std::{fs::File, io::Write, path::PathBuf, time::Duration};
use time::OffsetDateTime;
use tracing_subscriber::FmtSubscriber;

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

    /// Path to the neural network model file
    #[arg(long, value_name = "PATH", default_value = "models/policy.onnx")]
    model_path: PathBuf,
}

async fn run_nn_logging(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    // Create output directory and file
    let timestamp = OffsetDateTime::now_local()?.unix_timestamp();
    let output_dir = args
        .output_dir
        .unwrap_or_else(|| PathBuf::from(format!("nn_logs_{}", timestamp)));
    std::fs::create_dir_all(&output_dir)?;

    let mut nn_file = File::create(output_dir.join("nn_values.csv"))?;

    // Initialize neural network
    let mut nn_runner = NeuralNetworkRunner::new(args.model_path.to_str().unwrap())?;
    let obs_size = nn_runner.get_observation_size();

    // Write header
    write!(nn_file, "timestamp")?;
    for i in 0..obs_size {
        write!(nn_file, ",obs_{}", i)?;
    }
    writeln!(nn_file)?;

    // Initialize hardware
    let (imu, actuators, kbot_actuator_ids) = initialize_hardware(false, false).await?;

    // Set up timing
    let target_loop_interval = Duration::from_secs_f64(1.0 / args.rate);
    let mut next_loop_time = tokio::time::Instant::now();
    let start_time = next_loop_time;

    // Performance tracking
    let mut loop_count = 0;
    let mut loop_count_start = tokio::time::Instant::now();

    tracing::info!(
        "Logging neural network input values to {}",
        output_dir.display()
    );
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

        // Get neural network observation
        let (obs, _) = nn_runner
            .update_observation(&imu, &actuators, &kbot_actuator_ids, 1.0)
            .await?;

        // Write timestamp and all observation values
        write!(nn_file, "{:.6}", now)?;
        for value in obs.iter() {
            write!(nn_file, ",{:.6}", value)?;
        }
        writeln!(nn_file)?;

        // Track and report loop frequency
        loop_count += 1;
        if loop_count >= 100 {
            let elapsed = loop_count_start.elapsed();
            let loop_rate = loop_count as f32 / elapsed.as_secs_f32();
            tracing::info!("Loop rate: {:.1} Hz", loop_rate);

            // Flush file periodically
            nn_file.flush()?;

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

    // Final flush of file before exit
    nn_file.flush()?;
    tracing::info!("Neural network value logging complete");

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
    run_nn_logging(args).await
}
