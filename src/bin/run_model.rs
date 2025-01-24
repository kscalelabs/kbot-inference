// This script runs the neural network model to control the K-Bot robot.
//
// Run with:
//   cargo run --bin run_model -- <model_path> [--dry-run]
//
// Or using make:
//   make run
//   make dry-run

use kbot::{initialize_hardware, initialize_logging, nn::NeuralNetworkRunner};
use std::time::Duration;

async fn run_model(model_path: &str, dry_run: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut nn_runner = NeuralNetworkRunner::new(model_path)?;
    tracing::info!("Model loaded");

    let (imu, actuators, kbot_actuator_ids) = initialize_hardware(dry_run).await?;

    let target_loop_rate = 50.0;
    let target_loop_interval = Duration::from_millis((1000.0 / target_loop_rate) as u64);
    let mut next_loop_time = tokio::time::Instant::now();

    let mut loop_count = 0;
    let mut loop_count_start = tokio::time::Instant::now();
    let mut total_iteration_time = Duration::ZERO;

    loop {
        let sensor_time = nn_runner
            .update_observation(&imu, &actuators, &kbot_actuator_ids)
            .await?;
        let (actions, inference_time) = nn_runner.run_inference()?;
        let (commands, action_time) = nn_runner.run_iteration(actions, &actuators).await?;
        total_iteration_time += sensor_time + inference_time + action_time;

        loop_count += 1;
        if loop_count >= 100 {
            let total_elapsed = loop_count_start.elapsed();
            let loop_rate = loop_count as f32 / total_elapsed.as_secs_f32();
            let avg_iteration_time =
                total_iteration_time.as_secs_f32() * 1000.0 / loop_count as f32;

            tracing::info!(
                "Performance: Loop rate: {:.1} Hz, Average iteration time: {:.2}ms, Commands sent: {}",
                loop_rate,
                avg_iteration_time,
                commands.len()
            );

            loop_count = 0;
            loop_count_start = tokio::time::Instant::now();
            total_iteration_time = Duration::ZERO;
        }

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
    initialize_logging().await;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args.len() > 3 {
        return Err("Usage: program <model_path> [--dry-run]".into());
    }

    let model_path = &args[1];
    let dry_run = args.get(2).map_or(false, |arg| arg == "--dry-run");

    run_model(model_path, dry_run).await?;
    Ok(())
}
