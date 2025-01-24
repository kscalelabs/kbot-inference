// This script runs the neural network model to control the K-Bot robot.
//
// Run with:
//   cargo run --bin run_model -- <model_path> [--dry-run] [--slowdown-factor <factor>]
//
// Or using make:
//   make run
//   make dry-run
//   make slowdown-dry-run

use clap::Parser;
use kbot::{
    actuators::ActuatorCommand, initialize_hardware, initialize_logging, nn::NeuralNetworkRunner,
};
use std::{path::PathBuf, time::Duration};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the ONNX model file
    #[arg(value_name = "MODEL_PATH")]
    model_path: PathBuf,

    /// Run without hardware access
    #[arg(long)]
    dry_run: bool,

    /// Slow down execution by this factor (e.g., 2.0 runs at half speed)
    #[arg(long, value_name = "FACTOR", default_value = "1.0")]
    slowdown_factor: f64,
}

async fn run_model(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut nn_runner = NeuralNetworkRunner::new(args.model_path.to_str().unwrap())?;
    tracing::info!("Model loaded");

    let (imu, actuators, kbot_actuator_ids) = initialize_hardware(args.dry_run).await?;

    let target_loop_rate = 50.0;
    let target_loop_interval =
        Duration::from_millis((args.slowdown_factor * 1000.0 / target_loop_rate) as u64);
    let mut next_loop_time = tokio::time::Instant::now();

    let mut loop_count = 0;
    let mut loop_count_start = tokio::time::Instant::now();
    let mut total_iteration_time = Duration::ZERO;

    // Creates starting commands.
    let mut prev_commands = vec![];
    for id in kbot_actuator_ids.clone() {
        prev_commands.push(ActuatorCommand {
            actuator_id: id as u32,
            position: Some(0.0),
            velocity: None,
            torque: None,
        });
    }

    loop {
        let sensor_time = nn_runner
            .update_observation(&imu, &actuators, &kbot_actuator_ids)
            .await?;
        total_iteration_time += sensor_time;

        let (actions, inference_time) = nn_runner.run_inference()?;
        total_iteration_time += inference_time;

        // Takes the action.
        let commands = nn_runner.update_commands(actions).await?;
        if args.slowdown_factor == 1.0 {
            let action_time = nn_runner.take_action(commands.clone(), &actuators).await?;
            total_iteration_time += action_time;
        } else {
            let action_time = nn_runner
                .take_action_slowed(
                    prev_commands.clone(),
                    commands.clone(),
                    sensor_time + inference_time,
                    args.slowdown_factor as usize,
                    &actuators,
                )
                .await?;
            total_iteration_time += action_time;
        }

        // Update prev_commands for next iteration
        prev_commands = commands;

        loop_count += 1;
        if loop_count >= 100 {
            let total_elapsed = loop_count_start.elapsed();
            let loop_rate = loop_count as f32 / total_elapsed.as_secs_f32();
            let avg_iteration_time =
                total_iteration_time.as_secs_f32() * 1000.0 / loop_count as f32;

            tracing::info!(
                "Performance: Loop rate: {:.1} Hz, Average iteration time: {:.2}ms",
                loop_rate,
                avg_iteration_time
            );

            loop_count = 0;
            loop_count_start = tokio::time::Instant::now();
            total_iteration_time = Duration::ZERO;
        }

        // Sleep until the next loop time.
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
    let args = Args::parse();
    run_model(args).await
}
