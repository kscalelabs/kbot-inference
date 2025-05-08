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
use kbot::constants::HOME_POSITION;
use kbot::{
    actuators::ActuatorCommand, initialize_hardware, initialize_logging, nn::NeuralNetworkRunner,
};
use std::io::Write;
use std::{path::PathBuf, time::Duration};
use time;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the ONNX model file
    #[arg(value_name = "MODEL_PATH")]
    model_path: PathBuf,

    /// Run without hardware access
    #[arg(long)]
    dry_run: bool,

    /// Enable torque on actuators
    #[arg(long, default_value = "false")]
    torque_enabled: bool,

    /// Slow down execution by this factor (e.g., 2.0 runs at half speed)
    #[arg(long, value_name = "FACTOR", default_value = "1.0")]
    slowdown_factor: f64,

    /// Slowdown factor for moving to the initial home position.
    #[arg(long, value_name = "FACTOR", default_value = "50.0")]
    home_slowdown_factor: f64,

    /// Log neural network inputs and outputs to CSV files
    #[arg(long)]
    log_nn_io: bool,
}

async fn run_model(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut nn_runner = NeuralNetworkRunner::new(args.model_path.to_str().unwrap())?;
    tracing::info!("Model loaded");

    // Optionally create log writers
    let (mut obs_file, mut actions_file) = if args.log_nn_io {
        let timestamp = time::OffsetDateTime::now_local()?.unix_timestamp();
        let output_dir = PathBuf::from(format!("nn_logs_{}", timestamp));
        std::fs::create_dir_all(&output_dir)?;

        let obs_file = std::fs::File::create(output_dir.join("observations.csv"))?;
        let actions_file = std::fs::File::create(output_dir.join("actions.csv"))?;
        tracing::info!("Logging neural network I/O to {}", output_dir.display());
        (Some(obs_file), Some(actions_file))
    } else {
        (None, None)
    };

    // Write headers
    if let Some(file) = &mut obs_file {
        for i in 0..nn_runner.get_observation_size() {
            write!(
                file,
                "{}{}",
                if i == 0 { "" } else { "," },
                format!("obs_{}", i)
            )?;
        }
        writeln!(file)?;
    }
    if let Some(file) = &mut actions_file {
        for i in 0..NeuralNetworkRunner::get_action_size() {
            write!(
                file,
                "{}{}",
                if i == 0 { "" } else { "," },
                format!("action_{}", i)
            )?;
        }
        writeln!(file)?;
    }

    // Checks for positive slowdown factor.
    if args.slowdown_factor <= 0.0 {
        return Err("Slowdown factor must be greater than 0".into());
    }

    let target_loop_rate = 50.0;
    let target_loop_interval =
        Duration::from_millis((args.slowdown_factor * 1000.0 / target_loop_rate) as u64);
    let mut next_loop_time = tokio::time::Instant::now();

    let mut loop_count = 0;
    let mut loop_count_start = tokio::time::Instant::now();
    let mut total_iteration_time = Duration::ZERO;

    let (imu, actuators, kbot_actuator_ids) =
        initialize_hardware(args.dry_run, args.torque_enabled).await?;

    // Reads the actuator states to get the current positions.
    let mut prev_commands = vec![];
    if let Some(actuators) = &actuators {
        let actuator_states = actuators
            .get_actuators_state(kbot_actuator_ids.clone())
            .await?;
        for state in actuator_states {
            prev_commands.push(ActuatorCommand {
                actuator_id: state.actuator_id as u32,
                position: Some(state.position.unwrap_or(0.0)),
                velocity: None,
                torque: None,
            });
        }
    } else {
        for id in kbot_actuator_ids.clone() {
            prev_commands.push(ActuatorCommand {
                actuator_id: id as u32,
                position: Some(0.0),
                velocity: None,
                torque: None,
            });
        }
    }

    // First, we slowly move the actuators to the home position, then sleep for 5 seconds.
    if args.torque_enabled && !args.dry_run {
        tracing::info!("Moving to home position");
        let home_array = ndarray::Array2::from_shape_vec(
            (1, HOME_POSITION.len()),
            HOME_POSITION
                .iter()
                .map(|(_, pos)| *pos * std::f32::consts::PI / 180.0)
                .collect(),
        )?;
        let home_commands = NeuralNetworkRunner::update_commands(home_array).await?;
        NeuralNetworkRunner::take_action_slowed(
            prev_commands.clone(),
            home_commands.clone(),
            Duration::from_millis((args.home_slowdown_factor * 1000.0 / target_loop_rate) as u64),
            args.home_slowdown_factor as usize,
            &actuators,
        )
        .await?;
        prev_commands = home_commands;

        tracing::info!("Sleeping for 2 seconds");
        tokio::time::sleep(Duration::from_secs(2)).await;
    }

    tracing::info!("Starting main loop");
    loop {
        let (obs, sensor_time) = nn_runner
            .update_observation(
                &imu,
                &actuators,
                &kbot_actuator_ids,
                args.slowdown_factor as f32,
            )
            .await?;
        total_iteration_time += sensor_time;

        // Log observations if enabled
        if let Some(file) = &mut obs_file {
            for (i, &value) in obs.iter().enumerate() {
                write!(file, "{}{:.20e}", if i == 0 { "" } else { "," }, value)?;
            }
            writeln!(file)?;
        }

        let (actions, inference_time) = nn_runner.run_inference(obs)?;
        total_iteration_time += inference_time;

        // Log actions if enabled
        if let Some(file) = &mut actions_file {
            for (i, &value) in actions.iter().enumerate() {
                write!(file, "{}{:.20e}", if i == 0 { "" } else { "," }, value)?;
            }
            writeln!(file)?;
        }

        // Takes the action.
        let commands = NeuralNetworkRunner::update_commands(actions).await?;

        if args.slowdown_factor == 1.0 {
            let action_time =
                NeuralNetworkRunner::take_action(commands.clone(), &actuators).await?;
            total_iteration_time += action_time;
        } else {
            let action_time = NeuralNetworkRunner::take_action_slowed(
                prev_commands.clone(),
                commands.clone(),
                target_loop_interval,
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
