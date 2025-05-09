// This script gradually moves all actuators on the K-Bot robot to their zero positions.
//
// Run with:
//   cargo run --bin run_individual_sinusoid [--dry-run] [--torque-enabled] [--duration <seconds>]

use clap::Parser;
use kbot::{
    actuators::{Actuator, ActuatorCommand},
    initialize_hardware, initialize_logging,
    nn::NeuralNetworkRunner,
};
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run without hardware access
    #[arg(long)]
    dry_run: bool,

    /// Enable torque on actuators
    #[arg(long, default_value = "true")]
    torque_enabled: bool,

    /// Slowdown factor for moving to the initial home position.
    #[arg(long, value_name = "FACTOR", default_value = "50.0")]
    home_slowdown_factor: f64,

    /// Duration of the sinusoid.
    #[arg(long, value_name = "DURATION", default_value = "5.0")]
    duration: f32,

    /// Frequency of the sinusoid.
    #[arg(long, value_name = "FREQUENCY", default_value = "0.5")]
    frequency: f32,

    /// Amplitude of the sinusoid.
    #[arg(long, value_name = "AMPLITUDE", default_value = "0.25")]
    amplitude: f32,
}

async fn get_start_commands(
    actuators: &Option<Actuator>,
    kbot_actuator_ids: &Vec<u8>,
) -> Result<Vec<ActuatorCommand>, Box<dyn std::error::Error>> {
    let mut start_commands = vec![];
    if let Some(actuators) = actuators {
        let actuator_states = actuators
            .get_actuators_state(kbot_actuator_ids.clone())
            .await?;
        for state in actuator_states {
            start_commands.push(ActuatorCommand {
                actuator_id: state.actuator_id as u32,
                position: Some(state.position.unwrap_or(0.0)),
                velocity: None,
                torque: None,
            });
        }
    } else {
        for id in kbot_actuator_ids.clone() {
            start_commands.push(ActuatorCommand {
                actuator_id: id as u32,
                position: Some(0.0),
                velocity: None,
                torque: None,
            });
        }
    }
    Ok(start_commands)
}

async fn go_to_zero(
    actuators: &Option<Actuator>,
    kbot_actuator_ids: &Vec<u8>,
    home_slowdown_factor: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let start_commands = get_start_commands(actuators, kbot_actuator_ids).await?;
    let home_array = ndarray::Array2::zeros((1, NeuralNetworkRunner::get_action_size()));
    let home_commands = NeuralNetworkRunner::update_commands(home_array).await?;
    let target_loop_rate = 50.0;
    tracing::info!("Moving to zero positions...");
    NeuralNetworkRunner::take_action_slowed(
        start_commands.clone(),
        home_commands.clone(),
        Duration::from_millis((home_slowdown_factor * 1000.0 / target_loop_rate) as u64),
        home_slowdown_factor as usize,
        &actuators,
    )
    .await?;
    Ok(())
}

async fn move_to_zero(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let (_, actuators, kbot_actuator_ids) =
        initialize_hardware(args.dry_run, args.torque_enabled).await?;

    go_to_zero(&actuators, &kbot_actuator_ids, args.home_slowdown_factor).await?;

    for i in 0..kbot_actuator_ids.len() {
        tracing::info!("Moving actuator {} to sinusoid...", i);
        for t in 0..(args.duration * args.frequency) as u64 {
            let process_time = tokio::time::Instant::now();
            let start_commands = get_start_commands(&actuators, &kbot_actuator_ids).await?;
            let mut target_commands =
                ndarray::Array2::zeros((1, NeuralNetworkRunner::get_action_size()));
            target_commands[[0, i]] =
                args.amplitude * (2.0 * std::f32::consts::PI * args.frequency * t as f32).sin();
            let target_commands = NeuralNetworkRunner::update_commands(target_commands).await?;
            let target_loop_rate = 50.0;
            tracing::info!("Moving to target position...");
            NeuralNetworkRunner::take_action_slowed(
                start_commands.clone(),
                target_commands.clone(),
                Duration::from_millis(
                    (args.home_slowdown_factor * 1000.0 / target_loop_rate) as u64,
                ),
                args.home_slowdown_factor as usize,
                &actuators,
            )
            .await?;
            let process_time = process_time.elapsed();
            let desired_loop_time = Duration::from_secs_f32(1.0 / args.frequency);
            if process_time < desired_loop_time {
                let sleep_time = desired_loop_time - process_time;
                tracing::debug!("Sleeping for {:?} to maintain frequency", sleep_time);
                tokio::time::sleep(sleep_time).await;
            } else {
                tracing::warn!(
                    "Processing took longer than the desired loop time: {:?} > {:?}",
                    process_time,
                    desired_loop_time
                );
            }
        }
    }

    go_to_zero(&actuators, &kbot_actuator_ids, args.home_slowdown_factor).await?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    initialize_logging().await;
    let args = Args::parse();
    move_to_zero(args).await
}
