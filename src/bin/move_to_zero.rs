// This script gradually moves all actuators on the K-Bot robot to their zero positions.
//
// Run with:
//   cargo run --bin move_to_zero [--dry-run] [--torque-enabled] [--duration <seconds>]
//
// Or using make:
//   make move-to-zero

use clap::Parser;
use kbot::{
    actuators::ActuatorCommand, initialize_hardware, initialize_logging, nn::NeuralNetworkRunner,
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
}

async fn move_to_zero(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let (_, actuators, kbot_actuator_ids) =
        initialize_hardware(args.dry_run, args.torque_enabled).await?;

    tracing::info!("Starting gradual movement to zero positions...");

    // Start commands (current positions)
    let mut start_commands = vec![];
    if let Some(actuators) = &actuators {
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

    let home_array = ndarray::Array2::zeros((1, NeuralNetworkRunner::get_action_size()));
    let home_commands = NeuralNetworkRunner::update_commands(home_array).await?;
    let target_loop_rate = 50.0;

    NeuralNetworkRunner::take_action_slowed(
        start_commands.clone(),
        home_commands.clone(),
        Duration::from_millis((args.home_slowdown_factor * 1000.0 / target_loop_rate) as u64),
        args.home_slowdown_factor as usize,
        &actuators,
    )
    .await?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    initialize_logging().await;
    let args = Args::parse();
    move_to_zero(args).await
}
