// This script gradually moves all actuators on the K-Bot robot to their home positions.
//
// Run with:
//   cargo run --bin move_to_home [--dry-run] [--torque-enabled] [--duration <seconds>]
//
// Or using make:
//   make move-to-home

use clap::Parser;
use kbot::{
    actuators::ActuatorCommand, constants::HOME_POSITION, initialize_hardware, initialize_logging,
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
}

async fn move_to_home(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let (_, actuators, kbot_actuator_ids) =
        initialize_hardware(args.dry_run, args.torque_enabled).await?;

    tracing::info!("Starting gradual movement to home positions...");

    // Start commands (current positions)
    let mut start_commands = vec![];
    if let Some(actuators_ref) = &actuators {
        let actuator_states = actuators_ref
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
        // Dry run: assume starting at 0.0 for all configured actuators
        for id in kbot_actuator_ids {
            start_commands.push(ActuatorCommand {
                actuator_id: id as u32,
                position: Some(0.0),
                velocity: None,
                torque: None,
            });
        }
    }

    // Target commands (home positions from constants)
    let mut home_commands = Vec::new();
    for (actuator_id_usize, pos_f32) in HOME_POSITION.iter() {
        home_commands.push(ActuatorCommand {
            actuator_id: *actuator_id_usize as u32, // HOME_POSITION stores ID as usize
            position: Some(*pos_f32 as f64),
            velocity: None,
            torque: None,
        });
    }

    let target_loop_rate = 50.0; // Consistent with move_to_zero

    NeuralNetworkRunner::take_action_slowed(
        start_commands,
        home_commands,
        Duration::from_millis((args.home_slowdown_factor * 1000.0 / target_loop_rate) as u64),
        args.home_slowdown_factor as usize,
        &actuators,
    )
    .await?;

    tracing::info!("Successfully moved to home positions.");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    initialize_logging().await;
    let args = Args::parse();
    move_to_home(args).await
}
