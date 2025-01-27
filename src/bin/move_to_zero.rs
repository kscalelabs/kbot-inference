// This script gradually moves all actuators on the K-Bot robot to their zero positions.
//
// Run with:
//   cargo run --bin move_to_zero [--dry-run] [--torque-enabled] [--duration <seconds>]
//
// Or using make:
//   make move-to-zero

use clap::Parser;
use kbot::{actuators::ActuatorCommand, initialize_hardware, initialize_logging};
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

    /// Duration of movement in seconds
    #[arg(long, value_name = "SECONDS", default_value = "10.0")]
    duration: f64,
}

async fn move_to_zero(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    // Validate duration
    if args.duration <= 0.0 {
        return Err("Duration must be greater than 0".into());
    }

    let (_, actuators, kbot_actuator_ids) =
        initialize_hardware(args.dry_run, args.torque_enabled).await?;

    if let Some(actuators) = actuators {
        tracing::info!("Starting gradual movement to zero positions...");

        // Get current positions
        let actuator_states = actuators
            .get_actuators_state(kbot_actuator_ids.clone())
            .await?;

        // Start commands (current positions)
        let start_commands: Vec<ActuatorCommand> = actuator_states
            .iter()
            .map(|state| ActuatorCommand {
                actuator_id: state.actuator_id as u32,
                position: Some(state.position.unwrap_or(0.0)),
                velocity: None,
                torque: None,
            })
            .collect();

        // Target commands (zero positions)
        let target_commands: Vec<ActuatorCommand> = kbot_actuator_ids
            .iter()
            .map(|&id| ActuatorCommand {
                actuator_id: id as u32,
                position: Some(0.0),
                velocity: None,
                torque: None,
            })
            .collect();

        // Current commands (interpolated positions)
        let mut current_commands: Vec<ActuatorCommand> = start_commands.clone();

        let movement_duration = Duration::from_secs(args.duration as u64);
        let update_rate = 50; // 50 Hz
        let total_steps = (movement_duration.as_secs_f64() * update_rate as f64) as usize;
        let step_interval = Duration::from_millis((1000.0 / update_rate as f64) as u64);
        let mut next_update = tokio::time::Instant::now();

        tracing::info!(
            "Moving to zero over {} seconds...",
            movement_duration.as_secs()
        );

        // Gradually interpolate to zero position
        for step in 0..=total_steps {
            let progress = step as f64 / total_steps as f64;

            // Interpolate positions using array indices
            for ((current, start), target) in current_commands
                .iter_mut()
                .zip(start_commands.iter())
                .zip(target_commands.iter())
            {
                if let (Some(start_pos), Some(target_pos)) = (start.position, target.position) {
                    current.position = Some(start_pos + (target_pos - start_pos) * progress);
                }
            }

            // Take the action
            actuators
                .command_actuators(current_commands.clone())
                .await?;

            // Wait for the next update
            let now = tokio::time::Instant::now();
            if now < next_update {
                tokio::time::sleep(next_update - now).await;
            }
            next_update = now + step_interval;
            tracing::info!("Step: {} Progress: {:.2}%", step, progress * 100.0);
        }

        tracing::info!("All actuators moved to zero positions.");
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    initialize_logging().await;
    let args = Args::parse();
    move_to_zero(args).await
}
