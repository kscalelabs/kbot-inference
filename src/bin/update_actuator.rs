// This script updates actuator IDs on the K-Bot robot.
//
// Run with:
//   cargo run --bin update_actuator -- --old-id <ID> --new-id <ID>
//
// Or using make:
//   make update-actuator

use clap::Parser;
use kbot::actuators::{Actuator, ConfigureRequest};
use robstride::{ActuatorConfiguration, ActuatorType};
use std::time::Duration;
use tracing_subscriber::FmtSubscriber;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Specifies the port to use for the actuator
    #[arg(long, value_name = "PORT")]
    port: String,

    /// Current ID of the actuator to update
    #[arg(long, value_name = "OLD_ID")]
    old_id: u8,

    /// New ID to assign to the actuator
    #[arg(long, value_name = "NEW_ID")]
    new_id: u8,

    /// Maximum angle change for the actuator
    #[arg(long, value_name = "MAX_ANGLE_CHANGE", default_value = "90.0")]
    max_angle_change: f32,

    /// Maximum velocity for the actuator
    #[arg(long, value_name = "MAX_VELOCITY", default_value = "100.0")]
    max_velocity: f32,
}

async fn update_actuator_id(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize hardware with minimal setup (no IMU needed)
    let actuators = Actuator::new(
        vec![&args.port],
        Duration::from_millis(100),
        Duration::from_millis(20),
        &[(
            args.old_id,
            ActuatorConfiguration {
                actuator_type: ActuatorType::RobStride03,
                max_angle_change: Some(args.max_angle_change),
                max_velocity: Some(args.max_velocity),
                command_rate_hz: Some(100.0),
            },
        )],
    )
    .await?;

    tracing::info!(
        "Attempting to update actuator ID from {} to {}",
        args.old_id,
        args.new_id
    );

    // Updates the actuator ID.
    actuators
        .configure_actuator(ConfigureRequest {
            actuator_id: args.old_id as u32,
            kp: None,
            kd: None,
            max_torque: None,
            torque_enabled: None,
            zero_position: None,
            new_actuator_id: Some(args.new_id as u32),
        })
        .await?;

    tracing::info!("Successfully updated actuator ID");
    tracing::info!("Please update your robot configuration file with the new ID");

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Setting default subscriber failed");

    let args = Args::parse();
    update_actuator_id(args).await
}
