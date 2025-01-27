// This script zeros all actuators on the K-Bot robot.
//
// Run with:
//   cargo run --bin zero_actuators
//
// Or using make:
//   make zero-actuators

use kbot::{actuators::ConfigureRequest, initialize_hardware, initialize_logging};
use std::time::Duration;

async fn zero_actuators() -> Result<(), Box<dyn std::error::Error>> {
    // Always read the actuators, even if we're in dry run mode.
    let (_, actuators, kbot_actuator_ids) = initialize_hardware(false, false).await?;

    if let Some(actuators) = actuators {
        tracing::info!("Starting actuator zeroing process...");

        // First disable all actuators
        for actuator_id in &kbot_actuator_ids {
            let config = ConfigureRequest {
                actuator_id: *actuator_id as u32,
                kp: None,
                kd: None,
                max_torque: None,
                torque_enabled: Some(false),
                zero_position: None,
                new_actuator_id: None,
            };
            actuators.configure_actuator(config).await?;
            tracing::info!("Disabled actuator {}", actuator_id);
        }

        // Wait a moment for any motion to stop
        tracing::info!("Stopping actuators");
        tokio::time::sleep(Duration::from_secs(10)).await;

        // Then zero each actuator
        tracing::info!("Zeroing actuators");
        for actuator_id in &kbot_actuator_ids {
            let config = ConfigureRequest {
                actuator_id: *actuator_id as u32,
                kp: None,
                kd: None,
                max_torque: None,
                torque_enabled: None,
                zero_position: Some(true),
                new_actuator_id: None,
            };
            actuators.configure_actuator(config).await?;
            tracing::info!("Zeroed actuator {}", actuator_id);
        }

        // Wait another moment
        tokio::time::sleep(Duration::from_secs(1)).await;

        // Finally re-enable all actuators
        for actuator_id in &kbot_actuator_ids {
            let config = ConfigureRequest {
                actuator_id: *actuator_id as u32,
                kp: None,
                kd: None,
                max_torque: None,
                torque_enabled: Some(true),
                zero_position: None,
                new_actuator_id: None,
            };
            actuators.configure_actuator(config).await?;
            tracing::info!("Re-enabled actuator {}", actuator_id);
        }

        tracing::info!("All actuators have been zeroed successfully!");
    } else {
        tracing::info!("Dry run - would zero actuators: {:?}", kbot_actuator_ids);
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    initialize_logging().await;
    zero_actuators().await
}
