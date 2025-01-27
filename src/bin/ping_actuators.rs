// This script scans for actuators on specified CAN buses and reports which IDs are found.
//
// Run with:
//   cargo run --bin ping_actuators -- --ports <PORT1> [<PORT2> ...]
//
// Example:
//   cargo run --bin ping_actuators -- --ports can0 can1
//   cargo run --bin ping_actuators -- --ports /dev/ttyUSB0

use clap::Parser;
use kbot::actuators::Actuator;
use robstride::{ActuatorConfiguration, ActuatorType};
use std::time::Duration;
use tracing_subscriber::FmtSubscriber;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Ports to scan (e.g., "can0 can1" or "/dev/ttyUSB0")
    #[arg(long, value_name = "PORTS", num_args = 1.., value_delimiter = ' ')]
    ports: Vec<String>,

    /// Timeout duration in milliseconds for scanning
    #[arg(long, value_name = "TIMEOUT_MS", default_value = "100")]
    timeout_ms: u64,
}

async fn scan_for_actuators(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    // Create a minimal configuration for scanning
    // We'll use RobStride03 as a default type since we just want to scan
    let scan_config = vec![(
        0xFD, // Broadcast ID for scanning
        ActuatorConfiguration {
            actuator_type: ActuatorType::RobStride03,
            max_angle_change: None,
            max_velocity: None,
        },
    )];

    // Convert ports to string slices
    let port_refs: Vec<&str> = args.ports.iter().map(|s| s.as_str()).collect();

    // Initialize actuator with scanning configuration
    let actuator = Actuator::new(
        port_refs,
        Duration::from_millis(args.timeout_ms),
        Duration::from_millis(20),
        &scan_config,
    )
    .await?;

    // Get the list of known K-Bot actuator configurations for reference
    let kbot_configs = Actuator::create_kbot_actuators();
    let kbot_ids: Vec<u8> = kbot_configs.iter().map(|(id, _)| *id).collect();

    // Get current state of all possible actuators
    let states = actuator.get_actuators_state((1..=0xFF).collect()).await?;
    let num_actuators = states.len();

    // Print results
    println!("\n=== Actuator Scan Results ===");

    if states.is_empty() {
        println!("❌ No actuators found on ports: {:?}", args.ports);
        return Ok(());
    }

    for state in states {
        let id = state.actuator_id as u8;
        let is_kbot = kbot_ids.contains(&id);
        let status_icon = if state.online { "✓" } else { "✗" };

        println!("\n📍 Actuator ID: {}", id);
        println!(
            "   Type: {}",
            if is_kbot {
                "Known K-Bot actuator"
            } else {
                "Unknown actuator"
            }
        );
        println!(
            "   Status: {} {}",
            status_icon,
            if state.online { "Online" } else { "Offline" }
        );

        if let Some(temp) = state.temperature {
            println!("   Temperature: {:.1}°C", temp);
        }
        if let Some(pos) = state.position {
            println!("   Position: {:.1}°", pos);
        }
    }

    println!(
        "\n✨ Found {} actuator{}",
        num_actuators,
        if num_actuators == 1 { "" } else { "s" }
    );

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
    scan_for_actuators(args).await
}
