use clap::Parser;
use kbot::{initialize_hardware, initialize_logging};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run without hardware access
    #[arg(long)]
    dry_run: bool,
}

async fn get_stats(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let (imu, actuators, actuator_ids) = initialize_hardware(args.dry_run, false).await?;

    println!("\n=== Hardware Status ===\n");

    // Print IMU status
    println!("IMU Status:");
    match imu {
        Some(imu) => {
            let values = imu.get_values().await?;
            println!("  Connected: Yes");
            println!("  Roll: {:.2}°", values.roll);
            println!("  Pitch: {:.2}°", values.pitch);
            println!("  Yaw: {:.2}°", values.yaw);
            println!(
                "  Acceleration: ({:.2}, {:.2}, {:.2}) m/s²",
                values.accel_x, values.accel_y, values.accel_z
            );
            println!(
                "  Angular Velocity: ({:.2}, {:.2}, {:.2}) °/s",
                values.gyro_x, values.gyro_y, values.gyro_z
            );
        }
        None => println!("  Connected: No"),
    }

    // Print Actuator status
    println!("\nActuator Status:");
    match actuators {
        Some(actuators) => {
            println!("  Connected: Yes");
            println!("  Number of actuators: {}", actuator_ids.len());

            if !actuator_ids.is_empty() {
                let states = actuators.get_actuators_state(actuator_ids).await?;
                println!("\n  Individual Actuator States:");
                for state in states {
                    println!("    Actuator ID: {}", state.actuator_id);
                    println!("      Position: {:.2}°", state.position.unwrap_or(0.0));
                    println!("      Velocity: {:.2}°/s", state.velocity.unwrap_or(0.0));
                    println!("      Torque: {:.2}Nm", state.torque.unwrap_or(0.0));
                    println!("      Temperature: {}°C", state.temperature.unwrap_or(0.0));
                }
            }
        }
        None => println!("  Connected: No"),
    }

    println!("\n=== End Status ===\n");

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    initialize_logging().await;
    let args = Args::parse();
    get_stats(args).await
}
