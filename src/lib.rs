pub mod actuators;
pub mod constants;
pub mod imu;
pub mod nn;

use std::time::Duration;
use tracing_subscriber::FmtSubscriber;

pub async fn initialize_logging() {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Setting default subscriber failed");
}

pub async fn initialize_hardware(
    dry_run: bool,
    torque_enabled: bool,
) -> Result<(Option<imu::IMU>, Option<actuators::Actuator>, Vec<u8>), Box<dyn std::error::Error>> {
    if !dry_run {
        let kbot_actuators = actuators::Actuator::create_kbot_actuators();
        let kbot_actuator_ids = kbot_actuators.iter().map(|(id, _)| *id).collect::<Vec<_>>();

        let (imu, actuators) = tokio::try_join!(
            imu::IMU::new(&["/dev/ttyUSB0", "/dev/ttyCH341USB0"], 230400),
            actuators::Actuator::new(
                vec!["can0", "can1", "can2", "can3", "can4"],
                Duration::from_millis(100),
                Duration::from_millis(20),
                &kbot_actuators,
            )
        )?;

        // Disable torque on all actuators
        for id in &kbot_actuator_ids {
            let row = constants::ACTUATOR_KP_KD
                .iter()
                .find(|(i, _, _, _)| *i == *id as usize);
            if let Some(row) = row {
                let kp = row.1;
                let kd = row.2;
                let max_torque = row.3;
                if let Err(e) = actuators
                    .configure_actuator(actuators::ConfigureRequest {
                        actuator_id: *id as u32,
                        kp: Some(kp as f64 / 20.0),
                        kd: Some(kd as f64 / 10.0),
                        max_torque: Some(max_torque as f64 / 10.0),
                        torque_enabled: Some(torque_enabled),
                        zero_position: None,
                        new_actuator_id: None,
                    })
                    .await
                {
                    tracing::warn!("Failed to configure torque on actuator {}: {}", id, e);
                }
            } else {
                tracing::warn!("No kp and kd found for actuator {}", id);
            }
        }

        Ok((Some(imu), Some(actuators), kbot_actuator_ids))
    } else {
        tracing::info!("Running in dry run mode - no hardware access");
        Ok((None, None, vec![]))
    }
}
