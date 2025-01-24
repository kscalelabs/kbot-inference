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
) -> Result<(Option<imu::IMU>, Option<actuators::Actuator>, Vec<u8>), Box<dyn std::error::Error>> {
    if !dry_run {
        let kbot_actuators = actuators::Actuator::create_kbot_actuators();
        let kbot_actuator_ids = kbot_actuators.iter().map(|(id, _)| *id).collect::<Vec<_>>();

        let (imu, actuators) = tokio::try_join!(
            imu::IMU::new(&["/dev/ttyUSB0", "/dev/ttyCH341USB0"], 9600),
            actuators::Actuator::new(
                vec!["can0", "can1", "can2", "can3", "can4"],
                Duration::from_millis(100),
                Duration::from_millis(20),
                &kbot_actuators,
            )
        )?;

        Ok((Some(imu), Some(actuators), kbot_actuator_ids))
    } else {
        tracing::info!("Running in dry run mode - no hardware access");
        Ok((None, None, vec![]))
    }
}
