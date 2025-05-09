use chrono::Local;
use eyre::Result;
use imu::{HiwonderOutput, HiwonderReader, ImuData, ImuFrequency, ImuReader};
use serde_json;
use std::env;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::task::JoinHandle;
use tracing::{error, info};

const IMU_WRITE_TIMEOUT: Duration = Duration::from_secs(4);

pub struct IMU {
    data: Arc<RwLock<ImuData>>,
    _background_task: JoinHandle<()>,
}

impl IMU {
    pub async fn new(interfaces: &[&str], baud_rate: u32) -> Result<Self> {
        if interfaces.is_empty() {
            return Err(eyre::eyre!("No interfaces provided"));
        }

        let mut configured_imu_reader = None;

        for interface in interfaces {
            info!(
                "Attempting to initialize IMU with interface: {} at {} baud",
                interface, baud_rate
            );

            match HiwonderReader::new(interface, baud_rate, Duration::from_millis(100), true) {
                Ok(imu) => {
                    info!(
                        "Successfully created IMU reader for interface: {}",
                        interface
                    );
                    info!("Setting and verifying params...");

                    if let Err(e) = imu.set_output_mode(
                        HiwonderOutput::QUATERNION
                            | HiwonderOutput::ANGLE
                            | HiwonderOutput::GYRO
                            | HiwonderOutput::ACC,
                        IMU_WRITE_TIMEOUT,
                    ) {
                        error!(
                            "Failed to set output mode for {}: {}. Params might be default.",
                            interface, e
                        );
                    } else {
                        info!("Output mode set for {}", interface);
                    }

                    if let Err(e) = imu.set_frequency(ImuFrequency::Hz100, IMU_WRITE_TIMEOUT) {
                        error!(
                            "Failed to set frequency for {}: {}. Params might be default.",
                            interface, e
                        );
                    } else {
                        info!("100Hz frequency set for {}", interface);
                    }

                    if let Err(e) = imu.set_bandwidth(42, IMU_WRITE_TIMEOUT) {
                        error!(
                            "Failed to set bandwidth for {}: {}. Params might be default.",
                            interface, e
                        );
                    } else {
                        info!("Bandwidth set for {}", interface);
                    }

                    info!("Reading IMU parameters for interface {}...", interface);
                    match imu.read_all_registers(Duration::from_secs(1)) {
                        Ok(imu_parameters) => {
                            let hex_parameters: Vec<(String, Vec<String>)> = imu_parameters
                                .into_iter()
                                .map(|(name, values)| {
                                    let name_str = format!("{:?}", name);
                                    let hex_values =
                                        values.into_iter().map(|v| format!("{:#04x}", v)).collect();
                                    (name_str, hex_values)
                                })
                                .collect();

                            if let Ok(parameters_json) =
                                serde_json::to_string_pretty(&hex_parameters)
                            {
                                let now = Local::now();
                                let timestamp = now.format("%Y%m%d_%H%M%S").to_string();
                                let base_log_dir: String = env::var("KBOT_LOG_DIR")
                                    .unwrap_or_else(|_| "/tmp/kos-kbot".to_string());
                                let log_dir = format!("{}/{}", base_log_dir, timestamp);

                                if let Err(e) = std::fs::create_dir_all(&log_dir) {
                                    error!("Failed to create log directory {}: {}", log_dir, e);
                                } else {
                                    let log_path = format!(
                                        "{}/imu_parameters_{}.json",
                                        log_dir,
                                        interface.replace(['/', '\\', ':'], "_")
                                    );
                                    match std::fs::write(&log_path, parameters_json) {
                                        Ok(_) => info!(
                                            "IMU parameters for {} saved to {}",
                                            interface, log_path
                                        ),
                                        Err(e) => error!(
                                            "Failed to write IMU parameters for {} to {}: {}",
                                            interface, log_path, e
                                        ),
                                    }
                                }
                            } else {
                                error!(
                                    "Failed to serialize IMU parameters (hex) to JSON for interface {}",
                                    interface
                                );
                            }
                        }
                        Err(e) => {
                            error!("Failed to read IMU parameters for {}: {}", interface, e);
                        }
                    }
                    configured_imu_reader = Some(imu);
                    info!("Successfully configured IMU for interface: {}", interface);
                    break;
                }
                Err(e) => {
                    error!(
                        "Failed to create IMU reader for interface {}: {}",
                        interface, e
                    );
                }
            }
        }

        let imu_reader = configured_imu_reader.ok_or_else(|| {
            eyre::eyre!("Failed to initialize and configure IMU on any provided interface")
        })?;

        let data = Arc::new(RwLock::new(ImuData::default()));
        let data_clone = data.clone();

        let background_task = tokio::spawn(async move {
            loop {
                match imu_reader.get_data() {
                    Ok(raw_data) => {
                        if let Ok(mut imu_data_lock) = data_clone.write() {
                            *imu_data_lock = raw_data;
                        } else {
                            error!(
                                "IMU background task: Failed to acquire write lock for IMU data"
                            );
                        }
                    }
                    Err(e) => {
                        error!("IMU background task: Failed to get IMU data: {}", e);
                        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    }
                }
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
        });

        Ok(Self {
            data,
            _background_task: background_task,
        })
    }

    pub async fn get_values(&self) -> Result<ImuData> {
        self.data
            .read()
            .map_err(|e| eyre::eyre!("Lock error: {}", e))
            .map(|v| v.clone())
    }
}
