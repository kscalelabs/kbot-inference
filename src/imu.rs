use chrono::Local;
use eyre::Result;
use imu::{HiwonderOutput, HiwonderReader, ImuFrequency, ImuReader, Quaternion, Vector3};
use serde_json;
use std::env;
use std::fs;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::task::JoinHandle;
use tracing::{error, info};

#[derive(Debug, Clone)]
pub struct ImuValues {
    pub accel_x: f64,
    pub accel_y: f64,
    pub accel_z: f64,
    pub gyro_x: f64,
    pub gyro_y: f64,
    pub gyro_z: f64,
    pub roll: f64,
    pub pitch: f64,
    pub yaw: f64,
    pub quaternion_w: f64,
    pub quaternion_x: f64,
    pub quaternion_y: f64,
    pub quaternion_z: f64,
}

impl Default for ImuValues {
    fn default() -> Self {
        Self {
            accel_x: 0.0,
            accel_y: 0.0,
            accel_z: 0.0,
            gyro_x: 0.0,
            gyro_y: 0.0,
            gyro_z: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            quaternion_w: 0.0,
            quaternion_x: 0.0,
            quaternion_y: 0.0,
            quaternion_z: 0.0,
        }
    }
}

const IMU_WRITE_TIMEOUT: Duration = Duration::from_secs(4);

pub struct IMU {
    data: Arc<RwLock<ImuValues>>,
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
                Ok(mut imu) => {
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

        let mut imu_reader = configured_imu_reader.ok_or_else(|| {
            eyre::eyre!("Failed to initialize and configure IMU on any provided interface")
        })?;

        let data = Arc::new(RwLock::new(ImuValues::default()));
        let data_clone = data.clone();

        let background_task = tokio::spawn(async move {
            loop {
                match imu_reader.get_data() {
                    Ok(raw_data) => {
                        let angles = raw_data.euler.unwrap_or_default();
                        let velocities = raw_data.gyroscope.unwrap_or_default();
                        let accelerations = raw_data.accelerometer.unwrap_or_default();
                        let quaternion = raw_data.quaternion.unwrap_or_default();

                        if let Ok(mut imu_data_lock) = data_clone.write() {
                            imu_data_lock.accel_x = accelerations.x as f64;
                            imu_data_lock.accel_y = accelerations.y as f64;
                            imu_data_lock.accel_z = accelerations.z as f64;
                            imu_data_lock.gyro_x = velocities.x as f64;
                            imu_data_lock.gyro_y = velocities.y as f64;
                            imu_data_lock.gyro_z = velocities.z as f64;
                            imu_data_lock.roll = angles.x as f64;
                            imu_data_lock.pitch = angles.y as f64;
                            imu_data_lock.yaw = angles.z as f64;
                            imu_data_lock.quaternion_w = quaternion.w as f64;
                            imu_data_lock.quaternion_x = quaternion.x as f64;
                            imu_data_lock.quaternion_y = quaternion.y as f64;
                            imu_data_lock.quaternion_z = quaternion.z as f64;
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

    pub async fn get_values(&self) -> Result<ImuValues> {
        self.data
            .read()
            .map_err(|e| eyre::eyre!("Lock error: {}", e))
            .map(|v| v.clone())
    }
}
