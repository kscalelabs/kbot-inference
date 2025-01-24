use eyre::Result;
use hiwonder::{ImuFrequency, IMU as HiwonderIMU};
use std::sync::{Arc, Mutex, MutexGuard, RwLock};
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
    pub quaternion_x: f64,
    pub quaternion_y: f64,
    pub quaternion_z: f64,
    pub quaternion_w: f64,
}

impl Default for ImuValues {
    fn default() -> Self {
        ImuValues {
            accel_x: 0.0,
            accel_y: 0.0,
            accel_z: 0.0,
            gyro_x: 0.0,
            gyro_y: 0.0,
            gyro_z: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            quaternion_x: 0.0,
            quaternion_y: 0.0,
            quaternion_z: 0.0,
            quaternion_w: 0.0,
        }
    }
}

pub struct IMU {
    imu: Arc<Mutex<HiwonderIMU>>,
    data: Arc<RwLock<ImuValues>>,
}

impl IMU {
    pub async fn new(interface: &str, baud_rate: u32) -> Result<Self> {
        info!(
            "Initializing KBotIMU with interface: {} at {} baud",
            interface, baud_rate
        );

        let mut imu = match HiwonderIMU::new(interface, baud_rate) {
            Ok(imu) => {
                info!("Successfully created IMU reader");
                imu
            }
            Err(e) => {
                error!("Failed to create IMU reader: {}", e);
                return Err(eyre::eyre!("Failed to create IMU reader: {}", e));
            }
        };

        // Set the frequency to 100 Hz.
        imu.set_frequency(ImuFrequency::Hz100)?;

        Ok(IMU {
            imu: Arc::new(Mutex::new(imu)),
            data: Arc::new(RwLock::new(ImuValues::default())),
        })
    }

    async fn get_imu(&self) -> Result<MutexGuard<HiwonderIMU>> {
        let imu = self
            .imu
            .lock()
            .map_err(|e| eyre::eyre!("Failed to lock IMU: {}", e))?;
        Ok(imu)
    }

    pub async fn get_values(&self) -> Result<ImuValues> {
        match self.get_imu().await?.read_data() {
            Ok(Some((acc, gyro, angle, quat))) => {
                if let Ok(mut imu_data) = self.data.write() {
                    imu_data.accel_x = acc[0] as f64;
                    imu_data.accel_y = acc[1] as f64;
                    imu_data.accel_z = acc[2] as f64;
                    imu_data.gyro_x = gyro[0] as f64;
                    imu_data.gyro_y = gyro[1] as f64;
                    imu_data.gyro_z = gyro[2] as f64;
                    imu_data.roll = angle[0] as f64;
                    imu_data.pitch = angle[1] as f64;
                    imu_data.yaw = angle[2] as f64;
                    imu_data.quaternion_w = quat[0] as f64;
                    imu_data.quaternion_x = quat[1] as f64;
                    imu_data.quaternion_y = quat[2] as f64;
                    imu_data.quaternion_z = quat[3] as f64;
                    Ok(imu_data.clone())
                } else {
                    Err(eyre::eyre!("Failed to acquire write lock on IMU data"))
                }
            }
            Ok(None) => Ok(self
                .data
                .read()
                .map_err(|e| eyre::eyre!("Failed to read IMU data: {}", e))?
                .clone()),
            Err(e) => {
                error!("Error reading from IMU: {}", e);
                Err(eyre::eyre!("Error reading from IMU: {}", e))
            }
        }
    }
}
