use eyre::Result;
use hexmove::{HexmoveImuReader, ImuReader, Quaternion, Vector3};
use std::sync::{Arc, RwLock};
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
            accel_x: 0.0, accel_y: 0.0, accel_z: 0.0,
            gyro_x: 0.0,  gyro_y: 0.0,  gyro_z: 0.0,
            roll: 0.0,    pitch: 0.0,   yaw: 0.0,
            quaternion_w: 0.0, quaternion_x: 0.0,
            quaternion_y: 0.0, quaternion_z: 0.0,
        }
    }
}

pub struct IMU {
    data: Arc<RwLock<ImuValues>>,
    _bg: JoinHandle<()>,
}

impl IMU {
    pub async fn new(interfaces: &[&str], baud_rate: u32) -> Result<Self> {
        if interfaces.is_empty() {
            return Err(eyre::eyre!("No interfaces provided"));
        }
        // Initialize IMU hardware
        let mut imu_reader = None;
        for interface in interfaces {
            info!(
                "Attempting to initialize Hexmove IMU with interface: {} at {} baud",
                interface, baud_rate
            );

            let imu_reader = HexmoveImuReader::new(interface, 1, 1)
            .map_err(|e| format!("Failed to initialize IMU reader: {}", e))?;
    
        let mut imu_reader = imu_reader
        .ok_or_else(|| eyre::eyre!("Failed to initialize IMU on any provided interface"))?;

        let data = Arc::new(RwLock::new(ImuValues::default()));
        let data_clone = data.clone();

        // Spawn background task to continuously read IMU values
        let background_task = tokio::spawn(async move {
            let mut read_errors = 0;
            loop {
                let data = imu_reader
                    .get_data()
                    .map_err(|e| format!("Failed to get IMU data: {}", e))?;
                let angles = data.euler.unwrap_or(Vector3::default());
                let velocities = data.gyroscope.unwrap_or(Vector3::default());
                let accelerations = data.accelerometer.unwrap_or(Vector3::default());
                let quaternion = data.quaternion.unwrap_or(Quaternion::default());
                
                if let Ok(mut imu_data) = data_clone.write() {  
                    imu_data.accel_x = accelerations.x as f64;
                    imu_data.accel_y = accelerations.y as f64;
                    imu_data.accel_z = accelerations.z as f64;
                    imu_data.gyro_x = velocities.x as f64;
                    imu_data.gyro_y = velocities.y as f64;
                    imu_data.gyro_z = velocities.z as f64;
                    imu_data.roll = angles.x as f64;
                    imu_data.pitch = angles.y as f64;
                    imu_data.yaw = angles.z as f64;
                    imu_data.quaternion_w = quaternion.w as f64;
                    imu_data.quaternion_x = quaternion.x as f64;
                    imu_data.quaternion_y = quaternion.y as f64;
                    imu_data.quaternion_z = quaternion.z as f64;
                }
                
                read_errors = 0;
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
        });

        Ok(Self {
                data,
                _background_task: background_task,
            })
        }
        
    }

    pub async fn get_values(&self) -> Result<ImuValues> {
        self.data
            .read()
            .map_err(|e| eyre::eyre!("Lock error: {}", e))
            .map(|v| v.clone())
    }
}