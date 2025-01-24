use eyre::Result;
use robstride::{
    ActuatorConfiguration, ActuatorType, CH341Transport, ControlConfig, SocketCanTransport,
    Supervisor, TransportType,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

pub struct ActuatorCommand {
    pub actuator_id: u32,
    pub position: Option<f64>,
    pub velocity: Option<f64>,
    pub torque: Option<f64>,
}

pub struct ConfigureRequest {
    pub actuator_id: u32,
    pub kp: Option<f64>,
    pub kd: Option<f64>,
    pub max_torque: Option<f64>,
    pub torque_enabled: Option<bool>,
    pub zero_position: Option<bool>,
    pub new_actuator_id: Option<u32>,
}

pub struct ActionResult {
    pub actuator_id: u32,
    pub success: bool,
    pub error: Option<String>,
}

pub struct ActionResponse {
    pub success: bool,
    pub error: Option<String>,
}

pub struct ActuatorState {
    pub actuator_id: u32,
    pub position: Option<f64>,
    pub velocity: Option<f64>,
    pub torque: Option<f64>,
    pub temperature: Option<f64>,
    pub online: bool,
}

pub struct Actuator {
    supervisor: Arc<Mutex<Supervisor>>,
}

impl Actuator {
    pub async fn new(
        ports: Vec<&str>,
        actuator_timeout: Duration,
        polling_interval: Duration,
        actuators_config: &[(u8, ActuatorConfiguration)],
    ) -> Result<Self> {
        let mut supervisor = Supervisor::new(actuator_timeout)?;
        let mut found_motors = vec![false; actuators_config.len()];

        // Initialize transports for each port
        for port in &ports {
            let transport = match port {
                p if p.starts_with("/dev/tty") => {
                    let serial = CH341Transport::new(p.to_string()).await?;
                    TransportType::CH341(serial)
                }
                p if p.starts_with("can") => {
                    let can = SocketCanTransport::new(p.to_string()).await?;
                    TransportType::SocketCAN(can)
                }
                _ => return Err(eyre::eyre!("Invalid port: {}", port)),
            };

            supervisor
                .add_transport(port.to_string(), transport)
                .await?;
        }

        // Start supervisor runner
        let mut supervisor_runner = supervisor.clone_controller();
        let _supervisor_handle = tokio::spawn(async move {
            if let Err(e) = supervisor_runner.run(polling_interval).await {
                tracing::error!("Supervisor task failed: {}", e);
            }
        });

        // Scan for motors on each port
        for port in &ports {
            let discovered_ids = supervisor.scan_bus(0xFD, port, actuators_config).await?;

            println!("Discovered IDs: {:?}", discovered_ids);

            // Find unknown IDs by comparing against configured IDs
            let configured_ids: Vec<_> = actuators_config.iter().map(|(id, _)| id).collect();
            let unknown_ids: Vec<_> = discovered_ids
                .iter()
                .filter(|id| !configured_ids.contains(id))
                .collect();

            if !unknown_ids.is_empty() {
                tracing::warn!(
                    "Unknown motor IDs discovered on port {}: {:?}",
                    port,
                    unknown_ids
                );
            }

            // Mark found configured motors
            for (idx, (motor_id, _)) in actuators_config.iter().enumerate() {
                if discovered_ids.contains(motor_id) {
                    found_motors[idx] = true;
                }
            }
        }

        // Log warnings for missing motors
        for (idx, (motor_id, config)) in actuators_config.iter().enumerate() {
            if !found_motors[idx] {
                tracing::warn!(
                    "Configured motor not found - ID: {}, Type: {:?}",
                    motor_id,
                    config.actuator_type
                );
            }
        }

        Ok(Self {
            supervisor: Arc::new(Mutex::new(supervisor)),
        })
    }

    pub async fn command_actuators(
        &self,
        commands: Vec<ActuatorCommand>,
    ) -> Result<Vec<ActionResult>> {
        let mut results = vec![];
        for command in commands {
            let motor_id = command.actuator_id as u8;
            let mut supervisor = self.supervisor.lock().await;
            let result = supervisor
                .command(
                    motor_id,
                    command
                        .position
                        .map(|p| p.to_radians() as f32)
                        .unwrap_or(0.0),
                    command
                        .velocity
                        .map(|v| v.to_radians() as f32)
                        .unwrap_or(0.0),
                    command.torque.map(|t| t as f32).unwrap_or(0.0),
                )
                .await;

            results.push(ActionResult {
                actuator_id: command.actuator_id,
                success: result.is_ok(),
                error: result.err().map(|e| e.to_string()),
            });
        }
        Ok(results)
    }

    pub async fn configure_actuator(&self, config: ConfigureRequest) -> Result<ActionResponse> {
        let motor_id = config.actuator_id as u8;
        let mut supervisor = self.supervisor.lock().await;

        let control_config = ControlConfig {
            kp: config.kp.unwrap_or(0.0) as f32,
            kd: config.kd.unwrap_or(0.0) as f32,
            max_torque: Some(config.max_torque.unwrap_or(2.0) as f32),
            max_velocity: Some(5.0),
            max_current: Some(10.0),
        };

        let result = supervisor.configure(motor_id, control_config).await;

        if let Some(torque_enabled) = config.torque_enabled {
            if torque_enabled {
                supervisor.enable(motor_id).await?;
            } else {
                supervisor.disable(motor_id, true).await?;
            }
        }

        if let Some(true) = config.zero_position {
            supervisor.zero(motor_id).await?;
        }

        if let Some(new_id) = config.new_actuator_id {
            supervisor.change_id(motor_id, new_id as u8).await?;
        }

        Ok(ActionResponse {
            success: result.is_ok(),
            error: result.err().map(|e| e.to_string()),
        })
    }

    pub async fn get_actuators_state(&self, actuator_ids: Vec<u32>) -> Result<Vec<ActuatorState>> {
        let mut responses = vec![];
        let supervisor = self.supervisor.lock().await;

        for id in actuator_ids {
            if let Ok(Some((feedback, ts))) = supervisor.get_feedback(id as u8).await {
                responses.push(ActuatorState {
                    actuator_id: id,
                    online: ts.elapsed().unwrap_or(Duration::from_secs(1)) < Duration::from_secs(1),
                    position: Some(feedback.angle.to_degrees() as f64),
                    velocity: Some(feedback.velocity.to_degrees() as f64),
                    torque: Some(feedback.torque as f64),
                    temperature: Some(feedback.temperature as f64),
                });
            }
        }
        Ok(responses)
    }

    pub fn create_kbot_actuators() -> Vec<(u8, ActuatorConfiguration)> {
        vec![
            // Left Arm (11-15)
            (
                11,
                ActuatorConfiguration {
                    actuator_type: ActuatorType::RobStride03,
                    max_angle_change: Some(30.0f32.to_radians()),
                    max_velocity: Some(10.0f32.to_radians()),
                },
            ),
            (
                12,
                ActuatorConfiguration {
                    actuator_type: ActuatorType::RobStride03,
                    max_angle_change: Some(30.0f32.to_radians()),
                    max_velocity: Some(10.0f32.to_radians()),
                },
            ),
            (
                13,
                ActuatorConfiguration {
                    actuator_type: ActuatorType::RobStride02,
                    max_angle_change: Some(30.0f32.to_radians()),
                    max_velocity: Some(10.0f32.to_radians()),
                },
            ),
            (
                14,
                ActuatorConfiguration {
                    actuator_type: ActuatorType::RobStride02,
                    max_angle_change: Some(30.0f32.to_radians()),
                    max_velocity: Some(10.0f32.to_radians()),
                },
            ),
            (
                15,
                ActuatorConfiguration {
                    actuator_type: ActuatorType::RobStride02,
                    max_angle_change: Some(30.0f32.to_radians()),
                    max_velocity: Some(10.0f32.to_radians()),
                },
            ),
            // Right Arm (21-25)
            (
                21,
                ActuatorConfiguration {
                    actuator_type: ActuatorType::RobStride03,
                    max_angle_change: Some(30.0f32.to_radians()),
                    max_velocity: Some(10.0f32.to_radians()),
                },
            ),
            (
                22,
                ActuatorConfiguration {
                    actuator_type: ActuatorType::RobStride03,
                    max_angle_change: Some(30.0f32.to_radians()),
                    max_velocity: Some(10.0f32.to_radians()),
                },
            ),
            (
                23,
                ActuatorConfiguration {
                    actuator_type: ActuatorType::RobStride02,
                    max_angle_change: Some(30.0f32.to_radians()),
                    max_velocity: Some(10.0f32.to_radians()),
                },
            ),
            (
                24,
                ActuatorConfiguration {
                    actuator_type: ActuatorType::RobStride02,
                    max_angle_change: Some(30.0f32.to_radians()),
                    max_velocity: Some(10.0f32.to_radians()),
                },
            ),
            (
                25,
                ActuatorConfiguration {
                    actuator_type: ActuatorType::RobStride02,
                    max_angle_change: Some(30.0f32.to_radians()),
                    max_velocity: Some(10.0f32.to_radians()),
                },
            ),
            // Left Leg (31-35)
            (
                31,
                ActuatorConfiguration {
                    actuator_type: ActuatorType::RobStride04,
                    max_angle_change: Some(30.0f32.to_radians()),
                    max_velocity: Some(10.0f32.to_radians()),
                },
            ),
            (
                32,
                ActuatorConfiguration {
                    actuator_type: ActuatorType::RobStride03,
                    max_angle_change: Some(45.0f32.to_radians()),
                    max_velocity: Some(10.0f32.to_radians()),
                },
            ),
            (
                33,
                ActuatorConfiguration {
                    actuator_type: ActuatorType::RobStride03,
                    max_angle_change: Some(90.0f32.to_radians()),
                    max_velocity: Some(10.0f32.to_radians()),
                },
            ),
            (
                34,
                ActuatorConfiguration {
                    actuator_type: ActuatorType::RobStride04,
                    max_angle_change: Some(45.0f32.to_radians()),
                    max_velocity: Some(10.0f32.to_radians()),
                },
            ),
            (
                35,
                ActuatorConfiguration {
                    actuator_type: ActuatorType::RobStride02,
                    max_angle_change: Some(90.0f32.to_radians()),
                    max_velocity: Some(10.0f32.to_radians()),
                },
            ),
            // Right Leg (41-45)
            (
                41,
                ActuatorConfiguration {
                    actuator_type: ActuatorType::RobStride04,
                    max_angle_change: Some(30.0f32.to_radians()),
                    max_velocity: Some(10.0f32.to_radians()),
                },
            ),
            (
                42,
                ActuatorConfiguration {
                    actuator_type: ActuatorType::RobStride03,
                    max_angle_change: Some(30.0f32.to_radians()),
                    max_velocity: Some(10.0f32.to_radians()),
                },
            ),
            (
                43,
                ActuatorConfiguration {
                    actuator_type: ActuatorType::RobStride03,
                    max_angle_change: Some(90.0f32.to_radians()),
                    max_velocity: Some(10.0f32.to_radians()),
                },
            ),
            (
                44,
                ActuatorConfiguration {
                    actuator_type: ActuatorType::RobStride04,
                    max_angle_change: Some(45.0f32.to_radians()),
                    max_velocity: Some(10.0f32.to_radians()),
                },
            ),
            (
                45,
                ActuatorConfiguration {
                    actuator_type: ActuatorType::RobStride02,
                    max_angle_change: Some(90.0f32.to_radians()),
                    max_velocity: Some(10.0f32.to_radians()),
                },
            ),
        ]
    }
}
