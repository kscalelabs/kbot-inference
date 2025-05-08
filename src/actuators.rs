use eyre::Result;
use robstride::{
    ActuatorConfiguration, ActuatorType, CH341Transport, ControlConfig, SocketCanTransport,
    Supervisor, TransportType,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;

#[derive(Clone, Copy, Debug)]
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
            tracing::info!("Discovered IDs on {}: {:?}", port, discovered_ids);

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

    pub async fn command_actuators_slowed(
        &self,
        start_commands: Vec<ActuatorCommand>,
        end_commands: Vec<ActuatorCommand>,
        total_delay: Duration,
        num_steps: usize,
    ) -> Result<Vec<ActionResult>> {
        if total_delay.is_zero() {
            return Err(eyre::eyre!("Total delay must be greater than zero"));
        }
        if num_steps == 0 {
            return Err(eyre::eyre!("Number of steps must be greater than zero"));
        }

        let start_command_map: HashMap<u32, ActuatorCommand> = start_commands
            .into_iter()
            .map(|cmd| (cmd.actuator_id, cmd))
            .collect();
        let end_command_map: HashMap<u32, ActuatorCommand> = end_commands
            .into_iter()
            .map(|cmd| (cmd.actuator_id, cmd))
            .collect();

        // Make sure the start and end commands have the same actuator IDs
        if start_command_map
            .keys()
            .collect::<std::collections::HashSet<_>>()
            != end_command_map
                .keys()
                .collect::<std::collections::HashSet<_>>()
        {
            return Err(eyre::eyre!(
                "Start and end commands must have the same actuator IDs"
            ));
        }

        let step_delay = total_delay.div_f32(num_steps as f32);
        let mut final_results = vec![];

        let mut next_loop_time = tokio::time::Instant::now();

        for step in 0..num_steps {
            let t = step as f32 / (num_steps - 1) as f32;
            let mut interpolated_commands = vec![];

            // Interpolate commands for all actuator IDs present in either map
            for actuator_id in start_command_map
                .keys()
                .chain(end_command_map.keys())
                .copied()
                .collect::<std::collections::HashSet<_>>()
            {
                let start_cmd = start_command_map.get(&actuator_id);
                let end_cmd = end_command_map.get(&actuator_id);

                let interpolated_cmd = ActuatorCommand {
                    actuator_id,
                    position: match (
                        start_cmd.and_then(|c| c.position),
                        end_cmd.and_then(|c| c.position),
                    ) {
                        (Some(start), Some(end)) => Some(start * (1.0 - t as f64) + end * t as f64),
                        (Some(start), None) => Some(start),
                        (None, Some(end)) => Some(end),
                        (None, None) => None,
                    },
                    velocity: match (
                        start_cmd.and_then(|c| c.velocity),
                        end_cmd.and_then(|c| c.velocity),
                    ) {
                        (Some(start), Some(end)) => Some(start * (1.0 - t as f64) + end * t as f64),
                        (Some(start), None) => Some(start),
                        (None, Some(end)) => Some(end),
                        (None, None) => None,
                    },
                    torque: match (
                        start_cmd.and_then(|c| c.torque),
                        end_cmd.and_then(|c| c.torque),
                    ) {
                        (Some(start), Some(end)) => Some(start * (1.0 - t as f64) + end * t as f64),
                        (Some(start), None) => Some(start),
                        (None, Some(end)) => Some(end),
                        (None, None) => None,
                    },
                };
                interpolated_commands.push(interpolated_cmd);
            }

            let log_result = interpolated_commands
                .iter()
                .map(|c| (c.actuator_id, c.position, c.velocity, c.torque))
                .collect::<Vec<_>>();
            tracing::info!("Commands (slowed): {:?}", log_result);

            let results = self.command_actuators(interpolated_commands).await?;
            if step == num_steps - 1 {
                final_results = results;
            }

            // Sleep until the next loop time.
            next_loop_time += step_delay;
            if let Some(sleep_duration) =
                next_loop_time.checked_duration_since(tokio::time::Instant::now())
            {
                tokio::time::sleep(sleep_duration).await;
            }
        }

        Ok(final_results)
    }

    pub async fn configure_actuator(&self, config: ConfigureRequest) -> Result<ActionResponse> {
        let motor_id = config.actuator_id as u8;
        let mut supervisor = self.supervisor.lock().await;

        let control_config = ControlConfig {
            kp: config.kp.unwrap_or(0.0) as f32,
            kd: config.kd.unwrap_or(0.0) as f32,
            max_torque: config.max_torque.map(|t| t as f32),
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

    pub async fn get_actuators_state(&self, actuator_ids: Vec<u8>) -> Result<Vec<ActuatorState>> {
        let mut responses = vec![];
        let supervisor = self.supervisor.lock().await;

        for id in actuator_ids {
            if let Ok(Some((feedback, ts))) = supervisor.get_feedback(id as u8).await {
                responses.push(ActuatorState {
                    actuator_id: id as u32,
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
        // Load this from the config
        let max_angle_change = 0.01f32; // Percent
        let max_velocity = 0.01f32.to_radians();
        let command_rate_hz = 50.0f32; // Hz
        vec![

            (
                11,
                ActuatorConfiguration {
                    actuator_type: ActuatorType::RobStride03,
                    max_angle_change: Some(max_angle_change),
                    max_velocity: Some(max_velocity),
                    command_rate_hz: Some(command_rate_hz),
                },
            )
        ]
    }
}
