use crate::{
    actuators::{Actuator, ActuatorCommand},
    constants::*,
    imu::IMU,
};
use ndarray;
use ort::{session::builder::GraphOptimizationLevel, session::Session, Error as OrtError};
use std::time::Duration;

pub struct NeuralNetworkRunner {
    model: Session,
    obs: ndarray::Array2<f32>,
}

impl NeuralNetworkRunner {
    pub fn new(model_path: &str) -> Result<Self, OrtError> {
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        let obs = ndarray::Array2::<f32>::zeros((1, 66));

        // Populate the last command vector.
        let mut start_commands = vec![];
        for (_, actuator_id, _) in ACTUATOR_ID_MAP.iter() {
            start_commands.push(ActuatorCommand {
                actuator_id: *actuator_id as u32,
                position: Some(0.0),
                velocity: None,
                torque: None,
            });
        }

        Ok(Self { model, obs })
    }

    pub fn get_observation_size(&self) -> usize {
        self.obs.ncols()
    }

    pub fn get_action_size(&self) -> usize {
        ACTUATOR_ID_MAP.len()
    }

    pub async fn get_targets() -> Result<[f32; 3], Box<dyn std::error::Error>> {
        Ok([0.0, 0.0, 0.0]) // x_vel, y_vel, rot
    }

    pub async fn get_dof_pos_and_vel(
        actuators: &Option<Actuator>,
        actuator_ids: &Vec<u8>,
        slowdown_factor: f32,
    ) -> Result<[f32; 40], Box<dyn std::error::Error>> {
        if let Some(actuators) = actuators {
            let state = actuators.get_actuators_state(actuator_ids.to_vec()).await?;

            // Return array of 20 positions and 20 velocities
            let mut positions = state
                .iter()
                .map(|s| s.position)
                .map(|p| p.unwrap_or(0.0))
                .collect::<Vec<_>>();
            let mut velocities = state
                .iter()
                .map(|s| s.velocity)
                .map(|v| v.unwrap_or(0.0))
                .collect::<Vec<_>>();

            // Multiply velocities by slowdown factor
            if slowdown_factor != 1.0 {
                velocities = velocities
                    .iter()
                    .map(|v| v * slowdown_factor as f64)
                    .collect::<Vec<_>>();
            }

            // Subtract off the home position
            for (nn_idx, home_pos) in NN_HOME_POSITION.iter() {
                let pos = positions.get_mut(*nn_idx).ok_or_else(|| {
                    format!("Missing position for neural network index {}", nn_idx)
                })?;
                *pos = *pos - *home_pos as f64;
            }

            // Convert from degrees to radians.
            positions = positions
                .iter()
                .map(|p| p * std::f64::consts::PI / 180.0)
                .collect();
            velocities = velocities
                .iter()
                .map(|v| v * std::f64::consts::PI / 180.0)
                .collect();

            // Create array and copy positions and velocities into it
            let mut result = [0.0; 40];
            result[..20].copy_from_slice(&positions);
            result[20..].copy_from_slice(&velocities);

            Ok(result.map(|x| x as f32))
        } else {
            // Return zeros in dry run mode
            Ok([0.0; 40])
        }
    }

    pub fn euler_angles_to_gravity(
        roll: f32,
        pitch: f32,
    ) -> Result<[f32; 3], Box<dyn std::error::Error>> {
        // Convert roll and pitch to radians
        let (roll, pitch) = (roll.to_radians(), pitch.to_radians());

        // Calculate trigonometric values
        let (sr, cr) = roll.sin_cos();
        let (sp, cp) = pitch.sin_cos();

        // Gravity components
        let gx = -sp; // X-axis component
        let gy = sr * cp; // Y-axis component
        let gz = -cr * cp; // Z-axis component

        // Gravity in IMU frame
        Ok([gx, gy, gz])
    }

    pub async fn get_imu_values(imu: &Option<IMU>) -> Result<[f32; 3], Box<dyn std::error::Error>> {
        if let Some(imu) = imu {
            let imu_values = imu.get_values().await?;
            let gravity =
                Self::euler_angles_to_gravity(imu_values.roll as f32, imu_values.pitch as f32)?;

            // Return array of [ang_vel(3), linear_accel(3), projected_gravity(3)]
            Ok([
                // Just using the gravity vector for now.
                gravity[0], gravity[1], gravity[2],
            ])
        } else {
            // Return zeros in dry run mode
            Ok([0.0; 3])
        }
    }

    pub async fn update_commands(
        &mut self,
        actions: ndarray::Array2<f32>,
    ) -> Result<Vec<ActuatorCommand>, Box<dyn std::error::Error>> {
        // Convert from radians to degrees.
        let actions = actions * 180.0 / std::f32::consts::PI;

        // Apply scaling factor.
        let actions = actions * 0.5;

        // Add back the home position
        let mut final_actions = actions.to_owned();
        for (nn_idx, home_pos) in NN_HOME_POSITION.iter() {
            final_actions[[0, *nn_idx]] += home_pos;
        }

        // Clip to the desired actuator limits.
        for (nn_idx, lower_limit, upper_limit) in NN_JOINT_LIMITS.iter() {
            final_actions[[0, *nn_idx]] =
                final_actions[[0, *nn_idx]].clamp(*lower_limit, *upper_limit);
        }

        // Pair the neural network action ID with the actuator ID
        let commands: Vec<ActuatorCommand> = ACTUATOR_ID_MAP
            .iter()
            .map(|(nn_idx, actuator_id, is_inverted)| {
                let actuator_action = final_actions[[0, *nn_idx]];

                // Invert the actuator action to go from URDF space to actuator space if needed.
                let actuator_action = if *is_inverted {
                    -actuator_action
                } else {
                    actuator_action
                };

                ActuatorCommand {
                    actuator_id: *actuator_id as u32,
                    position: Some(actuator_action as f64),
                    velocity: None,
                    torque: None,
                }
            })
            .collect();

        Ok(commands)
    }

    pub async fn take_action(
        &mut self,
        commands: Vec<ActuatorCommand>,
        actuators: &Option<Actuator>,
    ) -> Result<Duration, Box<dyn std::error::Error>> {
        let action_start = tokio::time::Instant::now();
        if let Some(actuators) = actuators {
            actuators.command_actuators(commands.clone()).await?;
        } else {
            tracing::info!(
                "Commands (dry run): {:?}",
                commands
                    .iter()
                    .map(|c| (c.actuator_id, c.position.unwrap_or(0.0)))
                    .collect::<Vec<_>>()
            );
        }
        let action_time = action_start.elapsed();
        Ok(action_time)
    }

    pub async fn take_action_slowed(
        &mut self,
        start_commands: Vec<ActuatorCommand>,
        end_commands: Vec<ActuatorCommand>,
        total_delay: Duration,
        num_steps: usize,
        actuators: &Option<Actuator>,
    ) -> Result<Duration, Box<dyn std::error::Error>> {
        let action_start = tokio::time::Instant::now();
        if let Some(actuators) = actuators {
            actuators
                .command_actuators_slowed(start_commands, end_commands, total_delay, num_steps)
                .await?;
        } else {
            tracing::info!(
                "Commands (dry run): {:?}",
                end_commands
                    .iter()
                    .map(|c| (c.actuator_id, c.position.unwrap_or(0.0)))
                    .collect::<Vec<_>>()
            );
        }
        let action_time = action_start.elapsed();
        Ok(action_time)
    }

    pub async fn update_observation(
        &mut self,
        imu: &Option<IMU>,
        actuators: &Option<Actuator>,
        actuator_ids: &Vec<u8>,
        slowdown_factor: f32,
    ) -> Result<(ndarray::Array2<f32>, Duration), Box<dyn std::error::Error>> {
        let sensor_start = tokio::time::Instant::now();
        let (targets, dof_values, imu_values) = tokio::join!(
            Self::get_targets(),
            Self::get_dof_pos_and_vel(actuators, actuator_ids, slowdown_factor),
            Self::get_imu_values(imu)
        );
        let sensor_time = sensor_start.elapsed();

        // Update observation vector
        let targets = targets?;
        self.obs
            .slice_mut(ndarray::s![0, 0..3])
            .assign(&ndarray::Array1::from_vec(targets.to_vec()));

        let imu_values = imu_values?;
        self.obs
            .slice_mut(ndarray::s![0, 3..6])
            .assign(&ndarray::Array1::from_vec(imu_values.to_vec()));

        let dof_values = dof_values?;
        self.obs
            .slice_mut(ndarray::s![0, 6..46])
            .assign(&ndarray::Array1::from_vec(dof_values.to_vec()));

        Ok((self.obs.clone(), sensor_time))
    }

    pub fn run_inference(
        &mut self,
        obs: ndarray::Array2<f32>,
    ) -> Result<(ndarray::Array2<f32>, Duration), Box<dyn std::error::Error>> {
        let inference_start = tokio::time::Instant::now();
        let outputs = self.model.run(ort::inputs!["obs" => obs]?)?;
        let actions = outputs[0].try_extract_tensor::<f32>()?;
        let inference_time = inference_start.elapsed();

        let actions_array = actions.into_shape_with_order(ndarray::Ix2(1, 20))?;

        // Update the observation buffer with the new actions
        self.obs
            .slice_mut(ndarray::s![0, 46..66])
            .assign(&actions_array.slice(ndarray::s![0, ..]));

        Ok((actions_array.to_owned(), inference_time))
    }
}
