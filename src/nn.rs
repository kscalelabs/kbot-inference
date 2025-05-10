use crate::{
    actuators::{Actuator, ActuatorCommand},
    constants::*,
    imu::IMU,
};
use eyre::eyre;
use imu::{Quaternion, Vector3};
use ndarray;
use ort::{session::builder::GraphOptimizationLevel, session::Session, Error as OrtError};
use std::time::Duration;

pub struct NeuralNetworkRunner {
    model: Session,
    obs: ndarray::Array1<f32>,
    carry: ndarray::Array2<f32>,
}

impl NeuralNetworkRunner {
    pub fn new(model_path: &str) -> Result<Self, OrtError> {
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        let obs = ndarray::Array1::<f32>::zeros(49);
        let carry = ndarray::Array2::<f32>::zeros((5, 128));

        // Populate the last command vector.
        let mut start_commands = vec![];
        for (actuator_id, _) in ACTUATOR_ID_MAP.iter() {
            start_commands.push(ActuatorCommand {
                actuator_id: *actuator_id as u32,
                position: Some(0.0),
                velocity: None,
                torque: None,
            });
        }

        Ok(Self { model, obs, carry })
    }

    pub fn get_observation_size(&self) -> usize {
        self.obs.len()
    }

    pub fn get_action_size() -> usize {
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

            // Create vectors initialized with zeros
            let mut positions = vec![0.0; ACTUATOR_ID_MAP.len()];
            let mut velocities = vec![0.0; ACTUATOR_ID_MAP.len()];

            // Map actuator values to their neural network indices
            for actuator_id in actuator_ids.iter() {
                let state = state.iter().find(|s| s.actuator_id == *actuator_id as u32);
                let nn_index = ACTUATOR_ID_MAP
                    .iter()
                    .find(|(id, _)| *id == *actuator_id)
                    .map(|(_, nn_idx)| *nn_idx);

                if let Some(nn_index) = nn_index {
                    if let Some(state) = state {
                        positions[nn_index] = state.position.unwrap_or(0.0);
                        velocities[nn_index] = state.velocity.unwrap_or(0.0);
                    } else {
                        return Err(eyre!(
                            "Actuator ID {} not found in state response",
                            *actuator_id
                        )
                        .into());
                    }
                } else {
                    return Err(
                        eyre!("Actuator ID {} not found in ACTUATOR_ID_MAP", *actuator_id).into(),
                    );
                }
            }

            // Multiply velocities by slowdown factor
            if slowdown_factor != 1.0 {
                velocities = velocities
                    .iter()
                    .map(|v| v * slowdown_factor as f64)
                    .collect::<Vec<_>>();
            }

            // Convert from degrees to radians.
            positions = positions
                .iter()
                .map(|p| p * std::f64::consts::PI / 180.0)
                .collect();

            velocities = velocities
                .iter()
                .map(|v| v * std::f64::consts::PI / 180.0)
                .collect::<Vec<_>>();

            // Create array and copy positions into it
            let mut result = [0.0; 40];
            result[..20].copy_from_slice(&positions);
            result[20..40].copy_from_slice(&velocities);

            Ok(result.map(|x| x as f32))
        } else {
            // Return zeros in dry run mode
            Ok([0.0; 40])
        }
    }

    pub async fn get_imu_values(imu: &Option<IMU>) -> Result<[f32; 9], Box<dyn std::error::Error>> {
        if let Some(imu) = imu {
            let imu_values = imu.get_values().await?;

            let acc = match imu_values.accelerometer {
                Some(acc) => acc,
                None => return Err(eyre!("No accelerometer values").into()),
            };

            let gyro = match imu_values.gyroscope {
                Some(gyro) => gyro,
                None => return Err(eyre!("No gyroscope values").into()),
            };

            let quat = match imu_values.quaternion {
                Some(quat) => quat,
                None => return Err(eyre!("No quaternion values").into()),
            };

            let projected_gravity = quat.rotate_vector(Vector3::new(0.0, 0.0, -9.81), true);

            Ok([
                projected_gravity.x,
                projected_gravity.y,
                projected_gravity.z,
                acc.x,
                acc.y,
                acc.z,
                gyro.x,
                gyro.y,
                gyro.z,
            ])
        } else {
            // Return zeros in dry run mode
            Ok([0.0; 9])
        }
    }

    pub async fn update_commands(
        actions: ndarray::Array2<f32>,
    ) -> Result<Vec<ActuatorCommand>, Box<dyn std::error::Error>> {
        // Convert from radians to degrees.
        let actions = actions * 180.0 / std::f32::consts::PI;

        let mut final_actions = actions.to_owned();

        // Clip to the desired actuator limits.
        for (nn_idx, lower_limit, upper_limit) in NN_JOINT_LIMITS_DEGREES.iter() {
            final_actions[[0, *nn_idx]] =
                final_actions[[0, *nn_idx]].clamp(*lower_limit, *upper_limit);
        }

        // Pair the neural network action ID with the actuator ID
        let commands: Vec<ActuatorCommand> = ACTUATOR_ID_MAP
            .iter()
            .map(|(actuator_id, nn_idx)| {
                let actuator_action = final_actions[[0, *nn_idx]];

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
    ) -> Result<(ndarray::Array1<f32>, Duration), Box<dyn std::error::Error>> {
        let sensor_start = tokio::time::Instant::now();
        let (targets, imu_values, dof_values) = tokio::join!(
            Self::get_targets(),
            Self::get_imu_values(imu),
            Self::get_dof_pos_and_vel(actuators, actuator_ids, slowdown_factor),
        );
        let sensor_time = sensor_start.elapsed();

        // Update observation vector
        let dof_values = dof_values?;
        self.obs
            .slice_mut(ndarray::s![0..40])
            .assign(&ndarray::Array1::from_vec(dof_values.to_vec()));

        let imu_values = imu_values?;
        self.obs
            .slice_mut(ndarray::s![40..49])
            .assign(&ndarray::Array1::from_vec(imu_values.to_vec()));

        Ok((self.obs.clone(), sensor_time))
    }

    pub fn run_inference(
        &mut self,
        obs: ndarray::Array1<f32>,
    ) -> Result<(ndarray::Array2<f32>, Duration), Box<dyn std::error::Error>> {
        let inference_start = tokio::time::Instant::now();
        let inputs = ort::inputs![
            "args_tf_0" => obs,
            "args_tf_1" => self.carry.clone(),
        ]?;
        let outputs = self.model.run(inputs)?;
        let actions = outputs[0].try_extract_tensor::<f32>()?;
        let carry = outputs[1].try_extract_tensor::<f32>()?;
        let inference_time = inference_start.elapsed();

        let actions_array = actions.into_shape_with_order(ndarray::Ix2(1, 20))?;
        // update carry
        self.carry
            .slice_mut(ndarray::s![0..5, 0..128])
            .assign(&carry);

        Ok((actions_array.to_owned(), inference_time))
    }
}
