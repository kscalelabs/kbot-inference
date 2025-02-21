use crate::{
    actuators::{Actuator, ActuatorCommand},
    constants::*,
    imu::IMU,
};
use eyre::eyre;
use ndarray;
use ort::{session::builder::GraphOptimizationLevel, session::Session, Error as OrtError};
use std::time::Duration;

pub struct NeuralNetworkRunner {
    model: Session,
    x_vel: ndarray::Array1<f32>,
    y_vel: ndarray::Array1<f32>,
    rot: ndarray::Array1<f32>,
    t: ndarray::Array1<f32>,
    dof_pos: ndarray::Array1<f32>,
    dof_vel: ndarray::Array1<f32>,
    prev_actions: ndarray::Array1<f32>,
    projected_gravity: ndarray::Array1<f32>,
    buffer: ndarray::Array1<f32>,
}

#[derive(Clone)]
pub struct NetworkInput {
    pub x_vel: ndarray::Array1<f32>,
    pub y_vel: ndarray::Array1<f32>,
    pub rot: ndarray::Array1<f32>,
    pub t: ndarray::Array1<f32>,
    pub dof_pos: ndarray::Array1<f32>,
    pub dof_vel: ndarray::Array1<f32>,
    pub prev_actions: ndarray::Array1<f32>,
    pub projected_gravity: ndarray::Array1<f32>,
    pub buffer: ndarray::Array1<f32>,
}

impl NetworkInput {
    pub fn new() -> Self {
        Self {
            x_vel: ndarray::Array1::<f32>::zeros(1),
            y_vel: ndarray::Array1::<f32>::zeros(1),
            rot: ndarray::Array1::<f32>::zeros(1),
            t: ndarray::Array1::<f32>::zeros(1),
            dof_pos: ndarray::Array1::<f32>::zeros(10),
            dof_vel: ndarray::Array1::<f32>::zeros(10),
            prev_actions: ndarray::Array1::<f32>::zeros(10),
            projected_gravity: ndarray::Array1::<f32>::zeros(3),
            buffer: ndarray::Array1::<f32>::zeros(570),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.x_vel.iter()
            .chain(self.y_vel.iter())
            .chain(self.rot.iter())
            .chain(self.t.iter())
            .chain(self.dof_pos.iter())
            .chain(self.dof_vel.iter())
            .chain(self.prev_actions.iter())
            .chain(self.projected_gravity.iter())
            .chain(self.buffer.iter())
    }
}

impl NeuralNetworkRunner {
    pub fn new(model_path: &str) -> Result<Self, OrtError> {
        let model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        // Initialize all input buffers
        Ok(Self {
            model,
            x_vel: ndarray::Array1::<f32>::zeros(1),
            y_vel: ndarray::Array1::<f32>::zeros(1), 
            rot: ndarray::Array1::<f32>::zeros(1),
            t: ndarray::Array1::<f32>::zeros(1),
            dof_pos: ndarray::Array1::<f32>::zeros(10),
            dof_vel: ndarray::Array1::<f32>::zeros(10),
            prev_actions: ndarray::Array1::<f32>::zeros(10),
            projected_gravity: ndarray::Array1::<f32>::zeros(3),
            buffer: ndarray::Array1::<f32>::zeros(570),
        })
    }

    pub fn get_observation_size(&self) -> usize {
        self.x_vel.len() + self.y_vel.len() + self.rot.len() + self.t.len() + self.dof_pos.len() + self.dof_vel.len() + self.prev_actions.len() + self.projected_gravity.len() + self.buffer.len()
    }

    pub fn get_action_size() -> usize {
        ACTUATOR_ID_MAP.len()
    }

    pub async fn get_targets() -> Result<[f32; 3], Box<dyn std::error::Error>> {
        Ok([0.5, 0.0, 0.0]) // x_vel, y_vel, rot
    }

    pub async fn get_dof_pos_and_vel(
        actuators: &Option<Actuator>,
        actuator_ids: &Vec<u8>,
        slowdown_factor: f32,
    ) -> Result<[f32; 20], Box<dyn std::error::Error>> {
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
                    .find(|(id, _, _)| *id == *actuator_id)
                    .map(|(_, nn_idx, _)| *nn_idx);

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

            // Subtract off the home position
            for (nn_idx, home_pos) in NN_HOME_POSITION.iter() {
                let pos = positions.get_mut(*nn_idx).ok_or_else(|| {
                    format!("Missing position for neural network index {}", nn_idx)
                })?;
                *pos -= *home_pos as f64;
            }

            // Scale positions.
            // let scale = 0.5;
            // positions = positions
            //     .iter()
            //     .map(|p| p / scale as f64)
            //     .collect::<Vec<_>>();
            // velocities = velocities
            //     .iter()
            //     .map(|v| v / scale as f64)
            //     .collect::<Vec<_>>();

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
                .collect();

            // Create array and copy positions and velocities into it
            let mut result = [0.0; 20];
            result[..10].copy_from_slice(&positions);
            result[10..].copy_from_slice(&velocities);

            Ok(result.map(|x| x as f32))
        } else {
            // Return zeros in dry run mode
            Ok([0.0; 20])
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
            let gravity = Self::euler_angles_to_gravity(imu_values.roll as f32, imu_values.pitch as f32)?;
            
            Ok([-gravity[0], -gravity[1], gravity[2]])
        } else {
            // Return zeros in dry run mode
            Ok([0.0; 3])
        }
    }

    pub async fn update_commands(
        actions: ndarray::Array2<f32>,
    ) -> Result<Vec<ActuatorCommand>, Box<dyn std::error::Error>> {
        // Convert from radians to degrees.
        let actions = actions * 180.0 / std::f32::consts::PI;

        // Apply scaling factor.
        // let actions = actions * 0.5;

        // Add back the home position
        let mut final_actions = actions.to_owned();

        // Add back the home position.
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
            .map(|(actuator_id, nn_idx, is_inverted)| {
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
    ) -> Result<(NetworkInput, Duration), Box<dyn std::error::Error>> {
        let sensor_start = tokio::time::Instant::now();
        let (targets, imu_values, dof_values) = tokio::join!(
            Self::get_targets(),
            Self::get_imu_values(imu),
            Self::get_dof_pos_and_vel(actuators, actuator_ids, slowdown_factor),
        );
        let sensor_time = sensor_start.elapsed();

        let mut input = NetworkInput::new();
        
        // Update observation vector
        let targets = targets?;
        input.x_vel.slice_mut(ndarray::s![0..1])
            .assign(&ndarray::Array1::from_vec(targets[0..1].to_vec()));
        input.y_vel.slice_mut(ndarray::s![0..1])
            .assign(&ndarray::Array1::from_vec(targets[1..2].to_vec()));
        input.rot.slice_mut(ndarray::s![0..1])
            .assign(&ndarray::Array1::from_vec(targets[2..3].to_vec()));

        let imu_values = imu_values?;
        input.projected_gravity.slice_mut(ndarray::s![0..3])
            .assign(&ndarray::Array1::from_vec(imu_values.to_vec()));

        let dof_values = dof_values?;
        input.dof_pos.slice_mut(ndarray::s![0..10])
            .assign(&ndarray::Array1::from_vec(dof_values[0..10].to_vec()));
        input.dof_vel.slice_mut(ndarray::s![0..10])
            .assign(&ndarray::Array1::from_vec(dof_values[10..20].to_vec()));

        // Copy current state
        input.prev_actions = self.prev_actions.to_owned();
        input.buffer = self.buffer.to_owned();

        Ok((input, sensor_time))
    }

    pub fn run_inference(
        &mut self,
        input: NetworkInput,
        start_time: std::time::Instant,
    ) -> Result<(ndarray::Array2<f32>, Duration), Box<dyn std::error::Error>> {
        let inference_start = tokio::time::Instant::now();

        // Update timestamp in the input
        let mut input = input;
        input.t[[0]] = start_time.elapsed().as_secs_f32();

        let inputs = ort::inputs! {
            "x_vel.1" => input.x_vel,
            "y_vel.1" => input.y_vel,
            "rot.1" => input.rot,
            "t.1" => input.t,
            "dof_pos.1" => input.dof_pos,
            "dof_vel.1" => input.dof_vel,
            "prev_actions.1" => input.prev_actions,
            "projected_gravity.1" => input.projected_gravity,
            "buffer.1" => input.buffer,
        }?;

        let outputs = self.model.run(inputs)?;
        
        // Extract scaled actions for output
        let actions_scaled = outputs.get("actions_scaled")
            .ok_or_else(|| eyre!("Missing actions_scaled output"))?
            .try_extract_tensor::<f32>()?;
        let actions_array = actions_scaled.into_shape_with_order(ndarray::Ix2(1, Self::get_action_size()))?;
        
        // Update prev_actions and buffer from the model output
        let actions = outputs.get("actions")
            .ok_or_else(|| eyre!("Missing actions output"))?
            .try_extract_tensor::<f32>()?;
        let temp_actions_array = actions.into_shape_with_order(ndarray::Ix1(10))?;
        self.prev_actions = temp_actions_array.to_owned();

        let buffer = outputs.get("x.3")
            .ok_or_else(|| eyre!("Missing buffer output"))?
            .try_extract_tensor::<f32>()?;
        let buffer_array = buffer.into_shape_with_order(ndarray::Ix1(570))?;
        self.buffer = buffer_array.to_owned();

        let inference_time = inference_start.elapsed();
        Ok((actions_array.to_owned(), inference_time))
    }
}
