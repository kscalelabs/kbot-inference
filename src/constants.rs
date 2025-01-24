// Mapping from the neural network index to the actuator ID, with a flag
// indicating whether or not the actuator is oriented in the same direction
// on the real robot as it is in the URDF.
pub const ACTUATOR_ID_MAP: [(usize, u8, bool); 20] = [
    (0, 31, true),  // left_hip_pitch_04
    (1, 11, true),  // left_shoulder_pitch_03
    (2, 41, true),  // right_hip_pitch_04
    (3, 21, true),  // right_shoulder_pitch_03
    (4, 32, true),  // left_hip_roll_03
    (5, 12, true),  // left_shoulder_roll_03
    (6, 42, true),  // right_hip_roll_03
    (7, 22, true),  // right_shoulder_roll_03
    (8, 33, true),  // left_hip_yaw_03
    (9, 13, true),  // left_shoulder_yaw_02
    (10, 43, true), // right_hip_yaw_03
    (11, 23, true), // right_shoulder_yaw_02
    (12, 34, true), // left_knee_04
    (13, 14, true), // left_elbow_02
    (14, 44, true), // right_knee_04
    (15, 24, true), // right_elbow_02
    (16, 35, true), // left_ankle_02
    (17, 15, true), // left_wrist_02
    (18, 45, true), // right_ankle_02
    (19, 25, true), // right_wrist_02
];

// This is a mapping from the actuator ID to the PID gains. During training,
// we just use kp = 300 and kd = 5 for the 04 actuators, kp = 150 and kd = 5
// for the 03 actuators, and kp = 40 and kd = 5 for the 02 actuators.
pub const ACTUATOR_KP_KD: [(usize, f32, f32); 20] = [
    (11, 150.0, 5.0), // left_shoulder_pitch_03
    (12, 150.0, 5.0), // left_shoulder_roll_03
    (13, 40.0, 5.0),  // left_shoulder_yaw_02
    (14, 40.0, 5.0),  // left_elbow_02
    (15, 40.0, 5.0),  // left_wrist_02
    (21, 150.0, 5.0), // right_shoulder_pitch_03
    (22, 150.0, 5.0), // right_shoulder_roll_03
    (23, 40.0, 5.0),  // right_shoulder_yaw_02
    (24, 40.0, 5.0),  // right_elbow_02
    (25, 40.0, 5.0),  // right_wrist_02
    (31, 300.0, 5.0), // left_hip_pitch_04
    (32, 150.0, 5.0), // left_hip_roll_03
    (33, 150.0, 5.0), // left_hip_yaw_03
    (34, 300.0, 5.0), // left_knee_04
    (35, 40.0, 5.0),  // left_ankle_02
    (41, 300.0, 5.0), // right_hip_pitch_04
    (42, 150.0, 5.0), // right_hip_roll_03
    (43, 150.0, 5.0), // right_hip_yaw_03
    (44, 300.0, 5.0), // right_knee_04
    (45, 40.0, 5.0),  // right_ankle_02
];

// We define a "home position" for the robot when training the neural network,
// and the neural network inputs and outputs are relative to this home
// position. This means we need to subtract the home position from the
// absolute position when providing the input to the neural network, and add
// it back to the output.
pub const NN_HOME_POSITION: [(usize, f32); 20] = [
    (0, 60.0),   // left_hip_pitch_04
    (1, 0.0),    // left_shoulder_pitch_03
    (2, -60.0),  // right_hip_pitch_04
    (3, 0.0),    // right_shoulder_pitch_03
    (4, 0.0),    // left_hip_roll_03
    (5, 0.0),    // left_shoulder_roll_03
    (6, 0.0),    // right_hip_roll_03
    (7, 0.0),    // right_shoulder_roll_03
    (8, 0.0),    // left_hip_yaw_03
    (9, -80.0),  // left_shoulder_yaw_02
    (10, 0.0),   // right_hip_yaw_03
    (11, -80.0), // right_shoulder_yaw_02
    (12, -70.0), // left_knee_04
    (13, -90.0), // left_elbow_02
    (14, 70.0),  // right_knee_04
    (15, 90.0),  // right_elbow_02
    (16, 30.0),  // left_ankle_02
    (17, 0.0),   // left_wrist_02
    (18, 30.0),  // right_ankle_02
    (19, 0.0),   // right_wrist_02
];
