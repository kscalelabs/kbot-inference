// Mapping from the neural network index to the actuator ID. This also includes
// a flag indicating whether or not the real actuator orientation is flipped
// relative to the URDF model, and the lower and upper limits of the actuator.
pub const ACTUATOR_ID_MAP: [(usize, u8, bool); 20] = [
    (0, 31, false),  // left_hip_pitch_04
    (1, 11, false),  // left_shoulder_pitch_03
    (2, 41, false),  // right_hip_pitch_04
    (3, 21, false),  // right_shoulder_pitch_03
    (4, 32, true),   // left_hip_roll_03
    (5, 12, true),   // left_shoulder_roll_03
    (6, 42, true),   // right_hip_roll_03
    (7, 22, false),  // right_shoulder_roll_03
    (8, 33, false),  // left_hip_yaw_03
    (9, 13, false),  // left_shoulder_yaw_02
    (10, 43, false), // right_hip_yaw_03
    (11, 23, false), // right_shoulder_yaw_02
    (12, 34, false), // left_knee_04
    (13, 14, false), // left_elbow_02
    (14, 44, false), // right_knee_04
    (15, 24, false), // right_elbow_02
    (16, 35, false), // left_ankle_02
    (17, 15, false), // left_wrist_02
    (18, 45, true),  // right_ankle_02
    (19, 25, false), // right_wrist_02
];

// This is a mapping from the actuator ID to the PID gains. During training,
// we just use kp = 300 and kd = 5 for the 04 actuators, kp = 150 and kd = 5
// for the 03 actuators, and kp = 40 and kd = 5 for the 02 actuators.
pub const ACTUATOR_KP_KD: [(usize, f32, f32); 20] = [
    (11, 60.0, 5.0),  // left_shoulder_pitch_03
    (12, 60.0, 5.0),  // left_shoulder_roll_03
    (13, 20.0, 2.0),  // left_shoulder_yaw_02
    (14, 20.0, 2.0),  // left_elbow_02
    (15, 20.0, 2.0),  // left_wrist_02
    (21, 60.0, 5.0),  // right_shoulder_pitch_03
    (22, 60.0, 5.0),  // right_shoulder_roll_03
    (23, 20.0, 2.0),  // right_shoulder_yaw_02
    (24, 20.0, 2.0),  // right_elbow_02
    (25, 20.0, 2.0),  // right_wrist_02
    (31, 150.0, 5.0), // left_hip_pitch_04
    (32, 60.0, 5.0),  // left_hip_roll_03
    (33, 60.0, 5.0),  // left_hip_yaw_03
    (34, 150.0, 5.0), // left_knee_04
    (35, 20.0, 2.0),  // left_ankle_02
    (41, 150.0, 5.0), // right_hip_pitch_04
    (42, 60.0, 5.0),  // right_hip_roll_03
    (43, 60.0, 5.0),  // right_hip_yaw_03
    (44, 150.0, 5.0), // right_knee_04
    (45, 20.0, 2.0),  // right_ankle_02
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

pub const NN_JOINT_LIMITS: [(usize, f32, f32); 20] = [
    (0, -90.0, 90.0),    // left_hip_pitch_04
    (1, 0.0, 180.0),     // left_shoulder_pitch_03
    (2, -90.0, 90.0),    // right_hip_pitch_04
    (3, -180.0, 0.0),    // right_shoulder_pitch_03
    (4, -182.5, 20.0),   // left_hip_roll_03
    (5, -208.0, 27.5),   // left_shoulder_roll_03
    (6, -20.0, 182.5),   // right_hip_roll_03
    (7, -27.5, 208.0),   // right_shoulder_roll_03
    (8, -90.0, 90.0),    // left_hip_yaw_03
    (9, -90.0, 90.0),    // left_shoulder_yaw_02
    (10, -90.0, 90.0),   // right_hip_yaw_03
    (11, -90.0, 90.0),   // right_shoulder_yaw_02
    (12, 0.0, 120.0),    // left_knee_04
    (13, -145.0, 0.0),   // left_elbow_02
    (14, -120.0, 0.0),   // right_knee_04
    (15, 0.0, 145.0),    // right_elbow_02
    (16, -40.0, 40.0),   // left_ankle_02
    (17, -180.0, 180.0), // left_wrist_02
    (18, -40.0, 40.0),   // right_ankle_02
    (19, -180.0, 180.0), // right_wrist_02
];
