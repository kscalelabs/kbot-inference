// Mapping from the actuator ID to the neural network index.
pub const ACTUATOR_ID_MAP: [(u8, usize); 20] = [
    (11, 5),  // left_shoulder_pitch_03
    (12, 6),  // left_shoulder_roll_03
    (13, 7),  // left_shoulder_yaw_02
    (14, 8),  // left_elbow_02
    (15, 9),  // left_wrist_00
    (21, 0),  // right_shoulder_pitch_03
    (22, 1),  // right_shoulder_roll_03
    (23, 2),  // right_shoulder_yaw_02
    (24, 3),  // right_elbow_02
    (25, 4),  // right_wrist_00
    (31, 15), // left_hip_pitch_04
    (32, 16), // left_hip_roll_03
    (33, 17), // left_hip_yaw_03
    (34, 18), // left_knee_04
    (35, 19), // left_ankle_02
    (41, 10), // right_hip_pitch_04
    (42, 11), // right_hip_roll_03
    (43, 12), // right_hip_yaw_03
    (44, 13), // right_knee_04
    (45, 14), // right_ankle_02
];

// Kp values based on actuator types or common settings
const KP_20: f32 = 20.0; // For robstride_00 actuators
const KP_40: f32 = 40.0; // For robstride_02 actuators
const KP_100: f32 = 100.0; // For some robstride_03 actuators
const KP_150: f32 = 150.0; // For robstride_04 actuators
const KP_200: f32 = 200.0; // For other robstride_03 actuators

// Tau (soft torque limit) values based on actuator types
const TAU_9_8: f32 = 9.8; // For robstride_00 actuators
const TAU_11_9: f32 = 11.9; // For robstride_02 actuators
const TAU_42_0: f32 = 42.0; // For robstride_03 actuators
const TAU_84_0: f32 = 84.0; // For robstride_04 actuators

// Kd values are specified directly in the ACTUATOR_KP_KD array
// as they are highly specific to each joint configuration.

// This is a mapping from the actuator ID to the PID gains and torque limits.
// (Actuator ID, Kp, Kd, Tau_limit)
pub const ACTUATOR_KP_KD: [(usize, f32, f32, f32); 20] = [
    (11, KP_100, 8.284, TAU_42_0),  // left_shoulder_pitch_03
    (12, KP_100, 8.257, TAU_42_0),  // left_shoulder_roll_03
    (13, KP_40, 0.945, TAU_11_9),   // left_shoulder_yaw_02
    (14, KP_40, 1.266, TAU_11_9),   // left_elbow_02
    (15, KP_20, 0.295, TAU_9_8),    // left_wrist_00
    (21, KP_100, 8.284, TAU_42_0),  // right_shoulder_pitch_03
    (22, KP_100, 8.257, TAU_42_0),  // right_shoulder_roll_03
    (23, KP_40, 0.945, TAU_11_9),   // right_shoulder_yaw_02
    (24, KP_40, 1.266, TAU_11_9),   // right_elbow_02
    (25, KP_20, 0.295, TAU_9_8),    // right_wrist_00
    (31, KP_150, 24.722, TAU_84_0), // left_hip_pitch_04
    (32, KP_200, 26.387, TAU_42_0), // left_hip_roll_03
    (33, KP_100, 3.419, TAU_42_0),  // left_hip_yaw_03
    (34, KP_150, 8.654, TAU_84_0),  // left_knee_04
    (35, KP_40, 0.99, TAU_11_9),    // left_ankle_02
    (41, KP_150, 24.722, TAU_84_0), // right_hip_pitch_04
    (42, KP_200, 26.387, TAU_42_0), // right_hip_roll_03
    (43, KP_100, 3.419, TAU_42_0),  // right_hip_yaw_03
    (44, KP_150, 8.654, TAU_84_0),  // right_knee_04
    (45, KP_40, 0.99, TAU_11_9),    // right_ankle_02
];

// We define a "home position" for the robot
pub const HOME_POSITION: [(usize, f32); 20] = [
    (21, 0.0),   // right_shoulder_pitch_03
    (22, -10.0), // right_shoulder_roll_03
    (23, 0.0),   // right_shoulder_yaw_02
    (24, 90.0),  // right_elbow_02
    (25, 0.0),   // right_wrist_00
    (11, 0.0),   // left_shoulder_pitch_03
    (12, 10.0),  // left_shoulder_roll_03
    (13, 0.0),   // left_shoulder_yaw_02
    (14, -90.0), // left_elbow_02
    (15, 0.0),   // left_wrist_00
    (41, -25.0), // right_hip_pitch_04
    (42, 0.0),   // right_hip_roll_03
    (43, 0.0),   // right_hip_yaw_03
    (44, -50.0), // right_knee_04
    (45, 25.0),  // right_ankle_02
    (31, 25.0),  // left_hip_pitch_04
    (32, 0.0),   // left_hip_roll_03
    (33, 0.0),   // left_hip_yaw_03
    (34, 50.0),  // left_knee_04
    (35, -25.0), // left_ankle_02
];

// The commented-out values are the original URDF joint limits - I am
// adding software limits here to avoid some hardware failures.
pub const NN_JOINT_LIMITS: [(usize, f32, f32); 20] = [
    (0, -90.0, 90.0), // left_hip_pitch_04
    (1, 0.0, 180.0),  // left_shoulder_pitch_03
    (2, -90.0, 90.0), // right_hip_pitch_04
    (3, -180.0, 0.0), // right_shoulder_pitch_03
    // Software limit to avoid some hardware issues.
    // (4, -182.5, 20.0),   // left_hip_roll_03
    (4, -20.0, 20.0), // left_hip_roll_03
    // Software limit to avoid some hardware issues.
    // (5, -208.0, 27.5), // left_shoulder_roll_03
    (5, -90.0, 27.5), // left_shoulder_roll_03
    // Software limit to avoid some hardware issues.
    // (6, -20.0, 182.5),   // right_hip_roll_03
    (6, -20.0, 20.0), // right_hip_roll_03
    // (7, -27.5, 208.0),   // right_shoulder_roll_03
    (7, -27.5, 90.0),    // right_shoulder_roll_03
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
