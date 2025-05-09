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
const KP_00: f32 = 20.0; // For robstride_00 actuators
const KP_02: f32 = 40.0; // For robstride_02 actuators
const KP_03: f32 = 100.0; // For some robstride_03 actuators
const KP_04: f32 = 150.0; // For robstride_04 actuators
const KP_LARGE: f32 = 200.0; // For other robstride_03 actuators

// Tau (soft torque limit) values based on actuator types
const TAU_00: f32 = 9.8; // For robstride_00 actuators
const TAU_02: f32 = 11.9; // For robstride_02 actuators
const TAU_03: f32 = 42.0; // For robstride_03 actuators
const TAU_04: f32 = 84.0; // For robstride_04 actuators

// Kd values are specified directly in the ACTUATOR_KP_KD array
// as they are highly specific to each joint configuration.

// This is a mapping from the actuator ID to the PID gains and torque limits.
// (Actuator ID, Kp, Kd, Tau_limit)
pub const ACTUATOR_KP_KD: [(usize, f32, f32, f32); 20] = [
    (11, KP_03, 8.284, TAU_03),     // left_shoulder_pitch_03
    (12, KP_03, 8.257, TAU_03),     // left_shoulder_roll_03
    (13, KP_02, 0.945, TAU_02),     // left_shoulder_yaw_02
    (14, KP_02, 1.266, TAU_02),     // left_elbow_02
    (15, KP_00, 0.295, TAU_00),     // left_wrist_00
    (21, KP_03, 8.284, TAU_03),     // right_shoulder_pitch_03
    (22, KP_03, 8.257, TAU_03),     // right_shoulder_roll_03
    (23, KP_02, 0.945, TAU_02),     // right_shoulder_yaw_02
    (24, KP_02, 1.266, TAU_02),     // right_elbow_02
    (25, KP_00, 0.295, TAU_00),     // right_wrist_00
    (31, KP_04, 24.722, TAU_04),    // left_hip_pitch_04
    (32, KP_LARGE, 26.387, TAU_03), // left_hip_roll_03
    (33, KP_03, 3.419, TAU_03),     // left_hip_yaw_03
    (34, KP_04, 8.654, TAU_04),     // left_knee_04
    (35, KP_02, 0.99, TAU_02),      // left_ankle_02
    (41, KP_04, 24.722, TAU_04),    // right_hip_pitch_04
    (42, KP_LARGE, 26.387, TAU_03), // right_hip_roll_03
    (43, KP_03, 3.419, TAU_03),     // right_hip_yaw_03
    (44, KP_04, 8.654, TAU_04),     // right_knee_04
    (45, KP_02, 0.99, TAU_02),      // right_ankle_02
];

// We define a "home position" for the robot
pub const HOME_POSITION: [(usize, f32); 20] = [
    (21, 0.0),   // right_shoulder_pitch_03
    (22, -10.0), // right_shoulder_roll_03
    (23, 0.0),   // right_shoulder_yaw_02
    (24, 15.0),  // right_elbow_02
    (25, 0.0),   // right_wrist_00
    (11, 0.0),   // left_shoulder_pitch_03
    (12, 10.0),  // left_shoulder_roll_03
    (13, 0.0),   // left_shoulder_yaw_02
    (14, -15.0), // left_elbow_02
    (15, 0.0),   // left_wrist_00
    (41, -25.0), // right_hip_pitch_04
    (42, -5.0),  // right_hip_roll_03
    (43, 0.0),   // right_hip_yaw_03
    (44, -50.0), // right_knee_04
    (45, 25.0),  // right_ankle_02
    (31, 25.0),  // left_hip_pitch_04
    (32, 5.0),   // left_hip_roll_03
    (33, 0.0),   // left_hip_yaw_03
    (34, 50.0),  // left_knee_04
    (35, -25.0), // left_ankle_02
];

// The commented-out values are the original URDF joint limits - I am
// adding software limits here to avoid some hardware failures.
pub const NN_JOINT_LIMITS_DEGREES: [(usize, f32, f32); 20] = [
    (0, -180.00001984784282, 79.99997699027486), // right_shoulder_pitch_03
    (1, -95.00001206679983, 20.000008571513593), // right_shoulder_roll_03
    (2, -95.00001206679983, 95.00001206679983),  // right_shoulder_yaw_02
    (3, 0.0, 142.00002648027882),                // right_elbow_02
    (4, -99.99998556178845, 99.99998556178845),  // right_wrist_00
    (5, -79.99997699027486, 180.00001984784282), // left_shoulder_pitch_03
    (6, -20.000008571513593, 95.00001206679983), // left_shoulder_roll_03
    (7, -95.00001206679983, 95.00001206679983),  // left_shoulder_yaw_02
    (8, -142.00002648027882, 0.0),               // left_elbow_02
    (9, -99.99998556178845, 99.99998556178845),  // left_wrist_00
    (10, -126.99999140375387, 60.00002571454079), // right_hip_pitch_04
    (11, -129.99999841905884, 12.000028061219961), // right_hip_roll_03
    (12, -89.99998127603166, 89.99998127603166), // right_hip_yaw_03
    (13, -154.99998048556108, 0.0),              // right_knee_04
    (14, -13.000011301061788, 71.99999647998123), // right_ankle_02
    (15, -60.00002571454079, 126.99999140375387), // left_hip_pitch_04
    (16, -12.000028061219961, 129.99999841905884), // left_hip_roll_03
    (17, -89.99998127603166, 89.99998127603166), // left_hip_yaw_03
    (18, 0.0, 154.99998048556108),               // left_knee_04
    (19, -71.99999647998123, 13.000011301061788), // left_ankle_02
];
