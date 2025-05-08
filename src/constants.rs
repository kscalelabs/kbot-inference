// Mapping from the actuator ID to the neural network index.
pub const ACTUATOR_ID_MAP: [(u8, usize); 20] = [
    (11, 1),  // left_shoulder_pitch_03
    (12, 5),  // left_shoulder_roll_03
    (13, 9),  // left_shoulder_yaw_02
    (14, 13), // left_elbow_02
    (15, 17), // left_wrist_02
    (21, 3),  // right_shoulder_pitch_03
    (22, 7),  // right_shoulder_roll_03
    (23, 11), // right_shoulder_yaw_02
    (24, 15), // right_elbow_02s
    (25, 19), // right_wrist_02
    (31, 0),  // left_hip_pitch_04
    (32, 4),  // left_hip_roll_03
    (33, 8),  // left_hip_yaw_03
    (34, 12), // left_knee_04
    (35, 16), // left_ankle_02
    (41, 2),  // right_hip_pitch_04
    (42, 6),  // right_hip_roll_03
    (43, 10), // right_hip_yaw_03
    (44, 14), // right_knee_04
    (45, 18), // right_ankle_02
];

const KP_02: f32 = 20.0;
const KD_02: f32 = 2.0;
const TAU_02: f32 = 17.0;
// const TAU_02: f32 = 2.0;

const KP_03: f32 = 150.0;
const KD_03: f32 = 5.0;
const TAU_03: f32 = 60.0;
// const TAU_03: f32 = 5.0;

const KP_04: f32 = 200.0;
const KD_04: f32 = 5.0;
const TAU_04: f32 = 120.0;
// const TAU_04: f32 = 15.0;

// This is a mapping from the actuator ID to the PID gains and torque limits.
pub const ACTUATOR_KP_KD: [(usize, f32, f32, f32); 20] = [
    (11, KP_02, KD_02, TAU_02), // left_shoulder_pitch_03
    (12, KP_03, KD_03, TAU_03), // left_shoulder_roll_03
    (13, KP_02, KD_02, TAU_02), // left_shoulder_yaw_02
    (14, KP_02, KD_02, TAU_02), // left_elbow_02
    (15, KP_02, KD_02, TAU_02), // left_wrist_02
    (21, KP_02, KD_02, TAU_02), // right_shoulder_pitch_03
    (22, KP_03, KD_03, TAU_03), // right_shoulder_roll_03
    (23, KP_02, KD_02, TAU_02), // right_shoulder_yaw_02
    (24, KP_02, KD_02, TAU_02), // right_elbow_02
    (25, KP_02, KD_02, TAU_02), // right_wrist_02
    (31, KP_04, KD_04, TAU_04), // left_hip_pitch_04
    (32, KP_03, KD_03, TAU_03), // left_hip_roll_03
    (33, KP_02, KD_02, TAU_02), // left_hip_yaw_03
    (34, KP_04, KD_04, TAU_04), // left_knee_04
    (35, KP_02, KD_02, TAU_02), // left_ankle_02
    (41, KP_04, KD_04, TAU_04), // right_hip_pitch_04
    (42, KP_03, KD_03, TAU_03), // right_hip_roll_03
    (43, KP_02, KD_02, TAU_02), // right_hip_yaw_03
    (44, KP_04, KD_04, TAU_04), // right_knee_04
    (45, KP_02, KD_02, TAU_02), // right_ankle_02
];

// We define a "home position" for the robot when training the neural network,
// and the neural network inputs and outputs are relative to this home
// position. This means we need to subtract the home position from the
// absolute position when providing the input to the neural network, and add
// it back to the output.
pub const NN_HOME_POSITION: [(usize, f32); 20] = [
    (0, 30.0),   // left_hip_pitch_04
    (1, 0.0),    // left_shoulder_pitch_03
    (2, -30.0),  // right_hip_pitch_04
    (3, 0.0),    // right_shoulder_pitch_03
    (4, 0.0),    // left_hip_roll_03
    (5, 0.0),    // left_shoulder_roll_03
    (6, 0.0),    // right_hip_roll_03
    (7, 0.0),    // right_shoulder_roll_03
    (8, 0.0),    // left_hip_yaw_03
    (9, 0.0),    // left_shoulder_yaw_02
    (10, 0.0),   // right_hip_yaw_03
    (11, 0.0),   // right_shoulder_yaw_02
    (12, 60.0),  // left_knee_04
    (13, -90.0), // left_elbow_02
    (14, -60.0), // right_knee_04
    (15, 90.0),  // right_elbow_02
    (16, -30.0), // left_ankle_02
    (17, 0.0),   // left_wrist_02
    (18, -30.0), // right_ankle_02
    (19, 0.0),   // right_wrist_02
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
