// Mapping from the neural network index to the actuator ID. This also includes
// a flag indicating whether or not the real actuator orientation is flipped
// relative to the URDF model.
pub const ACTUATOR_ID_MAP: [(u8, usize, bool); 10] = [
    // (11, 1, false),  // left_shoulder_pitch_03
    // (12, 5, false),  // left_shoulder_roll_03
    // (13, 9, false),  // left_shoulder_yaw_02
    // (14, 13, false), // left_elbow_02
    // (15, 17, false), // left_wrist_02
    // (21, 3, false),  // right_shoulder_pitch_03
    // (22, 7, false),  // right_shoulder_roll_03
    // (23, 11, false), // right_shoulder_yaw_02
    // (24, 15, false), // right_elbow_02s
    // (25, 19, false), // right_wrist_02
    (31, 0, false),  // left_hip_pitch_04
    (32, 1, false),  // left_hip_roll_03
    (33, 2, false),  // left_hip_yaw_03
    (34, 3, false), // left_knee_04
    (35, 4, false), // left_ankle_02
    (41, 5, false),  // right_hip_pitch_04
    (42, 6, false),  // right_hip_roll_03
    (43, 7, false), // right_hip_yaw_03
    (44, 8, false), // right_knee_04
    (45, 9, false),  // right_ankle_02
];

const KP_02: f32 = 40.0;
const KD_02: f32 = 5.0;
const TAU_02: f32 = 20.0;
// const TAU_02: f32 = 2.0;

const KP_03: f32 = 120.0;
const KD_03: f32 = 5.0;
const TAU_03: f32 = 50.0;
// const TAU_03: f32 = 5.0;

const KP_04: f32 = 300.0;
const KD_04: f32 = 5.0;
const TAU_04: f32 = 70.0;
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
// Units are in degrees.
pub const NN_HOME_POSITION: [(usize, f32); 10] = [
    (0, 13.18), // left_hip_pitch_04
    (1, 0.0), // left_hip_roll_03
    (2, 0.0), // left_hip_yaw_03
    (3, 25.27), // left_knee_04
    (4, -11.17), // left_ankle_02
    (5, -13.18), // right_hip_pitch_04
    (6, 0.0), // right_hip_roll_03
    (7, 0.0), // right_hip_yaw_03
    (8, -25.27), // right_knee_04
    (9, 11.17), // right_ankle_02
];

// The commented-out values are the original URDF joint limits - I am
// adding software limits here to avoid some hardware failures.
pub const NN_JOINT_LIMITS: [(usize, f32, f32); 10] = [
    // (0, -90.0, 90.0), // left_hip_pitch_04
    // (1, 0.0, 180.0),  // left_shoulder_pitch_03
    // (2, -90.0, 90.0), // right_hip_pitch_04
    // (3, -180.0, 0.0), // right_shoulder_pitch_03
    // // Software limit to avoid some hardware issues.
    // // (4, -182.5, 20.0),   // left_hip_roll_03
    // (4, -20.0, 20.0), // left_hip_roll_03
    // // Software limit to avoid some hardware issues.
    // // (5, -208.0, 27.5), // left_shoulder_roll_03
    // (5, -90.0, 27.5), // left_shoulder_roll_03
    // // Software limit to avoid some hardware issues.
    // // (6, -20.0, 182.5),   // right_hip_roll_03
    // (6, -20.0, 20.0), // right_hip_roll_03
    // // (7, -27.5, 208.0),   // right_shoulder_roll_03
    // (7, -27.5, 90.0),    // right_shoulder_roll_03
    // (8, -90.0, 90.0),    // left_hip_yaw_03
    // (9, -90.0, 90.0),    // left_shoulder_yaw_02
    // (10, -90.0, 90.0),   // right_hip_yaw_03
    // (11, -90.0, 90.0),   // right_shoulder_yaw_02
    // (12, 0.0, 120.0),    // left_knee_04
    // (13, -145.0, 0.0),   // left_elbow_02
    // (14, -120.0, 0.0),   // right_knee_04
    // (15, 0.0, 145.0),    // right_elbow_02
    // (16, -40.0, 40.0),   // left_ankle_02
    // (17, -180.0, 180.0), // left_wrist_02
    // (18, -40.0, 40.0),   // right_ankle_02
    // (19, -180.0, 180.0), // right_wrist_02
    // ISAACGYM POLICY
    (0, -90.0, 90.0),   // left_hip_pitch_04
    (1, -20.0, 20.0),   // left_hip_roll_03
    (2, -90.0, 90.0),   // left_hip_yaw_03
    (3, 0.0, 120.0),    // left_knee_04
    (4, -40.0, 40.0),   // left_ankle_02
    (5, -90.0, 90.0),   // right_hip_pitch_04
    (6, -20.0, 20.0),   // right_hip_roll_03
    (7, -90.0, 90.0),   // right_hip_yaw_03
    (8, -120.0, 0.0),   // right_knee_04
    (9, -40.0, 40.0),   // right_ankle_02
];
