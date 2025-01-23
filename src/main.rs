use ort::Error as OrtError;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;

fn load_onnx_model(model_path: &str) -> Result<Session, OrtError> {
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    Ok(model)
}

fn run_model(model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let model = load_onnx_model(model_path).unwrap();
    println!("Model loaded");

    // let x_vel = ndarray::Array1::<f32>::zeros(1);
    // let y_vel = ndarray::Array1::<f32>::zeros(1);
    // let rot = ndarray::Array1::<f32>::zeros(1);
    // let mut t = ndarray::Array1::<f32>::zeros(1);
    // let dof_pos = ndarray::Array1::<f32>::zeros(10);
    // let dof_vel = ndarray::Array1::<f32>::zeros(10);
    // let mut prev_actions = ndarray::Array1::<f32>::zeros(10);
    // let imu_ang_vel = ndarray::Array1::<f32>::zeros(3);
    // let imu_euler_xyz = ndarray::Array1::<f32>::zeros(3);
    // let mut buffer = ndarray::Array1::<f32>::zeros(574);

    let mut obs = ndarray::Array2::<f32>::zeros((1, 69));

    for i in 0..50 {
        obs[(0, 0)] = i as f32 / 50.0;
        let outputs = model.run(ort::inputs!["obs" => obs.clone()]?)?;
        let actions = outputs[0].try_extract_tensor::<f32>().unwrap();
        println!("{:?}", actions);
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        // Add this line
        return Err("Expected one argument".into());
    }
    let model_path = &args[1];
    let _ =run_model(model_path);
    Ok(())
}
