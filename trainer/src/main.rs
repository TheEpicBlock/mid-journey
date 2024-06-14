use std::{env, fs::File, path::PathBuf};

use gpu::init_gpu;
use input::{Config, TrainingDataRaw};
use neural_network::train_nn;
use training_data::process_data;

mod input;
mod training_data;
mod neural_network;
mod misc;
mod shaders;
mod layer;
mod gpu;
#[allow(dead_code)]
mod color;
mod string;

#[tokio::main]
async fn main() {
    let args: Vec<_> = env::args_os().collect();
    if args.len() != 4 {
        println!("Usage: {:?} <training_data> <nn_config> <output_file>", args[0]);
        return;
    }

    let training_data_file = PathBuf::from(&args[1]);
    let config_file = PathBuf::from(&args[2]);
    let output_file = PathBuf::from(&args[3]);

    let data: TrainingDataRaw = serde_json::from_reader(File::open(training_data_file).expect("Can't open training data file")).unwrap();
    let config: Config = serde_json::from_reader(File::open(config_file).expect("Can't open training data file")).unwrap();

    let (data, truncated_data) = process_data(data, &config);

    println!("Starting trainig process!");
    println!("Training set contains {} entries", data.training.len());
    println!("Check/verify set contains {} entries", data.checking.len());
    println!("Total: {} entries", data.training.len() + data.checking.len());
    println!("{} entries were truncated due to configured input size", truncated_data);

    let gpu = init_gpu().await;
    let parameters = train_nn(&gpu, data, config).await;

    let json = layer::to_json(parameters, &gpu).await;
    serde_json::to_writer(File::create(output_file).expect("Couldn't open output file"), &json).unwrap();
}
