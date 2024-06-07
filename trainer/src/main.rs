use std::{env, fs::File, path::{self, Path, PathBuf}};

use input::{Config, TrainingDataRaw};
use training_data::process_data;

mod input;
mod training_data;

fn main() {
    let args: Vec<_> = env::args_os().collect();
    if args.len() != 3 { // TODO change to 4
        println!("Usage: {:?} <training_data> <nn_config> <output_file>", args[0]);
        return;
    }

    let training_data_file = PathBuf::try_from(&args[1]).unwrap();
    let config_file = PathBuf::try_from(&args[2]).unwrap();

    let data: TrainingDataRaw = serde_json::from_reader(File::open(training_data_file).unwrap()).unwrap();
    let config: Config = serde_json::from_reader(File::open(config_file).unwrap()).unwrap();

    let data = process_data(data, &config);

    println!("Starting trainig process!");
    println!("Training set contains {} entries", data.len());
    println!("Input length: {}", config.input_length);
}
