use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::{color::Color, input::{Config, TrainingDataRaw}, layer::MainType, string::string_to_data};

pub type GpuInputData = Vec<MainType>;

pub type DataSet = Vec<(GpuInputData, Color)>;

pub struct TrainingData {
    pub training: DataSet,
    pub checking: DataSet
}

pub fn process_data(raw: TrainingDataRaw, config: &Config) -> (TrainingData, u32) {
    let mut output = Vec::<(GpuInputData, Color)>::default();
    let mut truncated_data = 0;

    for (name, color_str) in raw {
        let mut name: &str = &name;
        if name.len() > config.input_length_max_chars() as usize {
            truncated_data += 1;
            name = &name[..(config.input_length_max_chars() as usize)];
        }
        output.push((string_to_data(name, config), Color::from_str(&color_str).unwrap()));
    }

    let mut training = Vec::<(GpuInputData, Color)>::default();
    let mut checking = Vec::<(GpuInputData, Color)>::default();
    
    // Random number, chosen by fair dice roll
    // (having this be deterministic should help reproducability)
    let mut rand = ChaCha20Rng::from_seed([4; 32]);

    for data in output {
        if rand.gen_ratio((10000f64*config.percentage_training) as u32, 10000) {
            training.push(data);
        } else {
            checking.push(data);
        }
    }

    (TrainingData { training, checking, }, truncated_data)
}
