use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::{color::Color, input::{Config, TrainingDataRaw}, layer::MainType};

pub type GpuInputData = Vec<MainType>;

pub type DataSet = Vec<(GpuInputData, Color)>;

pub struct TrainingData {
    pub training: DataSet,
    pub checking: DataSet
}

pub fn process_data(raw: TrainingDataRaw, config: &Config) -> (TrainingData, u32) {
    let mut output = Vec::<(GpuInputData, Color)>::default();
    let mut cut_data = 0;

    for (name, color_str) in raw {
        if name.len() > config.input_length as usize {
            cut_data += 1;
            continue;
        }
        output.push((string_to_data(&name, config), Color::from_str(&color_str).unwrap()));
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

    (TrainingData { training, checking, }, cut_data)
}

pub fn string_to_data(str: &str, config: &Config) -> GpuInputData {
    let mut output = vec![0 as MainType; config.input_length as usize];

    for (i, char) in str.chars().enumerate() {
        output[i] = convert_char(char, config);
    }

    return output;
}

fn convert_char(char: char, config: &Config) -> MainType {
    let char = char.to_ascii_uppercase();
    return *config.letter_mapping.get(&char).unwrap_or(&(0 as MainType));
}
