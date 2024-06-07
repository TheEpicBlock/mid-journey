use std::{collections::HashMap};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::{color::Color, input::{Config, TrainingDataRaw}};

pub struct TrainingData {
    pub training: HashMap<String, Color>,
    pub checking: HashMap<String, Color>
}

pub fn process_data(raw: TrainingDataRaw, config: &Config) -> (TrainingData, u32) {
    let mut output = HashMap::<String, Color>::default();
    let mut cut_data = 0;

    for (name, color_str) in raw {
        if name.len() > config.input_length as usize {
            cut_data += 1;
            continue;
        }
        output.insert(name, Color::from_str(&color_str).unwrap());
    }

    let mut training = HashMap::<String, Color>::default();
    let mut checking = HashMap::<String, Color>::default();
    
    // Random number, chosen by fair dice roll
    // (having this be deterministic should help reproducability)
    let mut rand = ChaCha20Rng::from_seed([4; 32]);

    for (name, color) in output {
        if rand.gen_ratio((10000f64*config.percentage_training) as u32, 10000) {
            training.insert(name, color);
        } else {
            checking.insert(name, color);
        }
    }

    return (TrainingData {
        training,
        checking,
    }, cut_data);
}
