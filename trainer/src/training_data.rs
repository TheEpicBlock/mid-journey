use std::{collections::HashMap, str::FromStr};

use color_processing::Color;

use crate::input::{Config, TrainingDataRaw};

pub type TrainingData = HashMap<String, Color>;

pub fn process_data(raw: TrainingDataRaw, config: &Config) -> TrainingData {
    let mut output: TrainingData = Default::default();

    for (name, color_str) in raw {
        if name.len() > config.input_length as usize {
            continue;
        }
        output.insert(name, color_str.parse().unwrap());
    }

    return output;
}
