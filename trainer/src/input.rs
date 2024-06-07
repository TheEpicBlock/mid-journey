use std::collections::BTreeMap;

use serde::Deserialize;

pub type TrainingDataRaw = BTreeMap<String, String>;

#[derive(Deserialize)]
pub struct Config {
    pub input_length: u32,
    pub percentage_training: f64
}