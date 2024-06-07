use std::collections::HashMap;

use serde::Deserialize;

pub type TrainingDataRaw = HashMap<String, String>;

#[derive(Deserialize)]
pub struct Config {
    pub input_length: u32
}