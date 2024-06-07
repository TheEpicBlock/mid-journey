use std::collections::BTreeMap;

use serde::Deserialize;

use crate::layer::Size;

pub type TrainingDataRaw = BTreeMap<String, String>;

#[derive(Deserialize)]
pub struct Config {
    pub input_length: Size,
    pub percentage_training: f64,
    pub layers: Vec<Size>
}