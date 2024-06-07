use std::collections::{BTreeMap, HashMap};

use serde::Deserialize;

use crate::layer::{MainType, Size};

pub type TrainingDataRaw = BTreeMap<String, String>;

#[derive(Deserialize)]
pub struct Config {
    pub input_length: Size,
    pub percentage_training: f64,
    pub layers: Vec<Size>,
    pub letter_mapping: HashMap<char, MainType>,
}