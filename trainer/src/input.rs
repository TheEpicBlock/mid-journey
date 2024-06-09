use std::collections::{BTreeMap, HashMap};

use serde::Deserialize;

use crate::layer::{MainType, Size};

pub type TrainingDataRaw = BTreeMap<String, String>;

#[derive(Deserialize)]
pub struct Config {
    pub input_length: Size,
    pub percentage_training: f64,
    layers: Vec<Size>,
    pub letter_mapping: HashMap<char, MainType>,
}

#[derive(Clone, Copy)]
pub struct LayerConfig {
    /// Size of the preceeding layer
    pub previous_size: Size,
    pub size: Size,
}

impl Config {
    pub fn layers(&self) -> Vec<LayerConfig> {
        let mut previous_size = self.input_length;
        let mut output = Vec::with_capacity(self.layers.len());

        for layer_size in self.layers.iter().copied() {
            output.push(LayerConfig {
                previous_size,
                size: layer_size,
            });
            previous_size = layer_size;
        }

        return output;
    }

    /// The number of layers in the neural network.
    /// Does not count the input layer, but does counts the output layer.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}