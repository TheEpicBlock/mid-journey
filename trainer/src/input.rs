use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

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
    /// Size of the layer afterwards. Will be None iff this is the last layer
    pub next_size: Option<Size>,
}

impl Config {
    pub fn layers(&self) -> Vec<LayerConfig> {
        let mut output = Vec::with_capacity(self.layers.len());

        for (i, layer_size) in self.layers.iter().copied().enumerate() {
            output.push(LayerConfig {
                previous_size: if i == 0 { self.input_length } else { self.layers[i-1] },
                size: layer_size,
                next_size: self.layers.get(i + 1).copied()
            });
        }

        return output;
    }

    /// The number of layers in the neural network.
    /// Does not count the input layer, but does counts the output layer.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

pub type JsonNetworkParameters = Vec<JsonNetworkLayer>;

#[derive(Serialize, Deserialize)]
pub struct JsonNetworkLayer {
    pub weights: Vec<MainType>,
    pub biases: Vec<MainType>,
}