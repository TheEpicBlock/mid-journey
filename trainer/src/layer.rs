use wgpu::{util::{BufferInitDescriptor, DeviceExt}, Buffer, BufferBinding, BufferDescriptor, BufferUsages, Device};

use crate::input::Config;

// Should match compute_forwards.wgsl
pub type MainType = f32;

/// Type for layer sizes
pub type Size = u64;

pub type WeightsAndBiases = Vec<Layer>;

pub struct Layer {
    pub weights: Buffer,
    pub biases: Buffer,
    pub compute_forwards_global_data: Buffer,
}

impl Layer {
    pub fn create(prev_size: Size, size: Size, device: &Device) -> Self {
        let weights = device.create_buffer(&BufferDescriptor {
            label: None,
            size: prev_size * size,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: true
        });
        let biases = device.create_buffer(&BufferDescriptor {
            label: None,
            size: prev_size * size,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: true
        });
        weights.slice(..).get_mapped_range_mut().iter_mut().for_each(|b| *b = 0);
        biases.slice(..).get_mapped_range_mut().iter_mut().for_each(|b| *b = 0);

        weights.unmap();
        biases.unmap();

        let compute_forwards = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[prev_size, size]),
            usage: BufferUsages::UNIFORM,
        });

        return Layer {
            weights,
            biases,
            compute_forwards_global_data: compute_forwards
        };
    }
}