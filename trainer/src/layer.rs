use std::ops::Deref;

use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Features};

use crate::{gpu::GpuDeviceData, input::Config, misc::size_of};

// Should match compute_forwards.wgsl
pub type MainType = f32;

/// Type for layer sizes
pub type Size = u64;

pub type WeightsAndBiases = Vec<LayerParameters>;

pub struct LayerValues {
    pub buffers: Vec<Buffer>,
    /// In bytes
    size_of_last_layer: Size,
}

pub struct LayerParameters {
    pub weights: Buffer,
    pub biases: Buffer,
}

impl LayerParameters {
    pub fn create(prev_size: Size, size: Size, device: &Device) -> Self {
        let weights = device.create_buffer(&BufferDescriptor {
            label: Some("nn layer weight"),
            size: prev_size * size * size_of::<MainType>(),
            usage: BufferUsages::STORAGE,
            mapped_at_creation: true
        });
        let biases = device.create_buffer(&BufferDescriptor {
            label: Some("nn layer biases"),
            size: size * size_of::<MainType>(),
            usage: BufferUsages::STORAGE,
            mapped_at_creation: true
        });
        weights.slice(..).get_mapped_range_mut().iter_mut().for_each(|b| *b = 0);
        biases.slice(..).get_mapped_range_mut().iter_mut().for_each(|b| *b = 0);

        weights.unmap();
        biases.unmap();

        LayerParameters {
            weights,
            biases,
        }
    }
}

impl LayerValues {
    pub fn create(gpu: &GpuDeviceData, config: &Config, invocations: usize) -> Self {
        LayerValues::create_with_input(gpu, config, invocations, |_|{})
    }

    pub fn create_with_input<F>(gpu: &GpuDeviceData, config: &Config, invocations: usize, initializer: F) -> Self
            where F: FnOnce(&mut [u8]) {
        let mut buffers = Vec::new();

        let input_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("nn layer inputs"),
            size: config.input_length * invocations as u64 * size_of::<MainType>(),
            usage: BufferUsages::STORAGE,
            mapped_at_creation: true
        });
        initializer(&mut input_buf.slice(..).get_mapped_range_mut());
        input_buf.unmap();
        buffers.push(input_buf);

        for (i, layer) in config.layers().iter().enumerate() {
            let mut usage = BufferUsages::STORAGE;
            if i == config.layers().len()-1 {
                usage |= BufferUsages::COPY_SRC;
            }
            buffers.push(gpu.device.create_buffer(&BufferDescriptor {
                label: Some("nn layer values"),
                size: layer.size * invocations as u64 * size_of::<MainType>(),
                usage,
                mapped_at_creation: false
            }));
        }

        Self {
            buffers,
            size_of_last_layer: config.layers().last().unwrap().size * invocations as u64 * size_of::<MainType>()
        }
    }

    pub fn read_output(&self, gpu: &GpuDeviceData, encoder: &mut CommandEncoder) -> LayerOutput<'_> {
        let output_buf = self.buffers.last().unwrap();
        if gpu.device.features().contains(Features::MAPPABLE_PRIMARY_BUFFERS) {
            return LayerOutput::Primary(output_buf);
        } else {
            let staging = gpu.device.create_buffer(&BufferDescriptor {
                label: Some("nn layer staging"),
                size: self.size_of_last_layer,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false
            });
            encoder.copy_buffer_to_buffer(output_buf, 0, &staging, 0, self.size_of_last_layer);
            return LayerOutput::Staging(staging);
        }
    }
}

pub enum LayerOutput<'a> {
    Primary(&'a Buffer),
    Staging(Buffer)
}

impl Deref for LayerOutput<'_> {
    type Target = Buffer;

    fn deref(&self) -> &Self::Target {
        match self {
            LayerOutput::Primary(x) => x,
            LayerOutput::Staging(x) => x,
        }
    }
}