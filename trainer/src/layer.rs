use std::ops::Deref;

use futures::{stream, try_join, StreamExt};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, CommandEncoderDescriptor, Device, Features};

use crate::{gpu::GpuDeviceData, input::{Config, JsonNetworkLayer, JsonNetworkParameters}, misc::{size_of, SliceExtension}};

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
        let mut usage = BufferUsages::STORAGE;

        if device.features().contains(Features::MAPPABLE_PRIMARY_BUFFERS) {
            usage |= BufferUsages::MAP_READ;
        } else {
            usage |= BufferUsages::COPY_SRC;
        }

        let weights = device.create_buffer(&BufferDescriptor {
            label: Some("nn layer weights"),
            size: prev_size * size * size_of::<MainType>(),
            usage,
            mapped_at_creation: true
        });
        let biases = device.create_buffer(&BufferDescriptor {
            label: Some("nn layer biases"),
            size: size * size_of::<MainType>(),
            usage,
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

    pub async fn to_json(&self, gpu: &GpuDeviceData) -> JsonNetworkLayer {
        let weights;
        let biases;

        if gpu.device.features().contains(Features::MAPPABLE_PRIMARY_BUFFERS) {
            weights = ReadableBuf::Primary(&self.weights);
            biases = ReadableBuf::Primary(&self.biases);
        } else {
            let weight_staging = gpu.device.create_buffer(&BufferDescriptor {
                label: Some("weight staging"),
                size: self.weights.size(),
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false
            });
            let bias_staging = gpu.device.create_buffer(&BufferDescriptor {
                label: Some("bias staging"),
                size: self.biases.size(),
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false
            });

            let mut command = gpu.device.create_command_encoder(&CommandEncoderDescriptor { label: None });
            command.copy_buffer_to_buffer(&self.weights, 0, &weight_staging, 0, self.weights.size());
            command.copy_buffer_to_buffer(&self.biases, 0, &bias_staging, 0, self.biases.size());
            gpu.queue.submit([command.finish()]);
            gpu.device.poll(wgpu::MaintainBase::Wait);

            weights = ReadableBuf::Staging(weight_staging);
            biases = ReadableBuf::Staging(bias_staging);
        }

        let weights_slice = weights.slice(..);
        let biases_slice = biases.slice(..);

        try_join!(
            weights_slice.map_buffer(&gpu.device, wgpu::MapMode::Read),
            biases_slice.map_buffer(&gpu.device, wgpu::MapMode::Read),
        ).unwrap();

        let weights_mapped = weights_slice.get_mapped_range();
        let weights_copy = Vec::from(bytemuck::cast_slice(&weights_mapped));
        let biases_mapped = biases_slice.get_mapped_range();
        let biases_copy = Vec::from(bytemuck::cast_slice(&biases_mapped));

        drop(weights_mapped);
        drop(biases_mapped);

        weights.unmap();
        biases.unmap();
        
        JsonNetworkLayer {
            weights: weights_copy,
            biases: biases_copy,
        }
    }
}

pub async fn to_json(parameters: WeightsAndBiases, gpu: &GpuDeviceData) -> JsonNetworkParameters {
    stream::iter(parameters).then(|l| async move { l.to_json(gpu).await }).collect().await
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
                if gpu.device.features().contains(Features::MAPPABLE_PRIMARY_BUFFERS) {
                    usage |= BufferUsages::MAP_READ;
                } else {
                    usage |= BufferUsages::COPY_SRC;
                }
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

    pub fn read_output(&self, gpu: &GpuDeviceData, encoder: &mut CommandEncoder) -> ReadableBuf<'_> {
        let output_buf = self.buffers.last().unwrap();
        if gpu.device.features().contains(Features::MAPPABLE_PRIMARY_BUFFERS) {
            return ReadableBuf::Primary(output_buf);
        } else {
            let staging = gpu.device.create_buffer(&BufferDescriptor {
                label: Some("nn layer staging"),
                size: self.size_of_last_layer,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false
            });
            encoder.copy_buffer_to_buffer(output_buf, 0, &staging, 0, self.size_of_last_layer);
            return ReadableBuf::Staging(staging);
        }
    }
}

pub enum ReadableBuf<'a> {
    Primary(&'a Buffer),
    Staging(Buffer)
}

impl Deref for ReadableBuf<'_> {
    type Target = Buffer;

    fn deref(&self) -> &Self::Target {
        match self {
            ReadableBuf::Primary(x) => x,
            ReadableBuf::Staging(x) => x,
        }
    }
}