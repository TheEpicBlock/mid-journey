
use futures::{future, stream, StreamExt};
use wgpu::{BindGroupDescriptor, BindGroupEntry, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor};

use crate::{color::{self, Color}, gpu::{init_gpu, GpuDeviceData}, input::Config, layer::{Layer, MainType, WeightsAndBiases}, misc::{ceil_div, size_of, SliceExtension}, training_data::{string_to_data, GpuInputData, TrainingData}};

pub async fn start_training(data: TrainingData, config: Config) {
    let gpu = init_gpu().await;

    // Init buffers for weights and biases
    let mut weights_and_biases = Vec::<Layer>::default();
    let mut prev_layer_size = config.input_length;
    for layer_size in config.layers.iter().copied() {
        weights_and_biases.push(Layer::create(prev_layer_size, layer_size, &gpu.device));
        prev_layer_size = layer_size;
    }

    eval_performance(data, &gpu, &config, &weights_and_biases).await;
}

pub fn calc_cost(expected: Color, actual: Color) -> MainType {
    let dl = actual.l - expected.l;
    let da = actual.a - expected.a;
    let db = actual.b - expected.b;
    return dl * dl + da * da + db * db;
}

pub async fn eval_performance(data: TrainingData, gpu: &GpuDeviceData, config: &Config, weights: &WeightsAndBiases) {
    let mut total_err = 0f64;
    let mut count = 0;
    let mut min = MainType::MAX;
    let mut max = MainType::MIN;

    stream::iter(data.checking).then(|data| async {
        let res = compute_forwards(data.0, gpu, config, weights).await;
        calc_cost(data.1, res)
    }).for_each(|result| {
        total_err += result as f64;
        count += 1;
        min = min.min(result);
        max = max.max(result);
        future::ready(())
    }).await;

    println!("Tested on {} datapoints. min/avg/max = ({}, {}, {})", count, min, total_err / (count as f64), max);
}

pub async fn compute_forwards(input: GpuInputData, gpu: &GpuDeviceData, config: &Config, weights: &WeightsAndBiases) -> color::Color {
    let buffer = |size, mapped| {
        gpu.device.create_buffer(&BufferDescriptor {
            label: None,
            size: size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: mapped
        })
    };

    // Initialize first buffers
    let input_buf = buffer(config.input_length * size_of::<MainType>(), true);
    {
        let mut input_data = input_buf.slice(..).get_mapped_range_mut();
        for (i, byte) in bytemuck::cast_slice(&input).iter().enumerate() {
            input_data[i] = *byte;
        }
    }
    input_buf.unmap();

    // Initialize intermediate/output buffers
    let mut buffers = Vec::new();
    buffers.push(input_buf);
    for layer in &config.layers {
        buffers.push(buffer(*layer * size_of::<MainType>(), false));
    }
    assert_eq!(config.layers.last(), Some(&3));

    // Output staging buffer
    let output_staging = gpu.device.create_buffer(&BufferDescriptor {
        label: None,
        size: (config.layers.last().unwrap() * size_of::<MainType>()),
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false
    });

    // Run stuff
    let mut command_encoder = gpu.device.create_command_encoder(&CommandEncoderDescriptor::default());
    for i in 0..config.layers.len() {
        let in_buf = &buffers[i];
        let out_buf = &buffers[i+1];
        let out_len = config.layers[i];
        let bind_group = gpu.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &gpu.compute_forwards.0,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(weights[i].compute_forwards_global_data.as_entire_buffer_binding())
                },
                BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(in_buf.as_entire_buffer_binding())
                },
                BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(weights[i].weights.as_entire_buffer_binding())
                },
                BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(weights[i].biases.as_entire_buffer_binding())
                },
                BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(out_buf.as_entire_buffer_binding())
                },
            ]
        });
        let mut pass_encoder = command_encoder.begin_compute_pass(&ComputePassDescriptor::default());
        pass_encoder.set_pipeline(&gpu.compute_forwards.1);
        pass_encoder.set_bind_group(0, &bind_group, &[]);
        pass_encoder.dispatch_workgroups(ceil_div(out_len as u32, 64), 0, 0);
    }
    command_encoder.copy_buffer_to_buffer(buffers.last().unwrap(), 0, &output_staging, 0, config.layers.last().unwrap() * size_of::<MainType>());
    gpu.queue.submit([command_encoder.finish()]);

    output_staging.slice(..).map_buffer(&gpu.device, wgpu::MapMode::Read).await.unwrap();
    let results: &[u8] = &output_staging.slice(..).get_mapped_range();
    let results: &[MainType] = bytemuck::cast_slice(results);

    color::Color::from(results)
}