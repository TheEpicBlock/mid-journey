
use wgpu::CommandEncoderDescriptor;

use crate::{color::Color, gpu::{init_gpu, GpuDeviceData}, input::Config, layer::{LayerParameters, LayerValues, MainType, WeightsAndBiases}, misc::{bind_group, SliceExtension}, shaders::{ComputePassExt, ShaderSet}, training_data::TrainingData};

pub async fn start_training(data: TrainingData, config: Config) {
    let gpu = init_gpu().await;

    // Init buffers for weights and biases
    let mut weights_and_biases = Vec::<LayerParameters>::default();
    for layer in config.layers() {
        weights_and_biases.push(LayerParameters::create(layer.previous_size, layer.size, &gpu.device));
    }

    eval_performance(data, &gpu, &config, &weights_and_biases).await;
}

pub fn calc_cost(expected: Color, actual: Color) -> MainType {
    let dl = actual.l - expected.l;
    let da = actual.a - expected.a;
    let db = actual.b - expected.b;
    return dl * dl + da * da + db * db;
}

pub async fn eval_performance(data: TrainingData, gpu: &GpuDeviceData, config: &Config, parameters: &WeightsAndBiases) {

    // Create the shaders and buffers to run all of the data at once
    let invocations = data.checking.len();
    let shaders = ShaderSet::compile(gpu, config, data.checking.len());
    let z_values = LayerValues::create(gpu, config, invocations);
    let a_values = LayerValues::create_with_input(gpu, config, invocations, |gpu_input| {
        Iterator::zip(
            data.checking.iter().map(|data| &data.0),
            bytemuck::cast_slice_mut::<_, MainType>(gpu_input).chunks_mut(config.input_length as usize)
        ).for_each(|(input_data, gpu_value)| {
            gpu_value.copy_from_slice(input_data);
        });
    });

    let mut bind_groups = Vec::new();
    for layer in 0..config.num_layers() {
        bind_groups.push(gpu.device.create_bind_group(&bind_group! {
            &gpu.shader_components.compute_forwards.0,
            0 => &parameters[layer].weights,
            1 => &parameters[layer].biases,
            2 => &a_values.buffers[layer],
            3 => &z_values.buffers[layer + 1],
            4 => &a_values.buffers[layer + 1],
        }));
    }

    let mut commands = gpu.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("Performance evaluation") });
    for layer in 0..config.num_layers() {
        let mut pass = commands.begin_compute_pass(&Default::default());
        pass.set_pipeline(&shaders[layer].compute_forwards);
        pass.set_bind_group(0, &bind_groups[layer], &[]);
        pass.dispatch(invocations, config.layers()[layer]);
    }
    let output = a_values.read_output(gpu, &mut commands);
    gpu.queue.submit([commands.finish()]);

    // Process NN results!
    let mut total_cost = 0f64;
    let mut count = 0;
    let mut min = MainType::MAX;
    let mut max = MainType::MIN;
    
    let output_buf = &output;
    {
        output_buf.slice(..).map_buffer(&gpu.device, wgpu::MapMode::Read).await.unwrap();
        let outputs = output_buf.slice(..).get_mapped_range();
        let outputs: &[Color] = bytemuck::cast_slice(&outputs);
        
        Iterator::zip(data.checking.iter().map(|data| data.1), outputs)
            .for_each(|(expected, nn_output)| {
                let cost = calc_cost(expected, *nn_output);
                total_cost += cost as f64;
                count += 1;
                min = min.min(cost);
                max = max.max(cost);
            });
    }
    output_buf.unmap();
    assert_eq!(count, data.checking.len());

    println!("Tested on {} datapoints. min/avg/max = ({}, {}, {})", count, min, total_cost / (count as f64), max);
}