
use wgpu::{BindGroup, BufferDescriptor, BufferUsages, CommandEncoderDescriptor};

use crate::{color::Color, gpu::GpuDeviceData, input::Config, layer::{LayerParameters, LayerValues, MainType, WeightsAndBiases}, misc::{bind_group, size_of, SliceExtension}, shaders::ShaderSet, string::string_to_data, training_data::{DataSet, TrainingData}};

pub async fn train_nn(gpu: &GpuDeviceData, data: TrainingData, config: Config) -> WeightsAndBiases {
    // Init buffers for weights and biases
    let mut network_parameters = Vec::<LayerParameters>::default();
    for layer in config.layers() {
        network_parameters.push(LayerParameters::create(layer.previous_size, layer.size, &gpu.device));
    }

    assert!(data.training.len() > 0, "No training data");

    eval_performance(&data.training, &gpu, &config, &network_parameters).await;

    let resources = TrainingResources::init(&gpu, config, &network_parameters, &data.training);

    for _ in 0..10 {
        for _ in 0..500 {
            run_training_step(&gpu, &resources);
        }
    
        gpu.device.poll(wgpu::MaintainBase::Wait);

        eval_performance(&data.training, &gpu, &resources.config, &network_parameters).await;
    }
    gpu.device.poll(wgpu::MaintainBase::Wait);
    gpu.device.start_capture();
    run_training_step(&gpu, &resources);
    gpu.device.stop_capture();

    gpu.device.poll(wgpu::MaintainBase::Wait);

    return network_parameters;
}

fn run_training_step(gpu: &GpuDeviceData, resources: &TrainingResources) {
    let config = &resources.config;
    let eval_resources = &resources.eval_resources;
    let shaders = &eval_resources.shaders;

    let mut commands = gpu.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("Performance evaluation") });

    // Run the NN forwards on the data
    for layer in 0..config.num_layers() {
        let mut pass = commands.begin_compute_pass(&Default::default());
        pass.set_bind_group(0, &eval_resources.bind_groups[layer], &[]);
        shaders[layer].compute_forwards.setup_pass(&mut pass);
    }

    // Run the backpropagation steps
    for layer in (0..config.num_layers()).rev() {
        let mut pass = commands.begin_compute_pass(&Default::default());
        pass.set_bind_group(0, &resources.backprop_bind_groups[layer], &[]);
        shaders[layer].backpropagation.setup_pass(&mut pass);
    }

    // Apply backpropagation steps
    for layer in 0..config.num_layers() {
        let mut pass = commands.begin_compute_pass(&Default::default());
        pass.set_bind_group(0, &resources.backprop_bias_apply_bind_groups[layer], &[]);
        shaders[layer].apply_backprop_biases.setup_pass(&mut pass);
        drop(pass);
        let mut pass = commands.begin_compute_pass(&Default::default());
        pass.set_bind_group(0, &resources.backprop_weight_apply_bind_groups[layer], &[]);
        shaders[layer].apply_backprop_weights.setup_pass(&mut pass);
    }

    // Submit everything
    gpu.queue.submit([commands.finish()]);
}

pub fn calc_cost(expected: Color, actual: Color) -> MainType {
    let dl = actual.l - expected.l;
    let da = actual.a - expected.a;
    let db = actual.b - expected.b;
    return (dl * dl + da * da + db * db) / 3.0;
}

pub async fn eval_single(data: &str, gpu: &GpuDeviceData, config: &Config, resources: &EvalResources) -> Color {
    let data = string_to_data(data, config);
    let input = resources.a_buffers.buffers.first().unwrap();
    gpu.queue.write_buffer(input, 0, bytemuck::cast_slice(&data));

    // Run the NN forwards on the data
    let mut commands = gpu.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("Performance evaluation") });
    for layer in 0..config.num_layers() {
        let mut pass = commands.begin_compute_pass(&Default::default());
        pass.set_bind_group(0, &resources.bind_groups[layer], &[]);
        resources.shaders[layer].compute_forwards.setup_pass(&mut pass);
    }
    let output = resources.a_buffers.read_output(gpu, &mut commands);
    gpu.queue.submit([commands.finish()]);

    let output_color;
    let output_buf = &output;
    {
        output_buf.slice(..).map_buffer(&gpu.device, wgpu::MapMode::Read).await.unwrap();
        let outputs = output_buf.slice(..).get_mapped_range();
        let outputs: &[Color] = bytemuck::cast_slice(&outputs);
        output_color = outputs[0];
    }
    output_buf.unmap();
    return output_color;
}

pub async fn eval_performance(data: &DataSet, gpu: &GpuDeviceData, config: &Config, parameters: &WeightsAndBiases) {
    if data.is_empty() {
        println!("Can't run performance evaluation. No data was marked to be used for checking");
        return;
    }

    let resources = EvalResources::init(gpu, config, parameters, data);

    // Run the NN forwards on the data
    let mut commands = gpu.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("Performance evaluation") });
    for layer in 0..config.num_layers() {
        let mut pass = commands.begin_compute_pass(&Default::default());
        pass.set_bind_group(0, &resources.bind_groups[layer], &[]);
        resources.shaders[layer].compute_forwards.setup_pass(&mut pass);
    }
    let output = resources.a_buffers.read_output(gpu, &mut commands);
    gpu.queue.submit([commands.finish()]);

    // Process NN results!
    let mut total_cost = 0f64;
    let mut count = 0;
    let mut min = MainType::MAX;
    let mut max = MainType::MIN;
    let white = Color::from_rgb((0.0, 0.0, 0.0));
    let mut vmin = MainType::MAX;
    let mut vmax = MainType::MIN;
    
    let output_buf = &output;
    {
        output_buf.slice(..).map_buffer(&gpu.device, wgpu::MapMode::Read).await.unwrap();
        let outputs = output_buf.slice(..).get_mapped_range();
        let outputs: &[Color] = bytemuck::cast_slice(&outputs);
        
        Iterator::zip(data.iter().map(|data| data.1), outputs)
            .for_each(|(expected, nn_output)| {
                let cost = calc_cost(expected, *nn_output);
                total_cost += cost as f64;
                count += 1;
                min = min.min(cost);
                max = max.max(cost);
                let variance = calc_cost(white, *nn_output);
                vmin = vmin.min(variance);
                vmax = vmax.max(variance);
            });
    }
    output_buf.unmap();
    assert_eq!(count, data.len());

    println!("Tested on {} datapoints. min/avg/max vmin/vmax = ({}, {}, {}) ({}, {})", count, min, total_cost / (count as f64), max, vmin, vmax);
}

/// Resources used during training. Superset of resources used during evals.
struct TrainingResources {
    config: Config,
    deriv_z_buffers: LayerValues,
    eval_resources: EvalResources,
    backprop_bind_groups: Vec<BindGroup>,
    backprop_bias_apply_bind_groups: Vec<BindGroup>,
    backprop_weight_apply_bind_groups: Vec<BindGroup>,
}

impl TrainingResources {
    fn init(gpu: &GpuDeviceData, config: Config, parameters: &WeightsAndBiases, data: &DataSet) -> Self {
        let invocations = data.len();
        // Resources needed to run the nn on the `training` dataset
        let eval_resources = EvalResources::init(gpu, &config, parameters, &data);

        let deriv_z_buffers = LayerValues::create(&gpu, &config, invocations);
        let expected_values_buf = gpu.device.create_buffer(&BufferDescriptor {
            label: Some("nn expected outputs"),
            size: config.layers().last().unwrap().size * invocations as u64 * size_of::<MainType>(),
            usage: BufferUsages::STORAGE,
            mapped_at_creation: true
        });
        {
            // Copy all of the expected values into the buffer
            let expected_values = &mut expected_values_buf.slice(..).get_mapped_range_mut();
            let expected_values: &mut [Color] = bytemuck::cast_slice_mut(expected_values);
            Iterator::zip(expected_values.iter_mut(), data).for_each(|(gpu_entry, data_entry)| *gpu_entry = data_entry.1);
        }
        expected_values_buf.unmap();

        let mut bind_groups = Vec::new();
        // Create bind groups for all but the final layer
        // (the layers don't include the input layer, but some of the buffers do. That's why the indexing is so awkward)
        for layer in 0..(config.num_layers() - 1) {
            bind_groups.push(gpu.device.create_bind_group(&bind_group! {
                &gpu.shader_components.backpropagation.0,
                0 => &parameters[layer + 1].weights,
                1 => &eval_resources.z_buffers.buffers[layer + 1],
                2 => &deriv_z_buffers.buffers[layer + 2],
                3 => &deriv_z_buffers.buffers[layer + 1],
            }));
        }
        // Create the bind groups for the final layer
        let last_layer_index = config.num_layers() - 1;
        bind_groups.push(gpu.device.create_bind_group(&bind_group! {
            &gpu.shader_components.backpropagation_start.0,
            0 => &eval_resources.a_buffers.buffers[last_layer_index+1],
            1 => &eval_resources.z_buffers.buffers[last_layer_index+1],
            2 => &expected_values_buf,
            3 => &deriv_z_buffers.buffers[last_layer_index+1]
        }));

        // Create other bind groups
        let mut bias_apply_bind_groups = Vec::new();
        for layer in 0..config.num_layers() {
            bias_apply_bind_groups.push(gpu.device.create_bind_group(&bind_group! {
                &eval_resources.shaders[layer].apply_backprop_biases.get_layout(),
                0 => &deriv_z_buffers.buffers[layer + 1],
                1 => &parameters[layer].biases,
            }));
        }
        let mut weight_apply_bind_groups = Vec::new();
        for layer in 0..config.num_layers() {
            weight_apply_bind_groups.push(gpu.device.create_bind_group(&bind_group! {
                &eval_resources.shaders[layer].apply_backprop_weights.get_layout(),
                0 => &eval_resources.a_buffers.buffers[layer],
                1 => &deriv_z_buffers.buffers[layer + 1],
                2 => &parameters[layer].weights,
            }));
        }

        Self {
            config,
            deriv_z_buffers,
            eval_resources,
            backprop_bind_groups: bind_groups,
            backprop_bias_apply_bind_groups: bias_apply_bind_groups,
            backprop_weight_apply_bind_groups: weight_apply_bind_groups,
        }
    }
}

/// Resources used during evaluations
pub struct EvalResources {
    invocations: usize,
    a_buffers: LayerValues,
    z_buffers: LayerValues,
    shaders: Vec<ShaderSet>,
    /// Bind groups for each of the `compute_forwards` invocations
    bind_groups: Vec<BindGroup>,
}

impl EvalResources {
    pub fn init(gpu: &GpuDeviceData, config: &Config, parameters: &WeightsAndBiases, data: &DataSet) -> Self {
        assert!(!data.is_empty());

        let invocations = data.len();
        let z_buffers = LayerValues::create(gpu, config, invocations);
        let a_buffers = LayerValues::create_with_input(gpu, config, invocations, |gpu_input| {
            Iterator::zip(
                data.iter().map(|data| &data.0),
                bytemuck::cast_slice_mut::<_, MainType>(gpu_input).chunks_exact_mut(config.input_length() as usize)
            ).for_each(|(input_data, gpu_value)| {
                gpu_value.copy_from_slice(input_data);
            });
        });
        let shaders = ShaderSet::compile(&gpu, config, invocations);

        let mut bind_groups = Vec::new();
        for layer in 0..config.num_layers() {
            bind_groups.push(gpu.device.create_bind_group(&bind_group! {
                &gpu.shader_components.compute_forwards.0,
                0 => &parameters[layer].weights,
                1 => &parameters[layer].biases,
                2 => &a_buffers.buffers[layer],
                3 => &z_buffers.buffers[layer + 1],
                4 => &a_buffers.buffers[layer + 1],
            }));
        }

        Self {
            invocations: data.len(),
            a_buffers,
            z_buffers,
            shaders,
            bind_groups,
        }
    }
}