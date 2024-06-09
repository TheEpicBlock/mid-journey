
use wgpu::{BindGroup, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, RenderBundleEncoderDescriptor};

use crate::{color::Color, gpu::{init_gpu, GpuDeviceData}, input::Config, layer::{LayerParameters, LayerValues, MainType, WeightsAndBiases}, misc::{bind_group, size_of, SliceExtension}, shaders::{ComputePassExt, ShaderSet}, training_data::{DataSet, GpuInputData, TrainingData}};

pub async fn start_training(data: TrainingData, config: Config) {
    let gpu = init_gpu().await;

    // Init buffers for weights and biases
    let mut network_parameters = Vec::<LayerParameters>::default();
    for layer in config.layers() {
        network_parameters.push(LayerParameters::create(layer.previous_size, layer.size, &gpu.device));
    }

    assert!(data.training.len() > 0, "No training data");

    let resources = TrainingResources::init(&gpu, config, &network_parameters, &data.training);

    gpu.device.start_capture();
    run_training_step(&gpu, &resources);
    gpu.device.stop_capture();

    for _ in 0..99 {
        run_training_step(&gpu, &resources);
    }
    
    gpu.device.poll(wgpu::MaintainBase::Wait);
}

fn run_training_step(gpu: &GpuDeviceData, resources: &TrainingResources) {
    let config = &resources.config;
    let eval_resources = &resources.eval_resources;
    let shaders = &eval_resources.shaders;

    let mut commands = gpu.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("Performance evaluation") });

    // Run the NN forwards on the data
    for layer in 0..config.num_layers() {
        let mut pass = commands.begin_compute_pass(&Default::default());
        pass.set_pipeline(&shaders[layer].compute_forwards);
        pass.set_bind_group(0, &eval_resources.bind_groups[layer], &[]);
        pass.dispatch(eval_resources.invocations, config.layers()[layer]);
    }

    // Run the backpropagation steps
    for layer in (0..config.num_layers()).rev() {
        let mut pass = commands.begin_compute_pass(&Default::default());
        pass.set_pipeline(&shaders[layer].backpropagation);
        pass.set_bind_group(0, &resources.backprop_bind_groups[layer], &[]);
        pass.dispatch(eval_resources.invocations, config.layers()[layer]);
    }

    // Submit everything
    gpu.queue.submit([commands.finish()]);
}

pub fn calc_cost(expected: Color, actual: Color) -> MainType {
    let dl = actual.l - expected.l;
    let da = actual.a - expected.a;
    let db = actual.b - expected.b;
    return dl * dl + da * da + db * db;
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
        pass.set_pipeline(&resources.shaders[layer].compute_forwards);
        pass.set_bind_group(0, &resources.bind_groups[layer], &[]);
        pass.dispatch(resources.invocations, config.layers()[layer]);
    }
    let output = resources.a_buffers.read_output(gpu, &mut commands);
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
        
        Iterator::zip(data.iter().map(|data| data.1), outputs)
            .for_each(|(expected, nn_output)| {
                let cost = calc_cost(expected, *nn_output);
                total_cost += cost as f64;
                count += 1;
                min = min.min(cost);
                max = max.max(cost);
            });
    }
    output_buf.unmap();
    assert_eq!(count, data.len());

    println!("Tested on {} datapoints. min/avg/max = ({}, {}, {})", count, min, total_cost / (count as f64), max);
}

/// Resources used during training. Superset of resources used during evals.
struct TrainingResources {
    config: Config,
    deriv_z_buffers: LayerValues,
    eval_resources: EvalResources,
    backprop_bind_groups: Vec<BindGroup>
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
        for layer in 0..(config.num_layers() - 1) {
            bind_groups.push(gpu.device.create_bind_group(&bind_group! {
                &gpu.shader_components.backpropagation.0,
                0 => &parameters[layer + 1].weights,
                1 => &eval_resources.z_buffers.buffers[layer],
                2 => &deriv_z_buffers.buffers[layer + 1],
                3 => &deriv_z_buffers.buffers[layer],
            }));
        }
        let last_layer_index = config.num_layers() - 1;
        bind_groups.push(gpu.device.create_bind_group(&bind_group! {
            &gpu.shader_components.backpropagation_start.0,
            0 => &eval_resources.a_buffers.buffers[last_layer_index],
            1 => &eval_resources.z_buffers.buffers[last_layer_index],
            2 => &expected_values_buf,
            3 => &deriv_z_buffers.buffers[last_layer_index]
        }));

        Self {
            config,
            deriv_z_buffers,
            eval_resources,
            backprop_bind_groups: bind_groups,
        }
    }
}

/// Resources used during evaluations
struct EvalResources {
    invocations: usize,
    a_buffers: LayerValues,
    z_buffers: LayerValues,
    shaders: Vec<ShaderSet>,
    /// Bind groups for each of the `compute_forwards` invocations
    bind_groups: Vec<BindGroup>,
}

impl EvalResources {
    fn init(gpu: &GpuDeviceData, config: &Config, parameters: &WeightsAndBiases, data: &DataSet) -> Self {
        let invocations = data.len();
        let z_buffers = LayerValues::create(gpu, config, invocations);
        let a_buffers = LayerValues::create_with_input(gpu, config, invocations, |gpu_input| {
            Iterator::zip(
                data.iter().map(|data| &data.0),
                bytemuck::cast_slice_mut::<_, MainType>(gpu_input).chunks_mut(config.input_length as usize)
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