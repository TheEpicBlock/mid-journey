use std::collections::HashMap;

use wgpu::{core::device, BindGroupLayout, CommandEncoder, ComputePass, ComputePipeline, ComputePipelineDescriptor, Device, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule};

use map_macro::hash_map;

use crate::{gpu::GpuDeviceData, input::{Config, LayerConfig}, misc::{bind_group_layout, ceil_div, floor_div, IterPow2}};

macro_rules! include_shader_str {
    ($($token:tt)*) => {
        concat!(include_str!("lib.wgsl"), include_str!($($token)*))
    };
}

macro_rules! include_shader {
    ($($token:tt)*) => {
        wgpu::ShaderModuleDescriptor {
            label: Some($($token)*),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_shader_str!($($token)*)))
        }
    };
}

pub fn create_pipeline(device: &Device, component: &ShaderComponent, name: &str, entrypoint: &str, constants: HashMap<String, f64>) -> ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(name),
        layout: Some(&device.create_pipeline_layout(
            &PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[ &component.0 ],
                push_constant_ranges: &[]
            }
        )),
        module: &component.1,
        entry_point: entrypoint,
        compilation_options: PipelineCompilationOptions {
            constants: &constants,
            zero_initialize_workgroup_memory: false,
        },
    })
}

pub struct ShaderComponent(pub BindGroupLayout, ShaderModule);

pub struct ShaderComponents {
    pub compute_forwards: ShaderComponent,
    pub backpropagation_start: ShaderComponent,
    pub backpropagation: ShaderComponent,
}

pub struct ShaderSet {
    pub compute_forwards: StandardShaderPipeline,
    /// Will be either `backpropagation_start` or `backpropagation` depending on if this is the final layer or not
    pub backpropagation: StandardShaderPipeline,
    pub apply_backprop_biases: BackpropApplyBiasShaderPipeline,
    pub apply_backprop_weights: BackpropApplyWeightShaderPipeline,
}

fn compute_forwards(device: &Device) -> ShaderComponent {
    let bind_group_layout = device.create_bind_group_layout(&bind_group_layout![
        { binding: 0, read_only: true },
        { binding: 1, read_only: true },
        { binding: 2, read_only: true },
        { binding: 3, read_only: false },
        { binding: 4, read_only: false },
    ]);

    let module = device.create_shader_module(include_shader!("compute_forwards.wgsl"));

    ShaderComponent(bind_group_layout, module)
}

fn backpropation_start(device: &Device) -> ShaderComponent {
    let bind_group_layout = device.create_bind_group_layout(&bind_group_layout![
        { binding: 0, read_only: true },
        { binding: 1, read_only: true },
        { binding: 2, read_only: true },
        { binding: 3, read_only: false },
    ]);

    let module = device.create_shader_module(include_shader!("backpropagation_start.wgsl"));

    ShaderComponent(bind_group_layout, module)
}

fn backpropation(device: &Device) -> ShaderComponent {
    let bind_group_layout = device.create_bind_group_layout(&bind_group_layout![
        { binding: 0, read_only: true },
        { binding: 1, read_only: true },
        { binding: 2, read_only: true },
        { binding: 3, read_only: false },
    ]);

    let module = device.create_shader_module(include_shader!("backpropagation.wgsl"));

    ShaderComponent(bind_group_layout, module)
}

fn apply_backprop_biases(device: &Device, workers_per_node: usize) -> ShaderComponent {
    let bind_group_layout = device.create_bind_group_layout(&bind_group_layout![
        { binding: 0, read_only: true },
        { binding: 1, read_only: false },
    ]);

    // Pipeline overridable constants at home
    let module = device.create_shader_module(
        wgpu::ShaderModuleDescriptor {
            label: Some("apply_backprop_biases.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                std::borrow::Cow::Owned(
                    include_shader_str!("apply_backprop_biases.wgsl").replace("${workers_per_node}", &workers_per_node.to_string())
                )
            )
        }
    );

    ShaderComponent(bind_group_layout, module)
}

fn apply_backprop_weights(device: &Device, workers_per_node: usize) -> ShaderComponent {
    let bind_group_layout = device.create_bind_group_layout(&bind_group_layout![
        { binding: 0, read_only: true },
        { binding: 1, read_only: true },
        { binding: 2, read_only: false },
    ]);

    // Pipeline overridable constants at home
    let module = device.create_shader_module(
        wgpu::ShaderModuleDescriptor {
            label: Some("apply_backprop_weights.wgsl"),
            source: wgpu::ShaderSource::Wgsl(
                std::borrow::Cow::Owned(
                    include_shader_str!("apply_backprop_weights.wgsl").replace("${workers_per_node}", &workers_per_node.to_string())
                )
            )
        }
    );

    ShaderComponent(bind_group_layout, module)
}

impl ShaderComponents {
    pub fn init(device: &Device) -> Self {
        Self {
            compute_forwards: compute_forwards(device),
            backpropagation_start: backpropation_start(device),
            backpropagation: backpropation(device),
        }
    }
}

impl ShaderSet {
    /// Compiles a set of shaders, the shaders are designed to
    /// evaluate multiple neural networks at once
    pub fn compile(gpu: &GpuDeviceData, config: &Config, invocations: usize) -> Vec<Self> {
        // Estimate the number of computations the shader does for each possible option and take the minimum
        // (we consider only powers of two as options)
        let workers_per_node = IterPow2::range(..=invocations)
            .filter(|n| *n <= gpu.device.limits().max_compute_workgroup_size_x as usize)
            .min_by_key(|n| (floor_div(invocations, *n) + (invocations % n) + n)).unwrap();

        // These shaders are a little special, they have some hacks around the lack of good pipeline overridable constant support
        let apply_backprop_biases = apply_backprop_biases(&gpu.device, workers_per_node);
        let apply_backprop_weights = apply_backprop_weights(&gpu.device, workers_per_node);

        let mut output = Vec::new();

        for (i, layer) in config.layers().into_iter().enumerate() {
            output.push(ShaderSet::compile_single(gpu, layer, invocations, i == config.layers().len() - 1, (&apply_backprop_biases, &apply_backprop_weights)));
        }

        return output;
    }

    fn compile_single(gpu: &GpuDeviceData, layer: LayerConfig, invocations: usize, final_layer: bool, apply_backprops: (&ShaderComponent, &ShaderComponent)) -> Self {
        let device = &gpu.device;
        let components = &gpu.shader_components;

        let compute_forwards = 
        create_pipeline(
            device,
            &components.compute_forwards,
            "Compute Forwards",
            "compute_forwards",
            hash_map! {
                "input_size".to_owned() => layer.previous_size as f64,
                "output_size".to_owned() => layer.size as f64,
                "invocations".to_owned() => invocations as f64,
            }
        );

        let backpropagation = if final_layer {
            create_pipeline(
                device,
                &components.backpropagation_start,
                "Backpropagation First Step",
                "backprop_from_cost",
                hash_map! {
                    "layer_size".to_owned() => layer.size as f64,
                    "invocations".to_owned() => invocations as f64,
                }
            )
        } else {
            create_pipeline(
                device,
                &components.backpropagation,
                "Backpropagation",
                "backprop_from_layer",
                hash_map! {
                    "layer_size".to_owned() => layer.size as f64,
                    "next_layer_size".to_owned() => layer.next_size.unwrap() as f64,
                    "invocations".to_owned() => invocations as f64,
                }
            )
        };

        let apply_backprop_biases = create_pipeline(
            device,
            &apply_backprops.0,
            "Backprop apply biases",
            "apply_biases",
            hash_map! {
                "layer_size".to_owned() => layer.size as f64,
                "invocations".to_owned() => invocations as f64,
            }
        );

        let apply_backprop_weights = create_pipeline(
            device,
            &apply_backprops.1,
            "Backprop apply weights",
            "apply_weights",
            hash_map! {
                "previous_layer_size".to_owned() => layer.previous_size as f64,
                "layer_size".to_owned() => layer.size as f64,
                "invocations".to_owned() => invocations as f64,
            }
        );

        Self {
            compute_forwards: StandardShaderPipeline {
                pipeline: compute_forwards,
                invocations: invocations as u32,
                layer_size: layer.size as u32,
            },
            backpropagation: StandardShaderPipeline {
                pipeline: backpropagation,
                invocations: invocations as u32,
                layer_size: layer.size as u32,
            },
            apply_backprop_biases: BackpropApplyBiasShaderPipeline {
                pipeline: apply_backprop_biases,
                layer_size: layer.size as u32,
            },
            apply_backprop_weights: BackpropApplyWeightShaderPipeline {
                pipeline: apply_backprop_weights,
                prev_layer_size: layer.previous_size as u32,
                layer_size: layer.size as u32,
            }
        }
    }
}

// Constants here should match the ones in lib.wgsl
const STD_WORKGROUP_SIZE: (u64, u64, u64) = (32, 2, 1);

pub struct StandardShaderPipeline {
    pipeline: ComputePipeline,
    invocations: u32,
    layer_size: u32,
}

impl StandardShaderPipeline {
    pub fn setup_pass<'a, 'b: 'a>(&'b self, pass: &mut ComputePass<'a>) {
        pass.set_pipeline(&self.pipeline);
        pass.dispatch_workgroups(
            ceil_div(self.invocations, STD_WORKGROUP_SIZE.0),
            ceil_div(self.layer_size, STD_WORKGROUP_SIZE.1),
            ceil_div(1, STD_WORKGROUP_SIZE.2)
        )
    }
}

pub struct BackpropApplyBiasShaderPipeline {
    pipeline: ComputePipeline,
    layer_size: u32,
}

impl BackpropApplyBiasShaderPipeline {
    pub fn setup_pass<'a, 'b: 'a>(&'b self, pass: &mut ComputePass<'a>) {
        pass.set_pipeline(&self.pipeline);
        // Should match constant in `apply_backprop_biases.wgsl`
        const WORKGROUP_SIZE: u32 = 8;
        pass.dispatch_workgroups(
            1,
            ceil_div(self.layer_size, WORKGROUP_SIZE as u64),
            1,
        )
    }

    pub fn get_layout(&self) -> BindGroupLayout{
        self.pipeline.get_bind_group_layout(0)
    }
}

pub struct BackpropApplyWeightShaderPipeline {
    pipeline: ComputePipeline,
    prev_layer_size: u32,
    layer_size: u32,
}

impl BackpropApplyWeightShaderPipeline {
    pub fn setup_pass<'a, 'b: 'a>(&'b self, pass: &mut ComputePass<'a>) {
        pass.set_pipeline(&self.pipeline);
        // Should match constant in `apply_backprop_weights.wgsl`
        const WORKGROUP_SIZE_A: u32 = 8;
        const WORKGROUP_SIZE_B: u32 = 1;
        pass.dispatch_workgroups(
            1,
            ceil_div(self.prev_layer_size, WORKGROUP_SIZE_A as u64),
            ceil_div(self.layer_size, WORKGROUP_SIZE_B as u64),
        )
    }

    pub fn get_layout(&self) -> BindGroupLayout{
        self.pipeline.get_bind_group_layout(0)
    }
}
