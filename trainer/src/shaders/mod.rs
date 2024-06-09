use wgpu::{BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, ComputePass, ComputePipeline, Device, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderStages};

use map_macro::hash_map;

use crate::{gpu::GpuDeviceData, input::{Config, LayerConfig}, misc::{bind_group_layout, ceil_div}};

macro_rules! include_shader {
    ($($token:tt)*) => {
        ShaderModuleDescriptor {
            label: Some($($token)*),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(concat!(include_str!("lib.wgsl"), include_str!($($token)*))))
        }
    };
}

pub struct ShaderComponent(pub BindGroupLayout, ShaderModule);

pub struct ShaderComponents {
    pub compute_forwards: ShaderComponent,
    pub backpropagation_start: ShaderComponent,
    pub backpropagation: ShaderComponent,
}

pub struct ShaderSet {
    pub compute_forwards: ComputePipeline,
    /// Will be either `backpropagation_start` or `backpropagation` depending on if this is the final layer or not
    pub backpropagation: ComputePipeline,
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
        let mut output = Vec::new();

        for (i, layer) in config.layers().into_iter().enumerate() {
            output.push(ShaderSet::compile_single(gpu, layer, invocations, i == config.layers().len() - 1));
        }

        return output;
    }

    fn compile_single(gpu: &GpuDeviceData, layer: LayerConfig, invocations: usize, final_layer: bool) -> Self {
        let device = &gpu.device;
        let components = &gpu.shader_components;

        let compute_forwards = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Compute Forwards"),
                layout: Some(&device.create_pipeline_layout(
                    &PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[ &components.compute_forwards.0 ],
                        push_constant_ranges: &[]
                    }
                )),
                module: &components.compute_forwards.1,
                entry_point: "compute_forwards",
                compilation_options: PipelineCompilationOptions {
                    constants: &hash_map! {
                        "input_size".to_owned() => layer.previous_size as f64,
                        "output_size".to_owned() => layer.size as f64,
                        "invocations".to_owned() => invocations as f64,
                    },
                    zero_initialize_workgroup_memory: false,
                },
            }
        );

        let backpropagation = if final_layer {
            device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("Backpropagation First Step"),
                    layout: Some(&device.create_pipeline_layout(
                        &PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[ &components.backpropagation_start.0 ],
                            push_constant_ranges: &[]
                        }
                    )),
                    module: &components.backpropagation_start.1,
                    entry_point: "backprop_from_cost",
                    compilation_options: PipelineCompilationOptions {
                        constants: &hash_map! {
                            "layer_size".to_owned() => layer.size as f64,
                            "invocations".to_owned() => invocations as f64,
                        },
                        zero_initialize_workgroup_memory: false,
                    },
                }
            )
        } else {
            device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("Backpropagation"),
                    layout: Some(&device.create_pipeline_layout(
                        &PipelineLayoutDescriptor {
                            label: None,
                            bind_group_layouts: &[ &components.backpropagation.0 ],
                            push_constant_ranges: &[]
                        }
                    )),
                    module: &components.backpropagation.1,
                    entry_point: "backprop_from_layer",
                    compilation_options: PipelineCompilationOptions {
                        constants: &hash_map! {
                            "layer_size".to_owned() => layer.size as f64,
                            "next_layer_size".to_owned() => layer.next_size.unwrap() as f64,
                            "invocations".to_owned() => invocations as f64,
                        },
                        zero_initialize_workgroup_memory: false,
                    },
                }
            )
        };

        Self {
            compute_forwards,
            backpropagation
        }
    }
}

pub trait ComputePassExt {
    fn dispatch(&mut self, invocations: usize, layer: LayerConfig);
}

impl ComputePassExt for ComputePass<'_> {
    fn dispatch(&mut self, invocations: usize, layer: LayerConfig) {
        self.dispatch_workgroups(ceil_div(invocations as u32, 64), ceil_div(layer.size as u32, 64), 1)
    }
}