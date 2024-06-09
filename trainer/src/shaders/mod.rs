use wgpu::{BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, ComputePass, ComputePipeline, Device, PipelineCompilationOptions, PipelineLayoutDescriptor, ShaderModule, ShaderModuleDescriptor, ShaderStages};

use map_macro::hash_map;

use crate::{gpu::GpuDeviceData, input::{Config, LayerConfig}, misc::ceil_div};

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
}

pub struct ShaderSet {
    pub compute_forwards: ComputePipeline,
}

fn compute_forwards(device: &Device) -> ShaderComponent {
    let bind_group_layout = device.create_bind_group_layout(
        &BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None
                    },
                    count: None,
                },
            ]
        }
    );

    let module = device.create_shader_module(include_shader!("compute_forwards.wgsl"));

    ShaderComponent(bind_group_layout, module)
}

impl ShaderComponents {
    pub fn init(device: &Device) -> Self {
        Self {
            compute_forwards: compute_forwards(device)
        }
    }
}

impl ShaderSet {
    /// Compiles a set of shaders, the shaders are designed to
    /// evaluate multiple neural networks at once
    pub fn compile(gpu: &GpuDeviceData, config: &Config, invocations: usize) -> Vec<Self> {
        let mut output = Vec::new();

        for layer in config.layers() {
            output.push(ShaderSet::compile_single(gpu, layer, invocations));
        }

        return output;
    }

    fn compile_single(gpu: &GpuDeviceData, layer: LayerConfig, invocations: usize) -> Self {
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

        Self {
            compute_forwards
        }
    }
}

pub trait ComputePassExt {
    fn dispatch(&mut self, invocations: usize, layer: LayerConfig);
}

impl ComputePassExt for ComputePass<'_> {
    fn dispatch(&mut self, invocations: usize, layer: LayerConfig) {
        self.dispatch_workgroups(ceil_div(invocations as u32, 64), ceil_div(layer.size as u32, 64), 0)
    }
}