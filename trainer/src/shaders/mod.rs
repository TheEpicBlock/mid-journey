use wgpu::{BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, ComputePipeline, Device, PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderStages};

macro_rules! include_shader {
    ($($token:tt)*) => {
        ShaderModuleDescriptor {
            label: Some($($token)*),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(concat!(include_str!("lib.wgsl"), include_str!($($token)*))))
        }
    };
}

pub struct Shader(pub BindGroupLayout, pub ComputePipeline);

pub fn compute_forwards(device: &Device) -> Shader {
    let bind_group_layout = device.create_bind_group_layout(
        &BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
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
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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

    let pipeline = device.create_compute_pipeline(
        &wgpu::ComputePipelineDescriptor {
            label: Some("Expand"),
            layout: Some(&device.create_pipeline_layout(
                &PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[]
                }
            )),
            module: &device.create_shader_module(include_shader!("compute_forwards.wgsl")),
            entry_point: "compute_forwards"
        }
    );

    return Shader(bind_group_layout, pipeline);
}