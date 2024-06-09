use wgpu::{Adapter, Device, DeviceDescriptor, Queue};

use crate::shaders::ShaderComponents;

pub struct GpuDeviceData {
    pub device: Device,
    pub queue: Queue,
    pub shader_components: ShaderComponents
}

pub async fn init_gpu() -> GpuDeviceData {
    let adapter = init_adapter().await;
    println!("Using gpu adapter: {:?}", adapter.get_info());
    let (device, queue) = adapter.request_device(&DeviceDescriptor::default(), None).await.expect("Failed to open GPU");

    let shader_components = ShaderComponents::init(&device);

    GpuDeviceData {
        device,
        queue,
        shader_components,
    }
}

pub async fn init_adapter() -> Adapter {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        dx12_shader_compiler: Default::default(),
        flags: Default::default(),
        gles_minor_version: Default::default(),
    });

    return instance.request_adapter(&&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        compatible_surface: None,
    }).await.expect("Couldn't request WebGPU adapter. Please ensure WebGPU is available for your device");
}