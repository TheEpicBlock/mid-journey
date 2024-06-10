use std::ops::{Add, Range, RangeTo};

use num_traits::One;
use wgpu::{BufferSlice, MapMode, BufferAsyncError, Device};

pub trait SliceExtension {
    async fn map_buffer(&self, device: &Device, mode: MapMode) -> Result<(), BufferAsyncError>;
}

impl SliceExtension for BufferSlice<'_> {
    async fn map_buffer(&self, device: &Device, mode: MapMode) -> Result<(), BufferAsyncError> {
        let (tx, rx) = futures_channel::oneshot::channel::<Result<(), BufferAsyncError>>();
        self.map_async(mode, |result| {
            tx.send(result).expect("Receiver should never be dropped");
        });
        while !device.poll(wgpu::MaintainBase::Poll).is_queue_empty() {
            tokio::task::yield_now().await;
        }
        device.poll(wgpu::MaintainBase::Wait);

        return rx.await.expect("Sender should never be dropper");
    }
}

pub fn size_of<T>() -> u64 {
    std::mem::size_of::<T>() as u64
}

pub fn ceil_div(a: u32, b: u64) -> u32 {
    (a as f64 / b as f64).ceil() as u32
}

pub fn floor_div(a: usize, b: usize) -> usize {
    (a as f64 / b as f64).floor() as usize
}

macro_rules! bind_group_layout {
    ($({ binding: $index:expr, read_only: $read_only:expr }),+$(,)?) => {
        wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                $(
                    wgpu::BindGroupLayoutEntry {
                        binding: $index,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: $read_only },
                            has_dynamic_offset: false,
                            min_binding_size: None
                        },
                        count: None,
                    }
                ),+
            ]
        }
    }
}

macro_rules! bind_group {
    ($layout:expr, $($index:expr => $buffer:expr),+$(,)?) => {
        wgpu::BindGroupDescriptor {
            label: None,
            layout: $layout,
            entries: &[
                $(
                    wgpu::BindGroupEntry {
                        binding: $index,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: $buffer,
                            offset: 0,
                            size: None,
                        })
                    }
                ),+
            ]
        }
    };
}

pub(crate) use bind_group;
pub(crate) use bind_group_layout;

pub struct IterPow2<T> {
    current: T,
    target: T,
}

impl <T> Iterator for IterPow2<T> where T: Add<Output = T> + Ord + Copy {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.target {
            return None;
        }

        let prev = self.current;
        self.current = self.current + self.current;
        return Some(prev);
    }
}

impl<T> IterPow2<T> where T: One {
    pub fn range(r: RangeTo<T>) -> Self {
        Self {
            current: T::one(),
            target: r.end,
        }
    }
}