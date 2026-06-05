use crate::graphics::{self, GraphicsContext, Texture};
use crate::world;
use anyhow::{Context, Result};
use ash::vk;
use crossbeam_channel::{Receiver, Sender};
use hecs::Entity;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub struct LoadedMeshData {
    pub vertex_buffer: vk::Buffer,
    pub vertex_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    pub index_memory: vk::DeviceMemory,
    pub index_count: u32,
    pub albedo_texture: Option<Texture>,
    pub aabb: graphics::occlusion::AABB,
}

pub enum ResourceLoadRequest {
    Mesh(Entity, world::Mesh),
    Texture(String, Vec<u8>),
}

pub enum ResourceLoadResult {
    Mesh(Entity, LoadedMeshData),
    Texture(String, Texture),
    Failed(ResourceLoadFailure),
}

pub struct ResourceLoadFailure {
    pub label: String,
    pub reason: String,
}

pub struct ResourceLoader {
    to_loader: Sender<ResourceLoadRequest>,
    from_loader: Receiver<ResourceLoadResult>,
    queued_count: Arc<AtomicUsize>,
}

impl ResourceLoader {
    pub fn new(graphics_context: Arc<GraphicsContext>) -> Self {
        let (to_loader, loader_rx) = crossbeam_channel::unbounded::<ResourceLoadRequest>();
        let (loader_tx, from_loader) = crossbeam_channel::unbounded::<ResourceLoadResult>();
        let queued_count = Arc::new(AtomicUsize::new(0));
        let worker_count = crate::runtime::background_worker_count();

        for worker_idx in 0..worker_count {
            let graphics_context = graphics_context.clone();
            let loader_rx = loader_rx.clone();
            let loader_tx = loader_tx.clone();
            let queued_count = queued_count.clone();
            let _ = std::thread::Builder::new()
                .name(format!("stfsc-loader-{worker_idx}"))
                .spawn(move || loop {
                    let Ok(request) = loader_rx.recv() else {
                        break;
                    };

                    queued_count.fetch_sub(1, Ordering::Relaxed);
                    let result = process_request(&graphics_context, request);
                    let _ = loader_tx.send(result);
                });
        }

        Self {
            to_loader,
            from_loader,
            queued_count,
        }
    }

    pub fn queue_mesh(&self, id: Entity, mesh: world::Mesh) {
        self.queued_count.fetch_add(1, Ordering::Relaxed);
        if self
            .to_loader
            .send(ResourceLoadRequest::Mesh(id, mesh))
            .is_err()
        {
            self.queued_count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    pub fn queue_texture(&self, texture_id: String, data: Vec<u8>) {
        self.queued_count.fetch_add(1, Ordering::Relaxed);
        if self
            .to_loader
            .send(ResourceLoadRequest::Texture(texture_id, data))
            .is_err()
        {
            self.queued_count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    pub fn poll_processed(&self) -> impl Iterator<Item = ResourceLoadResult> + '_ {
        self.from_loader.try_iter()
    }

    pub fn queued_count(&self) -> usize {
        self.queued_count.load(Ordering::Relaxed)
    }
}

fn process_request(
    graphics_context: &GraphicsContext,
    request: ResourceLoadRequest,
) -> ResourceLoadResult {
    match request {
        ResourceLoadRequest::Mesh(id, mesh) => match load_mesh(graphics_context, mesh) {
            Ok(loaded_data) => ResourceLoadResult::Mesh(id, loaded_data),
            Err(error) => ResourceLoadResult::Failed(ResourceLoadFailure {
                label: format!("mesh {:?}", id),
                reason: format!("{error:#}"),
            }),
        },
        ResourceLoadRequest::Texture(texture_id, data) => {
            log::info!("Texture loader: decoding '{}' in background", texture_id);
            match graphics_context.create_texture_from_bytes(&data) {
                Ok(texture) => ResourceLoadResult::Texture(texture_id, texture),
                Err(error) => ResourceLoadResult::Failed(ResourceLoadFailure {
                    label: texture_id,
                    reason: format!("{error:#}"),
                }),
            }
        }
    }
}

fn load_mesh(graphics_context: &GraphicsContext, mesh: world::Mesh) -> Result<LoadedMeshData> {
    let vert_size = (mesh.vertices.len() * std::mem::size_of::<world::Vertex>()) as vk::DeviceSize;
    let index_size = (mesh.indices.len() * std::mem::size_of::<u32>()) as vk::DeviceSize;

    let (vertex_buffer, vertex_memory) = graphics_context
        .create_buffer(
            vert_size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .context("failed to create vertex buffer")?;

    unsafe {
        let data_ptr = graphics_context
            .device
            .map_memory(vertex_memory, 0, vert_size, vk::MemoryMapFlags::empty())
            .context("failed to map vertex buffer")?;
        std::ptr::copy_nonoverlapping(
            mesh.vertices.as_ptr(),
            data_ptr as *mut world::Vertex,
            mesh.vertices.len(),
        );
        graphics_context.device.unmap_memory(vertex_memory);
    }

    let (index_buffer, index_memory) = graphics_context
        .create_buffer(
            index_size,
            vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .context("failed to create index buffer")?;

    unsafe {
        let data_ptr = graphics_context
            .device
            .map_memory(index_memory, 0, index_size, vk::MemoryMapFlags::empty())
            .context("failed to map index buffer")?;
        std::ptr::copy_nonoverlapping(
            mesh.indices.as_ptr(),
            data_ptr as *mut u32,
            mesh.indices.len(),
        );
        graphics_context.device.unmap_memory(index_memory);
    }

    let albedo_texture = if let Some(decoded) = &mesh.decoded_albedo {
        graphics_context
            .create_texture_from_raw(decoded.width, decoded.height, &decoded.data)
            .ok()
    } else if let Some(raw_bytes) = &mesh.albedo {
        graphics_context.create_texture_from_bytes(raw_bytes).ok()
    } else {
        None
    };

    let (min, max) = mesh
        .vertices
        .par_iter()
        .map(|vertex| {
            let pos = glam::Vec3::from(vertex.position);
            (pos, pos)
        })
        .reduce(
            || (glam::Vec3::splat(f32::MAX), glam::Vec3::splat(f32::MIN)),
            |(min_a, max_a), (min_b, max_b)| (min_a.min(min_b), max_a.max(max_b)),
        );

    let aabb = graphics::occlusion::AABB::new(min, max);

    Ok(LoadedMeshData {
        vertex_buffer,
        vertex_memory,
        index_buffer,
        index_memory,
        index_count: mesh.indices.len() as u32,
        albedo_texture,
        aabb,
    })
}
