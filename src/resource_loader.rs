use crate::graphics::{self, GraphicsContext, Texture};
use crate::world;
use anyhow::{Context, Result};
use ash::vk;
use crossbeam_channel::{Receiver, Sender, TrySendError};
use hecs::Entity;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceLoadPriority {
    High,
    Normal,
    Low,
}

impl Default for ResourceLoadPriority {
    fn default() -> Self {
        Self::Normal
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceQueueErrorKind {
    Full,
    Disconnected,
    ShuttingDown,
}

#[derive(Debug)]
pub struct ResourceQueueError<T> {
    kind: ResourceQueueErrorKind,
    item: T,
}

impl<T> ResourceQueueError<T> {
    fn new(kind: ResourceQueueErrorKind, item: T) -> Self {
        Self { kind, item }
    }

    pub fn kind(&self) -> ResourceQueueErrorKind {
        self.kind
    }

    pub fn into_inner(self) -> T {
        self.item
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ResourceLoaderStats {
    pub queued: usize,
    pub in_flight: usize,
    pub completed: usize,
    pub failed: usize,
    pub workers: usize,
    pub per_priority_capacity: usize,
}

#[derive(Default)]
struct ResourceLoaderCounters {
    queued: AtomicUsize,
    in_flight: AtomicUsize,
    completed: AtomicUsize,
    failed: AtomicUsize,
}

pub struct ResourceLoader {
    high_priority: Sender<ResourceLoadRequest>,
    normal_priority: Sender<ResourceLoadRequest>,
    low_priority: Sender<ResourceLoadRequest>,
    from_loader: Receiver<ResourceLoadResult>,
    counters: Arc<ResourceLoaderCounters>,
    shutdown: Arc<AtomicBool>,
    workers: Vec<JoinHandle<()>>,
    per_priority_capacity: usize,
}

impl ResourceLoader {
    pub fn new(graphics_context: Arc<GraphicsContext>) -> Self {
        let worker_count = crate::runtime::background_worker_count();
        let per_priority_capacity = (crate::runtime::resource_queue_capacity() / 3)
            .max(worker_count.saturating_mul(2))
            .max(4);

        let (high_priority, high_rx) =
            crossbeam_channel::bounded::<ResourceLoadRequest>(per_priority_capacity);
        let (normal_priority, normal_rx) =
            crossbeam_channel::bounded::<ResourceLoadRequest>(per_priority_capacity);
        let (low_priority, low_rx) =
            crossbeam_channel::bounded::<ResourceLoadRequest>(per_priority_capacity);
        let (loader_tx, from_loader) = crossbeam_channel::unbounded::<ResourceLoadResult>();
        let counters = Arc::new(ResourceLoaderCounters::default());
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut workers = Vec::with_capacity(worker_count);

        for worker_idx in 0..worker_count {
            let graphics_context = graphics_context.clone();
            let high_rx = high_rx.clone();
            let normal_rx = normal_rx.clone();
            let low_rx = low_rx.clone();
            let loader_tx = loader_tx.clone();
            let counters = counters.clone();
            let shutdown = shutdown.clone();
            match std::thread::Builder::new()
                .name(format!("stfsc-loader-{worker_idx}"))
                .spawn(move || {
                    while let Some(request) =
                        recv_next_request(&high_rx, &normal_rx, &low_rx, &shutdown)
                    {
                        counters.queued.fetch_sub(1, Ordering::Relaxed);
                        counters.in_flight.fetch_add(1, Ordering::Relaxed);

                        let result = process_request(&graphics_context, request);
                        counters.in_flight.fetch_sub(1, Ordering::Relaxed);

                        if matches!(result, ResourceLoadResult::Failed(_)) {
                            counters.failed.fetch_add(1, Ordering::Relaxed);
                        } else {
                            counters.completed.fetch_add(1, Ordering::Relaxed);
                        }

                        let _ = loader_tx.send(result);
                    }
                }) {
                Ok(worker) => workers.push(worker),
                Err(error) => log::error!("Failed to spawn resource loader worker: {error}"),
            }
        }

        Self {
            high_priority,
            normal_priority,
            low_priority,
            from_loader,
            counters,
            shutdown,
            workers,
            per_priority_capacity,
        }
    }

    pub fn queue_mesh(&self, id: Entity, mesh: world::Mesh) {
        if let Err(error) = self.queue_request(
            ResourceLoadRequest::Mesh(id, mesh),
            ResourceLoadPriority::Normal,
        ) {
            log::warn!("Dropped mesh load request: {:?}", error.kind());
        }
    }

    pub fn queue_mesh_with_priority(
        &self,
        id: Entity,
        mesh: world::Mesh,
        priority: ResourceLoadPriority,
    ) {
        if let Err(error) = self.queue_request(ResourceLoadRequest::Mesh(id, mesh), priority) {
            log::warn!("Dropped mesh load request: {:?}", error.kind());
        }
    }

    pub fn try_queue_mesh(
        &self,
        id: Entity,
        mesh: world::Mesh,
    ) -> std::result::Result<(), ResourceQueueError<(Entity, world::Mesh)>> {
        self.try_queue_mesh_with_priority(id, mesh, ResourceLoadPriority::Normal)
    }

    pub fn try_queue_mesh_with_priority(
        &self,
        id: Entity,
        mesh: world::Mesh,
        priority: ResourceLoadPriority,
    ) -> std::result::Result<(), ResourceQueueError<(Entity, world::Mesh)>> {
        self.try_queue_request(ResourceLoadRequest::Mesh(id, mesh), priority)
            .map_err(|error| match error.kind {
                ResourceQueueErrorKind::Full => match error.item {
                    ResourceLoadRequest::Mesh(id, mesh) => {
                        ResourceQueueError::new(ResourceQueueErrorKind::Full, (id, mesh))
                    }
                    ResourceLoadRequest::Texture(_, _) => unreachable!(),
                },
                ResourceQueueErrorKind::Disconnected => match error.item {
                    ResourceLoadRequest::Mesh(id, mesh) => {
                        ResourceQueueError::new(ResourceQueueErrorKind::Disconnected, (id, mesh))
                    }
                    ResourceLoadRequest::Texture(_, _) => unreachable!(),
                },
                ResourceQueueErrorKind::ShuttingDown => match error.item {
                    ResourceLoadRequest::Mesh(id, mesh) => {
                        ResourceQueueError::new(ResourceQueueErrorKind::ShuttingDown, (id, mesh))
                    }
                    ResourceLoadRequest::Texture(_, _) => unreachable!(),
                },
            })
    }

    pub fn queue_texture(&self, texture_id: String, data: Vec<u8>) {
        if let Err(error) = self.queue_request(
            ResourceLoadRequest::Texture(texture_id, data),
            ResourceLoadPriority::Normal,
        ) {
            log::warn!("Dropped texture load request: {:?}", error.kind());
        }
    }

    pub fn queue_texture_with_priority(
        &self,
        texture_id: String,
        data: Vec<u8>,
        priority: ResourceLoadPriority,
    ) {
        if let Err(error) =
            self.queue_request(ResourceLoadRequest::Texture(texture_id, data), priority)
        {
            log::warn!("Dropped texture load request: {:?}", error.kind());
        }
    }

    pub fn try_queue_texture(
        &self,
        texture_id: String,
        data: Vec<u8>,
    ) -> std::result::Result<(), ResourceQueueError<(String, Vec<u8>)>> {
        self.try_queue_texture_with_priority(texture_id, data, ResourceLoadPriority::Normal)
    }

    pub fn try_queue_texture_with_priority(
        &self,
        texture_id: String,
        data: Vec<u8>,
        priority: ResourceLoadPriority,
    ) -> std::result::Result<(), ResourceQueueError<(String, Vec<u8>)>> {
        self.try_queue_request(ResourceLoadRequest::Texture(texture_id, data), priority)
            .map_err(|error| match error.kind {
                ResourceQueueErrorKind::Full => match error.item {
                    ResourceLoadRequest::Texture(texture_id, data) => {
                        ResourceQueueError::new(ResourceQueueErrorKind::Full, (texture_id, data))
                    }
                    ResourceLoadRequest::Mesh(_, _) => unreachable!(),
                },
                ResourceQueueErrorKind::Disconnected => match error.item {
                    ResourceLoadRequest::Texture(texture_id, data) => ResourceQueueError::new(
                        ResourceQueueErrorKind::Disconnected,
                        (texture_id, data),
                    ),
                    ResourceLoadRequest::Mesh(_, _) => unreachable!(),
                },
                ResourceQueueErrorKind::ShuttingDown => match error.item {
                    ResourceLoadRequest::Texture(texture_id, data) => ResourceQueueError::new(
                        ResourceQueueErrorKind::ShuttingDown,
                        (texture_id, data),
                    ),
                    ResourceLoadRequest::Mesh(_, _) => unreachable!(),
                },
            })
    }

    pub fn poll_processed(&self) -> impl Iterator<Item = ResourceLoadResult> + '_ {
        self.from_loader.try_iter()
    }

    pub fn queued_count(&self) -> usize {
        self.counters.queued.load(Ordering::Relaxed)
    }

    pub fn stats(&self) -> ResourceLoaderStats {
        ResourceLoaderStats {
            queued: self.counters.queued.load(Ordering::Relaxed),
            in_flight: self.counters.in_flight.load(Ordering::Relaxed),
            completed: self.counters.completed.load(Ordering::Relaxed),
            failed: self.counters.failed.load(Ordering::Relaxed),
            workers: self.workers.len(),
            per_priority_capacity: self.per_priority_capacity,
        }
    }

    fn try_queue_request(
        &self,
        request: ResourceLoadRequest,
        priority: ResourceLoadPriority,
    ) -> std::result::Result<(), ResourceQueueError<ResourceLoadRequest>> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(ResourceQueueError::new(
                ResourceQueueErrorKind::ShuttingDown,
                request,
            ));
        }

        self.counters.queued.fetch_add(1, Ordering::Relaxed);
        let send_result = match priority {
            ResourceLoadPriority::High => self.high_priority.try_send(request),
            ResourceLoadPriority::Normal => self.normal_priority.try_send(request),
            ResourceLoadPriority::Low => self.low_priority.try_send(request),
        };

        match send_result {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(request)) => {
                self.counters.queued.fetch_sub(1, Ordering::Relaxed);
                Err(ResourceQueueError::new(
                    ResourceQueueErrorKind::Full,
                    request,
                ))
            }
            Err(TrySendError::Disconnected(request)) => {
                self.counters.queued.fetch_sub(1, Ordering::Relaxed);
                Err(ResourceQueueError::new(
                    ResourceQueueErrorKind::Disconnected,
                    request,
                ))
            }
        }
    }

    fn queue_request(
        &self,
        request: ResourceLoadRequest,
        priority: ResourceLoadPriority,
    ) -> std::result::Result<(), ResourceQueueError<ResourceLoadRequest>> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(ResourceQueueError::new(
                ResourceQueueErrorKind::ShuttingDown,
                request,
            ));
        }

        self.counters.queued.fetch_add(1, Ordering::Relaxed);
        let send_result = match priority {
            ResourceLoadPriority::High => self.high_priority.send(request),
            ResourceLoadPriority::Normal => self.normal_priority.send(request),
            ResourceLoadPriority::Low => self.low_priority.send(request),
        };

        match send_result {
            Ok(()) => Ok(()),
            Err(error) => {
                self.counters.queued.fetch_sub(1, Ordering::Relaxed);
                Err(ResourceQueueError::new(
                    ResourceQueueErrorKind::Disconnected,
                    error.0,
                ))
            }
        }
    }
}

impl Drop for ResourceLoader {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        for worker in self.workers.drain(..) {
            if worker.join().is_err() {
                log::warn!("Resource loader worker panicked during shutdown");
            }
        }
    }
}

fn recv_next_request(
    high_rx: &Receiver<ResourceLoadRequest>,
    normal_rx: &Receiver<ResourceLoadRequest>,
    low_rx: &Receiver<ResourceLoadRequest>,
    shutdown: &AtomicBool,
) -> Option<ResourceLoadRequest> {
    while !shutdown.load(Ordering::Acquire) {
        if let Ok(request) = high_rx.try_recv() {
            return Some(request);
        }
        if let Ok(request) = normal_rx.try_recv() {
            return Some(request);
        }
        if let Ok(request) = low_rx.try_recv() {
            return Some(request);
        }

        crossbeam_channel::select! {
            recv(high_rx) -> message => {
                if let Ok(request) = message {
                    return Some(request);
                }
            }
            recv(normal_rx) -> message => {
                if let Ok(request) = message {
                    return Some(request);
                }
            }
            recv(low_rx) -> message => {
                if let Ok(request) = message {
                    return Some(request);
                }
            }
            default(Duration::from_millis(2)) => {}
        }
    }

    None
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
    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
        anyhow::bail!("mesh contains no renderable geometry");
    }

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
