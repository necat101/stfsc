use crate::graphics::{self, GraphicsContext, Texture};
use crate::world;
use ash::vk;
use hecs::Entity;
use std::sync::{
    mpsc::{Receiver, Sender},
    Arc,
};

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
}

pub struct ResourceLoader {
    to_loader: Sender<ResourceLoadRequest>,
    from_loader: Receiver<ResourceLoadResult>,
}

impl ResourceLoader {
    pub fn new(graphics_context: Arc<GraphicsContext>) -> Self {
        let (to_loader, loader_rx) = std::sync::mpsc::channel::<ResourceLoadRequest>();
        let (loader_tx, from_loader) = std::sync::mpsc::channel::<ResourceLoadResult>();

        std::thread::spawn(move || {
            while let Ok(request) = loader_rx.recv() {
                match request {
                    ResourceLoadRequest::Mesh(id, mesh) => {
                        // Determine sizes
                        let vert_size = (mesh.vertices.len() * std::mem::size_of::<world::Vertex>())
                            as vk::DeviceSize;
                        let index_size =
                            (mesh.indices.len() * std::mem::size_of::<u32>()) as vk::DeviceSize;

                        // Create Vertex Buffer
                        let (vertex_buffer, vertex_memory) = graphics_context
                            .create_buffer(
                                vert_size,
                                vk::BufferUsageFlags::VERTEX_BUFFER,
                                vk::MemoryPropertyFlags::HOST_VISIBLE
                                    | vk::MemoryPropertyFlags::HOST_COHERENT,
                            )
                            .expect("Failed to create vertex buffer");

                        unsafe {
                            let data_ptr = graphics_context
                                .device
                                .map_memory(vertex_memory, 0, vert_size, vk::MemoryMapFlags::empty())
                                .expect("Failed to map vertex buffer");
                            std::ptr::copy_nonoverlapping(
                                mesh.vertices.as_ptr(),
                                data_ptr as *mut world::Vertex,
                                mesh.vertices.len(),
                            );
                            graphics_context.device.unmap_memory(vertex_memory);
                        }

                        // Create Index Buffer
                        let (index_buffer, index_memory) = graphics_context
                            .create_buffer(
                                index_size,
                                vk::BufferUsageFlags::INDEX_BUFFER,
                                vk::MemoryPropertyFlags::HOST_VISIBLE
                                    | vk::MemoryPropertyFlags::HOST_COHERENT,
                            )
                            .expect("Failed to create index buffer");

                        unsafe {
                            let data_ptr = graphics_context
                                .device
                                .map_memory(index_memory, 0, index_size, vk::MemoryMapFlags::empty())
                                .expect("Failed to map index buffer");
                            std::ptr::copy_nonoverlapping(
                                mesh.indices.as_ptr(),
                                data_ptr as *mut u32,
                                mesh.indices.len(),
                            );
                            graphics_context.device.unmap_memory(index_memory);
                        }

                        // Handle Albedo Texture
                        let mut albedo_texture = None;
                        if let Some(decoded) = &mesh.decoded_albedo {
                            // Use pre-decoded texture
                            if let Ok(tex) = graphics_context.create_texture_from_raw(
                                decoded.width,
                                decoded.height,
                                &decoded.data,
                            ) {
                                albedo_texture = Some(tex);
                            }
                        } else if let Some(raw_bytes) = &mesh.albedo {
                            // Decode raw PNG/JPG bytes on the fly
                            if let Ok(tex) = graphics_context.create_texture_from_bytes(raw_bytes) {
                                albedo_texture = Some(tex);
                            }
                        }

                        // Calculate AABB
                        let mut min = glam::Vec3::splat(f32::MAX);
                        let mut max = glam::Vec3::splat(f32::MIN);
                        for v in &mesh.vertices {
                            let pos = glam::Vec3::from(v.position);
                            min = min.min(pos);
                            max = max.max(pos);
                        }
                        let aabb = graphics::occlusion::AABB::new(min, max);

                        let loaded_data = LoadedMeshData {
                            vertex_buffer,
                            vertex_memory,
                            index_buffer,
                            index_memory,
                            index_count: mesh.indices.len() as u32,
                            albedo_texture,
                            aabb,
                        };

                        let _ = loader_tx.send(ResourceLoadResult::Mesh(id, loaded_data));
                    }
                    ResourceLoadRequest::Texture(texture_id, data) => {
                        println!("TEXTURE LOADER: Decoding '{}' in background", texture_id);
                        match graphics_context.create_texture_from_bytes(&data) {
                            Ok(texture) => {
                                let _ = loader_tx.send(ResourceLoadResult::Texture(texture_id, texture));
                            }
                            Err(e) => {
                                println!("TEXTURE LOADER ERROR: Failed to load '{}': {:?}", texture_id, e);
                            }
                        }
                    }
                }
            }
        });

        Self {
            to_loader,
            from_loader,
        }
    }

    pub fn queue_mesh(&self, id: Entity, mesh: world::Mesh) {
        let _ = self.to_loader.send(ResourceLoadRequest::Mesh(id, mesh));
    }

    pub fn queue_texture(&self, texture_id: String, data: Vec<u8>) {
        let _ = self.to_loader.send(ResourceLoadRequest::Texture(texture_id, data));
    }

    pub fn poll_processed(&self) -> impl Iterator<Item = ResourceLoadResult> + '_ {
        self.from_loader.try_iter()
    }
}
