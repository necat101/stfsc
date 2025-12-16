use ash::vk;
use std::sync::{Arc, mpsc::{Sender, Receiver}};
use crate::graphics::{self, GraphicsContext, Texture};
use crate::world;
use hecs::Entity;

pub struct LoadedMeshData {
    pub vertex_buffer: vk::Buffer,
    pub vertex_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    pub index_memory: vk::DeviceMemory,
    pub index_count: u32,
    pub albedo_texture: Option<Texture>,
}

pub struct ResourceLoader {
    to_loader: Sender<(Entity, world::Mesh)>,
    from_loader: Receiver<(Entity, LoadedMeshData)>,
}

impl ResourceLoader {
    pub fn new(graphics_context: Arc<GraphicsContext>) -> Self {
        let (to_loader, loader_rx) = std::sync::mpsc::channel::<(Entity, world::Mesh)>();
        let (loader_tx, from_loader) = std::sync::mpsc::channel::<(Entity, LoadedMeshData)>();
        
        std::thread::spawn(move || {
            while let Ok((id, mesh)) = loader_rx.recv() {
                // Determine sizes
                let vert_size = (mesh.vertices.len() * std::mem::size_of::<world::Vertex>()) as vk::DeviceSize;
                let index_size = (mesh.indices.len() * std::mem::size_of::<u32>()) as vk::DeviceSize;
                
                // Create Vertex Buffer
                let (vertex_buffer, vertex_memory) = unsafe {
                    graphics_context.create_buffer(
                        vert_size,
                        vk::BufferUsageFlags::VERTEX_BUFFER,
                        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                    ).expect("Failed to create vertex buffer")
                };

                unsafe {
                    let data_ptr = graphics_context.device.map_memory(
                        vertex_memory,
                        0,
                        vert_size,
                        vk::MemoryMapFlags::empty(),
                    ).expect("Failed to map vertex buffer");
                    std::ptr::copy_nonoverlapping(
                        mesh.vertices.as_ptr(),
                        data_ptr as *mut world::Vertex,
                        mesh.vertices.len(),
                    );
                    graphics_context.device.unmap_memory(vertex_memory);
                }

                // Create Index Buffer
                let (index_buffer, index_memory) = unsafe {
                    graphics_context.create_buffer(
                        index_size,
                        vk::BufferUsageFlags::INDEX_BUFFER,
                        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                    ).expect("Failed to create index buffer")
                };

                unsafe {
                    let data_ptr = graphics_context.device.map_memory(
                        index_memory,
                        0,
                        index_size,
                        vk::MemoryMapFlags::empty(),
                    ).expect("Failed to map index buffer");
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
                    // This call is now thread-safe due to queue_mutex in GraphicsContext
                    if let Ok(tex) = graphics_context.create_texture_from_raw(decoded.width, decoded.height, &decoded.data) {
                        albedo_texture = Some(tex);
                    }
                }
                
                let loaded_data = LoadedMeshData {
                    vertex_buffer,
                    vertex_memory,
                    index_buffer,
                    index_memory,
                    index_count: mesh.indices.len() as u32,
                    albedo_texture,
                };
                
                let _ = loader_tx.send((id, loaded_data));
            }
        });
        
        Self { to_loader, from_loader }
    }
    
    pub fn queue_upload(&self, id: Entity, mesh: world::Mesh) {
        let _ = self.to_loader.send((id, mesh));
    }
    
    pub fn poll_processed(&self) -> impl Iterator<Item = (Entity, LoadedMeshData)> + '_ {
        self.from_loader.try_iter()
    }
}
