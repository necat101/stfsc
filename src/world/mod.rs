use hecs::World;
use tokio::sync::mpsc;
use log::info;
use glam;
use std::sync::Arc;

pub struct GameWorld {
    pub ecs: World,
    // pub runtime: Runtime, // Removed unused runtime
    pub chunk_receiver: mpsc::Receiver<ChunkData>,
    pub chunk_sender: mpsc::Sender<ChunkData>,
    pub command_receiver: mpsc::Receiver<SceneUpdate>,
    pub command_sender: mpsc::Sender<SceneUpdate>,
}

pub struct ChunkData {
    pub id: u32,
    pub mesh: Option<Mesh>,
    pub transform: Option<Transform>,
}

#[derive(Clone, Debug)]
pub struct Transform {
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub albedo: Option<Vec<u8>>, // Raw image bytes (png/jpg)
    pub normal: Option<Vec<u8>>,
    pub metallic_roughness: Option<Vec<u8>>,
    
    // Runtime-only decoded data (not serialized)
    #[serde(skip)]
    pub decoded_albedo: Option<Arc<DecodedImage>>,
    #[serde(skip)]
    pub decoded_normal: Option<Arc<DecodedImage>>,
    #[serde(skip)]
    pub decoded_mr: Option<Arc<DecodedImage>>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 3],
    pub tangent: [f32; 4], // xyz + w (handedness)
}

#[derive(Clone, Debug)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct Material {
    pub color: [f32; 4],
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub enum SceneUpdate {
    Spawn { id: u32, position: [f32; 3], rotation: [f32; 4], color: [f32; 3] },
    SpawnMesh { id: u32, mesh: Mesh, position: [f32; 3], rotation: [f32; 4] },
    Move { id: u32, position: [f32; 3] },
}

impl GameWorld {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel(10);
        let (cmd_tx, cmd_rx) = mpsc::channel(100);
        let mut world = Self {
            ecs: World::new(),
            // runtime: Runtime::new().unwrap(),
            chunk_receiver: rx,
            chunk_sender: tx,
            command_receiver: cmd_rx,
            command_sender: cmd_tx,
        };
        world.spawn_default_scene();
        world
    }

    pub fn spawn_default_scene(&mut self) {
        // Spawn a few cubes
        for i in 0..5 {
            let x = (i as f32) * 1.5 - 3.0;
            let color = [0.5 + (i as f32) * 0.1, 0.2, 0.2]; // Gradient red
            let tangent = [1.0, 0.0, 0.0, 1.0]; // Default tangent
            self.ecs.spawn((
                Transform {
                    position: glam::Vec3::new(x, 0.0, -3.0),
                    rotation: glam::Quat::IDENTITY,
                    scale: glam::Vec3::ONE * 0.5,
                },
                Mesh {
                    vertices: vec![
                        Vertex { position: [-0.5, -0.5, 0.5], normal: [0.0, 0.0, 1.0], uv: [0.0, 0.0], color, tangent },
                        Vertex { position: [0.5, -0.5, 0.5], normal: [0.0, 0.0, 1.0], uv: [1.0, 0.0], color, tangent },
                        Vertex { position: [0.5, 0.5, 0.5], normal: [0.0, 0.0, 1.0], uv: [1.0, 1.0], color, tangent },
                        Vertex { position: [-0.5, 0.5, 0.5], normal: [0.0, 0.0, 1.0], uv: [0.0, 1.0], color, tangent },
                        Vertex { position: [-0.5, -0.5, -0.5], normal: [0.0, 0.0, -1.0], uv: [0.0, 0.0], color, tangent },
                        Vertex { position: [0.5, -0.5, -0.5], normal: [0.0, 0.0, -1.0], uv: [1.0, 0.0], color, tangent },
                        Vertex { position: [0.5, 0.5, -0.5], normal: [0.0, 0.0, -1.0], uv: [1.0, 1.0], color, tangent },
                        Vertex { position: [-0.5, 0.5, -0.5], normal: [0.0, 0.0, -1.0], uv: [0.0, 1.0], color, tangent },
                    ],
                    indices: vec![
                        0, 1, 2, 2, 3, 0, // Front
                        4, 5, 6, 6, 7, 4, // Back
                        4, 5, 1, 1, 0, 4, // Bottom
                        7, 6, 2, 2, 3, 7, // Top
                        4, 7, 3, 3, 0, 4, // Left
                        5, 6, 2, 2, 1, 5, // Right
                    ],
                    albedo: None,
                    normal: None,
                    metallic_roughness: None,
                    decoded_albedo: None,
                    decoded_normal: None,
                    decoded_mr: None,
                },
            ));
        }
    }

    pub fn update_streaming(&mut self) {
        // Check for loaded chunks
        while let Ok(chunk) = self.chunk_receiver.try_recv() {
            info!("Chunk {} loaded", chunk.id);
            // Add to ECS
            // self.ecs.spawn((chunk.mesh, chunk.rigid_body));
        }

        // Process Scene Updates
        while let Ok(cmd) = self.command_receiver.try_recv() {
            info!("Processing command: {:?}", cmd);
            match cmd {
                SceneUpdate::Spawn { id: _, position, rotation, color } => {
                    let tangent = [1.0, 0.0, 0.0, 1.0];
                    self.ecs.spawn((
                        Transform {
                            position: glam::Vec3::from(position),
                            rotation: glam::Quat::from_array(rotation),
                            scale: glam::Vec3::ONE,
                        },
                        Mesh {
                            vertices: vec![
                                Vertex { position: [-0.5, -0.5, 0.5], normal: [0.0, 0.0, 1.0], uv: [0.0, 0.0], color, tangent },
                                Vertex { position: [0.5, -0.5, 0.5], normal: [0.0, 0.0, 1.0], uv: [1.0, 0.0], color, tangent },
                                Vertex { position: [0.5, 0.5, 0.5], normal: [0.0, 0.0, 1.0], uv: [1.0, 1.0], color, tangent },
                                Vertex { position: [-0.5, 0.5, 0.5], normal: [0.0, 0.0, 1.0], uv: [0.0, 1.0], color, tangent },
                                Vertex { position: [-0.5, -0.5, -0.5], normal: [0.0, 0.0, -1.0], uv: [0.0, 0.0], color, tangent },
                                Vertex { position: [0.5, -0.5, -0.5], normal: [0.0, 0.0, -1.0], uv: [1.0, 0.0], color, tangent },
                                Vertex { position: [0.5, 0.5, -0.5], normal: [0.0, 0.0, -1.0], uv: [1.0, 1.0], color, tangent },
                                Vertex { position: [-0.5, 0.5, -0.5], normal: [0.0, 0.0, -1.0], uv: [0.0, 1.0], color, tangent },
                            ],
                            indices: vec![
                                0, 1, 2, 2, 3, 0, // Front
                                4, 5, 6, 6, 7, 4, // Back
                                4, 5, 1, 1, 0, 4, // Bottom
                                7, 6, 2, 2, 3, 7, // Top
                                4, 7, 3, 3, 0, 4, // Left
                                5, 6, 2, 2, 1, 5, // Right
                            ],
                            albedo: None,
                            normal: None,
                            metallic_roughness: None,
                            decoded_albedo: None,
                            decoded_normal: None,
                            decoded_mr: None,
                        },
                    ));
                    // info!("Spawn command received (ECS spawn disabled)");
                }
                SceneUpdate::SpawnMesh { id: _, mesh, position, rotation } => {
                     self.ecs.spawn((
                        Transform {
                            position: glam::Vec3::from(position),
                            rotation: glam::Quat::from_array(rotation),
                            scale: glam::Vec3::ONE,
                        },
                        mesh,
                    ));
                }
                SceneUpdate::Move { id: _, position: _ } => {
                    // TODO: Find entity by ID and update transform
                }
            }
        }

        // Simulate requesting a chunk (e.g. based on player position)
        // In a real game, this would check distance and only request if not loaded
        // For demo, we just spawn a task occasionally or once
        
        // Example: Spawn a task to load chunk 1
        // self.request_chunk(1);
    }

    pub fn request_chunk(&self, _chunk_id: u32) {
        // let tx = self.chunk_sender.clone();
        // self.runtime.spawn(async move { ... });
        // NOTE: Runtime removed. If async IO is needed, pass a handle or use a separate IO system.
    }
}

pub fn load_obj_from_bytes(data: &[u8]) -> anyhow::Result<Mesh> {
    let mut reader = std::io::Cursor::new(data);
    let (models, _materials) = tobj::load_obj_buf(
        &mut reader,
        &tobj::LoadOptions {
            single_index: true,
            triangulate: true,
            ..Default::default()
        },
        |p| tobj::load_mtl_buf(&mut std::io::Cursor::new(Vec::new())) // Ignore materials for now
    )?;

    if models.is_empty() {
        anyhow::bail!("No models found in buffer");
    }

    // Take the first model
    let mesh = &models[0].mesh;
    let mut vertices = Vec::new();
    
    for i in 0..mesh.positions.len() / 3 {
        let pos = [
            mesh.positions[i * 3],
            mesh.positions[i * 3 + 1],
            mesh.positions[i * 3 + 2],
        ];
        
        let normal = if !mesh.normals.is_empty() {
            [
                mesh.normals[i * 3],
                mesh.normals[i * 3 + 1],
                mesh.normals[i * 3 + 2],
            ]
        } else {
            [0.0, 1.0, 0.0] // Default normal (up)
        };
        
        let uv = if !mesh.texcoords.is_empty() {
            [
                mesh.texcoords[i * 2],
                1.0 - mesh.texcoords[i * 2 + 1], // Flip V for Vulkan/OBJ mismatch
            ]
        } else {
            [0.0, 0.0]
        };
        
        // TODO: Calc tangents
        let tangent = [1.0, 0.0, 0.0, 1.0];

        vertices.push(Vertex {
            position: pos,
            normal,
            uv,
            color: [1.0, 1.0, 1.0], // Default white color
            tangent,
        });
    }

    Ok(Mesh {
        vertices,
        indices: mesh.indices.clone(),
        albedo: None,
        normal: None,
        metallic_roughness: None,
        decoded_albedo: None,
        decoded_normal: None,
        decoded_mr: None,
    })
}
