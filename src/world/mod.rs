use hecs::World;
use tokio::runtime::Runtime;
use tokio::sync::mpsc;
use log::info;
use glam;

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

#[derive(Clone, Debug)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct Material {
    pub color: [f32; 4],
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub enum SceneUpdate {
    Spawn { id: u32, position: [f32; 3], color: [f32; 3] },
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
            self.ecs.spawn((
                Transform {
                    position: glam::Vec3::new(x, 0.0, -3.0),
                    rotation: glam::Quat::IDENTITY,
                    scale: glam::Vec3::ONE * 0.5,
                },
                Mesh {
                    vertices: vec![
                        Vertex { position: [-0.5, -0.5, 0.5], color },
                        Vertex { position: [0.5, -0.5, 0.5], color },
                        Vertex { position: [0.5, 0.5, 0.5], color },
                        Vertex { position: [-0.5, 0.5, 0.5], color },
                        Vertex { position: [-0.5, -0.5, -0.5], color },
                        Vertex { position: [0.5, -0.5, -0.5], color },
                        Vertex { position: [0.5, 0.5, -0.5], color },
                        Vertex { position: [-0.5, 0.5, -0.5], color },
                    ],
                    indices: vec![
                        0, 1, 2, 2, 3, 0, // Front
                        4, 5, 6, 6, 7, 4, // Back
                        4, 5, 1, 1, 0, 4, // Bottom
                        7, 6, 2, 2, 3, 7, // Top
                        4, 7, 3, 3, 0, 4, // Left
                        5, 6, 2, 2, 1, 5, // Right
                    ],
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
                SceneUpdate::Spawn { id, position, color } => {
                    self.ecs.spawn((
                        Transform {
                            position: glam::Vec3::from(position),
                            rotation: glam::Quat::IDENTITY,
                            scale: glam::Vec3::ONE,
                        },
                        Mesh {
                            vertices: vec![
                                Vertex { position: [-0.5, -0.5, 0.5], color },
                                Vertex { position: [0.5, -0.5, 0.5], color },
                                Vertex { position: [0.5, 0.5, 0.5], color },
                                Vertex { position: [-0.5, 0.5, 0.5], color },
                                Vertex { position: [-0.5, -0.5, -0.5], color },
                                Vertex { position: [0.5, -0.5, -0.5], color },
                                Vertex { position: [0.5, 0.5, -0.5], color },
                                Vertex { position: [-0.5, 0.5, -0.5], color },
                            ],
                            indices: vec![
                                0, 1, 2, 2, 3, 0, // Front
                                4, 5, 6, 6, 7, 4, // Back
                                4, 5, 1, 1, 0, 4, // Bottom
                                7, 6, 2, 2, 3, 7, // Top
                                4, 7, 3, 3, 0, 4, // Left
                                5, 6, 2, 2, 1, 5, // Right
                            ],
                        },
                    ));
                    // info!("Spawn command received (ECS spawn disabled)");
                }
                SceneUpdate::Move { id, position } => {
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

    pub fn request_chunk(&self, chunk_id: u32) {
        // let tx = self.chunk_sender.clone();
        // self.runtime.spawn(async move { ... });
        // NOTE: Runtime removed. If async IO is needed, pass a handle or use a separate IO system.
    }
}
