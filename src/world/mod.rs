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
    pub loaded_chunks: std::collections::HashSet<(i32, i32)>,
    /// Enable procedural world generation (buildings, props). Off by default.
    pub procedural_generation_enabled: bool,
    pub player_start_transform: Transform,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshHandle(pub usize);

/// Marker component for entities spawned from the editor (for tracking/deletion)
#[derive(Clone, Copy, Debug)]
pub struct EditorEntityId(pub u32);

#[derive(Debug)]
pub struct Procedural; // Tag for procedurally generated entities

#[derive(Clone, Debug)]
pub struct LODGroup {
    pub levels: Vec<LODLevel>,
}

#[derive(Clone, Debug)]
pub struct LODLevel {
    pub distance: f32, // Distance at which this LOD becomes active
    pub mesh: MeshHandle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RigidBodyHandle(pub rapier3d::prelude::RigidBodyHandle);

#[derive(Clone, Debug)]
pub struct Vehicle {
    pub speed: f32,          // Current speed
    pub max_speed: f32,      // Maximum speed
    pub steering: f32,
    pub accelerating: bool,  // Is the vehicle accelerating?
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentState {
    Idle,
    Walking,
    Running,
    Fleeing,  // Running away from danger
}

#[derive(Clone, Debug)]
pub struct CrowdAgent {
    pub velocity: glam::Vec3,
    pub target: glam::Vec3,
    pub state: AgentState,
    pub max_speed: f32,      // Speed varies by state
}

/// Light types for dynamic lighting
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[repr(u8)]
pub enum LightType {
    Point = 0,
    Spot = 1,
    Directional = 2,
}

impl Default for LightType {
    fn default() -> Self {
        LightType::Point
    }
}

/// ECS Component for dynamic lights
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LightComponent {
    pub light_type: LightType,
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,              // Attenuation distance (Point/Spot)
    pub inner_cone_angle: f32,   // Spot light inner cone (radians)
    pub outer_cone_angle: f32,   // Spot light outer cone (radians)
    pub cast_shadows: bool,
}

impl Default for LightComponent {
    fn default() -> Self {
        Self {
            light_type: LightType::Point,
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            range: 10.0,
            inner_cone_angle: 0.4,
            outer_cone_angle: 0.6,
            cast_shadows: false,
        }
    }
}

impl LightComponent {
    /// Create a point light
    pub fn point(color: [f32; 3], intensity: f32, range: f32) -> Self {
        Self {
            light_type: LightType::Point,
            color,
            intensity,
            range,
            ..Default::default()
        }
    }

    /// Create a spot light
    pub fn spot(color: [f32; 3], intensity: f32, range: f32, inner_angle: f32, outer_angle: f32) -> Self {
        Self {
            light_type: LightType::Spot,
            color,
            intensity,
            range,
            inner_cone_angle: inner_angle,
            outer_cone_angle: outer_angle,
            ..Default::default()
        }
    }

    /// Create a directional light
    pub fn directional(color: [f32; 3], intensity: f32) -> Self {
        Self {
            light_type: LightType::Directional,
            color,
            intensity,
            range: f32::MAX,
            ..Default::default()
        }
    }
}

/// ECS Component for 3D audio sources
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AudioSource {
    /// Sound file path or identifier
    pub sound_id: String,
    /// Volume (0.0 to 1.0)
    pub volume: f32,
    /// Whether to loop
    pub looping: bool,
    /// Max audible distance
    pub max_distance: f32,
    /// Whether currently playing
    pub playing: bool,
}

impl Default for AudioSource {
    fn default() -> Self {
        Self {
            sound_id: String::new(),
            volume: 1.0,
            looping: false,
            max_distance: 50.0,
            playing: false,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub enum SceneUpdate {
    Spawn { id: u32, primitive: u8, position: [f32; 3], rotation: [f32; 4], color: [f32; 3] },
    SpawnMesh { id: u32, mesh: Mesh, position: [f32; 3], rotation: [f32; 4] },
    Move { id: u32, position: [f32; 3] },
    DeleteEntity { id: u32 },
    ClearScene,
    /// Toggle procedural generation on/off (off by default)
    SetProceduralGeneration { enabled: bool },
    SetPlayerStart { position: [f32; 3], rotation: [f32; 4] },
    /// Spawn a dynamic light
    SpawnLight { 
        id: u32, 
        light_type: LightType, 
        position: [f32; 3], 
        direction: [f32; 3],  // For spot/directional
        color: [f32; 3], 
        intensity: f32, 
        range: f32,
        inner_cone: f32,
        outer_cone: f32,
    },
    /// Update an existing light's properties
    UpdateLight { 
        id: u32, 
        color: Option<[f32; 3]>, 
        intensity: Option<f32>,
        range: Option<f32>,
    },
    /// Spawn a 3D sound at a position
    SpawnSound {
        id: u32,
        sound_id: String,
        position: [f32; 3],
        volume: f32,
        looping: bool,
        max_distance: f32,
    },
    /// Stop a playing sound
    StopSound { id: u32 },
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
            loaded_chunks: std::collections::HashSet::new(),
            procedural_generation_enabled: false, // Off by default
            player_start_transform: Transform { 
                position: glam::Vec3::new(0.0, 1.7, 0.0), // Default standing height
                rotation: glam::Quat::IDENTITY, 
                scale: glam::Vec3::ONE 
            },
        };
        // world.spawn_default_scene(); // Moved to streaming logic or separate init
        world
    }

    pub fn update_streaming(&mut self, player_pos: glam::Vec3) {
        // Chunk Settings
        let chunk_size = 16.0; // 16 meters
        let load_radius = 4;   // 4 chunks radius

        // Calculate player chunk coord
        let pc_x = (player_pos.x / chunk_size).floor() as i32;
        let pc_z = (player_pos.z / chunk_size).floor() as i32;

        for x in (pc_x - load_radius)..=(pc_x + load_radius) {
            for z in (pc_z - load_radius)..=(pc_z + load_radius) {
                if !self.loaded_chunks.contains(&(x, z)) {
                    // Load this chunk
                    self.loaded_chunks.insert((x, z));
                    self.generate_chunk(x, z, chunk_size);
                }
            }
        }
        
        // Unloading can be added here (check if chunk is far)
        // ...
        
        // Process Scene Updates (Networking)
        while let Ok(cmd) = self.command_receiver.try_recv() {
            info!("Processing command: {:?}", cmd);
            match cmd {
                SceneUpdate::Spawn { id, primitive, position, rotation, color } => {
                    // Use MeshHandle(primitive index)
                    // The mesh library in lib.rs must be initialized with corresponding meshes
                    self.ecs.spawn((
                        EditorEntityId(id), // Track this entity's ID for deletion
                        Transform {
                            position: glam::Vec3::from(position),
                            rotation: glam::Quat::from_array(rotation),
                            scale: glam::Vec3::ONE,
                        },
                        MeshHandle(primitive as usize), // Use requested primitive
                        Material { color: [color[0], color[1], color[2], 1.0] },
                    ));
                    info!("Spawned entity with EditorEntityId({}) using MeshHandle({})", id, primitive);
                }
                SceneUpdate::SpawnMesh { id, mesh, position, rotation } => {
                     self.ecs.spawn((
                        EditorEntityId(id), // Track this entity's ID for deletion
                        Transform {
                            position: glam::Vec3::from(position),
                            rotation: glam::Quat::from_array(rotation),
                            scale: glam::Vec3::ONE,
                        },
                        mesh,
                        Material { color: [1.0, 1.0, 1.0, 1.0] }, // Default white for mesh
                    ));
                    info!("Spawned mesh with EditorEntityId({})", id);
                }
                SceneUpdate::Move { id, position } => {
                    // Find entity by EditorEntityId and update transform
                    for (entity, (editor_id, transform)) in self.ecs.query_mut::<(&EditorEntityId, &mut Transform)>() {
                        if editor_id.0 == id {
                            transform.position = glam::Vec3::from(position);
                            info!("Moved entity {} to {:?}", id, position);
                            break;
                        }
                    }
                }
                SceneUpdate::DeleteEntity { id } => {
                    // Find and despawn entity by EditorEntityId
                    let mut to_delete: Option<hecs::Entity> = None;
                    for (entity, editor_id) in self.ecs.query::<&EditorEntityId>().iter() {
                        if editor_id.0 == id {
                            to_delete = Some(entity);
                            break;
                        }
                    }
                    if let Some(entity) = to_delete {
                        let _ = self.ecs.despawn(entity);
                        info!("Deleted entity with EditorEntityId({})", id);
                    } else {
                        info!("DeleteEntity: No entity found with id {}", id);
                    }
                }
                SceneUpdate::ClearScene => {
                    // Clear all editor-spawned entities and procedural entities
                    let mut to_delete = Vec::new();
                    
                    for (entity, _) in self.ecs.query::<&EditorEntityId>().iter() {
                        to_delete.push(entity);
                    }
                    for (entity, _) in self.ecs.query::<&Procedural>().iter() {
                        to_delete.push(entity);
                    }
                    
                    let count = to_delete.len();
                    for entity in to_delete {
                        let _ = self.ecs.despawn(entity);
                    }
                    self.loaded_chunks.clear(); // Clear chunk history too
                    info!("ClearScene: Deleted {} entities", count);
                }
                SceneUpdate::SetProceduralGeneration { enabled } => {
                    self.procedural_generation_enabled = enabled;
                    if enabled {
                        // Clear loaded chunks to trigger regeneration
                        self.loaded_chunks.clear();
                        info!("Procedural generation ENABLED - chunks will generate");
                    } else {
                        // Clear existing procedural entities
                        let to_delete: Vec<hecs::Entity> = self.ecs.query::<&Procedural>()
                        .iter()
                        .map(|(entity, _)| entity)
                        .collect();
                        for entity in to_delete {
                            let _ = self.ecs.despawn(entity);
                        }
                        self.loaded_chunks.clear();
                        info!("Procedural generation DISABLED");
                    }
                }
                SceneUpdate::SetPlayerStart { position, rotation } => {
                    self.player_start_transform.position = glam::Vec3::from(position);
                    self.player_start_transform.rotation = glam::Quat::from_array(rotation);
                    info!("Player start set to: pos={:?}, rot={:?}", position, rotation);
                }
                SceneUpdate::SpawnLight { id, light_type, position, direction, color, intensity, range, inner_cone, outer_cone } => {
                    // Create a light entity with transform and light component
                    let light_component = LightComponent {
                        light_type,
                        color,
                        intensity,
                        range,
                        inner_cone_angle: inner_cone,
                        outer_cone_angle: outer_cone,
                        cast_shadows: false,
                    };
                    
                    // Calculate rotation from direction for spot/directional lights
                    let rotation = if direction[0].abs() > 0.001 || direction[1].abs() > 0.001 || direction[2].abs() > 0.001 {
                        let dir = glam::Vec3::from(direction).normalize();
                        // Rotation from default forward (-Z) to target direction
                        glam::Quat::from_rotation_arc(glam::Vec3::NEG_Z, dir)
                    } else {
                        glam::Quat::IDENTITY
                    };
                    
                    self.ecs.spawn((
                        EditorEntityId(id),
                        Transform {
                            position: glam::Vec3::from(position),
                            rotation,
                            scale: glam::Vec3::ONE,
                        },
                        light_component,
                    ));
                    info!("Spawned {} light with EditorEntityId({}), color={:?}, intensity={}", 
                          match light_type {
                              LightType::Point => "Point",
                              LightType::Spot => "Spot", 
                              LightType::Directional => "Directional"
                          }, id, color, intensity);
                }
                SceneUpdate::UpdateLight { id, color, intensity, range } => {
                    // Find entity by EditorEntityId and update light properties
                    for (_entity, (editor_id, light)) in self.ecs.query_mut::<(&EditorEntityId, &mut LightComponent)>() {
                        if editor_id.0 == id {
                            if let Some(c) = color {
                                light.color = c;
                            }
                            if let Some(i) = intensity {
                                light.intensity = i;
                            }
                            if let Some(r) = range {
                                light.range = r;
                            }
                            info!("Updated light {} properties", id);
                            break;
                        }
                    }
                }
                SceneUpdate::SpawnSound { id, sound_id, position, volume, looping, max_distance } => {
                    let audio_source = AudioSource {
                        sound_id: sound_id.clone(),
                        volume,
                        looping,
                        max_distance,
                        playing: true,
                    };
                    
                    self.ecs.spawn((
                        EditorEntityId(id),
                        Transform {
                            position: glam::Vec3::from(position),
                            rotation: glam::Quat::IDENTITY,
                            scale: glam::Vec3::ONE,
                        },
                        audio_source,
                    ));
                    info!("Spawned audio source with EditorEntityId({}), sound={}", id, sound_id);
                }
                SceneUpdate::StopSound { id } => {
                    // Find entity by EditorEntityId and stop the sound
                    for (_entity, (editor_id, audio)) in self.ecs.query_mut::<(&EditorEntityId, &mut AudioSource)>() {
                        if editor_id.0 == id {
                            audio.playing = false;
                            info!("Stopped sound {}", id);
                            break;
                        }
                    }
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

    fn generate_chunk(&mut self, cx: i32, cz: i32, size: f32) {
        // Check if procedural generation is enabled
        if !self.procedural_generation_enabled {
            return; // Procedural generation disabled
        }
        
        // Skip center chunks (clear spawn area)
        if cx.abs() <= 1 && cz.abs() <= 1 {
            return; // Don't generate buildings near spawn
        }
        
        // Deterministic generation based on chunk coord
        let mut seed = (cx as u32).wrapping_mul(73856093) ^ (cz as u32).wrapping_mul(19349663);
        let mut rand = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32) / (u32::MAX as f32)
        };
        
        // Spawn buildings/props in this chunk
        // 4x4 subgrid
        let count = 4;
        let step = size / count as f32;
        
        for i in 0..count {
            for j in 0..count {
                if rand() > 0.7 { // 30% chance of building
                    let lx = (i as f32) * step + step * 0.5 - size * 0.5;
                    let lz = (j as f32) * step + step * 0.5 - size * 0.5;
                    
                    let wx = (cx as f32) * size + lx;
                    let wz = (cz as f32) * size + lz;
                    
                    let height = 0.5 + rand() * 5.0; // 0.5 to 5.5m tall
                    
                    self.ecs.spawn((
                        Transform {
                            position: glam::Vec3::new(wx, height * 0.5, wz),
                            rotation: glam::Quat::IDENTITY,
                            scale: glam::Vec3::new(step * 0.8, height, step * 0.8),
                        },
                        MeshHandle(0),
                        Material { color: [0.7, 0.7, 0.8, 1.0] },
                        Procedural,
                    ));
                }
            }
        }
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

pub fn create_primitive(ptype: u8) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    
    // Helper to generate a vertex
    fn make_vert(pos: [f32;3], norm: [f32;3], uv: [f32;2]) -> Vertex {
        Vertex {
            position: pos, normal: norm, uv, color: [1.0, 1.0, 1.0], tangent: [1.0, 0.0, 0.0, 1.0]
        }
    }

    match ptype {
        0 => { // Cube
            // Front
            let off = vertices.len() as u32;
            vertices.push(make_vert([-0.5, -0.5,  0.5], [0.0, 0.0, 1.0], [0.0, 0.0]));
            vertices.push(make_vert([ 0.5, -0.5,  0.5], [0.0, 0.0, 1.0], [1.0, 0.0]));
            vertices.push(make_vert([ 0.5,  0.5,  0.5], [0.0, 0.0, 1.0], [1.0, 1.0]));
            vertices.push(make_vert([-0.5,  0.5,  0.5], [0.0, 0.0, 1.0], [0.0, 1.0]));
            indices.extend_from_slice(&[off, off+1, off+2, off+2, off+3, off]);
            
            // Back
            let off = vertices.len() as u32;
            vertices.push(make_vert([ 0.5, -0.5, -0.5], [0.0, 0.0, -1.0], [0.0, 0.0]));
            vertices.push(make_vert([-0.5, -0.5, -0.5], [0.0, 0.0, -1.0], [1.0, 0.0]));
            vertices.push(make_vert([-0.5,  0.5, -0.5], [0.0, 0.0, -1.0], [1.0, 1.0]));
            vertices.push(make_vert([ 0.5,  0.5, -0.5], [0.0, 0.0, -1.0], [0.0, 1.0]));
            indices.extend_from_slice(&[off, off+1, off+2, off+2, off+3, off]);
            
            // Top
            let off = vertices.len() as u32;
            vertices.push(make_vert([-0.5,  0.5,  0.5], [0.0, 1.0, 0.0], [0.0, 0.0]));
            vertices.push(make_vert([ 0.5,  0.5,  0.5], [0.0, 1.0, 0.0], [1.0, 0.0]));
            vertices.push(make_vert([ 0.5,  0.5, -0.5], [0.0, 1.0, 0.0], [1.0, 1.0]));
            vertices.push(make_vert([-0.5,  0.5, -0.5], [0.0, 1.0, 0.0], [0.0, 1.0]));
            indices.extend_from_slice(&[off, off+1, off+2, off+2, off+3, off]);
            
            // Bottom
            let off = vertices.len() as u32;
            vertices.push(make_vert([-0.5, -0.5, -0.5], [0.0, -1.0, 0.0], [0.0, 0.0]));
            vertices.push(make_vert([ 0.5, -0.5, -0.5], [0.0, -1.0, 0.0], [1.0, 0.0]));
            vertices.push(make_vert([ 0.5, -0.5,  0.5], [0.0, -1.0, 0.0], [1.0, 1.0]));
            vertices.push(make_vert([-0.5, -0.5,  0.5], [0.0, -1.0, 0.0], [0.0, 1.0]));
            indices.extend_from_slice(&[off, off+1, off+2, off+2, off+3, off]);
            
            // Right
            let off = vertices.len() as u32;
            vertices.push(make_vert([ 0.5, -0.5,  0.5], [1.0, 0.0, 0.0], [0.0, 0.0]));
            vertices.push(make_vert([ 0.5, -0.5, -0.5], [1.0, 0.0, 0.0], [1.0, 0.0]));
            vertices.push(make_vert([ 0.5,  0.5, -0.5], [1.0, 0.0, 0.0], [1.0, 1.0]));
            vertices.push(make_vert([ 0.5,  0.5,  0.5], [1.0, 0.0, 0.0], [0.0, 1.0]));
            indices.extend_from_slice(&[off, off+1, off+2, off+2, off+3, off]);

            // Left
            let off = vertices.len() as u32;
            vertices.push(make_vert([-0.5, -0.5, -0.5], [-1.0, 0.0, 0.0], [0.0, 0.0]));
            vertices.push(make_vert([-0.5, -0.5,  0.5], [-1.0, 0.0, 0.0], [1.0, 0.0]));
            vertices.push(make_vert([-0.5,  0.5,  0.5], [-1.0, 0.0, 0.0], [1.0, 1.0]));
            vertices.push(make_vert([-0.5,  0.5, -0.5], [-1.0, 0.0, 0.0], [0.0, 1.0]));
            indices.extend_from_slice(&[off, off+1, off+2, off+2, off+3, off]);
        }
        1 => { // Sphere (UV Sphere)
            let segments = 16;
            let rings = 16;
            for r in 0..=rings {
               let v = r as f32 / rings as f32;
               let theta = std::f32::consts::PI * v; // 0 to PI
               let y = -theta.cos() * 0.5;
               let ring_radius = theta.sin() * 0.5;
               
               for s in 0..=segments {
                   let u = s as f32 / segments as f32;
                   let phi = u * std::f32::consts::PI * 2.0;
                   let x = phi.cos() * ring_radius;
                   let z = phi.sin() * ring_radius;
                   let normal = glam::Vec3::new(x, y, z).normalize().to_array();
                   vertices.push(make_vert([x, y, z], normal, [u, v]));
               }
            }
            for r in 0..rings {
                for s in 0..segments {
                     let next_r = r + 1;
                     let current = r * (segments + 1) + s;
                     let next = next_r * (segments + 1) + s;
                     indices.extend_from_slice(&[
                         current, next, current + 1,
                         next, next + 1, current + 1
                     ]);
                }
            }
        }
        2 | 4 | 5 => { // Cylinder, Capsule (as Cylinder for now), Cone
            let segments = 16;
            let top_radius = if ptype == 5 { 0.0 } else { 0.5 }; // Cone tapers to 0
            let bottom_radius = 0.5;
            let y_min = -0.5;
            let y_max = 0.5;
            
            // Side
            for s in 0..=segments {
                 let u = s as f32 / segments as f32;
                 let angle = u * std::f32::consts::PI * 2.0;
                 let x = angle.cos();
                 let z = angle.sin();
                 
                 // Bottom
                 vertices.push(make_vert([x * bottom_radius, y_min, z * bottom_radius], [x, 0.0, z], [u, 0.0]));
                 // Top
                 vertices.push(make_vert([x * top_radius, y_max, z * top_radius], [x, 0.0, z], [u, 1.0]));
            }
            for s in 0..segments {
                let off = (s * 2) as u32;
                indices.extend_from_slice(&[off, off+1, off+2, off+1, off+3, off+2]);
            }
            
            // Caps (Simple fan)
            // Bottom Cap
            let center_idx = vertices.len() as u32;
            vertices.push(make_vert([0.0, y_min, 0.0], [0.0, -1.0, 0.0], [0.5, 0.5]));
            for s in 0..=segments {
                 let angle = (s as f32 / segments as f32) * std::f32::consts::PI * 2.0;
                 vertices.push(make_vert([angle.cos() * bottom_radius, y_min, angle.sin() * bottom_radius], [0.0, -1.0, 0.0], [0.5 + angle.cos()*0.5, 0.5 + angle.sin()*0.5]));
            }
            for s in 0..segments {
                indices.extend_from_slice(&[center_idx, center_idx + 1 + s as u32, center_idx + 2 + s as u32]);
            }

            // Top Cap (Cylinder only, or if radius > 0)
            if top_radius > 0.001 {
                let center_idx = vertices.len() as u32;
                vertices.push(make_vert([0.0, y_max, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5]));
                for s in 0..=segments {
                     let angle = (s as f32 / segments as f32) * std::f32::consts::PI * 2.0;
                     vertices.push(make_vert([angle.cos() * top_radius, y_max, angle.sin() * top_radius], [0.0, 1.0, 0.0], [0.5 + angle.cos()*0.5, 0.5 + angle.sin()*0.5]));
                }
                for s in 0..segments {
                    indices.extend_from_slice(&[center_idx, center_idx + 2 + s as u32, center_idx + 1 + s as u32]);
                }
            }
        }
        3 => { // Plane
             vertices.push(make_vert([-0.5, 0.0, -0.5], [0.0, 1.0, 0.0], [0.0, 1.0]));
             vertices.push(make_vert([ 0.5, 0.0, -0.5], [0.0, 1.0, 0.0], [1.0, 1.0]));
             vertices.push(make_vert([ 0.5, 0.0,  0.5], [0.0, 1.0, 0.0], [1.0, 0.0]));
             vertices.push(make_vert([-0.5, 0.0,  0.5], [0.0, 1.0, 0.0], [0.0, 0.0]));
             indices.extend_from_slice(&[0, 1, 2, 2, 3, 0]);
        }
        _ => {}
    }

    Mesh {
        vertices: vertices.iter().map(|v| Vertex {
            position: v.position, normal: v.normal, uv: v.uv, color: v.color, tangent: v.tangent
        }).collect(), // Avoid direct clone if struct not clone, but here re-constructing is safer
        indices,
        albedo: None, normal: None, metallic_roughness: None,
        decoded_albedo: None, decoded_normal: None, decoded_mr: None,
    }
}
