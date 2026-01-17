use glam;
use rapier3d;
use hecs::{World, Entity};
use log::{info, warn};
use std::sync::Arc;
use tokio::sync::mpsc;
use crate::physics::PhysicsWorld;

pub mod scripting;
pub mod fbx_loader;
pub mod animation;
pub mod gltf_loader;
use scripting::{FuckScript, ScriptContext, ScriptRegistry};
use animation::{Animator, AnimationState, AnimatorController, AnimationEventQueue};

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
    pub pending_audio_uploads: std::collections::HashMap<String, Vec<u8>>,
    pub pending_texture_uploads: std::collections::HashMap<String, Vec<u8>>,
    pub script_registry: Arc<ScriptRegistry>,
    pub respawn_enabled: bool,
    pub respawn_y: f32,
    pub pending_ui_events: Vec<crate::ui::UiEvent>,
    /// Multi-layer UI system
    pub ui_layers: crate::ui::UiLayerSet,
    /// Menu navigation stack for intermediate menus
    pub menu_stack: crate::ui::MenuStack,
    /// Flag for main.rs to check and release cursor when returning to pause menu
    pub cursor_should_release: bool,
}

pub struct ChunkData {
    pub id: u32,
    pub mesh: Option<Mesh>,
    pub transform: Option<Transform>,
}

#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,
}

#[derive(Clone, Copy, Debug)]
pub struct LocalTransform {
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,
}

#[derive(Clone, Copy, Debug)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Quaternion {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub orientation: Quaternion,
    pub position: Vector3,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub albedo: Option<Vec<u8>>, // Raw image bytes (png/jpg)
    pub normal: Option<Vec<u8>>,
    pub metallic_roughness: Option<Vec<u8>>,
    pub albedo_texture: Option<String>,

    // Bounds for physics/culling
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],

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
    /// Bone indices for skeletal animation (up to 4 influences per vertex)
    #[serde(default)]
    pub bone_indices: [u32; 4],
    /// Bone weights for skeletal animation (normalized, sum to 1.0)
    #[serde(default = "Vertex::default_bone_weights")]
    pub bone_weights: [f32; 4],
}

impl Vertex {
    /// Default bone weights (first bone gets full weight for static meshes)
    fn default_bone_weights() -> [f32; 4] {
        [1.0, 0.0, 0.0, 0.0]
    }

    /// Create a vertex with no bone influence (for static meshes)
    pub fn new_static(position: [f32; 3], normal: [f32; 3], uv: [f32; 2], color: [f32; 3], tangent: [f32; 4]) -> Self {
        Self {
            position,
            normal,
            uv,
            color,
            tangent,
            bone_indices: [0, 0, 0, 0],
            bone_weights: [1.0, 0.0, 0.0, 0.0],
        }
    }

    /// Create a skinned vertex with bone influences
    pub fn new_skinned(
        position: [f32; 3],
        normal: [f32; 3],
        uv: [f32; 2],
        color: [f32; 3],
        tangent: [f32; 4],
        bone_indices: [u32; 4],
        bone_weights: [f32; 4],
    ) -> Self {
        Self {
            position,
            normal,
            uv,
            color,
            tangent,
            bone_indices,
            bone_weights,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Material {
    pub color: [f32; 4],
    pub albedo_texture: Option<String>,  // Texture ID for custom albedo texture
    pub metallic: f32,
    pub roughness: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshHandle(pub usize);

/// Marker component for entities spawned from the editor (for tracking/deletion)
#[derive(Clone, Copy, Debug)]
pub struct EditorEntityId(pub u32);

#[derive(Debug)]
pub struct Procedural; // Tag for procedurally generated entities

/// Hierarchy component for parent-child relationships
#[derive(Clone, Copy, Debug)]
pub struct Hierarchy {
    pub parent: Option<Entity>,
}

/// Camera component for defining view perspectives
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub fov: f32,
    pub active: bool,
}

/// Marker component for entities spawned at engine startup (test scene, etc.)
/// These entities will be cleared when ClearScene is called
#[derive(Debug)]
pub struct StartupScene;

/// Ground plane component - stores the half-extents for shadow frustum calculations
#[derive(Clone, Copy, Debug)]
pub struct GroundPlane {
    /// Half-extents of the ground plane (X and Z size from center)
    pub half_extents: glam::Vec2,
}

impl GroundPlane {
    pub fn new(half_x: f32, half_z: f32) -> Self {
        Self {
            half_extents: glam::Vec2::new(half_x, half_z),
        }
    }
    
    /// Returns the maximum extent (for shadow frustum sizing)
    pub fn max_extent(&self) -> f32 {
        self.half_extents.x.max(self.half_extents.y)
    }
}

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
    pub speed: f32,     // Current speed
    pub max_speed: f32, // Maximum speed
    pub steering: f32,
    pub accelerating: bool, // Is the vehicle accelerating?
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentState {
    Idle,
    Walking,
    Running,
    Fleeing,   // Running away from danger
    Chasing,   // Following a target closely
    Approaching, // Moving toward a point of interest
}

#[derive(Clone, Debug)]
pub struct CrowdAgent {
    pub velocity: glam::Vec3,
    pub target: glam::Vec3,
    pub state: AgentState,
    pub max_speed: f32, // Speed varies by state
    pub stuck_timer: f32,
    pub last_pos: glam::Vec3,
}

#[derive(Clone, Copy, Debug)]
pub struct Player;

#[derive(Clone, Copy, Debug)]
pub struct Obstacle;

// Collision Layers
pub const LAYER_DEFAULT: u32 = 1;
pub const LAYER_ENVIRONMENT: u32 = 2;
pub const LAYER_PROP: u32 = 4;
pub const LAYER_CHARACTER: u32 = 8;
pub const LAYER_VEHICLE: u32 = 16;

/// Component to hold scripts on an entity
pub struct DynamicScript {
    pub script: Option<Box<dyn FuckScript>>,
    pub started: bool,
    pub enabled: bool,
}

impl DynamicScript {
    pub fn new(script: Box<dyn FuckScript>) -> Self {
        Self { script: Some(script), started: false, enabled: true }
    }
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
    pub range: f32,            // Attenuation distance (Point/Spot)
    pub inner_cone_angle: f32, // Spot light inner cone (radians)
    pub outer_cone_angle: f32, // Spot light outer cone (radians)
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
    pub fn spot(
        color: [f32; 3],
        intensity: f32,
        range: f32,
        inner_angle: f32,
        outer_angle: f32,
    ) -> Self {
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
    /// Runtime handle to the playing source (not serialized)
    #[serde(skip)]
    pub runtime_handle: Option<u32>,
}

impl Default for AudioSource {
    fn default() -> Self {
        Self {
            sound_id: String::new(),
            volume: 1.0,
            looping: false,
            max_distance: 50.0,
            playing: false,
            runtime_handle: None,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub enum SceneUpdate {
    Spawn {
        id: u32,
        primitive: u8,
        position: [f32; 3],
        rotation: [f32; 4],
        scale: [f32; 3],
        color: [f32; 3],
        albedo_texture: Option<String>,  // Texture ID for custom albedo texture
        collision_enabled: bool,
        layer: u32,
        #[serde(default)]
        is_static: bool,  // If true, object is not affected by gravity
    },
    SpawnMesh {
        id: u32,
        mesh: Mesh,
        position: [f32; 3],
        rotation: [f32; 4],
    },
    UpdateTransform {
        id: u32,
        position: Option<[f32; 3]>,
        rotation: Option<[f32; 4]>,
        scale: Option<[f32; 3]>,
    },
    DeleteEntity {
        id: u32,
    },
    ClearScene,
    /// Toggle procedural generation on/off (off by default)
    SetProceduralGeneration {
        enabled: bool,
    },
    SetPlayerStart {
        position: [f32; 3],
        rotation: [f32; 4],
    },
    /// Spawn a dynamic light
    SpawnLight {
        id: u32,
        light_type: LightType,
        position: [f32; 3],
        direction: [f32; 3], // For spot/directional
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
    /// Play a specific animation clip on an entity for preview
    PreviewAnimation {
        id: u32,
        clip_index: usize,
    },
    /// Preview animation at specific time (for editor scrubbing)
    PreviewAnimationAt {
        id: u32,
        clip_index: usize,
        time: f32,
    },
    /// Attach an entity to a parent
    AttachEntity {
        id: u32,
        parent_id: Option<u32>,
    },
    /// Set an entity's camera as the active rendering camera
    SetActiveCamera {
        id: u32,
        fov: f32,
    },
    /// Stop a playing sound
    StopSound {
        id: u32,
    },
    /// Upload audio data to the engine
    UploadSound {
        id: String,
        data: Vec<u8>,
    },
    /// Upload texture data to the engine
    UploadTexture {
        id: String,
        data: Vec<u8>,
    },
    /// Attach a script by name to an entity
    AttachScript {
        id: u32,
        name: String,
    },
    /// Attach an animator controller to an entity
    AttachAnimator {
        id: u32,
        config: crate::world::animation::AnimatorConfig,
    },
    /// Spawn a ground plane with shadow frustum sizing
    SpawnGroundPlane {
        id: u32,
        primitive: u8,
        position: [f32; 3],
        scale: [f32; 3],
        color: [f32; 3],
        half_extents: [f32; 2], // X and Z half-extents for shadow frustum
        albedo_texture: Option<String>, // Texture ID for custom albedo texture
        collision_enabled: bool,
        layer: u32,
    },
    /// Update an entity's material properties in real-time
    UpdateMaterial {
        id: u32,
        color: Option<[f32; 4]>,
        albedo_texture: Option<Option<String>>, // Some(Some(id)) to set, Some(None) to clear, None for no change
        metallic: Option<f32>,
        roughness: Option<f32>,
    },
    SetCollision {
        id: u32,
        enabled: bool,
        layer: u32,
    },
    /// Spawn an FBX mesh with full model data
    SpawnFbxMesh {
        id: u32,
        mesh_data: Vec<u8>,  // Raw FBX/OBJ file bytes
        position: [f32; 3],
        rotation: [f32; 4],
        scale: [f32; 3],
        #[serde(default)]
        albedo_texture: Option<String>,  // Optional texture ID
        #[serde(default)]
        collision_enabled: bool,
        #[serde(default)]
        layer: u32,
        #[serde(default)]
        is_static: bool,
    },
    /// Spawn a glTF/GLB mesh with full model data and animation support
    SpawnGltfMesh {
        id: u32,
        mesh_data: Vec<u8>,  // Raw GLB/GLTF file bytes
        position: [f32; 3],
        rotation: [f32; 4],
        scale: [f32; 3],
        #[serde(default)]
        collision_enabled: bool,
        #[serde(default)]
        layer: u32,
        #[serde(default)]
        is_static: bool,
    },
    SetRespawnSettings {
        enabled: bool,
        y_threshold: f32,
    },
    /// Set the UI layout for the default Hud layer (backwards compatible)
    SetUiLayout {
        layout: crate::ui::UiLayout,
    },
    /// Set a layout for a specific UI layer
    SetUiLayer {
        layer: crate::ui::UiLayer,
        layout: crate::ui::UiLayout,
    },
    /// Show a UI layer (make it visible)
    ShowUiLayer {
        layer: crate::ui::UiLayer,
    },
    /// Hide a UI layer
    HideUiLayer {
        layer: crate::ui::UiLayer,
    },
    /// Load a UI scene from file and set it as a layer
    LoadUiScene {
        layer: crate::ui::UiLayer,
        scene_path: String,
    },
    /// Push a menu onto the stack (menu_load in FuckScript)
    MenuLoad {
        alias: String,
    },
    /// Pop the current menu (go back)
    MenuBack,
    /// Show a popup by alias (for keybind-triggered overlays)
    ShowPopup {
        alias: String,
    },
    /// Hide a popup by alias
    HidePopup {
        alias: String,
    },
    /// Reset the pause menu to the default layout
    ResetPauseMenu,
}

// Assuming GameWorld struct definition is somewhere above this,
// and looks something like this (inferred from `new` function):
// pub struct GameWorld {
//     pub ecs: World,
//     pub chunk_receiver: mpsc::Receiver<Vec<(Transform, Mesh, Material)>>,
//     pub chunk_sender: mpsc::Sender<Vec<(Transform, Mesh, Material)>>,
//     pub command_receiver: mpsc::Receiver<SceneUpdate>,
//     pub command_sender: mpsc::Sender<SceneUpdate>,
//     pub loaded_chunks: std::collections::HashSet<(i32, i32)>,
//     pub procedural_generation_enabled: bool,
//     pub player_start_transform: Transform,
//     pub pending_audio_uploads: std::collections::HashMap<String, Vec<u8>>,
//     pub pending_texture_uploads: std::collections::HashMap<String, Vec<u8>>,
//     pub script_registry: Arc<ScriptRegistry>,
//     pub respawn_enabled: bool,
//     pub respawn_y: f32,
// }

// The user's instruction implies adding `pending_ui_events` to the GameWorld struct.
// Since the struct definition is not provided, I will add it to the `new` function
// and assume the struct definition will be updated elsewhere or is implicitly handled.
// However, the provided "Code Edit" snippet attempts to redefine `GameWorld` and its `new` function
// in a syntactically incorrect way. I will interpret the instruction as adding the field
// to the existing `GameWorld` struct and initializing it in the `new` function.

// To make the change syntactically correct, I will add the field to the `GameWorld` struct
// and initialize it in the `new` function. Since the `GameWorld` struct definition is not
// in the provided content, I will add the initialization to the `new` function.
// If the struct definition were present, the change would be made there.
// Given the provided "Code Edit" snippet, it seems the user intended to add the field
// and initialize it. I will add the initialization to the existing `new` function.

// If the GameWorld struct definition was available, the change would look like this:
// pub struct GameWorld {
//     // ... existing fields ...
//     pub pending_ui_events: Vec<crate::ui::UiEvent>,
// }

impl GameWorld {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::channel(10);
        let (cmd_tx, cmd_rx) = mpsc::channel(100);
        let world = Self {
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
                scale: glam::Vec3::ONE,
            },
            pending_audio_uploads: std::collections::HashMap::new(),
            pending_texture_uploads: std::collections::HashMap::new(),
            script_registry: {
                let mut reg = ScriptRegistry::new();
                reg.register("TestBounce", || scripting::TestBounceScript::new());
                reg.register("CrowdAgent", || scripting::CrowdAgentScript);
                reg.register("Vehicle", || scripting::VehicleScript);
                reg.register("PoliceAgent", || scripting::PoliceAgentScript);
                reg.register("TrafficAI", || scripting::TrafficAIScript);
                reg.register("WeaponNPC", || scripting::WeaponNPCScript);
                reg.register("CollisionLogger", || scripting::CollisionLoggerScript);
                reg.register("TouchToDestroy", || scripting::TouchToDestroyScript);
                Arc::new(reg)
            },
            respawn_enabled: false,
            respawn_y: -50.0,
            pending_ui_events: Vec::new(),
            ui_layers: crate::ui::UiLayerSet::new(),
            menu_stack: crate::ui::MenuStack::new(),
            cursor_should_release: false,
        };
        // world.spawn_default_scene(); // Moved to streaming logic or separate init
        world
    }

    /// Helper to find an entity by its EditorEntityId
    pub fn find_by_editor_id(&self, id: u32) -> Option<Entity> {
        for (entity, editor_id) in self.ecs.query::<&EditorEntityId>().iter() {
            if editor_id.0 == id {
                return Some(entity);
            }
        }
        None
    }

    /// Propagate world-space transforms from parents to children
    pub fn resolve_hierarchies(&mut self) {
        // Multi-pass to handle nested hierarchies (up to 8 levels)
        for _ in 0..8 {
            let mut updates = Vec::new();
            
            for (entity, (hierarchy, local_transform)) in self.ecs.query::<(&Hierarchy, &LocalTransform)>().iter() {
                if let Some(parent) = hierarchy.parent {
                    if let Ok(parent_transform) = self.ecs.get::<&Transform>(parent) {
                        // Calculate world-space transform:
                        // world_pos = parent_pos + (parent_rot * (parent_scale * local_pos))
                        let world_pos = parent_transform.position + (parent_transform.rotation * (parent_transform.scale * local_transform.position));
                        let world_rot = parent_transform.rotation * local_transform.rotation;
                        let world_scale = parent_transform.scale * local_transform.scale;
                        
                        // Only add to updates if it actually changed to avoid infinite loops
                        if let Ok(t) = self.ecs.get::<&Transform>(entity) {
                            if (t.position - world_pos).length_squared() > 0.0001 || 
                               (t.rotation.dot(world_rot)).abs() < 0.9999 ||
                               (t.scale - world_scale).length_squared() > 0.0001 {
                                updates.push((entity, world_pos, world_rot, world_scale));
                            }
                        } else {
                            updates.push((entity, world_pos, world_rot, world_scale));
                        }
                    }
                }
            }
            
            if updates.is_empty() { break; }
            
            for (entity, p, r, s) in updates {
                if let Ok(mut t) = self.ecs.get::<&mut Transform>(entity) {
                    t.position = p;
                    t.rotation = r;
                    t.scale = s;
                }
            }
        }
    }

    pub fn update_streaming(&mut self, player_pos: glam::Vec3, physics: &mut PhysicsWorld) {
        // Chunk Settings
        let chunk_size = 16.0; // 16 meters
        let load_radius = 4; // 4 chunks radius

        // Calculate player chunk coord
        let pc_x = (player_pos.x / chunk_size).floor() as i32;
        let pc_z = (player_pos.z / chunk_size).floor() as i32;

        let mut chunks_to_generate = Vec::new();
        for x in (pc_x - load_radius)..=(pc_x + load_radius) {
            for z in (pc_z - load_radius)..=(pc_z + load_radius) {
                if !self.loaded_chunks.contains(&(x, z)) {
                    chunks_to_generate.push((x, z));
                }
            }
        }

        if !chunks_to_generate.is_empty() {
            use rayon::prelude::*;
            // Parallelize the data generation for chunks (Read-only data access)
            // Use a local copy of procedural settings to avoid capturing &mut self
            let procedural_enabled = self.procedural_generation_enabled;
            
            let chunk_data: Vec<_> = chunks_to_generate.par_iter().map(|&(x, z)| {
                if !procedural_enabled { return Vec::new(); }
                self.calculate_chunk_entities(x, z, chunk_size)
            }).collect();

            // Sequential update of the world state
            for (i, entities) in chunk_data.into_iter().enumerate() {
                let (x, z) = chunks_to_generate[i];
                self.loaded_chunks.insert((x, z));
                for (transform, mesh, material) in entities {
                    self.ecs.spawn((transform, mesh, material, Procedural));
                }
            }
        }

        // Unloading can be added here (check if chunk is far)
        // ...

        // Process Scene Updates (Networking)
        while let Ok(cmd) = self.command_receiver.try_recv() {
            // info!("Processing command: {:?}", cmd); // Commented out to prevent massive binary dumps // Removed to prevent logging raw binary data from UploadSound
            match cmd {
                SceneUpdate::Spawn {
                    id,
                    primitive,
                    position,
                    rotation,
                    scale,
                    color,
                    albedo_texture,
                    collision_enabled,
                    layer,
                    is_static,
                } => {
                    self.ecs.spawn((
                        EditorEntityId(id), // Track this entity's ID for deletion
                        Transform {
                            position: glam::Vec3::from(position),
                            rotation: glam::Quat::from_array(rotation),
                            scale: glam::Vec3::from(scale),
                        },
                        LocalTransform {
                            position: glam::Vec3::from(position),
                            rotation: glam::Quat::from_array(rotation),
                            scale: glam::Vec3::from(scale),
                        },
                        MeshHandle(primitive as usize), // Use requested primitive
                        Material {
                            color: [color[0], color[1], color[2], 1.0],
                            albedo_texture,
                            metallic: 0.0,
                            roughness: 0.5,
                        },
                    ));

                    let target_entity = self.ecs.query::<&EditorEntityId>().iter().find(|(_, id_comp)| id_comp.0 == id).map(|(e, _)| e);
                    if let Some(entity) = target_entity {
                        if collision_enabled {
                            // Use is_static to control gravity. Planes (3) are always static.
                            let is_dynamic = !is_static && primitive != 3;
                            self.attach_physics_to_entity(entity, physics, primitive, position, rotation, scale, is_dynamic, layer);
                        }
                    }

                    info!(
                        "Spawned entity with EditorEntityId({}) using MeshHandle({}) (scale: {:?}, collision: {}, static: {})",
                        id, primitive, scale, collision_enabled, is_static
                    );
                }
                SceneUpdate::SpawnMesh {
                    id,
                    mesh,
                    position,
                    rotation,
                } => {
                    self.ecs.spawn((
                        EditorEntityId(id), // Track this entity's ID for deletion
                        Transform {
                            position: glam::Vec3::from(position),
                            rotation: glam::Quat::from_array(rotation),
                            scale: glam::Vec3::ONE,
                        },
                        LocalTransform {
                            position: glam::Vec3::from(position),
                            rotation: glam::Quat::from_array(rotation),
                            scale: glam::Vec3::ONE,
                        },
                        mesh,
                        Material {
                            color: [1.0, 1.0, 1.0, 1.0],
                            albedo_texture: None,
                            metallic: 0.0,
                            roughness: 0.5,
                        }, // Default white for mesh
                    ));
                    info!("Spawned mesh with EditorEntityId({})", id);
                }
                SceneUpdate::UpdateTransform { id, position, rotation, scale } => {
                    // Find entity by EditorEntityId and update transform
                    for (entity, (editor_id, transform)) in
                        self.ecs.query_mut::<(&EditorEntityId, &mut Transform)>()
                    {
                        if editor_id.0 == id {
                            if let Some(pos) = position {
                                transform.position = glam::Vec3::from(pos);
                            }
                            if let Some(rot) = rotation {
                                transform.rotation = glam::Quat::from_array(rot);
                            }
                            if let Some(s) = scale {
                                transform.scale = glam::Vec3::from(s);
                            }
                            
                            // Update LocalTransform too
                            if let Ok(mut local) = self.ecs.get::<&mut LocalTransform>(entity) {
                                if let Some(pos) = position { local.position = glam::Vec3::from(pos); }
                                if let Some(rot) = rotation { local.rotation = glam::Quat::from_array(rot); }
                                if let Some(s) = scale { local.scale = glam::Vec3::from(s); }
                            }

                            // Update physics if it exists
                            if let Ok(rb_handle) = self.ecs.get::<&RigidBodyHandle>(entity) {
                                physics.set_rigid_body_transform(rb_handle.0, position, rotation);
                            }

                            info!("Updated transform for entity {} (pos: {:?}, rot: {:?}, scale: {:?})", id, position, rotation, scale);
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
                        // Call on_disable before despawn
                        let mut script = None;
                        if let Ok(mut comp) = self.ecs.get::<&mut DynamicScript>(entity) {
                            script = comp.script.take();
                        }
                        if let Some(mut s) = script {
                            let mut ctx = ScriptContext::new(entity, &mut self.ecs, physics, 0.0);
                            s.on_disable(&mut ctx);
                        }
                        
                        // Remove physics rigid body if it exists
                        if let Ok(rb_handle) = self.ecs.get::<&RigidBodyHandle>(entity) {
                            physics.remove_rigid_body(rb_handle.0);
                        }

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
                    for (entity, _) in self.ecs.query::<&StartupScene>().iter() {
                        to_delete.push(entity);
                    }

                    let count = to_delete.len();
                    for entity in to_delete {
                        // Call on_disable before despawn
                        let mut script = None;
                        if let Ok(mut comp) = self.ecs.get::<&mut DynamicScript>(entity) {
                            script = comp.script.take();
                        }
                        if let Some(mut s) = script {
                            let mut ctx = ScriptContext::new(entity, &mut self.ecs, physics, 0.0);
                            s.on_disable(&mut ctx);
                        }
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
                        let to_delete: Vec<hecs::Entity> = self
                            .ecs
                            .query::<&Procedural>()
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
                    info!(
                        "Player start set to: pos={:?}, rot={:?}",
                        position, rotation
                    );
                }
                SceneUpdate::SpawnLight {
                    id,
                    light_type,
                    position,
                    direction,
                    color,
                    intensity,
                    range,
                    inner_cone,
                    outer_cone,
                } => {
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
                    let rotation = if direction[0].abs() > 0.001
                        || direction[1].abs() > 0.001
                        || direction[2].abs() > 0.001
                    {
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
                    info!(
                        "Spawned {} light with EditorEntityId({}), color={:?}, intensity={}",
                        match light_type {
                            LightType::Point => "Point",
                            LightType::Spot => "Spot",
                            LightType::Directional => "Directional",
                        },
                        id,
                        color,
                        intensity
                    );
                }
                SceneUpdate::UpdateLight {
                    id,
                    color,
                    intensity,
                    range,
                } => {
                    // Find entity by EditorEntityId and update light properties
                    for (_entity, (editor_id, light)) in self
                        .ecs
                        .query_mut::<(&EditorEntityId, &mut LightComponent)>()
                    {
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
                SceneUpdate::SpawnSound {
                    id,
                    sound_id,
                    position,
                    volume,
                    looping,
                    max_distance,
                } => {
                    let audio_source = AudioSource {
                        sound_id: sound_id.clone(),
                        volume,
                        looping,
                        max_distance,
                        playing: true,
                        runtime_handle: None,
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
                    info!(
                        "Spawned audio source with EditorEntityId({}), sound={}",
                        id, sound_id
                    );
                }
                SceneUpdate::StopSound { id } => {
                    // Find entity by EditorEntityId and stop the sound
                    for (_entity, (editor_id, audio)) in
                        self.ecs.query_mut::<(&EditorEntityId, &mut AudioSource)>()
                    {
                        if editor_id.0 == id {
                            audio.playing = false;
                            info!("Stopped sound {}", id);
                            break;
                        }
                    }
                }
                SceneUpdate::UploadSound { id, data } => {
                    info!("Received audio upload: {} ({} bytes)", id, data.len());
                    self.pending_audio_uploads.insert(id, data);
                }
                SceneUpdate::UploadTexture { id, data } => {
                    info!("Received texture upload: {} ({} bytes)", id, data.len());
                    self.pending_texture_uploads.insert(id, data);
                }
                SceneUpdate::AttachScript { id, name } => {
                    if let Some(script_box) = self.script_registry.create(&name) {
                        // Find entity by EditorEntityId
                        let mut target_entity = None;
                        for (entity, editor_id) in self.ecs.query::<&EditorEntityId>().iter() {
                            if editor_id.0 == id {
                                target_entity = Some(entity);
                                break;
                            }
                        }

                        if let Some(entity) = target_entity {
                            // Call on_disable on old script if it exists
                            let mut old_script = None;
                            if let Ok(mut comp) = self.ecs.get::<&mut DynamicScript>(entity) {
                                old_script = comp.script.take();
                            }
                            if let Some(mut s) = old_script {
                                let mut ctx = ScriptContext::new(entity, &mut self.ecs, physics, 0.0);
                                s.on_disable(&mut ctx);
                            }
                            self.ecs.insert_one(entity, DynamicScript::new(script_box)).expect("Failed to attach script");
                            info!("Attached script '{}' to entity {}", name, id);
                        } else {
                            warn!("AttachScript: No entity found with id {}", id);
                        }
                    } else {
                        warn!("AttachScript: Script '{}' not found in registry", name);
                    }
                }
                SceneUpdate::AttachAnimator { id, config } => {
                    // Find entity by EditorEntityId
                    let mut target_entity = None;
                    for (entity, editor_id) in self.ecs.query::<&EditorEntityId>().iter() {
                        if editor_id.0 == id {
                            target_entity = Some(entity);
                            break;
                        }
                    }

                    if let Some(entity) = target_entity {
                        use crate::world::animation::*;
                        
                        // Create state machine from config
                        let mut state_machine = AnimationStateMachine::new();
                        
                        // Add states
                        for state_cfg in &config.states {
                            state_machine.add_state(&state_cfg.name, state_cfg.clip_index);
                            
                            // Automatically add a transition from ANY to this state if a trigger is provided
                            if let Some(ref trigger) = state_cfg.trigger_param {
                                state_machine.transitions.push(
                                    StateTransition::new(
                                        "ANY",
                                        &state_cfg.name,
                                        TransitionCondition::Trigger(trigger.clone()),
                                        0.2, // Default transition duration
                                    )
                                );
                            }
                        }
                        
                        // Add transitions
                        for trans_cfg in &config.transitions {
                            let condition: TransitionCondition = trans_cfg.condition.clone().into();
                            state_machine.transitions.push(
                                StateTransition::new(
                                    &trans_cfg.from_state,
                                    &trans_cfg.to_state,
                                    condition,
                                    trans_cfg.duration,
                                ).with_priority(trans_cfg.priority)
                            );
                        }
                        
                        // Add parameters
                        for (name, param_cfg) in &config.parameters {
                            match param_cfg {
                                AnimParamConfig::Float(v) => state_machine.set_float(name, *v),
                                AnimParamConfig::Bool(v) => state_machine.set_bool(name, *v),
                                AnimParamConfig::Int(v) => state_machine.set_int(name, *v),
                                AnimParamConfig::Trigger => state_machine.set_trigger(name),
                            }
                        }
                        
                        // Set default state
                        if !config.default_state.is_empty() {
                            state_machine.current_state = config.default_state.clone();
                        }
                        
                        // Create controller
                        let mut controller = AnimatorController::new(state_machine);
                        controller.apply_root_motion = config.apply_root_motion;
                        controller.speed = config.speed;
                        
                        // Attach to entity
                        let _ = self.ecs.insert_one(entity, controller);
                        info!("Attached AnimatorController to entity {} with {} states, {} transitions", 
                              id, config.states.len(), config.transitions.len());
                    } else {
                        warn!("AttachAnimator: No entity found with id {}", id);
                    }
                }

                SceneUpdate::PreviewAnimation { id, clip_index } => {
                    if let Some(entity) = self.find_by_editor_id(id) {
                        if let Ok(mut animator) = self.ecs.get::<&mut Animator>(entity) {
                            animator.play(clip_index);
                            info!("Previewing animation clip {} on entity {}", clip_index, id);
                        }
                    }
                }

                SceneUpdate::PreviewAnimationAt { id, clip_index, time } => {
                    if let Some(entity) = self.find_by_editor_id(id) {
                        if let Ok(mut animator) = self.ecs.get::<&mut Animator>(entity) {
                            animator.play(clip_index);
                            animator.set_time(time);
                            // Set playing to false to stop automatic time updates during scrubbing
                            animator.playing = false; 
                            info!("Scrubbing animation clip {} on entity {} to time {}", clip_index, id, time);
                        }
                    }
                }

                SceneUpdate::AttachEntity { id, parent_id } => {
                    if let Some(entity) = self.find_by_editor_id(id) {
                        let parent = parent_id.and_then(|pid| self.find_by_editor_id(pid));
                        let _ = self.ecs.insert_one(entity, Hierarchy { parent });
                        info!("Attached entity {} to parent {:?}", id, parent_id);
                    }
                }

                SceneUpdate::SetActiveCamera { id, fov } => {
                    if let Some(entity) = self.find_by_editor_id(id) {
                        // Deactivate other cameras
                        for (e, cam) in self.ecs.query::<&mut Camera>().iter() {
                            if e != entity {
                                cam.active = false;
                            }
                        }
                        
                        let _ = self.ecs.insert_one(entity, Camera { fov, active: true });
                        info!("Set entity {} as active camera (FOV: {})", id, fov);
                    }
                }

                SceneUpdate::SpawnGroundPlane {
                    id,
                    primitive,
                    position,
                    scale,
                    color,
                    half_extents,
                    albedo_texture,
                    collision_enabled,
                    layer,
                } => {
                    self.ecs.spawn((
                        EditorEntityId(id),
                        Transform {
                            position: glam::Vec3::from(position),
                            rotation: glam::Quat::IDENTITY,
                            scale: glam::Vec3::from(scale),
                        },
                        MeshHandle(primitive as usize),
                        Material {
                            color: [color[0], color[1], color[2], 1.0],
                            albedo_texture,
                            metallic: 0.0,
                            roughness: 0.5,
                        },
                        GroundPlane::new(half_extents[0], half_extents[1]),
                    ));

                    let target_entity = self.ecs.query::<&EditorEntityId>().iter().find(|(_, id_comp)| id_comp.0 == id).map(|(e, _)| e);
                    if let Some(entity) = target_entity {
                        if collision_enabled {
                            // Ground is usually Environment
                            self.attach_physics_to_entity(entity, physics, primitive, position, [0.0, 0.0, 0.0, 1.0], scale, false, layer);
                        }
                    }

                    info!(
                        "Spawned GroundPlane with EditorEntityId({}) - half_extents: {:?} (collision: {})",
                        id, half_extents, collision_enabled
                    );
                }
                SceneUpdate::UpdateMaterial { id, color, albedo_texture, metallic, roughness } => {
                    for (_entity, (editor_id, mat)) in self.ecs.query_mut::<(&EditorEntityId, &mut Material)>() {
                        if editor_id.0 == id {
                            if let Some(c) = color {
                                mat.color = c;
                            }
                            if let Some(tex) = albedo_texture {
                                mat.albedo_texture = tex;
                            }
                            if let Some(m) = metallic {
                                mat.metallic = m;
                            }
                            if let Some(r) = roughness {
                                mat.roughness = r;
                            }
                            info!("Updated material for entity {}", id);
                            break;
                        }
                    }
                }
                SceneUpdate::SetCollision { id, enabled, layer } => {
                    let mut target_entity = None;
                    for (entity, editor_id) in self.ecs.query::<&EditorEntityId>().iter() {
                        if editor_id.0 == id {
                            target_entity = Some(entity);
                            break;
                        }
                    }

                    if let Some(entity) = target_entity {
                        // Always remove existing body to ensure clean state update (layer change or disable)
                         let mut rb_to_remove = None;
                        if let Ok(rb_handle) = self.ecs.get::<&RigidBodyHandle>(entity) {
                            rb_to_remove = Some(rb_handle.0);
                        }
                        if let Some(handle) = rb_to_remove {
                            physics.remove_rigid_body(handle);
                            let _ = self.ecs.remove_one::<RigidBodyHandle>(entity);
                            // log only if we are actually disabling
                            if !enabled {
                                info!("Disabled collision for entity {}", id);
                            }
                        }

                        if enabled {
                            let transform = *self.ecs.get::<&Transform>(entity).expect("Entity must have transform");
                            let primitive = if let Ok(h) = self.ecs.get::<&MeshHandle>(entity) { h.0 as u8 } else { 0 };
                            // Infer dynamic. For primitives in editor, we used true unless ground (primitive 3).
                            let is_dynamic = primitive != 3; 
                            self.attach_physics_to_entity(entity, physics, primitive, transform.position.into(), transform.rotation.into(), transform.scale.into(), is_dynamic, layer);
                            info!("Updated collision for entity {} (Layer: {})", id, layer);
                        }
                    }
                }
                SceneUpdate::SpawnGltfMesh {
                    id,
                    mesh_data,
                    position,
                    rotation,
                    scale,
                    collision_enabled,
                    layer,
                    is_static,
                } => {
                    match gltf_loader::load_gltf_with_animations(&mesh_data) {
                        Ok(model_scene) => {
                            // Merge meshes for rendering
                            let mut mesh = fbx_loader::merge_fbx_meshes(&model_scene);
                            
                            if mesh.vertices.is_empty() {
                                warn!("glTF file contains no mesh data for entity {}", id);
                            } else {
                                let vert_count = mesh.vertices.len();
                                let idx_count = mesh.indices.len();
                                
                                // Find the texture bytes for this mesh's albedo_texture
                                if let Some(tex_name) = &mesh.albedo_texture {
                                    for texture in &model_scene.textures {
                                        if &texture.name == tex_name {
                                            mesh.albedo = Some(texture.data.clone());
                                            break;
                                        }
                                    }
                                }
                                
                                // Fallback: if no texture was found by name but textures exist, use the first one
                                if mesh.albedo.is_none() && !model_scene.textures.is_empty() {
                                    let first_texture = &model_scene.textures[0];
                                    mesh.albedo = Some(first_texture.data.clone());
                                    if mesh.albedo_texture.is_none() {
                                        mesh.albedo_texture = Some(first_texture.name.clone());
                                    }
                                }

                                let albedo_texture = mesh.albedo_texture.clone();

                                let entity = self.ecs.spawn((
                                    EditorEntityId(id),
                                    Transform {
                                        position: glam::Vec3::from(position),
                                        rotation: glam::Quat::from_array(rotation),
                                        scale: glam::Vec3::from(scale),
                                    },
                                    mesh,
                                    Material {
                                        color: [1.0, 1.0, 1.0, 1.0],
                                        albedo_texture,
                                        metallic: 0.0,
                                        roughness: 0.5,
                                    },
                                ));

                                // Store skeleton and animations if present for the Animator to find later
                                if let Some(skeleton) = model_scene.skeleton {
                                    self.ecs.insert_one(entity, skeleton).unwrap();
                                }
                                
                                if !model_scene.animations.is_empty() {
                                    // Normally we would store these in a global registry or on the entity
                                    // For now, let's just log it
                                    info!("glTF model {} loaded with {} animations", id, model_scene.animations.len());
                                }

                                if collision_enabled {
                                    self.attach_physics_to_entity(
                                        entity,
                                        physics,
                                        255, // 255 = custom mesh using its AABB
                                        position,
                                        rotation,
                                        scale,
                                        !is_static,
                                        layer,
                                    );
                                }

                                info!(
                                    "Spawned glTF mesh EditorEntityId({}) at pos {:?} scale {:?} - {} vertices, {} indices (collision: {}, static: {})",
                                    id,
                                    position,
                                    scale,
                                    vert_count,
                                    idx_count,
                                    collision_enabled,
                                    is_static
                                );
                            }
                        }
                        Err(e) => {
                            warn!("Failed to load glTF for entity {}: {}", id, e);
                        }
                    }
                }
                SceneUpdate::SetRespawnSettings { enabled, y_threshold } => {
                    self.respawn_enabled = enabled;
                    self.respawn_y = y_threshold;
                    info!("Respawn settings updated: enabled={}, y={}", enabled, y_threshold);
                }
                SceneUpdate::SpawnFbxMesh {
                    id,
                    mesh_data,
                    position,
                    rotation,
                    scale,
                    albedo_texture,
                    collision_enabled,
                    layer,
                    is_static,
                } => {
                    match fbx_loader::load_fbx_from_bytes(&mesh_data) {
                        Ok(fbx_scene) => {
                            // Merge all meshes from the FBX file
                            let mut mesh = fbx_loader::merge_fbx_meshes(&fbx_scene);
                            mesh.albedo_texture = albedo_texture.clone();
                            
                            if mesh.vertices.is_empty() {
                                warn!("FBX file contains no mesh data for entity {}", id);
                            } else {
                                let vert_count = mesh.vertices.len();
                                let idx_count = mesh.indices.len();
                                let entity = self.ecs.spawn((
                                    EditorEntityId(id),
                                    Transform {
                                        position: glam::Vec3::from(position),
                                        rotation: glam::Quat::from_array(rotation),
                                        scale: glam::Vec3::from(scale),
                                    },
                                    mesh,
                                    Material {
                                        color: [1.0, 1.0, 1.0, 1.0],
                                        albedo_texture,
                                        metallic: 0.0,
                                        roughness: 0.5,
                                    },
                                ));

                                if collision_enabled {
                                    self.attach_physics_to_entity(
                                        entity,
                                        physics,
                                        255, // 255 = custom mesh using its AABB
                                        position,
                                        rotation,
                                        scale,
                                        !is_static,
                                        layer,
                                    );
                                }

                                info!(
                                    "Spawned mesh EditorEntityId({}) at pos {:?} scale {:?} - {} vertices, {} indices (collision: {}, static: {})",
                                    id,
                                    position,
                                    scale,
                                    vert_count,
                                    idx_count,
                                    collision_enabled,
                                    is_static
                                );
                            }
                        }
                        Err(e) => {
                            warn!("Failed to load FBX for entity {}: {}", id, e);
                        }
                    }
                }
                SceneUpdate::SetUiLayout { layout } => {
                    // Backwards compatible: SetUiLayout sets the Hud layer and makes it visible
                    self.ui_layers.set_layer(crate::ui::UiLayer::Hud, layout);
                    self.ui_layers.show(crate::ui::UiLayer::Hud);
                    info!("UI Layout updated (Hud layer): {} buttons, {} panels, {} texts", 
                        self.ui_layers.get_layer(&crate::ui::UiLayer::Hud).map(|l| l.buttons.len()).unwrap_or(0),
                        self.ui_layers.get_layer(&crate::ui::UiLayer::Hud).map(|l| l.panels.len()).unwrap_or(0),
                        self.ui_layers.get_layer(&crate::ui::UiLayer::Hud).map(|l| l.texts.len()).unwrap_or(0)
                    );
                }
                SceneUpdate::SetUiLayer { layer, layout } => {
                    info!("Setting UI layer {:?}", layer);
                    let should_show = matches!(layout.layer_type, 
                        crate::ui::UiLayerType::InGameOverlay | crate::ui::UiLayerType::MainMenu);
                    
                    self.ui_layers.set_layer(layer.clone(), layout);
                    
                    if should_show {
                        self.ui_layers.show(layer);
                    }
                }
                SceneUpdate::ShowUiLayer { layer } => {
                    info!("Showing UI layer {:?}", layer);
                    self.ui_layers.show(layer);
                }
                SceneUpdate::HideUiLayer { layer } => {
                    info!("Hiding UI layer {:?}", layer);
                    self.ui_layers.hide(&layer);
                }
                SceneUpdate::LoadUiScene { layer, scene_path } => {
                    // Load UI scene from assets/ui/{scene_path}.json
                    let path = format!("assets/ui/{}.json", scene_path);
                    match std::fs::read_to_string(&path) {
                        Ok(data) => {
                            match serde_json::from_str::<crate::ui::UiLayout>(&data) {
                                Ok(layout) => {
                                    self.ui_layers.set_layer(layer.clone(), layout);
                                    self.ui_layers.show(layer);
                                    info!("Loaded UI scene from {}", path);
                                }
                                Err(e) => warn!("Failed to parse UI scene {}: {}", path, e),
                            }
                        }
                        Err(e) => warn!("Failed to load UI scene {}: {}", path, e),
                    }
                }
                SceneUpdate::MenuLoad { alias } => {
                    // Push the current menu (if any) and show the new one
                    info!("MenuLoad: Loading menu '{}'", alias);
                    
                    // Check if the target menu is an IntermediateMenu - if so, hide the pause menu
                    let custom_layer = crate::ui::UiLayer::Custom(alias.clone());
                    if let Some(layout) = self.ui_layers.get_layer(&custom_layer) {
                        if layout.layer_type == crate::ui::UiLayerType::IntermediateMenu {
                            self.ui_layers.hide(&crate::ui::UiLayer::PauseMenu);
                        }
                    }
                    
                    self.menu_stack.push(&alias);
                    // Show the menu as a Custom layer
                    self.ui_layers.show(custom_layer);
                }
                SceneUpdate::MenuBack => {
                    // Pop the current menu and go back to the previous one
                    if let Some(current) = self.menu_stack.pop() {
                        info!("MenuBack: Closing menu '{}'", current);
                        self.ui_layers.hide(&crate::ui::UiLayer::Custom(current));
                        
                        // If menu stack is now empty, restore the pause menu
                        if self.menu_stack.is_empty() {
                            self.ui_layers.show(crate::ui::UiLayer::PauseMenu);
                        }
                    } else {
                        info!("MenuBack: No menu to go back from");
                    }
                }
                SceneUpdate::ShowPopup { alias } => {
                    info!("ShowPopup: Showing popup '{}'", alias);
                    self.ui_layers.show(crate::ui::UiLayer::Custom(alias));
                }
                SceneUpdate::HidePopup { alias } => {
                    info!("HidePopup: Hiding popup '{}'", alias);
                    self.ui_layers.hide(&crate::ui::UiLayer::Custom(alias));
                }
                SceneUpdate::ResetPauseMenu => {
                    info!("ResetPauseMenu: Resetting pause menu to default");
                    let default_layout = crate::ui::create_default_pause_menu_layout();
                    self.ui_layers.set_layer(crate::ui::UiLayer::PauseMenu, default_layout);
                }
            }
        }

        // Example: Spawn a task to load chunk 1
        // self.request_chunk(1);
    }

    /// Update all animated entities - ticks Animator components and uploads bone matrices
    /// Call this each frame to advance skeletal animations
    pub fn update_animations(&mut self, physics: &mut PhysicsWorld, dt: f32) {
        // Collect entities with Animator
        let entities: Vec<hecs::Entity> = self.ecs
            .query::<&Animator>()
            .iter()
            .map(|(e, _)| e)
            .collect();

        for entity in entities {
            let mut params = std::collections::HashMap::new();

            // 1. Process AnimatorController (if present) to drive state machine
            if let Ok(mut controller) = self.ecs.get::<&mut AnimatorController>(entity) {
                if controller.enabled {
                    // Evaluate state machine for transitions
                    if let Some((resource, blend_duration)) = controller.state_machine.evaluate() {
                        // If entity also has an Animator, crossfade to new resource
                        if let Ok(mut animator) = self.ecs.get::<&mut Animator>(entity) {
                            animator.crossfade_to(resource, blend_duration);
                        }
                    }
                    // Copy parameters for blend tree evaluation
                    params = controller.state_machine.parameters.clone();
                }
            }

            // 2. Update Animator (low-level playback and blending)
            let matrices = if let Ok(mut animator) = self.ecs.get::<&mut Animator>(entity) {
                let speed_mult = if let Ok(controller) = self.ecs.get::<&AnimatorController>(entity) {
                    controller.speed
                } else {
                    1.0
                };
                
                // Fetch event queue if present
                let mut queue = self.ecs.get::<&mut AnimationEventQueue>(entity).ok();

                // Use controller-global speed multiplier if present
                let old_speed = animator.speed;
                animator.speed *= speed_mult;
                let m = animator.update_with_params(dt, &params, &mut queue);
                animator.speed = old_speed;
                Some(m)
            } else {
                None
            };

            // 3. Update AnimationState (GPU matrix buffer)
            if let Some(matrices) = matrices {
                if let Ok(mut anim_state) = self.ecs.get::<&mut AnimationState>(entity) {
                    anim_state.update_from(matrices);
                }
            }

            // 4. Apply Root Motion (if enabled)
            let apply_rm = if let Ok(controller) = self.ecs.get::<&AnimatorController>(entity) {
                controller.apply_root_motion
            } else {
                false
            };

            if apply_rm {
                if let Ok(animator) = self.ecs.get::<&mut Animator>(entity) {
                    if animator.root_motion.has_motion {
                        let delta_pos = animator.root_motion.delta_position;
                        let delta_rot = animator.root_motion.delta_rotation;

                        // Apply to Transform
                        if let Ok(mut transform) = self.ecs.get::<&mut Transform>(entity) {
                            // Rotate position delta by current character rotation
                            let rotated_delta = transform.rotation * delta_pos;
                            transform.position += rotated_delta;
                            transform.rotation = transform.rotation * delta_rot;

                            // Apply to RigidBody (if present)
                            if let Ok(rb_handle) = self.ecs.get::<&RigidBodyHandle>(entity) {
                                if let Some(body) = physics.rigid_body_set.get_mut(rb_handle.0) {
                                    // Teleport physics body to match transform (for kinematic-like RM)
                                    // Alternatively, apply as velocity if preferred
                                    let p = transform.position;
                                    let r = transform.rotation;
                                    body.set_next_kinematic_translation(rapier3d::na::Vector3::new(p.x, p.y, p.z));
                                    body.set_next_kinematic_rotation(rapier3d::na::UnitQuaternion::from_quaternion(
                                        rapier3d::na::Quaternion::new(r.w, r.x, r.y, r.z)
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Clear animation event queues at end of frame
        for (_, queue) in self.ecs.query::<&mut AnimationEventQueue>().iter() {
            queue.clear();
        }
    }

    pub fn update_logic(&mut self, physics: &mut PhysicsWorld, dt: f32, ui_events: Vec<crate::ui::UiEvent>) {
        // Resolve hierarchical transforms BEFORE script/physics updates if we want them to affect visibility,
        // or AFTER if we want them to reflect the latest parent movement.
        // Let's do it AFTER script/physics logic in the main loop to ensure we have the latest world positions.
        self.resolve_hierarchies();

        // Drain pending UI events
        let mut events = ui_events;
        for event in events.drain(..) {
            self.dispatch_ui_event(physics, &event);
        }

        // 0.5 Update Animations - tick all skeletal animators
        self.update_animations(physics, dt);

        // 1. Process Collision Events
        let collision_events = {
            if let Ok(mut events) = physics.event_collector.collision_events.lock() {
                std::mem::take(&mut *events)
            } else {
                Vec::new()
            }
        };

        // 1. Process Collision Events - Dispatch sequentially for now
        for event in collision_events {
            match event {
                rapier3d::prelude::CollisionEvent::Started(c1, c2, _flags) => {
                    self.dispatch_collision_event(physics, c1, c2, true);
                }
                rapier3d::prelude::CollisionEvent::Stopped(c1, c2, _flags) => {
                    self.dispatch_collision_event(physics, c1, c2, false);
                }
            }
        }

        // 1.1 Parallel Agent Logic Update
        self.update_agents_parallel(dt);

        // 2. Update Scripts
        let entities: Vec<Entity> = self.ecs.query::<&DynamicScript>()
            .iter()
            .map(|(e, _)| e)
            .collect();

        for entity in entities {
            let (script, mut started, enabled) = if let Ok(mut s) = self.ecs.get::<&mut DynamicScript>(entity) {
                (s.script.take(), s.started, s.enabled)
            } else {
                continue;
            };

            if let Some(mut s) = script {
                if enabled {
                    // Check for animation events to dispatch
                    let anim_events: Vec<String> = if let Ok(queue) = self.ecs.get::<&AnimationEventQueue>(entity) {
                        queue.events.iter().map(|e| e.name.clone()).collect()
                    } else {
                        Vec::new()
                    };

                    let mut ctx = ScriptContext::new(entity, &mut self.ecs, physics, dt);

                    // Dispatch animation events
                    for event_name in anim_events {
                        s.on_animation_event(&mut ctx, &event_name);
                    }

                    if !started {
                        s.on_start(&mut ctx);
                        s.on_enable(&mut ctx);
                        started = true;
                    }
                    s.on_update(&mut ctx);
                }
                
                // Put it back
                if let Ok(mut comp) = self.ecs.get::<&mut DynamicScript>(entity) {
                    comp.script = Some(s);
                    comp.started = started;
                }
            }
        }
    }

    fn update_agents_parallel(&mut self, dt: f32) {
        use rayon::prelude::*;

        // Gather all agents and their transforms
        let agent_data: Vec<_> = self.ecs.query_mut::<(&mut CrowdAgent, &Transform)>()
            .into_iter()
            .map(|(e, (a, t))| (e, a.clone(), t.clone()))
            .collect();

        // Find player pos (global data for agents)
        let mut player_pos = None;
        for (_e, (t, _p)) in self.ecs.query::<(&Transform, &Player)>().iter() {
            player_pos = Some(t.position);
            break;
        }

        // Parallel update
        let results: Vec<_> = agent_data.into_par_iter().map(|(entity, mut agent, transform)| {
            let pos = transform.position;
            
            // Re-implementing the core CrowdAgent logic from scripting.rs for parallel speed
            let mut seed = (entity.id() as u32).wrapping_mul(12345) ^ (pos.x * 100.0) as u32;
            let mut rand = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed as f32) / (u32::MAX as f32)
            };

            if (pos - agent.last_pos).length() < 0.1 * dt {
                agent.stuck_timer += dt;
            } else {
                agent.stuck_timer = 0.0;
            }
            agent.last_pos = pos;

            if agent.stuck_timer > 1.5 {
                agent.target = glam::Vec3::new((rand() - 0.5) * 60.0, pos.y, (rand() - 0.5) * 60.0);
                agent.velocity = glam::Vec3::new(rand() - 0.5, 0.0, rand() - 0.5).normalize() * 2.0;
                agent.stuck_timer = 0.0;
            }

            if let Some(p_pos) = player_pos {
                if agent.state == AgentState::Fleeing && (p_pos - pos).length() < 12.0 {
                    let away = (pos - p_pos).normalize();
                    agent.target = pos + away * 15.0;
                }
            }

            let to_target = agent.target - pos;
            let dist = to_target.length();

            if dist < 1.5 || agent.target.length_squared() < 0.001 {
                agent.target = glam::Vec3::new((rand() - 0.5) * 60.0, pos.y, (rand() - 0.5) * 60.0);
                agent.velocity = (agent.target - pos).normalize() * 0.1;
                (entity, agent, None)
            } else {
                let desired = to_target.normalize() * agent.max_speed;
                let steer_force = if agent.state == AgentState::Fleeing { 20.0 } else { 8.0 };
                let steering = (desired - agent.velocity) * steer_force;

                agent.velocity += steering * dt;
                if agent.velocity.length() > agent.max_speed {
                    agent.velocity = agent.velocity.normalize() * agent.max_speed;
                }

                let new_pos = pos + agent.velocity * dt;
                let new_rot = if agent.velocity.length_squared() > 0.1 {
                    let angle = agent.velocity.x.atan2(agent.velocity.z);
                    glam::Quat::from_rotation_y(angle)
                } else {
                    transform.rotation
                };
                
                (entity, agent, Some((new_pos, new_rot)))
            }
        }).collect();

        // Apply results
        for (entity, agent, transform_update) in results {
            if let Ok(mut a) = self.ecs.get::<&mut CrowdAgent>(entity) {
                *a = agent;
            }
            if let Some((pos, rot)) = transform_update {
                if let Ok(mut t) = self.ecs.get::<&mut Transform>(entity) {
                    t.position = pos;
                    t.rotation = rot;
                }
            }
        }
    }

    fn dispatch_collision_event(&mut self, physics: &mut PhysicsWorld, c1: rapier3d::prelude::ColliderHandle, c2: rapier3d::prelude::ColliderHandle, started: bool) {
        let e1 = self.get_entity_from_collider(physics, c1);
        let e2 = self.get_entity_from_collider(physics, c2);

        if let Some(ent1) = e1 {
            if let Some(ent2) = e2 {
                self.call_collision_on_script(physics, ent1, ent2, started);
                self.call_collision_on_script(physics, ent2, ent1, started);
            }
        }
    }

    fn get_entity_from_collider(&self, physics: &PhysicsWorld, handle: rapier3d::prelude::ColliderHandle) -> Option<Entity> {
        let collider = physics.collider_set.get(handle)?;
        let rb_handle = collider.parent()?;
        let rb = physics.rigid_body_set.get(rb_handle)?;
        let bits = rb.user_data as u64;
        if bits == 0 { return None; }
        Entity::from_bits(bits)
    }

    fn call_collision_on_script(&mut self, physics: &mut PhysicsWorld, entity: Entity, other: Entity, started: bool) {
        let script = if let Ok(mut s) = self.ecs.get::<&mut DynamicScript>(entity) {
            s.script.take()
        } else {
            None
        };

        if let Some(mut s) = script {
            {
                let mut ctx = ScriptContext::new(entity, &mut self.ecs, physics, 0.0);
                if started {
                    s.on_collision_start(&mut ctx, other);
                } else {
                    s.on_collision_end(&mut ctx, other);
                }
            }
            if let Ok(mut comp) = self.ecs.get::<&mut DynamicScript>(entity) {
                comp.script = Some(s);
            }
        }
    }

    fn dispatch_ui_event(&mut self, physics: &mut PhysicsWorld, event: &crate::ui::UiEvent) {
        // First, check if callback is a built-in engine command
        let crate::ui::UiEvent::ButtonClicked { callback, .. } = event;
        // Parse menu_load("alias") pattern
            if callback.starts_with("menu_load(") && callback.ends_with(")") {
                let inner = &callback[10..callback.len()-1];
                // Extract alias from quotes (single or double)
                let alias = inner.trim().trim_matches(|c| c == '"' || c == '\'');
                if !alias.is_empty() {
                    info!("Built-in menu_load: Loading menu '{}'", alias);
                    
                    // Special case: "main" or "pause" means go back to the pause menu
                    if alias == "main" || alias == "pause" {
                        info!("menu_load: '{}' -> showing PauseMenu", alias);
                        // Hide all intermediate menus and clear the stack
                        while let Some(current) = self.menu_stack.pop() {
                            self.ui_layers.hide(&crate::ui::UiLayer::Custom(current));
                        }
                        // Show the pause menu
                        self.ui_layers.show(crate::ui::UiLayer::PauseMenu);
                        // Signal main.rs to release cursor (we're returning to pause menu)
                        self.cursor_should_release = true;
                        return;
                    }
                    
                    // Check if the target menu is an IntermediateMenu - if so, hide the pause menu
                    // so the intermediate menu appears on its own (not overlayed)
                    let custom_layer = crate::ui::UiLayer::Custom(alias.to_string());
                    if let Some(layout) = self.ui_layers.get_layer(&custom_layer) {
                        info!("menu_load: Found layer '{}' with type {:?}", alias, layout.layer_type);
                        if layout.layer_type == crate::ui::UiLayerType::IntermediateMenu {
                            // Hide the pause menu while the intermediate menu is shown
                            info!("menu_load: Hiding PauseMenu for IntermediateMenu");
                            self.ui_layers.hide(&crate::ui::UiLayer::PauseMenu);
                        }
                    } else {
                        info!("menu_load: Layer '{}' NOT FOUND in ui_layers", alias);
                    }
                    
                    self.menu_stack.push(alias);
                    self.ui_layers.show(custom_layer);
                    info!("menu_load: Showing layer '{}', menu_stack depth: {}", alias, self.menu_stack.depth());
                    return; // Don't pass to scripts
                }
            }
            // Parse menu_back() pattern
            if callback == "menu_back()" || callback == "menu_back" {
                if let Some(current) = self.menu_stack.pop() {
                    info!("Built-in menu_back: Closing menu '{}'", current);
                    self.ui_layers.hide(&crate::ui::UiLayer::Custom(current));
                    
                    // If menu stack is now empty and we're still paused, restore the pause menu
                    if self.menu_stack.is_empty() {
                        self.ui_layers.show(crate::ui::UiLayer::PauseMenu);
                        self.cursor_should_release = true;
                    }
                }
                return;
            }
            // Parse resume() pattern - unpause the game
            if callback == "resume()" || callback == "resume" {
                // This would require passing state back to main loop - for now just hide pause menu
                info!("Built-in resume: Player wants to resume game");
                // The pause state is managed in main.rs, but we can at least hide the menu
                return;
            }
            // Parse quit() pattern - exit the game
            if callback == "quit()" || callback == "quit" {
                info!("Built-in quit: Player wants to quit game");
                // Quit is handled in main.rs via the callback check - just return here
                return;
            }
        
        // Then pass to scripts for custom callbacks
        let entities: Vec<Entity> = self.ecs.query::<&DynamicScript>()
            .iter()
            .map(|(e, _)| e)
            .collect();

        for entity in entities {
            let script = if let Ok(mut s) = self.ecs.get::<&mut DynamicScript>(entity) {
                s.script.take()
            } else {
                None
            };

            if let Some(mut s) = script {
                {
                    let mut ctx = ScriptContext::new(entity, &mut self.ecs, physics, 0.0);
                    s.on_ui_event(&mut ctx, event);
                }
                if let Ok(mut comp) = self.ecs.get::<&mut DynamicScript>(entity) {
                    comp.script = Some(s);
                }
            }
        }
    }

    pub fn request_chunk(&self, _chunk_id: u32) {
        // let tx = self.chunk_sender.clone();
        // self.runtime.spawn(async move { ... });
        // NOTE: Runtime removed. If async IO is needed, pass a handle or use a separate IO system.
    }

    pub fn attach_physics_to_entity(
        &mut self,
        entity: Entity,
        physics: &mut PhysicsWorld,
        primitive: u8,
        position: [f32; 3],
        rotation: [f32; 4],
        scale: [f32; 3],
        is_dynamic: bool,
        layer: u32,
    ) {
        let entity_bits = entity.to_bits().get() as u128;
        
        // Determine collision mask based on layer
        // Props (Cubes) should not collide with Characters
        let filter = if layer == LAYER_PROP {
            LAYER_ENVIRONMENT | LAYER_PROP | LAYER_VEHICLE
        } else if layer == LAYER_CHARACTER {
            LAYER_ENVIRONMENT | LAYER_CHARACTER | LAYER_VEHICLE
        } else if layer == LAYER_ENVIRONMENT {
            u32::MAX
        } else {
            u32::MAX
        };

        let rb_handle = match primitive {
            0 | 3 => {
                // Cube or Plane (use box)
                physics.add_box_rigid_body(
                    entity_bits,
                    position,
                    [scale[0] * 0.5, scale[1] * 0.5, scale[2] * 0.5],
                    is_dynamic,
                    layer,
                    filter,
                )
            }
            1 => {
                // Sphere
                physics.add_sphere_rigid_body(entity_bits, position, scale[0] * 0.5, is_dynamic, layer, filter)
            }
            2 | 5 => {
                // Cylinder or Cone (as cylinder)
                physics.add_cylinder_rigid_body(
                    entity_bits,
                    position,
                    scale[1] * 0.5,
                    scale[0] * 0.5,
                    is_dynamic,
                    layer,
                    filter,
                )
            }
            4 => {
                // Capsule
                physics.add_capsule_rigid_body(
                    entity_bits,
                    position,
                    scale[1] * 0.5,
                    scale[0] * 0.5,
                    is_dynamic,
                    layer,
                    filter,
                )
            }
            255 => {
                // Custom Mesh: Use its AABB to create a box collider
                let mut half_extents = [scale[0] * 0.5, scale[1] * 0.5, scale[2] * 0.5];
                let mut y_offset = 0.0;
                
                if let Ok(mesh) = self.ecs.get::<&Mesh>(entity) {
                    let min = glam::Vec3::from(mesh.aabb_min);
                    let max = glam::Vec3::from(mesh.aabb_max);
                    let extent = max - min;
                    half_extents = [
                        extent.x * scale[0] * 0.5,
                        extent.y * scale[1] * 0.5,
                        extent.z * scale[2] * 0.5,
                    ];
                    
                    // If normalization is Base-Origin, the bottom is at Y=0 and top at Y=1 (roughly)
                    // The Rapier box is centered at its origin, so we need to offset it UP by half-height
                    // to match the visual mesh which starts at 0 and goes UP.
                    y_offset = half_extents[1];
                    info!("Custom mesh physics: half_extents={:?}, y_offset={}", half_extents, y_offset);
                }

                physics.add_box_rigid_body_with_offset(
                    entity_bits,
                    position,
                    y_offset,
                    half_extents,
                    is_dynamic,
                    layer,
                    filter,
                )
            }
            _ => physics.add_box_rigid_body(
                entity_bits,
                position,
                [scale[0] * 0.5, scale[1] * 0.5, scale[2] * 0.5],
                is_dynamic,
                layer,
                filter,
            ),
        };

        // Sync rotation if fixed (Rapier fixed bodies can still have rotation set at creation or via set_rotation)
        if let Some(rb) = physics.rigid_body_set.get_mut(rb_handle) {
            rb.set_rotation(
                rapier3d::na::UnitQuaternion::from_quaternion(rapier3d::na::Quaternion::new(
                    rotation[3],
                    rotation[0],
                    rotation[1],
                    rotation[2],
                )),
                true,
            );
        }

        let _ = self.ecs.insert_one(entity, RigidBodyHandle(rb_handle));
    }

    fn calculate_chunk_entities(&self, cx: i32, cz: i32, size: f32) -> Vec<(Transform, MeshHandle, Material)> {
        let mut entities = Vec::new();
        
        // Skip center chunks (clear spawn area)
        if cx.abs() <= 1 && cz.abs() <= 1 {
            return entities;
        }

        // Deterministic generation based on chunk coord
        let mut seed = (cx as u32).wrapping_mul(73856093) ^ (cz as u32).wrapping_mul(19349663);
        let mut rand = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as f32) / (u32::MAX as f32)
        };

        let count = 4;
        let step = size / count as f32;

        for i in 0..count {
            for j in 0..count {
                if rand() > 0.7 {
                    let lx = (i as f32) * step + step * 0.5 - size * 0.5;
                    let lz = (j as f32) * step + step * 0.5 - size * 0.5;

                    let wx = (cx as f32) * size + lx;
                    let wz = (cz as f32) * size + lz;

                    let height = 0.5 + rand() * 5.0;

                    entities.push((
                        Transform {
                            position: glam::Vec3::new(wx, height * 0.5, wz),
                            rotation: glam::Quat::IDENTITY,
                            scale: glam::Vec3::new(step * 0.8, height, step * 0.8),
                        },
                        MeshHandle(0),
                        Material {
                            color: [0.7, 0.7, 0.8, 1.0],
                            albedo_texture: None,
                            metallic: 0.0,
                            roughness: 0.5,
                        },
                    ));
                }
            }
        }
        entities
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
        |_p| tobj::load_mtl_buf(&mut std::io::Cursor::new(Vec::new())), // Ignore materials for now
    )?;

    if models.is_empty() {
        anyhow::bail!("No models found in buffer");
    }

    info!("OBJ loaded: {} model(s) found", models.len());

    let mut all_vertices = Vec::new();
    let mut all_indices = Vec::new();

    // Merge ALL models in the OBJ file
    for (model_idx, model) in models.iter().enumerate() {
        let mesh = &model.mesh;
        let base_vertex_idx = all_vertices.len() as u32;
        
        let vertex_count = mesh.positions.len() / 3;
        let has_normals = mesh.normals.len() >= vertex_count * 3;
        info!("  Model {} '{}': {} vertices, {} indices, has_normals: {}", 
            model_idx, model.name, vertex_count, mesh.indices.len(), has_normals);

        // First pass: create vertices with placeholder normals if needed
        for i in 0..vertex_count {
            let pos = [
                mesh.positions[i * 3],
                mesh.positions[i * 3 + 1],
                mesh.positions[i * 3 + 2],
            ];

            let normal = if has_normals {
                [
                    mesh.normals[i * 3],
                    mesh.normals[i * 3 + 1],
                    mesh.normals[i * 3 + 2],
                ]
            } else {
                [0.0, 0.0, 0.0] // Will be computed from faces
            };

            let uv = if mesh.texcoords.len() > i * 2 + 1 {
                [
                    mesh.texcoords[i * 2],
                    1.0 - mesh.texcoords[i * 2 + 1], // Flip V for Vulkan/OBJ mismatch
                ]
            } else {
                [0.0, 0.0]
            };

            let tangent = [1.0, 0.0, 0.0, 1.0];

            all_vertices.push(Vertex {
                position: pos,
                normal,
                uv,
                color: [1.0, 1.0, 1.0],
                tangent,
                bone_indices: [0, 0, 0, 0],
                bone_weights: [1.0, 0.0, 0.0, 0.0],
            });
        }

        // Offset indices for this model
        for &idx in &mesh.indices {
            all_indices.push(idx + base_vertex_idx);
        }
    }

    // If no normals were provided, compute face normals and accumulate to vertices
    let needs_normals = all_vertices.iter().any(|v| v.normal == [0.0, 0.0, 0.0]);
    if needs_normals && all_indices.len() >= 3 {
        info!("Computing face normals for {} triangles", all_indices.len() / 3);
        
        // Accumulate face normals to vertices
        for tri in all_indices.chunks(3) {
            if tri.len() < 3 { continue; }
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;
            
            if i0 >= all_vertices.len() || i1 >= all_vertices.len() || i2 >= all_vertices.len() {
                continue;
            }
            
            let p0 = glam::Vec3::from(all_vertices[i0].position);
            let p1 = glam::Vec3::from(all_vertices[i1].position);
            let p2 = glam::Vec3::from(all_vertices[i2].position);
            
            let edge1 = p1 - p0;
            let edge2 = p2 - p0;
            let face_normal = edge1.cross(edge2);
            
            // Accumulate (will normalize later)
            for &idx in &[i0, i1, i2] {
                let v = &mut all_vertices[idx];
                v.normal[0] += face_normal.x;
                v.normal[1] += face_normal.y;
                v.normal[2] += face_normal.z;
            }
        }
        
        // Normalize accumulated normals
        for v in &mut all_vertices {
            let n = glam::Vec3::from(v.normal);
            let len = n.length();
            if len > 0.0001 {
                let normalized = n / len;
                v.normal = normalized.to_array();
            } else {
                // Fallback for degenerate cases
                v.normal = [0.0, 0.0, 1.0];
            }
        }
    }

    // Normalize mesh bounds: Base-Origin (bottom at Y=0, centered on X/Z)
    if !all_vertices.is_empty() {
        let mut min = glam::Vec3::splat(f32::MAX);
        let mut max = glam::Vec3::splat(f32::MIN);
        
        for v in &all_vertices {
            let p = glam::Vec3::from(v.position);
            min = min.min(p);
            max = max.max(p);
        }
        
        let center = (min + max) * 0.5;
        let extent = max - min;
        let max_extent = extent.x.max(extent.y).max(extent.z);
        
        if max_extent > 0.0001 {
            let scale = 1.0 / max_extent; // Normalize to 1 unit
            
            info!("Normalizing mesh (Base-Origin): min_y={}, center_xz={:?}, scale={}", min.y, (center.x, center.z), scale);
            
            for v in &mut all_vertices {
                let p = glam::Vec3::from(v.position);
                // Center on X and Z, align bottom of Y to 0
                let normalized = glam::Vec3::new(
                    (p.x - center.x) * scale,
                    (p.y - min.y) * scale,
                    (p.z - center.z) * scale,
                );
                v.position = normalized.to_array();
            }
        }
    }

    // Generate UVs using planar projection if none were provided
    // Use the normalized positions (now in roughly -0.5 to 0.5 range) mapped to 0-1
    let needs_uvs = all_vertices.iter().all(|v| v.uv == [0.0, 0.0]);
    if needs_uvs && !all_vertices.is_empty() {
        info!("Generating planar UV coordinates for mesh without UVs");
        
        // Find bounds of normalized mesh for proper UV mapping
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        
        for v in &all_vertices {
            min_x = min_x.min(v.position[0]);
            max_x = max_x.max(v.position[0]);
            min_y = min_y.min(v.position[1]);
            max_y = max_y.max(v.position[1]);
        }
        
        let range_x = (max_x - min_x).max(0.0001);
        let range_y = (max_y - min_y).max(0.0001);
        
        // Map normalized positions to UV [0, 1]
        for v in &mut all_vertices {
            // X position -> U coordinate
            let u = (v.position[0] - min_x) / range_x;
            // Y position -> V coordinate (flip for typical image orientation)
            let v_coord = 1.0 - (v.position[1] - min_y) / range_y;
            v.uv = [u, v_coord];
        }
        
        info!("Generated UVs for {} vertices", all_vertices.len());
    }

    info!("OBJ total: {} vertices, {} indices", all_vertices.len(), all_indices.len());

    // Final bounds check for Mesh struct
    let mut aabb_min = [0.0, 0.0, 0.0];
    let mut aabb_max = [0.0, 0.0, 0.0];
    if !all_vertices.is_empty() {
        let mut min = glam::Vec3::splat(f32::MAX);
        let mut max = glam::Vec3::splat(f32::MIN);
        for v in &all_vertices {
            let p = glam::Vec3::from(v.position);
            min = min.min(p);
            max = max.max(p);
        }
        aabb_min = min.to_array();
        aabb_max = max.to_array();
    }

    Ok(Mesh {
        vertices: all_vertices,
        indices: all_indices,
        albedo: None,
        normal: None,
        metallic_roughness: None,
        albedo_texture: None,
        aabb_min,
        aabb_max,
        decoded_albedo: None,
        decoded_normal: None,
        decoded_mr: None,
    })
}

pub fn create_primitive(ptype: u8) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Helper to generate a vertex
    fn make_vert(pos: [f32; 3], norm: [f32; 3], uv: [f32; 2]) -> Vertex {
        Vertex {
            position: pos,
            normal: norm,
            uv,
            color: [1.0, 1.0, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            bone_indices: [0, 0, 0, 0],
            bone_weights: [1.0, 0.0, 0.0, 0.0],
        }
    }

    match ptype {
        0 => {
            // Cube
            // Front
            let off = vertices.len() as u32;
            vertices.push(make_vert([-0.5, -0.5, 0.5], [0.0, 0.0, 1.0], [0.0, 0.0]));
            vertices.push(make_vert([0.5, -0.5, 0.5], [0.0, 0.0, 1.0], [1.0, 0.0]));
            vertices.push(make_vert([0.5, 0.5, 0.5], [0.0, 0.0, 1.0], [1.0, 1.0]));
            vertices.push(make_vert([-0.5, 0.5, 0.5], [0.0, 0.0, 1.0], [0.0, 1.0]));
            indices.extend_from_slice(&[off, off + 1, off + 2, off + 2, off + 3, off]);

            // Back
            let off = vertices.len() as u32;
            vertices.push(make_vert([0.5, -0.5, -0.5], [0.0, 0.0, -1.0], [0.0, 0.0]));
            vertices.push(make_vert([-0.5, -0.5, -0.5], [0.0, 0.0, -1.0], [1.0, 0.0]));
            vertices.push(make_vert([-0.5, 0.5, -0.5], [0.0, 0.0, -1.0], [1.0, 1.0]));
            vertices.push(make_vert([0.5, 0.5, -0.5], [0.0, 0.0, -1.0], [0.0, 1.0]));
            indices.extend_from_slice(&[off, off + 1, off + 2, off + 2, off + 3, off]);

            // Top
            let off = vertices.len() as u32;
            vertices.push(make_vert([-0.5, 0.5, 0.5], [0.0, 1.0, 0.0], [0.0, 0.0]));
            vertices.push(make_vert([0.5, 0.5, 0.5], [0.0, 1.0, 0.0], [1.0, 0.0]));
            vertices.push(make_vert([0.5, 0.5, -0.5], [0.0, 1.0, 0.0], [1.0, 1.0]));
            vertices.push(make_vert([-0.5, 0.5, -0.5], [0.0, 1.0, 0.0], [0.0, 1.0]));
            indices.extend_from_slice(&[off, off + 1, off + 2, off + 2, off + 3, off]);

            // Bottom
            let off = vertices.len() as u32;
            vertices.push(make_vert([-0.5, -0.5, -0.5], [0.0, -1.0, 0.0], [0.0, 0.0]));
            vertices.push(make_vert([0.5, -0.5, -0.5], [0.0, -1.0, 0.0], [1.0, 0.0]));
            vertices.push(make_vert([0.5, -0.5, 0.5], [0.0, -1.0, 0.0], [1.0, 1.0]));
            vertices.push(make_vert([-0.5, -0.5, 0.5], [0.0, -1.0, 0.0], [0.0, 1.0]));
            indices.extend_from_slice(&[off, off + 1, off + 2, off + 2, off + 3, off]);

            // Right
            let off = vertices.len() as u32;
            vertices.push(make_vert([0.5, -0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 0.0]));
            vertices.push(make_vert([0.5, -0.5, -0.5], [1.0, 0.0, 0.0], [1.0, 0.0]));
            vertices.push(make_vert([0.5, 0.5, -0.5], [1.0, 0.0, 0.0], [1.0, 1.0]));
            vertices.push(make_vert([0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 1.0]));
            indices.extend_from_slice(&[off, off + 1, off + 2, off + 2, off + 3, off]);

            // Left
            let off = vertices.len() as u32;
            vertices.push(make_vert([-0.5, -0.5, -0.5], [-1.0, 0.0, 0.0], [0.0, 0.0]));
            vertices.push(make_vert([-0.5, -0.5, 0.5], [-1.0, 0.0, 0.0], [1.0, 0.0]));
            vertices.push(make_vert([-0.5, 0.5, 0.5], [-1.0, 0.0, 0.0], [1.0, 1.0]));
            vertices.push(make_vert([-0.5, 0.5, -0.5], [-1.0, 0.0, 0.0], [0.0, 1.0]));
            indices.extend_from_slice(&[off, off + 1, off + 2, off + 2, off + 3, off]);
        }
        1 => {
            // Sphere (UV Sphere)
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
                        current,
                        next,
                        current + 1,
                        next,
                        next + 1,
                        current + 1,
                    ]);
                }
            }
        }
        3 => {
            // Plane (XZ Quad)
            // Facing Up (Normal 0,1,0)
            let off = vertices.len() as u32;
            vertices.push(make_vert([-0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.0, 0.0]));  // BL
            vertices.push(make_vert([0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [1.0, 0.0]));   // BR
            vertices.push(make_vert([0.5, 0.0, -0.5], [0.0, 1.0, 0.0], [1.0, 1.0]));  // TR
            vertices.push(make_vert([-0.5, 0.0, -0.5], [0.0, 1.0, 0.0], [0.0, 1.0])); // TL
            indices.extend_from_slice(&[off, off + 1, off + 2, off + 2, off + 3, off]);
        }
        2 | 4 | 5 => {
            // Cylinder, Capsule (as Cylinder for now), Cone
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
                vertices.push(make_vert(
                    [x * bottom_radius, y_min, z * bottom_radius],
                    [x, 0.0, z],
                    [u, 0.0],
                ));
                // Top
                vertices.push(make_vert(
                    [x * top_radius, y_max, z * top_radius],
                    [x, 0.0, z],
                    [u, 1.0],
                ));
            }
            for s in 0..segments {
                let off = (s * 2) as u32;
                indices.extend_from_slice(&[off, off + 1, off + 2, off + 1, off + 3, off + 2]);
            }

            // Caps (Simple fan)
            // Bottom Cap
            let center_idx = vertices.len() as u32;
            vertices.push(make_vert([0.0, y_min, 0.0], [0.0, -1.0, 0.0], [0.5, 0.5]));
            for s in 0..=segments {
                let angle = (s as f32 / segments as f32) * std::f32::consts::PI * 2.0;
                vertices.push(make_vert(
                    [
                        angle.cos() * bottom_radius,
                        y_min,
                        angle.sin() * bottom_radius,
                    ],
                    [0.0, -1.0, 0.0],
                    [0.5 + angle.cos() * 0.5, 0.5 + angle.sin() * 0.5],
                ));
            }
            for s in 0..segments {
                indices.extend_from_slice(&[
                    center_idx,
                    center_idx + 1 + s as u32,
                    center_idx + 2 + s as u32,
                ]);
            }

            // Top Cap (Cylinder only, or if radius > 0)
            if top_radius > 0.001 {
                let center_idx = vertices.len() as u32;
                vertices.push(make_vert([0.0, y_max, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5]));
                for s in 0..=segments {
                    let angle = (s as f32 / segments as f32) * std::f32::consts::PI * 2.0;
                    vertices.push(make_vert(
                        [angle.cos() * top_radius, y_max, angle.sin() * top_radius],
                        [0.0, 1.0, 0.0],
                        [0.5 + angle.cos() * 0.5, 0.5 + angle.sin() * 0.5],
                    ));
                }
                for s in 0..segments {
                    indices.extend_from_slice(&[
                        center_idx,
                        center_idx + 2 + s as u32,
                        center_idx + 1 + s as u32,
                    ]);
                }
            }
        }
        _ => {}
    }

    let mut aabb_min = [0.0, 0.0, 0.0];
    let mut aabb_max = [0.0, 0.0, 0.0];
    if !vertices.is_empty() {
        let mut min = glam::Vec3::splat(f32::MAX);
        let mut max = glam::Vec3::splat(f32::MIN);
        for v in &vertices {
            let p = glam::Vec3::from(v.position);
            min = min.min(p);
            max = max.max(p);
        }
        aabb_min = min.to_array();
        aabb_max = max.to_array();
    }

    Mesh {
        vertices: vertices
            .iter()
            .map(|v| Vertex {
                position: v.position,
                normal: v.normal,
                uv: v.uv,
                color: v.color,
                tangent: v.tangent,
                bone_indices: [0, 0, 0, 0],
                bone_weights: [1.0, 0.0, 0.0, 0.0],
            })
            .collect(),
        indices,
        albedo: None,
        normal: None,
        metallic_roughness: None,
        albedo_texture: None,
        aabb_min,
        aabb_max,
        decoded_albedo: None,
        decoded_normal: None,
        decoded_mr: None,
    }
}
