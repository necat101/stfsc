use crate::physics::PhysicsWorld;
use crate::project::scene::{MAX_SCENE_SCRIPTS_PER_SCENE, MAX_SCRIPTS_PER_ENTITY};
use glam;
use hecs::{Entity, World};
use log::{info, warn};
use rapier3d;
use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::sync::mpsc;

pub mod animation;
pub mod fbx_loader;
pub mod gltf_loader;
pub mod sandbox;
pub mod scripting;
#[allow(dead_code, unused_imports)]
mod generated_fuckscript {
    include!(concat!(env!("OUT_DIR"), "/generated_fuckscript.rs"));
}
use animation::{AnimParam, AnimationEventQueue, AnimationState, Animator, AnimatorController};
use scripting::{
    FuckScript, ScriptContext, ScriptRegistry, XrHapticRequest, XrInputEvent, XrInputSnapshot,
};

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
    /// Seeded sandbox/open-world settings used by survival and creative style projects.
    pub sandbox_settings: sandbox::SandboxWorldSettings,
    /// Runtime day/night clock for sandbox-aware gameplay.
    pub sandbox_clock: sandbox::SandboxClock,
    pub player_start_transform: Transform,
    pub pending_audio_uploads: std::collections::HashMap<String, Vec<u8>>,
    pub pending_texture_uploads: std::collections::HashMap<String, Vec<u8>>,
    pub script_registry: Arc<ScriptRegistry>,
    pub respawn_enabled: bool,
    pub respawn_y: f32,
    pub pending_ui_events: Vec<crate::ui::UiEvent>,
    /// Scene graph known to the runtime. World and UI scenes are tracked separately.
    pub scene_hierarchy: crate::project::scene::SceneHierarchy,
    /// Scene commands requested by UI callbacks or scripts during logic update.
    pub pending_scene_commands: Vec<SceneUpdate>,
    /// Currently active 3D scene path/name, if one was loaded through the scene manager.
    pub active_world_scene: Option<String>,
    /// Stack of previous 3D scenes for push/pop transitions.
    pub world_scene_stack: Vec<String>,
    /// Stack of active UI scene aliases for push/pop UI navigation.
    pub ui_scene_stack: Vec<String>,
    /// Multi-layer UI system
    pub ui_layers: crate::ui::UiLayerSet,
    /// Menu navigation stack for intermediate menus
    pub menu_stack: crate::ui::MenuStack,
    /// Flag for main.rs to check and release cursor when returning to pause menu
    pub cursor_should_release: bool,
    /// Latest XR rig/controller snapshot exposed to scripts.
    pub xr_input: XrInputSnapshot,
    /// Previous XR snapshot for edge detection and debugging.
    pub previous_xr_input: XrInputSnapshot,
    /// XR input events waiting to be dispatched to scripts.
    pub pending_xr_events: Vec<XrInputEvent>,
    /// Haptic pulses requested by scripts, drained by the platform runtime.
    pub pending_xr_haptics: Vec<XrHapticRequest>,
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
    pub fn new_static(
        position: [f32; 3],
        normal: [f32; 3],
        uv: [f32; 2],
        color: [f32; 3],
        tangent: [f32; 4],
    ) -> Self {
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
    pub albedo_texture: Option<String>, // Texture ID for custom albedo texture
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

#[derive(Debug)]
pub struct SceneScriptController {
    pub scene_id: String,
}

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
    Fleeing,     // Running away from danger
    Chasing,     // Following a target closely
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

#[derive(Clone, Debug)]
pub struct Projectile {
    pub velocity: glam::Vec3,
    pub damage: f32,
    pub lifetime: f32,
    pub age: f32,
    pub owner: Option<Entity>,
    pub gravity_scale: f32,
}

/// Component to hold scripts on an entity
pub struct ScriptSlot {
    pub name: String,
    pub script: Option<Box<dyn FuckScript>>,
    pub awoken: bool,
    pub started: bool,
    pub enabled: bool,
}

impl ScriptSlot {
    pub fn new(name: impl Into<String>, script: Box<dyn FuckScript>) -> Self {
        Self {
            name: name.into(),
            script: Some(script),
            awoken: false,
            started: false,
            enabled: true,
        }
    }
}

/// Component to hold one or more scripts on an entity.
pub struct DynamicScript {
    pub scripts: Vec<ScriptSlot>,
}

impl DynamicScript {
    pub fn new(script: Box<dyn FuckScript>) -> Self {
        Self::from_named("Script", script)
    }

    pub fn from_named(name: impl Into<String>, script: Box<dyn FuckScript>) -> Self {
        Self {
            scripts: vec![ScriptSlot::new(name, script)],
        }
    }

    pub fn empty() -> Self {
        Self {
            scripts: Vec::new(),
        }
    }

    pub fn push_named(&mut self, name: impl Into<String>, script: Box<dyn FuckScript>) -> bool {
        if self.scripts.len() >= MAX_SCRIPTS_PER_ENTITY {
            return false;
        }
        self.scripts.push(ScriptSlot::new(name, script));
        true
    }

    pub fn is_empty(&self) -> bool {
        self.scripts.is_empty()
    }

    fn take_slot(&mut self, index: usize) -> Option<ScriptSlot> {
        if index < self.scripts.len() {
            Some(self.scripts.remove(index))
        } else {
            None
        }
    }

    fn insert_slot(&mut self, index: usize, slot: ScriptSlot) {
        let index = index.min(self.scripts.len());
        self.scripts.insert(index, slot);
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
        albedo_texture: Option<String>, // Texture ID for custom albedo texture
        collision_enabled: bool,
        layer: u32,
        #[serde(default)]
        is_static: bool, // If true, object is not affected by gravity
    },
    SpawnProjectile {
        id: u32,
        position: [f32; 3],
        rotation: [f32; 4],
        velocity: [f32; 3],
        radius: f32,
        lifetime: f32,
        damage: f32,
        color: [f32; 3],
        layer: u32,
        gravity_scale: f32,
        owner: Option<u64>,
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
    /// Register or replace the scene hierarchy known to the runtime.
    SetSceneHierarchy {
        hierarchy: crate::project::scene::SceneHierarchy,
    },
    /// Generic transition target for both 3D and UI scenes.
    TransitionScene {
        target: crate::project::scene::SceneRef,
        #[serde(default)]
        mode: crate::project::scene::SceneTransitionMode,
    },
    /// Load a 3D world scene from a JSON scene file.
    LoadWorldScene {
        scene_path: String,
        #[serde(default)]
        mode: crate::project::scene::SceneTransitionMode,
    },
    /// Request a UI scene from gameplay or scripts.
    CallUiScene {
        scene_path: String,
        layer: crate::ui::UiLayer,
        #[serde(default)]
        mode: crate::project::scene::SceneTransitionMode,
    },
    /// Request a 3D scene from UI callbacks.
    CallWorldScene {
        scene_path: String,
        #[serde(default)]
        mode: crate::project::scene::SceneTransitionMode,
    },
    /// Toggle procedural generation on/off (off by default)
    SetProceduralGeneration {
        enabled: bool,
    },
    /// Replace sandbox/open-world settings without committing to a specific game.
    SetSandboxSettings {
        settings: sandbox::SandboxWorldSettings,
    },
    /// Switch between survival rules and free-building god mode.
    SetGameMode {
        mode: sandbox::GameMode,
    },
    /// Set deterministic day/night time for sandbox worlds.
    SetSandboxClock {
        time_of_day: f32,
        day_index: u32,
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
    /// Replace the full script component stack on an entity.
    SetScripts {
        id: u32,
        names: Vec<String>,
    },
    /// Replace the script stack for a scene-level controller entity.
    SetSceneScripts {
        scene_id: String,
        names: Vec<String>,
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
        mesh_data: Vec<u8>, // Raw FBX/OBJ file bytes
        position: [f32; 3],
        rotation: [f32; 4],
        scale: [f32; 3],
        #[serde(default)]
        albedo_texture: Option<String>, // Optional texture ID
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
        mesh_data: Vec<u8>, // Raw GLB/GLTF file bytes
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
            sandbox_settings: sandbox::SandboxWorldSettings::default(),
            sandbox_clock: sandbox::SandboxClock::default(),
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
                reg.register("EnemyTracker", || scripting::EnemyTrackerScript::default());
                reg.register("WeaponNPC", || scripting::WeaponNPCScript);
                reg.register("GunWeapon", || scripting::GunWeaponScript::default());
                reg.register("BowWeapon", || scripting::BowWeaponScript::default());
                reg.register("Projectile", || scripting::ProjectileScript);
                reg.register("CollisionLogger", || scripting::CollisionLoggerScript);
                reg.register("TouchToDestroy", || scripting::TouchToDestroyScript);
                reg.register("HeadAnchor", || {
                    scripting::XrPoseAnchorScript::new(scripting::XrTrackedTarget::Head)
                });
                reg.register("LeftHandAnchor", || {
                    scripting::XrPoseAnchorScript::new(scripting::XrTrackedTarget::LeftGrip)
                });
                reg.register("RightHandAnchor", || {
                    scripting::XrPoseAnchorScript::new(scripting::XrTrackedTarget::RightGrip)
                });
                reg.register("LeftAimAnchor", || {
                    scripting::XrPoseAnchorScript::new(scripting::XrTrackedTarget::LeftAim)
                });
                reg.register("RightAimAnchor", || {
                    scripting::XrPoseAnchorScript::new(scripting::XrTrackedTarget::RightAim)
                });
                reg.register("TriggerHaptics", || scripting::TriggerHapticsScript);
                reg.register(scripting::CUSTOM_FUCKSCRIPT_RUNTIME_NAME, || {
                    scripting::CustomFuckScript
                });
                generated_fuckscript::register_generated_scripts(&mut reg);
                Arc::new(reg)
            },
            respawn_enabled: false,
            respawn_y: -50.0,
            pending_ui_events: Vec::new(),
            scene_hierarchy: crate::project::scene::SceneHierarchy::default(),
            pending_scene_commands: Vec::new(),
            active_world_scene: None,
            world_scene_stack: Vec::new(),
            ui_scene_stack: Vec::new(),
            ui_layers: crate::ui::UiLayerSet::new(),
            menu_stack: crate::ui::MenuStack::new(),
            cursor_should_release: false,
            xr_input: XrInputSnapshot::default(),
            previous_xr_input: XrInputSnapshot::default(),
            pending_xr_events: Vec::new(),
            pending_xr_haptics: Vec::new(),
        };
        // world.spawn_default_scene(); // Moved to streaming logic or separate init
        world
    }

    pub fn set_sandbox_settings(&mut self, mut settings: sandbox::SandboxWorldSettings) {
        settings.clamp_runtime_limits();
        self.sandbox_clock = sandbox::SandboxClock::from_settings(&settings);
        self.sandbox_settings = settings;
        self.loaded_chunks.clear();
    }

    pub fn set_game_mode(&mut self, mode: sandbox::GameMode) {
        self.sandbox_settings.game_mode = mode;
    }

    pub fn advance_sandbox_clock(&mut self, dt: f32) {
        if self.sandbox_settings.enabled {
            self.sandbox_clock.advance(dt);
        }
    }

    pub fn plan_sandbox_chunk(&self, cx: i32, cz: i32) -> sandbox::SandboxChunkPlan {
        let generator = sandbox::SandboxChunkGenerator::new(self.sandbox_settings.seed);
        generator.generate_chunk(
            sandbox::SandboxChunkCoord { x: cx, z: cz },
            &self.sandbox_settings,
        )
    }

    pub fn plan_sandbox_chunks_around(
        &self,
        player_pos: glam::Vec3,
    ) -> Vec<sandbox::SandboxChunkPlan> {
        let center = sandbox::SandboxChunkCoord::from_world_pos(
            player_pos,
            self.sandbox_settings.chunk_size,
        );
        let generator = sandbox::SandboxChunkGenerator::new(self.sandbox_settings.seed);
        generator.generate_window(
            center,
            self.sandbox_settings.load_radius,
            &self.sandbox_settings,
        )
    }

    pub fn set_xr_input_snapshot(&mut self, mut snapshot: XrInputSnapshot) {
        snapshot.populate_builtin_actions_from_controls();

        let previous = self.xr_input.clone();
        if snapshot.frame_index <= previous.frame_index {
            snapshot.frame_index = previous.frame_index.saturating_add(1);
        }
        snapshot.update_edges_from(&previous);

        self.pending_xr_events
            .extend(snapshot.diff_events(&previous));
        self.previous_xr_input = previous;
        self.xr_input = snapshot;
    }

    pub fn drain_xr_haptics(&mut self) -> Vec<XrHapticRequest> {
        std::mem::take(&mut self.pending_xr_haptics)
    }

    pub fn queue_scene_command(&mut self, command: SceneUpdate) {
        self.pending_scene_commands.push(command);
    }

    fn flush_pending_scene_commands(&mut self) {
        let pending = std::mem::take(&mut self.pending_scene_commands);
        for command in pending {
            match self.command_sender.try_send(command) {
                Ok(()) => {}
                Err(tokio::sync::mpsc::error::TrySendError::Full(command)) => {
                    self.pending_scene_commands.push(command);
                    warn!("Scene command queue is full; deferring remaining scene commands");
                    break;
                }
                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                    warn!("Scene command queue is closed; dropping pending scene commands");
                    break;
                }
            }
        }
    }

    fn expand_scene_transition(
        &mut self,
        target: crate::project::scene::SceneRef,
        mode: crate::project::scene::SceneTransitionMode,
        commands: &mut VecDeque<SceneUpdate>,
    ) {
        match target.domain {
            crate::project::scene::SceneDomain::World3d => {
                self.expand_world_scene_load(&target.path, mode, commands);
            }
            crate::project::scene::SceneDomain::Ui => {
                let alias = target.alias.clone().unwrap_or_else(|| {
                    Self::scene_stem(&target.path).unwrap_or_else(|| target.name.clone())
                });
                let layer = target
                    .layer
                    .clone()
                    .unwrap_or_else(|| crate::ui::UiLayer::Custom(alias));
                self.expand_ui_scene_load(&target.path, Some(layer), mode, commands);
            }
        }
    }

    fn expand_world_scene_load(
        &mut self,
        scene_path: &str,
        mode: crate::project::scene::SceneTransitionMode,
        commands: &mut VecDeque<SceneUpdate>,
    ) {
        use crate::project::scene::SceneTransitionMode;

        if mode == SceneTransitionMode::Pop {
            if let Some(previous) = self.world_scene_stack.pop() {
                self.expand_world_scene_load(&previous, SceneTransitionMode::Replace, commands);
            } else {
                warn!("World scene pop requested but the world scene stack is empty");
            }
            return;
        }

        let Some(resolved_path) =
            Self::resolve_scene_file(scene_path, crate::project::scene::SceneDomain::World3d)
        else {
            warn!("World scene '{}' could not be resolved", scene_path);
            return;
        };

        let scene_json = match std::fs::read_to_string(&resolved_path) {
            Ok(data) => data,
            Err(err) => {
                warn!(
                    "Failed to read world scene '{}': {}",
                    resolved_path.display(),
                    err
                );
                return;
            }
        };

        let scene = match serde_json::from_str::<crate::project::scene::Scene>(&scene_json) {
            Ok(scene) => scene,
            Err(scene_err) => {
                match serde_json::from_str::<crate::project::scene::WorldScene>(&scene_json) {
                    Ok(world_scene) => crate::project::scene::Scene::from_world_scene(world_scene),
                    Err(world_err) => {
                        warn!(
                        "Failed to parse world scene '{}': {}; world-only parse also failed: {}",
                        resolved_path.display(),
                        scene_err,
                        world_err
                    );
                        return;
                    }
                }
            }
        };

        match mode {
            SceneTransitionMode::Replace => {
                self.world_scene_stack.clear();
                commands.push_back(SceneUpdate::ClearScene);
            }
            SceneTransitionMode::Push => {
                if let Some(current) = self.active_world_scene.clone() {
                    self.world_scene_stack.push(current);
                }
                commands.push_back(SceneUpdate::ClearScene);
            }
            SceneTransitionMode::Additive | SceneTransitionMode::Overlay => {}
            SceneTransitionMode::Pop => {}
        }

        let scene_id = resolved_path.to_string_lossy().to_string();
        if mode != SceneTransitionMode::Additive || self.active_world_scene.is_none() {
            self.active_world_scene = Some(scene_id);
        }

        let scene_dir = resolved_path.parent();
        for update in Self::runtime_updates_for_scene(&scene, scene_dir) {
            commands.push_back(update);
        }

        info!(
            "Expanded world scene '{}' with {} entities via {:?}",
            scene.name,
            scene.entities.len(),
            mode
        );
    }

    fn expand_ui_scene_load(
        &mut self,
        scene_path: &str,
        requested_layer: Option<crate::ui::UiLayer>,
        mode: crate::project::scene::SceneTransitionMode,
        _commands: &mut VecDeque<SceneUpdate>,
    ) {
        use crate::project::scene::SceneTransitionMode;

        if mode == SceneTransitionMode::Pop {
            self.pop_ui_scene();
            return;
        }

        if let Some(layer) = requested_layer.clone() {
            if self.ui_layers.get_layer(&layer).is_some() {
                let alias = match &layer {
                    crate::ui::UiLayer::Custom(alias) => alias.clone(),
                    crate::ui::UiLayer::Hud => "hud".to_string(),
                    crate::ui::UiLayer::PauseMenu => "pause".to_string(),
                    crate::ui::UiLayer::MainMenu => "main".to_string(),
                };
                self.show_existing_ui_layer(layer, alias, mode);
                return;
            }
        }

        let Some(resolved_path) =
            Self::resolve_scene_file(scene_path, crate::project::scene::SceneDomain::Ui)
        else {
            warn!("UI scene '{}' could not be resolved", scene_path);
            return;
        };

        let scene_json = match std::fs::read_to_string(&resolved_path) {
            Ok(data) => data,
            Err(err) => {
                warn!(
                    "Failed to read UI scene '{}': {}",
                    resolved_path.display(),
                    err
                );
                return;
            }
        };

        let (name, alias, layer_type, mut layout) = if let Ok(ui_scene) =
            serde_json::from_str::<crate::project::scene::UiScene>(&scene_json)
        {
            (
                ui_scene.name,
                ui_scene.alias,
                ui_scene.layer_type,
                ui_scene.layout,
            )
        } else if let Ok(layout) = serde_json::from_str::<crate::ui::UiLayout>(&scene_json) {
            let alias = Self::scene_stem(scene_path).unwrap_or_else(|| "ui".to_string());
            (alias.clone(), alias, layout.layer_type, layout)
        } else if let Ok(scene) = serde_json::from_str::<crate::project::scene::Scene>(&scene_json)
        {
            let Some(ui_scene) = scene.ui_scenes().into_iter().next() else {
                warn!(
                    "Scene '{}' did not contain any legacy UI layouts",
                    resolved_path.display()
                );
                return;
            };
            (
                ui_scene.name,
                ui_scene.alias,
                ui_scene.layer_type,
                ui_scene.layout,
            )
        } else {
            warn!("Failed to parse UI scene '{}'", resolved_path.display());
            return;
        };

        layout.layer_type = layer_type;
        let layer =
            requested_layer.unwrap_or_else(|| Self::ui_layer_for_type(layer_type, alias.as_str()));

        self.apply_ui_scene_layer(layer, alias, layout, mode);
        info!("Loaded UI scene '{}' via {:?}", name, mode);
    }

    fn show_existing_ui_layer(
        &mut self,
        layer: crate::ui::UiLayer,
        alias: String,
        mode: crate::project::scene::SceneTransitionMode,
    ) {
        if mode == crate::project::scene::SceneTransitionMode::Replace {
            self.ui_layers.visible.clear();
            self.menu_stack.clear();
            self.ui_scene_stack.clear();
        }

        self.ui_layers.show(layer.clone());
        self.track_ui_scene_layer(layer, alias, mode);
    }

    fn apply_ui_scene_layer(
        &mut self,
        layer: crate::ui::UiLayer,
        alias: String,
        layout: crate::ui::UiLayout,
        mode: crate::project::scene::SceneTransitionMode,
    ) {
        if mode == crate::project::scene::SceneTransitionMode::Replace {
            self.ui_layers.visible.clear();
            self.menu_stack.clear();
            self.ui_scene_stack.clear();
        }

        self.ui_layers.set_layer(layer.clone(), layout);
        self.ui_layers.show(layer.clone());
        self.track_ui_scene_layer(layer, alias, mode);
    }

    fn track_ui_scene_layer(
        &mut self,
        layer: crate::ui::UiLayer,
        alias: String,
        mode: crate::project::scene::SceneTransitionMode,
    ) {
        if matches!(
            mode,
            crate::project::scene::SceneTransitionMode::Push
                | crate::project::scene::SceneTransitionMode::Overlay
                | crate::project::scene::SceneTransitionMode::Additive
                | crate::project::scene::SceneTransitionMode::Replace
        ) && !self
            .ui_scene_stack
            .iter()
            .any(|existing| existing == &alias)
        {
            self.ui_scene_stack.push(alias.clone());
        }

        if let crate::ui::UiLayer::Custom(custom_alias) = layer {
            if !self
                .menu_stack
                .iter()
                .any(|existing| existing == &custom_alias)
            {
                self.menu_stack.push(&custom_alias);
            }
        }
    }

    fn pop_ui_scene(&mut self) {
        if let Some(alias) = self.ui_scene_stack.pop() {
            self.ui_layers
                .hide(&crate::ui::UiLayer::Custom(alias.clone()));
            if self.menu_stack.current() == Some(alias.as_str()) {
                let _ = self.menu_stack.pop();
            }
            info!("Popped UI scene '{}'", alias);
        } else if let Some(alias) = self.menu_stack.pop() {
            self.ui_layers
                .hide(&crate::ui::UiLayer::Custom(alias.clone()));
            info!("Popped menu scene '{}'", alias);
        } else {
            warn!("UI scene pop requested but the UI scene stack is empty");
        }
    }

    fn ui_layer_for_type(layer_type: crate::ui::UiLayerType, alias: &str) -> crate::ui::UiLayer {
        match layer_type {
            crate::ui::UiLayerType::PauseOverlay => crate::ui::UiLayer::PauseMenu,
            crate::ui::UiLayerType::MainMenu => crate::ui::UiLayer::MainMenu,
            crate::ui::UiLayerType::InGameOverlay => crate::ui::UiLayer::Hud,
            crate::ui::UiLayerType::IntermediateMenu | crate::ui::UiLayerType::Popup => {
                crate::ui::UiLayer::Custom(alias.to_string())
            }
        }
    }

    fn runtime_updates_for_scene(
        scene: &crate::project::scene::Scene,
        scene_dir: Option<&Path>,
    ) -> Vec<SceneUpdate> {
        let mut updates = vec![
            SceneUpdate::SetSandboxSettings {
                settings: scene.sandbox.clone(),
            },
            SceneUpdate::SetProceduralGeneration {
                enabled: scene.sandbox.enabled,
            },
            SceneUpdate::SetRespawnSettings {
                enabled: scene.respawn_enabled,
                y_threshold: scene.respawn_y,
            },
            SceneUpdate::SetSceneHierarchy {
                hierarchy: scene.hierarchy.clone(),
            },
        ];

        let scene_script_names = scene.scene_script_names();
        if !scene_script_names.is_empty() {
            updates.push(SceneUpdate::SetSceneScripts {
                scene_id: scene.name.clone(),
                names: scene_script_names,
            });
        }

        for entity in &scene.entities {
            updates.extend(Self::runtime_updates_for_scene_entity(entity, scene_dir));
        }

        for ui_scene in scene.ui_scenes() {
            let layer = Self::ui_layer_for_type(ui_scene.layer_type, ui_scene.alias.as_str());
            updates.push(SceneUpdate::SetUiLayer {
                layer,
                layout: ui_scene.layout,
            });
        }

        updates
    }

    fn runtime_updates_for_scene_entity(
        entity: &crate::project::scene::SceneEntity,
        scene_dir: Option<&Path>,
    ) -> Vec<SceneUpdate> {
        let mut updates = Vec::new();
        let collision_enabled = entity.effective_collision_enabled();
        let layer = entity.effective_layer();

        if let (Some(texture_id), Some(texture_data)) = (
            &entity.material.albedo_texture,
            &entity.material.albedo_texture_data,
        ) {
            updates.push(SceneUpdate::UploadTexture {
                id: texture_id.clone(),
                data: texture_data.clone(),
            });
        }

        match &entity.entity_type {
            crate::project::scene::EntityType::Camera => {
                updates.push(SceneUpdate::SetPlayerStart {
                    position: entity.position,
                    rotation: entity.rotation,
                });
                updates.push(SceneUpdate::SetActiveCamera {
                    id: entity.id,
                    fov: entity.fov,
                });
            }
            crate::project::scene::EntityType::Light {
                light_type,
                intensity,
                range,
                color,
            } => {
                let engine_light_type = match light_type {
                    crate::project::scene::LightType::Point => LightType::Point,
                    crate::project::scene::LightType::Spot => LightType::Spot,
                    crate::project::scene::LightType::Directional => LightType::Directional,
                };
                updates.push(SceneUpdate::SpawnLight {
                    id: entity.id,
                    light_type: engine_light_type,
                    position: entity.position,
                    direction: [0.0, -1.0, 0.0],
                    color: *color,
                    intensity: *intensity,
                    range: *range,
                    inner_cone: 0.4,
                    outer_cone: 0.6,
                });
            }
            crate::project::scene::EntityType::AudioSource {
                sound_id,
                volume,
                looping,
                max_distance,
                audio_data,
            } => {
                if let Some(data) = audio_data {
                    updates.push(SceneUpdate::UploadSound {
                        id: sound_id.clone(),
                        data: data.clone(),
                    });
                }
                updates.push(SceneUpdate::SpawnSound {
                    id: entity.id,
                    sound_id: sound_id.clone(),
                    position: entity.position,
                    volume: *volume,
                    looping: *looping,
                    max_distance: *max_distance,
                });
            }
            crate::project::scene::EntityType::Ground => {
                updates.push(SceneUpdate::SpawnGroundPlane {
                    id: entity.id,
                    primitive: 0,
                    position: entity.position,
                    scale: entity.scale,
                    color: entity.material.albedo_color,
                    half_extents: [entity.scale[0] / 2.0, entity.scale[2] / 2.0],
                    albedo_texture: entity.material.albedo_texture.clone(),
                    collision_enabled,
                    layer,
                });
            }
            crate::project::scene::EntityType::Mesh { path } => {
                let Some(mesh_path) = Self::resolve_asset_file(path, scene_dir) else {
                    warn!("Mesh asset '{}' could not be resolved", path);
                    return updates;
                };
                match std::fs::read(&mesh_path) {
                    Ok(mesh_data) => {
                        let path_lower = mesh_path.to_string_lossy().to_lowercase();
                        if path_lower.ends_with(".glb") || path_lower.ends_with(".gltf") {
                            updates.push(SceneUpdate::SpawnGltfMesh {
                                id: entity.id,
                                mesh_data,
                                position: entity.position,
                                rotation: entity.rotation,
                                scale: entity.scale,
                                collision_enabled,
                                layer,
                                is_static: entity.is_static,
                            });
                        } else {
                            updates.push(SceneUpdate::SpawnFbxMesh {
                                id: entity.id,
                                mesh_data,
                                position: entity.position,
                                rotation: entity.rotation,
                                scale: entity.scale,
                                albedo_texture: entity.material.albedo_texture.clone(),
                                collision_enabled,
                                layer,
                                is_static: entity.is_static,
                            });
                        }
                    }
                    Err(err) => warn!(
                        "Failed to read mesh asset '{}': {}",
                        mesh_path.display(),
                        err
                    ),
                }
            }
            _ => {
                let primitive = match &entity.entity_type {
                    crate::project::scene::EntityType::Primitive(p) => match p {
                        crate::project::scene::PrimitiveType::Cube => 0,
                        crate::project::scene::PrimitiveType::Sphere => 1,
                        crate::project::scene::PrimitiveType::Cylinder => 2,
                        crate::project::scene::PrimitiveType::Plane => 3,
                        crate::project::scene::PrimitiveType::Capsule => 4,
                        crate::project::scene::PrimitiveType::Cone => 5,
                    },
                    crate::project::scene::EntityType::Vehicle => 0,
                    crate::project::scene::EntityType::Building { .. } => 0,
                    crate::project::scene::EntityType::CrowdAgent { .. } => 4,
                    _ => 0,
                };
                updates.push(SceneUpdate::Spawn {
                    id: entity.id,
                    primitive,
                    position: entity.position,
                    rotation: entity.rotation,
                    scale: entity.scale,
                    color: entity.material.albedo_color,
                    albedo_texture: entity.material.albedo_texture.clone(),
                    collision_enabled,
                    layer,
                    is_static: entity.is_static,
                });
            }
        }

        if let Some(config) = &entity.animator_config {
            updates.push(SceneUpdate::AttachAnimator {
                id: entity.id,
                config: config.clone(),
            });
        }

        let script_names = entity.runtime_script_names();
        if !script_names.is_empty() {
            updates.push(SceneUpdate::SetScripts {
                id: entity.id,
                names: script_names,
            });
        }

        if entity.parent_id.is_some() {
            updates.push(SceneUpdate::AttachEntity {
                id: entity.id,
                parent_id: entity.parent_id,
            });
        }

        updates
    }

    fn resolve_scene_file(
        scene_path: &str,
        domain: crate::project::scene::SceneDomain,
    ) -> Option<PathBuf> {
        let raw = Path::new(scene_path);
        let mut candidates = Vec::new();

        if raw.is_absolute() {
            Self::push_json_candidate(&mut candidates, raw.to_path_buf());
        } else {
            Self::push_json_candidate(&mut candidates, raw.to_path_buf());
            let dirs: &[&str] = match domain {
                crate::project::scene::SceneDomain::World3d => {
                    &["scenes", "assets/scenes", "assets/bundle/scenes"]
                }
                crate::project::scene::SceneDomain::Ui => &[
                    "ui",
                    "assets/ui",
                    "assets/bundle/ui",
                    "assets/bundle/scenes",
                    "scenes",
                ],
            };

            for dir in dirs {
                Self::push_json_candidate(&mut candidates, Path::new(dir).join(raw));
            }
        }

        candidates.into_iter().find(|path| path.exists())
    }

    fn resolve_asset_file(asset_path: &str, scene_dir: Option<&Path>) -> Option<PathBuf> {
        let raw = Path::new(asset_path);
        let mut candidates = Vec::new();

        if raw.is_absolute() {
            candidates.push(raw.to_path_buf());
        } else {
            candidates.push(raw.to_path_buf());
            if let Some(dir) = scene_dir {
                candidates.push(dir.join(raw));
            }
            for dir in ["assets/models", "assets", "models"] {
                candidates.push(Path::new(dir).join(raw));
            }
        }

        candidates.into_iter().find(|path| path.exists())
    }

    fn push_json_candidate(candidates: &mut Vec<PathBuf>, path: PathBuf) {
        candidates.push(path.clone());
        if path.extension().is_none() {
            let mut with_json = path;
            with_json.set_extension("json");
            candidates.push(with_json);
        }
    }

    fn scene_stem(path: &str) -> Option<String> {
        Path::new(path)
            .file_stem()
            .map(|stem| stem.to_string_lossy().to_string())
    }

    fn callback_arg(callback: &str, names: &[&str]) -> Option<String> {
        for name in names {
            let prefix = format!("{}(", name);
            if callback.starts_with(&prefix) && callback.ends_with(')') {
                let inner = &callback[prefix.len()..callback.len() - 1];
                let arg = inner
                    .split(',')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .trim_matches(|c| c == '"' || c == '\'');
                if !arg.is_empty() {
                    return Some(arg.to_string());
                }
            }
        }
        None
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

    fn take_script_slots(&mut self, entity: Entity) -> Vec<ScriptSlot> {
        if let Ok(mut comp) = self.ecs.get::<&mut DynamicScript>(entity) {
            std::mem::take(&mut comp.scripts)
        } else {
            Vec::new()
        }
    }

    fn disable_script_slots(
        &mut self,
        physics: &mut PhysicsWorld,
        entity: Entity,
        slots: Vec<ScriptSlot>,
    ) {
        for mut slot in slots {
            if !self.ecs.contains(entity) {
                break;
            }

            if let Some(mut script) = slot.script.take() {
                let xr = self.xr_input.clone();
                let mut ctx = ScriptContext::new_with_xr_and_scene_commands(
                    entity,
                    &mut self.ecs,
                    physics,
                    0.0,
                    xr,
                    Some(&mut self.pending_xr_haptics),
                    Some(&mut self.pending_scene_commands),
                );
                if slot.awoken {
                    script.on_disable(&mut ctx);
                }
                script.on_destroy(&mut ctx);
            }
        }
    }

    fn disable_all_scripts(&mut self, physics: &mut PhysicsWorld, entity: Entity) {
        let slots = self.take_script_slots(entity);
        self.disable_script_slots(physics, entity, slots);
    }

    fn call_each_script<F>(
        &mut self,
        physics: &mut PhysicsWorld,
        entity: Entity,
        dt: f32,
        mut call: F,
    ) where
        F: FnMut(&mut dyn FuckScript, &mut ScriptContext),
    {
        let script_count = self
            .ecs
            .get::<&DynamicScript>(entity)
            .map(|comp| comp.scripts.len())
            .unwrap_or(0);

        for script_index in 0..script_count {
            let slot = if let Ok(mut comp) = self.ecs.get::<&mut DynamicScript>(entity) {
                comp.take_slot(script_index)
            } else {
                break;
            };

            let Some(mut slot) = slot else {
                continue;
            };

            if slot.enabled {
                if let Some(mut script) = slot.script.take() {
                    let xr = self.xr_input.clone();
                    let mut ctx = ScriptContext::new_with_xr_and_scene_commands(
                        entity,
                        &mut self.ecs,
                        physics,
                        dt,
                        xr,
                        Some(&mut self.pending_xr_haptics),
                        Some(&mut self.pending_scene_commands),
                    );
                    call(script.as_mut(), &mut ctx);
                    slot.script = Some(script);
                }
            }

            if let Ok(mut comp) = self.ecs.get::<&mut DynamicScript>(entity) {
                comp.insert_slot(script_index, slot);
            } else {
                break;
            }
        }
    }

    /// Propagate world-space transforms from parents to children
    pub fn resolve_hierarchies(&mut self) {
        // Multi-pass to handle nested hierarchies (up to 8 levels)
        for _ in 0..8 {
            let mut updates = Vec::new();

            for (entity, (hierarchy, local_transform)) in
                self.ecs.query::<(&Hierarchy, &LocalTransform)>().iter()
            {
                if let Some(parent) = hierarchy.parent {
                    if let Ok(parent_transform) = self.ecs.get::<&Transform>(parent) {
                        // Calculate world-space transform:
                        // world_pos = parent_pos + (parent_rot * (parent_scale * local_pos))
                        let world_pos = parent_transform.position
                            + (parent_transform.rotation
                                * (parent_transform.scale * local_transform.position));
                        let world_rot = parent_transform.rotation * local_transform.rotation;
                        let world_scale = parent_transform.scale * local_transform.scale;

                        // Only add to updates if it actually changed to avoid infinite loops
                        if let Ok(t) = self.ecs.get::<&Transform>(entity) {
                            if (t.position - world_pos).length_squared() > 0.0001
                                || (t.rotation.dot(world_rot)).abs() < 0.9999
                                || (t.scale - world_scale).length_squared() > 0.0001
                            {
                                updates.push((entity, world_pos, world_rot, world_scale));
                            }
                        } else {
                            updates.push((entity, world_pos, world_rot, world_scale));
                        }
                    }
                }
            }

            if updates.is_empty() {
                break;
            }

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

            let chunk_data: Vec<_> = chunks_to_generate
                .par_iter()
                .map(|&(x, z)| {
                    if !procedural_enabled {
                        return Vec::new();
                    }
                    self.calculate_chunk_entities(x, z, chunk_size)
                })
                .collect();

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

        // Process Scene Updates (Networking + scene-manager expansions)
        let mut scene_commands = VecDeque::new();
        while let Ok(cmd) = self.command_receiver.try_recv() {
            scene_commands.push_back(cmd);
        }

        while let Some(cmd) = scene_commands.pop_front() {
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

                    let target_entity = self
                        .ecs
                        .query::<&EditorEntityId>()
                        .iter()
                        .find(|(_, id_comp)| id_comp.0 == id)
                        .map(|(e, _)| e);
                    if let Some(entity) = target_entity {
                        if collision_enabled {
                            // Use is_static to control gravity. Planes (3) are always static.
                            let is_dynamic = !is_static && primitive != 3;
                            self.attach_physics_to_entity(
                                entity, physics, primitive, position, rotation, scale, is_dynamic,
                                layer,
                            );
                        }
                    }

                    info!(
                        "Spawned entity with EditorEntityId({}) using MeshHandle({}) (scale: {:?}, collision: {}, static: {})",
                        id, primitive, scale, collision_enabled, is_static
                    );
                }
                SceneUpdate::SpawnProjectile {
                    id,
                    position,
                    rotation,
                    velocity,
                    radius,
                    lifetime,
                    damage,
                    color,
                    layer,
                    gravity_scale,
                    owner,
                } => {
                    let radius = radius.max(0.01);
                    let lifetime = lifetime.max(0.01);
                    let velocity_vec = glam::Vec3::from(velocity);
                    let owner_entity = owner.and_then(Entity::from_bits);

                    let entity = self.ecs.spawn((
                        EditorEntityId(id),
                        Transform {
                            position: glam::Vec3::from(position),
                            rotation: glam::Quat::from_array(rotation),
                            scale: glam::Vec3::splat(radius * 2.0),
                        },
                        LocalTransform {
                            position: glam::Vec3::from(position),
                            rotation: glam::Quat::from_array(rotation),
                            scale: glam::Vec3::splat(radius * 2.0),
                        },
                        MeshHandle(1),
                        Material {
                            color: [color[0], color[1], color[2], 1.0],
                            albedo_texture: None,
                            metallic: 0.0,
                            roughness: 0.45,
                        },
                        Projectile {
                            velocity: velocity_vec,
                            damage: damage.max(0.0),
                            lifetime,
                            age: 0.0,
                            owner: owner_entity,
                            gravity_scale,
                        },
                    ));

                    let rb_handle = physics.add_sphere_rigid_body(
                        entity.to_bits().get() as u128,
                        position,
                        radius,
                        true,
                        layer,
                        u32::MAX,
                    );

                    if let Some(body) = physics.rigid_body_set.get_mut(rb_handle) {
                        body.set_rotation(
                            rapier3d::na::UnitQuaternion::from_quaternion(
                                rapier3d::na::Quaternion::new(
                                    rotation[3],
                                    rotation[0],
                                    rotation[1],
                                    rotation[2],
                                ),
                            ),
                            true,
                        );
                        body.set_linvel(
                            rapier3d::na::Vector3::new(velocity[0], velocity[1], velocity[2]),
                            true,
                        );
                        body.set_gravity_scale(gravity_scale, true);
                    }

                    let _ = self.ecs.insert_one(entity, RigidBodyHandle(rb_handle));
                    if let Some(script_box) = self.script_registry.create("Projectile") {
                        let _ = self.ecs.insert_one(
                            entity,
                            DynamicScript::from_named("Projectile", script_box),
                        );
                    }

                    info!(
                        "Spawned projectile {} velocity {:?}, lifetime {:.2}, damage {:.1}",
                        id, velocity, lifetime, damage
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
                SceneUpdate::UpdateTransform {
                    id,
                    position,
                    rotation,
                    scale,
                } => {
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
                                if let Some(pos) = position {
                                    local.position = glam::Vec3::from(pos);
                                }
                                if let Some(rot) = rotation {
                                    local.rotation = glam::Quat::from_array(rot);
                                }
                                if let Some(s) = scale {
                                    local.scale = glam::Vec3::from(s);
                                }
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
                        self.disable_all_scripts(physics, entity);

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
                    for (entity, _) in self.ecs.query::<&SceneScriptController>().iter() {
                        to_delete.push(entity);
                    }

                    let count = to_delete.len();
                    for entity in to_delete {
                        // Call on_disable before despawn
                        self.disable_all_scripts(physics, entity);
                        let _ = self.ecs.despawn(entity);
                    }
                    self.loaded_chunks.clear(); // Clear chunk history too
                    info!("ClearScene: Deleted {} entities", count);
                }
                SceneUpdate::SetSceneHierarchy { hierarchy } => {
                    let root_count = hierarchy.roots.len();
                    let transition_count = hierarchy.transitions.len();
                    self.scene_hierarchy = hierarchy;
                    info!(
                        "Scene hierarchy registered: {} roots, {} transitions",
                        root_count, transition_count
                    );
                }
                SceneUpdate::TransitionScene { target, mode } => {
                    self.expand_scene_transition(target, mode, &mut scene_commands);
                }
                SceneUpdate::LoadWorldScene { scene_path, mode } => {
                    self.expand_world_scene_load(&scene_path, mode, &mut scene_commands);
                }
                SceneUpdate::CallUiScene {
                    scene_path,
                    layer,
                    mode,
                } => {
                    self.expand_ui_scene_load(&scene_path, Some(layer), mode, &mut scene_commands);
                }
                SceneUpdate::CallWorldScene { scene_path, mode } => {
                    self.expand_world_scene_load(&scene_path, mode, &mut scene_commands);
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
                SceneUpdate::SetSandboxSettings { settings } => {
                    let enabled = settings.enabled;
                    let seed = settings.seed;
                    let profile = settings.profile;
                    self.set_sandbox_settings(settings);
                    info!(
                        "Sandbox settings updated: enabled={}, seed={}, profile={:?}",
                        enabled, seed, profile
                    );
                }
                SceneUpdate::SetGameMode { mode } => {
                    self.set_game_mode(mode);
                    info!("Sandbox game mode set to {:?}", mode);
                }
                SceneUpdate::SetSandboxClock {
                    time_of_day,
                    day_index,
                } => {
                    self.sandbox_clock.set_time(time_of_day, day_index);
                    info!(
                        "Sandbox clock set to day {}, time {:.3}",
                        self.sandbox_clock.day_index, self.sandbox_clock.time_of_day
                    );
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
                            let has_script_component =
                                self.ecs.get::<&DynamicScript>(entity).is_ok();

                            if has_script_component {
                                let mut comp = self
                                    .ecs
                                    .get::<&mut DynamicScript>(entity)
                                    .expect("DynamicScript disappeared while attaching script");
                                if !comp.push_named(name.clone(), script_box) {
                                    warn!(
                                        "AttachScript: Entity {} already has the max of {} scripts",
                                        id, MAX_SCRIPTS_PER_ENTITY
                                    );
                                }
                            } else {
                                self.ecs
                                    .insert_one(
                                        entity,
                                        DynamicScript::from_named(&name, script_box),
                                    )
                                    .expect("Failed to attach script");
                            }
                            info!("Attached script '{}' to entity {}", name, id);
                        } else {
                            warn!("AttachScript: No entity found with id {}", id);
                        }
                    } else {
                        warn!("AttachScript: Script '{}' not found in registry", name);
                    }
                }
                SceneUpdate::SetScripts { id, names } => {
                    let mut target_entity = None;
                    for (entity, editor_id) in self.ecs.query::<&EditorEntityId>().iter() {
                        if editor_id.0 == id {
                            target_entity = Some(entity);
                            break;
                        }
                    }

                    if let Some(entity) = target_entity {
                        self.disable_all_scripts(physics, entity);
                        let _ = self.ecs.remove_one::<DynamicScript>(entity);

                        let mut component = DynamicScript::empty();
                        for name in names.into_iter().take(MAX_SCRIPTS_PER_ENTITY) {
                            let name = name.trim();
                            if name.is_empty() {
                                continue;
                            }

                            if let Some(script_box) = self.script_registry.create(name) {
                                let _ = component.push_named(name.to_string(), script_box);
                            } else {
                                warn!("SetScripts: Script '{}' not found in registry", name);
                            }
                        }

                        if !component.is_empty() {
                            if let Err(err) = self.ecs.insert_one(entity, component) {
                                warn!(
                                    "SetScripts: Failed to attach scripts to entity {}: {:?}",
                                    id, err
                                );
                            }
                        }

                        info!("SetScripts: Updated script stack for entity {}", id);
                    } else {
                        warn!("SetScripts: No entity found with id {}", id);
                    }
                }
                SceneUpdate::SetSceneScripts { scene_id, names } => {
                    let mut target_entity = None;
                    for (entity, controller) in self.ecs.query::<&SceneScriptController>().iter() {
                        if controller.scene_id == scene_id {
                            target_entity = Some(entity);
                            break;
                        }
                    }

                    let entity = if let Some(entity) = target_entity {
                        self.disable_all_scripts(physics, entity);
                        let _ = self.ecs.remove_one::<DynamicScript>(entity);
                        entity
                    } else {
                        self.ecs.spawn((
                            SceneScriptController {
                                scene_id: scene_id.clone(),
                            },
                            Transform {
                                position: glam::Vec3::ZERO,
                                rotation: glam::Quat::IDENTITY,
                                scale: glam::Vec3::ONE,
                            },
                            LocalTransform {
                                position: glam::Vec3::ZERO,
                                rotation: glam::Quat::IDENTITY,
                                scale: glam::Vec3::ONE,
                            },
                        ))
                    };

                    let mut component = DynamicScript::empty();
                    for name in names.into_iter().take(MAX_SCENE_SCRIPTS_PER_SCENE) {
                        let name = name.trim();
                        if name.is_empty() {
                            continue;
                        }

                        if let Some(script_box) = self.script_registry.create(name) {
                            let _ = component.push_named(name.to_string(), script_box);
                        } else {
                            warn!(
                                "SetSceneScripts: Script '{}' not found in registry for scene '{}'",
                                name, scene_id
                            );
                        }
                    }

                    if component.is_empty() {
                        let _ = self.ecs.despawn(entity);
                    } else if let Err(err) = self.ecs.insert_one(entity, component) {
                        warn!(
                            "SetSceneScripts: Failed to attach scripts to scene '{}': {:?}",
                            scene_id, err
                        );
                    }

                    info!(
                        "SetSceneScripts: Updated script stack for scene '{}'",
                        scene_id
                    );
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
                            if state_cfg.is_blend_tree {
                                let root = match state_cfg.blend_tree_type {
                                    1 => BlendNode::Blend2D {
                                        param_x: state_cfg.blend_param.clone(),
                                        param_y: state_cfg
                                            .blend_param_y
                                            .clone()
                                            .unwrap_or_else(|| "VelocityY".to_string()),
                                        children: vec![(
                                            0.0,
                                            0.0,
                                            BlendNode::Clip(state_cfg.clip_index),
                                        )],
                                    },
                                    _ => BlendNode::Blend1D {
                                        param: state_cfg.blend_param.clone(),
                                        children: vec![(
                                            0.0,
                                            BlendNode::Clip(state_cfg.clip_index),
                                        )],
                                    },
                                };
                                state_machine.add_blend_tree(
                                    &state_cfg.name,
                                    BlendTree::new(&state_cfg.name, root),
                                );
                            } else {
                                state_machine.add_state(&state_cfg.name, state_cfg.clip_index);
                            }

                            if let Some(state) = state_machine.states.get_mut(&state_cfg.name) {
                                state.speed_multiplier = state_cfg.speed_multiplier;
                            }

                            // Automatically add a transition from ANY to this state if a trigger is provided
                            if let Some(ref trigger) = state_cfg.trigger_param {
                                state_machine.transitions.push(StateTransition::new(
                                    ANY_STATE,
                                    &state_cfg.name,
                                    TransitionCondition::Trigger(trigger.clone()),
                                    0.2, // Default transition duration
                                ));
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
                                )
                                .with_priority(trans_cfg.priority),
                            );
                        }

                        // Add parameters
                        for (name, param_cfg) in &config.parameters {
                            match param_cfg {
                                AnimParamConfig::Float(v) => state_machine.set_float(name, *v),
                                AnimParamConfig::Bool(v) => state_machine.set_bool(name, *v),
                                AnimParamConfig::Int(v) => state_machine.set_int(name, *v),
                                AnimParamConfig::Trigger => {
                                    state_machine
                                        .parameters
                                        .insert(name.clone(), AnimParam::Trigger(false));
                                }
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

                        if let Ok(mut animator) = self.ecs.get::<&mut Animator>(entity) {
                            animator.speed = config.speed;
                            animator.layers.clear();

                            let bone_count = animator.skeleton.bones.len();
                            let spine_start = animator
                                .skeleton
                                .bones
                                .iter()
                                .position(|bone| {
                                    let name = bone.name.to_ascii_lowercase();
                                    name.contains("spine")
                                        || name.contains("chest")
                                        || name.contains("neck")
                                        || name.contains("head")
                                })
                                .unwrap_or(bone_count / 2);

                            for layer_cfg in &config.layers {
                                if let Some(clip) =
                                    animator.clips.get(layer_cfg.clip_index).cloned()
                                {
                                    let mut layer = AnimationLayer::named(&layer_cfg.name, clip);
                                    layer.weight = layer_cfg.weight.clamp(0.0, 1.0);
                                    layer.target_weight = layer.weight;
                                    layer.additive = layer_cfg.additive;
                                    layer.sync_to_base = layer_cfg.sync_to_base;
                                    layer.avatar_mask = match layer_cfg.mask_type {
                                        1 => Some(AvatarMask::upper_body_humanoid(
                                            bone_count,
                                            spine_start,
                                        )),
                                        2 => Some(AvatarMask::lower_body_humanoid(
                                            bone_count,
                                            spine_start,
                                        )),
                                        3 => Some(AvatarMask::from_bones(
                                            &layer_cfg.name,
                                            bone_count,
                                            &layer_cfg.mask_bones,
                                        )),
                                        _ => None,
                                    };
                                    animator.add_layer(layer);
                                }
                            }

                            animator.look_at_ik =
                                if config.look_at_ik_enabled && !config.look_at_bones.is_empty() {
                                    Some(if config.look_at_bones.len() >= 4 {
                                        LookAtIK::humanoid(
                                            config.look_at_bones[0],
                                            config.look_at_bones[1],
                                            config.look_at_bones[2],
                                            config.look_at_bones[3],
                                        )
                                    } else {
                                        LookAtIK::new(config.look_at_bones[0])
                                    })
                                } else {
                                    None
                                };

                            animator.foot_ik =
                                if config.foot_ik_enabled && config.foot_ik_bones.len() >= 7 {
                                    Some(FootIK::new(
                                        config.foot_ik_bones[0],
                                        config.foot_ik_bones[1],
                                        config.foot_ik_bones[2],
                                        config.foot_ik_bones[3],
                                        config.foot_ik_bones[4],
                                        config.foot_ik_bones[5],
                                        config.foot_ik_bones[6],
                                    ))
                                } else {
                                    None
                                };
                        }

                        if self.ecs.get::<&AnimationEventQueue>(entity).is_err() {
                            let _ = self.ecs.insert_one(entity, AnimationEventQueue::new());
                        }

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

                SceneUpdate::PreviewAnimationAt {
                    id,
                    clip_index,
                    time,
                } => {
                    if let Some(entity) = self.find_by_editor_id(id) {
                        if let Ok(mut animator) = self.ecs.get::<&mut Animator>(entity) {
                            animator.play(clip_index);
                            animator.set_time(time);
                            // Set playing to false to stop automatic time updates during scrubbing
                            animator.playing = false;
                            info!(
                                "Scrubbing animation clip {} on entity {} to time {}",
                                clip_index, id, time
                            );
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

                    let target_entity = self
                        .ecs
                        .query::<&EditorEntityId>()
                        .iter()
                        .find(|(_, id_comp)| id_comp.0 == id)
                        .map(|(e, _)| e);
                    if let Some(entity) = target_entity {
                        if collision_enabled {
                            // Ground is usually Environment
                            self.attach_physics_to_entity(
                                entity,
                                physics,
                                primitive,
                                position,
                                [0.0, 0.0, 0.0, 1.0],
                                scale,
                                false,
                                layer,
                            );
                        }
                    }

                    info!(
                        "Spawned GroundPlane with EditorEntityId({}) - half_extents: {:?} (collision: {})",
                        id, half_extents, collision_enabled
                    );
                }
                SceneUpdate::UpdateMaterial {
                    id,
                    color,
                    albedo_texture,
                    metallic,
                    roughness,
                } => {
                    for (_entity, (editor_id, mat)) in
                        self.ecs.query_mut::<(&EditorEntityId, &mut Material)>()
                    {
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
                            let transform = *self
                                .ecs
                                .get::<&Transform>(entity)
                                .expect("Entity must have transform");
                            let primitive = if let Ok(h) = self.ecs.get::<&MeshHandle>(entity) {
                                h.0 as u8
                            } else {
                                0
                            };
                            // Infer dynamic. For primitives in editor, we used true unless ground (primitive 3).
                            let is_dynamic = primitive != 3;
                            self.attach_physics_to_entity(
                                entity,
                                physics,
                                primitive,
                                transform.position.into(),
                                transform.rotation.into(),
                                transform.scale.into(),
                                is_dynamic,
                                layer,
                            );
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

                                // Attach skeletal animation runtime if present.
                                if let Some(skeleton) = model_scene.skeleton.clone() {
                                    self.ecs.insert_one(entity, skeleton.clone()).unwrap();

                                    if !model_scene.animations.is_empty() {
                                        let clips = model_scene
                                            .animations
                                            .iter()
                                            .map(|clip| Arc::new(clip.clone()))
                                            .collect();
                                        let animator =
                                            Animator::new(Arc::new(skeleton.clone()), clips);
                                        let anim_state = AnimationState::new(skeleton.bones.len());
                                        let _ = self.ecs.insert(
                                            entity,
                                            (animator, anim_state, AnimationEventQueue::new()),
                                        );
                                    }
                                }

                                if !model_scene.animations.is_empty() {
                                    info!(
                                        "glTF model {} loaded with {} animations",
                                        id,
                                        model_scene.animations.len()
                                    );
                                }

                                if collision_enabled {
                                    self.attach_physics_to_entity(
                                        entity, physics,
                                        255, // 255 = custom mesh using its AABB
                                        position, rotation, scale, !is_static, layer,
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
                SceneUpdate::SetRespawnSettings {
                    enabled,
                    y_threshold,
                } => {
                    self.respawn_enabled = enabled;
                    self.respawn_y = y_threshold;
                    info!(
                        "Respawn settings updated: enabled={}, y={}",
                        enabled, y_threshold
                    );
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
                                        entity, physics,
                                        255, // 255 = custom mesh using its AABB
                                        position, rotation, scale, !is_static, layer,
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
                    info!(
                        "UI Layout updated (Hud layer): {} buttons, {} panels, {} texts",
                        self.ui_layers
                            .get_layer(&crate::ui::UiLayer::Hud)
                            .map(|l| l.buttons.len())
                            .unwrap_or(0),
                        self.ui_layers
                            .get_layer(&crate::ui::UiLayer::Hud)
                            .map(|l| l.panels.len())
                            .unwrap_or(0),
                        self.ui_layers
                            .get_layer(&crate::ui::UiLayer::Hud)
                            .map(|l| l.texts.len())
                            .unwrap_or(0)
                    );
                }
                SceneUpdate::SetUiLayer { layer, layout } => {
                    info!("Setting UI layer {:?}", layer);
                    let should_show = matches!(
                        layout.layer_type,
                        crate::ui::UiLayerType::InGameOverlay | crate::ui::UiLayerType::MainMenu
                    );

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
                    self.expand_ui_scene_load(
                        &scene_path,
                        Some(layer),
                        crate::project::scene::SceneTransitionMode::Additive,
                        &mut scene_commands,
                    );
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
                    self.ui_layers
                        .set_layer(crate::ui::UiLayer::PauseMenu, default_layout);
                }
            }
        }

        // Example: Spawn a task to load chunk 1
        // self.request_chunk(1);
    }

    /// Update all animated entities - ticks Animator components and uploads bone matrices
    /// Call this each frame to advance skeletal animations
    pub fn update_animations(&mut self, physics: &mut PhysicsWorld, dt: f32) {
        // Clear previous frame events before generating this frame's events.
        for (_, queue) in self.ecs.query::<&mut AnimationEventQueue>().iter() {
            queue.clear();
        }

        // Collect entities with Animator.
        let entities: Vec<hecs::Entity> = self
            .ecs
            .query::<&Animator>()
            .iter()
            .map(|(e, _)| e)
            .collect();

        struct AnimationJob {
            entity: hecs::Entity,
            animator: Animator,
            params: std::collections::HashMap<String, AnimParam>,
            dt: f32,
            speed_multiplier: f32,
        }

        struct AnimationResult {
            entity: hecs::Entity,
            animator: Animator,
            matrices: Vec<glam::Mat4>,
            events: Vec<crate::world::fbx_loader::AnimationEvent>,
        }

        let mut jobs = Vec::with_capacity(entities.len());

        for entity in entities {
            let mut params = std::collections::HashMap::new();
            let mut speed_mult = 1.0f32;

            let (current_time, current_duration) = self
                .ecs
                .get::<&Animator>(entity)
                .map(|animator| (animator.time, animator.current_duration()))
                .unwrap_or((0.0, 0.0));

            // 1. Process AnimatorController (if present) to drive state machine
            let mut transition = None;
            if let Ok(mut controller) = self.ecs.get::<&mut AnimatorController>(entity) {
                if controller.enabled {
                    controller
                        .state_machine
                        .apply_parameter_curves(current_time);

                    // Evaluate state machine for transitions
                    transition = controller
                        .state_machine
                        .evaluate_with_time(current_time, current_duration);

                    // Copy parameters for blend tree evaluation
                    params = controller.state_machine.parameters.clone();
                    speed_mult =
                        controller.speed * controller.state_machine.current_speed_multiplier();
                }
            }

            if let Some((resource, blend_duration)) = transition {
                if let Ok(mut animator) = self.ecs.get::<&mut Animator>(entity) {
                    animator.crossfade_to(resource, blend_duration);
                }
            }

            if let Ok(animator) = self.ecs.get::<&Animator>(entity) {
                jobs.push(AnimationJob {
                    entity,
                    animator: (*animator).clone(),
                    params,
                    dt,
                    speed_multiplier: speed_mult,
                });
            }
        }

        let results: Vec<AnimationResult> = {
            use rayon::prelude::*;
            jobs.into_par_iter()
                .map(|mut job| {
                    let old_speed = job.animator.speed;
                    job.animator.speed *= job.speed_multiplier;
                    let (matrices, events) =
                        job.animator.update_collect_events(job.dt, &job.params);
                    job.animator.speed = old_speed;

                    AnimationResult {
                        entity: job.entity,
                        animator: job.animator,
                        matrices,
                        events,
                    }
                })
                .collect()
        };

        for result in results {
            let entity = result.entity;
            let root_motion = result.animator.root_motion;
            let matrices = result.matrices;
            let events = result.events;

            if let Ok(mut animator) = self.ecs.get::<&mut Animator>(entity) {
                *animator = result.animator;
            }

            // 3. Update AnimationState (GPU matrix buffer)
            if let Ok(mut anim_state) = self.ecs.get::<&mut AnimationState>(entity) {
                anim_state.update_from(matrices);
            }

            if !events.is_empty() {
                let mut pending_events = Some(events);
                let had_queue = {
                    if let Ok(mut queue) = self.ecs.get::<&mut AnimationEventQueue>(entity) {
                        if let Some(events) = pending_events.take() {
                            queue.events.extend(events);
                        }
                        true
                    } else {
                        false
                    }
                };

                if !had_queue {
                    if let Some(events) = pending_events.take() {
                        let _ = self.ecs.insert_one(entity, AnimationEventQueue { events });
                    }
                }
            }

            // 4. Apply Root Motion (if enabled)
            let apply_rm = if let Ok(controller) = self.ecs.get::<&AnimatorController>(entity) {
                controller.apply_root_motion
            } else {
                false
            };

            if apply_rm && root_motion.has_motion {
                let delta_pos = root_motion.delta_position;
                let delta_rot = root_motion.delta_rotation;

                if let Ok(mut transform) = self.ecs.get::<&mut Transform>(entity) {
                    let rotated_delta = transform.rotation * delta_pos;
                    transform.position += rotated_delta;
                    transform.rotation = transform.rotation * delta_rot;

                    if let Ok(rb_handle) = self.ecs.get::<&RigidBodyHandle>(entity) {
                        if let Some(body) = physics.rigid_body_set.get_mut(rb_handle.0) {
                            let p = transform.position;
                            let r = transform.rotation;
                            body.set_next_kinematic_translation(rapier3d::na::Vector3::new(
                                p.x, p.y, p.z,
                            ));
                            body.set_next_kinematic_rotation(
                                rapier3d::na::UnitQuaternion::from_quaternion(
                                    rapier3d::na::Quaternion::new(r.w, r.x, r.y, r.z),
                                ),
                            );
                        }
                    }
                }
            }
        }
    }

    pub fn update_logic(
        &mut self,
        physics: &mut PhysicsWorld,
        dt: f32,
        ui_events: Vec<crate::ui::UiEvent>,
    ) {
        self.advance_sandbox_clock(dt);

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
        let xr_events = std::mem::take(&mut self.pending_xr_events);
        let entities: Vec<Entity> = self
            .ecs
            .query::<&DynamicScript>()
            .iter()
            .map(|(e, _)| e)
            .collect();

        for entity in entities {
            let script_count = self
                .ecs
                .get::<&DynamicScript>(entity)
                .map(|comp| comp.scripts.len())
                .unwrap_or(0);

            for script_index in 0..script_count {
                let slot = if let Ok(mut comp) = self.ecs.get::<&mut DynamicScript>(entity) {
                    comp.take_slot(script_index)
                } else {
                    break;
                };

                let Some(mut slot) = slot else {
                    continue;
                };

                if slot.enabled {
                    if let Some(mut s) = slot.script.take() {
                        // Check for animation events to dispatch
                        let anim_events: Vec<String> =
                            if let Ok(queue) = self.ecs.get::<&AnimationEventQueue>(entity) {
                                queue.events.iter().map(|e| e.name.clone()).collect()
                            } else {
                                Vec::new()
                            };

                        let xr = self.xr_input.clone();
                        let mut ctx = ScriptContext::new_with_xr_and_scene_commands(
                            entity,
                            &mut self.ecs,
                            physics,
                            dt,
                            xr,
                            Some(&mut self.pending_xr_haptics),
                            Some(&mut self.pending_scene_commands),
                        );

                        if !slot.awoken {
                            s.on_awake(&mut ctx);
                            slot.awoken = true;
                        }
                        if !slot.started {
                            s.on_start(&mut ctx);
                            s.on_enable(&mut ctx);
                            slot.started = true;
                        }
                        for event in &xr_events {
                            s.on_xr_input(&mut ctx, event);
                        }
                        let xr_frame = ctx.xr.clone();
                        s.on_xr_frame(&mut ctx, &xr_frame);

                        // Dispatch animation events
                        for event_name in anim_events {
                            s.on_animation_event(&mut ctx, &event_name);
                        }

                        s.on_fixed_update(&mut ctx);
                        s.on_update(&mut ctx);
                        s.on_late_update(&mut ctx);
                        slot.script = Some(s);
                    }
                }

                if let Ok(mut comp) = self.ecs.get::<&mut DynamicScript>(entity) {
                    comp.insert_slot(script_index, slot);
                } else {
                    break;
                }
            }
        }

        self.flush_pending_scene_commands();
    }

    fn update_agents_parallel(&mut self, dt: f32) {
        use rayon::prelude::*;

        // Gather all agents and their transforms
        let agent_data: Vec<_> = self
            .ecs
            .query_mut::<(&mut CrowdAgent, &Transform)>()
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
        let results: Vec<_> = agent_data
            .into_par_iter()
            .map(|(entity, mut agent, transform)| {
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
                    agent.target =
                        glam::Vec3::new((rand() - 0.5) * 60.0, pos.y, (rand() - 0.5) * 60.0);
                    agent.velocity =
                        glam::Vec3::new(rand() - 0.5, 0.0, rand() - 0.5).normalize() * 2.0;
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
                    agent.target =
                        glam::Vec3::new((rand() - 0.5) * 60.0, pos.y, (rand() - 0.5) * 60.0);
                    agent.velocity = (agent.target - pos).normalize() * 0.1;
                    (entity, agent, None)
                } else {
                    let desired = to_target.normalize() * agent.max_speed;
                    let steer_force = if agent.state == AgentState::Fleeing {
                        20.0
                    } else {
                        8.0
                    };
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
            })
            .collect();

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

    fn dispatch_collision_event(
        &mut self,
        physics: &mut PhysicsWorld,
        c1: rapier3d::prelude::ColliderHandle,
        c2: rapier3d::prelude::ColliderHandle,
        started: bool,
    ) {
        let e1 = self.get_entity_from_collider(physics, c1);
        let e2 = self.get_entity_from_collider(physics, c2);

        if started {
            println!("DEBUG: COLLISION START between {:?} and {:?}", e1, e2);
        } else {
            println!("DEBUG: COLLISION STOP between {:?} and {:?}", e1, e2);
        }

        if let Some(ent1) = e1 {
            if let Some(ent2) = e2 {
                self.call_collision_on_script(physics, ent1, ent2, started);
                self.call_collision_on_script(physics, ent2, ent1, started);
            }
        }
    }

    fn get_entity_from_collider(
        &self,
        physics: &PhysicsWorld,
        handle: rapier3d::prelude::ColliderHandle,
    ) -> Option<Entity> {
        let collider = physics.collider_set.get(handle)?;
        let rb_handle = collider.parent()?;
        let rb = physics.rigid_body_set.get(rb_handle)?;
        let bits = rb.user_data as u64;
        if bits == 0 {
            // println!("DEBUG: Collider {:?} has no user_data", handle);
            return None;
        }
        let entity = Entity::from_bits(bits);
        // println!("DEBUG: Collider {:?} mapped to entity {:?}", handle, entity);
        entity
    }

    fn call_collision_on_script(
        &mut self,
        physics: &mut PhysicsWorld,
        entity: Entity,
        other: Entity,
        started: bool,
    ) {
        self.call_each_script(physics, entity, 0.0, |script, ctx| {
            if started {
                script.on_collision_start(ctx, other);
            } else {
                script.on_collision_end(ctx, other);
            }
        });
    }

    fn dispatch_ui_event(&mut self, physics: &mut PhysicsWorld, event: &crate::ui::UiEvent) {
        // First, check if callback is a built-in engine command
        let crate::ui::UiEvent::ButtonClicked { callback, .. } = event;
        // Parse menu_load("alias") pattern
        if callback.starts_with("menu_load(") && callback.ends_with(")") {
            let inner = &callback[10..callback.len() - 1];
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
                    info!(
                        "menu_load: Found layer '{}' with type {:?}",
                        alias, layout.layer_type
                    );
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
                info!(
                    "menu_load: Showing layer '{}', menu_stack depth: {}",
                    alias,
                    self.menu_stack.depth()
                );
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

        if let Some(scene_path) = Self::callback_arg(
            callback,
            &[
                "load_world_scene",
                "call_world_scene",
                "load_scene",
                "scene_load",
            ],
        ) {
            info!("Built-in world scene callback: loading '{}'", scene_path);
            self.queue_scene_command(SceneUpdate::CallWorldScene {
                scene_path,
                mode: crate::project::scene::SceneTransitionMode::Replace,
            });
            return;
        }

        if let Some(scene_path) =
            Self::callback_arg(callback, &["load_ui_scene", "call_ui_scene", "ui_scene"])
        {
            let alias = Self::scene_stem(&scene_path).unwrap_or_else(|| "ui".to_string());
            info!("Built-in UI scene callback: loading '{}'", scene_path);
            self.queue_scene_command(SceneUpdate::CallUiScene {
                scene_path,
                layer: crate::ui::UiLayer::Custom(alias),
                mode: crate::project::scene::SceneTransitionMode::Push,
            });
            return;
        }

        if callback == "ui_scene_back()" || callback == "ui_scene_back" {
            self.queue_scene_command(SceneUpdate::CallUiScene {
                scene_path: String::new(),
                layer: crate::ui::UiLayer::Custom(String::new()),
                mode: crate::project::scene::SceneTransitionMode::Pop,
            });
            return;
        }

        if callback == "world_scene_back()" || callback == "world_scene_back" {
            self.queue_scene_command(SceneUpdate::CallWorldScene {
                scene_path: String::new(),
                mode: crate::project::scene::SceneTransitionMode::Pop,
            });
            return;
        }

        // Then pass to scripts for custom callbacks
        let entities: Vec<Entity> = self
            .ecs
            .query::<&DynamicScript>()
            .iter()
            .map(|(e, _)| e)
            .collect();

        for entity in entities {
            self.call_each_script(physics, entity, 0.0, |script, ctx| {
                script.on_ui_event(ctx, event);
            });
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
                physics.add_sphere_rigid_body(
                    entity_bits,
                    position,
                    scale[0] * 0.5,
                    is_dynamic,
                    layer,
                    filter,
                )
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
                    info!(
                        "Custom mesh physics: half_extents={:?}, y_offset={}",
                        half_extents, y_offset
                    );
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

    fn calculate_chunk_entities(
        &self,
        cx: i32,
        cz: i32,
        size: f32,
    ) -> Vec<(Transform, MeshHandle, Material)> {
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
        info!(
            "  Model {} '{}': {} vertices, {} indices, has_normals: {}",
            model_idx,
            model.name,
            vertex_count,
            mesh.indices.len(),
            has_normals
        );

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
        info!(
            "Computing face normals for {} triangles",
            all_indices.len() / 3
        );

        // Accumulate face normals to vertices
        for tri in all_indices.chunks(3) {
            if tri.len() < 3 {
                continue;
            }
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

            info!(
                "Normalizing mesh (Base-Origin): min_y={}, center_xz={:?}, scale={}",
                min.y,
                (center.x, center.z),
                scale
            );

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

    info!(
        "OBJ total: {} vertices, {} indices",
        all_vertices.len(),
        all_indices.len()
    );

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
            vertices.push(make_vert([-0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.0, 0.0])); // BL
            vertices.push(make_vert([0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [1.0, 0.0])); // BR
            vertices.push(make_vert([0.5, 0.0, -0.5], [0.0, 1.0, 0.0], [1.0, 1.0])); // TR
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

#[cfg(test)]
mod tests {
    use super::*;

    struct NoopScript;

    impl FuckScript for NoopScript {}

    #[test]
    fn dynamic_script_can_stack_multiple_scripts() {
        let mut component = DynamicScript::from_named("First", Box::new(NoopScript));
        assert!(component.push_named("Second", Box::new(NoopScript)));

        assert_eq!(component.scripts.len(), 2);
        assert_eq!(component.scripts[0].name, "First");
        assert_eq!(component.scripts[1].name, "Second");

        let slot = component.take_slot(0).expect("first script slot exists");
        assert_eq!(slot.name, "First");
        component.insert_slot(0, slot);
        assert_eq!(component.scripts[0].name, "First");
    }

    #[test]
    fn dynamic_script_stack_is_capped() {
        let mut component = DynamicScript::empty();
        for index in 0..MAX_SCRIPTS_PER_ENTITY {
            assert!(component.push_named(format!("Script{}", index), Box::new(NoopScript)));
        }

        assert!(!component.push_named("Overflow", Box::new(NoopScript)));
        assert_eq!(component.scripts.len(), MAX_SCRIPTS_PER_ENTITY);
    }

    #[test]
    fn scene_script_stack_is_capped_at_scene_limit() {
        let mut world = GameWorld::new();
        let mut physics = PhysicsWorld::new();
        world
            .command_sender
            .try_send(SceneUpdate::SetSceneScripts {
                scene_id: "test-scene".to_string(),
                names: vec![
                    "TestBounce".to_string(),
                    "CrowdAgent".to_string(),
                    "Vehicle".to_string(),
                    "PoliceAgent".to_string(),
                    "TrafficAI".to_string(),
                    "EnemyTracker".to_string(),
                    "WeaponNPC".to_string(),
                    "GunWeapon".to_string(),
                    "BowWeapon".to_string(),
                    "Projectile".to_string(),
                ],
            })
            .expect("scene script command should enqueue");

        world.update_streaming(glam::Vec3::ZERO, &mut physics);

        let controller = world
            .ecs
            .query::<(&SceneScriptController, &DynamicScript)>()
            .iter()
            .find(|(_, (controller, _))| controller.scene_id == "test-scene")
            .map(|(_, (_, scripts))| scripts.scripts.len())
            .expect("scene script controller should exist");
        assert_eq!(controller, MAX_SCENE_SCRIPTS_PER_SCENE);
        assert_eq!(MAX_SCRIPTS_PER_ENTITY, 32);
    }

    #[test]
    fn set_scripts_vehicle_drives_through_runtime_update_path() {
        let mut world = GameWorld::new();
        let mut physics = PhysicsWorld::new();

        world
            .command_sender
            .try_send(SceneUpdate::Spawn {
                id: 3,
                primitive: 0,
                position: [0.0, 2.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [2.0, 1.0, 4.0],
                color: [0.8, 0.8, 0.8],
                albedo_texture: None,
                collision_enabled: true,
                layer: LAYER_VEHICLE,
                is_static: false,
            })
            .expect("spawn command should enqueue");
        world
            .command_sender
            .try_send(SceneUpdate::SetScripts {
                id: 3,
                names: vec!["Vehicle".to_string()],
            })
            .expect("script command should enqueue");

        world.update_streaming(glam::Vec3::ZERO, &mut physics);

        let vehicle_entity = world.find_by_editor_id(3).expect("vehicle should spawn");
        let rb_handle = world
            .ecs
            .get::<&RigidBodyHandle>(vehicle_entity)
            .expect("vehicle should have physics")
            .0;
        let script_names = world
            .ecs
            .get::<&DynamicScript>(vehicle_entity)
            .expect("vehicle should have dynamic script")
            .scripts
            .iter()
            .map(|slot| slot.name.clone())
            .collect::<Vec<_>>();
        assert_eq!(script_names, vec!["Vehicle"]);

        world.update_logic(&mut physics, 1.0 / 60.0, Vec::new());

        let body = physics
            .rigid_body_set
            .get(rb_handle)
            .expect("vehicle body should remain alive");
        assert!(
            body.linvel().z < -0.001,
            "vehicle script should drive forward, linvel was {:?}",
            body.linvel()
        );
    }

    #[test]
    fn set_scripts_vehicle_changes_position_after_full_physics_frames() {
        let mut world = GameWorld::new();
        let mut physics = PhysicsWorld::new();

        for command in [
            SceneUpdate::SpawnGroundPlane {
                id: 1,
                primitive: 0,
                position: [0.0, -1.1, 0.0],
                scale: [100.0, 2.0, 100.0],
                color: [0.8, 0.8, 0.8],
                half_extents: [50.0, 50.0],
                albedo_texture: None,
                collision_enabled: true,
                layer: LAYER_ENVIRONMENT,
            },
            SceneUpdate::Spawn {
                id: 3,
                primitive: 0,
                position: [0.0, 2.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [2.0, 1.0, 4.0],
                color: [0.8, 0.8, 0.8],
                albedo_texture: None,
                collision_enabled: true,
                layer: LAYER_VEHICLE,
                is_static: false,
            },
            SceneUpdate::SetScripts {
                id: 3,
                names: vec!["Vehicle".to_string()],
            },
        ] {
            world
                .command_sender
                .try_send(command)
                .expect("scene command should enqueue");
        }

        world.update_streaming(glam::Vec3::ZERO, &mut physics);
        let vehicle_entity = world.find_by_editor_id(3).expect("vehicle should spawn");
        let rb_handle = world
            .ecs
            .get::<&RigidBodyHandle>(vehicle_entity)
            .expect("vehicle should have physics")
            .0;
        let start_z = physics
            .rigid_body_set
            .get(rb_handle)
            .unwrap()
            .translation()
            .z;
        let start_rotation = *physics.rigid_body_set.get(rb_handle).unwrap().rotation();

        for _ in 0..60 {
            physics.step_with_dt(1.0 / 60.0);
            world.update_logic(&mut physics, 1.0 / 60.0, Vec::new());
        }

        let body = physics.rigid_body_set.get(rb_handle).unwrap();
        assert!(
            body.translation().z < start_z - 1.0,
            "vehicle should move visibly forward, start_z={}, end_z={}, linvel={:?}",
            start_z,
            body.translation().z,
            body.linvel()
        );
        assert!(
            start_rotation.angle_to(body.rotation()) < 0.05,
            "vehicle should cruise mostly straight before reaching map edge, start_rot={:?}, end_rot={:?}",
            start_rotation,
            body.rotation()
        );
    }

    #[test]
    fn set_scripts_vehicle_steers_when_near_map_edge() {
        let mut world = GameWorld::new();
        let mut physics = PhysicsWorld::new();

        for command in [
            SceneUpdate::SpawnGroundPlane {
                id: 1,
                primitive: 0,
                position: [0.0, -1.1, 0.0],
                scale: [100.0, 2.0, 100.0],
                color: [0.8, 0.8, 0.8],
                half_extents: [50.0, 50.0],
                albedo_texture: None,
                collision_enabled: true,
                layer: LAYER_ENVIRONMENT,
            },
            SceneUpdate::Spawn {
                id: 3,
                primitive: 0,
                position: [0.0, 2.0, -44.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [2.0, 1.0, 4.0],
                color: [0.8, 0.8, 0.8],
                albedo_texture: None,
                collision_enabled: true,
                layer: LAYER_VEHICLE,
                is_static: false,
            },
            SceneUpdate::SetScripts {
                id: 3,
                names: vec!["Vehicle".to_string()],
            },
        ] {
            world
                .command_sender
                .try_send(command)
                .expect("scene command should enqueue");
        }

        world.update_streaming(glam::Vec3::ZERO, &mut physics);
        let vehicle_entity = world.find_by_editor_id(3).expect("vehicle should spawn");
        let rb_handle = world
            .ecs
            .get::<&RigidBodyHandle>(vehicle_entity)
            .expect("vehicle should have physics")
            .0;
        let start_rotation = *physics.rigid_body_set.get(rb_handle).unwrap().rotation();
        let mut min_z = physics
            .rigid_body_set
            .get(rb_handle)
            .unwrap()
            .translation()
            .z;

        for _ in 0..75 {
            physics.step_with_dt(1.0 / 60.0);
            world.update_logic(&mut physics, 1.0 / 60.0, Vec::new());
            min_z = min_z.min(
                physics
                    .rigid_body_set
                    .get(rb_handle)
                    .unwrap()
                    .translation()
                    .z,
            );
        }

        let body = physics.rigid_body_set.get(rb_handle).unwrap();
        assert!(
            start_rotation.angle_to(body.rotation()) > 0.2,
            "vehicle should steer hard near edge, start_rot={:?}, end_rot={:?}",
            start_rotation,
            body.rotation()
        );
        assert!(
            min_z >= -50.0,
            "vehicle should turn before running off the map edge, min_z={}, final_pos={:?}",
            min_z,
            body.translation(),
        );
    }

    #[test]
    fn set_scripts_vehicle_stays_on_ground_during_demo_loop() {
        let mut world = GameWorld::new();
        let mut physics = PhysicsWorld::new();

        for command in [
            SceneUpdate::SpawnGroundPlane {
                id: 1,
                primitive: 0,
                position: [0.0, -1.1, 0.0],
                scale: [100.0, 2.0, 100.0],
                color: [0.8, 0.8, 0.8],
                half_extents: [50.0, 50.0],
                albedo_texture: None,
                collision_enabled: true,
                layer: LAYER_ENVIRONMENT,
            },
            SceneUpdate::Spawn {
                id: 3,
                primitive: 0,
                position: [0.0, 2.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [2.0, 1.0, 4.0],
                color: [0.8, 0.8, 0.8],
                albedo_texture: None,
                collision_enabled: true,
                layer: LAYER_VEHICLE,
                is_static: false,
            },
            SceneUpdate::SetScripts {
                id: 3,
                names: vec!["Vehicle".to_string()],
            },
        ] {
            world
                .command_sender
                .try_send(command)
                .expect("scene command should enqueue");
        }

        world.update_streaming(glam::Vec3::ZERO, &mut physics);
        let vehicle_entity = world.find_by_editor_id(3).expect("vehicle should spawn");
        let rb_handle = world
            .ecs
            .get::<&RigidBodyHandle>(vehicle_entity)
            .expect("vehicle should have physics")
            .0;
        let mut max_abs_x = 0.0f32;
        let mut max_abs_z = 0.0f32;

        for _ in 0..420 {
            physics.step_with_dt(1.0 / 60.0);
            world.update_logic(&mut physics, 1.0 / 60.0, Vec::new());
            let pos = physics.rigid_body_set.get(rb_handle).unwrap().translation();
            max_abs_x = max_abs_x.max(pos.x.abs());
            max_abs_z = max_abs_z.max(pos.z.abs());
        }

        assert!(
            max_abs_x <= 50.0 && max_abs_z <= 50.0,
            "vehicle should stay on the 100x100 ground during the demo loop, max_abs_x={}, max_abs_z={}",
            max_abs_x,
            max_abs_z
        );
    }
}
