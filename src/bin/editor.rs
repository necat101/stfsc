use eframe::egui;
use rayon::prelude::*;
use std::io::Write;
use std::net::TcpStream;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use stfsc_engine::world::{
    LightType, SceneUpdate, LAYER_ENVIRONMENT, LAYER_PROP, LAYER_CHARACTER, LAYER_VEHICLE, LAYER_DEFAULT,
    animation::{EditorKeyframe, KeyframeChannel, KeyframeValue, KeyframeInterpolation},
    fbx_loader::ModelScene,
};
use stfsc_engine::graphics::occlusion::{Frustum, AABB};
use glam::{Mat4, Vec3 as GVec3, Quat as GQuat};
use std::collections::HashMap;
use std::sync::Arc;

fn main() -> Result<(), eframe::Error> {
    if let Err(e) = std::process::Command::new("adb")
        .args(&["forward", "tcp:8080", "tcp:8080"])
        .output()
    {
        println!("Warning: adb forward failed: {}", e);
    } else {
        println!("ADB port forwarded.");
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 1000.0])
            .with_title("STFSC Editor - 556 Engine"),
        ..Default::default()
    };
    eframe::run_native(
        "STFSC Editor",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Box::new(EditorApp::new())
        }),
    )
}

// ============================================================================
// 3D MATH
// ============================================================================
#[derive(Clone, Copy)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}
impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
    fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    fn normalize(&self) -> Self {
        let l = self.length().max(0.0001);
        Self::new(self.x / l, self.y / l, self.z / l)
    }
    fn cross(&self, b: &Vec3) -> Vec3 {
        Vec3::new(
            self.y * b.z - self.z * b.y,
            self.z * b.x - self.x * b.z,
            self.x * b.y - self.y * b.x,
        )
    }
    fn dot(&self, b: &Vec3) -> f32 {
        self.x * b.x + self.y * b.y + self.z * b.z
    }
    fn sub(&self, b: &Vec3) -> Vec3 {
        Vec3::new(self.x - b.x, self.y - b.y, self.z - b.z)
    }
    fn add(&self, b: &Vec3) -> Vec3 {
        Vec3::new(self.x + b.x, self.y + b.y, self.z + b.z)
    }
    fn mul(&self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}

/// Pre-processed vertex for optimized software rendering
#[derive(Clone, Copy)]
struct ProcessedVertex {
    pos: egui::Pos2,
    color: egui::Color32,
    uv: egui::Pos2,
    depth: f32,
    world_pos: GVec3,
    is_valid: bool,
}

#[derive(Clone, Copy)]
struct Camera3D {
    target: Vec3,
    distance: f32,
    yaw: f32,
    pitch: f32,
    fov: f32,
}
impl Camera3D {
    fn new() -> Self {
        Self {
            target: Vec3::zero(),
            distance: 50.0,
            yaw: 0.5,
            pitch: 0.6,
            fov: 60.0,
        }
    }
    fn get_position(&self) -> Vec3 {
        Vec3::new(
            self.target.x + self.distance * self.pitch.cos() * self.yaw.sin(),
            self.target.y + self.distance * self.pitch.sin(),
            self.target.z + self.distance * self.pitch.cos() * self.yaw.cos(),
        )
    }
    fn get_forward(&self) -> Vec3 {
        self.target.sub(&self.get_position()).normalize()
    }
    fn get_right(&self) -> Vec3 {
        self.get_forward()
            .cross(&Vec3::new(0.0, 1.0, 0.0))
            .normalize()
    }
    fn project(&self, world: Vec3, size: egui::Vec2) -> Option<egui::Pos2> {
        let cam = self.get_position();
        let fwd = self.get_forward();
        let right = self.get_right();
        let up = right.cross(&fwd);
        let rel = world.sub(&cam);
        let z = rel.dot(&fwd);
        if z < 0.5 {
            return None;
        }
        let x = rel.dot(&right);
        let y = rel.dot(&up);
        let aspect = size.x / size.y;
        let scale = (self.fov.to_radians() * 0.5).tan();
        let ndc_x = x / (z * scale * aspect);
        let ndc_y = y / (z * scale);
        if ndc_x.abs() > 1.5 || ndc_y.abs() > 1.5 {
            return None;
        }
        Some(egui::pos2(
            size.x * 0.5 + ndc_x * size.x * 0.5,
            size.y * 0.5 - ndc_y * size.y * 0.5,
        ))
    }
    /// Project without culling off-screen points - used for mesh rendering where triangles may span view boundaries
    fn project_unclamped(&self, world: Vec3, size: egui::Vec2) -> (egui::Pos2, bool) {
        let cam = self.get_position();
        let fwd = self.get_forward();
        let right = self.get_right();
        let up = right.cross(&fwd);
        let rel = world.sub(&cam);
        let z = rel.dot(&fwd);
        if z < 0.01 {
            // Behind camera - return invalid
            return (egui::Pos2::ZERO, false);
        }
        let x = rel.dot(&right);
        let y = rel.dot(&up);
        let aspect = size.x / size.y;
        let scale = (self.fov.to_radians() * 0.5).tan();
        let ndc_x = x / (z * scale * aspect);
        let ndc_y = y / (z * scale);
        // Don't reject based on NDC bounds - allow off-screen vertices
        (egui::pos2(
            size.x * 0.5 + ndc_x * size.x * 0.5,
            size.y * 0.5 - ndc_y * size.y * 0.5,
        ), true)
    }
    fn get_ray(&self, screen_pos: egui::Pos2, size: egui::Vec2) -> (Vec3, Vec3) {
        let aspect = size.x / size.y;
        let scale = (self.fov.to_radians() * 0.5).tan();
        let ndc_x = (screen_pos.x / size.x - 0.5) * 2.0;
        let ndc_y = -(screen_pos.y / size.y - 0.5) * 2.0;
        let fwd = self.get_forward();
        let right = self.get_right();
        let up = right.cross(&fwd);
        let dir = fwd
            .add(&right.mul(ndc_x * scale * aspect))
            .add(&up.mul(ndc_y * scale))
            .normalize();
        (self.get_position(), dir)
    }
    
    /// Project a world point to screen coordinates, returning depth (view-space Z) as well
    fn get_view_proj(&self, aspect: f32) -> Mat4 {
        let cam_pos = self.get_position();
        let cam_pos_g = GVec3::new(cam_pos.x, cam_pos.y, cam_pos.z);
        let fwd = self.get_forward();
        let fwd_g = GVec3::new(fwd.x, fwd.y, fwd.z);
        let right = self.get_right();
        let right_g = GVec3::new(right.x, right.y, right.z);
        let up_g = right_g.cross(fwd_g);
        
        let view = Mat4::look_at_rh(cam_pos_g, cam_pos_g + fwd_g, up_g);
        let proj = Mat4::perspective_rh(self.fov.to_radians(), aspect, 0.1, 1000.0);
        proj * view
    }
}
    


// ============================================================================
// PRIMITIVES LIBRARY (Unity-style)
// ============================================================================
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
enum PrimitiveType {
    Cube,
    Sphere,
    Cylinder,
    Plane,
    Capsule,
    Cone,
}

impl PrimitiveType {
    fn all() -> Vec<PrimitiveType> {
        vec![
            PrimitiveType::Cube,
            PrimitiveType::Sphere,
            PrimitiveType::Cylinder,
            PrimitiveType::Plane,
            PrimitiveType::Capsule,
            PrimitiveType::Cone,
        ]
    }
    fn name(&self) -> &str {
        match self {
            PrimitiveType::Cube => "Cube",
            PrimitiveType::Sphere => "Sphere",
            PrimitiveType::Cylinder => "Cylinder",
            PrimitiveType::Plane => "Plane",
            PrimitiveType::Capsule => "Capsule",
            PrimitiveType::Cone => "Cone",
        }
    }
    fn icon(&self) -> &str {
        match self {
            PrimitiveType::Cube => "🟧",
            PrimitiveType::Sphere => "⚪",
            PrimitiveType::Cylinder => "🔷",
            PrimitiveType::Plane => "▬",
            PrimitiveType::Capsule => "💊",
            PrimitiveType::Cone => "🔺",
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Material {
    name: String,
    albedo_color: [f32; 3],
    metallic: f32,
    roughness: f32,
    albedo_texture: Option<String>,
    #[serde(skip)]
    albedo_texture_data: Option<Vec<u8>>, // Raw texture bytes for deployment
}
impl Default for Material {
    fn default() -> Self {
        Self {
            name: "Default".into(),
            albedo_color: [0.8, 0.8, 0.8],
            metallic: 0.0,
            roughness: 0.5,
            albedo_texture: None,
            albedo_texture_data: None,
        }
    }
}

// ============================================================================
// SCENE DATA
// ============================================================================
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct SceneEntity {
    id: u32,
    name: String,
    position: [f32; 3],
    rotation: [f32; 4],
    scale: [f32; 3],
    entity_type: EntityType,
    material: Material,
    script: Option<String>,
    #[serde(default)]
    collision_enabled: bool,
    #[serde(default = "default_layer")]
    layer: u32,
    #[serde(default)]
    is_static: bool,  // If true, object is not affected by gravity
    #[serde(default)]
    animator_config: Option<stfsc_engine::world::animation::AnimatorConfig>,
    #[serde(default)]
    parent_id: Option<u32>,
    #[serde(default = "default_fov")]
    fov: f32,
    #[serde(skip)]
    deployed: bool,
}

fn default_fov() -> f32 {
    0.785 // PI/4
}

fn default_layer() -> u32 {
    LAYER_DEFAULT
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum EntityType {
    Primitive(PrimitiveType),
    Mesh {
        path: String,
    },
    Vehicle,
    CrowdAgent {
        state: String,
        speed: f32,
    },
    Building {
        height: f32,
    },
    Ground,
    Camera,
    /// Dynamic light source
    Light {
        light_type: LightTypeEditor,
        intensity: f32,
        range: f32,
        color: [f32; 3],
    },
    /// 3D Audio Source
    AudioSource {
        sound_id: String,
        volume: f32,
        looping: bool,
        max_distance: f32,
        #[serde(skip)] // Don't serialize large audio data
        audio_data: Option<Vec<u8>>,
    },
}

/// Light types for editor
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
enum LightTypeEditor {
    Point,
    Spot,
    Directional,
}

/// A named UI layout with a FuckScript alias for menu_load()
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
struct NamedUiLayout {
    /// Display name shown in tab
    pub name: String,
    /// Alias for menu_load("alias") in FuckScript
    pub alias: String,
    /// Layer type determines behavior (pause, input blocking, etc.)
    #[serde(default)]
    pub layer_type: stfsc_engine::ui::UiLayerType,
    /// The actual UI layout
    pub layout: stfsc_engine::ui::UiLayout,
}

impl Default for NamedUiLayout {
    fn default() -> Self {
        Self {
            name: "Main".to_string(),
            alias: "main".to_string(),
            layer_type: stfsc_engine::ui::UiLayerType::InGameOverlay,
            layout: stfsc_engine::ui::UiLayout::default(),
        }
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
struct Scene {
    name: String,
    version: String,
    entities: Vec<SceneEntity>,
    respawn_enabled: bool,
    respawn_y: f32,
    /// Multiple named UI layouts with tabs
    #[serde(default)]
    pub ui_layouts: Vec<NamedUiLayout>,
    /// Legacy single layout (for backwards compatibility on load)
    #[serde(default, skip_serializing)]
    #[allow(dead_code)]
    pub ui_layout: stfsc_engine::ui::UiLayout,
}

impl Scene {
    fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            version: "1.0".into(),
            entities: Vec::new(),
            respawn_enabled: false,
            respawn_y: -20.0,
            ui_layouts: vec![NamedUiLayout::default()],
            ui_layout: stfsc_engine::ui::UiLayout::default(),
        }
    }
    
    /// Get current layout for editing (ensures at least one exists)
    #[allow(dead_code)]
    fn get_or_create_layout(&mut self, idx: usize) -> &mut NamedUiLayout {
        if self.ui_layouts.is_empty() {
            // Migrate legacy ui_layout if present
            if !self.ui_layout.buttons.is_empty() || !self.ui_layout.panels.is_empty() || !self.ui_layout.texts.is_empty() {
                self.ui_layouts.push(NamedUiLayout {
                    name: "Main".to_string(),
                    alias: "main".to_string(),
                    layer_type: stfsc_engine::ui::UiLayerType::InGameOverlay,
                    layout: std::mem::take(&mut self.ui_layout),
                });
            } else {
                self.ui_layouts.push(NamedUiLayout::default());
            }
        }
        let idx = idx.min(self.ui_layouts.len() - 1);
        &mut self.ui_layouts[idx]
    }

    fn create_test_scene() -> Self {
        let mut s = Scene::new("Test");
        s.respawn_enabled = true;
        s.respawn_y = -20.0;
        let m = Material::default();

        // Ground - positioned to match main.rs floor (avoiding z-fighting)
        // main.rs floor is at Y=-1.1 with scale 200x2x200, top at Y=-0.1
        // We position this slightly lower so they don't z-fight
        s.entities.push(SceneEntity {
            id: 1,
            name: "Ground".into(),
            position: [0.0, -1.1, 0.0],  // Lowered to match main.rs floor exactly (top at -0.1)
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [200.0, 2.0, 200.0],  // Match size with main.rs floor (200x200)
            entity_type: EntityType::Ground,
            material: m.clone(),
            script: None,
            collision_enabled: true,
            layer: LAYER_ENVIRONMENT,
            is_static: false,
            animator_config: None,
            parent_id: None,
            fov: 0.785,
            deployed: false,
        });
        // Physics Cube
        s.entities.push(SceneEntity {
            id: 2,
            name: "Physics Cube".into(),
            position: [0.0, 10.0, -5.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            entity_type: EntityType::Primitive(PrimitiveType::Cube),
            material: Material {
                albedo_color: [1.0, 0.5, 0.2],
                ..m.clone()
            },
            script: None,
            collision_enabled: true,
            layer: LAYER_PROP,
            is_static: false,
            animator_config: None,
            parent_id: None,
            fov: 0.785,
            deployed: false,
        });
        // Vehicle
        s.entities.push(SceneEntity {
            id: 3,
            name: "Vehicle".into(),
            position: [5.0, 2.0, -8.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [2.0, 1.0, 4.0],
            entity_type: EntityType::Vehicle,
            material: m.clone(),
            script: Some("Vehicle".into()),
            collision_enabled: true,
            layer: LAYER_VEHICLE,
            is_static: false,
            animator_config: None,
            parent_id: None,
            fov: 0.785,
            deployed: false,
        });
        // Agents
        for i in 0..8 {
            let (state, speed, color) = if i < 2 {
                ("Fleeing", 8.0, [1.0, 0.2, 0.2])
            } else if i < 4 {
                ("Running", 5.0, [0.2, 1.0, 0.2])
            } else {
                ("Walking", 2.0, [1.0, 1.0, 1.0])
            };
            s.entities.push(SceneEntity {
                id: 100 + i,
                name: format!("Agent {} ({})", i, state),
                position: [(i as f32 - 4.0) * 5.0, 1.0, -15.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [0.5, 1.8, 0.5],
                entity_type: EntityType::CrowdAgent {
                    state: state.into(),
                    speed,
                },
                material: Material {
                    albedo_color: color,
                    ..m.clone()
                },
                script: Some("CrowdAgent".into()),
                collision_enabled: false,
                layer: LAYER_CHARACTER,
                is_static: false,
                animator_config: None,
                parent_id: None,
                fov: 0.785,
                deployed: false,
            });
        }
        // Buildings
        for i in 0..4 {
            let h = 8.0 + (i as f32) * 5.0;
            s.entities.push(SceneEntity {
                id: 200 + i,
                name: format!("Building {}", i + 1),
                position: [25.0 + (i as f32) * 12.0, h / 2.0, -40.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [6.0, h, 6.0],
                entity_type: EntityType::Building { height: h },
                material: Material {
                    albedo_color: [1.0, 0.5, 0.2],
                    ..m.clone()
                },
                script: None,
                collision_enabled: false,
                layer: LAYER_ENVIRONMENT,
                is_static: true,  // Buildings are static by default
                animator_config: None,
                parent_id: None,
                fov: 0.785,
                deployed: false,
            });
        }
        // Player Start - required for correct spawn position
        s.entities.push(SceneEntity {
            id: 0,
            name: "Player Start".into(),
            position: [0.0, 1.7, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [0.5, 0.5, 0.5],
            entity_type: EntityType::Camera,
            material: Material {
                albedo_color: [0.3, 0.3, 1.0],
                ..m
            },
            script: None,
            collision_enabled: false,
            layer: LAYER_DEFAULT,
            is_static: true,
            animator_config: None,
            parent_id: None,
            fov: 0.785,
            deployed: false,
        });
        s
    }
}

enum AppCommand {
    Connect(String),
    Send(SceneUpdate),
}
enum AppEvent {
    Connected,
    ConnectionError(String),
    #[allow(dead_code)]
    SendError(String),
    StatusUpdate(String),
}

// ============================================================================
// PENDING ACTIONS (for unsaved changes dialog)
// ============================================================================
#[derive(Clone)]
enum PendingAction {
    NewScene,
    OpenScene(String),
    LoadTestScene,
    Exit,
}

// ============================================================================
// UNDO/REDO ACTION HISTORY
// ============================================================================
#[derive(Clone, PartialEq)]
enum UiElementHistory {
    Button(stfsc_engine::ui::Button),
    Panel(stfsc_engine::ui::Panel),
    Text(stfsc_engine::ui::Text),
}

#[derive(Clone)]
enum EditorAction {
    /// Entity was added - undo by removing, redo by adding back
    AddEntity { entity: SceneEntity },
    /// Entity was deleted - undo by adding back, redo by removing
    DeleteEntity { entity: SceneEntity },
    /// Entity was modified - stores before and after states
    ModifyEntity { 
        id: u32,
        before: SceneEntity,
        after: SceneEntity,
    },
    /// UI layout was modified (e.g. template applied)
    ModifyUiLayout {
        layout_idx: usize,
        before: NamedUiLayout,
        after: NamedUiLayout,
    },
    /// UI element was added
    AddUiElement {
        layout_idx: usize,
        element_type: String, // "Button", "Panel", "Text"
        element: UiElementHistory,
    },
    /// UI element was deleted
    DeleteUiElement {
        layout_idx: usize,
        element_type: String,
        element_idx: usize,
        element: UiElementHistory,
    },
    /// UI element was modified (e.g. moved)
    ModifyUiElement {
        layout_idx: usize,
        element_type: String,
        element_idx: usize,
        before: UiElementHistory,
        after: UiElementHistory,
    },
}

// ============================================================================
// EDITOR VIEW TABS
// ============================================================================

/// Active editor view tab
#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum EditorView {
    #[default]
    SceneEditor,
    AnimationEditor,
}

/// State for the animation editor
#[derive(Clone, Debug)]
struct AnimationEditorState {
    /// Currently selected animation clip index
    selected_clip_idx: Option<usize>,
    /// Current playback time (seconds)
    current_time: f32,
    /// Is animation playing
    is_playing: bool,
    /// Playback speed multiplier
    playback_speed: f32,
    /// Selected bone indices for keyframe editing
    selected_bones: Vec<usize>,
    /// Selected keyframe indices
    selected_keyframes: Vec<usize>,
    /// Timeline zoom level
    timeline_zoom: f32,
    /// Timeline scroll offset
    timeline_scroll: f32,
    /// Auto-key mode (automatically create keyframes on transform change)
    auto_key: bool,
    /// Last playback update time
    last_update: std::time::Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
enum RenderQuality {
    #[default]
    High, // Solid meshes + textures
    Low,  // Wireframe
}

impl Default for AnimationEditorState {
    fn default() -> Self {
        Self {
            selected_clip_idx: None,
            current_time: 0.0,
            is_playing: false,
            playback_speed: 1.0,
            selected_bones: Vec::new(),
            selected_keyframes: Vec::new(),
            timeline_zoom: 1.0,
            timeline_scroll: 0.0,
            auto_key: false,
            last_update: std::time::Instant::now(),
        }
    }
}

// ============================================================================
// EDITOR
// ============================================================================
struct EditorApp {
    ip: String,
    status: String,
    command_tx: Sender<AppCommand>,
    event_rx: Receiver<AppEvent>,
    is_connected: bool,
    scenes: Vec<String>,
    current_scene: Option<Scene>,
    selected_scene_idx: Option<usize>,
    selected_entity_id: Option<u32>,
    show_new_scene_dialog: bool,
    new_scene_name: String,
    #[allow(dead_code)]
    show_primitives_panel: bool,
    procedural_generation_enabled: bool, // Toggle for Quest procedural world gen

    camera: Camera3D,
    drag_button: Option<egui::PointerButton>,
    last_mouse: egui::Pos2,
    dragging_id: Option<u32>,
    drag_start_entity: Option<SceneEntity>,  // Entity state at drag start for undo
    inspector_start_entity: Option<SceneEntity>,  // Entity state at inspector edit start for undo
    last_selected_id: Option<u32>,  // Track selection changes

    // Scene file management
    current_scene_path: Option<String>,    // Path to currently loaded scene file
    scene_dirty: bool,                      // True if scene has unsaved changes
    recent_scenes: Vec<String>,             // Recently opened scene files
    show_open_scene_dialog: bool,           // Open scene file browser dialog
    show_save_as_dialog: bool,              // Save as dialog
    save_as_path: String,                   // Buffer for save-as filename input
    open_scene_files: Vec<String>,          // Cached list of scene files for browser
    show_unsaved_warning: bool,             // Unsaved changes warning dialog
    pending_action: Option<PendingAction>,  // What to do after unsaved warning

    // Clipboard and History
    clipboard: Option<SceneEntity>,         // Copied entity for paste
    undo_stack: Vec<EditorAction>,          // History of actions for undo
    redo_stack: Vec<EditorAction>,          // Undone actions for redo
    
    // Model import
    show_import_model_dialog: bool,         // Show import model file browser
    import_model_files: Vec<String>,        // Cached list of model files for browser

    // UI Editor
    show_ui_editor: bool,
    selected_ui_element_type: Option<String>, // "Button", "Panel", "Text"
    selected_ui_id: Option<u32>,
    selected_child_idx: Option<usize>, // For child elements within a panel
    selected_layout_idx: usize, // Currently selected UI layout tab
    
    // UI Dragging History
    ui_drag_start_element: Option<UiElementHistory>, // State before drag started
    ui_inspector_start_element: Option<UiElementHistory>, // State before inspector edit
    last_selected_ui_type: Option<String>,
    last_selected_ui_id: Option<u32>,
    
    // Editor view tabs
    active_view: EditorView,
    animation_editor_state: AnimationEditorState,

    // Rendering Quality
    render_quality: RenderQuality,
    // Local assets for editor rendering
    model_cache: HashMap<String, Arc<ModelScene>>,
    egui_textures: HashMap<String, egui::TextureHandle>,
    
    // Interaction
    gizmo_axis: Option<usize>, // 0=X, 1=Y, 2=Z
}

impl EditorApp {
    fn new() -> Self {
        let (command_tx, command_rx) = channel::<AppCommand>();
        let (event_tx, event_rx) = channel::<AppEvent>();

        thread::spawn(move || {
            let mut stream: Option<TcpStream> = None;
            while let Ok(cmd) = command_rx.recv() {
                match cmd {
                    AppCommand::Connect(ip) => {
                        let _ = event_tx.send(AppEvent::StatusUpdate(format!("Connecting...")));
                        let addr = if ip.contains(':') {
                            ip.clone()
                        } else {
                            format!("{}:8080", ip)
                        };
                        match TcpStream::connect_timeout(
                            &addr.parse().unwrap(),
                            std::time::Duration::from_secs(5),
                        ) {
                            Ok(s) => {
                                stream = Some(s);
                                let _ = event_tx.send(AppEvent::Connected);
                            }
                            Err(e) => {
                                let _ = event_tx.send(AppEvent::ConnectionError(e.to_string()));
                            }
                        }
                    }
                     AppCommand::Send(update) => {
                        if let Some(s) = &mut stream {
                            let bytes = bincode::serialize(&update).unwrap();
                            let len = bytes.len() as u32;
                            println!("EDITOR: Sending SceneUpdate (len: {})", len);
                            if s.write_all(&len.to_le_bytes()).is_ok() && s.write_all(&bytes).is_ok() {
                                // Success
                                let _ = s.flush();
                            } else {
                                println!("EDITOR ERROR: Failed to write to stream. Disconnecting.");
                                stream = None;
                                let _ = event_tx.send(AppEvent::ConnectionError("Connection lost".into()));
                            }
                        } else {
                            println!("EDITOR WARN: Not connected, cannot send command.");
                        }
                    }
                }
            }
        });

        // Create assets/textures if it doesn't exist
        let _ = std::fs::create_dir_all("assets/textures");

        Self {
            ip: "127.0.0.1:8080".into(),
            status: "Disconnected".into(),
            command_tx,
            event_rx,
            is_connected: false,
            scenes: vec!["Test".into()],
            current_scene: Some(Scene::create_test_scene()),
            selected_scene_idx: Some(0),
            selected_entity_id: None,
            show_new_scene_dialog: false,
            new_scene_name: String::new(),
            show_primitives_panel: false,
            procedural_generation_enabled: false, // Off by default
            camera: Camera3D::new(),
            drag_button: None,
            last_mouse: egui::pos2(0.0, 0.0),
            dragging_id: None,
            drag_start_entity: None,
            inspector_start_entity: None,
            last_selected_id: None,
            // Scene file management
            current_scene_path: None,
            scene_dirty: false,
            recent_scenes: Vec::new(),
            show_open_scene_dialog: false,
            show_save_as_dialog: false,
            save_as_path: String::new(),
            open_scene_files: Vec::new(),
            show_unsaved_warning: false,
            pending_action: None,
            // Clipboard and History
            clipboard: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            // Model import
            show_import_model_dialog: false,
            import_model_files: Vec::new(),
            show_ui_editor: false,
            selected_ui_element_type: None,
            selected_ui_id: None,
            selected_child_idx: None,
            selected_layout_idx: 0,
            ui_drag_start_element: None,
            ui_inspector_start_element: None,
            last_selected_ui_type: None,
            last_selected_ui_id: None,
            // Editor view tabs
            active_view: EditorView::SceneEditor,
            animation_editor_state: AnimationEditorState::default(),
            // Rendering Quality
            render_quality: RenderQuality::High,
            model_cache: {
                let mut cache = HashMap::new();
                for ptype in PrimitiveType::all() {
                    let mesh = stfsc_engine::world::create_primitive(match ptype {
                        PrimitiveType::Cube => 0,
                        PrimitiveType::Sphere => 1,
                        PrimitiveType::Cylinder => 2,
                        PrimitiveType::Plane => 3,
                        PrimitiveType::Capsule => 4,
                        PrimitiveType::Cone => 5,
                    });
                    let mut scene = ModelScene::new();
                    scene.meshes.push(mesh);
                    cache.insert(format!("primitive://{}", ptype.name().to_lowercase()), Arc::new(scene));
                }
                cache
            },
            egui_textures: HashMap::new(),
            gizmo_axis: None,
        }
    }

    fn deploy_entity_to_quest(&self, entity: &SceneEntity) {
        match &entity.entity_type {
            EntityType::Camera => {
                let _ = self
                    .command_tx
                    .send(AppCommand::Send(SceneUpdate::SetPlayerStart {
                        position: entity.position,
                        rotation: entity.rotation,
                    }));
            }
            EntityType::Light {
                light_type,
                intensity,
                range,
                color,
            } => {
                // Convert editor light type to engine light type
                let engine_light_type = match light_type {
                    LightTypeEditor::Point => LightType::Point,
                    LightTypeEditor::Spot => LightType::Spot,
                    LightTypeEditor::Directional => LightType::Directional,
                };

                // Add tiny random offset to prevent singularities/Z-fighting if multiple lights are spawned at exact grid points
                // or if camera is exactly at light position
                let epsilon_x = (rand::random::<f32>() - 0.5) * 0.001;
                let epsilon_y = (rand::random::<f32>() - 0.5) * 0.001;
                let epsilon_z = (rand::random::<f32>() - 0.5) * 0.001;
                let pos = (glam::Vec3::from(entity.position) + glam::Vec3::new(epsilon_x, epsilon_y, epsilon_z)).to_array();

                let _ = self
                    .command_tx
                    .send(AppCommand::Send(SceneUpdate::SpawnLight {
                        id: entity.id,
                        light_type: engine_light_type,
                        position: pos,
                        direction: [0.0, -1.0, 0.0], // Default down direction
                        color: *color,
                        intensity: *intensity,
                        range: *range,
                        inner_cone: 0.4,
                        outer_cone: 0.6,
                    }));
            }
            EntityType::AudioSource {
                sound_id,
                volume,
                looping,
                max_distance,
                audio_data,
            } => {
                // First, upload the audio data if we have it
                if let Some(data) = audio_data {
                    let _ = self
                        .command_tx
                        .send(AppCommand::Send(SceneUpdate::UploadSound {
                            id: sound_id.clone(),
                            data: data.clone(),
                        }));
                }
                // Then spawn the sound source
                let _ = self
                    .command_tx
                    .send(AppCommand::Send(SceneUpdate::SpawnSound {
                        id: entity.id,
                        sound_id: sound_id.clone(),
                        position: entity.position,
                        volume: *volume,
                        looping: *looping,
                        max_distance: *max_distance,
                    }));
            }
            _ => {
                let color = entity.material.albedo_color;
                
                // Upload texture if present (for any entity type including Ground)
                if let (Some(texture_id), Some(texture_data)) = (&entity.material.albedo_texture, &entity.material.albedo_texture_data) {
                    let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::UploadTexture {
                        id: texture_id.clone(),
                        data: texture_data.clone(),
                    }));
                }
                
                // Special handling for Ground entities - send SpawnGroundPlane for shadow frustum sizing
                if matches!(&entity.entity_type, EntityType::Ground) {
                    let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::SpawnGroundPlane {
                        id: entity.id,
                        primitive: 0, // Cube
                        position: entity.position,
                        scale: entity.scale,
                        color,
                        half_extents: [entity.scale[0] / 2.0, entity.scale[2] / 2.0], // Half of X and Z scale
                        albedo_texture: entity.material.albedo_texture.clone(),
                        collision_enabled: entity.collision_enabled,
                        layer: entity.layer,
                    }));
                } else if let EntityType::Mesh { path } = &entity.entity_type {
                    // Load mesh from file and send via SpawnFbxMesh
                    if let Ok(data) = std::fs::read(path) {
                        self.deploy_mesh_entity(entity, &data);
                    } else {
                        println!("EDITOR WARN: Failed to load mesh file: {}", path);
                    }
                } else {
                    let primitive = match &entity.entity_type {
                        EntityType::Primitive(p) => match p {
                            PrimitiveType::Cube => 0,
                            PrimitiveType::Sphere => 1,
                            PrimitiveType::Cylinder => 2,
                            PrimitiveType::Plane => 3,
                            PrimitiveType::Capsule => 4,
                            PrimitiveType::Cone => 5,
                        },
                        EntityType::Vehicle => 0,         // Use Cube for vehicle body
                        EntityType::Building { .. } => 0, // Use Cube for buildings
                        EntityType::CrowdAgent { .. } => 4, // Use Capsule for agents
                        _ => 0,                           // Default to Cube
                    };

                    let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::Spawn {
                        id: entity.id,
                        primitive,
                        position: entity.position,
                        rotation: entity.rotation,
                        scale: entity.scale,
                        color,
                        albedo_texture: entity.material.albedo_texture.clone(),
                        collision_enabled: entity.collision_enabled,
                        layer: entity.layer,
                        is_static: entity.is_static,
                    }));
                }
            }
        }

        // Deploy script if assigned
        if let Some(script_name) = &entity.script {
            if !script_name.is_empty() {
                let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::AttachScript {
                    id: entity.id,
                    name: script_name.clone(),
                }));
            }
        }
    }

    fn deploy_all_to_quest(&self) {
        if let Some(scene) = &self.current_scene {
            // Send global scene settings first
            let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::SetRespawnSettings {
                enabled: scene.respawn_enabled,
                y_threshold: scene.respawn_y,
            }));
            
            // Deploy all entities
            for entity in &scene.entities {
                self.deploy_entity_to_quest(entity);
            }
            
            // Deploy all UI layouts
            for nl in &scene.ui_layouts {
                let layer = match nl.layer_type {
                    stfsc_engine::ui::UiLayerType::PauseOverlay => stfsc_engine::ui::UiLayer::PauseMenu,
                    stfsc_engine::ui::UiLayerType::MainMenu => stfsc_engine::ui::UiLayer::MainMenu,
                    stfsc_engine::ui::UiLayerType::InGameOverlay => stfsc_engine::ui::UiLayer::Hud,
                    _ => stfsc_engine::ui::UiLayer::Custom(nl.alias.clone()),
                };
                // Sync layer_type to inner layout
                let mut layout_to_send = nl.layout.clone();
                layout_to_send.layer_type = nl.layer_type;
                let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::SetUiLayer {
                    layer,
                    layout: layout_to_send,
                }));
            }
        }
    }

    fn add_primitive(&mut self, ptype: PrimitiveType) {
        if let Some(scene) = &mut self.current_scene {
            let id = scene.entities.iter().map(|e| e.id).max().unwrap_or(0) + 1;
            let new_entity = SceneEntity {
                id,
                name: format!("{} {}", ptype.name(), id),
                position: [0.0, 2.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0, 1.0, 1.0],
                entity_type: EntityType::Primitive(ptype),
                material: Material::default(),
                script: None,
                collision_enabled: true,
                layer: LAYER_PROP, // Default new primitives to Props
                is_static: false,
                animator_config: None,
                parent_id: None,
                fov: 0.785,
                deployed: false,
            };
            // Push to undo stack
            self.undo_stack.push(EditorAction::AddEntity { entity: new_entity.clone() });
            self.redo_stack.clear();
            
            scene.entities.push(new_entity);
            self.selected_entity_id = Some(id);
            self.scene_dirty = true;
        }
    }

    // ========================================================================
    // SCENE FILE MANAGEMENT
    // ========================================================================
    
    /// Mark the current scene as having unsaved changes
    #[allow(dead_code)]
    fn mark_scene_dirty(&mut self) {
        self.scene_dirty = true;
    }

    /// Save the current scene to the specified path
    fn save_scene_to_path(&mut self, path: &str) -> Result<(), String> {
        if let Some(scene) = &self.current_scene {
            // Ensure scenes directory exists
            std::fs::create_dir_all("scenes").map_err(|e| e.to_string())?;
            
            let json = serde_json::to_string_pretty(scene)
                .map_err(|e| format!("Serialization error: {}", e))?;
            
            std::fs::write(path, json)
                .map_err(|e| format!("Write error: {}", e))?;
            
            self.current_scene_path = Some(path.to_string());
            self.scene_dirty = false;
            self.add_to_recent(path);
            self.status = format!("Saved: {}", path);
            Ok(())
        } else {
            Err("No scene to save".into())
        }
    }

    /// Load a scene from the specified file path
    fn load_scene_from_path(&mut self, ctx: &egui::Context, path: &str) -> Result<(), String> {
        let contents = std::fs::read_to_string(path)
            .map_err(|e| format!("Read error: {}", e))?;
        
        let mut scene: Scene = serde_json::from_str(&contents)
            .map_err(|e| format!("Parse error: {}", e))?;
        
        // Reload texture bytes from disk for each entity with a texture path
        for entity in &mut scene.entities {
            if let Some(texture_path) = entity.material.albedo_texture.clone() {
                // Try multiple paths:
                // 1. As stored (absolute or relative to current project root)
                // 2. Relative to the scene file being loaded
                // 3. In the project's assets/textures folder
                let paths_to_try = vec![
                    texture_path.clone(),
                    format!("{}/{}", std::path::Path::new(path).parent().unwrap_or(std::path::Path::new(".")).display(), texture_path),
                    format!("assets/textures/{}", std::path::Path::new(&texture_path).file_name().unwrap_or_default().to_string_lossy()),
                ];
                
                let mut loaded_path = None;
                for p in paths_to_try {
                    if let Ok(bytes) = std::fs::read(&p) {
                        entity.material.albedo_texture_data = Some(bytes);
                        loaded_path = Some(p);
                        break;
                    }
                }
                
                if let Some(p) = loaded_path {
                    // Update to the working path for future sessions
                entity.material.albedo_texture = Some(p);
                
                // Pre-cache texture for editor software renderer
                if let Some(data) = &entity.material.albedo_texture_data {
                    let tex_name = std::path::Path::new(entity.material.albedo_texture.as_ref().unwrap())
                        .file_name().unwrap_or_default().to_string_lossy().to_string();
                    self.load_egui_texture(ctx, &tex_name, data);
                }
            } else {
                    eprintln!("Warning: Could not load texture at '{}'", texture_path);
                }
            }
            
            // Cache model for editor rendering
            if let EntityType::Mesh { path } = &entity.entity_type {
                let _ = self.cache_model_at_path(ctx, path);
            }
        }
        
        self.current_scene = Some(scene);
        self.current_scene_path = Some(path.to_string());
        self.scene_dirty = false;
        self.selected_entity_id = None;
        // Clear undo/redo on scene load
        self.undo_stack.clear();
        self.redo_stack.clear();
        self.add_to_recent(path);
        self.status = format!("Loaded: {}", path);
        Ok(())
    }

    /// Scan the scenes directory for .json files
    fn scan_scene_files(&mut self) {
        self.open_scene_files.clear();
        if let Ok(entries) = std::fs::read_dir("scenes") {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "json") {
                    if let Some(path_str) = path.to_str() {
                        self.open_scene_files.push(path_str.to_string());
                    }
                }
            }
        }
        self.open_scene_files.sort();
    }

    /// Scan for 3D model files (OBJ, FBX) in common directories
    fn scan_model_files() -> Vec<String> {
        let mut files = Vec::new();
        let search_dirs = ["assets/models", "models", "assets", "."];
        
        for dir in search_dirs {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        if let Some(ext) = path.extension() {
                            let ext_lower = ext.to_string_lossy().to_lowercase();
                            if ext_lower == "obj" || ext_lower == "fbx" || ext_lower == "glb" || ext_lower == "gltf" {
                                if let Some(path_str) = path.to_str() {
                                    files.push(path_str.to_string());
                                }
                            }
                        }
                    }
                }
            }
        }
        
        files.sort();
        files
    }
    
    /// Cache a 3D model and its textures for editor rendering
    fn cache_model_at_path(&mut self, ctx: &egui::Context, path: &str) -> Result<Arc<ModelScene>, String> {
        if let Some(cached) = self.model_cache.get(path) {
            return Ok(cached.clone());
        }

        let data = std::fs::read(path).map_err(|e| format!("Failed to read file {}: {}", path, e))?;
        let ext = std::path::Path::new(path)
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();

        let model_scene = if ext == "glb" || ext == "gltf" {
            stfsc_engine::world::gltf_loader::load_gltf_with_animations(&data)
                .map_err(|e| format!("glTF Error: {}", e))?
        } else if ext == "obj" {
            stfsc_engine::world::fbx_loader::ModelScene::from_obj_bytes(&data)
                .map_err(|e| format!("OBJ Error: {}", e))?
        } else {
            return Err(format!("Unsupported model format: .{}", ext));
        };

        // Load textures into egui
    for tex in &model_scene.textures {
        self.load_egui_texture(ctx, &tex.name, &tex.data);
    }

        let arc_model = Arc::new(model_scene);
        self.model_cache.insert(path.to_string(), arc_model.clone());
        Ok(arc_model)
    }

    /// Import a 3D model from file path and add to scene
    /// Import a 3D model (glTF/OBJ/FBX) and add it as a Mesh entity
    fn import_model_from_path(&mut self, ctx: &egui::Context, path: &str) -> Result<(), String> {
        // Read the file
        let data = std::fs::read(path).map_err(|e| format!("Failed to read file: {}", e))?;
        
        // Get filename for entity name
        let filename = std::path::Path::new(path)
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("Model")
            .to_string();
        
        // Detect file extension
        let ext = std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();
        
        let id = if let Some(scene) = &self.current_scene {
            scene.entities.iter().map(|e| e.id).max().unwrap_or(0) + 1
        } else {
            return Err("No scene loaded".into());
        };
        
        // Check if this is a glTF file with animations
        let (animator_config, animation_info) = if ext == "glb" || ext == "gltf" || ext == "obj" {
            match self.cache_model_at_path(ctx, path) {
                Ok(model_scene) => {
                    if !model_scene.animations.is_empty() {
                        // Create AnimatorConfig from animations
                        let mut config = stfsc_engine::world::animation::AnimatorConfig::new();
                        
                        for (i, anim) in model_scene.animations.iter().enumerate() {
                            config.add_state(&anim.name, i);
                            config.clip_names.push(anim.name.clone());
                            config.clip_durations.push(anim.duration);
                        }
                        
                        // Populate bone metadata if skeleton exists
                        if let Some(skeleton) = &model_scene.skeleton {
                            for bone in &skeleton.bones {
                                config.bone_names.push(bone.name.clone());
                                config.bone_parents.push(bone.parent_index);
                                config.bone_transforms.push(bone.local_transform);
                            }
                        }
                        
                        // Set first animation as default
                        if let Some(first_anim) = model_scene.animations.first() {
                            config.default_state = first_anim.name.clone();
                        }
                        
                        let info = format!("({} animations)", model_scene.animations.len());
                        (Some(config), info)
                    } else {
                        (None, String::new())
                    }
                }
                Err(e) => {
                    log::warn!("Could not cache model: {}", e);
                    (None, String::new())
                }
            }
        } else {
            (None, String::new())
        };
        
        let new_entity = SceneEntity {
            id,
            name: format!("{}{}", filename, if !animation_info.is_empty() { format!(" {}", animation_info) } else { String::new() }),
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            entity_type: EntityType::Mesh { path: path.to_string() },
            material: Material::default(),
            script: None,
            collision_enabled: false,
            layer: LAYER_PROP,
            is_static: animator_config.is_none(),  // Animated models are not static
            animator_config,
            parent_id: None,
            fov: 0.785,
            deployed: false,
        };
        
        // If connected, deploy immediately (before mutating scene)
        if self.is_connected {
            let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::SpawnFbxMesh {
                id: new_entity.id,
                mesh_data: data,
                position: new_entity.position,
                rotation: new_entity.rotation,
                scale: new_entity.scale,
                albedo_texture: None,  // No texture on initial import
                collision_enabled: new_entity.collision_enabled,
                layer: new_entity.layer,
                is_static: new_entity.is_static,
            }));
            
            // Also send AnimatorConfig if present
            if let Some(ref anim_config) = new_entity.animator_config {
                let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::AttachAnimator {
                    id: new_entity.id,
                    config: anim_config.clone(),
                }));
            }
        }
        
        // Now add to scene
        if let Some(scene) = &mut self.current_scene {
            scene.entities.push(new_entity);
        }
        
        self.selected_entity_id = Some(id);
        self.scene_dirty = true;
        self.status = format!("Imported: {}{}", filename, animation_info);
        
        Ok(())
    }
    
    /// Deploy a mesh entity to the Quest via SpawnGltfMesh or SpawnFbxMesh
    fn deploy_mesh_entity(&self, entity: &SceneEntity, mesh_data: &[u8]) {
        // Upload texture if present (mainly for OBJ/FBX fallback)
        if let (Some(texture_id), Some(texture_data)) = (&entity.material.albedo_texture, &entity.material.albedo_texture_data) {
            let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::UploadTexture {
                id: texture_id.clone(),
                data: texture_data.clone(),
            }));
        }
        
        let is_gltf = if let EntityType::Mesh { path } = &entity.entity_type {
            let path_lower = path.to_lowercase();
            path_lower.ends_with(".glb") || path_lower.ends_with(".gltf")
        } else {
            false
        };

        if is_gltf {
            let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::SpawnGltfMesh {
                id: entity.id,
                mesh_data: mesh_data.to_vec(),
                position: entity.position,
                rotation: entity.rotation,
                scale: entity.scale,
                collision_enabled: entity.collision_enabled,
                layer: entity.layer,
                is_static: entity.is_static,
            }));
        } else {
            let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::SpawnFbxMesh {
                id: entity.id,
                mesh_data: mesh_data.to_vec(),
                position: entity.position,
                rotation: entity.rotation,
                scale: entity.scale,
                albedo_texture: entity.material.albedo_texture.clone(),
                collision_enabled: entity.collision_enabled,
                layer: entity.layer,
                is_static: entity.is_static,
            }));
        }
    }

    /// Add a path to the recent scenes list (max 5 entries)
    fn add_to_recent(&mut self, path: &str) {
        // Remove if already present
        self.recent_scenes.retain(|p| p != path);
        // Add to front
        self.recent_scenes.insert(0, path.to_string());
        // Keep max 5
        self.recent_scenes.truncate(5);
    }

    /// Check for unsaved changes and prompt user if needed
    /// Returns true if it's safe to proceed, false if action was deferred
    fn check_unsaved_and_set_action(&mut self, action: PendingAction) -> bool {
        if self.scene_dirty {
            self.pending_action = Some(action);
            self.show_unsaved_warning = true;
            false
        } else {
            true
        }
    }

    /// Decode and load a texture into egui's texture manager
    fn load_egui_texture(&mut self, ctx: &egui::Context, name: &str, data: &[u8]) {
        if self.egui_textures.contains_key(name) {
            return;
        }

        if let Ok(img) = image::load_from_memory(data) {
            let rgba = img.to_rgba8();
            let pixels = rgba.as_flat_samples();
            let color_img = egui::ColorImage::from_rgba_unmultiplied(
                [img.width() as usize, img.height() as usize],
                pixels.as_slice(),
            );
            let handle = ctx.load_texture(
                name,
                color_img,
                egui::TextureOptions::LINEAR,
            );
            self.egui_textures.insert(name.to_string(), handle);
        }
    }

    /// Execute the pending action after user confirms in unsaved dialog
    fn execute_pending_action(&mut self, ctx: &egui::Context) {
        if let Some(action) = self.pending_action.take() {
            match action {
                PendingAction::NewScene => {
                    self.show_new_scene_dialog = true;
                    self.new_scene_name.clear();
                }
                PendingAction::OpenScene(path) => {
                    if let Err(e) = self.load_scene_from_path(ctx, &path) {
                        self.status = format!("Error: {}", e);
                    }
                }
                PendingAction::LoadTestScene => {
                    self.current_scene = Some(Scene::create_test_scene());
                    self.current_scene_path = None;
                    self.scene_dirty = false;
                    self.status = "Loaded Test Engine Scene".into();
                }
                PendingAction::Exit => {
                    std::process::exit(0);
                }
            }
        }
        self.show_unsaved_warning = false;
    }

    /// Get the window title with dirty indicator
    #[allow(dead_code)]
    fn get_window_title(&self) -> String {
        let scene_name = self.current_scene.as_ref()
            .map(|s| s.name.as_str())
            .unwrap_or("Untitled");
        let dirty = if self.scene_dirty { "*" } else { "" };
        format!("STFSC Editor - 556 Engine - [{}]{}", scene_name, dirty)
    }

    // ========================================================================
    // CLIPBOARD & UNDO/REDO
    // ========================================================================
    
    /// Copy selected entity to clipboard
    fn copy_selected(&mut self) {
        if let Some(id) = self.selected_entity_id {
            if let Some(scene) = &self.current_scene {
                if let Some(entity) = scene.entities.iter().find(|e| e.id == id) {
                    self.clipboard = Some(entity.clone());
                    self.status = format!("Copied: {}", entity.name);
                }
            }
        }
    }

    /// Paste entity from clipboard with new ID and offset position
    fn paste_from_clipboard(&mut self) {
        if let Some(copied) = self.clipboard.clone() {
            if let Some(scene) = &mut self.current_scene {
                let new_id = scene.entities.iter().map(|e| e.id).max().unwrap_or(0) + 1;
                let mut new_entity = copied.clone();
                new_entity.id = new_id;
                new_entity.position[0] += 2.0; // Offset pasted entity
                scene.entities.push(new_entity);
                self.selected_entity_id = Some(new_id);
                self.scene_dirty = true;
            }
        }
    }

    /// Compute character-space and skinning matrices for an entity's current animation state
    fn compute_animated_pose(
        entity: &SceneEntity,
        model_cache: &HashMap<String, Arc<ModelScene>>,
        anim_state: &AnimationEditorState,
    ) -> Option<(Vec<Mat4>, Vec<Mat4>)> {
        let anim_config = entity.animator_config.as_ref()?;
        
        let clip_idx = anim_state.selected_clip_idx?;
        let clip_name = anim_config.clip_names.get(clip_idx)?;
        
        if let EntityType::Mesh { path } = &entity.entity_type {
            if let Some(model) = model_cache.get(path) {
                if let Some(clip) = model.animations.iter().find(|a| a.name == *clip_name) {
                    if let Some(skeleton) = &model.skeleton {
                        let time = anim_state.current_time;
                        let local_transforms = clip.sample(time, skeleton);
                        
                        let bone_count = skeleton.bones.len();
                        let mut global_transforms = vec![Mat4::IDENTITY; bone_count];
                        let mut skinning_matrices = vec![Mat4::IDENTITY; bone_count];

                        for i in 0..bone_count {
                            let bone = &skeleton.bones[i];
                            let local = local_transforms.get(i).copied().unwrap_or(bone.local_transform);

                            global_transforms[i] = if let Some(parent_idx) = bone.parent_index {
                                if parent_idx < bone_count {
                                    global_transforms[parent_idx] * local
                                } else {
                                    local
                                }
                            } else {
                                local
                            };

                            skinning_matrices[i] = global_transforms[i] * bone.inverse_bind_matrix;
                        }
                        
                        return Some((global_transforms, skinning_matrices));
                    }
                }
            }
        }
        None
    }
    /// Push action to undo stack (clears redo stack)
    #[allow(dead_code)]
    fn push_undo_action(&mut self, action: EditorAction) {
        self.undo_stack.push(action);
        self.redo_stack.clear();
        // Limit history to 50 actions
        if self.undo_stack.len() > 50 {
            self.undo_stack.remove(0);
        }
    }

    /// Undo the last action
    fn undo(&mut self) {
        if let Some(action) = self.undo_stack.pop() {
            match &action {
                EditorAction::AddEntity { entity } => {
                    // Undo add = delete
                    if let Some(scene) = &mut self.current_scene {
                        scene.entities.retain(|e| e.id != entity.id);
                        if self.selected_entity_id == Some(entity.id) {
                            self.selected_entity_id = None;
                        }
                    }
                    self.status = format!("Undo: Removed {}", entity.name);
                }
                EditorAction::DeleteEntity { entity } => {
                    // Undo delete = add back
                    if let Some(scene) = &mut self.current_scene {
                        scene.entities.push(entity.clone());
                        self.selected_entity_id = Some(entity.id);
                    }
                    self.status = format!("Undo: Restored {}", entity.name);
                }
                EditorAction::ModifyEntity { id, before, .. } => {
                    // Undo modify = restore before state
                    if let Some(scene) = &mut self.current_scene {
                        if let Some(e) = scene.entities.iter_mut().find(|e| e.id == *id) {
                            *e = before.clone();
                        }
                    }
                    self.status = format!("Undo: Restored entity #{}", id);
                }
                EditorAction::ModifyUiLayout { layout_idx, before, .. } => {
                    if let Some(scene) = &mut self.current_scene {
                        if let Some(layout) = scene.ui_layouts.get_mut(*layout_idx) {
                            *layout = before.clone();
                        }
                    }
                    self.status = format!("Undo: Restored UI Layout #{}", layout_idx);
                }
                EditorAction::AddUiElement { layout_idx, element_type, element: _ } => {
                    // Undo add = remove
                    if let Some(scene) = &mut self.current_scene {
                        if let Some(nl) = scene.ui_layouts.get_mut(*layout_idx) {
                            match element_type.as_str() {
                                "Button" => { nl.layout.buttons.pop(); }
                                "Panel" => { nl.layout.panels.pop(); }
                                "Text" => { nl.layout.texts.pop(); }
                                _ => {}
                            }
                        }
                    }
                    self.status = format!("Undo: Removed UI {}", element_type);
                }
                EditorAction::DeleteUiElement { layout_idx, element_type, element_idx, element } => {
                    // Undo delete = insert back
                    if let Some(scene) = &mut self.current_scene {
                        if let Some(nl) = scene.ui_layouts.get_mut(*layout_idx) {
                            match element {
                                UiElementHistory::Button(b) => nl.layout.buttons.insert(*element_idx, b.clone()),
                                UiElementHistory::Panel(p) => nl.layout.panels.insert(*element_idx, p.clone()),
                                UiElementHistory::Text(t) => nl.layout.texts.insert(*element_idx, t.clone()),
                            }
                        }
                    }
                    self.status = format!("Undo: Restored UI {}", element_type);
                }
                EditorAction::ModifyUiElement { layout_idx, element_type, element_idx, before, .. } => {
                    if let Some(scene) = &mut self.current_scene {
                        if let Some(nl) = scene.ui_layouts.get_mut(*layout_idx) {
                            match before {
                                UiElementHistory::Button(b) => { if let Some(target) = nl.layout.buttons.get_mut(*element_idx) { *target = b.clone(); } }
                                UiElementHistory::Panel(p) => { if let Some(target) = nl.layout.panels.get_mut(*element_idx) { *target = p.clone(); } }
                                UiElementHistory::Text(t) => { if let Some(target) = nl.layout.texts.get_mut(*element_idx) { *target = t.clone(); } }
                            }
                        }
                    }
                    self.status = format!("Undo: Restored UI {}", element_type);
                }
            }
            self.redo_stack.push(action);
            self.scene_dirty = true;
        }
    }

    /// Redo the last undone action
    fn redo(&mut self) {
        if let Some(action) = self.redo_stack.pop() {
            match &action {
                EditorAction::AddEntity { entity } => {
                    // Redo add = add again
                    if let Some(scene) = &mut self.current_scene {
                        scene.entities.push(entity.clone());
                        self.selected_entity_id = Some(entity.id);
                    }
                    self.status = format!("Redo: Added {}", entity.name);
                }
                EditorAction::DeleteEntity { entity } => {
                    // Redo delete = delete again
                    if let Some(scene) = &mut self.current_scene {
                        scene.entities.retain(|e| e.id != entity.id);
                        if self.selected_entity_id == Some(entity.id) {
                            self.selected_entity_id = None;
                        }
                    }
                    self.status = format!("Redo: Deleted {}", entity.name);
                }
                EditorAction::ModifyEntity { id, after, .. } => {
                    // Redo modify = apply after state
                    if let Some(scene) = &mut self.current_scene {
                        if let Some(e) = scene.entities.iter_mut().find(|e| e.id == *id) {
                            *e = after.clone();
                        }
                    }
                    self.status = format!("Redo: Modified entity #{}", id);
                }
                EditorAction::ModifyUiLayout { layout_idx, after, .. } => {
                    if let Some(scene) = &mut self.current_scene {
                        if let Some(layout) = scene.ui_layouts.get_mut(*layout_idx) {
                            *layout = after.clone();
                        }
                    }
                    self.status = format!("Redo: Modified UI Layout #{}", layout_idx);
                }
                EditorAction::AddUiElement { layout_idx, element_type, element } => {
                    if let Some(scene) = &mut self.current_scene {
                        if let Some(nl) = scene.ui_layouts.get_mut(*layout_idx) {
                            match element {
                                UiElementHistory::Button(b) => nl.layout.buttons.push(b.clone()),
                                UiElementHistory::Panel(p) => nl.layout.panels.push(p.clone()),
                                UiElementHistory::Text(t) => nl.layout.texts.push(t.clone()),
                            }
                        }
                    }
                    self.status = format!("Redo: Added UI {}", element_type);
                }
                EditorAction::DeleteUiElement { layout_idx, element_type, element_idx, .. } => {
                    if let Some(scene) = &mut self.current_scene {
                        if let Some(nl) = scene.ui_layouts.get_mut(*layout_idx) {
                            match element_type.as_str() {
                                "Button" => { nl.layout.buttons.remove(*element_idx); }
                                "Panel" | "PanelChild" => { nl.layout.panels.remove(*element_idx); }
                                "Text" => { nl.layout.texts.remove(*element_idx); }
                                _ => {}
                            }
                        }
                    }
                    self.status = format!("Redo: Deleted UI {}", element_type);
                }
                EditorAction::ModifyUiElement { layout_idx, element_type, element_idx, after, .. } => {
                    if let Some(scene) = &mut self.current_scene {
                        if let Some(nl) = scene.ui_layouts.get_mut(*layout_idx) {
                            match after {
                                UiElementHistory::Button(b) => { if let Some(target) = nl.layout.buttons.get_mut(*element_idx) { *target = b.clone(); } }
                                UiElementHistory::Panel(p) => { if let Some(target) = nl.layout.panels.get_mut(*element_idx) { *target = p.clone(); } }
                                UiElementHistory::Text(t) => { if let Some(target) = nl.layout.texts.get_mut(*element_idx) { *target = t.clone(); } }
                            }
                        }
                    }
                    self.status = format!("Redo: Modified UI {}", element_type);
                }
            }
            self.undo_stack.push(action);
            self.scene_dirty = true;
        }
    }

    fn draw_3d_viewport(&mut self, ui: &mut egui::Ui) -> egui::Rect {
        let available = ui.available_size();
        let (response, painter) = ui.allocate_painter(available, egui::Sense::click_and_drag());
        let rect = response.rect;

        // Camera controls - improved drag detection
        if response.drag_started() {
            // Unwrapping 0,0 is bad if we rely on it for raycasting. Use hover_pos or interact_pos from input if None.
            let start_pos = response
                .interact_pointer_pos()
                .or_else(|| ui.input(|i| i.pointer.interact_pos()))
                .or_else(|| ui.input(|i| i.pointer.hover_pos()))
                .unwrap_or(egui::pos2(0.0, 0.0));

            self.last_mouse = start_pos;

            self.drag_button = ui.input(|i| {
                // Allow "M" key + Left Click to simulate Middle Click (Drag Object)
                if i.key_down(egui::Key::M) && i.pointer.button_down(egui::PointerButton::Primary) {
                    Some(egui::PointerButton::Middle)
                } else if i.pointer.button_down(egui::PointerButton::Secondary) {
                    Some(egui::PointerButton::Secondary)
                } else if i.pointer.button_down(egui::PointerButton::Middle) {
                    Some(egui::PointerButton::Middle)
                } else if i.pointer.button_down(egui::PointerButton::Primary) {
                    Some(egui::PointerButton::Primary)
                } else {
                    None
                }
            });

            // Check for entity drag start (Middle Mouse)
            if self.drag_button == Some(egui::PointerButton::Middle) {
                if let Some(scene) = &self.current_scene {
                    for entity in &scene.entities {
                        let pos =
                            Vec3::new(entity.position[0], entity.position[1], entity.position[2]);
                        if let Some(center) = self.camera.project(pos, available) {
                            let c = egui::pos2(rect.min.x + center.x, rect.min.y + center.y);
                            let size = (8.0 + 400.0 / self.camera.distance).clamp(4.0, 25.0);
                            if (start_pos.x - c.x).abs() < size + 5.0
                                && (start_pos.y - c.y).abs() < size + 5.0
                            {
                                self.dragging_id = Some(entity.id);
                                self.drag_start_entity = Some(entity.clone()); // Capture state for undo
                                self.selected_entity_id = Some(entity.id);
                                break;
                            }
                        }
                    }
                }
            }
        }

        if response.drag_released() {
            // Push undo for move if entity position changed
            if let (Some(start_entity), Some(drag_id)) = (self.drag_start_entity.take(), self.dragging_id) {
                if let Some(scene) = &self.current_scene {
                    if let Some(after_entity) = scene.entities.iter().find(|e| e.id == drag_id) {
                        if start_entity.position != after_entity.position {
                            self.undo_stack.push(EditorAction::ModifyEntity {
                                id: drag_id,
                                before: start_entity,
                                after: after_entity.clone(),
                            });
                            self.redo_stack.clear();
                        }
                    }
                }
            }
            self.drag_button = None;
            self.dragging_id = None;
        }

        if response.dragged() {
            let mouse = response
                .interact_pointer_pos()
                .or_else(|| ui.input(|i| i.pointer.interact_pos()))
                .or_else(|| ui.input(|i| i.pointer.hover_pos()))
                .unwrap_or(self.last_mouse);

            let delta = egui::vec2(mouse.x - self.last_mouse.x, mouse.y - self.last_mouse.y);
            self.last_mouse = mouse;

            if let Some(drag_id) = self.dragging_id {
                let rect_size = rect.size();
                let rel_x = mouse.x - rect.min.x;
                let rel_y = mouse.y - rect.min.y;
                let (cam_pos, ray_dir) = self.camera.get_ray(egui::pos2(rel_x, rel_y), rect_size);

                if let Some(scene) = &mut self.current_scene {
                    if let Some(entity) = scene.entities.iter_mut().find(|e| e.id == drag_id) {
                        if ray_dir.y.abs() > 0.001 {
                            let t = (entity.position[1] - cam_pos.y) / ray_dir.y;
                            if t > 0.0 {
                                let new_pos = cam_pos.add(&ray_dir.mul(t));
                                entity.position[0] = new_pos.x;
                                entity.position[2] = new_pos.z;
                                self.scene_dirty = true;
                            }
                        }
                    }
                }
            } else {
                match self.drag_button {
                    Some(egui::PointerButton::Secondary) => {
                        self.camera.yaw -= delta.x * 0.01;
                        self.camera.pitch = (self.camera.pitch - delta.y * 0.01).clamp(-1.5, 1.5);
                    }
                    Some(egui::PointerButton::Middle) | Some(egui::PointerButton::Primary) => {
                        let right = self.camera.get_right();
                        let pan_speed = self.camera.distance * 0.002;
                        self.camera.target.x -= right.x * delta.x * pan_speed;
                        self.camera.target.z -= right.z * delta.x * pan_speed;
                        self.camera.target.y += delta.y * pan_speed;
                    }
                    None | Some(_) => {}
                }
            }
        }

        // Camera focus for Animation Editor
        if self.active_view == EditorView::AnimationEditor {
            if let Some(id) = self.selected_entity_id {
                if let Some(scene) = &self.current_scene {
                   if let Some(entity) = scene.entities.iter().find(|e| e.id == id) {
                       let target = Vec3::new(entity.position[0], entity.position[1] + 1.2, entity.position[2]);
                       // Smooth camera interpolation
                       self.camera.target.x = self.camera.target.x * 0.95 + target.x * 0.05;
                       self.camera.target.y = self.camera.target.y * 0.95 + target.y * 0.05;
                       self.camera.target.z = self.camera.target.z * 0.95 + target.z * 0.05;
                       // Ensure comfortable viewing distance
                       if self.camera.distance > 15.0 {
                           self.camera.distance = self.camera.distance * 0.98 + 15.0 * 0.02;
                       }
                   }
                }
            }
        }

        // Zoom
        let scroll = ui.input(|i| i.scroll_delta.y);
        self.camera.distance = (self.camera.distance - scroll * 1.5).clamp(0.1, 1000.0);

        // Background - Unity-style sky for High mode, dark for Low mode
        if self.render_quality == RenderQuality::High {
            // Unity-style bright sky gradient (bottom = horizon light blue, top = sky blue)
            let sky_horizon = egui::Color32::from_rgb(147, 180, 210);   // Light blue at horizon
            let sky_zenith = egui::Color32::from_rgb(82, 127, 175);     // Deeper blue at top
            
            let steps = 16;
            for i in 0..steps {
                let t = i as f32 / steps as f32;
                let h = rect.height();
                let r = egui::Rect::from_min_max(
                    egui::pos2(rect.min.x, rect.min.y + t * h),
                    egui::pos2(rect.max.x, rect.min.y + (t + 1.0/steps as f32) * h),
                );
                // Inverse t: sky is brighter at top (t=0) in this loop runs top-to-bottom
                let inv_t = 1.0 - t;
                let color = egui::Color32::from_rgb(
                    (sky_zenith.r() as f32 * inv_t + sky_horizon.r() as f32 * t) as u8,
                    (sky_zenith.g() as f32 * inv_t + sky_horizon.g() as f32 * t) as u8,
                    (sky_zenith.b() as f32 * inv_t + sky_horizon.b() as f32 * t) as u8,
                );
                painter.rect_filled(r, 0.0, color);
            }
        } else {
            // Low Quality: Dark professional gradient
            let bg_color_top = egui::Color32::from_rgb(20, 22, 26);
            let bg_color_bottom = egui::Color32::from_rgb(45, 50, 60);
            
            painter.rect_filled(rect, 0.0, bg_color_top);
            
            let steps = 12;
            for i in 0..steps {
                let t = i as f32 / steps as f32;
                let h = rect.height();
                let r = egui::Rect::from_min_max(
                    egui::pos2(rect.min.x, rect.min.y + t * h),
                    egui::pos2(rect.max.x, rect.min.y + (t + 1.0/steps as f32) * h),
                );
                let color = egui::Color32::from_rgb(
                    (bg_color_top.r() as f32 * (1.0 - t) + bg_color_bottom.r() as f32 * t) as u8,
                    (bg_color_top.g() as f32 * (1.0 - t) + bg_color_bottom.g() as f32 * t) as u8,
                    (bg_color_top.b() as f32 * (1.0 - t) + bg_color_bottom.b() as f32 * t) as u8,
                );
                painter.rect_filled(r, 0.0, color);
            }
        }

        // Professional fading grid - colors adjusted for background
        let grid_size = 500.0; // Increased for "infinite" feel
        let grid_step = 10.0;
        let sub_step = 2.0;
        let cam_dist = self.camera.distance;
        // Fade grid faster at distance
        let base_alpha = (200.0 / (cam_dist * 0.1).max(1.0)).clamp(10.0, 150.0);
        
        // Grid colors based on mode (darker for bright sky, lighter for dark)
        let (major_color, minor_color) = if self.render_quality == RenderQuality::High {
            // Dark grid on bright sky (like Unity)
            ([40, 60, 80], [100, 110, 120])
        } else {
            // Light grid on dark background
            ([140, 150, 170], [100, 110, 130])
        };
        
        // Render sub-grid for High Quality
        if self.render_quality == RenderQuality::High {
            for i in -100..=100 {
                let val = i as f32 * sub_step;
                if val.abs() > grid_size { continue; }
                if i % 5 == 0 { continue; } // Skip major lines
                
                // Exponential edge fade for smoother horizon
                let dist_ratio = val.abs() / grid_size;
                let edge_fade = (1.0 - dist_ratio * dist_ratio).max(0.0);
                let alpha = (base_alpha * 0.25 * edge_fade) as u8;
                let stroke = egui::Stroke::new(0.3, egui::Color32::from_rgba_unmultiplied(minor_color[0], minor_color[1], minor_color[2], alpha));
                
                // X
                if let (Some(p1), Some(p2)) = (self.camera.project(Vec3::new(val, 0.0, -grid_size), available), self.camera.project(Vec3::new(val, 0.0, grid_size), available)) {
                    painter.line_segment([egui::pos2(rect.min.x + p1.x, rect.min.y + p1.y), egui::pos2(rect.min.x + p2.x, rect.min.y + p2.y)], stroke);
                }
                // Z
                if let (Some(p1), Some(p2)) = (self.camera.project(Vec3::new(-grid_size, 0.0, val), available), self.camera.project(Vec3::new(grid_size, 0.0, val), available)) {
                    painter.line_segment([egui::pos2(rect.min.x + p1.x, rect.min.y + p1.y), egui::pos2(rect.min.x + p2.x, rect.min.y + p2.y)], stroke);
                }
            }
        }

        for i in -50..=50 {
            let val = i as f32 * grid_step;
            if val.abs() > grid_size { continue; }
            let is_major = i % 5 == 0;
            
            // Dist alpha - fade at edges
            let dist_ratio = val.abs() / grid_size;
            let edge_fade = (1.0 - dist_ratio * dist_ratio).max(0.0);
            let alpha = (base_alpha * edge_fade) as u8;
            
            // X-lines
            if let (Some(p1), Some(p2)) = (
                self.camera.project(Vec3::new(val, 0.0, -grid_size), available),
                self.camera.project(Vec3::new(val, 0.0, grid_size), available),
            ) {
                let stroke = if is_major { 
                    egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(major_color[0], major_color[1], major_color[2], alpha))
                } else {
                    egui::Stroke::new(0.5, egui::Color32::from_rgba_unmultiplied(minor_color[0], minor_color[1], minor_color[2], alpha / 2))
                };
                painter.line_segment(
                    [egui::pos2(rect.min.x + p1.x, rect.min.y + p1.y), egui::pos2(rect.min.x + p2.x, rect.min.y + p2.y)],
                    stroke,
                );
            }
            
            // Z-lines
            if let (Some(p1), Some(p2)) = (
                self.camera.project(Vec3::new(-grid_size, 0.0, val), available),
                self.camera.project(Vec3::new(grid_size, 0.0, val), available),
            ) {
                let stroke = if is_major { 
                    egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(major_color[0], major_color[1], major_color[2], alpha))
                } else {
                    egui::Stroke::new(0.5, egui::Color32::from_rgba_unmultiplied(minor_color[0], minor_color[1], minor_color[2], alpha / 2))
                };
                painter.line_segment(
                    [egui::pos2(rect.min.x + p1.x, rect.min.y + p1.y), egui::pos2(rect.min.x + p2.x, rect.min.y + p2.y)],
                    stroke,
                );
            }
        }

        // Origin axes
        let origin = Vec3::zero();
        if let Some(o) = self.camera.project(origin, available) {
            let o = egui::pos2(rect.min.x + o.x, rect.min.y + o.y);
            for (axis, color) in [
                (Vec3::new(5.0, 0.0, 0.0), egui::Color32::from_rgb(255, 60, 60)),
                (Vec3::new(0.0, 5.0, 0.0), egui::Color32::from_rgb(60, 255, 60)),
                (Vec3::new(0.0, 0.0, 5.0), egui::Color32::from_rgb(60, 60, 255)),
            ] {
                if let Some(p) = self.camera.project(axis, available) {
                    painter.line_segment(
                        [o, egui::pos2(rect.min.x + p.x, rect.min.y + p.y)],
                        egui::Stroke::new(1.5, color),
                    );
                }
            }
        }

        // Draw entities - parallel projection computation
        let mut clicked_entity: Option<u32> = None;
        if let Some(scene) = &self.current_scene {
            let camera = self.camera; // Copy for parallel access
            let selected_id = self.selected_entity_id;
            let model_cache = &self.model_cache;
            let anim_state = &self.animation_editor_state;
            let render_quality = self.render_quality;
            
            // Parallel: Compute all entity render data (projections, colors, etc.)
            let white_pixel_uv = egui::pos2(0.0, 0.0);
            let cam_pos = camera.get_position();
            let cam_pos_g = GVec3::new(cam_pos.x, cam_pos.y, cam_pos.z);
            let cam_fwd = {
                let fwd = camera.get_forward();
                GVec3::new(fwd.x, fwd.y, fwd.z)
            };
            
            // Extract frustum once for culling
            let vp = camera.get_view_proj(available.x / available.y);
            let frustum = Frustum::from_view_proj(vp);

            let mut entity_renders: Vec<_> = scene.entities.par_iter().map(|entity| {
                let pos = Vec3::new(entity.position[0], entity.position[1], entity.position[2]);
                let is_selected = selected_id == Some(entity.id);
                // Ground and Plane entities should be double-sided (no backface culling)
                let is_double_sided = matches!(&entity.entity_type, EntityType::Ground | EntityType::Primitive(PrimitiveType::Plane));
                
                let base_color = [
                    (entity.material.albedo_color[0] * 200.0) as u8,
                    (entity.material.albedo_color[1] * 200.0) as u8,
                    (entity.material.albedo_color[2] * 200.0) as u8,
                ];

                // Standard projection for icons/centers
                let center_proj = camera.project(pos, available)
                    .map(|p| egui::pos2(rect.min.x + p.x, rect.min.y + p.y));
                
                let mut high_res_meshes = Vec::new();

                // 1. Frustum Culling at Entity Level
                // 1. Precise Frustum Culling at Entity Level
                let entity_scale = GVec3::from(entity.scale);
                let entity_pos = GVec3::from(entity.position);
                
                let mesh_path = match &entity.entity_type {
                    EntityType::Mesh { path } => Some(path.clone()),
                    EntityType::Primitive(ptype) => Some(format!("primitive://{}", ptype.name().to_lowercase())),
                    EntityType::Ground => Some("primitive://cube".to_string()),
                    EntityType::Vehicle => Some("primitive://cube".to_string()),
                    EntityType::Building { .. } => Some("primitive://cube".to_string()),
                    EntityType::CrowdAgent { .. } => Some("primitive://capsule".to_string()),
                    _ => None,
                };

                let entity_aabb = if let Some(path) = &mesh_path {
                    if let Some(model) = model_cache.get(path) {
                        // Use accurate mesh AABB if available
                        let mut min = GVec3::splat(f32::MAX);
                        let mut max = GVec3::splat(f32::MIN);
                        for mesh in model.meshes.iter().chain(model.skinned_meshes.iter().map(|s| &s.mesh)) {
                            min = min.min(GVec3::from(mesh.aabb_min));
                            max = max.max(GVec3::from(mesh.aabb_max));
                        }
                        let center = (min + max) * 0.5;
                        let extent = (max - min) * 0.5;
                        let world_center = entity_pos + GQuat::from_array(entity.rotation) * (center * entity_scale);
                        let world_extent = entity_scale * extent * 1.2; // Extra padding for rotation
                        AABB::from_center_extents(world_center, world_extent)
                    } else {
                        AABB::from_center_extents(entity_pos, entity_scale * 1.5)
                    }
                } else {
                    AABB::from_center_extents(entity_pos, entity_scale * 1.5) // Conservative bounds
                };
                let is_visible = frustum.intersects_aabb(&entity_aabb);

                if render_quality == RenderQuality::High && is_visible {
                    // Reuse mesh_path from frustum culling above
                    if let Some(ref path) = mesh_path {
                        if let Some(model) = model_cache.get(path) {
                            let transform = Mat4::from_scale_rotation_translation(
                                entity_scale,
                                GQuat::from_array(entity.rotation),
                                entity_pos,
                            );
                            
                            let pose = Self::compute_animated_pose(entity, model_cache, anim_state);
                            let mat_color = GVec3::new(
                                entity.material.albedo_color[0],
                                entity.material.albedo_color[1],
                                entity.material.albedo_color[2],
                            );
                            
                            // Balanced lighting - slightly angled for depth perception
                            let light_dir = GVec3::new(0.4, 0.8, 0.3).normalize();
                            
                            // Process all meshes (regular and skinned)
                            let all_meshes: Vec<&stfsc_engine::world::Mesh> = model.meshes.iter()
                                .chain(model.skinned_meshes.iter().map(|s| &s.mesh))
                                .collect();
                            
                            for mesh in all_meshes {
                                // 2. Vertex Pre-processing Pass (Parallel within mesh)
                                let processed_verts: Vec<ProcessedVertex> = mesh.vertices.par_iter().map(|v| {
                                    // Transform
                                    let (world_pos, world_normal) = if let Some((_, skinning_matrices)) = &pose {
                                        let mut skinned_pos = GVec3::ZERO;
                                        let mut skinned_norm = GVec3::ZERO;
                                        for i in 0..4 {
                                            let weight = v.bone_weights[i];
                                            if weight > 0.0 {
                                                let bone_idx = v.bone_indices[i] as usize;
                                                if let Some(mat) = skinning_matrices.get(bone_idx) {
                                                    skinned_pos += mat.transform_point3(GVec3::from(v.position)) * weight;
                                                    skinned_norm += mat.transform_vector3(GVec3::from(v.normal)) * weight;
                                                }
                                            }
                                        }
                                        (transform.transform_point3(skinned_pos), transform.transform_vector3(skinned_norm).normalize())
                                    } else {
                                        (transform.transform_point3(GVec3::from(v.position)), transform.transform_vector3(GVec3::from(v.normal)).normalize())
                                    };

                                    let depth = (world_pos - cam_pos_g).dot(cam_fwd);
                                    // Project vertex to screen space
                                    // Use project_unclamped for large geometry where vertices may be off-screen
                                    let (proj_pos, proj_valid) = camera.project_unclamped(Vec3::new(world_pos.x, world_pos.y, world_pos.z), available);
                                    let screen_pos = egui::pos2(rect.min.x + proj_pos.x, rect.min.y + proj_pos.y);
                                    // Vertex is valid if projection succeeded AND depth is positive
                                    let is_valid = proj_valid && depth > 0.1;

                                    // Shade
                                    let mut color = egui::Color32::BLACK;
                                    if is_valid {
                                        let view_dir = (cam_pos_g - world_pos).normalize();
                                        // Bright, balanced ambient lighting
                                        let ambient = 0.35;
                                        let diffuse = world_normal.dot(light_dir).max(0.0) * 0.65;
                                        let half_dir = (light_dir + view_dir).normalize();
                                        let specular = world_normal.dot(half_dir).max(0.0).powf(24.0) * 0.2;
                                        let base_col = mat_color * GVec3::from(v.color);
                                        let mut shaded = base_col * (diffuse + ambient) + GVec3::splat(specular);
                                        
                                        if is_selected { shaded = shaded.lerp(GVec3::new(1.0, 0.85, 0.4), 0.25); }

                                        // Apply fog
                                        let fog_color = GVec3::new(0.50, 0.65, 0.85);
                                        let fog_dist = (depth - 20.0).max(0.0) * 0.002;
                                        let fog_factor = (1.0f32 - (-fog_dist).exp()).clamp(0.0f32, 1.0f32);
                                        shaded = shaded.lerp(fog_color, fog_factor);

                                        color = egui::Color32::from_rgb(
                                            (shaded.x.clamp(0.0, 1.0) * 255.0) as u8,
                                            (shaded.y.clamp(0.0, 1.0) * 255.0) as u8,
                                            (shaded.z.clamp(0.0, 1.0) * 255.0) as u8,
                                        );
                                    }

                                    ProcessedVertex {
                                        pos: screen_pos,
                                        color,
                                        uv: egui::pos2(v.uv[0], v.uv[1]),
                                        depth,
                                        world_pos,
                                        is_valid,
                                    }
                                }).collect();

                                // 3. Triangle Collection & Sorting pass (Parallel with stable indices)
                                let num_tris = mesh.indices.len() / 3;
                                let valid_tris: Vec<(usize, [usize; 3], f32)> = (0..num_tris).into_par_iter().filter_map(|tri_idx| {
                                    let base = tri_idx * 3;
                                    if base + 2 >= mesh.indices.len() { return None; }
                                    let indices = [mesh.indices[base] as usize, mesh.indices[base + 1] as usize, mesh.indices[base + 2] as usize];
                                    let v0 = &processed_verts[indices[0]];
                                    let v1 = &processed_verts[indices[1]];
                                    let v2 = &processed_verts[indices[2]];

                                    // Basic vertex validity check - require all vertices to be in front of camera
                                    if !v0.is_valid || !v1.is_valid || !v2.is_valid { return None; }

                                    // NO BACKFACE CULLING - matches client's vk::CullModeFlags::NONE
                                    // The GPU hardware doesn't cull backfaces, so neither should the editor

                                    // Only reject truly degenerate triangles (zero area)
                                    let area = ((v1.pos.x - v0.pos.x) * (v2.pos.y - v0.pos.y) - (v2.pos.x - v0.pos.x) * (v1.pos.y - v0.pos.y)).abs() * 0.5;
                                    if area < 0.001 { return None; }

                                    let avg_depth = (v0.depth + v1.depth + v2.depth) / 3.0;
                                    Some((tri_idx, indices, avg_depth))
                                }).collect();

                                // Stable sort by depth (far to near), preserves original order for equal depths
                                let mut valid_tris = valid_tris;
                                valid_tris.sort_by(|a, b| {
                                    match b.2.partial_cmp(&a.2) {
                                        Some(std::cmp::Ordering::Equal) | None => a.0.cmp(&b.0),
                                        Some(ord) => ord,
                                    }
                                });

                                // 4. Final Egui Mesh Construction
                                // Calculate average depth for this sub-mesh BEFORE consuming valid_tris
                                let mesh_avg_depth = if !valid_tris.is_empty() {
                                    valid_tris.iter().map(|(_, _, d)| *d).sum::<f32>() / valid_tris.len() as f32
                                } else {
                                    0.0
                                };
                                
                                let mut egui_vertices = Vec::with_capacity(valid_tris.len() * 3);
                                let mut egui_indices = Vec::with_capacity(valid_tris.len() * 3);
                                let has_texture = mesh.albedo_texture.is_some() || entity.material.albedo_texture.is_some();

                                for (_, indices, _) in valid_tris {
                                    let base_idx = egui_vertices.len() as u32;
                                    for &idx in &indices {
                                        let v = &processed_verts[idx];
                                        egui_vertices.push(egui::epaint::Vertex {
                                            pos: v.pos,
                                            uv: if has_texture { v.uv } else { white_pixel_uv },
                                            color: v.color,
                                        });
                                    }
                                    egui_indices.push(base_idx);
                                    egui_indices.push(base_idx + 1);
                                    egui_indices.push(base_idx + 2);
                                }

                                if !egui_indices.is_empty() {
                                    high_res_meshes.push((egui_vertices, egui_indices, mesh.albedo_texture.clone(), mesh_avg_depth));
                                }
                            }
                        }
                    }
                }

                // Culling for icons/lines (project corners for bounding box culling)
                let half = Vec3::new(entity.scale[0] * 0.5, entity.scale[1] * 0.5, entity.scale[2] * 0.5);
                let corners = [
                    Vec3::new(pos.x - half.x, pos.y - half.y, pos.z - half.z),
                    Vec3::new(pos.x + half.x, pos.y - half.y, pos.z - half.z),
                    Vec3::new(pos.x + half.x, pos.y + half.y, pos.z - half.z),
                    Vec3::new(pos.x - half.x, pos.y + half.y, pos.z - half.z),
                    Vec3::new(pos.x - half.x, pos.y - half.y, pos.z + half.z),
                    Vec3::new(pos.x + half.x, pos.y - half.y, pos.z + half.z),
                    Vec3::new(pos.x + half.x, pos.y + half.y, pos.z + half.z),
                    Vec3::new(pos.x - half.x, pos.y + half.y, pos.z + half.z),
                ];
                let proj: Vec<Option<egui::Pos2>> = corners.iter().map(|c| {
                    camera.project(*c, available).map(|p| egui::pos2(rect.min.x + p.x, rect.min.y + p.y))
                }).collect();
                
                let dist_to_cam = (entity_pos - cam_pos_g).length();
                (entity.id, is_selected, base_color, proj, center_proj, high_res_meshes, dist_to_cam, pos, Vec3::new(half.x, half.y, half.z), entity.material.albedo_texture.clone())
            }).collect();
            
            // Sort entities by distance from camera (far to near)
            entity_renders.sort_by(|a, b| b.6.partial_cmp(&a.6).unwrap_or(std::cmp::Ordering::Equal));
            
            // Sequential: Draw using pre-computed projections (already sorted far-to-near)
            for (id, is_selected, base_color, proj, center_proj, high_res_meshes, _dist, pos, half, entity_albedo) in entity_renders {
                let base = egui::Color32::from_rgb(base_color[0], base_color[1], base_color[2]);
                let wire_color = if is_selected { egui::Color32::GOLD } else { base };
                let stroke = egui::Stroke::new(if is_selected { 2.5 } else { 1.0 }, wire_color);
                
                // Draw wireframe/Solid - Low = ALWAYS wireframe, High = solid meshes if available
                if self.render_quality == RenderQuality::High && !high_res_meshes.is_empty() {
                    // Sort sub-meshes by depth (far to near) for correct occlusion
                    let mut high_res_meshes = high_res_meshes;
                    high_res_meshes.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
                    
                    // High Quality: Render solid lit triangles with smooth Gouraud shading and textures
                    for (egui_vertices, indices, texture_name, _depth) in high_res_meshes {
                        let mut mesh = egui::Mesh::default();
                        mesh.vertices = egui_vertices;
                        mesh.indices = indices;
                        
                        // Apply texture if available
                        let mut final_texture = texture_name;
                        if final_texture.is_none() {
                            if let Some(entity_tex) = &entity_albedo {
                                // Try to get the filename of the entity texture
                                let tex_name = std::path::Path::new(entity_tex)
                                    .file_name().unwrap_or_default().to_string_lossy().to_string();
                                final_texture = Some(tex_name);
                            }
                        }

                        if let Some(name) = final_texture {
                            if let Some(handle) = self.egui_textures.get(&name) {
                                mesh.texture_id = handle.id();
                            }
                        }
                        
                        painter.add(egui::Shape::mesh(mesh));
                    }
                    
                    // Realistic Mesh Projection Shadows for High Quality
                    // Project the entire mesh silhouette onto the ground plane (Y=0)
                    let shadow_dist = pos.y;
                    let shadow_size = half.x.max(half.z) * 2.0;

                    // Optimization: Skip shadows for small or far objects
                    if shadow_dist > 0.0 && shadow_dist < 50.0 && shadow_size > 0.1 {
                        let shadow_alpha = (60.0 / (shadow_dist * 0.5 + 1.0)).clamp(5.0, 40.0) as u8;
                        
                        let shadow_pos = Vec3::new(pos.x, 0.0, pos.z);
                        if let Some(sp) = camera.project(shadow_pos, available) {
                            let sp = egui::pos2(rect.min.x + sp.x, rect.min.y + sp.y);
                            
                            // Transform the extents to screen space to get an ellipse
                            // Use actual half-extents for more accurate shadow sizing
                            let extent_x = Vec3::new(pos.x + half.x, 0.0, pos.z);
                            let extent_z = Vec3::new(pos.x, 0.0, pos.z + half.z);
                            
                            if let (Some(px), Some(pz)) = (camera.project(extent_x, available), camera.project(extent_z, available)) {
                                let dx = px.x - (sp.x - rect.min.x);
                                let dy = px.y - (sp.y - rect.min.y);
                                let radius_x = (dx * dx + dy * dy).sqrt();
                                
                                let dzx = pz.x - (sp.x - rect.min.x);
                                let dzy = pz.y - (sp.y - rect.min.y);
                                let radius_z = (dzx * dzx + dzy * dzy).sqrt();

                                // Draw a soft blob shadow using multiple approximated ellipses for "softness"
                                // Optimized: Fewer steps and lower poly count for shadows
                                for i in 0..2 {
                                    let t = 1.0 - (i as f32 / 2.0);
                                    let alpha = (shadow_alpha as f32 * t * t) as u8;
                                    let s_color = egui::Color32::from_rgba_unmultiplied(5, 10, 15, alpha);
                                    
                                    // Approximate ellipse with a 8-sided polygon for faster results
                                    let mut points = Vec::with_capacity(8);
                                    let rad_x = radius_x * (1.0 + i as f32 * 0.6);
                                    let rad_z = radius_z * (1.0 + i as f32 * 0.6);
                                    for j in 0..8 {
                                        let angle = (j as f32 / 8.0) * std::f32::consts::TAU;
                                        points.push(egui::pos2(
                                            sp.x + angle.cos() * rad_x,
                                            sp.y + angle.sin() * rad_z,
                                        ));
                                    }
                                    painter.add(egui::Shape::convex_polygon(points, s_color, egui::Stroke::NONE));
                                }
                            }
                        }
                    }
                    
                    // Also draw a subtle wireframe overlay on selected objects for clarity
                    if is_selected {
                        for (i, j) in [
                            (0, 1), (1, 2), (2, 3), (3, 0),
                            (4, 5), (5, 6), (6, 7), (7, 4),
                            (0, 4), (1, 5), (2, 6), (3, 7),
                        ] {
                            if let (Some(p1), Some(p2)) = (proj[i], proj[j]) {
                                painter.line_segment([p1, p2], egui::Stroke::new(1.5, egui::Color32::from_rgba_unmultiplied(255, 200, 50, 180)));
                            }
                        }
                    }
                } else {
                    // Low Quality: Always render wireframe bounding boxes
                    for (i, j) in [
                        (0, 1), (1, 2), (2, 3), (3, 0),
                        (4, 5), (5, 6), (6, 7), (7, 4),
                        (0, 4), (1, 5), (2, 6), (3, 7),
                    ] {
                        if let (Some(p1), Some(p2)) = (proj[i], proj[j]) {
                            painter.line_segment([p1, p2], stroke);
                        }
                    }
                }
                
                // Center dot + click detection
                // High mode = NO dots (solid meshes only), Low mode = outline circles
                if let Some(c) = center_proj {
                    let size = (8.0 + 400.0 / self.camera.distance).clamp(4.0, 25.0);
                    
                    if self.render_quality == RenderQuality::Low {
                        // Low: outline only - transparent with stroke
                        painter.circle_stroke(c, size, egui::Stroke::new(1.5, wire_color));
                    }
                    // High: No center dot - only solid mesh rendering
                    
                    if is_selected {
                        painter.circle_stroke(c, size + 2.0, egui::Stroke::new(2.0, egui::Color32::GOLD));
                    }
                    
                    // Click detection hitbox (invisible in High mode)
                    if response.clicked() {
                        if (self.last_mouse.x - response.interact_pointer_pos().unwrap_or(self.last_mouse).x).abs() < 2.0 {
                            if let Some(click) = response.interact_pointer_pos() {
                                if (click.x - c.x).abs() < size + 5.0 && (click.y - c.y).abs() < size + 5.0 {
                                    clicked_entity = Some(id);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Draw Skeleton for selected entity in Animation Editor
        if self.active_view == EditorView::AnimationEditor {
            if let Some(id) = self.selected_entity_id {
                if let Some(scene) = &self.current_scene {
                    if let Some(entity) = scene.entities.iter().find(|e| e.id == id) {
                        self.draw_skeleton(&painter, entity, &self.camera, rect, available);
                    }
                }
            }
        }
        
        // ========================================================================
        // 3D TRANSLATE GIZMO
        // ========================================================================
        if let Some(selected_id) = self.selected_entity_id {
            if let Some(scene) = &mut self.current_scene {
                if let Some(entity_idx) = scene.entities.iter().position(|e| e.id == selected_id) {
                    let pos = scene.entities[entity_idx].position;
                    let entity_pos = Vec3::new(pos[0], pos[1], pos[2]);
                    let gizmo_scale = (self.camera.distance * 0.15).max(0.5);
                    let axes = [
                        (Vec3::new(1.0, 0.0, 0.0), egui::Color32::from_rgb(255, 60, 60), "X"),
                        (Vec3::new(0.0, 1.0, 0.0), egui::Color32::from_rgb(60, 255, 60), "Y"),
                        (Vec3::new(0.0, 0.0, 1.0), egui::Color32::from_rgb(60, 60, 255), "Z"),
                    ];

                    let mut hovered_axis = None;
                    let mouse_pos = ui.input(|i| i.pointer.interact_pos());
                    
                    for (i, (dir, color, _label)) in axes.iter().enumerate() {
                        let start = entity_pos;
                        let scaled_dir = Vec3::new(dir.x * gizmo_scale, dir.y * gizmo_scale, dir.z * gizmo_scale);
                        let end = entity_pos.add(&scaled_dir);
                        
                        if let (Some(p1), Some(p2)) = (self.camera.project(start, available), self.camera.project(end, available)) {
                            let p1 = egui::pos2(rect.min.x + p1.x, rect.min.y + p1.y);
                            let p2 = egui::pos2(rect.min.x + p2.x, rect.min.y + p2.y);
                            
                            let is_dragging = self.gizmo_axis == Some(i);
                            let mut draw_color = *color;
                            if is_dragging { draw_color = egui::Color32::WHITE; }
                            
                            // Check for hover
                            if let Some(m) = mouse_pos {
                                let dist_to_segment = {
                                    let l2 = p1.distance_sq(p2);
                                    if l2 == 0.0 { p1.distance(m) }
                                    else {
                                        let t = ((m.x - p1.x) * (p2.x - p1.x) + (m.y - p1.y) * (p2.y - p1.y)) / l2;
                                        let t = t.clamp(0.0, 1.0);
                                        m.distance(egui::pos2(p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y)))
                                    }
                                };
                                if dist_to_segment < 10.0 {
                                    hovered_axis = Some(i);
                                    if !is_dragging { draw_color = egui::Color32::YELLOW; }
                                }
                            }
                            
                            painter.line_segment([p1, p2], egui::Stroke::new(if is_dragging || hovered_axis == Some(i) { 4.0 } else { 2.5 }, draw_color));
                            
                            // Draw Arrowhead (Cone)
                            if let Some(cone_model) = self.model_cache.get("primitive://cone") {
                                let cone_transform = Mat4::from_scale_rotation_translation(
                                    GVec3::splat(gizmo_scale * 0.1),
                                    GQuat::from_rotation_arc(GVec3::Y, GVec3::new(dir.x, dir.y, dir.z)),
                                    GVec3::new(end.x, end.y, end.z)
                                );
                                for mesh in &cone_model.meshes {
                                    let mut egui_mesh = egui::Mesh::default();
                                    for v in &mesh.vertices {
                                        let world_v = cone_transform.transform_point3(GVec3::from(v.position));
                                        if let Some(p) = self.camera.project(Vec3::new(world_v.x, world_v.y, world_v.z), available) {
                                            egui_mesh.vertices.push(egui::epaint::Vertex {
                                                pos: egui::pos2(rect.min.x + p.x, rect.min.y + p.y),
                                                uv: egui::pos2(0.0, 0.0),
                                                color: draw_color,
                                            });
                                        }
                                    }
                                    egui_mesh.indices = mesh.indices.clone();
                                    painter.add(egui::Shape::mesh(egui_mesh));
                                }
                            }
                        }
                    }

                    // Selection / Dragging logic
                    if ui.input(|i| i.pointer.any_pressed()) {
                        if let Some(i) = hovered_axis {
                            self.gizmo_axis = Some(i);
                        }
                    }
                    if ui.input(|i| !i.pointer.any_down()) {
                        self.gizmo_axis = None;
                    }

                    if let Some(axis_idx) = self.gizmo_axis {
                        let axis_dir = axes[axis_idx].0;
                        let mouse_delta = ui.input(|i| i.pointer.delta());
                        
                        // Convert mouse delta to world space delta
                        // Simple approach: project mouse delta onto axis in screen space
                        if let (Some(p1), Some(p2)) = (self.camera.project(entity_pos, available), self.camera.project(entity_pos.add(&axis_dir), available)) {
                            let screen_dir = egui::vec2(p2.x - p1.x, p2.y - p1.y);
                            let screen_len = screen_dir.length();
                            if screen_len > 0.0 {
                                let normalized_screen_dir = screen_dir / screen_len;
                                let projection = mouse_delta.x * normalized_screen_dir.x + mouse_delta.y * normalized_screen_dir.y;
                                let world_delta = projection / screen_len; // Scale by reciprocal of screen-space unit length
                                
                                let mut new_pos = scene.entities[entity_idx].position;
                                match axis_idx {
                                    0 => new_pos[0] += world_delta,
                                    1 => new_pos[1] += world_delta,
                                    2 => new_pos[2] += world_delta,
                                    _ => {}
                                }
                                scene.entities[entity_idx].position = new_pos;
                                self.scene_dirty = true;
                                
                                // Deploy to engine
                                if self.is_connected {
                                    let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::UpdateTransform {
                                        id: selected_id,
                                        position: Some(new_pos),
                                        rotation: None,
                                        scale: None,
                                    }));
                                }
                            }
                        }
                    }
                }
            }
        }
        
        if let Some(id) = clicked_entity {
            if self.gizmo_axis.is_none() { // Don't change selection while dragging gizmo
                self.selected_entity_id = Some(id);
            }
        }

        // Overlay
        painter.text(
            rect.min + egui::vec2(10.0, 10.0),
            egui::Align2::LEFT_TOP,
            "Right-drag: Orbit | Middle-drag or M+Left-drag: Move Object | Left-drag: Pan",
            egui::FontId::proportional(11.0),
            egui::Color32::WHITE,
        );
        
        rect
    }

    fn draw_skeleton(&self, painter: &egui::Painter, entity: &SceneEntity, camera: &Camera3D, rect: egui::Rect, available: egui::Vec2) {
        if let Some(config) = &entity.animator_config {
            if config.bone_transforms.is_empty() { return; }
            
            // Try to get animated pose first
            let animated_pose = Self::compute_animated_pose(entity, &self.model_cache, &self.animation_editor_state);
            
            let entity_mat = Mat4::from_scale_rotation_translation(
                GVec3::from(entity.scale),
                GQuat::from_array(entity.rotation),
                GVec3::from(entity.position)
            );
            
            let globals = if let Some((global_transforms, _)) = animated_pose {
                global_transforms
            } else {
                // Fallback to bind pose
                let mut base_globals = vec![Mat4::IDENTITY; config.bone_transforms.len()];
                for i in 0..config.bone_transforms.len() {
                    if let Some(parent_idx) = config.bone_parents[i] {
                        if parent_idx < base_globals.len() {
                            base_globals[i] = base_globals[parent_idx] * config.bone_transforms[i];
                        }
                    } else {
                        base_globals[i] = config.bone_transforms[i];
                    }
                }
                base_globals
            };
            
            // Draw bones
            let bone_color = egui::Color32::from_rgb(100, 200, 255);
            let joint_color = egui::Color32::WHITE;
            
            for i in 0..globals.len() {
                let world_mat = entity_mat * globals[i];
                let pos = world_mat.transform_point3(GVec3::ZERO);
                let v3_pos = Vec3::new(pos.x, pos.y, pos.z);
                
                if let Some(p1) = camera.project(v3_pos, available) {
                    let p1_egui = egui::pos2(rect.min.x + p1.x, rect.min.y + p1.y);
                    
                    // Draw joint
                    painter.circle_filled(p1_egui, 3.0, joint_color);
                    
                    // Draw line to parent
                    if let Some(parent_idx) = config.bone_parents[i] {
                        if parent_idx < globals.len() {
                            let p_world_mat = entity_mat * globals[parent_idx];
                            let p_pos = p_world_mat.transform_point3(GVec3::ZERO);
                            let v3_p_pos = Vec3::new(p_pos.x, p_pos.y, p_pos.z);
                            
                            if let Some(p2) = camera.project(v3_p_pos, available) {
                                let p2_egui = egui::pos2(rect.min.x + p2.x, rect.min.y + p2.y);
                                painter.line_segment([p1_egui, p2_egui], egui::Stroke::new(1.5, bone_color));
                            }
                        }
                    }
                    
                    // Label bone if selected
                    if self.animation_editor_state.selected_bones.contains(&i) {
                        painter.text(
                            p1_egui + egui::vec2(5.0, 5.0),
                            egui::Align2::LEFT_TOP,
                            &config.bone_names[i],
                            egui::FontId::proportional(10.0),
                            egui::Color32::YELLOW,
                        );
                        painter.circle_stroke(p1_egui, 5.0, egui::Stroke::new(1.0, egui::Color32::YELLOW));
                    }
                }
            }
        }
    }

    fn draw_keyframe_inspector(&mut self, ui: &mut egui::Ui) {
        ui.heading("🔍 Keyframe Inspector");
        ui.add_space(8.0);
        
        let state = &mut self.animation_editor_state;
        if state.selected_keyframes.is_empty() {
            ui.vertical_centered(|ui| {
                ui.add_space(20.0);
                ui.label(egui::RichText::new("No keyframe selected").weak());
                ui.label(egui::RichText::new("Click a keyframe in the timeline to edit its properties").small());
            });
            return;
        }
        
        if state.selected_keyframes.len() > 1 {
            ui.label(format!("Multiple keyframes selected ({}).", state.selected_keyframes.len()));
            return;
        }
        
        let kf_idx = state.selected_keyframes[0];
        
        if let Some(scene) = &mut self.current_scene {
            if let Some(entity) = self.selected_entity_id
                .and_then(|id| scene.entities.iter_mut().find(|e| e.id == id))
            {
                if let Some(config) = &mut entity.animator_config {
                    if let Some(kf) = config.editor_keyframes.get_mut(kf_idx) {
                        egui::Grid::new("kf_inspector_grid")
                            .num_columns(2)
                            .spacing([10.0, 10.0])
                            .striped(true)
                            .show(ui, |ui| {
                                ui.label("Time:");
                                if ui.add(egui::DragValue::new(&mut kf.time).speed(0.01).clamp_range(0.0..=60.0).suffix("s")).changed() {
                                    self.scene_dirty = true;
                                }
                                ui.end_row();
                                
                                ui.label("Bone:");
                                let bone_name = config.bone_names.get(kf.bone_index).cloned().unwrap_or_else(|| "Unknown".to_string());
                                ui.label(egui::RichText::new(format!("#{} {}", kf.bone_index, bone_name)).strong());
                                ui.end_row();
                                
                                ui.label("Channel:");
                                ui.label(format!("{:?}", kf.channel));
                                ui.end_row();
                                
                                ui.label("Interpolation:");
                                if egui::ComboBox::from_id_source("kf_interp")
                                    .selected_text(format!("{:?}", kf.interpolation))
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut kf.interpolation, KeyframeInterpolation::Linear, "Linear");
                                        ui.selectable_value(&mut kf.interpolation, KeyframeInterpolation::Step, "Step");
                                        ui.selectable_value(&mut kf.interpolation, KeyframeInterpolation::Bezier, "Bezier");
                                    }).response.changed() {
                                        self.scene_dirty = true;
                                    }
                                ui.end_row();
                            });
                        
                        ui.add_space(8.0);
                        ui.separator();
                        ui.add_space(8.0);
                        
                        ui.label("Value:");
                        match &mut kf.value {
                            KeyframeValue::Float(f) => {
                                if ui.add(egui::DragValue::new(f).speed(0.1)).changed() {
                                    self.scene_dirty = true;
                                }
                            }
                            KeyframeValue::Vec3(v) => {
                                ui.horizontal(|ui| {
                                    ui.label("X"); if ui.add(egui::DragValue::new(&mut v[0]).speed(0.05)).changed() { self.scene_dirty = true; }
                                    ui.label("Y"); if ui.add(egui::DragValue::new(&mut v[1]).speed(0.05)).changed() { self.scene_dirty = true; }
                                    ui.label("Z"); if ui.add(egui::DragValue::new(&mut v[2]).speed(0.05)).changed() { self.scene_dirty = true; }
                                });
                            }
                            KeyframeValue::Quat(q) => {
                                egui::Grid::new("quat_grid").show(ui, |ui| {
                                    ui.label("X"); if ui.add(egui::DragValue::new(&mut q[0]).speed(0.02)).changed() { self.scene_dirty = true; }
                                    ui.label("Y"); if ui.add(egui::DragValue::new(&mut q[1]).speed(0.02)).changed() { self.scene_dirty = true; }
                                    ui.end_row();
                                    ui.label("Z"); if ui.add(egui::DragValue::new(&mut q[2]).speed(0.02)).changed() { self.scene_dirty = true; }
                                    ui.label("W"); if ui.add(egui::DragValue::new(&mut q[3]).speed(0.02)).changed() { self.scene_dirty = true; }
                                    ui.end_row();
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    
    // ========================================================================
    // ANIMATION EDITOR
    // ========================================================================
    
    fn draw_animation_editor(&mut self, ui: &mut egui::Ui) {
        ui.heading("🎭 Animation Editor");
        
        // Get selected entity with animator
        let has_animator = self.selected_entity_id.map(|id| {
            self.current_scene.as_ref()
                .and_then(|s| s.entities.iter().find(|e| e.id == id))
                .and_then(|e| e.animator_config.as_ref())
                .is_some()
        }).unwrap_or(false);
        
        if !has_animator {
            ui.add_space(20.0);
            ui.label("⚠ Select an entity with an Animator to edit animations.");
            ui.add_space(10.0);
            ui.label("To add an Animator:");
            ui.label("1. Select an entity in the Hierarchy (left panel)");
            ui.label("2. Enable '🎬 Enable Animator' in the Inspector (right panel)");
            ui.label("3. Import a 3D model with animations (.glb/.gltf)");
            return;
        }
        
        // Toolbar
        ui.horizontal(|ui| {
            self.draw_animation_toolbar(ui);
        });
        
        ui.separator();
        
        // Main content: split vertically
        // Top: Preview viewport + Bone hierarchy
        // Bottom: Timeline/Dopesheet
        
        let available = ui.available_size();
        let timeline_height = 220.0_f32.min(available.y * 0.4);
        let top_height = (available.y - timeline_height - 20.0).max(100.0);
        
        // Split top section into Hierarchy | Preview | Inspector
        ui.horizontal(|ui| {
            // Left: Bone hierarchy (220px)
            ui.allocate_ui(egui::vec2(220.0, top_height), |ui| {
                ui.vertical(|ui| {
                    egui::Frame::group(ui.style()).show(ui, |ui| {
                        self.draw_bone_hierarchy(ui);
                    });
                });
            });
            
            ui.separator();
            
            // Middle: Animation preview viewport (flexible)
            let preview_width = (ui.available_width() - 270.0).max(100.0);
            ui.allocate_ui(egui::vec2(preview_width, top_height), |ui| {
                ui.vertical(|ui| {
                    self.draw_animation_preview(ui);
                });
            });
            
            ui.separator();
            
            // Right: Keyframe Inspector (250px)
            ui.allocate_ui(egui::vec2(250.0, top_height), |ui| {
                ui.vertical(|ui| {
                    egui::Frame::group(ui.style()).show(ui, |ui| {
                        self.draw_keyframe_inspector(ui);
                    });
                });
            });
        });
        
        ui.separator();
        
        // Bottom: Timeline/Dopesheet
        ui.allocate_ui(egui::vec2(available.x, timeline_height), |ui| {
            self.draw_animation_timeline(ui);
        });
    }
    
    fn draw_animation_toolbar(&mut self, ui: &mut egui::Ui) {
        // Clip selector
        ui.label("Clip:");
        if let Some(scene) = &self.current_scene {
            if let Some(entity) = self.selected_entity_id
                .and_then(|id| scene.entities.iter().find(|e| e.id == id)) 
            {
                if let Some(config) = &entity.animator_config {
                    egui::ComboBox::from_id_source("anim_clip_select")
                        .width(150.0)
                        .selected_text(
                            self.animation_editor_state.selected_clip_idx
                                .and_then(|i| config.clip_names.get(i))
                                .map(|s| s.as_str())
                                .unwrap_or("Select clip...")
                        )
                        .show_ui(ui, |ui| {
                            for (i, name) in config.clip_names.iter().enumerate() {
                                let duration = config.clip_durations.get(i).copied().unwrap_or(0.0);
                                let text = format!("{} ({:.2}s)", name, duration);
                                if ui.selectable_label(
                                    self.animation_editor_state.selected_clip_idx == Some(i),
                                    &text
                                ).clicked() {
                                    self.animation_editor_state.selected_clip_idx = Some(i);
                                    self.animation_editor_state.current_time = 0.0;
                                }
                            }
                        });
                }
            }
        }
        
        ui.separator();
        
        // Playback controls
        let state = &mut self.animation_editor_state;
        
        if ui.button("⏮").on_hover_text("Go to start").clicked() {
            state.current_time = 0.0;
        }
        
        if state.is_playing {
            if ui.button("⏸").on_hover_text("Pause").clicked() {
                state.is_playing = false;
            }
        } else {
            if ui.button("▶").on_hover_text("Play").clicked() {
                state.is_playing = true;
                state.last_update = std::time::Instant::now();
            }
        }
        
        if ui.button("⏹").on_hover_text("Stop").clicked() {
            state.is_playing = false;
            state.current_time = 0.0;
        }
        
        if ui.button("⏭").on_hover_text("Go to end").clicked() {
            // Set to clip duration
            if let Some(scene) = &self.current_scene {
                if let Some(entity) = self.selected_entity_id
                    .and_then(|id| scene.entities.iter().find(|e| e.id == id))
                {
                    if let Some(config) = &entity.animator_config {
                        if let Some(idx) = state.selected_clip_idx {
                            if let Some(duration) = config.clip_durations.get(idx) {
                                state.current_time = *duration;
                            }
                        }
                    }
                }
            }
        }
        
        ui.separator();
        
        // Speed control
        ui.label("Speed:");
        ui.add(egui::DragValue::new(&mut self.animation_editor_state.playback_speed)
            .speed(0.1)
            .clamp_range(0.1..=3.0)
            .suffix("x"));
        
        ui.separator();
        
        // Auto-key toggle
        ui.checkbox(&mut self.animation_editor_state.auto_key, "🔑 Auto-Key")
            .on_hover_text("Automatically create keyframes when modifying transforms");
        
        ui.separator();
        
        // Time display
        let time = self.animation_editor_state.current_time;
        ui.label(format!("⏱ {:.2}s", time));
        
        ui.separator();
        
        // Add keyframe button
        if ui.button("➕ Add Keyframe").on_hover_text("Add keyframe at current time for selected bones").clicked() {
            if let Some(entity_id) = self.selected_entity_id {
                let time = self.animation_editor_state.current_time;
                let bones = self.animation_editor_state.selected_bones.clone();
                
                if let Some(scene) = &mut self.current_scene {
                    if let Some(entity) = scene.entities.iter_mut().find(|e| e.id == entity_id) {
                        if let Some(config) = &mut entity.animator_config {
                            for &bone_idx in &bones {
                                // Add default keyframes for Position X, Y, Z for now
                                config.editor_keyframes.push(EditorKeyframe {
                                    time,
                                    bone_index: bone_idx,
                                    channel: KeyframeChannel::PositionX,
                                    value: KeyframeValue::Float(0.0),
                                    interpolation: KeyframeInterpolation::Linear,
                                });
                            }
                            self.scene_dirty = true;
                        }
                    }
                }
            }
        }
        
        // Delete keyframe button
        if ui.button("🗑 Delete").on_hover_text("Delete selected keyframes").clicked() {
            if let Some(entity_id) = self.selected_entity_id {
                if let Some(scene) = &mut self.current_scene {
                    if let Some(entity) = scene.entities.iter_mut().find(|e| e.id == entity_id) {
                        if let Some(config) = &mut entity.animator_config {
                            let mut kf_indices = self.animation_editor_state.selected_keyframes.clone();
                            kf_indices.sort_unstable_by(|a, b| b.cmp(a)); // Delete from end
                            
                            for idx in kf_indices {
                                if idx < config.editor_keyframes.len() {
                                    config.editor_keyframes.remove(idx);
                                }
                            }
                            self.animation_editor_state.selected_keyframes.clear();
                            self.scene_dirty = true;
                        }
                    }
                }
            }
        }
    }
    
    fn draw_animation_timeline(&mut self, ui: &mut egui::Ui) {
        egui::Frame::dark_canvas(ui.style()).show(ui, |ui| {
            let (response, painter) = ui.allocate_painter(
                ui.available_size(),
                egui::Sense::click_and_drag(),
            );
            
            let rect = response.rect;
            let state = &mut self.animation_editor_state;
            
            // Get clip duration
            let duration = self.current_scene.as_ref()
                .and_then(|s| self.selected_entity_id
                    .and_then(|id| s.entities.iter().find(|e| e.id == id)))
                .and_then(|e| e.animator_config.as_ref())
                .and_then(|c| state.selected_clip_idx
                    .and_then(|i| c.clip_durations.get(i)))
                .copied()
                .unwrap_or(1.0)
                .max(0.1);
            
            let header_height = 25.0;
            let track_height = 20.0;
            let time_offset = 100.0; // Left margin for labels
            let usable_width = (rect.width() - time_offset).max(100.0);
            let pixels_per_second = usable_width / duration * state.timeline_zoom;
            
            // Draw timeline header with time markers
            painter.rect_filled(
                egui::Rect::from_min_size(rect.min, egui::vec2(rect.width(), header_height)),
                0.0,
                egui::Color32::from_gray(40),
            );
            
            // Draw time markers
            let step = if state.timeline_zoom > 2.0 { 0.25 } else if state.timeline_zoom > 1.0 { 0.5 } else { 1.0 };
            let mut t = 0.0;
            while t <= duration {
                let x = rect.min.x + time_offset + t * pixels_per_second;
                if x < rect.max.x {
                    painter.line_segment(
                        [egui::pos2(x, rect.min.y + 15.0), egui::pos2(x, rect.min.y + header_height)],
                        egui::Stroke::new(1.0, egui::Color32::from_gray(100)),
                    );
                    painter.text(
                        egui::pos2(x + 2.0, rect.min.y + 3.0),
                        egui::Align2::LEFT_TOP,
                        format!("{:.1}s", t),
                        egui::FontId::proportional(10.0),
                        egui::Color32::from_gray(180),
                    );
                }
                t += step;
            }
            
            // Draw "Timeline" label
            painter.text(
                egui::pos2(rect.min.x + 5.0, rect.min.y + 5.0),
                egui::Align2::LEFT_TOP,
                "Timeline",
                egui::FontId::proportional(11.0),
                egui::Color32::WHITE,
            );
            
            // Draw keyframe tracks (dopesheet style)
            let tracks_start_y = rect.min.y + header_height;
            let num_tracks = ((rect.height() - header_height) / track_height) as usize;
            
            for i in 0..num_tracks.min(20) {
                let y = tracks_start_y + (i as f32) * track_height;
                
                if y + track_height > rect.max.y {
                    break;
                }
                
                // Track background
                let bg_color = if i % 2 == 0 { 
                    egui::Color32::from_gray(30) 
                } else { 
                    egui::Color32::from_gray(25) 
                };
                painter.rect_filled(
                    egui::Rect::from_min_size(
                        egui::pos2(rect.min.x, y),
                        egui::vec2(rect.width(), track_height)
                    ),
                    0.0,
                    bg_color,
                );
                
                // Track label
                let is_selected = state.selected_bones.contains(&i);
                let text_color = if is_selected { 
                    egui::Color32::from_rgb(100, 200, 255) 
                } else { 
                    egui::Color32::from_gray(150) 
                };
                painter.text(
                    egui::pos2(rect.min.x + 5.0, y + 3.0),
                    egui::Align2::LEFT_TOP,
                    format!("Bone {}", i),
                    egui::FontId::proportional(11.0),
                    text_color,
                );
                
                // Draw separator line
                painter.line_segment(
                    [egui::pos2(rect.min.x + time_offset - 5.0, y), egui::pos2(rect.min.x + time_offset - 5.0, y + track_height)],
                    egui::Stroke::new(1.0, egui::Color32::from_gray(50)),
                );

                // Draw keyframes for this track/bone
                if let Some(scene) = &self.current_scene {
                    if let Some(entity) = self.selected_entity_id
                        .and_then(|id| scene.entities.iter().find(|e| e.id == id))
                    {
                        if let Some(config) = &entity.animator_config {
                            for (kf_idx, kf) in config.editor_keyframes.iter().enumerate() {
                                if kf.bone_index == i {
                                    let kf_x = rect.min.x + time_offset + kf.time * pixels_per_second;
                                    if kf_x >= rect.min.x + time_offset && kf_x <= rect.max.x {
                                        let is_kf_selected = state.selected_keyframes.contains(&kf_idx);
                                        let kf_color = if is_kf_selected {
                                            egui::Color32::from_rgb(255, 255, 0)
                                        } else {
                                            egui::Color32::from_gray(200)
                                        };
                                        
                                        // Draw diamond shape for keyframe
                                        let size = 5.0;
                                        let center = egui::pos2(kf_x, y + track_height / 2.0);
                                        painter.add(egui::Shape::convex_polygon(
                                            vec![
                                                center + egui::vec2(0.0, -size),
                                                center + egui::vec2(size, 0.0),
                                                center + egui::vec2(0.0, size),
                                                center + egui::vec2(-size, 0.0),
                                            ],
                                            kf_color,
                                            egui::Stroke::NONE,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // Draw current time playhead
            let playhead_x = rect.min.x + time_offset + state.current_time * pixels_per_second;
            if playhead_x >= rect.min.x + time_offset && playhead_x <= rect.max.x {
                painter.line_segment(
                    [egui::pos2(playhead_x, rect.min.y), egui::pos2(playhead_x, rect.max.y)],
                    egui::Stroke::new(2.0, egui::Color32::from_rgb(255, 80, 80)),
                );
                
                // Playhead handle
                painter.circle_filled(
                    egui::pos2(playhead_x, rect.min.y + 12.0),
                    6.0,
                    egui::Color32::from_rgb(255, 80, 80),
                );
            }
            
            // Handle drag to scrub time
            if response.dragged() && !ui.input(|i| i.modifiers.shift) {
                if let Some(pos) = response.interact_pointer_pos() {
                    let x = pos.x - rect.min.x - time_offset;
                    let new_time = (x / pixels_per_second).clamp(0.0, duration);
                    state.current_time = new_time;
                }
            }
            
            // Handle keyframe selection
            if response.clicked() {
                if let Some(mouse_pos) = response.interact_pointer_pos() {
                    let mut found_kf = None;
                    if let Some(scene) = &self.current_scene {
                        if let Some(entity) = self.selected_entity_id
                            .and_then(|id| scene.entities.iter().find(|e| e.id == id))
                        {
                            if let Some(config) = &entity.animator_config {
                                for (kf_idx, kf) in config.editor_keyframes.iter().enumerate() {
                                    let kf_x = rect.min.x + time_offset + kf.time * pixels_per_second;
                                    let y = tracks_start_y + (kf.bone_index as f32) * track_height;
                                    let center = egui::pos2(kf_x, y + track_height / 2.0);
                                    if mouse_pos.distance(center) < 8.0 {
                                        found_kf = Some(kf_idx);
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    
                    if let Some(kf_idx) = found_kf {
                        if ui.input(|i| i.modifiers.shift) {
                            if state.selected_keyframes.contains(&kf_idx) {
                                state.selected_keyframes.retain(|&idx| idx != kf_idx);
                            } else {
                                state.selected_keyframes.push(kf_idx);
                            }
                        } else {
                            state.selected_keyframes = vec![kf_idx];
                        }
                    } else if mouse_pos.x > rect.min.x + time_offset {
                        // Clicked empty space in timeline (not labels)
                        if !ui.input(|i| i.modifiers.shift) {
                            state.selected_keyframes.clear();
                        }
                    }
                }
            }
            
            // Handle scroll to zoom
            let scroll = ui.input(|i| i.scroll_delta.y);
            if scroll != 0.0 && response.hovered() {
                state.timeline_zoom = (state.timeline_zoom + scroll * 0.01).clamp(0.5, 5.0);
            }
        });
    }
    
    fn draw_bone_hierarchy(&mut self, ui: &mut egui::Ui) {
        ui.heading("🦴 Bones");
        ui.add_space(4.0);
        
        egui::ScrollArea::vertical().show(ui, |ui| {
            if let Some(scene) = &self.current_scene {
                if let Some(entity) = self.selected_entity_id
                    .and_then(|id| scene.entities.iter().find(|e| e.id == id))
                {
                    if let Some(config) = &entity.animator_config {
                        if config.clip_names.is_empty() {
                            ui.label("No animation clips found.");
                            ui.add_space(4.0);
                            ui.label("Import a model with animations (.glb/.gltf)");
                        } else {
                            // Show bone list
                            ui.label(egui::RichText::new("Select bones to edit:").small());
                            ui.add_space(4.0);
                            
                            if config.bone_names.is_empty() {
                                for i in 0..16 {
                                    let selected = self.animation_editor_state.selected_bones.contains(&i);
                                    let label = format!("  🦴 Bone {}", i);
                                    if ui.selectable_label(selected, label).clicked() {
                                        if selected { self.animation_editor_state.selected_bones.retain(|&b| b != i); }
                                        else { self.animation_editor_state.selected_bones.push(i); }
                                    }
                                }
                            } else {
                                for (i, name) in config.bone_names.iter().enumerate() {
                                    let selected = self.animation_editor_state.selected_bones.contains(&i);
                                    
                                    // Simple indentation logic
                                    let mut depth = 0;
                                    let mut curr = config.bone_parents.get(i).copied().flatten();
                                    while let Some(parent_idx) = curr {
                                        depth += 1;
                                        curr = config.bone_parents.get(parent_idx).copied().flatten();
                                        if depth > 10 { break; } // Safety
                                    }
                                    
                                    let indent = "  ".repeat(depth);
                                    let label = format!("{}🦴 {}", indent, name);
                                    
                                    if ui.selectable_label(selected, label).clicked() {
                                        if selected { self.animation_editor_state.selected_bones.retain(|&b| b != i); }
                                        else { self.animation_editor_state.selected_bones.push(i); }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    }
    
    fn draw_animation_preview(&mut self, ui: &mut egui::Ui) {
        // Update playback time if playing
        if self.animation_editor_state.is_playing {
            let now = std::time::Instant::now();
            let dt = now.duration_since(self.animation_editor_state.last_update).as_secs_f32();
            self.animation_editor_state.last_update = now;
            
            self.animation_editor_state.current_time += dt * self.animation_editor_state.playback_speed;
            
            // Loop at end of clip
            if let Some(scene) = &self.current_scene {
                if let Some(entity) = self.selected_entity_id
                    .and_then(|id| scene.entities.iter().find(|e| e.id == id))
                {
                    if let Some(config) = &entity.animator_config {
                        if let Some(idx) = self.animation_editor_state.selected_clip_idx {
                            if let Some(&duration) = config.clip_durations.get(idx) {
                                if self.animation_editor_state.current_time >= duration {
                                    self.animation_editor_state.current_time = 0.0;
                                }
                            }
                        }
                    }
                }
            }
            
            // Request repaint for continuous playback
            ui.ctx().request_repaint();
        }
        
        // Show 3D Viewport in preview area
        let rect = self.draw_3d_viewport(ui);
        
        // Add Animation Info Overlay on top of the 3D Viewport
        let painter = ui.painter().with_clip_rect(rect);
        
        // Draw "Animation Preview" label (top left)
        painter.text(
            egui::pos2(rect.min.x + 10.0, rect.min.y + 10.0),
            egui::Align2::LEFT_TOP,
            "🎭 Animation Preview",
            egui::FontId::proportional(14.0),
            egui::Color32::from_gray(180),
        );
        
        // Draw current time and clip info (top center)
        let clip_info = self.current_scene.as_ref()
            .and_then(|s| self.selected_entity_id
                .and_then(|id| s.entities.iter().find(|e| e.id == id)))
            .and_then(|e| e.animator_config.as_ref())
            .and_then(|c| self.animation_editor_state.selected_clip_idx
                .and_then(|i| c.clip_names.get(i).map(|name| (name.clone(), c.clip_durations.get(i).copied().unwrap_or(0.0)))))
            .map(|(name, dur)| format!("{} ({:.2}s)", name, dur))
            .unwrap_or_else(|| "No clip selected".to_string());
        
        painter.text(
            egui::pos2(rect.center().x, rect.min.y + 15.0),
            egui::Align2::CENTER_TOP,
            format!("🎬 Clip: {} | ⏱ Time: {:.2}s", clip_info, self.animation_editor_state.current_time),
            egui::FontId::proportional(16.0),
            egui::Color32::WHITE,
        );
        
        // Draw play indicator if playing (top right)
        if self.animation_editor_state.is_playing {
            painter.text(
                egui::pos2(rect.max.x - 10.0, rect.min.y + 10.0),
                egui::Align2::RIGHT_TOP,
                "▶ Playing",
                egui::FontId::proportional(12.0),
                egui::Color32::from_rgb(100, 255, 100),
            );
        }
        
        // Send preview update to engine if connected
        if self.is_connected {
            if let Some(id) = self.selected_entity_id {
                if let Some(clip_idx) = self.animation_editor_state.selected_clip_idx {
                    let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::PreviewAnimationAt {
                        id,
                        clip_index: clip_idx,
                        time: self.animation_editor_state.current_time,
                    }));
                }
            }
        }
    }
}

impl eframe::App for EditorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(ev) = self.event_rx.try_recv() {
            match ev {
                AppEvent::Connected => {
                    self.status = "Connected ✓".into();
                    self.is_connected = true;
                }
                AppEvent::ConnectionError(e) => {
                    self.status = format!("Error: {}", e);
                    self.is_connected = false;
                }
                AppEvent::SendError(e) => {
                    self.status = format!("Send Error: {}", e);
                }
                AppEvent::StatusUpdate(m) => {
                    self.status = m;
                }
            }
        }

        // Keyboard shortcuts (global)
        ctx.input(|i| {
            if i.modifiers.ctrl {
                if i.key_pressed(egui::Key::C) {
                    // Ctrl+C: Copy
                }
                if i.key_pressed(egui::Key::V) {
                    // Ctrl+V: Paste
                }
                if i.key_pressed(egui::Key::Z) {
                    // Ctrl+Z: Undo
                }
                if i.key_pressed(egui::Key::Y) {
                    // Ctrl+Y: Redo
                }
            }
        });

        // Now handle the shortcuts outside the input closure to avoid borrow issues
        let do_copy = ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::C));
        let do_paste = ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::V));
        let do_undo = ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Z));
        let do_redo = ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Y));

        if do_copy {
            self.copy_selected();
        }
        if do_paste {
            self.paste_from_clipboard();
        }
        if do_undo {
            self.undo();
        }
        if do_redo {
            self.redo();
        }

        // MENU
        egui::TopBottomPanel::top("menu").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("📄 New Scene").clicked() {
                        if self.check_unsaved_and_set_action(PendingAction::NewScene) {
                            self.show_new_scene_dialog = true;
                            self.new_scene_name.clear();
                        }
                        ui.close_menu();
                    }
                    if ui.button("🧪 Test Engine Scene").clicked() {
                        if self.check_unsaved_and_set_action(PendingAction::LoadTestScene) {
                            self.current_scene = Some(Scene::create_test_scene());
                            self.current_scene_path = None;
                            self.scene_dirty = false;
                            if !self.scenes.contains(&"Test".to_string()) {
                                self.scenes.push("Test".into());
                            }
                            self.status = "Loaded Test Engine Scene".into();
                        }
                        ui.close_menu();
                    }
                    ui.separator();
                    
                    // Open Scene
                    if ui.button("📂 Open Scene...").clicked() {
                        self.scan_scene_files();
                        self.show_open_scene_dialog = true;
                        ui.close_menu();
                    }
                    
                    // Recent Scenes submenu
                    if !self.recent_scenes.is_empty() {
                        ui.menu_button("📋 Recent Scenes", |ui| {
                            let recent_clone = self.recent_scenes.clone();
                            for path in &recent_clone {
                                let display_name = std::path::Path::new(path)
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or(path);
                                if ui.button(display_name).clicked() {
                                    let path_clone = path.clone();
                                    if self.check_unsaved_and_set_action(PendingAction::OpenScene(path_clone.clone())) {
                                        if let Err(e) = self.load_scene_from_path(ctx, &path_clone) {
                                            self.status = format!("Error: {}", e);
                                        }
                                    }
                                    ui.close_menu();
                                }
                            }
                        });
                    }
                    
                    ui.separator();
                    
                    // Save Scene (to known path or trigger Save As)
                    if ui.button("💾 Save Scene").clicked() {
                        if let Some(path) = self.current_scene_path.clone() {
                            if let Err(e) = self.save_scene_to_path(&path) {
                                self.status = format!("Error: {}", e);
                            }
                        } else if let Some(scene) = &self.current_scene {
                            // Default to scene name if no path
                            let path = format!("scenes/{}.json", scene.name.to_lowercase().replace(" ", "_"));
                            if let Err(e) = self.save_scene_to_path(&path) {
                                self.status = format!("Error: {}", e);
                            }
                        }
                        ui.close_menu();
                    }
                    
                    // Save Scene As
                    if ui.button("💾 Save Scene As...").clicked() {
                        if let Some(scene) = &self.current_scene {
                            self.save_as_path = format!("scenes/{}.json", scene.name.to_lowercase().replace(" ", "_"));
                        } else {
                            self.save_as_path = "scenes/untitled.json".to_string();
                        }
                        self.show_save_as_dialog = true;
                        ui.close_menu();
                    }
                    
                    ui.separator();
                    if ui.button("Exit").clicked() {
                        if self.check_unsaved_and_set_action(PendingAction::Exit) {
                            std::process::exit(0);
                        }
                        ui.close_menu();
                    }
                });
                ui.menu_button("Edit", |ui| {
                    if ui
                        .add_enabled(
                            self.selected_entity_id.is_some(),
                            egui::Button::new("🗑 Delete Selected"),
                        )
                        .clicked()
                    {
                        if let Some(id) = self.selected_entity_id {
                            if self.is_connected {
                                let _ = self
                                    .command_tx
                                    .send(AppCommand::Send(SceneUpdate::DeleteEntity { id }));
                            }
                            if let Some(scene) = &mut self.current_scene {
                                scene.entities.retain(|e| e.id != id);
                            }
                            self.selected_entity_id = None;
                        }
                        ui.close_menu();
                    }
                });
                ui.menu_button("View", |ui| {
                    ui.label("Rendering Quality");
                    ui.selectable_value(&mut self.render_quality, RenderQuality::Low, "Low (Wireframe)");
                    ui.selectable_value(&mut self.render_quality, RenderQuality::High, "High (Solid)");
                    
                    ui.separator();
                    ui.label("Camera");
                    if ui.button("🔄 Reset Camera").clicked() {
                        self.camera = Camera3D::new();
                        ui.close_menu();
                    }
                    if ui.button("🔍 Focus Selected").clicked() {
                        if let (Some(id), Some(scene)) = (self.selected_entity_id, &self.current_scene) {
                            if let Some(e) = scene.entities.iter().find(|e| e.id == id) {
                                self.camera.target = Vec3::new(e.position[0], e.position[1], e.position[2]);
                            }
                        }
                        ui.close_menu();
                    }
                });
                ui.menu_button("GameObject", |ui| {
                    ui.label("3D Primitives");
                    ui.separator();
                    for ptype in PrimitiveType::all() {
                        if ui
                            .button(format!("{} {}", ptype.icon(), ptype.name()))
                            .clicked()
                        {
                            self.add_primitive(ptype);
                            ui.close_menu();
                        }
                    }
                    ui.separator();
                    if ui.button("🚗 Vehicle").clicked() {
                        if let Some(scene) = &mut self.current_scene {
                            let id = scene.entities.iter().map(|e| e.id).max().unwrap_or(0) + 1;
                            scene.entities.push(SceneEntity {
                                id,
                                name: format!("Vehicle {}", id),
                                position: [0.0, 1.0, 0.0],
                                rotation: [0.0, 0.0, 0.0, 1.0],
                                scale: [2.0, 1.0, 4.0],
                                entity_type: EntityType::Vehicle,
                                material: Material {
                                    albedo_color: [1.0, 1.0, 0.0],
                                    ..Default::default()
                                },
                                script: None,
                                collision_enabled: true,
                                layer: LAYER_VEHICLE,
                                is_static: false,
                                animator_config: None,
                                parent_id: None,
                                fov: 0.785,
                                deployed: false,
                            });
                            self.selected_entity_id = Some(id);
                        }
                        ui.close_menu();
                    }
                    if ui.button("🚶 Crowd Agent").clicked() {
                        if let Some(scene) = &mut self.current_scene {
                            let id = scene.entities.iter().map(|e| e.id).max().unwrap_or(0) + 1;
                            scene.entities.push(SceneEntity {
                                id,
                                name: format!("Agent {}", id),
                                position: [0.0, 1.0, 0.0],
                                rotation: [0.0, 0.0, 0.0, 1.0],
                                scale: [0.5, 1.8, 0.5],
                                entity_type: EntityType::CrowdAgent {
                                    state: "Walking".into(),
                                    speed: 2.0,
                                },
                                material: Material::default(),
                                script: Some("CrowdAgent".into()),
                                collision_enabled: true,
                                layer: LAYER_CHARACTER,
                                is_static: false,
                                animator_config: None,
                                parent_id: None,
                                fov: 0.785,
                                deployed: false,
                            });
                            self.selected_entity_id = Some(id);
                        }
                        ui.close_menu();
                    }
                    if ui.button("🎥 Player Start").clicked() {
                        if let Some(scene) = &mut self.current_scene {
                            let id = scene.entities.iter().map(|e| e.id).max().unwrap_or(0) + 1;
                            scene.entities.push(SceneEntity {
                                id,
                                name: "Player Start".into(),
                                position: [0.0, 1.7, 0.0],
                                rotation: [0.0, 0.0, 0.0, 1.0],
                                scale: [0.5, 0.5, 0.5],
                                entity_type: EntityType::Camera,
                                material: Material {
                                    albedo_color: [0.3, 0.3, 1.0],
                                    ..Default::default()
                                },
                                script: None,
                                collision_enabled: false,
                                layer: LAYER_DEFAULT,
                                is_static: true,
                                animator_config: None,
                                parent_id: None,
                                fov: 0.785,
                                deployed: false,
                            });
                            self.selected_entity_id = Some(id);
                        }
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.label("💡 Lights");
                    if ui.button("💡 Point Light").clicked() {
                        if let Some(scene) = &mut self.current_scene {
                            let id = scene.entities.iter().map(|e| e.id).max().unwrap_or(0) + 1;
                            scene.entities.push(SceneEntity {
                                id,
                                name: format!("Point Light {}", id),
                                position: [0.0, 5.0, 0.0],
                                rotation: [0.0, 0.0, 0.0, 1.0],
                                scale: [0.5, 0.5, 0.5],
                                entity_type: EntityType::Light {
                                    light_type: LightTypeEditor::Point,
                                    intensity: 5.0,
                                    range: 20.0,
                                    color: [1.0, 1.0, 0.9],
                                },
                                material: Material {
                                    albedo_color: [1.0, 1.0, 0.5],
                                    ..Default::default()
                                },
                                script: None,
                                collision_enabled: false,
                                layer: LAYER_DEFAULT,
                                is_static: true,
                                animator_config: None,
                                parent_id: None,
                                fov: 0.785,
                                deployed: false,
                            });
                            self.selected_entity_id = Some(id);
                        }
                        ui.close_menu();
                    }
                    if ui.button("🔦 Spot Light").clicked() {
                        if let Some(scene) = &mut self.current_scene {
                            let id = scene.entities.iter().map(|e| e.id).max().unwrap_or(0) + 1;
                            scene.entities.push(SceneEntity {
                                id,
                                name: format!("Spot Light {}", id),
                                position: [0.0, 10.0, 0.0],
                                rotation: [0.0, 0.0, 0.0, 1.0],
                                scale: [0.5, 0.5, 0.5],
                                entity_type: EntityType::Light {
                                    light_type: LightTypeEditor::Spot,
                                    intensity: 10.0,
                                    range: 30.0,
                                    color: [1.0, 1.0, 1.0],
                                },
                                material: Material {
                                    albedo_color: [1.0, 0.8, 0.3],
                                    ..Default::default()
                                },
                                script: None,
                                collision_enabled: false,
                                layer: LAYER_DEFAULT,
                                is_static: true,
                                animator_config: None,
                                parent_id: None,
                                fov: 0.785,
                                deployed: false,
                            });
                            self.selected_entity_id = Some(id);
                        }
                        ui.close_menu();
                    }
                    if ui.button("☀️ Directional Light").clicked() {
                        if let Some(scene) = &mut self.current_scene {
                            let id = scene.entities.iter().map(|e| e.id).max().unwrap_or(0) + 1;
                            scene.entities.push(SceneEntity {
                                id,
                                name: format!("Sun Light {}", id),
                                position: [50.0, 100.0, 50.0],
                                rotation: [0.0, 0.0, 0.0, 1.0],
                                scale: [1.0, 1.0, 1.0],
                                entity_type: EntityType::Light {
                                    light_type: LightTypeEditor::Directional,
                                    intensity: 3.0,
                                    range: 1000.0,
                                    color: [1.0, 0.95, 0.8],
                                },
                                material: Material {
                                    albedo_color: [1.0, 0.9, 0.5],
                                    ..Default::default()
                                },
                                script: None,
                                collision_enabled: false,
                                layer: LAYER_DEFAULT,
                                is_static: true,
                                animator_config: None,
                                parent_id: None,
                                fov: 0.785,
                                deployed: false,
                            });
                            self.selected_entity_id = Some(id);
                        }
                        ui.close_menu();
                    }

                    ui.separator();
                    if ui.button("🔊 Audio Source").clicked() {
                        if let Some(scene) = &mut self.current_scene {
                            let id = scene.entities.iter().map(|e| e.id).max().unwrap_or(0) + 1;
                            scene.entities.push(SceneEntity {
                                id,
                                name: format!("Audio Source {}", id),
                                position: [0.0, 2.0, 0.0],
                                rotation: [0.0, 0.0, 0.0, 1.0],
                                scale: [0.5, 0.5, 0.5],
                                entity_type: EntityType::AudioSource {
                                    sound_id: "test_sound".into(),
                                    volume: 1.0,
                                    looping: true,
                                    max_distance: 20.0,
                                    audio_data: None,
                                },
                                material: Material {
                                    albedo_color: [0.2, 0.8, 1.0],
                                    ..Default::default()
                                },
                                script: None,
                                collision_enabled: false,
                                layer: LAYER_DEFAULT,
                                is_static: true,
                                animator_config: None,
                                parent_id: None,
                                fov: 0.785,
                                deployed: false,
                            });
                            self.selected_entity_id = Some(id);
                        }
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.label("📦 Import");
                    if ui.button("📦 Import 3D Model...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("3D Models", &["obj", "glb", "gltf", "fbx"])
                            .pick_file()
                        {
                            if let Some(path_str) = path.to_str() {
                                if let Err(e) = self.import_model_from_path(ctx, path_str) {
                                    self.status = format!("Import failed: {}", e);
                                }
                            }
                        }
                        ui.close_menu();
                    }
                    if ui.button("🖼 UI Editor").clicked() {
                        self.show_ui_editor = !self.show_ui_editor;
                        ui.close_menu();
                    }
                });
                ui.menu_button("Scene", |ui| {
                    if ui.button("🚀 Deploy All to Quest").clicked() {
                        if self.is_connected {
                            self.deploy_all_to_quest();
                            self.status = "Deployed scene to Quest!".into();
                        } else {
                            self.status = "Not connected!".into();
                        }
                        ui.close_menu();
                    }
                    if ui.button("🗑 Clear Quest Scene").clicked() {
                        if self.is_connected {
                            let _ = self
                                .command_tx
                                .send(AppCommand::Send(SceneUpdate::ClearScene));
                        }
                        ui.close_menu();
                    }
                    ui.separator();
                    // Procedural generation toggle
                    let label = if self.procedural_generation_enabled {
                        "🌆 Procedural Gen: ON"
                    } else {
                        "🌆 Procedural Gen: OFF"
                    };
                    if ui.button(label).clicked() {
                        self.procedural_generation_enabled = !self.procedural_generation_enabled;
                        if self.is_connected {
                            let _ = self.command_tx.send(AppCommand::Send(
                                SceneUpdate::SetProceduralGeneration {
                                    enabled: self.procedural_generation_enabled,
                                },
                            ));
                            self.status = format!(
                                "Procedural Gen: {}",
                                if self.procedural_generation_enabled {
                                    "ON"
                                } else {
                                    "OFF"
                                }
                            );
                        }
                        ui.close_menu();
                    }

                    ui.separator();
                    ui.label("Respawn System");
                    if let Some(scene) = &mut self.current_scene {
                        let mut changed = false;
                        if ui.checkbox(&mut scene.respawn_enabled, "Enable Min Y Respawn").changed() {
                            changed = true;
                        }
                        ui.horizontal(|ui| {
                            ui.label("Min Y:");
                            if ui.add(egui::DragValue::new(&mut scene.respawn_y).speed(0.1)).changed() {
                                changed = true;
                            }
                        });

                        if changed && self.is_connected {
                            let _ = self.command_tx.send(AppCommand::Send(
                                SceneUpdate::SetRespawnSettings {
                                    enabled: scene.respawn_enabled,
                                    y_threshold: scene.respawn_y,
                                },
                            ));
                            self.scene_dirty = true;
                        }
                    }
                });


                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let c = if self.is_connected {
                        egui::Color32::GREEN
                    } else {
                        egui::Color32::RED
                    };
                    ui.colored_label(c, &self.status);
                });
            });
        });

        // STATUS BAR
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!(
                    "Scene: {}",
                    self.current_scene
                        .as_ref()
                        .map(|s| s.name.as_str())
                        .unwrap_or("None")
                ));
                ui.separator();
                ui.label(format!(
                    "Entities: {}",
                    self.current_scene
                        .as_ref()
                        .map(|s| s.entities.len())
                        .unwrap_or(0)
                ));
                if let Some(id) = self.selected_entity_id {
                    ui.separator();
                    ui.label(format!("Selected: #{}", id));
                }
            });
        });

        // LEFT - PROJECT
        egui::SidePanel::left("project")
            .default_width(160.0)
            .show(ctx, |ui| {
                ui.heading("Project");

                egui::CollapsingHeader::new("🔌 Connection")
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.add(egui::TextEdit::singleline(&mut self.ip).desired_width(90.0));
                            if ui
                                .button(if self.is_connected { "✓" } else { "→" })
                                .clicked()
                            {
                                let _ = self.command_tx.send(AppCommand::Connect(self.ip.clone()));
                            }
                        });
                    });

                egui::CollapsingHeader::new("📁 Scenes")
                    .default_open(true)
                    .show(ui, |ui| {
                        for (i, name) in self.scenes.iter().enumerate() {
                            if ui
                                .selectable_label(
                                    self.selected_scene_idx == Some(i),
                                    format!(" 📄 {}", name),
                                )
                                .clicked()
                            {
                                self.selected_scene_idx = Some(i);
                                if name == "Test" {
                                    self.current_scene = Some(Scene::create_test_scene());
                                }
                            }
                        }
                    });

                ui.add_space(10.0);
                egui::CollapsingHeader::new("🧱 Primitives")
                    .default_open(true)
                    .show(ui, |ui| {
                        for ptype in PrimitiveType::all() {
                            if ui
                                .button(format!("{} {}", ptype.icon(), ptype.name()))
                                .clicked()
                            {
                                self.add_primitive(ptype);
                            }
                        }
                    });
            });

        // RIGHT - INSPECTOR
        let command_tx = self.command_tx.clone();
        egui::SidePanel::right("inspector").default_width(260.0).show(ctx, |ui| {
            ui.heading("Inspector");
            ui.separator();
            
            let mut delete_id: Option<u32> = None;
            let mut deploy_id: Option<u32> = None;
            let mut inspector_dirty = false; // Track if any inspector field changed
            let mut texture_to_cache: Option<(String, Vec<u8>)> = None;
            
            // Capture entity state when selection changes (for undo tracking)
            if self.selected_entity_id != self.last_selected_id {
                // Push undo for previous entity if it was modified
                if let (Some(start), Some(prev_id)) = (self.inspector_start_entity.take(), self.last_selected_id) {
                    if let Some(scene) = &self.current_scene {
                        if let Some(after) = scene.entities.iter().find(|e| e.id == prev_id) {
                            // Only push if something changed (compare key fields)
                            // Comparing all fields including material and script
                            if start.name != after.name || start.position != after.position 
                               || start.scale != after.scale 
                               || start.material.albedo_color != after.material.albedo_color
                               || start.material.albedo_texture != after.material.albedo_texture
                               || start.material.metallic != after.material.metallic
                               || start.material.roughness != after.material.roughness
                               || start.script != after.script
                               || start.collision_enabled != after.collision_enabled {
                                self.undo_stack.push(EditorAction::ModifyEntity {
                                    id: prev_id,
                                    before: start,
                                    after: after.clone(),
                                });
                                self.redo_stack.clear();
                            }
                        }
                    }
                }
                // Capture new entity's state
                if let Some(id) = self.selected_entity_id {
                    if let Some(scene) = &self.current_scene {
                        if let Some(entity) = scene.entities.iter().find(|e| e.id == id) {
                            self.inspector_start_entity = Some(entity.clone());
                        }
                    }
                }
                self.last_selected_id = self.selected_entity_id;
            }
            
            if let Some(id) = self.selected_entity_id {
                if let Some(scene) = &mut self.current_scene {
                    // Pre-collect potential parents for the dropdown
                    let parent_options: Vec<(u32, String)> = scene.entities.iter()
                        .map(|e| (e.id, e.name.clone()))
                        .collect();

                    if let Some(entity) = scene.entities.iter_mut().find(|e| e.id == id) {
                        ui.horizontal(|ui| { 
                            ui.label("Name:"); 
                            if ui.text_edit_singleline(&mut entity.name).changed() {
                                inspector_dirty = true;
                            }
                        });
                        
                        // Hierarchy: Parent Dropdown
                        ui.horizontal(|ui| {
                            ui.label("🔗 Parent:");
                            let parent_name = if let Some(pid) = entity.parent_id {
                                parent_options.iter()
                                    .find(|(id, _)| *id == pid)
                                    .map(|(_, name)| name.clone())
                                    .unwrap_or_else(|| format!("Unknown ({})", pid))
                            } else {
                                "None".to_string()
                            };
                            
                            egui::ComboBox::from_id_source(format!("parent_{}", entity.id))
                                .selected_text(&parent_name)
                                .show_ui(ui, |ui| {
                                    if ui.selectable_label(entity.parent_id.is_none(), "None").clicked() {
                                        entity.parent_id = None;
                                        inspector_dirty = true;
                                    }
                                    for (other_id, other_name) in &parent_options {
                                        if *other_id == entity.id { continue; } // Can't parent to self
                                        if ui.selectable_label(entity.parent_id == Some(*other_id), other_name).clicked() {
                                            entity.parent_id = Some(*other_id);
                                            inspector_dirty = true;
                                        }
                                    }
                                });
                        });

                        if let EntityType::Camera = entity.entity_type {
                            ui.horizontal(|ui| {
                                ui.label("🎥 FOV:");
                                if ui.add(egui::Slider::new(&mut entity.fov, 0.1..=3.0)).changed() {
                                    inspector_dirty = true;
                                }
                            });
                        }

                        ui.horizontal(|ui| {
                            ui.label("📜 Script:");
                            let mut script_name = entity.script.clone().unwrap_or_default();
                            if ui.text_edit_singleline(&mut script_name).changed() {
                                entity.script = if script_name.is_empty() { None } else { Some(script_name) };
                                inspector_dirty = true;
                            }
                        });
                        
                        // Animator Controller Section
                        ui.add_space(4.0);
                        let has_animator = entity.animator_config.is_some();
                        let mut enable_animator = has_animator;
                        if ui.checkbox(&mut enable_animator, "🎬 Enable Animator").changed() {
                            if enable_animator && !has_animator {
                                entity.animator_config = Some(stfsc_engine::world::animation::AnimatorConfig::new());
                            } else if !enable_animator && has_animator {
                                entity.animator_config = None;
                            }
                            inspector_dirty = true;
                        }
                        
                        if let Some(anim_config) = &mut entity.animator_config {
                            egui::CollapsingHeader::new("🎬 Animator Controller")
                                .default_open(true)
                                .show(ui, |ui| {
                                    // Root Motion & Speed
                                    ui.horizontal(|ui| {
                                        ui.checkbox(&mut anim_config.apply_root_motion, "Apply Root Motion");
                                        ui.add(egui::DragValue::new(&mut anim_config.speed).speed(0.05).prefix("Speed: ").clamp_range(0.0..=5.0));
                                    });
                                    
                                    // Clips List (Metadata)
                                    ui.collapsing("🎬 All Animations", |ui| {
                                        if anim_config.clip_names.is_empty() {
                                            ui.label("No clips found.");
                                        } else {
                                            for (i, name) in anim_config.clip_names.iter().enumerate() {
                                                let duration = anim_config.clip_durations.get(i).copied().unwrap_or(0.0);
                                                ui.horizontal(|ui| {
                                                    ui.label(format!("{}: {}", i, name));
                                                    
                                                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                                        if ui.button("▶ Play").clicked() {
                                                            let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::PreviewAnimation {
                                                                id: entity.id,
                                                                clip_index: i,
                                                            }));
                                                        }
                                                        ui.label(format!("{:.2}s", duration));
                                                    });
                                                });
                                            }
                                        }
                                    });

                                    // States
                                    ui.collapsing("📦 States", |ui| {
                                        let mut remove_idx = None;
                                        let param_names: Vec<String> = anim_config.parameters.iter().map(|(p, _)| p.clone()).collect();
                                        
                                        for (idx, state) in anim_config.states.iter_mut().enumerate() {
                                            ui.group(|ui| {
                                                ui.horizontal(|ui| {
                                                    ui.text_edit_singleline(&mut state.name);
                                                    ui.add(egui::DragValue::new(&mut state.clip_index).prefix("Clip: "));
                                                    if ui.small_button("🗑").clicked() {
                                                        remove_idx = Some(idx);
                                                    }
                                                });
                                                
                                                ui.horizontal(|ui| {
                                                    ui.add(egui::DragValue::new(&mut state.speed_multiplier).speed(0.05).prefix("Speed: "));
                                                    
                                                    ui.add_space(8.0);
                                                    ui.label("Trigger:");
                                                    let current_trigger = state.trigger_param.clone().unwrap_or_else(|| "None".to_string());
                                                    egui::ComboBox::from_id_source(format!("trigger_{}", idx))
                                                        .selected_text(&current_trigger)
                                                        .show_ui(ui, |ui| {
                                                            if ui.selectable_label(state.trigger_param.is_none(), "None").clicked() {
                                                                state.trigger_param = None;
                                                                inspector_dirty = true;
                                                            }
                                                            for param in &param_names {
                                                                if ui.selectable_label(state.trigger_param.as_ref() == Some(param), param).clicked() {
                                                                    state.trigger_param = Some(param.clone());
                                                                    inspector_dirty = true;
                                                                }
                                                            }
                                                        });
                                                });
                                            });
                                        }
                                        if let Some(idx) = remove_idx {
                                            anim_config.states.remove(idx);
                                            inspector_dirty = true;
                                        }
                                        if ui.small_button("+ Add State").clicked() {
                                            let name = format!("State{}", anim_config.states.len());
                                            anim_config.add_state(&name, 0);
                                            inspector_dirty = true;
                                        }
                                    });
                                    
                                    // Default State
                                    ui.horizontal(|ui| {
                                        ui.label("Default:");
                                        egui::ComboBox::from_id_source("default_state")
                                            .selected_text(&anim_config.default_state)
                                            .show_ui(ui, |ui| {
                                                for state in &anim_config.states {
                                                    ui.selectable_value(&mut anim_config.default_state, state.name.clone(), &state.name);
                                                }
                                            });
                                    });
                                    
                                    // Transitions
                                    ui.collapsing("↔️ Transitions", |ui| {
                                        let state_names: Vec<String> = anim_config.states.iter().map(|s| s.name.clone()).collect();
                                        let mut remove_idx = None;
                                        for (idx, trans) in anim_config.transitions.iter_mut().enumerate() {
                                            ui.horizontal(|ui| {
                                                egui::ComboBox::from_id_source(format!("from_{}", idx))
                                                    .width(60.0)
                                                    .selected_text(&trans.from_state)
                                                    .show_ui(ui, |ui| {
                                                        ui.selectable_value(&mut trans.from_state, "ANY".to_string(), "ANY");
                                                        for name in &state_names {
                                                            ui.selectable_value(&mut trans.from_state, name.clone(), name);
                                                        }
                                                    });
                                                ui.label("→");
                                                egui::ComboBox::from_id_source(format!("to_{}", idx))
                                                    .width(60.0)
                                                    .selected_text(&trans.to_state)
                                                    .show_ui(ui, |ui| {
                                                        for name in &state_names {
                                                            ui.selectable_value(&mut trans.to_state, name.clone(), name);
                                                        }
                                                    });
                                                ui.add(egui::DragValue::new(&mut trans.duration).speed(0.01).prefix("t:").clamp_range(0.0..=2.0));
                                                if ui.small_button("🗑").clicked() {
                                                    remove_idx = Some(idx);
                                                }
                                            });
                                        }
                                        if let Some(idx) = remove_idx {
                                            anim_config.transitions.remove(idx);
                                            inspector_dirty = true;
                                        }
                                        if ui.small_button("+ Add Transition").clicked() {
                                            use stfsc_engine::world::animation::TransitionConditionConfig;
                                            anim_config.add_transition("", "", TransitionConditionConfig::AnimationComplete, 0.2);
                                            inspector_dirty = true;
                                        }
                                    });
                                    
                                    // Parameters
                                    ui.collapsing("⚙️ Parameters", |ui| {
                                        let mut remove_idx = None;
                                        for (idx, (name, param)) in anim_config.parameters.iter_mut().enumerate() {
                                            ui.horizontal(|ui| {
                                                ui.text_edit_singleline(name);
                                                match param {
                                                    stfsc_engine::world::animation::AnimParamConfig::Float(v) => {
                                                        ui.label("Float");
                                                        ui.add(egui::DragValue::new(v).speed(0.05));
                                                    }
                                                    stfsc_engine::world::animation::AnimParamConfig::Bool(v) => {
                                                        ui.checkbox(v, "Bool");
                                                    }
                                                    stfsc_engine::world::animation::AnimParamConfig::Int(v) => {
                                                        ui.label("Int");
                                                        ui.add(egui::DragValue::new(v));
                                                    }
                                                    stfsc_engine::world::animation::AnimParamConfig::Trigger => {
                                                        ui.label("Trigger");
                                                    }
                                                }
                                                if ui.small_button("🗑").clicked() {
                                                    remove_idx = Some(idx);
                                                }
                                            });
                                        }
                                        if let Some(idx) = remove_idx {
                                            anim_config.parameters.remove(idx);
                                            inspector_dirty = true;
                                        }
                                        ui.horizontal(|ui| {
                                            if ui.small_button("+ Float").clicked() {
                                                anim_config.add_float(&format!("Param{}", anim_config.parameters.len()), 0.0);
                                                inspector_dirty = true;
                                            }
                                            if ui.small_button("+ Bool").clicked() {
                                                anim_config.add_bool(&format!("Flag{}", anim_config.parameters.len()), false);
                                                inspector_dirty = true;
                                            }
                                            if ui.small_button("+ Trigger").clicked() {
                                                anim_config.add_trigger(&format!("Trigger{}", anim_config.parameters.len()));
                                                inspector_dirty = true;
                                            }
                                        });
                                    });
                                    
                                    // IK Toggles
                                    ui.horizontal(|ui| {
                                        ui.checkbox(&mut anim_config.look_at_ik_enabled, "Look-At IK");
                                        ui.checkbox(&mut anim_config.foot_ik_enabled, "Foot IK");
                                    });
                                    
                                    // Deploy Animator button
                                    if ui.button("🚀 Deploy Animator").clicked() {
                                        let _ = command_tx.send(AppCommand::Send(SceneUpdate::AttachAnimator {
                                            id: entity.id,
                                            config: anim_config.clone(),
                                        }));
                                    }
                                });
                        }

                        if ui.checkbox(&mut entity.collision_enabled, "🧱 Toggle Collision").changed() {
                            inspector_dirty = true;
                            if self.is_connected {
                                let _ = command_tx.send(AppCommand::Send(SceneUpdate::SetCollision {
                                    id: entity.id,
                                    enabled: entity.collision_enabled,
                                    layer: entity.layer,
                                }));
                            }
                        }
                        if ui.checkbox(&mut entity.is_static, "⚓ Static (No Gravity)").changed() {
                            inspector_dirty = true;
                            // Static toggle requires re-deploy to take effect
                        }
                        
                        ui.horizontal(|ui| {
                            ui.label("Layer:");
                            egui::ComboBox::from_id_source("layer_combo")
                                .selected_text(match entity.layer {
                                    LAYER_DEFAULT => "Default",
                                    LAYER_ENVIRONMENT => "Environment",
                                    LAYER_PROP => "Prop",
                                    LAYER_CHARACTER => "Character",
                                    LAYER_VEHICLE => "Vehicle",
                                    _ => "Custom",
                                })
                                .show_ui(ui, |ui| {
                                    if ui.selectable_value(&mut entity.layer, LAYER_DEFAULT, "Default").changed() { 
                                        inspector_dirty = true;
                                        if entity.collision_enabled && self.is_connected {
                                            let _ = command_tx.send(AppCommand::Send(SceneUpdate::SetCollision {
                                                id: entity.id, enabled: true, layer: entity.layer
                                            }));
                                        }
                                    }
                                    if ui.selectable_value(&mut entity.layer, LAYER_ENVIRONMENT, "Environment").changed() { 
                                        inspector_dirty = true;
                                        if entity.collision_enabled && self.is_connected {
                                            let _ = command_tx.send(AppCommand::Send(SceneUpdate::SetCollision {
                                                id: entity.id, enabled: true, layer: entity.layer
                                            }));
                                        }
                                    }
                                    if ui.selectable_value(&mut entity.layer, LAYER_PROP, "Prop").changed() { 
                                        inspector_dirty = true;
                                        if entity.collision_enabled && self.is_connected {
                                            let _ = command_tx.send(AppCommand::Send(SceneUpdate::SetCollision {
                                                id: entity.id, enabled: true, layer: entity.layer
                                            }));
                                        }
                                    }
                                    if ui.selectable_value(&mut entity.layer, LAYER_CHARACTER, "Character").changed() { 
                                        inspector_dirty = true;
                                        if entity.collision_enabled && self.is_connected {
                                            let _ = command_tx.send(AppCommand::Send(SceneUpdate::SetCollision {
                                                id: entity.id, enabled: true, layer: entity.layer
                                            }));
                                        }
                                    }
                                    if ui.selectable_value(&mut entity.layer, LAYER_VEHICLE, "Vehicle").changed() { 
                                        inspector_dirty = true;
                                        if entity.collision_enabled && self.is_connected {
                                            let _ = command_tx.send(AppCommand::Send(SceneUpdate::SetCollision {
                                                id: entity.id, enabled: true, layer: entity.layer
                                            }));
                                        }
                                    }
                                });
                        });
                        ui.label(format!("ID: {} | Deployed: {}", entity.id, if entity.deployed { "✓" } else { "✗" }));
                        
                        ui.add_space(8.0);
                        ui.label("📍 Transform");
                        let mut transform_dirty = false;
                        ui.vertical(|ui| {
                            ui.horizontal(|ui| {
                                if ui.add(egui::DragValue::new(&mut entity.position[0]).speed(0.1).prefix("X:")).changed() { transform_dirty = true; }
                                if ui.add(egui::DragValue::new(&mut entity.position[1]).speed(0.1).prefix("Y:")).changed() { transform_dirty = true; }
                                if ui.add(egui::DragValue::new(&mut entity.position[2]).speed(0.1).prefix("Z:")).changed() { transform_dirty = true; }
                            });
                            ui.horizontal(|ui| {
                                if ui.add(egui::DragValue::new(&mut entity.scale[0]).speed(0.1).prefix("X:")).changed() { transform_dirty = true; }
                                if ui.add(egui::DragValue::new(&mut entity.scale[1]).speed(0.1).prefix("Y:")).changed() { transform_dirty = true; }
                                if ui.add(egui::DragValue::new(&mut entity.scale[2]).speed(0.1).prefix("Z:")).changed() { transform_dirty = true; }
                            });
                        });
                        
                        if transform_dirty {
                            inspector_dirty = true;
                            if self.is_connected {
                                let _ = command_tx.send(AppCommand::Send(SceneUpdate::UpdateTransform {
                                    id: entity.id,
                                    position: Some(entity.position),
                                    rotation: Some(entity.rotation),
                                    scale: Some(entity.scale),
                                }));
                            }
                        }
                        
                        ui.add_space(8.0);
                        ui.label("🎨 Material");
                        ui.horizontal(|ui| {
                            ui.label("Albedo:");
                            let mut c = egui::Color32::from_rgb(
                                (entity.material.albedo_color[0]*255.0) as u8,
                                (entity.material.albedo_color[1]*255.0) as u8,
                                (entity.material.albedo_color[2]*255.0) as u8);
                            if ui.color_edit_button_srgba(&mut c).changed() {
                                entity.material.albedo_color = [c.r() as f32/255.0, c.g() as f32/255.0, c.b() as f32/255.0];
                                inspector_dirty = true;
                            }
                        });
                        if ui.add(egui::Slider::new(&mut entity.material.metallic, 0.0..=1.0).text("Metallic")).changed() { inspector_dirty = true; }
                        if ui.add(egui::Slider::new(&mut entity.material.roughness, 0.0..=1.0).text("Roughness")).changed() { inspector_dirty = true; }
                        
                        // Albedo Texture Drag & Drop
                        ui.horizontal(|ui| {
                            ui.label("Albedo Texture:");
                            let texture_display_name = entity.material.albedo_texture.as_ref()
                                .and_then(|p| std::path::Path::new(p).file_name())
                                .map(|n| n.to_string_lossy().to_string())
                                .unwrap_or_else(|| "(drop image)".into());
                            
                            // Create a drop zone frame
                            let drop_zone = egui::Frame::none()
                                .stroke(egui::Stroke::new(1.0, egui::Color32::GRAY))
                                .inner_margin(4.0)
                                .show(ui, |ui| {
                                    ui.label(&texture_display_name);
                                });
                            
                            // Warning for missing files
                            if entity.material.albedo_texture.is_some() && entity.material.albedo_texture_data.is_none() {
                                ui.colored_label(egui::Color32::RED, "⚠ File missing on disk");
                                if ui.link("Help").on_hover_text("The editor can't find this texture file. Drag it back into this box to re-import it into the project's assets folder.").clicked() {
                                    // Just show tooltip
                                }
                            }
                            
                            let drop_rect = drop_zone.response.rect;
                            
                            // Check for dropped files anywhere (not just when hovered)
                            let dropped_file = ui.ctx().input(|i| {
                                i.raw.dropped_files.first().and_then(|f| f.path.clone())
                            });
                            
                            // If a file was dropped and mouse is over our drop zone
                            if let Some(path) = dropped_file {
                                let mouse_pos = ui.ctx().input(|i| i.pointer.hover_pos());
                                if let Some(pos) = mouse_pos {
                                    if drop_rect.contains(pos) {
                                        if let Ok(bytes) = std::fs::read(&path) {
                                            if let Some(filename) = path.file_name() {
                                                let dest_path = format!("assets/textures/{}", filename.to_string_lossy());
                                                // Copy to local assets if it's not already there
                                                if path.to_string_lossy() != dest_path {
                                                    let _ = std::fs::copy(&path, &dest_path);
                                                }
                                                
                                                entity.material.albedo_texture = Some(dest_path.clone());
                                                entity.material.albedo_texture_data = Some(bytes.clone());
                                                inspector_dirty = true;
                                                
                                                // Upload immediately if connected
                                                let texture_id = filename.to_string_lossy().to_string();
                                                
                                                // Buffer for editor's rendering cache (apply after scene borrow ends)
                                                texture_to_cache = Some((texture_id.clone(), bytes.clone()));
                                                
                                                let _ = command_tx.send(AppCommand::Send(SceneUpdate::UploadTexture {
                                                    id: texture_id.clone(),
                                                    data: bytes,
                                                }));
                                                
                                                // Link to material
                                                let _ = command_tx.send(AppCommand::Send(SceneUpdate::UpdateMaterial {
                                                    id: entity.id,
                                                    color: None,
                                                    albedo_texture: Some(Some(texture_id)),
                                                    metallic: None,
                                                    roughness: None,
                                                }));
                                            }
                                        }
                                    }
                                }
                            }
                            
                            // Clear button
                            if entity.material.albedo_texture.is_some() {
                                if ui.small_button("❌").clicked() {
                                    entity.material.albedo_texture = None;
                                    entity.material.albedo_texture_data = None;
                                    inspector_dirty = true;
                                    
                                    let _ = command_tx.send(AppCommand::Send(SceneUpdate::UpdateMaterial {
                                        id: entity.id,
                                        color: None,
                                        albedo_texture: Some(None),
                                        metallic: None,
                                        roughness: None,
                                    }));
                                }
                            }
                        });
                        
                        // Real-time material updates for color/PBR
                        if inspector_dirty && self.is_connected {
                            let _ = command_tx.send(AppCommand::Send(SceneUpdate::UpdateMaterial {
                                id: entity.id,
                                color: Some([entity.material.albedo_color[0], entity.material.albedo_color[1], entity.material.albedo_color[2], 1.0]),
                                albedo_texture: None, // Only update color/PBR here
                                metallic: Some(entity.material.metallic),
                                roughness: Some(entity.material.roughness),
                            }));
                        }
                        
                        ui.add_space(8.0);
                        match &mut entity.entity_type {
                            EntityType::CrowdAgent { state, speed } => {
                                ui.label("🚶 Crowd Agent");
                                egui::ComboBox::from_label("State").selected_text(state.as_str()).show_ui(ui, |ui| {
                                    for s in ["Walking", "Running", "Fleeing"] { ui.selectable_value(state, s.into(), s); }
                                });
                                ui.add(egui::Slider::new(speed, 0.0..=15.0).text("Speed"));
                            }
                            EntityType::Building { height } => {
                                ui.label("🏢 Building");
                                if ui.add(egui::Slider::new(height, 2.0..=50.0).text("Height")).changed() {
                                    // Update Y-scale to match height
                                    entity.scale[1] = *height;
                                    // Also adjust Y-position to keep base at ground level (assuming pivot at center)
                                    // Default building position is h/2.0
                                    entity.position[1] = *height / 2.0;
                                    
                                    if self.is_connected {
                                        let _ = command_tx.send(AppCommand::Send(SceneUpdate::UpdateTransform {
                                            id: entity.id,
                                            position: Some(entity.position),
                                            rotation: Some(entity.rotation),
                                            scale: Some(entity.scale),
                                        }));
                                    }
                                }
                            }
                            EntityType::AudioSource { sound_id, volume, looping, max_distance, audio_data } => {
                                ui.label("🔊 Audio Source");
                                ui.horizontal(|ui| { 
                                    ui.label("Sound ID:"); 
                                    let response = ui.text_edit_singleline(sound_id); 
                                    
                                    // Drag & Drop Logic
                                    if response.hovered() {
                                        ui.ctx().input(|i| {
                                            if !i.raw.dropped_files.is_empty() {
                                                if let Some(file) = i.raw.dropped_files.first() {
                                                    if let Some(path) = &file.path {
                                                        if let Ok(bytes) = std::fs::read(path) {
                                                            if let Some(name) = path.file_name() {
                                                                let name_str = name.to_string_lossy().to_string();
                                                                *sound_id = name_str.clone();
                                                                *audio_data = Some(bytes.clone()); // Store for deploy
                                                                
                                                                // Upload to Engine immediately
                                                                let _ = command_tx.send(AppCommand::Send(SceneUpdate::UploadSound {
                                                                    id: name_str,
                                                                    data: bytes,
                                                                }));
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        });
                                        response.on_hover_text("Drag .mp3/.ogg file here to upload");
                                    }
                                });
                                ui.add(egui::Slider::new(volume, 0.0..=1.0).text("Volume"));
                                ui.add(egui::Slider::new(max_distance, 1.0..=100.0).text("Max Dist"));
                                ui.checkbox(looping, "Looping");
                            }
                            EntityType::Primitive(p) => { ui.label(format!("{} {}", p.icon(), p.name())); }
                            EntityType::Camera => { ui.label("🎥 Player Start"); }
                            EntityType::Light { light_type, intensity, range, color } => {
                                let icon = match light_type {
                                    LightTypeEditor::Point => "💡",
                                    LightTypeEditor::Spot => "🔦",
                                    LightTypeEditor::Directional => "☀️",
                                };
                                ui.label(format!("{} {:?} Light", icon, light_type));
                                ui.add(egui::Slider::new(intensity, 0.1..=50.0).text("Intensity"));
                                ui.add(egui::Slider::new(range, 1.0..=100.0).text("Range"));
                                ui.horizontal(|ui| {
                                    ui.label("Color:");
                                    let mut c = egui::Color32::from_rgb(
                                        (color[0] * 255.0) as u8,
                                        (color[1] * 255.0) as u8,
                                        (color[2] * 255.0) as u8);
                                    if ui.color_edit_button_srgba(&mut c).changed() {
                                        color[0] = c.r() as f32 / 255.0;
                                        color[1] = c.g() as f32 / 255.0;
                                        color[2] = c.b() as f32 / 255.0;
                                    }
                                });
                            }
                            _ => {}
                        }
                        
                        ui.add_space(15.0);
                        ui.horizontal(|ui| {
                            if ui.button("🚀 Deploy").clicked() { deploy_id = Some(entity.id); }
                            if ui.button("🗑 Delete").clicked() { delete_id = Some(entity.id); }
                        });
                    }
                }
            } else {
                ui.label("No entity selected");
            }
            
            // Handle actions outside borrow
            if let Some(id) = deploy_id {
                if let Some(scene) = &mut self.current_scene {
                    if let Some(e) = scene.entities.iter_mut().find(|e| e.id == id) {
                        if self.is_connected {
                            match &e.entity_type {
                                EntityType::Camera => {
                                    let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::SetActiveCamera {
                                        id: e.id, fov: e.fov,
                                    }));
                                },
                                EntityType::Mesh { path } => {
                                    // Load file data and send correct spawn command
                                    if let Ok(data) = std::fs::read(path) {
                                        let path_lower = path.to_lowercase();
                                        if path_lower.ends_with(".glb") || path_lower.ends_with(".gltf") {
                                            let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::SpawnGltfMesh {
                                                id: e.id,
                                                mesh_data: data,
                                                position: e.position,
                                                rotation: e.rotation,
                                                scale: e.scale,
                                                collision_enabled: e.collision_enabled,
                                                layer: e.layer,
                                                is_static: e.is_static,
                                            }));
                                        } else {
                                            // Assume FBX/OBJ for SpawnFbxMesh
                                            let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::SpawnFbxMesh {
                                                id: e.id,
                                                mesh_data: data,
                                                position: e.position,
                                                rotation: e.rotation,
                                                scale: e.scale,
                                                albedo_texture: e.material.albedo_texture.clone(),
                                                collision_enabled: e.collision_enabled,
                                                layer: e.layer,
                                                is_static: e.is_static,
                                            }));
                                        }
                                    } else {
                                        self.status = format!("Error: Could not read mesh file {}", path);
                                    }
                                },
                                _ => {
                                    let color = e.material.albedo_color;
                                    let primitive = match &e.entity_type {
                                        EntityType::Primitive(p) => match p {
                                            PrimitiveType::Cube => 0, PrimitiveType::Sphere => 1, PrimitiveType::Cylinder => 2,
                                            PrimitiveType::Plane => 3, PrimitiveType::Capsule => 4, PrimitiveType::Cone => 5,
                                        },
                                        EntityType::CrowdAgent { .. } => 4, _ => 0
                                    };
                                    let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::Spawn {
                                        id: e.id, primitive, position: e.position, rotation: e.rotation, scale: e.scale, color,
                                        albedo_texture: e.material.albedo_texture.clone(),
                                        collision_enabled: e.collision_enabled,
                                        layer: e.layer,
                                        is_static: e.is_static,
                                    }));
                                }
                            }

                            // Always send parenting command if parent exists
                            if let Some(parent_id) = e.parent_id {
                                let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::AttachEntity {
                                    id: e.id,
                                    parent_id: Some(parent_id),
                                }));
                            }

                            e.deployed = true;
                            self.status = format!("Deployed #{}", id);
                        }
                    }
                }
            }
            if let Some(id) = delete_id {
                // Save entity before deletion for undo
                if let Some(scene) = &self.current_scene {
                    if let Some(entity) = scene.entities.iter().find(|e| e.id == id) {
                        self.undo_stack.push(EditorAction::DeleteEntity { entity: entity.clone() });
                        self.redo_stack.clear();
                    }
                }
                if self.is_connected { let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::DeleteEntity { id })); }
                if let Some(scene) = &mut self.current_scene { scene.entities.retain(|e| e.id != id); }
                self.selected_entity_id = None;
                self.scene_dirty = true;
            }
            
            // Apply inspector dirty tracking
            if inspector_dirty {
                self.scene_dirty = true;
            }
            
            // Finally, apply texture cache update after side panel borrow ends
            if let Some((name, data)) = texture_to_cache {
                self.load_egui_texture(ctx, &name, &data);
            }
        });

        // BOTTOM - HIERARCHY
        egui::TopBottomPanel::bottom("hierarchy")
            .resizable(true)
            .default_height(120.0)
            .show(ctx, |ui| {
                ui.heading("Hierarchy");
                egui::ScrollArea::horizontal().show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        if let Some(scene) = &self.current_scene {
                            for entity in &scene.entities {
                                let icon = match &entity.entity_type {
                                    EntityType::Primitive(p) => p.icon(),
                                    EntityType::Vehicle => "🚗",
                                    EntityType::CrowdAgent { .. } => "🚶",
                                    EntityType::Building { .. } => "🏢",
                                    EntityType::Ground => "🌍",
                                    EntityType::Mesh { .. } => "📦",
                                    EntityType::Camera => "🎥",
                                    EntityType::AudioSource { .. } => "🔊",
                                    EntityType::Light { light_type, .. } => match light_type {
                                        LightTypeEditor::Point => "💡",
                                        LightTypeEditor::Spot => "🔦",
                                        LightTypeEditor::Directional => "☀️",
                                    },
                                };
                                let deployed = if entity.deployed { "✓" } else { "" };
                                if ui
                                    .selectable_label(
                                        self.selected_entity_id == Some(entity.id),
                                        format!("{}{} {}", icon, deployed, entity.name),
                                    )
                                    .clicked()
                                {
                                    self.selected_entity_id = Some(entity.id);
                                }
                            }
                        }
                    });
                });
            });

        // CENTER - 3D VIEWPORT with TABS
        egui::CentralPanel::default().show(ctx, |ui| {
            // Tab bar at top
            ui.horizontal(|ui| {
                if ui.selectable_label(
                    self.active_view == EditorView::SceneEditor,
                    "🎬 Scene Editor"
                ).clicked() {
                    self.active_view = EditorView::SceneEditor;
                }
                if ui.selectable_label(
                    self.active_view == EditorView::AnimationEditor,
                    "🎭 Animation Editor"
                ).clicked() {
                    self.active_view = EditorView::AnimationEditor;
                }
            });
            ui.separator();
            
            // Conditional content based on active tab
            match self.active_view {
                EditorView::SceneEditor => {
                    ui.heading("3D Scene Viewport");
                    let _ = self.draw_3d_viewport(ui);
                }
                EditorView::AnimationEditor => {
                    self.draw_animation_editor(ui);
                }
            }
        });

        // New Scene Dialog
        if self.show_new_scene_dialog {
            egui::Window::new("New Scene")
                .collapsible(false)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Name:");
                        ui.text_edit_singleline(&mut self.new_scene_name);
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Create").clicked() && !self.new_scene_name.is_empty() {
                            self.scenes.push(self.new_scene_name.clone());
                            self.current_scene = Some(Scene::new(&self.new_scene_name));
                            self.current_scene_path = None;
                            self.scene_dirty = false;
                            self.selected_scene_idx = Some(self.scenes.len() - 1);
                            self.selected_entity_id = None;
                            self.new_scene_name.clear();
                            self.show_new_scene_dialog = false;
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_new_scene_dialog = false;
                        }
                    });
                });
        }

        // Save Scene As Dialog
        if self.show_save_as_dialog {
            egui::Window::new("Save Scene As")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("File path:");
                        ui.text_edit_singleline(&mut self.save_as_path);
                    });
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("💾 Save").clicked() {
                            let path = self.save_as_path.clone();
                            // Ensure .json extension
                            let path = if !path.ends_with(".json") {
                                format!("{}.json", path)
                            } else {
                                path
                            };
                            if let Err(e) = self.save_scene_to_path(&path) {
                                self.status = format!("Error: {}", e);
                            }
                            self.show_save_as_dialog = false;
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_save_as_dialog = false;
                        }
                    });
                });
        }

        // Open Scene Dialog
        if self.show_open_scene_dialog {
            egui::Window::new("Open Scene")
                .collapsible(false)
                .resizable(true)
                .min_width(400.0)
                .show(ctx, |ui| {
                    ui.label("Select a scene file to open:");
                    ui.add_space(4.0);
                    
                    egui::ScrollArea::vertical()
                        .max_height(300.0)
                        .show(ui, |ui| {
                            let files_clone = self.open_scene_files.clone();
                            for file in &files_clone {
                                let display_name = std::path::Path::new(file)
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or(file);
                                if ui.selectable_label(false, format!("📄 {}", display_name)).clicked() {
                                    let file_path = file.clone();
                                    if self.check_unsaved_and_set_action(PendingAction::OpenScene(file_path.clone())) {
                                        if let Err(e) = self.load_scene_from_path(ctx, &file_path) {
                                            self.status = format!("Error: {}", e);
                                        }
                                    }
                                    self.show_open_scene_dialog = false;
                                }
                            }
                            if files_clone.is_empty() {
                                ui.label("No scene files found in 'scenes/' directory.");
                            }
                        });
                    
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("🔄 Refresh").clicked() {
                            self.scan_scene_files();
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_open_scene_dialog = false;
                        }
                    });
                });
        }

        // Import 3D Model Dialog
        if self.show_import_model_dialog {
            egui::Window::new("📦 Import 3D Model")
                .collapsible(false)
                .resizable(true)
                .min_width(450.0)
                .show(ctx, |ui| {
                    ui.label("Select a 3D model file to import:");
                    ui.label("Supported formats: .obj (FBX files should be converted to OBJ using Blender)");
                    ui.add_space(4.0);
                    
                    egui::ScrollArea::vertical()
                        .max_height(300.0)
                        .show(ui, |ui| {
                            let files_clone = self.import_model_files.clone();
                            for file in &files_clone {
                                let display_name = std::path::Path::new(file)
                                    .file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or(file);
                                let icon = if file.ends_with(".fbx") { "⚠️" } else { "📦" };
                                let tooltip = if file.ends_with(".fbx") {
                                    "FBX files must be converted to OBJ format using Blender"
                                } else {
                                    "Click to import this model"
                                };
                                if ui.selectable_label(false, format!("{} {}", icon, display_name))
                                    .on_hover_text(tooltip)
                                    .clicked() 
                                {
                                    if file.ends_with(".fbx") {
                                        self.status = "FBX files must be converted to OBJ using Blender. File → Export → Wavefront (.obj)".into();
                                    } else {
                                        let file_path = file.clone();
                                        if let Err(e) = self.import_model_from_path(ctx, &file_path) {
                                            self.status = format!("Import error: {}", e);
                                        }
                                        self.show_import_model_dialog = false;
                                    }
                                }
                            }
                            if files_clone.is_empty() {
                                ui.label("No model files found.");
                                ui.label("Place .obj files in: assets/models/, models/, or project root.");
                            }
                        });
                    
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("🔄 Refresh").clicked() {
                            self.import_model_files = Self::scan_model_files();
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_import_model_dialog = false;
                        }
                    });
                });
        }

        // UI Editor Window
        if self.show_ui_editor {
            let mut ui_dirty = false;
            let mut open = true;
            
            egui::Window::new("🖼 UI Editor")
                .default_size([800.0, 500.0])
                .min_width(600.0)
                .min_height(350.0)
                .max_width(1200.0)
                .max_height(800.0)
                .resizable(true)
                .open(&mut open)
                .show(ctx, |ui| {
                    // Ensure at least one layout exists before any UI code
                    if let Some(scene) = &mut self.current_scene {
                        if scene.ui_layouts.is_empty() {
                            scene.ui_layouts.push(NamedUiLayout::default());
                        }
                        // Ensure selected_layout_idx is valid
                        if self.selected_layout_idx >= scene.ui_layouts.len() {
                            self.selected_layout_idx = 0;
                        }
                    }
                    
                    // Tab bar for layout tabs
                    ui.horizontal(|ui| {
                        if let Some(scene) = &mut self.current_scene {
                            let mut tab_to_remove = None;
                            for (idx, layout) in scene.ui_layouts.iter().enumerate() {
                                let selected = self.selected_layout_idx == idx;
                                let tab_text = format!("{}", layout.name);
                                if ui.selectable_label(selected, &tab_text).clicked() {
                                    self.selected_layout_idx = idx;
                                    self.selected_ui_id = None;
                                    self.selected_ui_element_type = None;
                                }
                                // Right-click to delete tab (if more than one)
                                if scene.ui_layouts.len() > 1 {
                                    if ui.small_button("✕").clicked() {
                                        tab_to_remove = Some(idx);
                                    }
                                }
                            }
                            // Add new tab button
                            if ui.button("➕").on_hover_text("Add new layout").clicked() {
                                let new_idx = scene.ui_layouts.len();
                                scene.ui_layouts.push(NamedUiLayout {
                                    name: format!("Layout {}", new_idx + 1),
                                    alias: format!("layout_{}", new_idx + 1),
                                    layer_type: stfsc_engine::ui::UiLayerType::InGameOverlay,
                                    layout: stfsc_engine::ui::UiLayout::default(),
                                });
                                self.selected_layout_idx = new_idx;
                            }
                            
                            if let Some(idx) = tab_to_remove {
                                scene.ui_layouts.remove(idx);
                                if self.selected_layout_idx >= scene.ui_layouts.len() && self.selected_layout_idx > 0 {
                                    self.selected_layout_idx -= 1;
                                }
                                ui_dirty = true;
                            }
                        }
                    });
                    
                    ui.separator();
                    
                    // Layout name and alias editor
                    ui.horizontal(|ui| {
                        if let Some(scene) = &mut self.current_scene {
                            if let Some(layout) = scene.ui_layouts.get_mut(self.selected_layout_idx) {
                                ui.label("Name:");
                                if ui.text_edit_singleline(&mut layout.name).changed() {
                                    ui_dirty = true;
                                }
                                ui.separator();
                                ui.label("Alias:");
                                if ui.text_edit_singleline(&mut layout.alias).changed() {
                                    ui_dirty = true;
                                }
                                ui.label(egui::RichText::new("→ menu_load(\"").monospace());
                                ui.label(egui::RichText::new(&layout.alias).strong());
                                ui.label(egui::RichText::new("\")").monospace());
                            }
                        }
                    });
                    
                    // Layer type selector
                    ui.horizontal(|ui| {
                        if let Some(scene) = &mut self.current_scene {
                            if let Some(layout) = scene.ui_layouts.get_mut(self.selected_layout_idx) {
                                ui.label("Layer Type:");
                                let current_type_name = match layout.layer_type {
                                    stfsc_engine::ui::UiLayerType::MainMenu => "Main Menu",
                                    stfsc_engine::ui::UiLayerType::PauseOverlay => "Pause Overlay",
                                    stfsc_engine::ui::UiLayerType::IntermediateMenu => "Intermediate Menu",
                                    stfsc_engine::ui::UiLayerType::InGameOverlay => "In-Game Overlay",
                                    stfsc_engine::ui::UiLayerType::Popup => "Popup",
                                };
                                egui::ComboBox::from_id_source("layer_type_combo")
                                    .selected_text(current_type_name)
                                    .show_ui(ui, |ui| {
                                        if ui.selectable_label(layout.layer_type == stfsc_engine::ui::UiLayerType::InGameOverlay, "In-Game Overlay").clicked() {
                                            layout.layer_type = stfsc_engine::ui::UiLayerType::InGameOverlay;
                                            ui_dirty = true;
                                        }
                                        if ui.selectable_label(layout.layer_type == stfsc_engine::ui::UiLayerType::PauseOverlay, "Pause Overlay").clicked() {
                                            layout.layer_type = stfsc_engine::ui::UiLayerType::PauseOverlay;
                                            ui_dirty = true;
                                        }
                                        if ui.selectable_label(layout.layer_type == stfsc_engine::ui::UiLayerType::IntermediateMenu, "Intermediate Menu").clicked() {
                                            layout.layer_type = stfsc_engine::ui::UiLayerType::IntermediateMenu;
                                            ui_dirty = true;
                                        }
                                        if ui.selectable_label(layout.layer_type == stfsc_engine::ui::UiLayerType::MainMenu, "Main Menu").clicked() {
                                            layout.layer_type = stfsc_engine::ui::UiLayerType::MainMenu;
                                            ui_dirty = true;
                                        }
                                        if ui.selectable_label(layout.layer_type == stfsc_engine::ui::UiLayerType::Popup, "Popup").clicked() {
                                            layout.layer_type = stfsc_engine::ui::UiLayerType::Popup;
                                            ui_dirty = true;
                                        }
                                    });
                                // Show behavior hints
                                let hint = match layout.layer_type {
                                    stfsc_engine::ui::UiLayerType::MainMenu => "⚡ Blocks input, loads before scene",
                                    stfsc_engine::ui::UiLayerType::PauseOverlay => "⏸ Pauses game, blocks input",
                                    stfsc_engine::ui::UiLayerType::IntermediateMenu => "📑 For nested menus (settings)",
                                    stfsc_engine::ui::UiLayerType::InGameOverlay => "🎮 Always visible during gameplay",
                                    stfsc_engine::ui::UiLayerType::Popup => "💬 Keybind-triggered, transient",
                                };
                                ui.label(egui::RichText::new(hint).weak().italics());
                            }
                        }
                    });
                    
                    ui.separator();

                    // Toolbar
                    ui.horizontal(|ui| {
                        let mut pending_action = None;
                        if ui.button("➕ Button").clicked() {
                           if let Some(scene) = &mut self.current_scene {
                               if let Some(named_layout) = scene.ui_layouts.get_mut(self.selected_layout_idx) {
                                   let id = (named_layout.layout.buttons.len() + named_layout.layout.panels.len() + named_layout.layout.texts.len()) as u32 + 1;
                                   let btn = stfsc_engine::ui::Button::new(id, "New Button", 100.0, 100.0, 200.0, 50.0);
                                   pending_action = Some(EditorAction::AddUiElement {
                                       layout_idx: self.selected_layout_idx,
                                       element_type: "Button".into(),
                                       element: UiElementHistory::Button(btn.clone()),
                                   });
                                   named_layout.layout.buttons.push(btn);
                                   ui_dirty = true;
                               }
                           }
                        }
                        if let Some(a) = pending_action.take() { self.push_undo_action(a); }

                        if ui.button("➕ Panel").clicked() {
                            if let Some(scene) = &mut self.current_scene {
                                if let Some(named_layout) = scene.ui_layouts.get_mut(self.selected_layout_idx) {
                                    let panel = stfsc_engine::ui::Panel::new(100.0, 100.0, 200.0, 200.0);
                                    pending_action = Some(EditorAction::AddUiElement {
                                        layout_idx: self.selected_layout_idx,
                                        element_type: "Panel".into(),
                                        element: UiElementHistory::Panel(panel.clone()),
                                    });
                                    named_layout.layout.panels.push(panel);
                                    ui_dirty = true;
                                }
                            }
                        }
                        if let Some(a) = pending_action.take() { self.push_undo_action(a); }

                        if ui.button("➕ Text").clicked() {
                            if let Some(scene) = &mut self.current_scene {
                                if let Some(named_layout) = scene.ui_layouts.get_mut(self.selected_layout_idx) {
                                    let text = stfsc_engine::ui::Text::new("New Text", 100.0, 100.0);
                                    pending_action = Some(EditorAction::AddUiElement {
                                        layout_idx: self.selected_layout_idx,
                                        element_type: "Text".into(),
                                        element: UiElementHistory::Text(text.clone()),
                                    });
                                    named_layout.layout.texts.push(text);
                                    ui_dirty = true;
                                }
                            }
                        }
                        if let Some(a) = pending_action.take() { self.push_undo_action(a); }

                        ui.separator();
                        if ui.button("🗑 Clear All").clicked() {
                            if let Some(scene) = &mut self.current_scene {
                                if let Some(named_layout) = scene.ui_layouts.get_mut(self.selected_layout_idx) {
                                    let before = named_layout.clone();
                                    named_layout.layout = stfsc_engine::ui::UiLayout::default();
                                    pending_action = Some(EditorAction::ModifyUiLayout {
                                        layout_idx: self.selected_layout_idx,
                                        before,
                                        after: named_layout.clone(),
                                    });
                                    self.selected_ui_id = None;
                                    ui_dirty = true;
                                }
                            }
                        }
                        if let Some(a) = pending_action.take() { self.push_undo_action(a); }

                        egui::menu::menu_button(ui, "📋 Templates", |ui| {
                            let mut template_action = None;
                            if ui.button("🎮 HUD (Health/Ammo)").clicked() {
                                if let Some(scene) = &mut self.current_scene {
                                    if let Some(nl) = scene.ui_layouts.get_mut(self.selected_layout_idx) {
                                        let before = nl.clone();
                                        nl.layout = stfsc_engine::ui::UiLayout::default();
                                        let mut health_panel = stfsc_engine::ui::Panel::new(20.0, 20.0, 200.0, 30.0);
                                        health_panel.color = [0.8, 0.2, 0.2, 0.8];
                                        nl.layout.panels.push(health_panel);
                                        let mut health_text = stfsc_engine::ui::Text::new("HEALTH: 100", 30.0, 27.0);
                                        health_text.font_size = 20.0;
                                        nl.layout.texts.push(health_text);
                                        let mut ammo_panel = stfsc_engine::ui::Panel::new(1700.0, 980.0, 180.0, 60.0);
                                        ammo_panel.color = [0.15, 0.15, 0.2, 0.8];
                                        nl.layout.panels.push(ammo_panel);
                                        let mut ammo_text = stfsc_engine::ui::Text::new("30 / 120", 1720.0, 1000.0);
                                        ammo_text.font_size = 28.0;
                                        nl.layout.texts.push(ammo_text);
                                        template_action = Some(EditorAction::ModifyUiLayout {
                                            layout_idx: self.selected_layout_idx,
                                            before,
                                            after: nl.clone(),
                                        });
                                        ui_dirty = true;
                                    }
                                }
                                ui.close_menu();
                            }
                            if ui.button("⏸ Pause Menu").clicked() {
                                if let Some(scene) = &mut self.current_scene {
                                    if let Some(nl) = scene.ui_layouts.get_mut(self.selected_layout_idx) {
                                        let before = nl.clone();
                                        nl.layout = stfsc_engine::ui::UiLayout::default();
                                        let mut overlay = stfsc_engine::ui::Panel::new(0.0, 0.0, 1920.0, 1080.0);
                                        overlay.color = [0.0, 0.0, 0.0, 0.7];
                                        nl.layout.panels.push(overlay);
                                        let mut title = stfsc_engine::ui::Text::new("PAUSED", 960.0, 200.0);
                                        title.font_size = 64.0;
                                        nl.layout.texts.push(title);
                                        for (i, (label, cb)) in [("RESUME", "on_resume"), ("SETTINGS", "on_settings"), ("QUIT", "on_quit")].into_iter().enumerate() {
                                            let mut btn = stfsc_engine::ui::Button::new(i as u32 + 1, label, 810.0, 350.0 + i as f32 * 100.0, 300.0, 70.0);
                                            btn.on_click = Some(cb.to_string());
                                            nl.layout.buttons.push(btn);
                                        }
                                        template_action = Some(EditorAction::ModifyUiLayout {
                                            layout_idx: self.selected_layout_idx,
                                            before,
                                            after: nl.clone(),
                                        });
                                        ui_dirty = true;
                                    }
                                }
                                ui.close_menu();
                            }
                            if ui.button("🏠 Main Menu").clicked() {
                                if let Some(scene) = &mut self.current_scene {
                                    if let Some(nl) = scene.ui_layouts.get_mut(self.selected_layout_idx) {
                                        let before = nl.clone();
                                        nl.layout = stfsc_engine::ui::UiLayout::default();
                                        let mut header = stfsc_engine::ui::Panel::new(0.0, 0.0, 1920.0, 80.0);
                                        header.color = [0.1, 0.1, 0.15, 0.9];
                                        nl.layout.panels.push(header);
                                        let mut title = stfsc_engine::ui::Text::new("556 DOWNTOWN", 960.0, 40.0);
                                        title.font_size = 48.0;
                                        nl.layout.texts.push(title);
                                        for (i, (label, cb)) in [("PLAY", "on_play"), ("SETTINGS", "on_settings"), ("QUIT", "on_quit")].into_iter().enumerate() {
                                            let mut btn = stfsc_engine::ui::Button::new(i as u32 + 1, label, 810.0, 400.0 + i as f32 * 80.0, 300.0, 60.0);
                                            btn.on_click = Some(cb.to_string());
                                            nl.layout.buttons.push(btn);
                                        }
                                        template_action = Some(EditorAction::ModifyUiLayout {
                                            layout_idx: self.selected_layout_idx,
                                            before,
                                            after: nl.clone(),
                                        });
                                        ui_dirty = true;
                                    }
                                }
                                ui.close_menu();
                            }
                            // Push template undo if any applied
                            if let Some(a) = template_action { self.push_undo_action(a); }
                        });
                        ui.separator();
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("🚀 Push to Engine").clicked() {
                                if let Some(scene) = &self.current_scene {
                                    if let Some(nl) = scene.ui_layouts.get(self.selected_layout_idx) {
                                        // Determine the correct UiLayer based on layer_type
                                        let layer = match nl.layer_type {
                                            stfsc_engine::ui::UiLayerType::PauseOverlay => stfsc_engine::ui::UiLayer::PauseMenu,
                                            stfsc_engine::ui::UiLayerType::MainMenu => stfsc_engine::ui::UiLayer::MainMenu,
                                            stfsc_engine::ui::UiLayerType::InGameOverlay => stfsc_engine::ui::UiLayer::Hud,
                                            _ => stfsc_engine::ui::UiLayer::Custom(nl.alias.clone()),
                                        };
                                        // IMPORTANT: Sync the layer_type to the inner UiLayout before sending
                                        let mut layout_to_send = nl.layout.clone();
                                        layout_to_send.layer_type = nl.layer_type;
                                        let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::SetUiLayer {
                                            layer,
                                            layout: layout_to_send,
                                        }));
                                    }
                                }
                            }
                        });
                    });

                    ui.separator();

                    // Main content: fixed sizes to prevent window growth
                    let available = ui.available_size();
                    let canvas_width = (available.x * 0.68).min(700.0).max(300.0);
                    let props_width = (available.x - canvas_width - 20.0).min(250.0).max(150.0);

                    ui.horizontal(|ui| {
                        // Canvas area with fixed max size
                        ui.vertical(|ui| {
                            ui.set_width(canvas_width);
                            let aspect = 16.0 / 9.0;
                            let canvas_height = (canvas_width / aspect).min(350.0);
                            let canvas_size = egui::vec2(canvas_width, canvas_height);

                            let (response, painter) = ui.allocate_painter(canvas_size, egui::Sense::click_and_drag());
                            let rect = response.rect;
                            
                            // Dark background with grid
                            painter.rect_filled(rect, 4.0, egui::Color32::from_gray(20));
                            let scale = canvas_size.x / 1920.0;

                            // Draw grid lines
                            for i in 0..=4 {
                                let x = rect.min.x + (i as f32 * canvas_size.x / 4.0);
                                painter.line_segment([egui::pos2(x, rect.min.y), egui::pos2(x, rect.max.y)], egui::Stroke::new(1.0, egui::Color32::from_gray(40)));
                            }
                            for i in 0..=3 {
                                let y = rect.min.y + (i as f32 * canvas_size.y / 3.0);
                                painter.line_segment([egui::pos2(rect.min.x, y), egui::pos2(rect.max.x, y)], egui::Stroke::new(1.0, egui::Color32::from_gray(40)));
                            }

                            // Track what element is under cursor for direct drag
                            let pointer = response.interact_pointer_pos().unwrap_or(egui::pos2(-1000.0, -1000.0));
                            let mut clicked_element: Option<(String, u32)> = None;
                            let mut drag_target: Option<(String, u32)> = None;

                            if let Some(scene) = &self.current_scene {
                              if let Some(nl) = scene.ui_layouts.get(self.selected_layout_idx) {
                                // 1. Check Texts (usually on top)
                                for (i, text) in nl.layout.texts.iter().enumerate().rev() {
                                    let pos = egui::pos2(rect.min.x + text.position[0] * scale, rect.min.y + text.position[1] * scale);
                                    let galley = painter.layout_no_wrap(text.content.clone(), egui::FontId::proportional(text.font_size * scale), egui::Color32::WHITE);
                                    let elem_rect = egui::Rect::from_min_size(pos, galley.size());
                                    if elem_rect.contains(pointer) {
                                        if response.drag_started() { 
                                            drag_target = Some(("Text".into(), i as u32));
                                            self.ui_drag_start_element = Some(UiElementHistory::Text(text.clone()));
                                        }
                                        if response.clicked() { clicked_element = Some(("Text".into(), i as u32)); }
                                        break;
                                    }
                                }

                                // 2. Check Layout Buttons
                                if clicked_element.is_none() && drag_target.is_none() {
                                    for (i, button) in nl.layout.buttons.iter().enumerate().rev() {
                                        let pos = egui::pos2(rect.min.x + button.panel.position[0] * scale, rect.min.y + button.panel.position[1] * scale);
                                        let size = egui::vec2(button.panel.size[0] * scale, button.panel.size[1] * scale);
                                        let elem_rect = egui::Rect::from_min_size(pos, size);
                                        if elem_rect.contains(pointer) {
                                            if response.drag_started() { 
                                                drag_target = Some(("Button".into(), i as u32));
                                                self.ui_drag_start_element = Some(UiElementHistory::Button(button.clone()));
                                            }
                                            if response.clicked() { clicked_element = Some(("Button".into(), i as u32)); }
                                            break;
                                        }
                                    }
                                }

                                // 3. Check Panels and their children
                                if clicked_element.is_none() && drag_target.is_none() {
                                    for (i, panel) in nl.layout.panels.iter().enumerate().rev() {
                                        let panel_pos = egui::pos2(rect.min.x + panel.position[0] * scale, rect.min.y + panel.position[1] * scale);
                                        
                                        // Check children first (they are on top of the panel)
                                        let mut child_found = false;
                                        for (ci, child) in panel.children.iter().enumerate().rev() {
                                            match child {
                                                stfsc_engine::ui::PanelChild::Button(btn) => {
                                                    let pos = egui::pos2(panel_pos.x + btn.panel.position[0] * scale, panel_pos.y + btn.panel.position[1] * scale);
                                                    let size = egui::vec2(btn.panel.size[0] * scale, btn.panel.size[1] * scale);
                                                    let elem_rect = egui::Rect::from_min_size(pos, size);
                                                    if elem_rect.contains(pointer) {
                                                        if response.drag_started() { 
                                                            drag_target = Some(("PanelChild".into(), i as u32));
                                                            self.selected_child_idx = Some(ci);
                                                            // We store the whole panel for undo since children are part of it
                                                            self.ui_drag_start_element = Some(UiElementHistory::Panel(panel.clone()));
                                                        }
                                                        if response.clicked() { 
                                                            clicked_element = Some(("PanelChild".into(), i as u32));
                                                            self.selected_child_idx = Some(ci);
                                                        }
                                                        child_found = true;
                                                        break;
                                                    }
                                                }
                                                stfsc_engine::ui::PanelChild::Text(txt) => {
                                                    let pos = egui::pos2(panel_pos.x + txt.position[0] * scale, panel_pos.y + txt.position[1] * scale);
                                                    let galley = painter.layout_no_wrap(txt.content.clone(), egui::FontId::proportional(txt.font_size * scale), egui::Color32::WHITE);
                                                    let elem_rect = egui::Rect::from_min_size(pos, galley.size());
                                                    if elem_rect.contains(pointer) {
                                                        if response.drag_started() { 
                                                            drag_target = Some(("PanelChild".into(), i as u32));
                                                            self.selected_child_idx = Some(ci);
                                                            self.ui_drag_start_element = Some(UiElementHistory::Panel(panel.clone()));
                                                        }
                                                        if response.clicked() { 
                                                            clicked_element = Some(("PanelChild".into(), i as u32));
                                                            self.selected_child_idx = Some(ci);
                                                        }
                                                        child_found = true;
                                                        break;
                                                    }
                                                }
                                            }
                                        }

                                        if child_found { break; }

                                        // Finally check the panel itself
                                        let size = egui::vec2(panel.size[0] * scale, panel.size[1] * scale);
                                        let panel_rect = egui::Rect::from_min_size(panel_pos, size);
                                        if panel_rect.contains(pointer) {
                                            if response.drag_started() { 
                                                drag_target = Some(("Panel".into(), i as u32));
                                                self.ui_drag_start_element = Some(UiElementHistory::Panel(panel.clone()));
                                            }
                                            if response.clicked() { clicked_element = Some(("Panel".into(), i as u32)); }
                                            break;
                                        }
                                    }
                                }
                              }
                            }

                            // Handle selection on click
                            if let Some((etype, id)) = clicked_element {
                                self.selected_ui_element_type = Some(etype);
                                self.selected_ui_id = Some(id);
                            }

                            // Handle drag - select and move simultaneously
                            if let Some((etype, id)) = drag_target {
                                self.selected_ui_element_type = Some(etype);
                                self.selected_ui_id = Some(id);
                            }

                            // Apply drag delta to selected element
                            if response.dragged() {
                                if let (Some(etype), Some(id), Some(scene)) = (&self.selected_ui_element_type, self.selected_ui_id, &mut self.current_scene) {
                                    if let Some(nl) = scene.ui_layouts.get_mut(self.selected_layout_idx) {
                                        let delta = response.drag_delta() / scale;
                                        match etype.as_str() {
                                            "Button" => if let Some(btn) = nl.layout.buttons.get_mut(id as usize) {
                                                btn.panel.position[0] += delta.x;
                                                btn.panel.position[1] += delta.y;
                                                ui_dirty = true;
                                            }
                                            "Panel" => if let Some(p) = nl.layout.panels.get_mut(id as usize) {
                                                p.position[0] += delta.x;
                                                p.position[1] += delta.y;
                                                ui_dirty = true;
                                            }
                                            "Text" => if let Some(t) = nl.layout.texts.get_mut(id as usize) {
                                                t.position[0] += delta.x;
                                                t.position[1] += delta.y;
                                                ui_dirty = true;
                                            }
                                            "PanelChild" => if let (Some(p), Some(ci)) = (nl.layout.panels.get_mut(id as usize), self.selected_child_idx) {
                                                if let Some(child) = p.children.get_mut(ci) {
                                                    match child {
                                                        stfsc_engine::ui::PanelChild::Button(btn) => {
                                                            btn.panel.position[0] += delta.x;
                                                            btn.panel.position[1] += delta.y;
                                                        }
                                                        stfsc_engine::ui::PanelChild::Text(txt) => {
                                                            txt.position[0] += delta.x;
                                                            txt.position[1] += delta.y;
                                                        }
                                                    }
                                                    ui_dirty = true;
                                                }
                                            }
                                            _ => {}
                                        }
                                    }
                                }
                            }

                            // Push undo on drag release
                            if response.drag_released() {
                                if let (Some(before), Some(etype), Some(id), Some(scene)) = (self.ui_drag_start_element.take(), &self.selected_ui_element_type, self.selected_ui_id, &mut self.current_scene) {
                                    if let Some(nl) = scene.ui_layouts.get(self.selected_layout_idx) {
                                        let after = match etype.as_str() {
                                            "Button" => nl.layout.buttons.get(id as usize).map(|b| UiElementHistory::Button(b.clone())),
                                            "Panel" | "PanelChild" => nl.layout.panels.get(id as usize).map(|p| UiElementHistory::Panel(p.clone())),
                                            "Text" => nl.layout.texts.get(id as usize).map(|t| UiElementHistory::Text(t.clone())),
                                            _ => None,
                                        };

                                        if let Some(after_state) = after {
                                            // Only push if something changed (rough check via serialization/clonability is hard, 
                                            // we just push for now like with 3D objects)
                                            self.push_undo_action(EditorAction::ModifyUiElement {
                                                layout_idx: self.selected_layout_idx,
                                                element_type: etype.clone(),
                                                element_idx: id as usize,
                                                before,
                                                after: after_state,
                                            });
                                        }
                                    }
                                }
                            }

                            // Now draw all elements
                            if let Some(scene) = &self.current_scene {
                              if let Some(nl) = scene.ui_layouts.get(self.selected_layout_idx) {
                                // Draw panels
                                for (i, panel) in nl.layout.panels.iter().enumerate() {
                                    let pos = egui::pos2(rect.min.x + panel.position[0] * scale, rect.min.y + panel.position[1] * scale);
                                    let size = egui::vec2(panel.size[0] * scale, panel.size[1] * scale);
                                    let panel_rect = egui::Rect::from_min_size(pos, size);
                                    let color = egui::Color32::from_rgba_unmultiplied(
                                        (panel.color[0] * 255.0) as u8, (panel.color[1] * 255.0) as u8,
                                        (panel.color[2] * 255.0) as u8, (panel.color[3] * 255.0) as u8,
                                    );
                                    painter.rect_filled(panel_rect, 0.0, color);
                                    if self.selected_ui_id == Some(i as u32) && self.selected_ui_element_type.as_deref() == Some("Panel") {
                                        painter.rect_stroke(panel_rect, 0.0, egui::Stroke::new(2.0, egui::Color32::YELLOW));
                                    }
                                    
                                    // Draw child elements inside panel
                                    for (ci, child) in panel.children.iter().enumerate() {
                                        match child {
                                            stfsc_engine::ui::PanelChild::Button(btn) => {
                                                let child_pos = egui::pos2(
                                                    pos.x + btn.panel.position[0] * scale,
                                                    pos.y + btn.panel.position[1] * scale
                                                );
                                                let child_size = egui::vec2(btn.panel.size[0] * scale, btn.panel.size[1] * scale);
                                                let child_rect = egui::Rect::from_min_size(child_pos, child_size);
                                                painter.rect_filled(child_rect, 4.0, egui::Color32::from_rgb(60, 60, 80));
                                                painter.text(child_rect.center(), egui::Align2::CENTER_CENTER, &btn.label.content, egui::FontId::proportional(12.0 * scale), egui::Color32::WHITE);
                                                // Highlight if this child is selected
                                                if self.selected_ui_id == Some(i as u32) && self.selected_ui_element_type.as_deref() == Some("PanelChild") && self.selected_child_idx == Some(ci) {
                                                    painter.rect_stroke(child_rect, 4.0, egui::Stroke::new(2.0, egui::Color32::from_rgb(0, 200, 255)));
                                                }
                                            }
                                            stfsc_engine::ui::PanelChild::Text(txt) => {
                                                let child_pos = egui::pos2(
                                                    pos.x + txt.position[0] * scale,
                                                    pos.y + txt.position[1] * scale
                                                );
                                                let txt_color = egui::Color32::from_rgba_unmultiplied(
                                                    (txt.color[0] * 255.0) as u8, (txt.color[1] * 255.0) as u8,
                                                    (txt.color[2] * 255.0) as u8, (txt.color[3] * 255.0) as u8,
                                                );
                                                let galley = painter.layout_no_wrap(txt.content.clone(), egui::FontId::proportional(txt.font_size * scale), txt_color);
                                                painter.galley(child_pos, galley);
                                            }
                                        }
                                    }
                                }
                                // Draw buttons
                                for (i, button) in nl.layout.buttons.iter().enumerate() {
                                    let pos = egui::pos2(rect.min.x + button.panel.position[0] * scale, rect.min.y + button.panel.position[1] * scale);
                                    let size = egui::vec2(button.panel.size[0] * scale, button.panel.size[1] * scale);
                                    let btn_rect = egui::Rect::from_min_size(pos, size);
                                    painter.rect_filled(btn_rect, 4.0, egui::Color32::from_rgb(60, 60, 80));
                                    painter.text(btn_rect.center(), egui::Align2::CENTER_CENTER, &button.label.content, egui::FontId::proportional(14.0 * scale), egui::Color32::WHITE);
                                    if self.selected_ui_id == Some(i as u32) && self.selected_ui_element_type.as_deref() == Some("Button") {
                                        painter.rect_stroke(btn_rect, 4.0, egui::Stroke::new(2.0, egui::Color32::YELLOW));
                                    }
                                }
                                // Draw texts
                                for (i, text) in nl.layout.texts.iter().enumerate() {
                                    let pos = egui::pos2(rect.min.x + text.position[0] * scale, rect.min.y + text.position[1] * scale);
                                    let color = egui::Color32::from_rgba_unmultiplied(
                                        (text.color[0] * 255.0) as u8, (text.color[1] * 255.0) as u8,
                                        (text.color[2] * 255.0) as u8, (text.color[3] * 255.0) as u8,
                                    );
                                    let galley = painter.layout_no_wrap(text.content.clone(), egui::FontId::proportional(text.font_size * scale), color);
                                    let text_rect = egui::Rect::from_min_size(pos, galley.size());
                                    painter.galley(pos, galley);
                                    if self.selected_ui_id == Some(i as u32) && self.selected_ui_element_type.as_deref() == Some("Text") {
                                        painter.rect_stroke(text_rect, 0.0, egui::Stroke::new(1.0, egui::Color32::YELLOW));
                                    }
                                }
                              }
                            }
                        });

                        ui.separator();

                                // Properties panel
                                ui.vertical(|ui| {
                                    ui.set_width(props_width);
                                    egui::ScrollArea::vertical().show(ui, |ui| {
                                        ui.heading("Properties");
                                        
                                        let mut pending_undo = None;
                                        if let Some(scene) = &mut self.current_scene {
                                            // Capture UI state when selection changes (for undo tracking)
                                            if (self.selected_ui_id, self.selected_ui_element_type.clone()) != (self.last_selected_ui_id, self.last_selected_ui_type.clone()) {
                                                // Push undo for previous element if it was modified
                                                if let (Some(start), Some(prev_type), Some(prev_id)) = (self.ui_inspector_start_element.take(), self.last_selected_ui_type.take(), self.last_selected_ui_id.take()) {
                                                    if let Some(nl) = scene.ui_layouts.get_mut(self.selected_layout_idx) {
                                                        let current = match prev_type.as_str() {
                                                            "Button" => nl.layout.buttons.get(prev_id as usize).map(|b| UiElementHistory::Button(b.clone())),
                                                            "Panel" => nl.layout.panels.get(prev_id as usize).map(|p| UiElementHistory::Panel(p.clone())),
                                                            "Text" => nl.layout.texts.get(prev_id as usize).map(|t| UiElementHistory::Text(t.clone())),
                                                            _ => None,
                                                        };
                                                        if let Some(after) = current {
                                                            if start != after {
                                                                pending_undo = Some(EditorAction::ModifyUiElement {
                                                                    layout_idx: self.selected_layout_idx,
                                                                    element_type: prev_type,
                                                                    element_idx: prev_id as usize,
                                                                    before: start,
                                                                    after,
                                                                });
                                                            }
                                                        }
                                                    }
                                                }
                                                // Capture new element's state
                                                if let (Some(id), Some(etype)) = (self.selected_ui_id, &self.selected_ui_element_type) {
                                                    if let Some(nl) = scene.ui_layouts.get_mut(self.selected_layout_idx) {
                                                        let state = match etype.as_str() {
                                                            "Button" => nl.layout.buttons.get(id as usize).map(|b| UiElementHistory::Button(b.clone())),
                                                            "Panel" | "PanelChild" => nl.layout.panels.get(id as usize).map(|p| UiElementHistory::Panel(p.clone())),
                                                            "Text" => nl.layout.texts.get(id as usize).map(|t| UiElementHistory::Text(t.clone())),
                                                            _ => None,
                                                        };
                                                        self.ui_inspector_start_element = state;
                                                        self.last_selected_ui_type = Some(etype.clone());
                                                        self.last_selected_ui_id = Some(id);
                                                    }
                                                } else {
                                                    self.last_selected_ui_type = None;
                                                    self.last_selected_ui_id = None;
                                                }
                                            }

                                            if let (Some(id), Some(etype)) = (self.selected_ui_id, &self.selected_ui_element_type.clone()) {
                                                if let Some(nl) = scene.ui_layouts.get_mut(self.selected_layout_idx) {
                                                    match etype.as_str() {
                                            "Button" => if let Some(button) = nl.layout.buttons.get_mut(id as usize) {
                                                ui.label(format!("Button #{}", button.id));
                                                ui.horizontal(|ui| { ui.label("Label:"); if ui.text_edit_singleline(&mut button.label.content).changed() { ui_dirty = true; } });
                                                ui.horizontal(|ui| { ui.label("On Click:"); 
                                                    let mut cb = button.on_click.clone().unwrap_or_default();
                                                    if ui.text_edit_singleline(&mut cb).changed() {
                                                        button.on_click = if cb.is_empty() { None } else { Some(cb) };
                                                        ui_dirty = true;
                                                    }
                                                });
                                                ui.label("Position:");
                                                ui.horizontal(|ui| {
                                                    ui.add(egui::DragValue::new(&mut button.panel.position[0]).prefix("X: ").speed(1.0));
                                                    ui.add(egui::DragValue::new(&mut button.panel.position[1]).prefix("Y: ").speed(1.0));
                                                });
                                                ui.label("Size:");
                                                ui.horizontal(|ui| {
                                                    ui.add(egui::DragValue::new(&mut button.panel.size[0]).prefix("W: ").speed(1.0));
                                                    ui.add(egui::DragValue::new(&mut button.panel.size[1]).prefix("H: ").speed(1.0));
                                                });
                                                ui.separator();
                                                if ui.button("🗑 Delete").clicked() {
                                                    let button = nl.layout.buttons.get(id as usize).cloned();
                                                    if let Some(b) = button {
                                                        pending_undo = Some(EditorAction::DeleteUiElement {
                                                            layout_idx: self.selected_layout_idx,
                                                            element_type: "Button".into(),
                                                            element_idx: id as usize,
                                                            element: UiElementHistory::Button(b),
                                                        });
                                                        nl.layout.buttons.remove(id as usize);
                                                        self.selected_ui_id = None;
                                                        ui_dirty = true;
                                                    }
                                                }
                                            }
                                            "Panel" => if let Some(panel) = nl.layout.panels.get_mut(id as usize) {
                                                ui.label("Panel");
                                                
                                                // Color picker
                                                ui.horizontal(|ui| {
                                                    ui.label("Color:");
                                                    let mut c = egui::Color32::from_rgba_unmultiplied(
                                                        (panel.color[0] * 255.0) as u8,
                                                        (panel.color[1] * 255.0) as u8,
                                                        (panel.color[2] * 255.0) as u8,
                                                        (panel.color[3] * 255.0) as u8,
                                                    );
                                                    if ui.color_edit_button_srgba(&mut c).changed() {
                                                        panel.color = [
                                                            c.r() as f32 / 255.0,
                                                            c.g() as f32 / 255.0,
                                                            c.b() as f32 / 255.0,
                                                            c.a() as f32 / 255.0,
                                                        ];
                                                        ui_dirty = true;
                                                    }
                                                });
                                                
                                                // Background texture
                                                ui.horizontal(|ui| {
                                                    ui.label("Background:");
                                                    let mut tex_id = panel.texture_id.clone().unwrap_or_default();
                                                    let response = ui.text_edit_singleline(&mut tex_id);
                                                    if response.changed() {
                                                        panel.texture_id = if tex_id.is_empty() { None } else { Some(tex_id) };
                                                        ui_dirty = true;
                                                    }
                                                    // Drag-and-drop for image files
                                                    if response.hovered() {
                                                        ui.ctx().input(|i| {
                                                            if !i.raw.dropped_files.is_empty() {
                                                                if let Some(file) = i.raw.dropped_files.first() {
                                                                    if let Some(path) = &file.path {
                                                                        if let Some(name) = path.file_name() {
                                                                            let name_str = name.to_string_lossy().to_string();
                                                                            panel.texture_id = Some(name_str);
                                                                            ui_dirty = true;
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        });
                                                        response.on_hover_text("Drag image file here").on_hover_cursor(egui::CursorIcon::PointingHand);
                                                    }
                                                });
                                                
                                                ui.label("Position:");
                                                ui.horizontal(|ui| {
                                                    ui.add(egui::DragValue::new(&mut panel.position[0]).prefix("X: ").speed(1.0));
                                                    ui.add(egui::DragValue::new(&mut panel.position[1]).prefix("Y: ").speed(1.0));
                                                });
                                                ui.label("Size:");
                                                ui.horizontal(|ui| {
                                                    ui.add(egui::DragValue::new(&mut panel.size[0]).prefix("W: ").speed(1.0));
                                                    ui.add(egui::DragValue::new(&mut panel.size[1]).prefix("H: ").speed(1.0));
                                                });
                                                
                                                // Child elements section
                                                ui.separator();
                                                let children_count = panel.children.len();
                                                ui.label(format!("Children: {}", children_count));
                                                
                                                // Compute next child ID before the closure to avoid borrow conflicts
                                                let next_child_id = children_count as u32 + 100;
                                                
                                                let mut add_button = false;
                                                let mut add_text = false;
                                                ui.horizontal(|ui| {
                                                    if ui.button("➕ Button").clicked() {
                                                        add_button = true;
                                                    }
                                                    if ui.button("➕ Text").clicked() {
                                                        add_text = true;
                                                    }
                                                });
                                                if add_button {
                                                    panel.children.push(stfsc_engine::ui::PanelChild::Button(
                                                        stfsc_engine::ui::Button::new(next_child_id, "Button", 10.0, 10.0, 100.0, 30.0)
                                                    ));
                                                    ui_dirty = true;
                                                }
                                                if add_text {
                                                    panel.children.push(stfsc_engine::ui::PanelChild::Text(
                                                        stfsc_engine::ui::Text::new("Text", 10.0, 10.0)
                                                    ));
                                                    ui_dirty = true;
                                                }


                                                // List existing children with editing controls
                                                let mut child_to_remove = None;
                                                for ci in 0..panel.children.len() {
                                                    ui.push_id(ci, |ui| {
                                                        let child = &mut panel.children[ci];
                                                        match child {
                                                            stfsc_engine::ui::PanelChild::Button(btn) => {
                                                                ui.collapsing(format!("Button: {}", &btn.label.content), |ui| {
                                                                    ui.horizontal(|ui| {
                                                                        ui.label("Label:");
                                                                        if ui.text_edit_singleline(&mut btn.label.content).changed() {
                                                                            ui_dirty = true;
                                                                        }
                                                                    });
                                                                    ui.horizontal(|ui| {
                                                                        ui.label("On Click:");
                                                                        let mut cb = btn.on_click.clone().unwrap_or_default();
                                                                        if ui.text_edit_singleline(&mut cb).changed() {
                                                                            btn.on_click = if cb.is_empty() { None } else { Some(cb) };
                                                                            ui_dirty = true;
                                                                        }
                                                                    });
                                                                    ui.label("Position (relative):");
                                                                    ui.horizontal(|ui| {
                                                                        if ui.add(egui::DragValue::new(&mut btn.panel.position[0]).prefix("X: ").speed(1.0)).changed() { ui_dirty = true; }
                                                                        if ui.add(egui::DragValue::new(&mut btn.panel.position[1]).prefix("Y: ").speed(1.0)).changed() { ui_dirty = true; }
                                                                    });
                                                                    ui.label("Size:");
                                                                    ui.horizontal(|ui| {
                                                                        if ui.add(egui::DragValue::new(&mut btn.panel.size[0]).prefix("W: ").speed(1.0)).changed() { ui_dirty = true; }
                                                                        if ui.add(egui::DragValue::new(&mut btn.panel.size[1]).prefix("H: ").speed(1.0)).changed() { ui_dirty = true; }
                                                                    });
                                                                    if ui.small_button("❌ Remove").clicked() {
                                                                        child_to_remove = Some(ci);
                                                                    }
                                                                });
                                                            }
                                                            stfsc_engine::ui::PanelChild::Text(txt) => {
                                                                ui.collapsing(format!("Text: {}", &txt.content), |ui| {
                                                                    ui.horizontal(|ui| {
                                                                        ui.label("Content:");
                                                                        if ui.text_edit_singleline(&mut txt.content).changed() {
                                                                            ui_dirty = true;
                                                                        }
                                                                    });
                                                                    ui.label("Position (relative):");
                                                                    ui.horizontal(|ui| {
                                                                        if ui.add(egui::DragValue::new(&mut txt.position[0]).prefix("X: ").speed(1.0)).changed() { ui_dirty = true; }
                                                                        if ui.add(egui::DragValue::new(&mut txt.position[1]).prefix("Y: ").speed(1.0)).changed() { ui_dirty = true; }
                                                                    });
                                                                    if ui.add(egui::Slider::new(&mut txt.font_size, 8.0..=72.0).text("Size")).changed() {
                                                                        ui_dirty = true;
                                                                    }
                                                                    if ui.small_button("❌ Remove").clicked() {
                                                                        child_to_remove = Some(ci);
                                                                    }
                                                                });
                                                            }
                                                        }
                                                    });
                                                }
                                                if let Some(idx) = child_to_remove {
                                                    panel.children.remove(idx);
                                                    ui_dirty = true;
                                                }
                                                
                                                ui.separator();
                                                if ui.button("🗑 Delete").clicked() {
                                                    let panel = nl.layout.panels.get(id as usize).cloned();
                                                    if let Some(p) = panel {
                                                        pending_undo = Some(EditorAction::DeleteUiElement {
                                                            layout_idx: self.selected_layout_idx,
                                                            element_type: "Panel".into(),
                                                            element_idx: id as usize,
                                                            element: UiElementHistory::Panel(p),
                                                        });
                                                        nl.layout.panels.remove(id as usize);
                                                        self.selected_ui_id = None;
                                                        ui_dirty = true;
                                                    }
                                                }
                                            }
                                            "Text" => if let Some(text) = nl.layout.texts.get_mut(id as usize) {
                                                ui.label("Text");
                                                ui.horizontal(|ui| { ui.label("Content:"); if ui.text_edit_singleline(&mut text.content).changed() { ui_dirty = true; } });
                                                ui.add(egui::Slider::new(&mut text.font_size, 8.0..=128.0).text("Size"));
                                                ui.label("Position:");
                                                ui.horizontal(|ui| {
                                                    ui.add(egui::DragValue::new(&mut text.position[0]).prefix("X: ").speed(1.0));
                                                    ui.add(egui::DragValue::new(&mut text.position[1]).prefix("Y: ").speed(1.0));
                                                });
                                                ui.separator();
                                                if ui.button("🗑 Delete").clicked() {
                                                    let text = nl.layout.texts.get(id as usize).cloned();
                                                    if let Some(t) = text {
                                                        pending_undo = Some(EditorAction::DeleteUiElement {
                                                            layout_idx: self.selected_layout_idx,
                                                            element_type: "Text".into(),
                                                            element_idx: id as usize,
                                                            element: UiElementHistory::Text(t),
                                                        });
                                                        nl.layout.texts.remove(id as usize);
                                                        self.selected_ui_id = None;
                                                        ui_dirty = true;
                                                    }
                                                }
                                            }
                                            _ => {}
                                        }
                                      }
                                    }
                                    if let Some(a) = pending_undo { self.push_undo_action(a); }
                                } else {
                                    ui.label("Click an element to edit");
                                    ui.separator();
                                    ui.heading("⌨ Keybinds");
                                    if let Some(scene) = &mut self.current_scene {
                                      if let Some(nl) = scene.ui_layouts.get_mut(self.selected_layout_idx) {
                                        let mut to_remove = None;
                                        for (i, bind) in nl.layout.keybinds.iter_mut().enumerate() {
                                            ui.horizontal(|ui| {
                                                ui.text_edit_singleline(&mut bind.key);
                                                ui.label("→");
                                                ui.text_edit_singleline(&mut bind.callback);
                                                if ui.button("❌").clicked() { to_remove = Some(i); }
                                            });
                                        }
                                        if let Some(idx) = to_remove { nl.layout.keybinds.remove(idx); ui_dirty = true; }
                                        if ui.button("➕ Add Keybind").clicked() {
                                            nl.layout.keybinds.push(stfsc_engine::ui::Keybind { key: "Space".into(), callback: "on_space".into() });
                                            ui_dirty = true;
                                        }
                                      }
                            }
                                }
                        });
                    });
                });
            });
            if ui_dirty { self.scene_dirty = true; }
            if !open { self.show_ui_editor = false; }
        }

        // Unsaved Changes Warning Dialog
        if self.show_unsaved_warning {
            egui::Window::new("⚠️ Unsaved Changes")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
                .show(ctx, |ui| {
                    let scene_name = self.current_scene.as_ref()
                        .map(|s| s.name.as_str())
                        .unwrap_or("Untitled");
                    ui.label(format!("Save changes to \"{}\"?", scene_name));
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("💾 Save").clicked() {
                            // Save first, then execute pending action
                            if let Some(path) = self.current_scene_path.clone() {
                                if let Err(e) = self.save_scene_to_path(&path) {
                                    self.status = format!("Error: {}", e);
                                    self.show_unsaved_warning = false;
                                    self.pending_action = None;
                                    return;
                                }
                            } else if let Some(scene) = &self.current_scene {
                                let path = format!("scenes/{}.json", scene.name.to_lowercase().replace(" ", "_"));
                                if let Err(e) = self.save_scene_to_path(&path) {
                                    self.status = format!("Error: {}", e);
                                    self.show_unsaved_warning = false;
                                    self.pending_action = None;
                                    return;
                                }
                            }
                            self.execute_pending_action(ctx);
                        }
                        if ui.button("🚫 Don't Save").clicked() {
                            self.scene_dirty = false; // Discard changes
                            self.execute_pending_action(ctx);
                        }
                        if ui.button("Cancel").clicked() {
                            self.show_unsaved_warning = false;
                            self.pending_action = None;
                        }
                    });
                });
        }

        ctx.request_repaint();
    }
}
