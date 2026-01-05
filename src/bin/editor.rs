use eframe::egui;
use rayon::prelude::*;
use std::io::Write;
use std::net::TcpStream;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use stfsc_engine::world::{LightType, SceneUpdate, LAYER_ENVIRONMENT, LAYER_PROP, LAYER_CHARACTER, LAYER_VEHICLE, LAYER_DEFAULT};

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
        if ndc_x.abs() > 2.0 || ndc_y.abs() > 2.0 {
            return None;
        }
        Some(egui::pos2(
            size.x * 0.5 + ndc_x * size.x * 0.5,
            size.y * 0.5 - ndc_y * size.y * 0.5,
        ))
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
    #[serde(skip)]
    deployed: bool,
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
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
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
            position: [0.0, -1.1, 0.0],  // Lowered to match main.rs floor exactly
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [100.0, 2.0, 100.0],  // Match thickness with main.rs floor
            entity_type: EntityType::Ground,
            material: m.clone(),
            script: None,
            collision_enabled: true,
            layer: LAYER_ENVIRONMENT,
            is_static: false,
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
            
            for entity in &scene.entities {
                self.deploy_entity_to_quest(entity);
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
    fn load_scene_from_path(&mut self, path: &str) -> Result<(), String> {
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
                } else {
                    eprintln!("Warning: Could not load texture at '{}'", texture_path);
                }
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
                            if ext_lower == "obj" || ext_lower == "fbx" {
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
    
    /// Import a 3D model from file path and add to scene
    fn import_model_from_path(&mut self, path: &str) -> Result<(), String> {
        // Read the file
        let data = std::fs::read(path).map_err(|e| format!("Failed to read file: {}", e))?;
        
        // Get filename for entity name
        let filename = std::path::Path::new(path)
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("Model")
            .to_string();
        
        let id = if let Some(scene) = &self.current_scene {
            scene.entities.iter().map(|e| e.id).max().unwrap_or(0) + 1
        } else {
            return Err("No scene loaded".into());
        };
        
        let new_entity = SceneEntity {
            id,
            name: filename.clone(),
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
            entity_type: EntityType::Mesh { path: path.to_string() },
            material: Material::default(),
            script: None,
            collision_enabled: false,
            layer: LAYER_PROP,
            is_static: true,  // Imported meshes are static by default (no gravity)
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
        }
        
        // Now add to scene
        if let Some(scene) = &mut self.current_scene {
            scene.entities.push(new_entity);
        }
        
        self.selected_entity_id = Some(id);
        self.scene_dirty = true;
        self.status = format!("Imported: {}", filename);
        
        Ok(())
    }
    
    /// Deploy a mesh entity to the Quest via SpawnFbxMesh
    fn deploy_mesh_entity(&self, entity: &SceneEntity, mesh_data: &[u8]) {
        // Upload texture if present
        if let (Some(texture_id), Some(texture_data)) = (&entity.material.albedo_texture, &entity.material.albedo_texture_data) {
            let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::UploadTexture {
                id: texture_id.clone(),
                data: texture_data.clone(),
            }));
        }
        
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

    /// Execute the pending action after user confirms in unsaved dialog
    fn execute_pending_action(&mut self) {
        if let Some(action) = self.pending_action.take() {
            match action {
                PendingAction::NewScene => {
                    self.show_new_scene_dialog = true;
                    self.new_scene_name.clear();
                }
                PendingAction::OpenScene(path) => {
                    if let Err(e) = self.load_scene_from_path(&path) {
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
                let mut new_entity = copied;
                new_entity.id = new_id;
                new_entity.name = format!("{} (Copy)", new_entity.name);
                // Offset position slightly so it's visible
                new_entity.position[0] += 1.0;
                new_entity.position[2] += 1.0;
                new_entity.deployed = false;
                
                // Push to undo stack
                self.undo_stack.push(EditorAction::AddEntity { entity: new_entity.clone() });
                self.redo_stack.clear(); // Clear redo on new action
                
                scene.entities.push(new_entity);
                self.selected_entity_id = Some(new_id);
                self.scene_dirty = true;
                self.status = format!("Pasted entity #{}", new_id);
            }
        }
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
            }
            self.undo_stack.push(action);
            self.scene_dirty = true;
        }
    }

    fn draw_3d_viewport(&mut self, ui: &mut egui::Ui) {
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
                    // Reverse iterate to find top-most object? Or just check all.
                    // Basic check using project() distance
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
                        // Only push if position actually changed
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

            // Handle Object Dragging
            if let Some(drag_id) = self.dragging_id {
                let rect_size = rect.size();
                // Ensure raycast uses relative coordinates properly
                let rel_x = mouse.x - rect.min.x;
                let rel_y = mouse.y - rect.min.y;
                let (cam_pos, ray_dir) = self.camera.get_ray(egui::pos2(rel_x, rel_y), rect_size);

                if let Some(scene) = &mut self.current_scene {
                    if let Some(entity) = scene.entities.iter_mut().find(|e| e.id == drag_id) {
                        // Ray-Plane Intersection (Y = entity.y)
                        // P = O + tD
                        // P.y = PlaneY
                        // O.y + t*D.y = PlaneY  =>  t = (PlaneY - O.y) / D.y
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
                        // Right-drag: Orbit
                        self.camera.yaw -= delta.x * 0.01;
                        self.camera.pitch = (self.camera.pitch - delta.y * 0.01).clamp(-1.5, 1.5);
                    }
                    Some(egui::PointerButton::Middle) | Some(egui::PointerButton::Primary) => {
                        // Middle/Left-drag: Pan (if not dragging object)
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

        // Zoom
        let scroll = ui.input(|i| i.scroll_delta.y);
        self.camera.distance = (self.camera.distance - scroll * 0.5).clamp(5.0, 200.0);

        // Background
        painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(30, 40, 60));

        // Grid
        let grid_spacing = 5.0;
        for i in -20..=20 {
            let x = i as f32 * grid_spacing;
            if let (Some(p1), Some(p2)) = (
                self.camera.project(Vec3::new(x, 0.0, -100.0), available),
                self.camera.project(Vec3::new(x, 0.0, 100.0), available),
            ) {
                painter.line_segment(
                    [
                        egui::pos2(rect.min.x + p1.x, rect.min.y + p1.y),
                        egui::pos2(rect.min.x + p2.x, rect.min.y + p2.y),
                    ],
                    egui::Stroke::new(0.5, egui::Color32::from_rgb(50, 50, 60)),
                );
            }
            if let (Some(p1), Some(p2)) = (
                self.camera.project(Vec3::new(-100.0, 0.0, x), available),
                self.camera.project(Vec3::new(100.0, 0.0, x), available),
            ) {
                painter.line_segment(
                    [
                        egui::pos2(rect.min.x + p1.x, rect.min.y + p1.y),
                        egui::pos2(rect.min.x + p2.x, rect.min.y + p2.y),
                    ],
                    egui::Stroke::new(0.5, egui::Color32::from_rgb(50, 50, 60)),
                );
            }
        }

        // Origin axes
        let origin = Vec3::zero();
        if let Some(o) = self.camera.project(origin, available) {
            let o = egui::pos2(rect.min.x + o.x, rect.min.y + o.y);
            for (axis, color) in [
                (Vec3::new(5.0, 0.0, 0.0), egui::Color32::RED),
                (Vec3::new(0.0, 5.0, 0.0), egui::Color32::GREEN),
                (Vec3::new(0.0, 0.0, 5.0), egui::Color32::BLUE),
            ] {
                if let Some(p) = self.camera.project(axis, available) {
                    painter.line_segment(
                        [o, egui::pos2(rect.min.x + p.x, rect.min.y + p.y)],
                        egui::Stroke::new(2.0, color),
                    );
                }
            }
        }

        // Draw entities - parallel projection computation
        let mut clicked_entity: Option<u32> = None;
        if let Some(scene) = &self.current_scene {
            let camera = self.camera; // Copy for parallel access
            let selected_id = self.selected_entity_id;
            
            // Parallel: Compute all entity render data (projections, colors, etc.)
            let entity_renders: Vec<_> = scene.entities.par_iter().map(|entity| {
                let pos = Vec3::new(entity.position[0], entity.position[1], entity.position[2]);
                let half = Vec3::new(
                    entity.scale[0] * 0.5,
                    entity.scale[1] * 0.5,
                    entity.scale[2] * 0.5,
                );
                let is_selected = selected_id == Some(entity.id);
                
                let base_color = [
                    (entity.material.albedo_color[0] * 200.0) as u8,
                    (entity.material.albedo_color[1] * 200.0) as u8,
                    (entity.material.albedo_color[2] * 200.0) as u8,
                ];
                
                // Box corners
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
                
                // Project all 8 corners + center in parallel (within this entity)
                let proj: Vec<Option<egui::Pos2>> = corners.iter().map(|c| {
                    camera.project(*c, available).map(|p| egui::pos2(rect.min.x + p.x, rect.min.y + p.y))
                }).collect();
                
                let center_proj = camera.project(pos, available)
                    .map(|p| egui::pos2(rect.min.x + p.x, rect.min.y + p.y));
                
                (entity.id, is_selected, base_color, proj, center_proj)
            }).collect();
            
            // Sequential: Draw using pre-computed projections
            for (id, is_selected, base_color, proj, center_proj) in entity_renders {
                let base = egui::Color32::from_rgb(base_color[0], base_color[1], base_color[2]);
                let wire_color = if is_selected { egui::Color32::GOLD } else { base };
                let stroke = egui::Stroke::new(if is_selected { 2.5 } else { 1.0 }, wire_color);
                
                // Draw wireframe
                for (i, j) in [
                    (0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7),
                ] {
                    if let (Some(p1), Some(p2)) = (proj[i], proj[j]) {
                        painter.line_segment([p1, p2], stroke);
                    }
                }
                
                // Center dot + click detection
                if let Some(c) = center_proj {
                    let size = (8.0 + 400.0 / self.camera.distance).clamp(4.0, 25.0);
                    painter.circle_filled(c, size, base.linear_multiply(0.6));
                    if is_selected {
                        painter.circle_stroke(c, size + 2.0, egui::Stroke::new(2.0, egui::Color32::GOLD));
                    }
                    
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
        if let Some(id) = clicked_entity {
            self.selected_entity_id = Some(id);
        }

        // Overlay
        painter.text(
            rect.min + egui::vec2(10.0, 10.0),
            egui::Align2::LEFT_TOP,
            "Right-drag: Orbit | Middle-drag or M+Left-drag: Move Object | Left-drag: Pan",
            egui::FontId::proportional(11.0),
            egui::Color32::WHITE,
        );
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
                                        if let Err(e) = self.load_scene_from_path(&path_clone) {
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
                                deployed: false,
                            });
                            self.selected_entity_id = Some(id);
                        }
                        ui.close_menu();
                    }
                    ui.separator();
                    ui.label("📦 Import");
                    if ui.button("📦 Import 3D Model (.obj)").clicked() {
                        // Scan for model files
                        self.import_model_files = Self::scan_model_files();
                        self.show_import_model_dialog = true;
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
                ui.menu_button("View", |ui| {
                    if ui.button("Reset Camera").clicked() {
                        self.camera = Camera3D::new();
                        ui.close_menu();
                    }
                    if ui.button("Focus Selected").clicked() {
                        if let (Some(id), Some(scene)) =
                            (self.selected_entity_id, &self.current_scene)
                        {
                            if let Some(e) = scene.entities.iter().find(|e| e.id == id) {
                                self.camera.target =
                                    Vec3::new(e.position[0], e.position[1], e.position[2]);
                            }
                        }
                        ui.close_menu();
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
                    if let Some(entity) = scene.entities.iter_mut().find(|e| e.id == id) {
                        ui.horizontal(|ui| { 
                            ui.label("Name:"); 
                            if ui.text_edit_singleline(&mut entity.name).changed() {
                                inspector_dirty = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("📜 Script:");
                            let mut script_name = entity.script.clone().unwrap_or_default();
                            if ui.text_edit_singleline(&mut script_name).changed() {
                                entity.script = if script_name.is_empty() { None } else { Some(script_name) };
                                inspector_dirty = true;
                            }
                        });
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
                        ui.vertical(|ui| {
                            ui.horizontal(|ui| {
                                if ui.add(egui::DragValue::new(&mut entity.position[0]).speed(0.1).prefix("X:")).changed() { inspector_dirty = true; }
                                if ui.add(egui::DragValue::new(&mut entity.position[1]).speed(0.1).prefix("Y:")).changed() { inspector_dirty = true; }
                                if ui.add(egui::DragValue::new(&mut entity.position[2]).speed(0.1).prefix("Z:")).changed() { inspector_dirty = true; }
                            });
                            ui.horizontal(|ui| {
                                if ui.add(egui::DragValue::new(&mut entity.scale[0]).speed(0.1).prefix("X:")).changed() { inspector_dirty = true; }
                                if ui.add(egui::DragValue::new(&mut entity.scale[1]).speed(0.1).prefix("Y:")).changed() { inspector_dirty = true; }
                                if ui.add(egui::DragValue::new(&mut entity.scale[2]).speed(0.1).prefix("Z:")).changed() { inspector_dirty = true; }
                            });
                        });
                        
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
                                ui.add(egui::Slider::new(height, 2.0..=50.0).text("Height"));
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
                            if let EntityType::Camera = e.entity_type {
                                let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::SetPlayerStart {
                                    position: e.position, rotation: e.rotation,
                                }));
                            } else {
                                let color = e.material.albedo_color;
                                let primitive = match &e.entity_type {
                                    EntityType::Primitive(p) => match p {
                                        PrimitiveType::Cube => 0, PrimitiveType::Sphere => 1, PrimitiveType::Cylinder => 2,
                                        PrimitiveType::Plane => 3, PrimitiveType::Capsule => 4, PrimitiveType::Cone => 5,
                                    },
                                    EntityType::Ground => 0, EntityType::Vehicle => 0, EntityType::Building { .. } => 0, EntityType::CrowdAgent { .. } => 4, _ => 0
                                };
                                let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::Spawn {
                                    id: e.id, primitive, position: e.position, rotation: e.rotation, color,
                                    albedo_texture: e.material.albedo_texture.clone(),
                                    collision_enabled: e.collision_enabled,
                                    layer: e.layer,
                                    is_static: e.is_static,
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

        // CENTER - 3D VIEWPORT
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("3D Scene Viewport");
            self.draw_3d_viewport(ui);
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
                                        if let Err(e) = self.load_scene_from_path(&file_path) {
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
                                        if let Err(e) = self.import_model_from_path(&file_path) {
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
                        if ui.button("➕ Button").clicked() {
                           if let Some(scene) = &mut self.current_scene {
                               if let Some(named_layout) = scene.ui_layouts.get_mut(self.selected_layout_idx) {
                                   let id = (named_layout.layout.buttons.len() + named_layout.layout.panels.len() + named_layout.layout.texts.len()) as u32 + 1;
                                   // Use Button::new to get proper default colors (dark panel, white text)
                                   named_layout.layout.buttons.push(stfsc_engine::ui::Button::new(id, "New Button", 100.0, 100.0, 200.0, 50.0));
                                   ui_dirty = true;
                               }
                           }
                        }
                        if ui.button("➕ Panel").clicked() {
                            if let Some(scene) = &mut self.current_scene {
                                if let Some(named_layout) = scene.ui_layouts.get_mut(self.selected_layout_idx) {
                                    named_layout.layout.panels.push(stfsc_engine::ui::Panel::new(100.0, 100.0, 200.0, 200.0));
                                    ui_dirty = true;
                                }
                            }
                        }
                        if ui.button("➕ Text").clicked() {
                            if let Some(scene) = &mut self.current_scene {
                                if let Some(named_layout) = scene.ui_layouts.get_mut(self.selected_layout_idx) {
                                    named_layout.layout.texts.push(stfsc_engine::ui::Text::new("New Text", 100.0, 100.0));
                                    ui_dirty = true;
                                }
                            }
                        }
                        ui.separator();
                        if ui.button("🗑 Clear All").clicked() {
                            if let Some(scene) = &mut self.current_scene {
                                if let Some(named_layout) = scene.ui_layouts.get_mut(self.selected_layout_idx) {
                                    named_layout.layout = stfsc_engine::ui::UiLayout::default();
                                    self.selected_ui_id = None;
                                    ui_dirty = true;
                                }
                            }
                        }
                        egui::menu::menu_button(ui, "📋 Templates", |ui| {
                            if ui.button("🎮 HUD (Health/Ammo)").clicked() {
                                if let Some(scene) = &mut self.current_scene {
                                    if let Some(nl) = scene.ui_layouts.get_mut(self.selected_layout_idx) {
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
                                        ui_dirty = true;
                                    }
                                }
                                ui.close_menu();
                            }
                            if ui.button("⏸ Pause Menu").clicked() {
                                if let Some(scene) = &mut self.current_scene {
                                    if let Some(nl) = scene.ui_layouts.get_mut(self.selected_layout_idx) {
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
                                        ui_dirty = true;
                                    }
                                }
                                ui.close_menu();
                            }
                            if ui.button("🏠 Main Menu").clicked() {
                                if let Some(scene) = &mut self.current_scene {
                                    if let Some(nl) = scene.ui_layouts.get_mut(self.selected_layout_idx) {
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
                                        ui_dirty = true;
                                    }
                                }
                                ui.close_menu();
                            }
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
                                        let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::SetUiLayer {
                                            layer,
                                            layout: nl.layout.clone(),
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
                                // Check buttons first (on top)
                                for (i, button) in nl.layout.buttons.iter().enumerate().rev() {
                                    let pos = egui::pos2(rect.min.x + button.panel.position[0] * scale, rect.min.y + button.panel.position[1] * scale);
                                    let size = egui::vec2(button.panel.size[0] * scale, button.panel.size[1] * scale);
                                    let elem_rect = egui::Rect::from_min_size(pos, size);
                                    if elem_rect.contains(pointer) {
                                        if response.drag_started() { drag_target = Some(("Button".into(), i as u32)); }
                                        if response.clicked() { clicked_element = Some(("Button".into(), i as u32)); }
                                        break;
                                    }
                                }
                                // Check panels
                                if clicked_element.is_none() && drag_target.is_none() {
                                    for (i, panel) in nl.layout.panels.iter().enumerate().rev() {
                                        let pos = egui::pos2(rect.min.x + panel.position[0] * scale, rect.min.y + panel.position[1] * scale);
                                        let size = egui::vec2(panel.size[0] * scale, panel.size[1] * scale);
                                        let elem_rect = egui::Rect::from_min_size(pos, size);
                                        if elem_rect.contains(pointer) {
                                            if response.drag_started() { drag_target = Some(("Panel".into(), i as u32)); }
                                            if response.clicked() { clicked_element = Some(("Panel".into(), i as u32)); }
                                            break;
                                        }
                                    }
                                }
                                // Check texts
                                if clicked_element.is_none() && drag_target.is_none() {
                                    for (i, text) in nl.layout.texts.iter().enumerate().rev() {
                                        let pos = egui::pos2(rect.min.x + text.position[0] * scale, rect.min.y + text.position[1] * scale);
                                        let galley = painter.layout_no_wrap(text.content.clone(), egui::FontId::proportional(text.font_size * scale), egui::Color32::WHITE);
                                        let elem_rect = egui::Rect::from_min_size(pos, galley.size());
                                        if elem_rect.contains(pointer) {
                                            if response.drag_started() { drag_target = Some(("Text".into(), i as u32)); }
                                            if response.clicked() { clicked_element = Some(("Text".into(), i as u32)); }
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
                                            _ => {}
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
                                if let (Some(id), Some(etype)) = (self.selected_ui_id, &self.selected_ui_element_type.clone()) {
                                    if let Some(scene) = &mut self.current_scene {
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
                                                    nl.layout.buttons.remove(id as usize);
                                                    self.selected_ui_id = None;
                                                    ui_dirty = true;
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
                                                    nl.layout.panels.remove(id as usize);
                                                    self.selected_ui_id = None;
                                                    ui_dirty = true;
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
                                                    nl.layout.texts.remove(id as usize);
                                                    self.selected_ui_id = None;
                                                    ui_dirty = true;
                                                }
                                            }
                                            _ => {}
                                        }
                                      }
                                    }
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
                            self.execute_pending_action();
                        }
                        if ui.button("🚫 Don't Save").clicked() {
                            self.scene_dirty = false; // Discard changes
                            self.execute_pending_action();
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
