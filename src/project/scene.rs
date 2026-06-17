use crate::ui::{UiLayer, UiLayerType, UiLayout};
use crate::world::animation::AnimatorConfig;
use crate::world::sandbox::SandboxWorldSettings;
use crate::world::{LAYER_CHARACTER, LAYER_DEFAULT, LAYER_ENVIRONMENT, LAYER_PROP, LAYER_VEHICLE};
use serde::{Deserialize, Serialize};

// ============================================================================
// PRIMITIVES LIBRARY
// ============================================================================
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum PrimitiveType {
    Cube,
    Sphere,
    Cylinder,
    Plane,
    Capsule,
    Cone,
}

impl PrimitiveType {
    pub fn all() -> Vec<PrimitiveType> {
        vec![
            PrimitiveType::Cube,
            PrimitiveType::Sphere,
            PrimitiveType::Cylinder,
            PrimitiveType::Plane,
            PrimitiveType::Capsule,
            PrimitiveType::Cone,
        ]
    }

    pub fn name(&self) -> &str {
        match self {
            PrimitiveType::Cube => "Cube",
            PrimitiveType::Sphere => "Sphere",
            PrimitiveType::Cylinder => "Cylinder",
            PrimitiveType::Plane => "Plane",
            PrimitiveType::Capsule => "Capsule",
            PrimitiveType::Cone => "Cone",
        }
    }

    pub fn icon(&self) -> &str {
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Material {
    pub name: String,
    pub albedo_color: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub albedo_texture: Option<String>,
    #[serde(skip)]
    pub albedo_texture_data: Option<Vec<u8>>, // Raw texture bytes for deployment
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

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ShaderPreset {
    StandardPbr,
    Unlit,
    Transparent,
    Toon,
    VertexColor,
    SkinnedPbr,
}

impl ShaderPreset {
    pub fn all() -> Vec<Self> {
        vec![
            Self::StandardPbr,
            Self::Unlit,
            Self::Transparent,
            Self::Toon,
            Self::VertexColor,
            Self::SkinnedPbr,
        ]
    }

    pub fn name(&self) -> &str {
        match self {
            Self::StandardPbr => "Standard PBR",
            Self::Unlit => "Unlit",
            Self::Transparent => "Transparent",
            Self::Toon => "Toon",
            Self::VertexColor => "Vertex Color",
            Self::SkinnedPbr => "Skinned PBR",
        }
    }
}

impl Default for ShaderPreset {
    fn default() -> Self {
        Self::StandardPbr
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MaterialSlot {
    pub name: String,
    pub material: Material,
    #[serde(default)]
    pub shader: ShaderPreset,
    #[serde(default)]
    pub script: Option<String>,
    #[serde(default)]
    pub bone_name: Option<String>,
    #[serde(default)]
    pub bone_index: Option<usize>,
}

impl MaterialSlot {
    pub fn from_material(name: impl Into<String>, material: Material) -> Self {
        Self {
            name: name.into(),
            material,
            shader: ShaderPreset::default(),
            script: None,
            bone_name: None,
            bone_index: None,
        }
    }
}

impl Default for MaterialSlot {
    fn default() -> Self {
        Self::from_material("Element 0", Material::default())
    }
}

// ============================================================================
// SCENE DATA
// ============================================================================

fn default_fov() -> f32 {
    0.785 // PI/4
}

fn default_layer() -> u32 {
    0 // Default layer
}

fn default_true() -> bool {
    true
}

// ============================================================================
// SCENE HIERARCHY
// ============================================================================

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SceneDomain {
    /// A real-time 3D scene made of ECS entities, physics, scripts, lights, etc.
    World3d,
    /// A 2D UI scene made of UI layout data and input callbacks.
    Ui,
}

impl Default for SceneDomain {
    fn default() -> Self {
        Self::World3d
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SceneTransitionMode {
    /// Replace the currently active scene in the same domain.
    Replace,
    /// Add the target scene without clearing the current domain.
    Additive,
    /// Push the target on that domain's stack.
    Push,
    /// Show the target as an overlay. UI scenes treat this like a visible layer.
    Overlay,
    /// Pop the current scene stack and return to the previous scene where supported.
    Pop,
}

impl Default for SceneTransitionMode {
    fn default() -> Self {
        Self::Replace
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SceneRef {
    pub name: String,
    pub path: String,
    #[serde(default)]
    pub domain: SceneDomain,
    #[serde(default)]
    pub alias: Option<String>,
    #[serde(default)]
    pub layer: Option<UiLayer>,
    #[serde(default)]
    pub children: Vec<SceneRef>,
}

impl SceneRef {
    pub fn world(name: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
            domain: SceneDomain::World3d,
            alias: None,
            layer: None,
            children: Vec::new(),
        }
    }

    pub fn ui(
        name: impl Into<String>,
        path: impl Into<String>,
        alias: impl Into<String>,
        layer: UiLayer,
    ) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
            domain: SceneDomain::Ui,
            alias: Some(alias.into()),
            layer: Some(layer),
            children: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneTransition {
    pub name: String,
    #[serde(default)]
    pub from: Option<SceneRef>,
    pub to: SceneRef,
    #[serde(default)]
    pub mode: SceneTransitionMode,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SceneHierarchy {
    #[serde(default)]
    pub roots: Vec<SceneRef>,
    #[serde(default)]
    pub transitions: Vec<SceneTransition>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldScene {
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub entities: Vec<SceneEntity>,
    #[serde(default)]
    pub scene_scripts: Vec<String>,
    #[serde(default)]
    pub scene_script_components: Vec<ScriptComponent>,
    #[serde(default)]
    pub respawn_enabled: bool,
    #[serde(default)]
    pub respawn_y: f32,
    #[serde(default)]
    pub sandbox: SandboxWorldSettings,
    #[serde(default)]
    pub hierarchy: SceneHierarchy,
    #[serde(default)]
    pub transitions: Vec<SceneTransition>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UiScene {
    pub name: String,
    pub version: String,
    pub alias: String,
    #[serde(default)]
    pub layer_type: UiLayerType,
    pub layout: UiLayout,
    #[serde(default)]
    pub hierarchy: SceneHierarchy,
    #[serde(default)]
    pub transitions: Vec<SceneTransition>,
}

pub const MAX_SCRIPTS_PER_ENTITY: usize = 32;
pub const MAX_SCENE_SCRIPTS_PER_SCENE: usize = 8;
pub const CUSTOM_FUCKSCRIPT_NAME: &str = "CustomFuckScript";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ScriptCompileMode {
    BuiltinNative,
    EditableNativeCache,
    CustomNativeCache,
}

impl Default for ScriptCompileMode {
    fn default() -> Self {
        Self::BuiltinNative
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ScriptComponent {
    pub name: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub compile_mode: ScriptCompileMode,
    #[serde(default)]
    pub cache_key: Option<String>,
}

impl ScriptComponent {
    pub fn builtin(name: impl Into<String>) -> Self {
        let name = name.into();
        if name == CUSTOM_FUCKSCRIPT_NAME {
            return Self::custom();
        }

        Self {
            source: builtin_script_template(&name),
            name,
            enabled: true,
            compile_mode: ScriptCompileMode::BuiltinNative,
            cache_key: None,
        }
    }

    pub fn custom() -> Self {
        Self {
            name: CUSTOM_FUCKSCRIPT_NAME.to_string(),
            enabled: true,
            source: custom_script_template(),
            compile_mode: ScriptCompileMode::CustomNativeCache,
            cache_key: None,
        }
    }

    pub fn runtime_name(&self) -> String {
        if !matches!(self.compile_mode, ScriptCompileMode::BuiltinNative) {
            if let Some(cache_key) = self.cache_key.as_ref().map(|key| key.trim()) {
                if !cache_key.is_empty() {
                    return cache_key.to_string();
                }
            }
        }

        self.name.trim().to_string()
    }

    pub fn has_editable_source(&self) -> bool {
        !self.source.trim().is_empty()
            || matches!(
                self.compile_mode,
                ScriptCompileMode::EditableNativeCache | ScriptCompileMode::CustomNativeCache
            )
    }
}

pub fn builtin_script_template(name: &str) -> String {
    format!(
        "// Editable cache source for {name}.\n// Built-in presets still run as native Rust unless this source is promoted\n// into generated bindings during project export.\n\nscript {name} {{\n    on_update(ctx) {{\n        // Add high-level FuckScript logic here.\n    }}\n}}\n"
    )
}

pub fn custom_script_template() -> String {
    "// Custom FuckScript\n// Export will include this source in the native script cache manifest.\n\nscript CustomFuckScript {\n    on_start(ctx) {\n    }\n\n    on_update(ctx) {\n    }\n}\n".to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneEntity {
    pub id: u32,
    pub name: String,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub entity_type: EntityType,
    pub material: Material,
    #[serde(default)]
    pub material_slots: Vec<MaterialSlot>,
    #[serde(default)]
    pub active_material_slot: usize,
    /// Legacy primary script field kept so old scene files still load cleanly.
    pub script: Option<String>,
    #[serde(default)]
    pub scripts: Vec<String>,
    #[serde(default)]
    pub script_components: Vec<ScriptComponent>,
    #[serde(default)]
    pub collision_enabled: bool,
    #[serde(default = "default_layer")]
    pub layer: u32,
    #[serde(default)]
    pub is_static: bool,
    #[serde(default)]
    pub animator_config: Option<AnimatorConfig>,
    #[serde(default)]
    pub parent_id: Option<u32>,
    #[serde(default = "default_fov")]
    pub fov: f32,
    // Transient flags
    #[serde(skip)]
    pub deployed: bool,
}

impl SceneEntity {
    pub fn ensure_material_slots(&mut self) -> &mut Vec<MaterialSlot> {
        if self.material_slots.is_empty() {
            self.material_slots.push(MaterialSlot::from_material(
                "Element 0",
                self.material.clone(),
            ));
        }

        if self.active_material_slot >= self.material_slots.len() {
            self.active_material_slot = 0;
        }

        &mut self.material_slots
    }

    pub fn sync_primary_material_from_active_slot(&mut self) {
        if self.material_slots.is_empty() {
            return;
        }

        if self.active_material_slot >= self.material_slots.len() {
            self.active_material_slot = 0;
        }

        if let Some(slot) = self.material_slots.get(self.active_material_slot) {
            self.material = slot.material.clone();
        }
    }

    pub fn script_names(&self) -> Vec<String> {
        if !self.script_components.is_empty() {
            return self
                .script_components
                .iter()
                .filter(|component| component.enabled)
                .map(ScriptComponent::runtime_name)
                .filter(|name| !name.is_empty())
                .take(MAX_SCRIPTS_PER_ENTITY)
                .collect();
        }

        if !self.scripts.is_empty() {
            return Self::normalize_script_names(self.scripts.clone());
        }

        self.script
            .iter()
            .map(|name| name.trim())
            .filter(|name| !name.is_empty())
            .map(ToOwned::to_owned)
            .collect()
    }

    pub fn runtime_script_names(&self) -> Vec<String> {
        let mut names = self.script_names();
        if matches!(self.entity_type, EntityType::Vehicle)
            && !names.iter().any(|name| name == "Vehicle")
        {
            names.insert(0, "Vehicle".to_string());
            names.truncate(MAX_SCRIPTS_PER_ENTITY);
        }
        names
    }

    pub fn effective_collision_enabled(&self) -> bool {
        self.collision_enabled
            || matches!(self.entity_type, EntityType::Ground | EntityType::Vehicle)
    }

    pub fn effective_layer(&self) -> u32 {
        if self.layer != 0 {
            return self.layer;
        }

        match self.entity_type {
            EntityType::Ground | EntityType::Building { .. } => LAYER_ENVIRONMENT,
            EntityType::Vehicle => LAYER_VEHICLE,
            EntityType::CrowdAgent { .. } => LAYER_CHARACTER,
            EntityType::Primitive(_) => LAYER_PROP,
            _ => LAYER_DEFAULT,
        }
    }

    pub fn set_script_names(&mut self, names: Vec<String>) {
        let names = Self::normalize_script_names(names);
        let old_components = std::mem::take(&mut self.script_components);
        let script_components = names
            .iter()
            .enumerate()
            .map(|(index, name)| {
                let mut component = old_components
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| ScriptComponent::builtin(name));
                component.name = name.clone();
                component
            })
            .collect();

        self.script = names.first().cloned();
        self.scripts = names.clone();
        self.script_components = script_components;
    }

    pub fn ensure_script_components(&mut self) -> &mut Vec<ScriptComponent> {
        if self.script_components.is_empty() {
            let names = if !self.scripts.is_empty() {
                Self::normalize_script_names(self.scripts.clone())
            } else {
                Self::normalize_script_names(
                    self.script
                        .iter()
                        .map(|script| script.to_string())
                        .collect::<Vec<_>>(),
                )
            };
            self.script_components = names.into_iter().map(ScriptComponent::builtin).collect();
        }
        self.clamp_script_components();
        &mut self.script_components
    }

    pub fn sync_legacy_script_fields(&mut self) {
        let names: Vec<String> = self
            .script_components
            .iter()
            .filter(|component| component.enabled)
            .map(ScriptComponent::runtime_name)
            .filter(|name| !name.is_empty())
            .take(MAX_SCRIPTS_PER_ENTITY)
            .collect();
        self.script = names.first().cloned();
        self.scripts = names;
    }

    pub fn clamp_script_components(&mut self) {
        self.script_components.truncate(MAX_SCRIPTS_PER_ENTITY);
    }

    fn normalize_script_names(names: Vec<String>) -> Vec<String> {
        names
            .into_iter()
            .map(|name| name.trim().to_string())
            .filter(|name| !name.is_empty())
            .take(MAX_SCRIPTS_PER_ENTITY)
            .collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum EntityType {
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
        light_type: LightType,
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
        #[serde(skip)]
        audio_data: Option<Vec<u8>>,
    },
}

/// Light types
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum LightType {
    Point,
    Spot,
    Directional,
}

/// A named UI layout
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct NamedUiLayout {
    pub name: String,
    pub alias: String,
    #[serde(default)]
    pub layer_type: UiLayerType,
    pub layout: UiLayout,
}

impl Default for NamedUiLayout {
    fn default() -> Self {
        Self {
            name: "Main".to_string(),
            alias: "main".to_string(),
            layer_type: UiLayerType::InGameOverlay,
            layout: UiLayout::default(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Scene {
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub domain: SceneDomain,
    pub entities: Vec<SceneEntity>,
    #[serde(default)]
    pub scene_scripts: Vec<String>,
    #[serde(default)]
    pub scene_script_components: Vec<ScriptComponent>,
    pub respawn_enabled: bool,
    pub respawn_y: f32,
    /// Per-scene sandbox settings. Disabled by default so existing scenes keep current behavior.
    #[serde(default)]
    pub sandbox: SandboxWorldSettings,
    /// Hierarchical child scene graph. New projects should reference UI scenes here instead of
    /// embedding them directly in `ui_layouts`.
    #[serde(default)]
    pub hierarchy: SceneHierarchy,
    /// Named scene transitions authored at this scene level.
    #[serde(default)]
    pub transitions: Vec<SceneTransition>,
    /// Multiple named UI layouts with tabs
    #[serde(default)]
    pub ui_layouts: Vec<NamedUiLayout>,
    /// Legacy single layout support
    #[serde(default, skip_serializing)]
    #[allow(dead_code)]
    pub ui_layout: UiLayout,
}

impl Scene {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            version: "1.0".into(),
            domain: SceneDomain::World3d,
            entities: Vec::new(),
            scene_scripts: Vec::new(),
            scene_script_components: Vec::new(),
            respawn_enabled: false,
            respawn_y: -20.0,
            sandbox: SandboxWorldSettings::default(),
            hierarchy: SceneHierarchy::default(),
            transitions: Vec::new(),
            ui_layouts: vec![NamedUiLayout::default()],
            ui_layout: UiLayout::default(),
        }
    }

    pub fn world_scene(&self) -> WorldScene {
        WorldScene {
            name: self.name.clone(),
            version: self.version.clone(),
            entities: self.entities.clone(),
            scene_scripts: self.scene_scripts.clone(),
            scene_script_components: self.scene_script_components.clone(),
            respawn_enabled: self.respawn_enabled,
            respawn_y: self.respawn_y,
            sandbox: self.sandbox.clone(),
            hierarchy: self.hierarchy.clone(),
            transitions: self.transitions.clone(),
        }
    }

    pub fn ui_scenes(&self) -> Vec<UiScene> {
        self.ui_layouts
            .iter()
            .map(|named| {
                let mut layout = named.layout.clone();
                layout.layer_type = named.layer_type;
                UiScene {
                    name: named.name.clone(),
                    version: self.version.clone(),
                    alias: named.alias.clone(),
                    layer_type: named.layer_type,
                    layout,
                    hierarchy: SceneHierarchy::default(),
                    transitions: Vec::new(),
                }
            })
            .collect()
    }

    pub fn from_world_scene(world: WorldScene) -> Self {
        Self {
            name: world.name,
            version: world.version,
            domain: SceneDomain::World3d,
            entities: world.entities,
            scene_scripts: world.scene_scripts,
            scene_script_components: world.scene_script_components,
            respawn_enabled: world.respawn_enabled,
            respawn_y: world.respawn_y,
            sandbox: world.sandbox,
            hierarchy: world.hierarchy,
            transitions: world.transitions,
            ui_layouts: Vec::new(),
            ui_layout: UiLayout::default(),
        }
    }

    pub fn get_or_create_layout(&mut self, idx: usize) -> &mut NamedUiLayout {
        if self.ui_layouts.is_empty() {
            // Migrate legacy ui_layout if needed (simplified)
            self.ui_layouts.push(NamedUiLayout::default());
        }
        let idx = idx.min(self.ui_layouts.len() - 1);
        &mut self.ui_layouts[idx]
    }

    pub fn scene_script_names(&self) -> Vec<String> {
        if !self.scene_script_components.is_empty() {
            return self
                .scene_script_components
                .iter()
                .filter(|component| component.enabled)
                .map(ScriptComponent::runtime_name)
                .filter(|name| !name.is_empty())
                .take(MAX_SCENE_SCRIPTS_PER_SCENE)
                .collect();
        }

        Self::normalize_scene_script_names(self.scene_scripts.clone())
    }

    pub fn ensure_scene_script_components(&mut self) -> &mut Vec<ScriptComponent> {
        if self.scene_script_components.is_empty() {
            self.scene_script_components =
                Self::normalize_scene_script_names(self.scene_scripts.clone())
                    .into_iter()
                    .map(ScriptComponent::builtin)
                    .collect();
        }
        self.clamp_scene_script_components();
        &mut self.scene_script_components
    }

    pub fn sync_legacy_scene_script_fields(&mut self) {
        self.clamp_scene_script_components();
        self.scene_scripts = self
            .scene_script_components
            .iter()
            .filter(|component| component.enabled)
            .map(ScriptComponent::runtime_name)
            .filter(|name| !name.is_empty())
            .take(MAX_SCENE_SCRIPTS_PER_SCENE)
            .collect();
    }

    pub fn clamp_scene_script_components(&mut self) {
        self.scene_script_components
            .truncate(MAX_SCENE_SCRIPTS_PER_SCENE);
    }

    fn normalize_scene_script_names(names: Vec<String>) -> Vec<String> {
        names
            .into_iter()
            .map(|name| name.trim().to_string())
            .filter(|name| !name.is_empty())
            .take(MAX_SCENE_SCRIPTS_PER_SCENE)
            .collect()
    }

    pub fn create_test_scene() -> Self {
        let mut s = Scene::new("Test");
        s.respawn_enabled = true;
        s.respawn_y = -20.0;
        let m = Material::default();

        // Ground
        s.entities.push(SceneEntity {
            id: 1,
            name: "Ground".into(),
            position: [0.0, -1.1, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [200.0, 2.0, 200.0],
            entity_type: EntityType::Ground,
            material: m.clone(),
            material_slots: vec![],
            active_material_slot: 0,
            script: None,
            scripts: vec![],
            script_components: vec![],
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
            material_slots: vec![],
            active_material_slot: 0,
            script: None,
            scripts: vec![],
            script_components: vec![],
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
            material_slots: vec![],
            active_material_slot: 0,
            script: Some("Vehicle".into()),
            scripts: vec!["Vehicle".into()],
            script_components: vec![ScriptComponent::builtin("Vehicle")],
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
                material_slots: vec![],
                active_material_slot: 0,
                script: Some("CrowdAgent".into()),
                scripts: vec!["CrowdAgent".into()],
                script_components: vec![ScriptComponent::builtin("CrowdAgent")],
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
                material_slots: vec![],
                active_material_slot: 0,
                script: None,
                scripts: vec![],
                script_components: vec![],
                collision_enabled: false,
                layer: LAYER_ENVIRONMENT,
                is_static: true,
                animator_config: None,
                parent_id: None,
                fov: 0.785,
                deployed: false,
            });
        }
        // Player Start
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
            material_slots: vec![],
            active_material_slot: 0,
            script: None,
            scripts: vec![],
            script_components: vec![],
            collision_enabled: false,
            layer: LAYER_DEFAULT,
            is_static: true,
            animator_config: None,
            parent_id: None,
            fov: 0.785,
            deployed: false,
        });

        // Add a Directional Light (Sun) so we can see things!
        s.entities.push(SceneEntity {
            id: 999,
            name: "Sun".into(),
            position: [10.0, 50.0, 10.0],
            rotation: [-0.707, 0.0, 0.0, 0.707], // Pointing down
            scale: [1.0, 1.0, 1.0],
            entity_type: EntityType::Light {
                light_type: LightType::Directional,
                intensity: 5.0,
                range: 1000.0,
                color: [1.0, 1.0, 0.9],
            },
            material: Material::default(),
            material_slots: vec![],
            active_material_slot: 0,
            script: None,
            scripts: vec![],
            script_components: vec![],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vehicle_runtime_scripts_keep_builtin_driver_with_extra_script() {
        let mut scene = Scene::create_test_scene();
        let vehicle = scene
            .entities
            .iter_mut()
            .find(|entity| matches!(entity.entity_type, EntityType::Vehicle))
            .expect("test scene should contain a vehicle");
        vehicle.script_components = vec![ScriptComponent {
            name: "CustomVehicleLogic".to_string(),
            enabled: true,
            source: "script CustomVehicleLogic {}".to_string(),
            compile_mode: ScriptCompileMode::CustomNativeCache,
            cache_key: Some("__stfsc_script_test".to_string()),
        }];
        vehicle.sync_legacy_script_fields();

        let names = vehicle.runtime_script_names();
        assert_eq!(names[0], "Vehicle");
        assert!(names.iter().any(|name| name == "__stfsc_script_test"));
    }

    #[test]
    fn legacy_ground_and_vehicle_get_runtime_physics_defaults() {
        let mut scene = Scene::create_test_scene();
        let ground = scene
            .entities
            .iter_mut()
            .find(|entity| matches!(entity.entity_type, EntityType::Ground))
            .expect("test scene should contain ground");
        ground.collision_enabled = false;
        ground.layer = 0;
        assert!(ground.effective_collision_enabled());
        assert_eq!(ground.effective_layer(), LAYER_ENVIRONMENT);

        let vehicle = scene
            .entities
            .iter_mut()
            .find(|entity| matches!(entity.entity_type, EntityType::Vehicle))
            .expect("test scene should contain vehicle");
        vehicle.collision_enabled = false;
        vehicle.layer = 0;
        assert!(vehicle.effective_collision_enabled());
        assert_eq!(vehicle.effective_layer(), LAYER_VEHICLE);
    }

    #[test]
    fn scene_script_components_clamp_to_scene_limit_without_changing_entity_limit() {
        let mut scene = Scene::new("Procedural");
        scene.scene_script_components = (0..(MAX_SCENE_SCRIPTS_PER_SCENE + 3))
            .map(|index| ScriptComponent::builtin(format!("SceneScript{}", index)))
            .collect();
        scene.sync_legacy_scene_script_fields();

        assert_eq!(
            scene.scene_script_components.len(),
            MAX_SCENE_SCRIPTS_PER_SCENE
        );
        assert_eq!(scene.scene_scripts.len(), MAX_SCENE_SCRIPTS_PER_SCENE);
        assert_eq!(MAX_SCRIPTS_PER_ENTITY, 32);
    }
}
