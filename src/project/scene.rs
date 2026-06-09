use crate::ui::{UiLayerType, UiLayout};
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

pub const MAX_SCRIPTS_PER_ENTITY: usize = 32;
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
    pub entities: Vec<SceneEntity>,
    pub respawn_enabled: bool,
    pub respawn_y: f32,
    /// Per-scene sandbox settings. Disabled by default so existing scenes keep current behavior.
    #[serde(default)]
    pub sandbox: SandboxWorldSettings,
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
            entities: Vec::new(),
            respawn_enabled: false,
            respawn_y: -20.0,
            sandbox: SandboxWorldSettings::default(),
            ui_layouts: vec![NamedUiLayout::default()],
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
