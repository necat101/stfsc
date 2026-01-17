use serde::{Deserialize, Serialize};
use crate::world::animation::AnimatorConfig;
use crate::ui::{UiLayout, UiLayerType};
use crate::world::{LAYER_ENVIRONMENT, LAYER_PROP, LAYER_CHARACTER, LAYER_VEHICLE, LAYER_DEFAULT};


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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneEntity {
    pub id: u32,
    pub name: String,
    pub position: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
    pub entity_type: EntityType,
    pub material: Material,
    pub script: Option<String>,
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
