//! Bundled Project Loading
//!
//! Detects and loads pre-bundled project assets for standalone game builds.
//! This allows exported APKs to run without an editor connection.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(target_os = "android")]
use log::info;

/// Manifest for a bundled project
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BundledProjectManifest {
    /// Project name
    pub name: String,
    /// Path to the startup scene (relative to bundle)
    pub startup_scene: String,
    /// List of all bundled scene files
    pub scenes: Vec<String>,
    /// List of all bundled model files  
    pub models: Vec<String>,
    /// List of all bundled texture files
    pub textures: Vec<String>,
}

impl Default for BundledProjectManifest {
    fn default() -> Self {
        Self {
            name: "Bundled Project".into(),
            startup_scene: "main.bincode".into(),
            scenes: Vec::new(),
            models: Vec::new(),
            textures: Vec::new(),
        }
    }
}

/// A bundled project with pre-loaded scene data and assets
pub struct BundledProject {
    pub manifest: BundledProjectManifest,
    /// Pre-serialized scene updates as raw bytes (scene_name -> data)
    pub scene_data: HashMap<String, Vec<u8>>,
    /// Pre-loaded asset bytes (path -> data)
    pub assets: HashMap<String, Vec<u8>>,
}

impl Default for BundledProject {
    fn default() -> Self {
        Self {
            manifest: BundledProjectManifest::default(),
            scene_data: HashMap::new(),
            assets: HashMap::new(),
        }
    }
}

impl BundledProject {
    /// Create a new empty bundled project
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if bundled content exists and load the manifest
    #[cfg(target_os = "android")]
    pub fn load() -> Option<Self> {
        // info!("BundledProject::load() - asset loading requires AndroidApp context");
        None
    }
    
    /// Load bundled project with asset manager from AndroidApp
    #[cfg(target_os = "android")]
    pub fn load_with_asset_manager(&mut self, asset_manager: &ndk::asset::AssetManager) {
        // Try to open the bundle manifest
        if let Some(mut manifest_asset) = asset_manager.open(&std::ffi::CString::new("bundle/project.json").unwrap()) {
            if let Ok(manifest_bytes) = manifest_asset.buffer() {
                if let Ok(manifest) = serde_json::from_slice::<BundledProjectManifest>(manifest_bytes) {
                    info!("Loaded bundled project: {}", manifest.name);
                    self.manifest = manifest.clone();
                    
                    // Load scene data
                    for scene_name in &manifest.scenes {
                        let scene_path = std::ffi::CString::new(format!("bundle/scenes/{}", scene_name)).unwrap();
                        if let Some(mut scene_asset) = asset_manager.open(&scene_path) {
                            if let Ok(data) = scene_asset.buffer() {
                                info!("Loaded bundled scene: {} ({} bytes)", scene_name, data.len());
                                self.scene_data.insert(scene_name.clone(), data.to_vec());
                            }
                        }
                    }

                    // Load models
                    for model_path in &manifest.models {
                        let path = std::ffi::CString::new(format!("bundle/models/{}", model_path)).unwrap();
                        if let Some(mut model_asset) = asset_manager.open(&path) {
                            if let Ok(data) = model_asset.buffer() {
                                info!("Loaded bundled model: {} ({} bytes)", model_path, data.len());
                                self.assets.insert(model_path.clone(), data.to_vec());
                            }
                        }
                    }

                    // Load textures
                    for texture_path in &manifest.textures {
                        let path = std::ffi::CString::new(format!("bundle/textures/{}", texture_path)).unwrap();
                        if let Some(mut texture_asset) = asset_manager.open(&path) {
                            if let Ok(data) = texture_asset.buffer() {
                                info!("Loaded bundled texture: {} ({} bytes)", texture_path, data.len());
                                self.assets.insert(texture_path.clone(), data.to_vec());
                            }
                        }
                    }
                }
            }
        }
    }
    
    /// Non-Android stub - bundling only supported on Android for now
    #[cfg(not(target_os = "android"))]
    pub fn load() -> Option<Self> {
        // On desktop, check for bundle folder in current directory
        let manifest_path = std::path::Path::new("assets/bundle/project.json");
        if !manifest_path.exists() {
            return None;
        }
        
        let manifest_bytes = std::fs::read(manifest_path).ok()?;
        let manifest: BundledProjectManifest = serde_json::from_slice(&manifest_bytes).ok()?;
        
        let mut scene_data = HashMap::new();
        for scene_name in &manifest.scenes {
            let scene_path = format!("assets/bundle/scenes/{}", scene_name);
            if let Ok(data) = std::fs::read(&scene_path) {
                scene_data.insert(scene_name.clone(), data);
            }
        }

        let mut assets = HashMap::new();
        // Load models
        for model_path in &manifest.models {
            let path = format!("assets/bundle/models/{}", model_path);
            if let Ok(data) = std::fs::read(&path) {
                assets.insert(model_path.clone(), data);
            }
        }
        // Load textures
        for texture_path in &manifest.textures {
            let path = format!("assets/bundle/textures/{}", texture_path);
            if let Ok(data) = std::fs::read(&path) {
                assets.insert(texture_path.clone(), data);
            }
        }
        
        Some(BundledProject {
            manifest,
            scene_data,
            assets,
        })
    }
    
    /// Get the startup scene data
    pub fn get_startup_scene_data(&self) -> Option<&Vec<u8>> {
        self.scene_data.get(&self.manifest.startup_scene)
    }
}

/// Serialize scene updates to bytes for bundling
pub fn serialize_scene_updates(updates: &[crate::world::SceneUpdate]) -> Result<Vec<u8>, String> {
    bincode::serialize(updates).map_err(|e| format!("Serialization error: {}", e))
}

/// Deserialize scene updates from bundled bytes
pub fn deserialize_scene_updates(data: &[u8]) -> Result<Vec<crate::world::SceneUpdate>, String> {
    bincode::deserialize(data).map_err(|e| format!("Deserialization error: {}", e))
}
