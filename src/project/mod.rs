use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::fs;
use std::collections::HashMap;
use std::time::SystemTime;

pub mod export;
pub mod scene;

// ============================================================================
// CONSTANTS
// ============================================================================
pub const PROJECT_FORMAT_VERSION: &str = "1.0";
pub const ENGINE_VERSION: &str = "0.1.0";

// ============================================================================
// OPTIMIZATION & QUALITY SETTINGS
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptLevel {
    Debug,
    #[default]
    Release,
    ReleaseLTO,
}

impl OptLevel {
    pub fn name(&self) -> &'static str {
        match self {
            OptLevel::Debug => "Debug",
            OptLevel::Release => "Release",
            OptLevel::ReleaseLTO => "Release (LTO)",
        }
    }
    
    pub fn all() -> Vec<OptLevel> {
        vec![OptLevel::Debug, OptLevel::Release, OptLevel::ReleaseLTO]
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GraphicsQuality {
    Low,
    #[default]
    Medium,
    High,
    Ultra,
}

impl GraphicsQuality {
    pub fn name(&self) -> &'static str {
        match self {
            GraphicsQuality::Low => "Low",
            GraphicsQuality::Medium => "Medium",
            GraphicsQuality::High => "High",
            GraphicsQuality::Ultra => "Ultra",
        }
    }
    
    pub fn all() -> Vec<GraphicsQuality> {
        vec![GraphicsQuality::Low, GraphicsQuality::Medium, GraphicsQuality::High, GraphicsQuality::Ultra]
    }
}

// ============================================================================
// BUILD CONFIGURATION
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BuildConfiguration {
    pub optimization_level: OptLevel,
    pub graphics_quality: GraphicsQuality,
    pub enable_hand_tracking: bool,
    pub enable_passthrough: bool,
    pub enable_face_tracking: bool,
    pub target_fps: u32,
    /// Additional cargo features to enable
    pub cargo_features: Vec<String>,
    /// Custom environment variables for build
    pub env_vars: HashMap<String, String>,
}

impl Default for BuildConfiguration {
    fn default() -> Self {
        Self {
            optimization_level: OptLevel::Release,
            graphics_quality: GraphicsQuality::Medium,
            enable_hand_tracking: true,
            enable_passthrough: true,
            enable_face_tracking: false,
            target_fps: 72,
            cargo_features: Vec::new(),
            env_vars: HashMap::new(),
        }
    }
}

impl BuildConfiguration {
    pub fn for_linux() -> Self {
        Self {
            target_fps: 144,
            graphics_quality: GraphicsQuality::Ultra,
            enable_hand_tracking: false,
            enable_passthrough: false,
            enable_face_tracking: false,
            ..Default::default()
        }
    }
    
    pub fn for_quest3() -> Self {
        Self {
            target_fps: 90,
            graphics_quality: GraphicsQuality::High,
            ..Default::default()
        }
    }
    
    pub fn for_quest_pro() -> Self {
        Self {
            target_fps: 90,
            graphics_quality: GraphicsQuality::High,
            enable_face_tracking: true,
            ..Default::default()
        }
    }
    
    pub fn for_pc() -> Self {
        Self {
            target_fps: 120,
            graphics_quality: GraphicsQuality::Ultra,
            enable_hand_tracking: false,
            enable_passthrough: false,
            enable_face_tracking: false,
            ..Default::default()
        }
    }
}

// ============================================================================
// INPUT MAPPING
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub enum InputSource {
    // VR Controllers
    TriggerLeft,
    TriggerRight,
    GripLeft,
    GripRight,
    ButtonA,
    ButtonB,
    ButtonX,
    ButtonY,
    ThumbstickLeft,
    ThumbstickRight,
    ThumbstickLeftClick,
    ThumbstickRightClick,
    MenuButton,
    // Keyboard
    Key(String),
    // Mouse
    MouseButton(u8),
    MouseMove,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InputBinding {
    pub source: InputSource,
    /// Threshold for analog inputs (0.0 - 1.0)
    pub threshold: f32,
    /// Whether this binding is active
    pub enabled: bool,
}

impl Default for InputBinding {
    fn default() -> Self {
        Self {
            source: InputSource::ButtonA,
            threshold: 0.5,
            enabled: true,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct InputMappings {
    /// Map of action name to list of bindings
    pub actions: HashMap<String, Vec<InputBinding>>,
}

impl InputMappings {
    pub fn default_vr_mappings() -> Self {
        let mut actions = HashMap::new();
        
        // Common VR game actions
        actions.insert("jump".to_string(), vec![
            InputBinding { source: InputSource::ButtonA, threshold: 0.5, enabled: true },
        ]);
        actions.insert("interact".to_string(), vec![
            InputBinding { source: InputSource::TriggerRight, threshold: 0.7, enabled: true },
        ]);
        actions.insert("grab".to_string(), vec![
            InputBinding { source: InputSource::GripRight, threshold: 0.7, enabled: true },
        ]);
        actions.insert("grab_left".to_string(), vec![
            InputBinding { source: InputSource::GripLeft, threshold: 0.7, enabled: true },
        ]);
        actions.insert("pause".to_string(), vec![
            InputBinding { source: InputSource::MenuButton, threshold: 0.5, enabled: true },
        ]);
        actions.insert("fire".to_string(), vec![
            InputBinding { source: InputSource::TriggerRight, threshold: 0.9, enabled: true },
        ]);
        actions.insert("aim".to_string(), vec![
            InputBinding { source: InputSource::TriggerLeft, threshold: 0.5, enabled: true },
        ]);
        actions.insert("sprint".to_string(), vec![
            InputBinding { source: InputSource::ThumbstickLeftClick, threshold: 0.5, enabled: true },
        ]);
        
        Self { actions }
    }
}

// ============================================================================
// ASSET MANIFEST
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AssetEntry {
    /// Relative path from project root
    pub path: String,
    /// File size in bytes
    pub size: u64,
    /// Last modified timestamp (Unix epoch seconds)
    pub modified: u64,
    /// Asset-specific metadata
    pub metadata: HashMap<String, String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct AssetManifest {
    pub models: Vec<AssetEntry>,
    pub textures: Vec<AssetEntry>,
    pub audio: Vec<AssetEntry>,
    pub scenes: Vec<AssetEntry>,
    pub scripts: Vec<AssetEntry>,
    pub ui_layouts: Vec<AssetEntry>,
    /// Last time the manifest was refreshed
    pub last_scan: u64,
}

impl AssetManifest {
    /// Scan project directory and populate asset lists
    pub fn scan_from_path(root: &Path) -> Self {
        let mut manifest = Self::default();
        manifest.last_scan = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        
        // Scan models
        manifest.models = Self::scan_directory(root, "assets/models", &["glb", "gltf", "obj", "fbx"]);
        
        // Scan textures
        manifest.textures = Self::scan_directory(root, "assets/textures", &["png", "jpg", "jpeg", "ktx2", "dds"]);
        
        // Scan audio
        manifest.audio = Self::scan_directory(root, "audio", &["wav", "ogg", "mp3"]);
        manifest.audio.extend(Self::scan_directory(root, "assets/audio", &["wav", "ogg", "mp3"]));
        
        // Scan scenes
        manifest.scenes = Self::scan_directory(root, "scenes", &["json"]);
        
        // Scan scripts
        manifest.scripts = Self::scan_directory(root, "scripts", &["lua", "rhai", "json"]);
        
        // Scan UI layouts
        manifest.ui_layouts = Self::scan_directory(root, "ui", &["json"]);
        
        manifest
    }
    
    fn scan_directory(root: &Path, subdir: &str, extensions: &[&str]) -> Vec<AssetEntry> {
        let mut entries = Vec::new();
        let dir = root.join(subdir);
        
        if !dir.exists() {
            return entries;
        }
        
        Self::scan_recursive(&dir, root, extensions, &mut entries);
        entries
    }
    
    fn scan_recursive(dir: &Path, root: &Path, extensions: &[&str], entries: &mut Vec<AssetEntry>) {
        if let Ok(read_dir) = fs::read_dir(dir) {
            for entry in read_dir.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    Self::scan_recursive(&path, root, extensions, entries);
                } else if let Some(ext) = path.extension() {
                    let ext_str = ext.to_string_lossy().to_lowercase();
                    if extensions.iter().any(|e| *e == ext_str) {
                        if let Ok(metadata) = fs::metadata(&path) {
                            let relative = path.strip_prefix(root).unwrap_or(&path);
                            entries.push(AssetEntry {
                                path: relative.to_string_lossy().to_string(),
                                size: metadata.len(),
                                modified: metadata.modified()
                                    .ok()
                                    .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
                                    .map(|d| d.as_secs())
                                    .unwrap_or(0),
                                metadata: HashMap::new(),
                            });
                        }
                    }
                }
            }
        }
    }
    
    /// Get total asset count
    pub fn total_count(&self) -> usize {
        self.models.len() + self.textures.len() + self.audio.len() 
            + self.scenes.len() + self.scripts.len() + self.ui_layouts.len()
    }
    
    /// Get total size of all assets
    pub fn total_size(&self) -> u64 {
        let sum_size = |entries: &[AssetEntry]| entries.iter().map(|e| e.size).sum::<u64>();
        sum_size(&self.models) + sum_size(&self.textures) + sum_size(&self.audio)
            + sum_size(&self.scenes) + sum_size(&self.scripts) + sum_size(&self.ui_layouts)
    }
}

// ============================================================================
// PROCEDURAL GENERATION SETTINGS
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ProceduralGenSettings {
    pub enabled: bool,
    pub seed: u64,
    pub density: f32,
}

// ============================================================================
// TARGET PLATFORM
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetPlatform {
    Linux,
    PC,
    Quest3,
    QuestPro,
}

impl TargetPlatform {
    pub fn name(&self) -> &'static str {
        match self {
            TargetPlatform::Linux => "Linux Desktop",
            TargetPlatform::PC => "PC (Windows)",
            TargetPlatform::Quest3 => "Meta Quest 3",
            TargetPlatform::QuestPro => "Meta Quest Pro",
        }
    }
    
    pub fn all() -> Vec<TargetPlatform> {
        vec![TargetPlatform::Linux, TargetPlatform::PC, TargetPlatform::Quest3, TargetPlatform::QuestPro]
    }
    
    pub fn default_build_config(&self) -> BuildConfiguration {
        match self {
            TargetPlatform::Linux => BuildConfiguration::for_linux(),
            TargetPlatform::PC => BuildConfiguration::for_pc(),
            TargetPlatform::Quest3 => BuildConfiguration::for_quest3(),
            TargetPlatform::QuestPro => BuildConfiguration::for_quest_pro(),
        }
    }
    
    /// Check if this is a VR platform
    pub fn is_vr(&self) -> bool {
        matches!(self, TargetPlatform::Quest3 | TargetPlatform::QuestPro)
    }
    
    /// Get the cargo target triple for cross-compilation
    pub fn cargo_target(&self) -> Option<&'static str> {
        match self {
            TargetPlatform::Linux => None, // Native, no cross-compile
            TargetPlatform::PC => Some("x86_64-pc-windows-gnu"),
            TargetPlatform::Quest3 | TargetPlatform::QuestPro => Some("aarch64-linux-android"),
        }
    }
}

impl Default for TargetPlatform {
    fn default() -> Self {
        TargetPlatform::Quest3
    }
}

// ============================================================================
// PROJECT METADATA
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProjectMetadata {
    /// Project format version for migrations
    pub version: String,
    /// Engine version this project was created/last saved with
    pub engine_version: String,
    /// Project name
    pub name: String,
    /// Optional description
    #[serde(default)]
    pub description: String,
    /// Optional author name
    #[serde(default)]
    pub author: String,
    /// Creation timestamp (ISO 8601)
    pub created_at: String,
    /// Last modification timestamp (ISO 8601)
    pub modified_at: String,
    /// Primary target platform
    pub target_platform: TargetPlatform,
    /// Procedural generation settings
    pub procedural_gen: ProceduralGenSettings,
    /// Build configurations per platform
    #[serde(default)]
    pub build_configs: HashMap<TargetPlatform, BuildConfiguration>,
    /// Default scene to load on startup
    #[serde(default)]
    pub startup_scene: Option<String>,
    /// Input action mappings
    #[serde(default)]
    pub input_mappings: InputMappings,
}

impl Default for ProjectMetadata {
    fn default() -> Self {
        let now = chrono_lite_now();
        let mut build_configs = HashMap::new();
        for platform in TargetPlatform::all() {
            build_configs.insert(platform, platform.default_build_config());
        }
        
        Self {
            version: PROJECT_FORMAT_VERSION.to_string(),
            engine_version: ENGINE_VERSION.to_string(),
            name: "Untitled Project".to_string(),
            description: String::new(),
            author: String::new(),
            created_at: now.clone(),
            modified_at: now,
            target_platform: TargetPlatform::Quest3,
            procedural_gen: ProceduralGenSettings::default(),
            build_configs,
            startup_scene: None,
            input_mappings: InputMappings::default_vr_mappings(),
        }
    }
}

/// Simple timestamp generation without external chrono dependency
fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let secs = duration.as_secs();
    
    // Convert to rough ISO 8601 format
    let days_since_epoch = secs / 86400;
    let remaining_secs = secs % 86400;
    let hours = remaining_secs / 3600;
    let minutes = (remaining_secs % 3600) / 60;
    let seconds = remaining_secs % 60;
    
    // Approximate year/month/day calculation (not accounting for leap years perfectly)
    let mut year = 1970;
    let mut remaining_days = days_since_epoch;
    loop {
        let days_in_year = if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }
    
    let days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 1;
    for &days in &days_in_months {
        let days = if month == 2 && year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) { 29 } else { days };
        if remaining_days < days {
            break;
        }
        remaining_days -= days;
        month += 1;
    }
    let day = remaining_days + 1;
    
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", year, month, day, hours, minutes, seconds)
}

// ============================================================================
// PROJECT VALIDATION
// ============================================================================

#[derive(Debug, Clone)]
pub enum ProjectWarning {
    MissingAsset(String),
    EmptyScene(String),
    NoScenes,
    OldFormatVersion(String),
}

#[derive(Debug, Clone)]
pub enum ProjectError {
    MissingProjectFile,
    CorruptedProjectFile(String),
    InvalidVersion(String),
}

// ============================================================================
// PROJECT
// ============================================================================

#[derive(Debug, Clone)]
pub struct Project {
    pub root_path: PathBuf,
    pub metadata: ProjectMetadata,
    /// Cached asset manifest (call refresh_assets to update)
    pub assets: AssetManifest,
}

impl Project {
    pub fn new(name: &str, root: PathBuf) -> Self {
        let metadata = ProjectMetadata {
            name: name.to_string(),
            ..Default::default()
        };
        
        Project {
            root_path: root,
            metadata,
            assets: AssetManifest::default(),
        }
    }

    pub fn save(&mut self) -> Result<(), String> {
        // Update modified timestamp
        self.metadata.modified_at = chrono_lite_now();
        
        let project_file = self.root_path.join("project.json");
        let json = serde_json::to_string_pretty(&self.metadata)
            .map_err(|e| format!("Failed to serialize metadata: {}", e))?;
        
        fs::create_dir_all(&self.root_path).map_err(|e| format!("Failed to create project directory: {}", e))?;
        fs::write(project_file, json).map_err(|e| format!("Failed to write project file: {}", e))?;
        
        // Ensure standard folders exist
        let dirs = ["assets", "assets/models", "assets/textures", "assets/audio", "scenes", "scripts", "ui"];
        for dir in dirs {
            fs::create_dir_all(self.root_path.join(dir)).map_err(|e| format!("Failed to create directory '{}': {}", dir, e))?;
        }
        
        Ok(())
    }

    pub fn load(root: PathBuf) -> Result<Self, String> {
        let project_file = root.join("project.json");
        if !project_file.exists() {
            return Err("Not a valid project directory (project.json missing)".into());
        }
        
        let contents = fs::read_to_string(&project_file).map_err(|e| format!("Failed to read project file: {}", e))?;
        
        // Try to load as new format first, then migrate if needed
        let metadata: ProjectMetadata = match serde_json::from_str(&contents) {
            Ok(m) => m,
            Err(_) => {
                // Try loading as legacy format and migrate
                Self::migrate_legacy_metadata(&contents)?
            }
        };
        
        let mut project = Project {
            root_path: root,
            metadata,
            assets: AssetManifest::default(),
        };
        
        // Refresh assets on load
        project.refresh_assets();
        
        Ok(project)
    }
    
    /// Migrate from older project.json format
    fn migrate_legacy_metadata(contents: &str) -> Result<ProjectMetadata, String> {
        #[derive(Deserialize)]
        struct LegacyMetadata {
            name: String,
            target_platform: TargetPlatform,
            procedural_gen: ProceduralGenSettings,
        }
        
        let legacy: LegacyMetadata = serde_json::from_str(contents)
            .map_err(|e| format!("Failed to parse legacy project file: {}", e))?;
        
        let now = chrono_lite_now();
        let mut build_configs = HashMap::new();
        for platform in TargetPlatform::all() {
            build_configs.insert(platform, platform.default_build_config());
        }
        
        Ok(ProjectMetadata {
            version: PROJECT_FORMAT_VERSION.to_string(),
            engine_version: ENGINE_VERSION.to_string(),
            name: legacy.name,
            description: String::new(),
            author: String::new(),
            created_at: now.clone(),
            modified_at: now,
            target_platform: legacy.target_platform,
            procedural_gen: legacy.procedural_gen,
            build_configs,
            startup_scene: None,
            input_mappings: InputMappings::default_vr_mappings(),
        })
    }
    
    /// Refresh the asset manifest by scanning the project directory
    pub fn refresh_assets(&mut self) {
        self.assets = AssetManifest::scan_from_path(&self.root_path);
    }
    
    /// Validate project integrity
    pub fn validate(&self) -> (Vec<ProjectWarning>, Vec<ProjectError>) {
        let mut warnings = Vec::new();
        let errors = Vec::new();
        
        // Check for scenes
        if self.assets.scenes.is_empty() {
            warnings.push(ProjectWarning::NoScenes);
        }
        
        // Check format version
        if self.metadata.version != PROJECT_FORMAT_VERSION {
            warnings.push(ProjectWarning::OldFormatVersion(self.metadata.version.clone()));
        }
        
        (warnings, errors)
    }
    
    pub fn get_asset_path(&self, relative_path: &str) -> PathBuf {
        self.root_path.join(relative_path)
    }
    
    pub fn resolve_path(&self, path: &str) -> PathBuf {
        let p = Path::new(path);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.root_path.join(path)
        }
    }
    
    /// Get the build configuration for the target platform (cloned for safety)
    pub fn get_build_config(&self) -> BuildConfiguration {
        self.metadata.build_configs
            .get(&self.metadata.target_platform)
            .cloned()
            .unwrap_or_else(|| self.metadata.target_platform.default_build_config())
    }
    
    /// Get mutable build configuration for the target platform
    pub fn get_build_config_mut(&mut self) -> &mut BuildConfiguration {
        let platform = self.metadata.target_platform;
        self.metadata.build_configs
            .entry(platform)
            .or_insert_with(|| platform.default_build_config())
    }
}

// ============================================================================
// RECENT PROJECTS
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RecentProject {
    pub name: String,
    pub path: String,
    pub last_opened: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct RecentProjects {
    pub projects: Vec<RecentProject>,
}

impl RecentProjects {
    const MAX_RECENT: usize = 10;
    
    /// Get the path to the recent projects file
    pub fn config_path() -> Option<PathBuf> {
        home::home_dir().map(|h| h.join(".stfsc").join("recent_projects.json"))
    }
    
    /// Load recent projects from disk
    pub fn load() -> Self {
        if let Some(path) = Self::config_path() {
            if let Ok(contents) = fs::read_to_string(&path) {
                if let Ok(recent) = serde_json::from_str(&contents) {
                    return recent;
                }
            }
        }
        Self::default()
    }
    
    /// Save recent projects to disk
    pub fn save(&self) -> Result<(), String> {
        if let Some(path) = Self::config_path() {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            let json = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
            fs::write(path, json).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    
    /// Add a project to the recent list (moves to front if already present)
    pub fn add(&mut self, name: &str, path: &str) {
        // Remove if already exists
        self.projects.retain(|p| p.path != path);
        
        // Add to front
        self.projects.insert(0, RecentProject {
            name: name.to_string(),
            path: path.to_string(),
            last_opened: chrono_lite_now(),
        });
        
        // Trim to max size
        self.projects.truncate(Self::MAX_RECENT);
    }
    
    /// Remove a project from the recent list
    pub fn remove(&mut self, path: &str) {
        self.projects.retain(|p| p.path != path);
    }
    
    /// Clear all recent projects
    pub fn clear(&mut self) {
        self.projects.clear();
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Format bytes to human-readable string
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    
    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
