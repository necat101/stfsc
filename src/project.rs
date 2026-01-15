use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::fs;

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ProceduralGenSettings {
    pub enabled: bool,
    pub seed: u64,
    pub density: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetPlatform {
    PC,
    Quest3,
    QuestPro,
}

impl TargetPlatform {
    pub fn name(&self) -> &'static str {
        match self {
            TargetPlatform::PC => "PC",
            TargetPlatform::Quest3 => "Oculus Quest 3",
            TargetPlatform::QuestPro => "Oculus Quest Pro",
        }
    }
    
    pub fn all() -> Vec<TargetPlatform> {
        vec![TargetPlatform::PC, TargetPlatform::Quest3, TargetPlatform::QuestPro]
    }
}

impl Default for TargetPlatform {
    fn default() -> Self {
        TargetPlatform::Quest3
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ProjectMetadata {
    pub name: String,
    pub target_platform: TargetPlatform,
    pub procedural_gen: ProceduralGenSettings,
}

impl Default for ProjectMetadata {
    fn default() -> Self {
        Self {
            name: "Untitled Project".to_string(),
            target_platform: TargetPlatform::Quest3,
            procedural_gen: ProceduralGenSettings::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Project {
    pub root_path: PathBuf,
    pub metadata: ProjectMetadata,
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
        }
    }

    pub fn save(&self) -> Result<(), String> {
        let project_file = self.root_path.join("project.json");
        let json = serde_json::to_string_pretty(&self.metadata)
            .map_err(|e| format!("Failed to serialize metadata: {}", e))?;
        
        fs::create_dir_all(&self.root_path).map_err(|e| format!("Failed to create project directory: {}", e))?;
        fs::write(project_file, json).map_err(|e| format!("Failed to write project file: {}", e))?;
        
        // Ensure standard folders exist
        let dirs = ["assets", "assets/models", "assets/textures", "scenes", "scripts"];
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
        
        let contents = fs::read_to_string(project_file).map_err(|e| format!("Failed to read project file: {}", e))?;
        let metadata: ProjectMetadata = serde_json::from_str(&contents).map_err(|e| format!("Failed to parse project file: {}", e))?;
        
        Ok(Project {
            root_path: root,
            metadata,
        })
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
}
