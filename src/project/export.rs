//! Project Export Module
//!
//! Handles packaging and exporting STFSC projects as distributable games
//! for various target platforms (Linux, Quest, etc.)

use std::fs;
use std::io::{self, BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::Sender;

use super::{Project, TargetPlatform, OptLevel};

// Cross-platform file permission helper
#[cfg(unix)]
fn make_executable(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))
}

#[cfg(windows)]
fn make_executable(_path: &Path) -> io::Result<()> {
    // Windows .exe files are inherently executable
    Ok(())
}

/// Result of an export operation
#[derive(Debug)]
pub struct ExportResult {
    pub success: bool,
    pub output_path: PathBuf,
    pub log: String,
    pub error: Option<String>,
}

/// Progress callback for export operations
pub type ProgressCallback = Box<dyn Fn(f32, &str) + Send>;

/// Project exporter for building distributable games
pub struct ProjectExporter {
    pub engine_root: PathBuf,
    pub output_dir: PathBuf,
}

impl ProjectExporter {
    /// Create a new exporter
    ///
    /// # Arguments
    /// * `engine_root` - Path to the STFSC engine source directory
    /// * `output_dir` - Where to place exported builds
    pub fn new(engine_root: PathBuf, output_dir: PathBuf) -> Self {
        Self {
            engine_root,
            output_dir,
        }
    }
    
    /// Run cargo command and stream output in real-time
    fn run_cargo_streaming(
        &self,
        args: &[&str],
        progress_tx: Option<&Sender<String>>,
    ) -> Result<String, String> {
        let mut child = Command::new("cargo")
            .args(args)
            .current_dir(&self.engine_root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn cargo: {}", e))?;
        
        let mut full_output = String::new();
        
        // Read stderr (where cargo sends most output)
        if let Some(stderr) = child.stderr.take() {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(line) = line {
                    full_output.push_str(&line);
                    full_output.push('\n');
                    if let Some(tx) = progress_tx {
                        let _ = tx.send(line);
                    }
                }
            }
        }
        
        // Wait for process to complete
        let status = child.wait().map_err(|e| format!("Failed to wait for cargo: {}", e))?;
        
        if status.success() {
            Ok(full_output)
        } else {
            Err(format!("Cargo failed with exit code: {:?}", status.code()))
        }
    }

    /// Export a project for the specified platform
    pub fn export(
        &self,
        project: &Project,
        platform: TargetPlatform,
        opt_level: OptLevel,
    ) -> ExportResult {
        let project_name = &project.metadata.name;
        let platform_dir = match platform {
            TargetPlatform::Linux => "linux",
            TargetPlatform::PC => "windows",
            TargetPlatform::Quest3 => "quest3",
            TargetPlatform::QuestPro => "quest_pro",
        };

        let export_path = self.output_dir.join(project_name).join(platform_dir);
        
        // Create export directory
        if let Err(e) = fs::create_dir_all(&export_path) {
            return ExportResult {
                success: false,
                output_path: export_path,
                log: String::new(),
                error: Some(format!("Failed to create export directory: {}", e)),
            };
        }

        // Export based on platform
        match platform {
            TargetPlatform::Linux => self.export_linux(project, &export_path, opt_level),
            TargetPlatform::Quest3 | TargetPlatform::QuestPro => {
                self.export_quest(project, &export_path, opt_level, platform)
            }
            TargetPlatform::PC => {
                // Windows desktop export
                self.export_windows(project, &export_path, opt_level)
            }
        }
    }
    
    /// Export a project with real-time progress streaming
    pub fn export_with_sender(
        &self,
        project: &Project,
        platform: TargetPlatform,
        opt_level: OptLevel,
        progress_tx: &Sender<String>,
    ) -> ExportResult {
        let project_name = &project.metadata.name;
        let platform_dir = match platform {
            TargetPlatform::Linux => "linux",
            TargetPlatform::PC => "windows",
            TargetPlatform::Quest3 => "quest3",
            TargetPlatform::QuestPro => "quest_pro",
        };

        let export_path = self.output_dir.join(project_name).join(platform_dir);
        
        // Create export directory
        if let Err(e) = fs::create_dir_all(&export_path) {
            return ExportResult {
                success: false,
                output_path: export_path,
                log: String::new(),
                error: Some(format!("Failed to create export directory: {}", e)),
            };
        }

        // Export based on platform with streaming
        match platform {
            TargetPlatform::Linux => self.export_linux_streaming(project, &export_path, opt_level, progress_tx),
            TargetPlatform::Quest3 | TargetPlatform::QuestPro => {
                self.export_quest_streaming(project, &export_path, opt_level, platform, progress_tx)
            }
            TargetPlatform::PC => {
                self.export_windows_streaming(project, &export_path, opt_level, progress_tx)
            }
        }
    }

    /// Export for Linux desktop
    fn export_linux(
        &self,
        project: &Project,
        export_path: &Path,
        opt_level: OptLevel,
    ) -> ExportResult {
        let mut log = String::new();
        
        log.push_str(&format!("=== Exporting {} for Linux ===\n", project.metadata.name));
        
        // Step 1: Copy assets
        log.push_str("Copying assets...\n");
        if let Err(e) = self.copy_assets(project, export_path) {
            return ExportResult {
                success: false,
                output_path: export_path.to_path_buf(),
                log,
                error: Some(format!("Failed to copy assets: {}", e)),
            };
        }
        log.push_str(&format!("  Copied {} assets\n", project.assets.total_count()));

        // Step 2: Build the engine binary
        log.push_str("Building Linux binary...\n");
        
        let cargo_args = match opt_level {
            OptLevel::Debug => vec!["build", "--bin", "stfsc_engine"],
            OptLevel::Release => vec!["build", "--release", "--bin", "stfsc_engine"],
            OptLevel::ReleaseLTO => vec!["build", "--release", "--bin", "stfsc_engine"],
        };

        let target_dir = match opt_level {
            OptLevel::Debug => "debug",
            OptLevel::Release | OptLevel::ReleaseLTO => "release",
        };

        let output = Command::new("cargo")
            .args(&cargo_args)
            .current_dir(&self.engine_root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();

        match output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                log.push_str(&stdout);
                log.push_str(&stderr);

                if !output.status.success() {
                    return ExportResult {
                        success: false,
                        output_path: export_path.to_path_buf(),
                        log,
                        error: Some("Cargo build failed".into()),
                    };
                }
            }
            Err(e) => {
                return ExportResult {
                    success: false,
                    output_path: export_path.to_path_buf(),
                    log,
                    error: Some(format!("Failed to run cargo: {}", e)),
                };
            }
        }

        // Step 3: Copy binary to export directory
        let binary_src = self.engine_root.join("target").join(target_dir).join("stfsc_engine");
        let binary_dst = export_path.join(&project.metadata.name);

        if let Err(e) = fs::copy(&binary_src, &binary_dst) {
            return ExportResult {
                success: false,
                output_path: export_path.to_path_buf(),
                log,
                error: Some(format!("Failed to copy binary: {}", e)),
            };
        }

        // Make executable
        let _ = make_executable(&binary_dst);

        log.push_str(&format!("Binary created: {}\n", binary_dst.display()));

        // Step 4: Create launch script
        let script_path = export_path.join("run.sh");
        let script_content = format!(
            "#!/bin/bash\ncd \"$(dirname \"$0\")\"\n./{}\n",
            project.metadata.name
        );
        if let Err(e) = fs::write(&script_path, script_content) {
            log.push_str(&format!("Warning: Failed to create launch script: {}\n", e));
        } else {
            let _ = make_executable(&script_path);
        }

        log.push_str("\n=== Export complete ===\n");

        ExportResult {
            success: true,
            output_path: export_path.to_path_buf(),
            log,
            error: None,
        }
    }
    
    /// Export for Linux desktop with streaming output
    fn export_linux_streaming(
        &self,
        project: &Project,
        export_path: &Path,
        opt_level: OptLevel,
        progress_tx: &Sender<String>,
    ) -> ExportResult {
        let mut log = String::new();
        
        let _ = progress_tx.send(format!("=== Exporting {} for Linux ===", project.metadata.name));
        
        // Step 1: Copy assets
        let _ = progress_tx.send("📦 Copying assets...".into());
        if let Err(e) = self.copy_assets(project, export_path) {
            return ExportResult {
                success: false,
                output_path: export_path.to_path_buf(),
                log,
                error: Some(format!("Failed to copy assets: {}", e)),
            };
        }
        let _ = progress_tx.send(format!("✓ Copied {} assets", project.assets.total_count()));

        // Step 2: Build with streaming
        let _ = progress_tx.send("🔨 Compiling Linux binary...".into());
        
        let cargo_args: Vec<&str> = match opt_level {
            OptLevel::Debug => vec!["build", "--bin", "stfsc_engine"],
            OptLevel::Release => vec!["build", "--release", "--bin", "stfsc_engine"],
            OptLevel::ReleaseLTO => vec!["build", "--release", "--bin", "stfsc_engine"],
        };

        let target_dir = match opt_level {
            OptLevel::Debug => "debug",
            OptLevel::Release | OptLevel::ReleaseLTO => "release",
        };

        match self.run_cargo_streaming(&cargo_args, Some(progress_tx)) {
            Ok(output) => {
                log.push_str(&output);
            }
            Err(e) => {
                return ExportResult {
                    success: false,
                    output_path: export_path.to_path_buf(),
                    log,
                    error: Some(e),
                };
            }
        }

        // Step 3: Copy binary
        let _ = progress_tx.send("📋 Copying binary...".into());
        let binary_src = self.engine_root.join("target").join(target_dir).join("stfsc_engine");
        let binary_dst = export_path.join(&project.metadata.name);

        if let Err(e) = fs::copy(&binary_src, &binary_dst) {
            return ExportResult {
                success: false,
                output_path: export_path.to_path_buf(),
                log,
                error: Some(format!("Failed to copy binary: {}", e)),
            };
        }

        let _ = make_executable(&binary_dst);

        // Create launch script
        let script_path = export_path.join("run.sh");
        let script_content = format!(
            "#!/bin/bash\ncd \"$(dirname \"$0\")\"\n./{}\n",
            project.metadata.name
        );
        let _ = fs::write(&script_path, script_content);
        let _ = make_executable(&script_path);

        let _ = progress_tx.send(format!("✅ Linux export complete: {}", export_path.display()));

        ExportResult {
            success: true,
            output_path: export_path.to_path_buf(),
            log,
            error: None,
        }
    }
    
    /// Export for Quest VR with streaming output  
    fn export_quest_streaming(
        &self,
        project: &Project,
        export_path: &Path,
        opt_level: OptLevel,
        _platform: TargetPlatform,
        progress_tx: &Sender<String>,
    ) -> ExportResult {
        let mut log = String::new();
        
        let _ = progress_tx.send(format!("=== Exporting {} for Quest ===", project.metadata.name));

        // Create bundle directory
        let engine_assets = self.engine_root.join("assets");
        let bundle_dir = engine_assets.join("bundle");
        let scenes_dir = bundle_dir.join("scenes");
        
        if let Err(e) = fs::create_dir_all(&scenes_dir) {
            return ExportResult {
                success: false,
                output_path: export_path.to_path_buf(),
                log,
                error: Some(format!("Failed to create bundle directory: {}", e)),
            };
        }
        let _ = progress_tx.send("📁 Created bundle directory".into());

        // Create manifest
        let _ = progress_tx.send("📝 Creating project manifest...".into());
        let startup_scene = if !project.assets.scenes.is_empty() {
            project.assets.scenes.first()
                .map(|s| {
                    let name = Path::new(&s.path).file_stem()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| "scene".into());
                    format!("{}.json", name)
                })
                .unwrap_or_else(|| "main.json".into())
        } else {
            "main.json".into()
        };
        
        let manifest = crate::bundle::BundledProjectManifest {
            name: project.metadata.name.clone(),
            startup_scene,
            scenes: project.assets.scenes.iter()
                .map(|s| {
                    let name = Path::new(&s.path).file_stem()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| "scene".into());
                    format!("{}.json", name)
                })
                .collect(),
            models: project.assets.models.iter()
                .filter_map(|m| Path::new(&m.path).file_name().map(|n| n.to_string_lossy().to_string()))
                .collect(),
            textures: project.assets.textures.iter()
                .filter_map(|t| Path::new(&t.path).file_name().map(|n| n.to_string_lossy().to_string()))
                .collect(),
        };
        
        let manifest_path = bundle_dir.join("project.json");
        if let Ok(json) = serde_json::to_string_pretty(&manifest) {
            let _ = fs::write(&manifest_path, &json);
        }

        // Bundle assets
        let _ = progress_tx.send("📦 Bundling scenes...".into());
        for scene_entry in &project.assets.scenes {
            let src_path = project.root_path.join(&scene_entry.path);
            if src_path.exists() {
                let scene_name = Path::new(&scene_entry.path).file_stem()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "scene".into());
                let dst_path = scenes_dir.join(format!("{}.json", scene_name));
                let _ = fs::copy(&src_path, &dst_path);
            }
        }

        let _ = progress_tx.send("📦 Bundling models...".into());
        let models_dst = bundle_dir.join("models");
        let _ = fs::create_dir_all(&models_dst);
        for model in &project.assets.models {
            let src = project.root_path.join(&model.path);
            if let Some(filename) = Path::new(&model.path).file_name() {
                let dst = models_dst.join(filename);
                if src.exists() {
                    let _ = fs::copy(&src, &dst);
                }
            }
        }

        let _ = progress_tx.send("📦 Bundling textures...".into());
        let textures_dst = bundle_dir.join("textures");
        let _ = fs::create_dir_all(&textures_dst);
        for texture in &project.assets.textures {
            let src = project.root_path.join(&texture.path);
            if let Some(filename) = Path::new(&texture.path).file_name() {
                let dst = textures_dst.join(filename);
                if src.exists() {
                    let _ = fs::copy(&src, &dst);
                }
            }
        }

        // Build APK with streaming
        let _ = progress_tx.send("🔨 Building Quest APK...".into());
        
        let cargo_args: Vec<&str> = match opt_level {
            OptLevel::Debug => vec!["apk", "build", "--lib"],
            OptLevel::Release | OptLevel::ReleaseLTO => vec!["apk", "build", "--release", "--lib"],
        };

        let target_dir = match opt_level {
            OptLevel::Debug => "debug",
            OptLevel::Release | OptLevel::ReleaseLTO => "release",
        };

        match self.run_cargo_streaming(&cargo_args, Some(progress_tx)) {
            Ok(output) => {
                log.push_str(&output);
            }
            Err(e) => {
                let _ = fs::remove_dir_all(&bundle_dir);
                return ExportResult {
                    success: false,
                    output_path: export_path.to_path_buf(),
                    log,
                    error: Some(e),
                };
            }
        }

        // Copy APK
        let _ = progress_tx.send("📋 Copying APK...".into());
        let apk_name = format!("{}.apk", project.metadata.name.replace(" ", "_"));
        let apk_src = self.engine_root.join("target").join(target_dir).join("apk").join("stfsc_engine.apk");
        let apk_dst = export_path.join(&apk_name);

        if apk_src.exists() {
            if let Err(e) = fs::copy(&apk_src, &apk_dst) {
                let _ = fs::remove_dir_all(&bundle_dir);
                return ExportResult {
                    success: false,
                    output_path: export_path.to_path_buf(),
                    log,
                    error: Some(format!("Failed to copy APK: {}", e)),
                };
            }
        } else {
            let _ = fs::remove_dir_all(&bundle_dir);
            return ExportResult {
                success: false,
                output_path: export_path.to_path_buf(),
                log,
                error: Some(format!("APK not found: {}", apk_src.display())),
            };
        }

        // Cleanup
        let _ = fs::remove_dir_all(&bundle_dir);

        let _ = progress_tx.send(format!("✅ Quest APK ready: {}", apk_dst.display()));

        ExportResult {
            success: true,
            output_path: export_path.to_path_buf(),
            log,
            error: None,
        }
    }

    /// Export for Quest VR
    fn export_quest(
        &self,
        project: &Project,
        export_path: &Path,
        opt_level: OptLevel,
        _platform: TargetPlatform,
    ) -> ExportResult {
        let mut log = String::new();
        
        log.push_str(&format!("=== Exporting {} for Quest ===\n", project.metadata.name));

        // Step 1: Create bundle directory structure in engine's assets folder
        let engine_assets = self.engine_root.join("assets");
        let bundle_dir = engine_assets.join("bundle");
        let scenes_dir = bundle_dir.join("scenes");
        
        if let Err(e) = fs::create_dir_all(&scenes_dir) {
            return ExportResult {
                success: false,
                output_path: export_path.to_path_buf(),
                log,
                error: Some(format!("Failed to create bundle directory: {}", e)),
            };
        }
        log.push_str("Created bundle directory structure\n");

        // Step 2: Create project manifest
        log.push_str("Creating project manifest...\n");
        let startup_scene = if !project.assets.scenes.is_empty() {
            // Use first scene as startup, or look for "main" scene
            project.assets.scenes.iter()
                .find(|s| s.path.to_lowercase().contains("main"))
                .or(project.assets.scenes.first())
                .map(|s| {
                    let name = Path::new(&s.path).file_stem()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| "scene".into());
                    format!("{}.json", name)
                })
                .unwrap_or_else(|| "main.json".into())
        } else {
            "main.json".into()
        };
        
        let manifest = crate::bundle::BundledProjectManifest {
            name: project.metadata.name.clone(),
            startup_scene: startup_scene.clone(),
            scenes: project.assets.scenes.iter()
                .map(|s| {
                    let name = Path::new(&s.path).file_stem()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| "scene".into());
                    format!("{}.json", name)
                })
                .collect(),
            models: project.assets.models.iter()
                .map(|m| Path::new(&m.path).file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default())
                .collect(),
            textures: project.assets.textures.iter()
                .map(|t| Path::new(&t.path).file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default())
                .collect(),
        };
        
        let manifest_path = bundle_dir.join("project.json");
        match serde_json::to_string_pretty(&manifest) {
            Ok(json) => {
                if let Err(e) = fs::write(&manifest_path, &json) {
                    return ExportResult {
                        success: false,
                        output_path: export_path.to_path_buf(),
                        log,
                        error: Some(format!("Failed to write manifest: {}", e)),
                    };
                }
                log.push_str(&format!("Manifest created: {}\n", manifest_path.display()));
            }
            Err(e) => {
                return ExportResult {
                    success: false,
                    output_path: export_path.to_path_buf(),
                    log,
                    error: Some(format!("Failed to serialize manifest: {}", e)),
                };
            }
        }

        // Step 3: Copy and serialize scenes
        log.push_str("Bundling scenes...\n");
        for scene_entry in &project.assets.scenes {
            let src_path = project.root_path.join(&scene_entry.path);
            if src_path.exists() {
                // Read the scene JSON and copy it to bundle (scenes will be parsed at runtime)
                let scene_name = Path::new(&scene_entry.path).file_stem()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "scene".into());
                
                // For now, just copy scene JSONs - the engine will parse them
                // In future: pre-serialize to SceneUpdate bincode
                let dst_path = scenes_dir.join(format!("{}.json", scene_name));
                if let Err(e) = fs::copy(&src_path, &dst_path) {
                    log.push_str(&format!("Warning: Failed to copy scene {}: {}\n", scene_name, e));
                } else {
                    log.push_str(&format!("  Bundled scene: {}\n", scene_name));
                }
            }
        }

        // Step 4: Copy models to bundle
        log.push_str("Bundling models...\n");
        let models_dst = bundle_dir.join("models");
        let _ = fs::create_dir_all(&models_dst);
        for model in &project.assets.models {
            let src = project.root_path.join(&model.path);
            let filename = Path::new(&model.path).file_name().unwrap_or_default();
            let dst = models_dst.join(filename);
            if src.exists() {
                if let Err(e) = fs::copy(&src, &dst) {
                    log.push_str(&format!("Warning: Failed to copy model {}: {}\n", 
                        filename.to_string_lossy(), e));
                }
            }
        }

        // Step 5: Copy textures to bundle
        log.push_str("Bundling textures...\n");
        let textures_dst = bundle_dir.join("textures");
        let _ = fs::create_dir_all(&textures_dst);
        for texture in &project.assets.textures {
            let src = project.root_path.join(&texture.path);
            let filename = Path::new(&texture.path).file_name().unwrap_or_default();
            let dst = textures_dst.join(filename);
            if src.exists() {
                if let Err(e) = fs::copy(&src, &dst) {
                    log.push_str(&format!("Warning: Failed to copy texture {}: {}\n", 
                        filename.to_string_lossy(), e));
                }
            }
        }

        log.push_str(&format!("Bundled {} models, {} textures\n", 
            project.assets.models.len(), project.assets.textures.len()));

        // Step 6: Build APK
        log.push_str("Building Quest APK...\n");
        
        let cargo_args = match opt_level {
            OptLevel::Debug => vec!["apk", "build"],
            OptLevel::Release | OptLevel::ReleaseLTO => vec!["apk", "build", "--release"],
        };

        let target_dir = match opt_level {
            OptLevel::Debug => "debug",
            OptLevel::Release | OptLevel::ReleaseLTO => "release",
        };

        let output = Command::new("cargo")
            .args(&cargo_args)
            .current_dir(&self.engine_root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();

        match output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                log.push_str(&stdout);
                log.push_str(&stderr);

                if !output.status.success() {
                    return ExportResult {
                        success: false,
                        output_path: export_path.to_path_buf(),
                        log,
                        error: Some("cargo apk build failed".into()),
                    };
                }
            }
            Err(e) => {
                return ExportResult {
                    success: false,
                    output_path: export_path.to_path_buf(),
                    log,
                    error: Some(format!("Failed to run cargo apk: {}", e)),
                };
            }
        }

        // Step 7: Copy APK to export directory
        let apk_name = format!("{}.apk", project.metadata.name.replace(" ", "_"));
        let apk_src = self.engine_root.join("target").join(target_dir).join("apk").join("stfsc_engine.apk");
        let apk_dst = export_path.join(&apk_name);

        if apk_src.exists() {
            if let Err(e) = fs::copy(&apk_src, &apk_dst) {
                return ExportResult {
                    success: false,
                    output_path: export_path.to_path_buf(),
                    log,
                    error: Some(format!("Failed to copy APK: {}", e)),
                };
            }
            log.push_str(&format!("APK created: {}\n", apk_dst.display()));
        } else {
            return ExportResult {
                success: false,
                output_path: export_path.to_path_buf(),
                log,
                error: Some(format!("APK not found at: {}", apk_src.display())),
            };
        }

        // Step 8: Cleanup - remove bundle from engine assets after build
        let _ = fs::remove_dir_all(&bundle_dir);

        log.push_str("\n=== Export complete ===\n");
        log.push_str("\nInstall with:\n");
        log.push_str(&format!("  adb install \"{}\"\n", apk_dst.display()));

        ExportResult {
            success: true,
            output_path: export_path.to_path_buf(),
            log,
            error: None,
        }
    }

    /// Export for Windows desktop
    fn export_windows(
        &self,
        project: &Project,
        export_path: &Path,
        opt_level: OptLevel,
    ) -> ExportResult {
        let mut log = String::new();
        
        log.push_str(&format!("=== Exporting {} for Windows ===\n", project.metadata.name));
        
        // Step 1: Copy assets
        log.push_str("Copying assets...\n");
        if let Err(e) = self.copy_assets(project, export_path) {
            return ExportResult {
                success: false,
                output_path: export_path.to_path_buf(),
                log,
                error: Some(format!("Failed to copy assets: {}", e)),
            };
        }
        log.push_str(&format!("  Copied {} assets\n", project.assets.total_count()));

        // Step 2: Build the engine binary
        log.push_str("Building Windows binary...\n");
        
        let cargo_args = match opt_level {
            OptLevel::Debug => vec!["build", "--bin", "stfsc_engine"],
            OptLevel::Release => vec!["build", "--release", "--bin", "stfsc_engine"],
            OptLevel::ReleaseLTO => vec!["build", "--release", "--bin", "stfsc_engine"],
        };

        let target_dir = match opt_level {
            OptLevel::Debug => "debug",
            OptLevel::Release | OptLevel::ReleaseLTO => "release",
        };

        let output = Command::new("cargo")
            .args(&cargo_args)
            .current_dir(&self.engine_root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();

        match output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                log.push_str(&stdout);
                log.push_str(&stderr);

                if !output.status.success() {
                    return ExportResult {
                        success: false,
                        output_path: export_path.to_path_buf(),
                        log,
                        error: Some("Cargo build failed".into()),
                    };
                }
            }
            Err(e) => {
                return ExportResult {
                    success: false,
                    output_path: export_path.to_path_buf(),
                    log,
                    error: Some(format!("Failed to run cargo: {}", e)),
                };
            }
        }

        // Step 3: Copy binary to export directory
        let binary_src = self.engine_root.join("target").join(target_dir).join("stfsc_engine.exe");
        let binary_dst = export_path.join(format!("{}.exe", project.metadata.name));

        if let Err(e) = fs::copy(&binary_src, &binary_dst) {
            return ExportResult {
                success: false,
                output_path: export_path.to_path_buf(),
                log,
                error: Some(format!("Failed to copy binary: {}", e)),
            };
        }

        // Windows .exe files are already executable, no need to set permissions
        log.push_str(&format!("Binary created: {}\n", binary_dst.display()));

        // Step 4: Create launch script (Windows batch file)
        let script_path = export_path.join("run.bat");
        let script_content = format!(
            "@echo off\r\ncd /D \"%~dp0\"\r\n\"{}.exe\"\r\n",
            project.metadata.name
        );
        if let Err(e) = fs::write(&script_path, script_content) {
            log.push_str(&format!("Warning: Failed to create launch script: {}\n", e));
        }

        log.push_str("\n=== Export complete ===\n");

        ExportResult {
            success: true,
            output_path: export_path.to_path_buf(),
            log,
            error: None,
        }
    }
    
    /// Export for Windows desktop with streaming output
    fn export_windows_streaming(
        &self,
        project: &Project,
        export_path: &Path,
        opt_level: OptLevel,
        progress_tx: &Sender<String>,
    ) -> ExportResult {
        let mut log = String::new();
        
        let _ = progress_tx.send(format!("=== Exporting {} for Windows ===", project.metadata.name));
        
        // Step 1: Copy assets
        let _ = progress_tx.send("📦 Copying assets...".into());
        if let Err(e) = self.copy_assets(project, export_path) {
            return ExportResult {
                success: false,
                output_path: export_path.to_path_buf(),
                log,
                error: Some(format!("Failed to copy assets: {}", e)),
            };
        }
        let _ = progress_tx.send(format!("✓ Copied {} assets", project.assets.total_count()));

        // Step 2: Build with streaming
        let _ = progress_tx.send("🔨 Compiling Windows binary...".into());
        
        let cargo_args: Vec<&str> = match opt_level {
            OptLevel::Debug => vec!["build", "--bin", "stfsc_engine"],
            OptLevel::Release => vec!["build", "--release", "--bin", "stfsc_engine"],
            OptLevel::ReleaseLTO => vec!["build", "--release", "--bin", "stfsc_engine"],
        };

        let target_dir = match opt_level {
            OptLevel::Debug => "debug",
            OptLevel::Release | OptLevel::ReleaseLTO => "release",
        };

        match self.run_cargo_streaming(&cargo_args, Some(progress_tx)) {
            Ok(output) => {
                log.push_str(&output);
            }
            Err(e) => {
                return ExportResult {
                    success: false,
                    output_path: export_path.to_path_buf(),
                    log,
                    error: Some(e),
                };
            }
        }

        // Step 3: Copy binary
        let _ = progress_tx.send("📋 Copying binary...".into());
        let binary_src = self.engine_root.join("target").join(target_dir).join("stfsc_engine.exe");
        let binary_dst = export_path.join(format!("{}.exe", project.metadata.name));

        if let Err(e) = fs::copy(&binary_src, &binary_dst) {
            return ExportResult {
                success: false,
                output_path: export_path.to_path_buf(),
                log,
                error: Some(format!("Failed to copy binary: {}", e)),
            };
        }

        // Create launch script (Windows batch file)
        let script_path = export_path.join("run.bat");
        let script_content = format!(
            "@echo off\r\ncd /D \"%~dp0\"\r\n\"{}.exe\"\r\n",
            project.metadata.name
        );
        let _ = fs::write(&script_path, script_content);

        let _ = progress_tx.send(format!("✅ Windows export complete: {}", export_path.display()));

        ExportResult {
            success: true,
            output_path: export_path.to_path_buf(),
            log,
            error: None,
        }
    }

    /// Copy project assets to export directory
    fn copy_assets(&self, project: &Project, export_path: &Path) -> io::Result<()> {
        let assets_dst = export_path.join("assets");
        fs::create_dir_all(&assets_dst)?;

        // Copy scenes
        let scenes_dst = export_path.join("scenes");
        fs::create_dir_all(&scenes_dst)?;
        for scene in &project.assets.scenes {
            let src = project.root_path.join(&scene.path);
            let dst = export_path.join(&scene.path);
            if let Some(parent) = dst.parent() {
                fs::create_dir_all(parent)?;
            }
            if src.exists() {
                fs::copy(&src, &dst)?;
            }
        }

        // Copy models
        for model in &project.assets.models {
            let src = project.root_path.join(&model.path);
            let dst = export_path.join(&model.path);
            if let Some(parent) = dst.parent() {
                fs::create_dir_all(parent)?;
            }
            if src.exists() {
                fs::copy(&src, &dst)?;
            }
        }

        // Copy textures
        for texture in &project.assets.textures {
            let src = project.root_path.join(&texture.path);
            let dst = export_path.join(&texture.path);
            if let Some(parent) = dst.parent() {
                fs::create_dir_all(parent)?;
            }
            if src.exists() {
                fs::copy(&src, &dst)?;
            }
        }

        // Copy audio
        for audio in &project.assets.audio {
            let src = project.root_path.join(&audio.path);
            let dst = export_path.join(&audio.path);
            if let Some(parent) = dst.parent() {
                fs::create_dir_all(parent)?;
            }
            if src.exists() {
                fs::copy(&src, &dst)?;
            }
        }

        // Copy scripts
        for script in &project.assets.scripts {
            let src = project.root_path.join(&script.path);
            let dst = export_path.join(&script.path);
            if let Some(parent) = dst.parent() {
                fs::create_dir_all(parent)?;
            }
            if src.exists() {
                fs::copy(&src, &dst)?;
            }
        }

        // Copy UI layouts
        for ui in &project.assets.ui_layouts {
            let src = project.root_path.join(&ui.path);
            let dst = export_path.join(&ui.path);
            if let Some(parent) = dst.parent() {
                fs::create_dir_all(parent)?;
            }
            if src.exists() {
                fs::copy(&src, &dst)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_export_result_creation() {
        let result = ExportResult {
            success: true,
            output_path: PathBuf::from("/tmp/test"),
            log: "Test log".into(),
            error: None,
        };
        assert!(result.success);
    }
}
