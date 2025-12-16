use eframe::egui;
use std::net::TcpStream;
use std::io::Write;
use stfsc_engine::world::{SceneUpdate, Mesh, Vertex, LightType};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use std::path::PathBuf;

fn main() -> Result<(), eframe::Error> {
    if let Err(e) = std::process::Command::new("adb")
        .args(&["forward", "tcp:8080", "tcp:8080"])
        .output() 
    { println!("Warning: adb forward failed: {}", e); }
    else { println!("ADB port forwarded."); }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 1000.0])
            .with_title("STFSC Editor - 556 Engine"),
        ..Default::default()
    };
    eframe::run_native("STFSC Editor", options, Box::new(|cc| {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        Box::new(EditorApp::new())
    }))
}

// ============================================================================
// 3D MATH
// ============================================================================
#[derive(Clone, Copy)]
struct Vec3 { x: f32, y: f32, z: f32 }
impl Vec3 {
    fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
    fn zero() -> Self { Self::new(0.0, 0.0, 0.0) }
    fn length(&self) -> f32 { (self.x*self.x + self.y*self.y + self.z*self.z).sqrt() }
    fn normalize(&self) -> Self { let l = self.length().max(0.0001); Self::new(self.x/l, self.y/l, self.z/l) }
    fn cross(&self, b: &Vec3) -> Vec3 { Vec3::new(self.y*b.z - self.z*b.y, self.z*b.x - self.x*b.z, self.x*b.y - self.y*b.x) }
    fn dot(&self, b: &Vec3) -> f32 { self.x*b.x + self.y*b.y + self.z*b.z }
    fn sub(&self, b: &Vec3) -> Vec3 { Vec3::new(self.x-b.x, self.y-b.y, self.z-b.z) }
    fn add(&self, b: &Vec3) -> Vec3 { Vec3::new(self.x+b.x, self.y+b.y, self.z+b.z) }
    fn mul(&self, s: f32) -> Vec3 { Vec3::new(self.x*s, self.y*s, self.z*s) }
}

struct Camera3D {
    target: Vec3, distance: f32, yaw: f32, pitch: f32, fov: f32,
}
impl Camera3D {
    fn new() -> Self { Self { target: Vec3::zero(), distance: 50.0, yaw: 0.5, pitch: 0.6, fov: 60.0 } }
    fn get_position(&self) -> Vec3 {
        Vec3::new(
            self.target.x + self.distance * self.pitch.cos() * self.yaw.sin(),
            self.target.y + self.distance * self.pitch.sin(),
            self.target.z + self.distance * self.pitch.cos() * self.yaw.cos(),
        )
    }
    fn get_forward(&self) -> Vec3 { self.target.sub(&self.get_position()).normalize() }
    fn get_right(&self) -> Vec3 { self.get_forward().cross(&Vec3::new(0.0, 1.0, 0.0)).normalize() }
    fn project(&self, world: Vec3, size: egui::Vec2) -> Option<egui::Pos2> {
        let cam = self.get_position();
        let fwd = self.get_forward();
        let right = self.get_right();
        let up = right.cross(&fwd);
        let rel = world.sub(&cam);
        let z = rel.dot(&fwd);
        if z < 0.5 { return None; }
        let x = rel.dot(&right);
        let y = rel.dot(&up);
        let aspect = size.x / size.y;
        let scale = (self.fov.to_radians() * 0.5).tan();
        let ndc_x = x / (z * scale * aspect);
        let ndc_y = y / (z * scale);
        if ndc_x.abs() > 2.0 || ndc_y.abs() > 2.0 { return None; }
        Some(egui::pos2(size.x * 0.5 + ndc_x * size.x * 0.5, size.y * 0.5 - ndc_y * size.y * 0.5))
    }
    fn get_ray(&self, screen_pos: egui::Pos2, size: egui::Vec2) -> (Vec3, Vec3) {
        let aspect = size.x / size.y;
        let scale = (self.fov.to_radians() * 0.5).tan();
        let ndc_x = (screen_pos.x / size.x - 0.5) * 2.0;
        let ndc_y = -(screen_pos.y / size.y - 0.5) * 2.0;
        let fwd = self.get_forward();
        let right = self.get_right();
        let up = right.cross(&fwd);
        let dir = fwd.add(&right.mul(ndc_x * scale * aspect)).add(&up.mul(ndc_y * scale)).normalize();
        (self.get_position(), dir)
    }
}

// ============================================================================
// PRIMITIVES LIBRARY (Unity-style)
// ============================================================================
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
enum PrimitiveType { Cube, Sphere, Cylinder, Plane, Capsule, Cone }

impl PrimitiveType {
    fn all() -> Vec<PrimitiveType> {
        vec![PrimitiveType::Cube, PrimitiveType::Sphere, PrimitiveType::Cylinder, 
             PrimitiveType::Plane, PrimitiveType::Capsule, PrimitiveType::Cone]
    }
    fn name(&self) -> &str {
        match self { 
            PrimitiveType::Cube => "Cube", PrimitiveType::Sphere => "Sphere",
            PrimitiveType::Cylinder => "Cylinder", PrimitiveType::Plane => "Plane",
            PrimitiveType::Capsule => "Capsule", PrimitiveType::Cone => "Cone",
        }
    }
    fn icon(&self) -> &str {
        match self {
            PrimitiveType::Cube => "🟧", PrimitiveType::Sphere => "⚪",
            PrimitiveType::Cylinder => "🔷", PrimitiveType::Plane => "▬",
            PrimitiveType::Capsule => "💊", PrimitiveType::Cone => "🔺",
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
}
impl Default for Material {
    fn default() -> Self { 
        Self { name: "Default".into(), albedo_color: [0.8, 0.8, 0.8], metallic: 0.0, roughness: 0.5, albedo_texture: None }
    }
}

// ============================================================================
// SCENE DATA
// ============================================================================
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct SceneEntity {
    id: u32, name: String, position: [f32; 3], rotation: [f32; 4], scale: [f32; 3],
    entity_type: EntityType, material: Material,
    #[serde(skip)] deployed: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum EntityType {
    Primitive(PrimitiveType),
    Mesh { path: String },
    Vehicle, CrowdAgent { state: String, speed: f32 }, Building { height: f32 }, Ground, Camera,
    /// Dynamic light source
    Light { light_type: LightTypeEditor, intensity: f32, range: f32, color: [f32; 3] },
}

/// Light types for editor
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
enum LightTypeEditor { Point, Spot, Directional }

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
struct Scene { name: String, version: String, entities: Vec<SceneEntity> }

impl Scene {
    fn new(name: &str) -> Self { Self { name: name.into(), version: "1.0".into(), entities: Vec::new() } }
    
    fn create_test_scene() -> Self {
        let mut s = Scene::new("Test");
        let m = Material::default();
        
        // Ground
        s.entities.push(SceneEntity { id: 1, name: "Ground".into(), position: [0.0, -0.5, 0.0], 
            rotation: [0.0,0.0,0.0,1.0], scale: [100.0,1.0,100.0], entity_type: EntityType::Ground, material: m.clone(), deployed: false });
        // Physics Cube
        s.entities.push(SceneEntity { id: 2, name: "Physics Cube".into(), position: [0.0, 10.0, -5.0],
            rotation: [0.0,0.0,0.0,1.0], scale: [1.0,1.0,1.0], 
            entity_type: EntityType::Primitive(PrimitiveType::Cube), 
            material: Material { albedo_color: [1.0, 0.5, 0.2], ..m.clone() }, deployed: false });
        // Vehicle
        s.entities.push(SceneEntity { id: 3, name: "Vehicle".into(), position: [5.0, 2.0, -8.0],
            rotation: [0.0,0.0,0.0,1.0], scale: [2.0,1.0,4.0], entity_type: EntityType::Vehicle, material: m.clone(), deployed: false });
        // Agents
        for i in 0..8 {
            let (state, speed, color) = if i < 2 { ("Fleeing", 8.0, [1.0,0.2,0.2]) } 
                else if i < 4 { ("Running", 5.0, [0.2,1.0,0.2]) } else { ("Walking", 2.0, [1.0,1.0,1.0]) };
            s.entities.push(SceneEntity { id: 100+i, name: format!("Agent {} ({})", i, state),
                position: [(i as f32 - 4.0) * 5.0, 1.0, -15.0], rotation: [0.0,0.0,0.0,1.0], scale: [0.5,1.8,0.5],
                entity_type: EntityType::CrowdAgent { state: state.into(), speed }, 
                material: Material { albedo_color: color, ..m.clone() }, deployed: false });
        }
        // Buildings
        for i in 0..4 {
            let h = 8.0 + (i as f32) * 5.0;
            s.entities.push(SceneEntity { id: 200+i, name: format!("Building {}", i+1),
                position: [25.0 + (i as f32)*12.0, h/2.0, -40.0], rotation: [0.0,0.0,0.0,1.0], scale: [6.0,h,6.0],
                entity_type: EntityType::Building { height: h }, 
                material: Material { albedo_color: [0.6, 0.6, 0.7], ..m.clone() }, deployed: false });
        }
        // Player Start - required for correct spawn position
        s.entities.push(SceneEntity { id: 0, name: "Player Start".into(), position: [0.0, 1.7, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0], scale: [0.5, 0.5, 0.5],
            entity_type: EntityType::Camera, material: Material { albedo_color: [0.3, 0.3, 1.0], ..m }, deployed: false });
        s
    }
}

enum AppCommand { Connect(String), Send(SceneUpdate) }
enum AppEvent { Connected, ConnectionError(String), SendError(String), StatusUpdate(String) }

// ============================================================================
// EDITOR
// ============================================================================
struct EditorApp {
    ip: String, status: String, command_tx: Sender<AppCommand>, event_rx: Receiver<AppEvent>, is_connected: bool,
    scenes: Vec<String>, current_scene: Option<Scene>, selected_scene_idx: Option<usize>, selected_entity_id: Option<u32>,
    show_new_scene_dialog: bool, new_scene_name: String,
    show_primitives_panel: bool,
    procedural_generation_enabled: bool, // Toggle for Quest procedural world gen

    camera: Camera3D, drag_button: Option<egui::PointerButton>, last_mouse: egui::Pos2, dragging_id: Option<u32>,
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
                        let addr = if ip.contains(':') { ip.clone() } else { format!("{}:8080", ip) };
                        match TcpStream::connect_timeout(&addr.parse().unwrap(), std::time::Duration::from_secs(5)) {
                            Ok(s) => { stream = Some(s); let _ = event_tx.send(AppEvent::Connected); }
                            Err(e) => { let _ = event_tx.send(AppEvent::ConnectionError(e.to_string())); }
                        }
                    }
                    AppCommand::Send(update) => {
                        if let Some(s) = &mut stream {
                            let bytes = bincode::serialize(&update).unwrap();
                            let len = bytes.len() as u32;
                            if s.write_all(&len.to_le_bytes()).is_ok() { let _ = s.write_all(&bytes); }
                        }
                    }
                }
            }
        });

        Self {
            ip: "127.0.0.1:8080".into(), status: "Disconnected".into(), command_tx, event_rx, is_connected: false,
            scenes: vec!["Test".into()], current_scene: Some(Scene::create_test_scene()), 
            selected_scene_idx: Some(0), selected_entity_id: None,
            show_new_scene_dialog: false, new_scene_name: String::new(),
            show_primitives_panel: false,
            procedural_generation_enabled: false, // Off by default
            camera: Camera3D::new(), drag_button: None, last_mouse: egui::pos2(0.0, 0.0), dragging_id: None,
        }
    }
    
    fn deploy_entity_to_quest(&self, entity: &SceneEntity) {
        match &entity.entity_type {
            EntityType::Camera => {
                let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::SetPlayerStart {
                    position: entity.position, rotation: entity.rotation,
                }));
            }
            EntityType::Light { light_type, intensity, range, color } => {
                // Convert editor light type to engine light type
                let engine_light_type = match light_type {
                    LightTypeEditor::Point => LightType::Point,
                    LightTypeEditor::Spot => LightType::Spot,
                    LightTypeEditor::Directional => LightType::Directional,
                };
                let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::SpawnLight {
                    id: entity.id,
                    light_type: engine_light_type,
                    position: entity.position,
                    direction: [0.0, -1.0, 0.0], // Default down direction
                    color: *color,
                    intensity: *intensity,
                    range: *range,
                    inner_cone: 0.4,
                    outer_cone: 0.6,
                }));
            }
            _ => {
                let color = entity.material.albedo_color;
                let primitive = match &entity.entity_type {
                    EntityType::Primitive(p) => match p {
                        PrimitiveType::Cube => 0,
                        PrimitiveType::Sphere => 1,
                        PrimitiveType::Cylinder => 2,
                        PrimitiveType::Plane => 3,
                        PrimitiveType::Capsule => 4,
                        PrimitiveType::Cone => 5,
                    },
                    EntityType::Ground => 0, // Use Cube for ground
                    EntityType::Vehicle => 0, // Use Cube for vehicle body
                    EntityType::Building { .. } => 0, // Use Cube for buildings
                    EntityType::CrowdAgent { .. } => 4, // Use Capsule for agents
                    _ => 0, // Default to Cube
                };
                
                let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::Spawn {
                    id: entity.id, primitive, position: entity.position, rotation: entity.rotation, color,
                }));
            }
        }
    }
    
    fn deploy_all_to_quest(&self) {
        if let Some(scene) = &self.current_scene {
            for entity in &scene.entities {
                self.deploy_entity_to_quest(entity);
            }
        }
    }
    
    fn add_primitive(&mut self, ptype: PrimitiveType) {
        if let Some(scene) = &mut self.current_scene {
            let id = scene.entities.iter().map(|e| e.id).max().unwrap_or(0) + 1;
            scene.entities.push(SceneEntity {
                id, name: format!("{} {}", ptype.name(), id),
                position: [0.0, 2.0, 0.0], rotation: [0.0,0.0,0.0,1.0], scale: [1.0,1.0,1.0],
                entity_type: EntityType::Primitive(ptype), material: Material::default(), deployed: false,
            });
            self.selected_entity_id = Some(id);
        }
    }
    
    fn draw_3d_viewport(&mut self, ui: &mut egui::Ui) {
        let available = ui.available_size();
        let (response, painter) = ui.allocate_painter(available, egui::Sense::click_and_drag());
        let rect = response.rect;
        
        // Camera controls - improved drag detection
        if response.drag_started() {
            // Unwrapping 0,0 is bad if we rely on it for raycasting. Use hover_pos or interact_pos from input if None.
            let start_pos = response.interact_pointer_pos()
                .or_else(|| ui.input(|i| i.pointer.interact_pos()))
                .or_else(|| ui.input(|i| i.pointer.hover_pos()))
                .unwrap_or(egui::pos2(0.0, 0.0));
                
            self.last_mouse = start_pos;
            
            self.drag_button = ui.input(|i| {
                // Allow "M" key + Left Click to simulate Middle Click (Drag Object)
                if i.key_down(egui::Key::M) && i.pointer.button_down(egui::PointerButton::Primary) {
                    Some(egui::PointerButton::Middle)
                }
                else if i.pointer.button_down(egui::PointerButton::Secondary) { Some(egui::PointerButton::Secondary) }
                else if i.pointer.button_down(egui::PointerButton::Middle) { Some(egui::PointerButton::Middle) }
                else if i.pointer.button_down(egui::PointerButton::Primary) { Some(egui::PointerButton::Primary) }
                else { None }
            });
            
            // Check for entity drag start (Middle Mouse)
            if self.drag_button == Some(egui::PointerButton::Middle) {
                 if let Some(scene) = &self.current_scene {
                     // Reverse iterate to find top-most object? Or just check all.
                     // Basic check using project() distance
                     for entity in &scene.entities {
                        let pos = Vec3::new(entity.position[0], entity.position[1], entity.position[2]);
                        if let Some(center) = self.camera.project(pos, available) {
                            let c = egui::pos2(rect.min.x + center.x, rect.min.y + center.y);
                            let size = (8.0 + 400.0 / self.camera.distance).clamp(4.0, 25.0);
                             if (start_pos.x - c.x).abs() < size + 5.0 && (start_pos.y - c.y).abs() < size + 5.0 {
                                self.dragging_id = Some(entity.id);
                                self.selected_entity_id = Some(entity.id);
                                break;
                             }
                        }
                     }
                 }
            }
        }
        
        if response.drag_released() { 
            self.drag_button = None; 
            self.dragging_id = None;
        }
        
        if response.dragged() {
            let mouse = response.interact_pointer_pos()
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
            if let (Some(p1), Some(p2)) = (self.camera.project(Vec3::new(x, 0.0, -100.0), available), 
                                            self.camera.project(Vec3::new(x, 0.0, 100.0), available)) {
                painter.line_segment([egui::pos2(rect.min.x + p1.x, rect.min.y + p1.y), 
                                      egui::pos2(rect.min.x + p2.x, rect.min.y + p2.y)], 
                                     egui::Stroke::new(0.5, egui::Color32::from_rgb(50, 50, 60)));
            }
            if let (Some(p1), Some(p2)) = (self.camera.project(Vec3::new(-100.0, 0.0, x), available), 
                                            self.camera.project(Vec3::new(100.0, 0.0, x), available)) {
                painter.line_segment([egui::pos2(rect.min.x + p1.x, rect.min.y + p1.y), 
                                      egui::pos2(rect.min.x + p2.x, rect.min.y + p2.y)], 
                                     egui::Stroke::new(0.5, egui::Color32::from_rgb(50, 50, 60)));
            }
        }
        
        // Origin axes
        let origin = Vec3::zero();
        if let Some(o) = self.camera.project(origin, available) {
            let o = egui::pos2(rect.min.x + o.x, rect.min.y + o.y);
            for (axis, color) in [(Vec3::new(5.0,0.0,0.0), egui::Color32::RED), 
                                   (Vec3::new(0.0,5.0,0.0), egui::Color32::GREEN), 
                                   (Vec3::new(0.0,0.0,5.0), egui::Color32::BLUE)] {
                if let Some(p) = self.camera.project(axis, available) {
                    painter.line_segment([o, egui::pos2(rect.min.x + p.x, rect.min.y + p.y)], egui::Stroke::new(2.0, color));
                }
            }
        }
        
        // Draw entities
        let mut clicked_entity: Option<u32> = None;
        if let Some(scene) = &self.current_scene {
            for entity in &scene.entities {
                let pos = Vec3::new(entity.position[0], entity.position[1], entity.position[2]);
                let half = Vec3::new(entity.scale[0]*0.5, entity.scale[1]*0.5, entity.scale[2]*0.5);
                let is_selected = self.selected_entity_id == Some(entity.id);
                
                let base_color = egui::Color32::from_rgb(
                    (entity.material.albedo_color[0] * 200.0) as u8,
                    (entity.material.albedo_color[1] * 200.0) as u8,
                    (entity.material.albedo_color[2] * 200.0) as u8,
                );
                let wire_color = if is_selected { egui::Color32::GOLD } else { base_color };
                let stroke = egui::Stroke::new(if is_selected { 2.5 } else { 1.0 }, wire_color);
                
                // Box corners
                let corners = [
                    Vec3::new(pos.x-half.x, pos.y-half.y, pos.z-half.z), Vec3::new(pos.x+half.x, pos.y-half.y, pos.z-half.z),
                    Vec3::new(pos.x+half.x, pos.y+half.y, pos.z-half.z), Vec3::new(pos.x-half.x, pos.y+half.y, pos.z-half.z),
                    Vec3::new(pos.x-half.x, pos.y-half.y, pos.z+half.z), Vec3::new(pos.x+half.x, pos.y-half.y, pos.z+half.z),
                    Vec3::new(pos.x+half.x, pos.y+half.y, pos.z+half.z), Vec3::new(pos.x-half.x, pos.y+half.y, pos.z+half.z),
                ];
                let proj: Vec<_> = corners.iter().map(|c| self.camera.project(*c, available)
                    .map(|p| egui::pos2(rect.min.x + p.x, rect.min.y + p.y))).collect();
                
                for (i, j) in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)] {
                    if let (Some(p1), Some(p2)) = (proj[i], proj[j]) {
                        painter.line_segment([p1, p2], stroke);
                    }
                }
                
                // Center dot + click detection
                if let Some(center) = self.camera.project(pos, available) {
                    let c = egui::pos2(rect.min.x + center.x, rect.min.y + center.y);
                    let size = (8.0 + 400.0 / self.camera.distance).clamp(4.0, 25.0);
                    painter.circle_filled(c, size, base_color.linear_multiply(0.6));
                    if is_selected { painter.circle_stroke(c, size + 2.0, egui::Stroke::new(2.0, egui::Color32::GOLD)); }
                    
                    if response.clicked() {
                        // Prevent clicking if we just dragged (simple check, or rely on clicked being false if dragged)
                         if (self.last_mouse.x - response.interact_pointer_pos().unwrap_or(self.last_mouse).x).abs() < 2.0 {
                            if let Some(click) = response.interact_pointer_pos() {
                                if (click.x - c.x).abs() < size + 5.0 && (click.y - c.y).abs() < size + 5.0 {
                                    clicked_entity = Some(entity.id);
                                }
                            }
                        }
                    }
                }
            }
        }
        if let Some(id) = clicked_entity { self.selected_entity_id = Some(id); }
        
        // Overlay
        painter.text(rect.min + egui::vec2(10.0, 10.0), egui::Align2::LEFT_TOP,
            "Right-drag: Orbit | Middle-drag or M+Left-drag: Move Object | Left-drag: Pan",
            egui::FontId::proportional(11.0), egui::Color32::WHITE);
    }
}

impl eframe::App for EditorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(ev) = self.event_rx.try_recv() {
            match ev {
                AppEvent::Connected => { self.status = "Connected ✓".into(); self.is_connected = true; }
                AppEvent::ConnectionError(e) => { self.status = format!("Error: {}", e); self.is_connected = false; }
                AppEvent::SendError(e) => { self.status = format!("Send Error: {}", e); }
                AppEvent::StatusUpdate(m) => { self.status = m; }
            }
        }

        // MENU
        egui::TopBottomPanel::top("menu").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("📄 New Scene").clicked() { self.show_new_scene_dialog = true; ui.close_menu(); }
                    if ui.button("🧪 Test Engine Scene").clicked() {
                        self.current_scene = Some(Scene::create_test_scene());
                        if !self.scenes.contains(&"Test".to_string()) { self.scenes.push("Test".into()); }
                        self.status = "Loaded Test Engine Scene".into();
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("💾 Save Scene").clicked() {
                        if let Some(scene) = &self.current_scene {
                            std::fs::create_dir_all("scenes").ok();
                            let path = format!("scenes/{}.json", scene.name.to_lowercase());
                            if let Ok(json) = serde_json::to_string_pretty(scene) {
                                match std::fs::write(&path, json) {
                                    Ok(_) => self.status = format!("Saved: {}", path),
                                    Err(e) => self.status = format!("Error: {}", e),
                                }
                            }
                        }
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Exit").clicked() { std::process::exit(0); }
                });
                ui.menu_button("Edit", |ui| {
                    if ui.add_enabled(self.selected_entity_id.is_some(), egui::Button::new("🗑 Delete Selected")).clicked() {
                        if let Some(id) = self.selected_entity_id {
                            if self.is_connected { let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::DeleteEntity { id })); }
                            if let Some(scene) = &mut self.current_scene { scene.entities.retain(|e| e.id != id); }
                            self.selected_entity_id = None;
                        }
                        ui.close_menu();
                    }
                });
                ui.menu_button("GameObject", |ui| {
                    ui.label("3D Primitives");
                    ui.separator();
                    for ptype in PrimitiveType::all() {
                        if ui.button(format!("{} {}", ptype.icon(), ptype.name())).clicked() {
                            self.add_primitive(ptype);
                            ui.close_menu();
                        }
                    }
                    ui.separator();
                    if ui.button("🚗 Vehicle").clicked() {
                        if let Some(scene) = &mut self.current_scene {
                            let id = scene.entities.iter().map(|e| e.id).max().unwrap_or(0) + 1;
                            scene.entities.push(SceneEntity {
                                id, name: format!("Vehicle {}", id), position: [0.0,1.0,0.0], rotation: [0.0,0.0,0.0,1.0], scale: [2.0,1.0,4.0],
                                entity_type: EntityType::Vehicle, material: Material { albedo_color: [1.0,1.0,0.0], ..Default::default() }, deployed: false,
                            });
                            self.selected_entity_id = Some(id);
                        }
                        ui.close_menu();
                    }
                    if ui.button("🚶 Crowd Agent").clicked() {
                        if let Some(scene) = &mut self.current_scene {
                            let id = scene.entities.iter().map(|e| e.id).max().unwrap_or(0) + 1;
                            scene.entities.push(SceneEntity {
                                id, name: format!("Agent {}", id), position: [0.0,1.0,0.0], rotation: [0.0,0.0,0.0,1.0], scale: [0.5,1.8,0.5],
                                entity_type: EntityType::CrowdAgent { state: "Walking".into(), speed: 2.0 }, material: Material::default(), deployed: false,
                            });
                            self.selected_entity_id = Some(id);
                        }
                        ui.close_menu();
                    }
                    if ui.button("🎥 Player Start").clicked() {
                        if let Some(scene) = &mut self.current_scene {
                            let id = scene.entities.iter().map(|e| e.id).max().unwrap_or(0) + 1;
                            scene.entities.push(SceneEntity {
                                id, name: "Player Start".into(), position: [0.0, 1.7, 0.0], rotation: [0.0, 0.0, 0.0, 1.0], scale: [0.5, 0.5, 0.5],
                                entity_type: EntityType::Camera, material: Material { albedo_color: [0.3, 0.3, 1.0], ..Default::default() }, deployed: false,
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
                                id, name: format!("Point Light {}", id), position: [0.0, 5.0, 0.0], rotation: [0.0, 0.0, 0.0, 1.0], scale: [0.5, 0.5, 0.5],
                                entity_type: EntityType::Light { light_type: LightTypeEditor::Point, intensity: 5.0, range: 20.0, color: [1.0, 1.0, 0.9] },
                                material: Material { albedo_color: [1.0, 1.0, 0.5], ..Default::default() }, deployed: false,
                            });
                            self.selected_entity_id = Some(id);
                        }
                        ui.close_menu();
                    }
                    if ui.button("🔦 Spot Light").clicked() {
                        if let Some(scene) = &mut self.current_scene {
                            let id = scene.entities.iter().map(|e| e.id).max().unwrap_or(0) + 1;
                            scene.entities.push(SceneEntity {
                                id, name: format!("Spot Light {}", id), position: [0.0, 10.0, 0.0], rotation: [0.0, 0.0, 0.0, 1.0], scale: [0.5, 0.5, 0.5],
                                entity_type: EntityType::Light { light_type: LightTypeEditor::Spot, intensity: 10.0, range: 30.0, color: [1.0, 1.0, 1.0] },
                                material: Material { albedo_color: [1.0, 0.8, 0.3], ..Default::default() }, deployed: false,
                            });
                            self.selected_entity_id = Some(id);
                        }
                        ui.close_menu();
                    }
                    if ui.button("☀️ Directional Light").clicked() {
                        if let Some(scene) = &mut self.current_scene {
                            let id = scene.entities.iter().map(|e| e.id).max().unwrap_or(0) + 1;
                            scene.entities.push(SceneEntity {
                                id, name: format!("Sun Light {}", id), position: [50.0, 100.0, 50.0], rotation: [0.0, 0.0, 0.0, 1.0], scale: [1.0, 1.0, 1.0],
                                entity_type: EntityType::Light { light_type: LightTypeEditor::Directional, intensity: 3.0, range: 1000.0, color: [1.0, 0.95, 0.8] },
                                material: Material { albedo_color: [1.0, 0.9, 0.5], ..Default::default() }, deployed: false,
                            });
                            self.selected_entity_id = Some(id);
                        }
                        ui.close_menu();
                    }
                });
                ui.menu_button("Scene", |ui| {
                    if ui.button("🚀 Deploy All to Quest").clicked() {
                        if self.is_connected { self.deploy_all_to_quest(); self.status = "Deployed scene to Quest!".into(); }
                        else { self.status = "Not connected!".into(); }
                        ui.close_menu();
                    }
                    if ui.button("🗑 Clear Quest Scene").clicked() {
                        if self.is_connected { let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::ClearScene)); }
                        ui.close_menu();
                    }
                    ui.separator();
                    // Procedural generation toggle
                    let label = if self.procedural_generation_enabled { "🌆 Procedural Gen: ON" } else { "🌆 Procedural Gen: OFF" };
                    if ui.button(label).clicked() {
                        self.procedural_generation_enabled = !self.procedural_generation_enabled;
                        if self.is_connected {
                            let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::SetProceduralGeneration { 
                                enabled: self.procedural_generation_enabled 
                            }));
                            self.status = format!("Procedural Gen: {}", if self.procedural_generation_enabled { "ON" } else { "OFF" });
                        }
                        ui.close_menu();
                    }
                });
                ui.menu_button("View", |ui| {
                    if ui.button("Reset Camera").clicked() { self.camera = Camera3D::new(); ui.close_menu(); }
                    if ui.button("Focus Selected").clicked() {
                        if let (Some(id), Some(scene)) = (self.selected_entity_id, &self.current_scene) {
                            if let Some(e) = scene.entities.iter().find(|e| e.id == id) {
                                self.camera.target = Vec3::new(e.position[0], e.position[1], e.position[2]);
                            }
                        }
                        ui.close_menu();
                    }
                });
                
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let c = if self.is_connected { egui::Color32::GREEN } else { egui::Color32::RED };
                    ui.colored_label(c, &self.status);
                });
            });
        });

        // STATUS BAR
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("Scene: {}", self.current_scene.as_ref().map(|s| s.name.as_str()).unwrap_or("None")));
                ui.separator();
                ui.label(format!("Entities: {}", self.current_scene.as_ref().map(|s| s.entities.len()).unwrap_or(0)));
                if let Some(id) = self.selected_entity_id { ui.separator(); ui.label(format!("Selected: #{}", id)); }
            });
        });

        // LEFT - PROJECT
        egui::SidePanel::left("project").default_width(160.0).show(ctx, |ui| {
            ui.heading("Project");
            
            egui::CollapsingHeader::new("🔌 Connection").default_open(true).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut self.ip).desired_width(90.0));
                    if ui.button(if self.is_connected { "✓" } else { "→" }).clicked() {
                        let _ = self.command_tx.send(AppCommand::Connect(self.ip.clone()));
                    }
                });
            });
            
            egui::CollapsingHeader::new("📁 Scenes").default_open(true).show(ui, |ui| {
                for (i, name) in self.scenes.iter().enumerate() {
                    if ui.selectable_label(self.selected_scene_idx == Some(i), format!(" 📄 {}", name)).clicked() {
                        self.selected_scene_idx = Some(i);
                        if name == "Test" { self.current_scene = Some(Scene::create_test_scene()); }
                    }
                }
            });
            
            ui.add_space(10.0);
            egui::CollapsingHeader::new("🧱 Primitives").default_open(true).show(ui, |ui| {
                for ptype in PrimitiveType::all() {
                    if ui.button(format!("{} {}", ptype.icon(), ptype.name())).clicked() {
                        self.add_primitive(ptype);
                    }
                }
            });
        });

        // RIGHT - INSPECTOR
        egui::SidePanel::right("inspector").default_width(260.0).show(ctx, |ui| {
            ui.heading("Inspector");
            ui.separator();
            
            let mut delete_id: Option<u32> = None;
            let mut deploy_id: Option<u32> = None;
            
            if let Some(id) = self.selected_entity_id {
                if let Some(scene) = &mut self.current_scene {
                    if let Some(entity) = scene.entities.iter_mut().find(|e| e.id == id) {
                        ui.horizontal(|ui| { ui.label("Name:"); ui.text_edit_singleline(&mut entity.name); });
                        ui.label(format!("ID: {} | Deployed: {}", entity.id, if entity.deployed { "✓" } else { "✗" }));
                        
                        ui.add_space(8.0);
                        ui.label("⚙ Transform");
                        egui::Grid::new("transform").show(ui, |ui| {
                            ui.label("Position"); 
                            ui.add(egui::DragValue::new(&mut entity.position[0]).speed(0.1).prefix("X:"));
                            ui.add(egui::DragValue::new(&mut entity.position[1]).speed(0.1).prefix("Y:"));
                            ui.add(egui::DragValue::new(&mut entity.position[2]).speed(0.1).prefix("Z:"));
                            ui.end_row();
                            ui.label("Scale");
                            ui.add(egui::DragValue::new(&mut entity.scale[0]).speed(0.1).prefix("X:"));
                            ui.add(egui::DragValue::new(&mut entity.scale[1]).speed(0.1).prefix("Y:"));
                            ui.add(egui::DragValue::new(&mut entity.scale[2]).speed(0.1).prefix("Z:"));
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
                            }
                        });
                        ui.add(egui::Slider::new(&mut entity.material.metallic, 0.0..=1.0).text("Metallic"));
                        ui.add(egui::Slider::new(&mut entity.material.roughness, 0.0..=1.0).text("Roughness"));
                        
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
                                }));
                            }
                            e.deployed = true;
                            self.status = format!("Deployed #{}", id);
                        }
                    }
                }
            }
            if let Some(id) = delete_id {
                if self.is_connected { let _ = self.command_tx.send(AppCommand::Send(SceneUpdate::DeleteEntity { id })); }
                if let Some(scene) = &mut self.current_scene { scene.entities.retain(|e| e.id != id); }
                self.selected_entity_id = None;
            }
        });

        // BOTTOM - HIERARCHY
        egui::TopBottomPanel::bottom("hierarchy").resizable(true).default_height(120.0).show(ctx, |ui| {
            ui.heading("Hierarchy");
            egui::ScrollArea::horizontal().show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    if let Some(scene) = &self.current_scene {
                        for entity in &scene.entities {
                            let icon = match &entity.entity_type {
                                EntityType::Primitive(p) => p.icon(),
                                EntityType::Vehicle => "🚗", EntityType::CrowdAgent {..} => "🚶",
                                EntityType::Building {..} => "🏢", EntityType::Ground => "🌍", EntityType::Mesh {..} => "📦",
                                EntityType::Camera => "🎥",
                                EntityType::Light { light_type, .. } => match light_type {
                                    LightTypeEditor::Point => "💡",
                                    LightTypeEditor::Spot => "🔦",
                                    LightTypeEditor::Directional => "☀️",
                                },
                            };
                            let deployed = if entity.deployed { "✓" } else { "" };
                            if ui.selectable_label(self.selected_entity_id == Some(entity.id), 
                                format!("{}{} {}", icon, deployed, entity.name)).clicked() {
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
            egui::Window::new("New Scene").collapsible(false).show(ctx, |ui| {
                ui.horizontal(|ui| { ui.label("Name:"); ui.text_edit_singleline(&mut self.new_scene_name); });
                ui.horizontal(|ui| {
                    if ui.button("Create").clicked() && !self.new_scene_name.is_empty() {
                        self.scenes.push(self.new_scene_name.clone());
                        self.current_scene = Some(Scene::new(&self.new_scene_name));
                        self.selected_scene_idx = Some(self.scenes.len() - 1);
                        self.new_scene_name.clear();
                        self.show_new_scene_dialog = false;
                    }
                    if ui.button("Cancel").clicked() { self.show_new_scene_dialog = false; }
                });
            });
        }
        
        ctx.request_repaint();
    }
}
