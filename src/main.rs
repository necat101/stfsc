use stfsc_engine::graphics::{GraphicsContext, GpuMesh, InstanceData, Texture, calculate_shadow_resolution};
use stfsc_engine::world::{GameWorld, MeshHandle, Material, Transform, RigidBodyHandle, Mesh, AudioSource};
use stfsc_engine::audio::{AudioSystem, AudioBuffer, AudioSourceProperties, AttenuationModel, AudioBufferHandle};
use stfsc_engine::physics::PhysicsWorld;
use stfsc_engine::resource_loader::{ResourceLoader, ResourceLoadResult};
use stfsc_engine::lighting::{self, LightUBO, GpuLightData};
use stfsc_engine::graphics::occlusion::OcclusionCuller;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{WindowBuilder, CursorGrabMode};
use winit::event::{Event, WindowEvent, DeviceEvent, ElementState, MouseButton};
use hecs::Entity;
use winit::keyboard::{PhysicalKey, KeyCode, Key, NamedKey};
use log::info;
use std::sync::{Arc, RwLock};
use ash::vk;
use std::collections::{HashMap, HashSet};
use rapier3d::prelude::{LockedAxes, vector};

// ============================================================================
// SCENE CONFIGURATION - Define these before graphics initialization
// This is where a "shader pre-compilation" phase would read scene metadata
// ============================================================================
const GROUND_PLANE_HALF_EXTENT: f32 = 100.0;  // Half-size of ground plane (200x200 total)

/// Cached texture entry with material descriptor set
struct TextureEntry {
    #[allow(dead_code)]
    texture: Texture,
    material_descriptor_set: vk::DescriptorSet,
}

fn main() {
    env_logger::init();

    info!("STFSC Engine - Linux Desktop Target");

    // ========================================================================
    // SCENE PRE-INITIALIZATION PHASE (like COD's shader compilation screen)
    // Calculate optimal shadow resolution based on scene requirements
    // ========================================================================
    let shadow_resolution = calculate_shadow_resolution(GROUND_PLANE_HALF_EXTENT);
    info!("Scene: Ground plane half-extent = {} units", GROUND_PLANE_HALF_EXTENT);
    info!("Scene: Calculated shadow map resolution = {}x{}", shadow_resolution, shadow_resolution);

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("STFSC Engine - 556 Downtown")
        .with_inner_size(winit::dpi::LogicalSize::new(1920.0, 1080.0))
        .build(&event_loop)
        .unwrap();

    // Create graphics context with scene-optimized shadow resolution
    let graphics_context = Arc::new(
        GraphicsContext::new_desktop_with_shadow_resolution(&window, Some(shadow_resolution))
            .expect("Failed to create GraphicsContext")
    );
    info!("Graphics: Shadow map initialized at {}x{}", 
          graphics_context.shadow_extent.width, 
          graphics_context.shadow_extent.height);
    
    // Create Default Textures
    let (albedo_tex, normal_tex, mr_tex) = graphics_context.create_default_pbr_textures().expect("Failed to create default textures");

    // Create Global Instance Buffer (10k instances)
    let max_instances = 10000;
    let (instance_buffer, instance_memory) = graphics_context.create_buffer(
        (max_instances * std::mem::size_of::<InstanceData>()) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    ).expect("Failed to create instance buffer");

    let instance_ptr = unsafe {
        graphics_context.device.map_memory(instance_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).expect("Failed to map instance buffer") as *mut InstanceData
    };

    // Create Light Uniform Buffer
    let light_buffer_size = std::mem::size_of::<LightUBO>() as u64;
    let (light_buffer, light_memory) = graphics_context.create_buffer(
        light_buffer_size,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    ).expect("Failed to create light buffer");

    let light_ptr = unsafe {
        graphics_context.device.map_memory(light_memory, 0, light_buffer_size, vk::MemoryMapFlags::empty()).expect("Failed to map light buffer") as *mut LightUBO
    };

    unsafe { *light_ptr = LightUBO::default(); }

    let global_descriptor_set = graphics_context.create_global_descriptor_set(
        graphics_context.shadow_depth_view,
        graphics_context.shadow_sampler,
        instance_buffer,
        light_buffer,
    ).expect("Failed to create global descriptor set");

    let material_descriptor_set = graphics_context.create_material_descriptor_set(&albedo_tex, &normal_tex, &mr_tex).expect("Failed to create material descriptor set");

    // Initialize Resource Loader
    let resource_loader = ResourceLoader::new(graphics_context.clone());

    // Initialize World & Physics
    let mut physics_world = PhysicsWorld::new();
    let mut game_world = GameWorld::new();
    let cmd_tx = game_world.command_sender.clone();
    
    struct GameState {
        physics: PhysicsWorld,
        world: GameWorld,
        player_entity: Entity,
        player_rb: RigidBodyHandle,
        player_position: glam::Vec3,
        player_yaw: f32,
        player_pitch: f32,
    }

    let player_pos = [0.0, 5.0, 5.0];
    let player_rb = physics_world.add_capsule_rigid_body(
        0, player_pos, 0.6, 0.3, true, 
        stfsc_engine::world::LAYER_DEFAULT, u32::MAX
    );
    // Lock rotations for player controller
    {
        if let Some(body) = physics_world.rigid_body_set.get_mut(player_rb) {
            body.set_locked_axes(LockedAxes::ROTATION_LOCKED, true);
            body.set_linear_damping(0.0);
        }
    }

    let player_entity = game_world.ecs.spawn((
        Transform {
            position: glam::Vec3::from(player_pos),
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
        },
        stfsc_engine::world::RigidBodyHandle(player_rb),
        stfsc_engine::world::Player,
    ));
    // Link user data
    if let Some(body) = physics_world.rigid_body_set.get_mut(player_rb) {
        body.user_data = player_entity.to_bits().get() as u128;
    }

    // --- TEST SCENE FOR ADVANCED SCRIPTING ---

    // 1. Static Physics Floor (using a Plane)
    // Plane is at Y=0 relative to pivot. We want it at Y=-0.1.
    // Scale is [200, 1, 200].
    let floor_pos = [0.0, -0.1, 0.0];
    let _floor_half_extents = [GROUND_PLANE_HALF_EXTENT, 0.1, GROUND_PLANE_HALF_EXTENT]; // Physics still needs thickness
    // For physics, a thin box is safer than a plane to prevent tunneling
    let collider_pos = [0.0, -1.1, 0.0];
    let collider_extents = [GROUND_PLANE_HALF_EXTENT, 1.0, GROUND_PLANE_HALF_EXTENT];
    let floor_rb = physics_world.add_box_rigid_body(0, collider_pos, collider_extents, false, stfsc_engine::world::LAYER_ENVIRONMENT, u32::MAX);
    
    let ground_scale = GROUND_PLANE_HALF_EXTENT * 2.0; // Full size = 2x half-extent
    let floor_id = game_world.ecs.spawn((
        stfsc_engine::world::StartupScene, // Marker for ClearScene
        Transform {
            position: glam::Vec3::from(floor_pos),
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::new(ground_scale, 1.0, ground_scale),
        },
        MeshHandle(3), // Plane
        Material { color: [0.15, 0.15, 0.15, 1.0], albedo_texture: None, metallic: 0.0, roughness: 0.5 },
        RigidBodyHandle(floor_rb),
        stfsc_engine::world::GroundPlane::new(GROUND_PLANE_HALF_EXTENT, GROUND_PLANE_HALF_EXTENT),
    ));
    physics_world.rigid_body_set.get_mut(floor_rb).unwrap().user_data = floor_id.to_bits().get() as u128;

    // 2. CollisionLogger Sphere (falls onto floor)
    let sphere_pos = [2.0, 8.0, 0.0];
    let sphere_rb = physics_world.add_sphere_rigid_body(0, sphere_pos, 0.5, true, stfsc_engine::world::LAYER_PROP, u32::MAX);
    let sphere_id = game_world.ecs.spawn((
        stfsc_engine::world::StartupScene, // Marker for ClearScene
        Transform {
            position: glam::Vec3::from(sphere_pos),
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::splat(1.0),
        },
        MeshHandle(1), // Sphere
        Material { color: [0.3, 0.8, 0.3, 1.0], albedo_texture: None, metallic: 0.0, roughness: 0.5 },
        RigidBodyHandle(sphere_rb),
    ));
    // Update physics user data with entity bits
    physics_world.rigid_body_set.get_mut(sphere_rb).unwrap().user_data = sphere_id.to_bits().get() as u128;
    // Attach Logger Script
    game_world.ecs.insert_one(sphere_id, stfsc_engine::world::DynamicScript::new(
        game_world.script_registry.create("CollisionLogger").unwrap()
    )).unwrap();

    // 3. TouchToDestroy Cube
    let cube_pos = [-2.0, 5.0, 0.0];
    let cube_rb = physics_world.add_box_rigid_body(0, cube_pos, [0.5, 0.5, 0.5], true, stfsc_engine::world::LAYER_PROP, stfsc_engine::world::LAYER_ENVIRONMENT | stfsc_engine::world::LAYER_PROP);
    let cube_id = game_world.ecs.spawn((
        stfsc_engine::world::StartupScene, // Marker for ClearScene
        Transform {
            position: glam::Vec3::from(cube_pos),
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::splat(1.0),
        },
        MeshHandle(0), 
        Material { color: [0.8, 0.3, 0.3, 1.0], albedo_texture: None, metallic: 0.0, roughness: 0.5 },
        RigidBodyHandle(cube_rb),
    ));
    physics_world.rigid_body_set.get_mut(cube_rb).unwrap().user_data = cube_id.to_bits().get() as u128;
    // Attach TouchToDestroy + TestBounce
    game_world.ecs.insert_one(cube_id, stfsc_engine::world::DynamicScript::new(
        game_world.script_registry.create("TouchToDestroy").unwrap()
    )).unwrap();
    // (Optional: Add TestBounce via separate DynamicScript if we supported multiple, 
    // but the current system replaced them. Let's stick to one primary test script per entity for now.)

    let game_state = Arc::new(RwLock::new(GameState {
        physics: physics_world,
        world: game_world,
        player_entity,
        player_rb: stfsc_engine::world::RigidBodyHandle(player_rb),
        player_position: glam::Vec3::from(player_pos), 
        player_yaw: -std::f32::consts::FRAC_PI_2, // Look towards -Z
        player_pitch: 0.0,
    }));

    // Initialize Audio System (kept on main thread - not Send/Sync)
    let mut audio_system = match AudioSystem::new() {
        Ok(sys) => {
            println!("AUDIO: Initialized successfully");
            Some(sys)
        }
        Err(e) => {
            println!("AUDIO: Failed to initialize: {:?}", e);
            None
        }
    };
    let mut audio_buffers: HashMap<String, AudioBufferHandle> = HashMap::new();
    let mut texture_cache: HashMap<String, TextureEntry> = HashMap::new();

    // Start TCP Listener
    let _game_state_net = game_state.clone();
    let cmd_tx_net = cmd_tx.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await.unwrap();
            println!("CONNECTED: Editor interface listening on 0.0.0.0:8080");
            info!("Editor interface listening on 0.0.0.0:8080");
            loop {
                if let Ok((mut socket, _)) = listener.accept().await {
                    let tx = cmd_tx_net.clone();
                    tokio::spawn(async move {
                        use tokio::io::AsyncReadExt;
                        loop {
                            let mut len_buf = [0u8; 4];
                            if socket.read_exact(&mut len_buf).await.is_err() { break; }
                            let len = u32::from_le_bytes(len_buf) as usize;
                            let mut data = vec![0u8; len];
                            if socket.read_exact(&mut data).await.is_err() { break; }
                            if let Ok(update) = bincode::deserialize::<stfsc_engine::world::SceneUpdate>(&data) {
                                match &update {
                                    stfsc_engine::world::SceneUpdate::UploadTexture { id, .. } => {
                                        println!("NETWORK: Received UploadTexture (ID: {}, Size: {} bytes)", id, len);
                                        info!("Received UploadTexture (ID: {}, Size: {} bytes)", id, len);
                                    }
                                    stfsc_engine::world::SceneUpdate::UploadSound { id, .. } => {
                                        println!("NETWORK: Received UploadSound (ID: {}, Size: {} bytes)", id, len);
                                        info!("Received UploadSound (ID: {}, Size: {} bytes)", id, len);
                                    }
                                    _ => {
                                        println!("NETWORK: Received SceneUpdate: {:?}", update);
                                        info!("Received client update: {:?}", update);
                                    }
                                }
                                let _ = tx.send(update).await;
                            } else {
                                println!("NETWORK ERROR: Failed to deserialize SceneUpdate (len: {})", len);
                                info!("Failed to deserialize SceneUpdate (len: {})", len);
                            }
                        }
                    });
                }
            }
        });
    });

    // Game Logic Thread
    let game_state_logic = game_state.clone();
    std::thread::spawn(move || {
        loop {
            let start = std::time::Instant::now();
            {
                if let Ok(mut state) = game_state_logic.write() {
                    state.physics.step();
            
                    // Sync player position FROM physics
                    let player_rb_handle = state.player_rb.0;
                    let (mut phys_pos, _rot) = {
                         let body = state.physics.rigid_body_set.get(player_rb_handle).unwrap();
                         (body.translation().clone(), body.rotation().clone())
                    };
                    
                    // Respawn Logic
                    if state.world.respawn_enabled && phys_pos.y < state.world.respawn_y {
                        let spawn_pos = state.world.player_start_transform.position;
                        if let Some(body) = state.physics.rigid_body_set.get_mut(player_rb_handle) {
                            body.set_translation(rapier3d::na::Vector3::new(spawn_pos.x, spawn_pos.y, spawn_pos.z), true);
                            body.set_linvel(rapier3d::na::Vector3::zeros(), true);
                            phys_pos = body.translation().clone();
                            println!("PLAYER: Respawned! (y < {})", state.world.respawn_y);
                        }
                    }

                    // Offset camera to eye level (0.8m above 0.9m center = 1.7m standing height)
                    let eye_offset = glam::Vec3::new(0.0, 0.8, 0.0); 
                    state.player_position = glam::Vec3::new(phys_pos.x, phys_pos.y, phys_pos.z) + eye_offset;
                    let pos = state.player_position; // Define pos as Glam Vec3 for update_streaming

                    // Sync player entity transform
                    let player_ent = state.player_entity;
                    if let Ok(mut t) = state.world.ecs.get::<&mut Transform>(player_ent) {
                        t.position = state.player_position;
                        t.rotation = glam::Quat::from_rotation_y(-state.player_yaw);
                    }

                    let GameState { world, physics, .. } = &mut *state;
                    world.update_streaming(pos, physics);
                    world.update_logic(physics, 0.016);
                    
                    // Sync physics to ECS - Parallelized gathering
                    use rayon::prelude::*;
                    let updates: Vec<_> = world.ecs.query::<(&Transform, &RigidBodyHandle)>()
                        .iter()
                        .map(|(id, (_, handle))| (id, handle.0)) // Copy handles out
                        .collect::<Vec<_>>() 
                        .into_par_iter()
                        .filter_map(|(id, handle_idx)| {
                            physics.rigid_body_set.get(handle_idx).map(|body| {
                                let p = body.translation();
                                let r = body.rotation();
                                (id, glam::Vec3::new(p.x, p.y, p.z), glam::Quat::from_xyzw(r.i, r.j, r.k, r.w))
                            })
                        })
                        .collect();

                    for (id, p, r) in updates {
                        if let Ok(mut t) = world.ecs.get::<&mut Transform>(id) {
                            t.position = p;
                            t.rotation = r;
                        }
                    }
                }
            }
            let elapsed = start.elapsed();
            if elapsed < std::time::Duration::from_millis(16) {
                std::thread::sleep(std::time::Duration::from_millis(16) - elapsed);
            }
        }
    });

    let mut mesh_library: Vec<GpuMesh> = Vec::new();
    let mut pending_uploads: HashSet<hecs::Entity> = HashSet::new();
    let mut occlusion_culler = OcclusionCuller::new();
    let mut prev_transforms: HashMap<u64, glam::Mat4> = HashMap::new();
    let mut prev_view_proj = glam::Mat4::IDENTITY;

    // Load Primitives
    for i in 0..6 {
        let mesh = stfsc_engine::world::create_primitive(i);
        let (vbo, vbo_mem) = graphics_context.create_buffer((mesh.vertices.len() * std::mem::size_of::<stfsc_engine::world::Vertex>()) as u64, vk::BufferUsageFlags::VERTEX_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT).unwrap();
        unsafe {
            let ptr = graphics_context.device.map_memory(vbo_mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap();
            std::ptr::copy_nonoverlapping(mesh.vertices.as_ptr(), ptr as *mut stfsc_engine::world::Vertex, mesh.vertices.len());
            graphics_context.device.unmap_memory(vbo_mem);
        }
        let (ibo, ibo_mem) = graphics_context.create_buffer((mesh.indices.len() * std::mem::size_of::<u32>()) as u64, vk::BufferUsageFlags::INDEX_BUFFER, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT).unwrap();
        unsafe {
            let ptr = graphics_context.device.map_memory(ibo_mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap();
            std::ptr::copy_nonoverlapping(mesh.indices.as_ptr(), ptr as *mut u32, mesh.indices.len());
            graphics_context.device.unmap_memory(ibo_mem);
        }
        mesh_library.push(GpuMesh {
            vertex_buffer: vbo, vertex_memory: vbo_mem,
            index_buffer: ibo, index_memory: ibo_mem,
            index_count: mesh.indices.len() as u32,
            material_descriptor_set,
            custom_textures: Vec::new(),
            aabb: {
                let mut min = glam::Vec3::splat(f32::MAX);
                let mut max = glam::Vec3::splat(f32::MIN);
                for v in &mesh.vertices {
                    let pos = glam::Vec3::from(v.position);
                    min = min.min(pos);
                    max = max.max(pos);
                }
                stfsc_engine::graphics::occlusion::AABB::new(min, max)
            }
        });
    }

    let mut cursor_captured = false;
    let mut keys_pressed: HashSet<PhysicalKey> = HashSet::new();
    let mut last_frame_time = std::time::Instant::now();

    // Main Loop
    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);

        match event {
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                if cursor_captured {
                    if let Ok(mut state) = game_state.write() {
                        let sensitivity = 0.002;
                        state.player_yaw += delta.0 as f32 * sensitivity; // Fixed: was -= (inverted)
                        state.player_pitch = (state.player_pitch - delta.1 as f32 * sensitivity).clamp(-1.5, 1.5);
                    }
                }
            }
            Event::WindowEvent { event: WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. }, .. } => {
                println!("CLICK: Left mouse button pressed, cursor_captured={}", cursor_captured);
                if !cursor_captured {
                    cursor_captured = true;
                    window.set_cursor_visible(false);
                    let result = window.set_cursor_grab(CursorGrabMode::Locked)
                        .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
                    println!("CURSOR: Captured (result: {:?})", result);
                }
            }
            Event::WindowEvent { 
                event: WindowEvent::KeyboardInput { 
                    event: key_event,
                    .. 
                }, .. 
            } => {
                let physical_key = key_event.physical_key;
                let state = key_event.state;
                
                match state {
                    ElementState::Pressed => {
                        keys_pressed.insert(physical_key);
                        
                        // Debug print for all keys in dev
                        println!("KEY: {:?} (Logical: {:?})", physical_key, key_event.logical_key);

                        // Escape OR Grave (backtick/tilde key) to release cursor
                        let is_escape = physical_key == PhysicalKey::Code(KeyCode::Escape) 
                            || key_event.logical_key == Key::Named(NamedKey::Escape);
                        let is_grave = physical_key == PhysicalKey::Code(KeyCode::Backquote)
                            || key_event.logical_key == Key::Character("`".into());

                        if is_escape || is_grave {
                            if cursor_captured {
                                println!("CURSOR: Releasing mouse grab (key: {:?}, logical: {:?})", physical_key, key_event.logical_key);
                                cursor_captured = false;
                                window.set_cursor_visible(true);
                                if let Err(e) = window.set_cursor_grab(CursorGrabMode::None) {
                                    println!("CURSOR ERROR: Failed to release grab: {:?}", e);
                                } else {
                                    println!("CURSOR: Grab released successfully");
                                }
                            }
                        }
                    }
                    ElementState::Released => {
                        keys_pressed.remove(&physical_key);
                    }
                }
            }
            Event::DeviceEvent { event: DeviceEvent::Key(key), .. } => {
                // Handle keyboard via DeviceEvent - more reliable when cursor is captured
                let physical_key = key.physical_key;
                match key.state {
                    ElementState::Pressed => {
                        keys_pressed.insert(physical_key);
                        // Debug for DeviceEvent keys
                        if matches!(physical_key, PhysicalKey::Code(KeyCode::KeyW) | PhysicalKey::Code(KeyCode::KeyA) | PhysicalKey::Code(KeyCode::KeyS) | PhysicalKey::Code(KeyCode::KeyD) | PhysicalKey::Code(KeyCode::Space)) {
                            println!("KEY (DeviceEvent): {:?}", physical_key);
                        }
                    }
                    ElementState::Released => {
                        keys_pressed.remove(&physical_key);
                    }
                }
                
                // Escape handling
                if physical_key == PhysicalKey::Code(KeyCode::Escape) && key.state == ElementState::Pressed {
                    if cursor_captured {
                        println!("CURSOR: Releasing mouse grab (DeviceEvent Escape)");
                        cursor_captured = false;
                        window.set_cursor_visible(true);
                        let _ = window.set_cursor_grab(CursorGrabMode::None);
                    }
                }
            }
            Event::WindowEvent { event: WindowEvent::Focused(focused), .. } => {
                println!("FOCUS: Window focused={}", focused);
                if !focused && cursor_captured {
                    println!("CURSOR: Releasing mouse grab (focus lost)");
                    cursor_captured = false;
                    window.set_cursor_visible(true);
                    let _ = window.set_cursor_grab(CursorGrabMode::None);
                }
            }
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                elwt.exit();
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {
                let dt = last_frame_time.elapsed().as_secs_f32();
                last_frame_time = std::time::Instant::now();

                // 0. Update Player Velocity - only when cursor is captured
                if cursor_captured {
                    if let Ok(mut state) = game_state.write() {
                        let mut move_dir = glam::Vec3::ZERO;
                        let forward = glam::Vec3::new(state.player_yaw.cos(), 0.0, state.player_yaw.sin()).normalize();
                        let right = glam::Vec3::new(-state.player_yaw.sin(), 0.0, state.player_yaw.cos()).normalize();

                        if keys_pressed.contains(&PhysicalKey::Code(KeyCode::KeyW)) { move_dir += forward; }
                        if keys_pressed.contains(&PhysicalKey::Code(KeyCode::KeyS)) { move_dir -= forward; }
                        if keys_pressed.contains(&PhysicalKey::Code(KeyCode::KeyA)) { move_dir -= right; }
                        if keys_pressed.contains(&PhysicalKey::Code(KeyCode::KeyD)) { move_dir += right; }

                        let mut speed = 5.0;
                        if keys_pressed.contains(&PhysicalKey::Code(KeyCode::ShiftLeft)) || keys_pressed.contains(&PhysicalKey::Code(KeyCode::ShiftRight)) {
                            speed = 12.0; // Sprint speed
                        }

                        let rb_handle = state.player_rb.0;
                        if let Some(body) = state.physics.rigid_body_set.get_mut(rb_handle) {
                            let current_vel = body.linvel();
                             let mut target_vel = glam::Vec3::ZERO;
                            if move_dir.length_squared() > 0.0 {
                                target_vel = move_dir.normalize() * speed;
                            }
                            
                            // Preserve Y velocity (gravity), set X/Z
                            let mut new_linvel = rapier3d::na::Vector3::new(target_vel.x, current_vel.y, target_vel.z);
                            
                            // Jump Logic
                             if keys_pressed.contains(&PhysicalKey::Code(KeyCode::Space)) {
                                 // Simple ground check: if nearly stationary vertically or raycasting (skipped for brevity)
                                 // We'll trust the user or check if Y velocity is small/negative (falling/grounded)
                                 // Ideally we use a shape cast. For now, allow jump if Vy is small.
                                 if current_vel.y.abs() < 0.1 {
                                     new_linvel.y = 5.0; // Jump force
                                     println!("PLAYER: Jump!");
                                 }
                            }
                            body.set_linvel(new_linvel, true);
                        }
                    }
                }

                // Audio Processing (on main thread - AudioSystem is not Send/Sync)
                if let Some(ref mut audio) = audio_system {
                    if let Ok(mut state) = game_state.write() {
                        // Process pending audio uploads
                        let pending: Vec<_> = state.world.pending_audio_uploads.drain().collect();
                        if !pending.is_empty() {
                            println!("AUDIO: Processing {} pending uploads", pending.len());
                        }
                        for (sound_id, data) in pending {
                            println!("AUDIO: Uploading buffer '{}' ({} bytes)", sound_id, data.len());
                            match audio.load_buffer_from_bytes(data) {
                                Ok(handle) => {
                                    println!("AUDIO: Loaded buffer for '{}'", sound_id);
                                    audio_buffers.insert(sound_id, handle);
                                }
                                Err(e) => {
                                    println!("AUDIO ERROR: Failed to load '{}': {:?}", sound_id, e);
                                }
                            }
                        }
                        
                        // Update listener position
                        let player_pos = state.player_position;
                        let player_yaw = state.player_yaw;
                        let player_pitch = state.player_pitch;
                        let rotation = glam::Quat::from_rotation_y(-player_yaw) * glam::Quat::from_rotation_x(player_pitch);
                        audio.set_listener_pose(player_pos, rotation);
                        
                        // Collect sources that need to be started
                        let mut sources_to_start = Vec::new();
                        let mut _audio_source_count = 0;
                        for (id, (transform, source)) in state.world.ecs.query::<(&Transform, &AudioSource)>().iter() {
                            _audio_source_count += 1;
                            if source.playing && source.runtime_handle.is_none() {
                                if let Some(&buffer_handle) = audio_buffers.get(&source.sound_id) {
                                    println!("AUDIO: Queuing sound '{}' for playback", source.sound_id);
                                    sources_to_start.push((id, transform.position, buffer_handle, source.volume, source.looping, source.max_distance));
                                } else if source.sound_id == "__test_tone__" || source.sound_id == "test_sound" {
                                    // Generate a test tone for debugging
                                    println!("AUDIO: Generating test tone for '{}'", source.sound_id);
                                    let buffer = AudioBuffer::test_tone(440.0, 2.0, 44100);
                                    let handle = audio.load_buffer(buffer);
                                    sources_to_start.push((id, transform.position, handle, source.volume, source.looping, source.max_distance));
                                } else {
                                    // Log missing buffer (will spam but helps debug)
                                    println!("AUDIO: Waiting for buffer '{}' (have {} buffers loaded)", source.sound_id, audio_buffers.len());
                                }
                            }
                        }
                        
                        // Start the sounds
                        for (entity_id, position, buffer_handle, volume, looping, max_distance) in sources_to_start {
                            let props = AudioSourceProperties {
                                position,
                                volume,
                                looping,
                                attenuation: AttenuationModel::InverseDistance {
                                    reference_distance: 1.0,
                                    max_distance,
                                    rolloff_factor: 1.0,
                                },
                                ..Default::default()
                            };
                            if let Some(audio_handle) = audio.play_3d(buffer_handle, props) {
                                println!("AUDIO: Started playing sound at {:?}", position);
                                if let Ok(mut src) = state.world.ecs.get::<&mut AudioSource>(entity_id) {
                                    src.runtime_handle = Some(audio_handle.0);
                                }
                            }
                        }
                    }
                    audio.update(dt);
                }

                // Texture Processing - Queue GPU textures for background loading
                if let Ok(mut state) = game_state.write() {
                    let pending: Vec<_> = state.world.pending_texture_uploads.drain().collect();
                    for (texture_id, data) in pending {
                        if !texture_cache.contains_key(&texture_id) {
                            println!("TEXTURE: Queuing background load for '{}' ({} bytes)", texture_id, data.len());
                            resource_loader.queue_texture(texture_id, data);
                        }
                    }
                }

                // 1. Wait for Fence
                unsafe {
                    graphics_context.device.wait_for_fences(&[graphics_context.fence], true, u64::MAX).unwrap();
                    graphics_context.device.reset_fences(&[graphics_context.fence]).unwrap();
                }

                // 2. Acquire Image
                let swapchain = graphics_context.swapchain.unwrap();
                let (image_index, _) = unsafe {
                    graphics_context.swapchain_loader.as_ref().unwrap().acquire_next_image(
                        swapchain, std::u64::MAX, graphics_context.image_available_semaphore.unwrap(), vk::Fence::null()
                    ).unwrap()
                };

                // 2. Resource Logic (Mesh Uploads)

                if let Ok(state) = game_state.try_read() {
                    for (id, (mesh, _)) in state.world.ecs.query::<(&Mesh, &Transform)>().iter() {
                        if state.world.ecs.get::<&GpuMesh>(id).is_ok() { continue; }
                        if pending_uploads.contains(&id) { continue; }
                        resource_loader.queue_mesh(id, mesh.clone());
                        pending_uploads.insert(id);
                    }
                }
                for result in resource_loader.poll_processed() {
                    match result {
                        ResourceLoadResult::Mesh(id, loaded_data) => {
                            pending_uploads.remove(&id);
                            let material_descriptor_set = graphics_context.create_material_descriptor_set(
                                loaded_data.albedo_texture.as_ref().unwrap_or(&albedo_tex), &normal_tex, &mr_tex
                            ).unwrap();
                            let gpu_mesh = GpuMesh {
                                vertex_buffer: loaded_data.vertex_buffer, vertex_memory: loaded_data.vertex_memory,
                                index_buffer: loaded_data.index_buffer, index_memory: loaded_data.index_memory,
                                index_count: loaded_data.index_count, material_descriptor_set,
                                custom_textures: Vec::new(), aabb: loaded_data.aabb,
                            };
                            if let Ok(mut state) = game_state.write() {
                                let _ = state.world.ecs.insert_one(id, gpu_mesh);
                            }
                        }
                        ResourceLoadResult::Texture(texture_id, texture) => {
                            // Create descriptor set with custom albedo + default normal/mr
                            match graphics_context.create_material_descriptor_set(&texture, &normal_tex, &mr_tex) {
                                Ok(descriptor_set) => {
                                    println!("TEXTURE: Background load complete for '{}' with descriptor set", texture_id);
                                    texture_cache.insert(texture_id, TextureEntry {
                                        texture,
                                        material_descriptor_set: descriptor_set,
                                    });
                                }
                                Err(e) => {
                                    println!("TEXTURE ERROR: Failed to create descriptor set for background load '{}': {:?}", texture_id, e);
                                }
                            }
                        }
                    }
                }

                // 3. Prepare Draw Data
                let mut batch_map: HashMap<usize, Vec<InstanceData>> = HashMap::new();
                let mut custom_draws: Vec<(GpuMesh, InstanceData, Option<vk::DescriptorSet>)> = Vec::new();
                let mut textured_draws: Vec<(usize, InstanceData, vk::DescriptorSet)> = Vec::new();
                let mut view_proj = glam::Mat4::IDENTITY;
                let mut player_pos = glam::Vec3::ZERO;
                
                if let Ok(state) = game_state.read() {
                    player_pos = state.player_position;
                    let forward = glam::Vec3::new(
                        state.player_yaw.cos() * state.player_pitch.cos(),
                        state.player_pitch.sin(),
                        state.player_yaw.sin() * state.player_pitch.cos(),
                    );
                    let view_matrix = glam::Mat4::look_at_rh(state.player_position, state.player_position + forward, glam::Vec3::Y);
                    let extent = graphics_context.swapchain_extent.unwrap_or(vk::Extent2D { width: 1920, height: 1080 });
                    let aspect = extent.width as f32 / (extent.height as f32).max(1.0);
                    
                    // Reversed-Z infinite far plane projection for maximum depth precision
                    // This eliminates z-fighting on large ground planes (556 Downtown open world)
                    let near = 0.1;
                    let fov = std::f32::consts::FRAC_PI_4;
                    let f = 1.0 / (fov / 2.0).tan();
                    
                    // Reversed-Z infinite projection matrix (near maps to 1.0, far maps to 0.0)
                    // This distributes depth precision evenly across the view frustum
                    let proj_matrix = glam::Mat4::from_cols(
                        glam::Vec4::new(f / aspect, 0.0, 0.0, 0.0),
                        glam::Vec4::new(0.0, -f, 0.0, 0.0),  // Flip Y for Vulkan
                        glam::Vec4::new(0.0, 0.0, 0.0, -1.0),
                        glam::Vec4::new(0.0, 0.0, near, 0.0),
                    );

                    view_proj = proj_matrix * view_matrix;

                    occlusion_culler.update_frustum(view_proj);

                    // Parallel preparation of render data
                    use rayon::prelude::*;
                    
                    // 1. Process regular MeshHandles
                    let mesh_data: Vec<_> = state.world.ecs.query::<(&MeshHandle, &Transform, &Material)>()
                        .iter()
                        .map(|(id, (h, t, m))| (id, *h, *t, m.clone()))
                        .collect();

                    let frustum = *occlusion_culler.get_frustum();

                    let mesh_results: Vec<_> = mesh_data.into_par_iter().filter_map(|(id, handle, transform, material)| {
                        if handle.0 >= mesh_library.len() { return None; }
                        
                        let gpu_mesh = &mesh_library[handle.0];
                        let model = glam::Mat4::from_scale_rotation_translation(transform.scale, transform.rotation, transform.position);
                        
                        if frustum.intersects_aabb(&gpu_mesh.aabb.transform(model)) {
                            let entity_id = id.id() as u64;
                            let color = material.color;
                            
                            // Check for custom texture (texture_cache is sync-friendly for reads)
                            let tex_info = material.albedo_texture.as_ref()
                                .and_then(|tex_id| texture_cache.get(tex_id).map(|entry| entry.material_descriptor_set));

                            Some((entity_id, handle.0, model, color, tex_info))
                        } else {
                            None
                        }
                    }).collect();

                    // 2. Process custom GpuMeshes
                    let custom_entities: Vec<_> = state.world.ecs.query::<(&GpuMesh, &Transform, &Material)>()
                        .iter()
                        .filter(|(id, _)| state.world.ecs.get::<&MeshHandle>(*id).is_err())
                        .map(|(id, (g, t, m))| (id, g.clone(), *t, m.clone()))
                        .collect();
                    
                    let custom_results: Vec<_> = custom_entities.into_par_iter().filter_map(|(id, gpu_mesh, transform, material)| {
                        let model = glam::Mat4::from_scale_rotation_translation(transform.scale, transform.rotation, transform.position);
                        if frustum.intersects_aabb(&gpu_mesh.aabb.transform(model)) {
                            let entity_id = id.id() as u64;
                            
                            // Check for custom texture (same as MeshHandles)
                            let tex_info = material.albedo_texture.as_ref()
                                .and_then(|tex_id| texture_cache.get(tex_id).map(|entry| entry.material_descriptor_set));
                            
                            Some((entity_id, gpu_mesh, model, material.color, tex_info))
                        } else {
                            None
                        }
                    }).collect();

                    // Apply results (sequential final batching)
                    for (entity_id, mesh_idx, model, color, tex_info) in mesh_results {
                        let prev_model = *prev_transforms.get(&entity_id).unwrap_or(&model);
                        prev_transforms.insert(entity_id, model);
                        
                        let instance = InstanceData { model, prev_model, color };
                        if let Some(desc_set) = tex_info {
                            textured_draws.push((mesh_idx, instance, desc_set));
                        } else {
                            batch_map.entry(mesh_idx).or_default().push(instance);
                        }
                    }

                    for (entity_id, gpu_mesh, model, color, tex_info) in custom_results {
                        let prev_model = *prev_transforms.get(&entity_id).unwrap_or(&model);
                        prev_transforms.insert(entity_id, model);
                        custom_draws.push((gpu_mesh, InstanceData { model, prev_model, color }, tex_info));
                    }

                    // Lights - Parallelized gathering
                    let light_data: Vec<_> = state.world.ecs.query::<(&stfsc_engine::world::LightComponent, &Transform)>()
                        .iter()
                        .map(|(id, (l, t))| (id, l.clone(), *t))
                        .collect::<Vec<_>>()
                        .into_par_iter()
                        .filter_map(|(_id, light, transform)| {
                            let direction = transform.rotation * glam::Vec3::NEG_Z;
                            Some(GpuLightData::from_light(&lighting::Light {
                                light_type: match light.light_type {
                                    stfsc_engine::world::LightType::Point => lighting::LightType::Point,
                                    stfsc_engine::world::LightType::Spot => lighting::LightType::Spot,
                                    stfsc_engine::world::LightType::Directional => lighting::LightType::Directional,
                                },
                                color: glam::Vec3::from_array([light.color[0], light.color[1], light.color[2]]),
                                intensity: light.intensity, range: light.range,
                                inner_cone_angle: light.inner_cone_angle, outer_cone_angle: light.outer_cone_angle,
                                cast_shadows: light.cast_shadows,
                            }, transform.position, direction, -1))
                        })
                        .collect();

                    let mut light_ubo = LightUBO::new();
                    for gpu_light in light_data {
                        if !light_ubo.add_light(gpu_light) { break; }
                    }
                    unsafe { *light_ptr = light_ubo; }
                }

                // 4. Upload Instances
                let mut instance_offset = 0;
                let mut draw_calls = Vec::new();
                for (mesh_idx, instances) in batch_map {
                    let count = instances.len();
                    if instance_offset + count > max_instances { break; }
                    unsafe { std::ptr::copy_nonoverlapping(instances.as_ptr(), instance_ptr.add(instance_offset), count); }
                    draw_calls.push((mesh_idx, instance_offset as u32, count as u32));
                    instance_offset += count;
                }
                let mut custom_draw_calls: Vec<(GpuMesh, u32, u32, Option<vk::DescriptorSet>)> = Vec::new();
                for (mesh, data, tex_info) in custom_draws {
                    if instance_offset + 1 > max_instances { break; }
                    unsafe { std::ptr::write(instance_ptr.add(instance_offset), data); }
                    custom_draw_calls.push((mesh, instance_offset as u32, 1u32, tex_info));
                    instance_offset += 1;
                }
                // Upload textured entities (individual draw calls with custom descriptor sets)
                let mut textured_draw_calls: Vec<(usize, u32, u32, vk::DescriptorSet)> = Vec::new();
                for (mesh_idx, data, descriptor_set) in textured_draws {
                    if instance_offset + 1 > max_instances { break; }
                    unsafe { std::ptr::write(instance_ptr.add(instance_offset), data); }
                    textured_draw_calls.push((mesh_idx, instance_offset as u32, 1u32, descriptor_set));
                    instance_offset += 1;
                }
                
                if !draw_calls.is_empty() || !custom_draw_calls.is_empty() || !textured_draw_calls.is_empty() {
                    println!("RENDER: Drawing {} batches, {} custom meshes, {} textured (Total instances: {})", 
                        draw_calls.len(), custom_draw_calls.len(), textured_draw_calls.len(), instance_offset);
                }

                // Dynamic shadow frustum based on GroundPlane component
                // Get ground center and extent for shadow coverage
                let (shadow_half_extent, ground_center) = if let Ok(state) = game_state.read() {
                    // Find the GroundPlane with largest extent and get its center
                    let mut max_extent = 50.0f32; // Default fallback
                    let mut ground_pos = glam::Vec3::ZERO; // Default center
                    for (_, (ground, transform)) in state.world.ecs.query::<(&stfsc_engine::world::GroundPlane, &Transform)>().iter() {
                        if ground.max_extent() > max_extent {
                            max_extent = ground.max_extent();
                            ground_pos = transform.position;
                        }
                    }
                    // Add 10% padding to ensure full coverage
                    (max_extent * 1.1, ground_pos)
                } else {
                    (120.0, glam::Vec3::ZERO) // Fallback if can't read state
                };
                
                // Shadow map snapping to prevent shimmering
                let world_size = shadow_half_extent * 2.0;
                let texel_size = world_size / graphics_context.shadow_extent.width as f32; 
                // Center shadow on ground plane, not player (for static ground)
                let mut shadow_center = glam::Vec3::new(ground_center.x, 0.0, ground_center.z);
                shadow_center.x = (shadow_center.x / texel_size).floor() * texel_size;
                shadow_center.z = (shadow_center.z / texel_size).floor() * texel_size;

                // Expand frustum by 100% to ensure the entire ground plane is covered
                // A 200x200 ground with half_extent=100 needs frustum_half=200 minimum
                let frustum_half = shadow_half_extent * 2.0;
                let shadow_proj_base = glam::Mat4::orthographic_rh(
                    -frustum_half, frustum_half, 
                    -frustum_half, frustum_half, 
                    0.1, frustum_half * 4.0  // Far plane for full depth coverage
                );
                let mut shadow_proj = shadow_proj_base;
                shadow_proj.col_mut(1).y *= -1.0; // Matrix Y flip

                // Correction for Vulkan Z [0, 1] range from glam's [-1, 1]
                let mut shadow_correction = glam::Mat4::IDENTITY;
                shadow_correction.col_mut(2).z = 0.5;
                shadow_correction.col_mut(3).z = 0.5;
                let shadow_proj = shadow_correction * shadow_proj;
                
                // Light pointing mostly down with very slight angle for visible shadows
                // Reduced XZ offset (1%) to prevent shadow frustum edge from cutting across ground
                let light_offset_xz = frustum_half * 0.01;
                let light_pos = shadow_center + glam::Vec3::new(light_offset_xz, frustum_half * 2.0, light_offset_xz);
                let light_view_proj = shadow_proj * glam::Mat4::look_at_rh(light_pos, shadow_center, glam::Vec3::Y);
                
                // Update LightUBO with correct directional light data
                let light_dir = (shadow_center - light_pos).normalize();
                let light_data = lighting::GpuLightData {
                    position_type: [light_pos.x, light_pos.y, light_pos.z, 2.0], // Type 2 = Directional
                    direction_range: [light_dir.x, light_dir.y, light_dir.z, 1000.0], // Infinite range
                    color_intensity: [1.0, 1.0, 1.0, 3.0], // Intensity 3.0
                    cone_shadow: [0.0, 0.0, 0.0, 0.0],
                };
                
                unsafe {
                    (*light_ptr).lights[0] = light_data;
                    (*light_ptr).num_lights = 1;
                    (*light_ptr).ambient = [0.15, 0.15, 0.15, 1.0];
                }

                // 5. Record Commands
                let cmd = graphics_context.command_buffer;
                unsafe {
                    graphics_context.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::RELEASE_RESOURCES).unwrap();
                    graphics_context.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)).unwrap();

                    // Set Viewport and Scissor (Dynamic State)
                    let viewport = vk::Viewport {
                        x: 0.0, y: 0.0,
                        width: graphics_context.swapchain_extent.unwrap().width as f32,
                        height: graphics_context.swapchain_extent.unwrap().height as f32,
                        min_depth: 0.0, max_depth: 1.0,
                    };
                    let scissor = vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: graphics_context.swapchain_extent.unwrap(),
                    };
                    graphics_context.device.cmd_set_viewport(cmd, 0, &[viewport]);
                    graphics_context.device.cmd_set_scissor(cmd, 0, &[scissor]);

                    // Shadow Pass
                    let shadow_rp_begin = vk::RenderPassBeginInfo::builder()
                        .render_pass(graphics_context.shadow_render_pass).framebuffer(graphics_context.shadow_framebuffer)
                        .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: graphics_context.shadow_extent })
                        .clear_values(&[vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } }]);  // Shadow uses standard depth (LESS)
                    graphics_context.device.cmd_begin_render_pass(cmd, &shadow_rp_begin, vk::SubpassContents::INLINE);
                    graphics_context.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, graphics_context.shadow_pipeline);
                    
                    // Set depth bias to prevent shadow acne (constant, clamp, slope)
                    // Much higher values needed for large 200+ unit ground planes with 2048 shadow map
                    // constant_bias: pushes all depths away, slope_bias: scales with surface angle
                    graphics_context.device.cmd_set_depth_bias(cmd, 200.0, 0.0, 25.0);
                    
                    graphics_context.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, graphics_context.shadow_pipeline_layout, 0, &[global_descriptor_set], &[]);
                    graphics_context.device.cmd_push_constants(cmd, graphics_context.shadow_pipeline_layout, vk::ShaderStageFlags::VERTEX, 0, bytemuck::bytes_of(&light_view_proj));
                    for (mesh_idx, first, count) in &draw_calls {
                        let m = &mesh_library[*mesh_idx];
                        graphics_context.device.cmd_bind_vertex_buffers(cmd, 0, &[m.vertex_buffer], &[0]);
                        graphics_context.device.cmd_bind_index_buffer(cmd, m.index_buffer, 0, vk::IndexType::UINT32);
                        graphics_context.device.cmd_draw_indexed(cmd, m.index_count, *count, 0, 0, *first);
                    }
                    for (m, first, count, _) in &custom_draw_calls {
                        graphics_context.device.cmd_bind_vertex_buffers(cmd, 0, &[m.vertex_buffer], &[0]);
                        graphics_context.device.cmd_bind_index_buffer(cmd, m.index_buffer, 0, vk::IndexType::UINT32);
                        graphics_context.device.cmd_draw_indexed(cmd, m.index_count, *count, 0, 0, *first);
                    }
                    graphics_context.device.cmd_end_render_pass(cmd);

                    // Main Pass
                    let clear_values = [
                        vk::ClearValue { color: vk::ClearColorValue { float32: [0.1, 0.1, 0.15, 1.0] } },
                        vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 } },  // Reversed-Z: clear to 0
                        vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] } },
                    ];
                    let main_rp_begin = vk::RenderPassBeginInfo::builder()
                        .render_pass(graphics_context.render_pass).framebuffer(graphics_context.swapchain_framebuffers[image_index as usize])
                        .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: graphics_context.swapchain_extent.unwrap() })
                        .clear_values(&clear_values);
                    graphics_context.device.cmd_begin_render_pass(cmd, &main_rp_begin, vk::SubpassContents::INLINE);
                    graphics_context.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, graphics_context.pipeline);
                    graphics_context.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, graphics_context.pipeline_layout, 0, &[global_descriptor_set], &[]);
                    
                    let mut push_data = Vec::new();
                    push_data.extend_from_slice(bytemuck::bytes_of(&view_proj));
                    push_data.extend_from_slice(bytemuck::bytes_of(&prev_view_proj));
                    push_data.extend_from_slice(bytemuck::bytes_of(&light_view_proj));
                    push_data.extend_from_slice(bytemuck::bytes_of(&glam::Vec4::from((player_pos, 1.0))));
                    graphics_context.device.cmd_push_constants(cmd, graphics_context.pipeline_layout, vk::ShaderStageFlags::VERTEX, 0, &push_data);

                    for (mesh_idx, first, count) in &draw_calls {
                        let m = &mesh_library[*mesh_idx];
                        graphics_context.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, graphics_context.pipeline_layout, 1, &[m.material_descriptor_set], &[]);
                        graphics_context.device.cmd_bind_vertex_buffers(cmd, 0, &[m.vertex_buffer], &[0]);
                        graphics_context.device.cmd_bind_index_buffer(cmd, m.index_buffer, 0, vk::IndexType::UINT32);
                        graphics_context.device.cmd_draw_indexed(cmd, m.index_count, *count, 0, 0, *first);
                    }
                    for (m, first, count, tex_info) in &custom_draw_calls {
                        // Bind material texture if available, otherwise fallback to GpuMesh's internal descriptor set
                        let descriptor_set = tex_info.unwrap_or(m.material_descriptor_set);
                        graphics_context.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, graphics_context.pipeline_layout, 1, &[descriptor_set], &[]);
                        graphics_context.device.cmd_bind_vertex_buffers(cmd, 0, &[m.vertex_buffer], &[0]);
                        graphics_context.device.cmd_bind_index_buffer(cmd, m.index_buffer, 0, vk::IndexType::UINT32);
                        graphics_context.device.cmd_draw_indexed(cmd, m.index_count, *count, 0, 0, *first);
                    }
                    
                    // Draw textured entities with their custom descriptor sets
                    for (mesh_idx, first, count, descriptor_set) in &textured_draw_calls {
                        let m = &mesh_library[*mesh_idx];
                        // Bind the entity's custom texture descriptor set
                        graphics_context.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, graphics_context.pipeline_layout, 1, &[*descriptor_set], &[]);
                        graphics_context.device.cmd_bind_vertex_buffers(cmd, 0, &[m.vertex_buffer], &[0]);
                        graphics_context.device.cmd_bind_index_buffer(cmd, m.index_buffer, 0, vk::IndexType::UINT32);
                        graphics_context.device.cmd_draw_indexed(cmd, m.index_count, *count, 0, 0, *first);
                    }

                    graphics_context.device.cmd_end_render_pass(cmd);
                    graphics_context.device.end_command_buffer(cmd).unwrap();
                }

                // 6. Submit & Present
                let wait_semaphores = [graphics_context.image_available_semaphore.unwrap()];
                let signal_semaphores = [graphics_context.render_finished_semaphore.unwrap()];
                let command_buffers = [cmd];
                let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

                let submit_info = vk::SubmitInfo::builder()
                    .wait_semaphores(&wait_semaphores)
                    .wait_dst_stage_mask(&wait_stages)
                    .command_buffers(&command_buffers)
                    .signal_semaphores(&signal_semaphores);
                
                let submit_infos = [submit_info.build()];
                unsafe {
                    let _lock = graphics_context.queue_mutex.lock().unwrap();
                    graphics_context.device.queue_submit(graphics_context.queue, &submit_infos, graphics_context.fence).unwrap();
                }

                let present_wait_semaphores = [graphics_context.render_finished_semaphore.unwrap()];
                let swapchains = [swapchain];
                let image_indices = [image_index];

                let present_info = vk::PresentInfoKHR::builder()
                    .wait_semaphores(&present_wait_semaphores)
                    .swapchains(&swapchains)
                    .image_indices(&image_indices);
                
                unsafe {
                    graphics_context.swapchain_loader.as_ref().unwrap().queue_present(graphics_context.queue, &present_info).unwrap();
                }

                prev_view_proj = view_proj;
            }
            _ => {}
        }
    }).unwrap();
}
