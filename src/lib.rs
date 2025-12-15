#[cfg(target_os = "android")]
use android_activity::{AndroidApp, MainEvent, PollEvent};
use log::{info, error, debug};
use std::sync::{Arc, RwLock};
use std::collections::HashMap; // Added HashMap
#[cfg(target_os = "android")]
use openxr as oxr;
#[cfg(target_os = "android")]
use ash::vk;

#[cfg(target_os = "android")]
use graphics::InstanceData; // Added InstanceData

#[cfg(target_os = "android")]
mod graphics;
#[cfg(target_os = "android")]
mod physics;
pub mod world;
#[cfg(target_os = "android")]
mod xr;

#[cfg(target_os = "android")]
use graphics::GraphicsContext;
#[cfg(target_os = "android")]
use xr::XrContext;
#[cfg(target_os = "android")]
use physics::PhysicsWorld;
#[cfg(target_os = "android")]
use rapier3d::prelude::{vector, point, Ray, QueryFilter, nalgebra};
use world::GameWorld;
use world::DecodedImage; // Import DecodedImage
#[cfg(target_os = "android")]
use graphics::Texture;
#[cfg(target_os = "android")]
use image::io::Reader as ImageReader;

// Import commonly used OpenXR types
#[cfg(target_os = "android")]
use oxr::{
    EventDataBuffer, SessionState, ViewConfigurationType, 
    ReferenceSpaceType, Posef, CompositionLayerProjectionView,
    SwapchainSubImage, Rect2Di, Offset2Di, Extent2Di,
    CompositionLayerProjection, EnvironmentBlendMode, Duration
};

#[cfg(target_os = "android")]
#[derive(Clone, Debug)]
pub struct GpuMesh {
    pub vertex_buffer: vk::Buffer,
    pub vertex_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    pub index_memory: vk::DeviceMemory,
    pub index_count: u32,
    pub material_descriptor_set: vk::DescriptorSet,
    pub custom_textures: Vec<Texture>,
}


#[derive(Debug)]
pub enum AndroidEvent {
    Resume,
    Pause,
    Destroy,
    InitWindow,
    WindowResized,
}

#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(app: AndroidApp) {
    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Info),
    );

    // Set panic hook to log panics
    std::panic::set_hook(Box::new(|panic_info| {
        error!("PANIC: {:?}", panic_info);
    }));

    info!("STFSC Engine Starting (Main Thread)...");

    let (event_tx, event_rx) = std::sync::mpsc::channel::<AndroidEvent>();
    let render_app = app.clone();

    // Spawn Render Thread
    std::thread::spawn(move || {
        render_loop(render_app, event_rx);
    });

    // Main Event Loop (Android Lifecycle)
    let mut running = true;
    loop {
        if !running { break; }
        app.poll_events(Some(std::time::Duration::from_millis(0)), |event| {
            match event {
                PollEvent::Main(MainEvent::Destroy) => {
                    info!("MainEvent::Destroy received on Main Thread");
                    let _ = event_tx.send(AndroidEvent::Destroy);
                    running = false;
                }
                PollEvent::Main(MainEvent::InitWindow { .. }) => {
                     let _ = event_tx.send(AndroidEvent::InitWindow);
                }
                PollEvent::Main(MainEvent::Resume { .. }) => {
                     let _ = event_tx.send(AndroidEvent::Resume);
                }
                PollEvent::Main(MainEvent::Pause { .. }) => {
                     let _ = event_tx.send(AndroidEvent::Pause);
                }
                PollEvent::Main(MainEvent::WindowResized { .. }) => {
                     let _ = event_tx.send(AndroidEvent::WindowResized);
                }
                _ => {}
            }
        });
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

#[cfg(target_os = "android")]
fn render_loop(app: AndroidApp, event_rx: std::sync::mpsc::Receiver<AndroidEvent>) {
    info!("Render Thread Started");

    // Wait for InitWindow before initializing Graphics/XR
    info!("Waiting for InitWindow...");
    // State flags that might change before InitWindow
    let mut activity_resumed = false;
    let mut window_ready = false;

    while !window_ready {
        match event_rx.recv() {
            Ok(AndroidEvent::InitWindow) => {
                 info!("InitWindow received, initializing engine...");
                 window_ready = true;
            }
            Ok(AndroidEvent::Resume) => {
                 info!("Resume received (during wait)");
                 activity_resumed = true;
            }
            Ok(AndroidEvent::Pause) => {
                 info!("Pause received (during wait)");
                 activity_resumed = false;
            }
            Ok(AndroidEvent::Destroy) => {
                 info!("Destroy received before InitWindow, existing.");
                 return;
            }
             _ => {} // Ignore others
        }
    }

    let mut quit = false;
    
    let (xr_instance, xr_system) = match XrContext::new() {
        Ok(ret) => ret,
        Err(e) => {
            error!("Failed to create OpenXR instance: {:?}", e);
            return;
        }
    };

    let graphics_context = match GraphicsContext::new(&xr_instance, xr_system) {
        Ok(ctx) => ctx,
        Err(e) => {
            error!("Failed to create Graphics Context: {:?}", e);
            return;
        }
    };
    
    let skybox_renderer = match graphics_context.create_skybox_pipeline(graphics_context.render_pass) {
        Ok(r) => r,
        Err(e) => {
            error!("Failed to create skybox renderer: {:?}", e);
            return;
        }
    };
    
    // Create Default Textures
    let (albedo_tex, normal_tex, mr_tex) = match graphics_context.create_default_pbr_textures() {
        Ok(texs) => texs,
        Err(e) => {
            error!("Failed to create default textures: {:?}", e);
            return;
        }
    };
    
    // Create Global Instance Buffer (10k instances)
    let max_instances = 10000;
    let (instance_buffer, instance_memory) = match graphics_context.create_buffer(
        (max_instances * std::mem::size_of::<InstanceData>()) as u64,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
    ) {
        Ok(ret) => ret,
        Err(e) => {
             error!("Failed to create instance buffer: {:?}", e);
             return;
        }
    };

    let instance_ptr = unsafe {
        match graphics_context.device.map_memory(instance_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) {
             Ok(ptr) => ptr as *mut InstanceData,
             Err(e) => {
                 error!("Failed to map instance buffer: {:?}", e);
                 return;
             }
        }
    };
    let global_descriptor_set = match graphics_context.create_global_descriptor_set(
        graphics_context.shadow_depth_view,
        graphics_context.shadow_sampler,
        instance_buffer
        ) {
        Ok(set) => set,
        Err(e) => {
            error!("Failed to create global descriptor set: {:?}", e);
            return;
        }
    };

    let material_descriptor_set = match graphics_context.create_material_descriptor_set(
        &albedo_tex, 
        &normal_tex, 
        &mr_tex
        ) {
        Ok(set) => set,
        Err(e) => {
            error!("Failed to create material descriptor set: {:?}", e);
            return;
        }
    };

    // Initialize Mesh Library
    let mut mesh_library: Vec<GpuMesh> = Vec::new();
    
    // Create primitives (0=Cube, 1=Sphere, 2=Cylinder, 3=Plane, 4=Capsule, 5=Cone)
    // Create primitives (0=Cube, 1=Sphere, 2=Cylinder, 3=Plane, 4=Capsule, 5=Cone)
    for i in 0..6 {
        let mesh = world::create_primitive(i);
        
        let (vbo, vbo_mem) = graphics_context.create_buffer(
            (mesh.vertices.len() * std::mem::size_of::<world::Vertex>()) as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        ).unwrap();
        
        unsafe {
            let ptr = graphics_context.device.map_memory(vbo_mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap();
            std::ptr::copy_nonoverlapping(mesh.vertices.as_ptr(), ptr as *mut world::Vertex, mesh.vertices.len());
            graphics_context.device.unmap_memory(vbo_mem);
        }

        let (ibo, ibo_mem) = graphics_context.create_buffer(
            (mesh.indices.len() * std::mem::size_of::<u32>()) as u64,
            vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        ).unwrap();

        unsafe {
            let ptr = graphics_context.device.map_memory(ibo_mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).unwrap();
            std::ptr::copy_nonoverlapping(mesh.indices.as_ptr(), ptr as *mut u32, mesh.indices.len());
            graphics_context.device.unmap_memory(ibo_mem);
        }

        let index_count = mesh.indices.len() as u32;

        mesh_library.push(GpuMesh {
            vertex_buffer: vbo,
            vertex_memory: vbo_mem,
            index_buffer: ibo,
            index_memory: ibo_mem,
            index_count,
            material_descriptor_set: material_descriptor_set, // Share same material for now
            custom_textures: Vec::new(),
        });
        info!("Initialized Primitive Mesh {}", i);
    }


    // Skybox Mesh (Hardcoded Cube)
    let skybox_vertices = [
        world::Vertex { position: [-1.0, -1.0,  1.0], normal: [0.0, 0.0, 0.0], uv: [0.0, 0.0], color: [0.0, 0.0, 0.0], tangent: [0.0, 0.0, 0.0, 0.0] },
        world::Vertex { position: [ 1.0, -1.0,  1.0], normal: [0.0, 0.0, 0.0], uv: [0.0, 0.0], color: [0.0, 0.0, 0.0], tangent: [0.0, 0.0, 0.0, 0.0] },
        world::Vertex { position: [ 1.0,  1.0,  1.0], normal: [0.0, 0.0, 0.0], uv: [0.0, 0.0], color: [0.0, 0.0, 0.0], tangent: [0.0, 0.0, 0.0, 0.0] },
        world::Vertex { position: [-1.0,  1.0,  1.0], normal: [0.0, 0.0, 0.0], uv: [0.0, 0.0], color: [0.0, 0.0, 0.0], tangent: [0.0, 0.0, 0.0, 0.0] },
        world::Vertex { position: [-1.0, -1.0, -1.0], normal: [0.0, 0.0, 0.0], uv: [0.0, 0.0], color: [0.0, 0.0, 0.0], tangent: [0.0, 0.0, 0.0, 0.0] },
        world::Vertex { position: [ 1.0, -1.0, -1.0], normal: [0.0, 0.0, 0.0], uv: [0.0, 0.0], color: [0.0, 0.0, 0.0], tangent: [0.0, 0.0, 0.0, 0.0] },
        world::Vertex { position: [ 1.0,  1.0, -1.0], normal: [0.0, 0.0, 0.0], uv: [0.0, 0.0], color: [0.0, 0.0, 0.0], tangent: [0.0, 0.0, 0.0, 0.0] },
        world::Vertex { position: [-1.0,  1.0, -1.0], normal: [0.0, 0.0, 0.0], uv: [0.0, 0.0], color: [0.0, 0.0, 0.0], tangent: [0.0, 0.0, 0.0, 0.0] },
    ];
    let skybox_indices = [
        0u32, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        4, 5, 1, 1, 0, 4,
        7, 6, 2, 2, 3, 7,
        4, 7, 3, 3, 0, 4,
        5, 6, 2, 2, 1, 5,
    ];

    let (skybox_vertex_buffer, skybox_vertex_memory) = graphics_context.create_buffer(
        (skybox_vertices.len() * std::mem::size_of::<world::Vertex>()) as u64,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
    ).unwrap();
    
    unsafe {
        let ptr = graphics_context.device.map_memory(skybox_vertex_memory, 0, (skybox_vertices.len() * std::mem::size_of::<world::Vertex>()) as u64, vk::MemoryMapFlags::empty()).unwrap();
        std::ptr::copy_nonoverlapping(skybox_vertices.as_ptr(), ptr as *mut world::Vertex, skybox_vertices.len());
        graphics_context.device.unmap_memory(skybox_vertex_memory);
    }

    let (skybox_index_buffer, skybox_index_memory) = graphics_context.create_buffer(
        (skybox_indices.len() * std::mem::size_of::<u32>()) as u64,
        vk::BufferUsageFlags::INDEX_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
    ).unwrap();

    unsafe {
        let ptr = graphics_context.device.map_memory(skybox_index_memory, 0, (skybox_indices.len() * std::mem::size_of::<u32>()) as u64, vk::MemoryMapFlags::empty()).unwrap();
        std::ptr::copy_nonoverlapping(skybox_indices.as_ptr(), ptr as *mut u32, skybox_indices.len());
        graphics_context.device.unmap_memory(skybox_index_memory);
    }

    let mut xr_context = match XrContext::create_session(&xr_instance, xr_system, &graphics_context) {
        Ok(ctx) => ctx,
        Err(e) => {
            error!("Failed to create OpenXR Session: {:?}", e);
            return;
        }
    };

    // Initialize Physics and World
    // Initialize Physics and World
    let physics_world = PhysicsWorld::new();
    let game_world = GameWorld::new();

    let cmd_tx = game_world.command_sender.clone();
    
    // Shared State
    struct GameState {
        physics: PhysicsWorld,
        world: GameWorld,
        player_position: glam::Vec3,
    }
    
    let game_state = Arc::new(RwLock::new(GameState {
        physics: physics_world,
        world: game_world,
        player_position: glam::Vec3::new(0.0, 1.7, 0.0), // Default standing height
    }));

    // =========================================================================
    // DEMO SCENE - DISABLED
    // The editor now controls scene content. Deploy "Test Engine Scene" from 
    // editor via File > Test Engine Scene, then Scene > Deploy All to Quest
    // =========================================================================
    // To re-enable: uncomment the block below
    /*
    // Spawn a Test Physics Block
    {
        if let Ok(mut state) = game_state.write() {
             // 1. Create Rigid Body
             let handle = state.physics.add_box_rigid_body([0.0, 10.0, -8.0], [0.5, 0.5, 0.5], true);
             state.physics.add_box_rigid_body([0.0, -2.0, 0.0], [100.0, 1.0, 100.0], false); // Ground Plane

             // 2. Create Entity
             state.world.ecs.spawn((
                 world::Transform {
                     position: glam::Vec3::new(0.0, 10.0, -8.0),
                     rotation: glam::Quat::IDENTITY,
                     scale: glam::Vec3::ONE,
                 },
                 world::MeshHandle(0),
                 world::RigidBodyHandle(handle),
             ));
             info!("Spawned Physics Test Cube + Ground");
             
             // 3. Spawn Test Vehicle
             let veh_handle = state.physics.add_box_rigid_body([5.0, 2.0, -8.0], [1.0, 0.5, 2.0], true);
             state.world.ecs.spawn((
                 world::Transform {
                     position: glam::Vec3::new(5.0, 2.0, -8.0),
                     rotation: glam::Quat::IDENTITY,
                     scale: glam::Vec3::ONE,
                 },
                 world::MeshHandle(0), // Re-use cube mesh for now
                 world::RigidBodyHandle(veh_handle),
                 world::Vehicle { speed: 10.0, max_speed: 30.0, steering: 0.0, accelerating: true },
             ));

             info!("Spawned Test Vehicle");
             
             // 4. Spawn Crowd (50 Agents)
             let mut seed: u32 = 999;
             let mut rand = || {
                 seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                 (seed as f32) / (u32::MAX as f32)
             };
             for i in 0..50 {
                 let x = (rand() - 0.5) * 40.0;
                 let z = (rand() - 0.5) * 40.0;
                 
                 // Some agents are fleeing (10% of them) for demo
                 let (agent_state, max_speed) = if i < 5 {
                     (world::AgentState::Fleeing, 8.0)  // Fast runners
                 } else if i < 15 {
                     (world::AgentState::Running, 5.0)  // Jogging
                 } else {
                     (world::AgentState::Walking, 2.0)  // Normal walking
                 };
                 
                 state.world.ecs.spawn((
                     world::Transform {
                         position: glam::Vec3::new(x, 1.0, z),
                         rotation: glam::Quat::IDENTITY,
                         scale: glam::Vec3::new(0.3, 1.7, 0.3), // Human-ish scale
                     },
                     world::MeshHandle(0),
                     world::CrowdAgent {
                         velocity: glam::Vec3::ZERO,
                         target: glam::Vec3::new((rand() - 0.5) * 40.0, 1.0, (rand() - 0.5) * 40.0),
                         state: agent_state,
                         max_speed,
                     }
                 ));
             }
             info!("Spawned 50 Crowd Agents (5 Fleeing, 10 Running, 35 Walking)");
        }
    }
    */
    info!("Quest started with empty scene - deploy from editor");

    // Game Loop Thread
    let game_state_thread = game_state.clone();
    let mut game_quit = false; // TODO: Handle quit signal properly across threads
    
    std::thread::spawn(move || {
        info!("Game Loop Thread Started");
        let mut frame_count: u64 = 0;
        while !game_quit {
            let start = std::time::Instant::now();
            
            // ===== PHASE 1: Physics Step (short lock) =====
            {
                if let Ok(mut state) = game_state_thread.try_write() {
                    state.physics.step();
                }
            }
            // Yield to render thread
            std::thread::yield_now();
            
            // ===== PHASE 2: World Streaming (short lock) =====
            {
                if let Ok(mut state) = game_state_thread.try_write() {
                    let player_pos = state.player_position;
                    state.world.update_streaming(player_pos);
                }
            }
            // Yield to render thread
            std::thread::yield_now();
            
            // ===== PHASE 3: Physics -> ECS Sync (collect outside lock, apply with lock) =====
            let physics_updates: Vec<_>;
            {
                if let Ok(state) = game_state_thread.try_read() {
                    use rayon::prelude::*;
                    let physics_bodies: Vec<_> = state.world.ecs
                        .query::<(&world::Transform, &world::RigidBodyHandle)>()
                        .iter()
                        .map(|(id, (_, handle))| (id, handle.0))
                        .collect();
                    
                    physics_updates = physics_bodies.par_iter()
                        .filter_map(|(id, handle)| {
                            state.physics.rigid_body_set.get(*handle).map(|body| {
                                let translation = body.translation();
                                let rotation = body.rotation();
                                (*id, 
                                 glam::Vec3::new(translation.x, translation.y, translation.z), 
                                 glam::Quat::from_xyzw(rotation.i, rotation.j, rotation.k, rotation.w))
                            })
                        })
                        .collect();
                } else {
                    physics_updates = Vec::new();
                }
            }
            // Apply physics updates with short write lock
            if !physics_updates.is_empty() {
                if let Ok(mut state) = game_state_thread.try_write() {
                    for (id, pos, rot) in physics_updates {
                        if let Ok(mut transform) = state.world.ecs.get::<&mut world::Transform>(id) {
                            transform.position = pos;
                            transform.rotation = rot;
                        }
                    }
                }
            }
            std::thread::yield_now();
            
            // ===== PHASE 4: Vehicle Logic (short lock per operation) =====
            {
                // Collect vehicle data
                let vehicle_data: Vec<_>;
                if let Ok(state) = game_state_thread.try_read() {
                    vehicle_data = state.world.ecs.query::<(&world::Vehicle, &world::RigidBodyHandle)>()
                        .iter()
                        .map(|(_, (v, h))| (v.speed, h.0))
                        .collect();
                } else {
                    vehicle_data = Vec::new();
                }
                
                // Process each vehicle with short lock
                for (speed, body_handle) in vehicle_data {
                    if let Ok(mut state) = game_state_thread.try_write() {
                        let ray_result = {
                            if let Some(body) = state.physics.rigid_body_set.get(body_handle) {
                                let position = body.translation();
                                let ray_origin = point![position.x, position.y, position.z];
                                let ray_dir = vector![0.0, -1.0, 0.0];
                                let max_dist = 1.5;
                                
                                state.physics.query_pipeline.cast_ray(
                                    &state.physics.rigid_body_set,
                                    &state.physics.collider_set,
                                    &Ray::new(ray_origin, ray_dir),
                                    max_dist,
                                    true,
                                    QueryFilter::default().exclude_rigid_body(body_handle)
                                ).map(|(_, toi)| (toi, max_dist))
                            } else {
                                None
                            }
                        };
                        
                        if let Some((toi, max_dist)) = ray_result {
                            if let Some(body) = state.physics.rigid_body_set.get_mut(body_handle) {
                                let stiffness = 200.0;
                                let damping = 10.0;
                                let compression = 1.0 - (toi / max_dist);
                                let up_force = vector![0.0, stiffness * compression, 0.0];
                                
                                body.add_force(up_force, true);
                                
                                let vel = *body.linvel();
                                body.add_force(-vel * damping, true);
                                
                                let rot = *body.rotation();
                                let forward_dir = rot.transform_vector(&vector![0.0, 0.0, -1.0]);
                                body.add_force(forward_dir * speed, true);
                            }
                        }
                    }
                }
            }
            std::thread::yield_now();
            
            // ===== PHASE 5: Crowd Logic (parallel compute, short lock for apply) =====
            {
                use rayon::prelude::*;
                let dt = 0.027 * 5.0;
                
                // Collect agents outside lock
                let agents_to_update: Vec<_>;
                if let Ok(state) = game_state_thread.try_read() {
                    agents_to_update = state.world.ecs
                        .query::<(&world::CrowdAgent, &world::Transform)>()
                        .iter()
                        .enumerate()
                        .filter(|(idx, _)| (*idx % 5) == ((frame_count % 5) as usize))
                        .map(|(_, (id, (agent, transform)))| {
                            (id, agent.clone(), transform.clone())
                        })
                        .collect();
                } else {
                    agents_to_update = Vec::new();
                }
                
                // Process in parallel WITHOUT lock
                let agent_updates: Vec<_> = agents_to_update.par_iter()
                    .map(|(id, agent, transform)| {
                        let pos = transform.position;
                        let target = agent.target;
                        let max_speed = agent.max_speed;
                        
                        let to_target = target - pos;
                        let dist = to_target.length();
                        
                        if dist < 1.0 {
                            let new_target = -agent.target;
                            (*id, agent.velocity, new_target, transform.position, transform.rotation)
                        } else {
                            let desired = to_target.normalize() * max_speed;
                            let steer_force = if agent.state == world::AgentState::Fleeing { 10.0 } else { 5.0 };
                            let steering = (desired - agent.velocity) * steer_force;
                            
                            let mut new_velocity = agent.velocity + steering * dt;
                            if new_velocity.length() > max_speed {
                                new_velocity = new_velocity.normalize() * max_speed;
                            }
                            
                            let new_position = pos + new_velocity * dt;
                            
                            let new_rotation = if new_velocity.length_squared() > 0.1 {
                                let angle = new_velocity.x.atan2(new_velocity.z);
                                glam::Quat::from_rotation_y(angle)
                            } else {
                                transform.rotation
                            };
                            
                            (*id, new_velocity, agent.target, new_position, new_rotation)
                        }
                    })
                    .collect();
                
                // Apply with short write lock
                if !agent_updates.is_empty() {
                    if let Ok(mut state) = game_state_thread.try_write() {
                        for (id, new_vel, new_target, new_pos, new_rot) in agent_updates {
                            if let Ok((mut agent, mut transform)) = state.world.ecs.query_one_mut::<(&mut world::CrowdAgent, &mut world::Transform)>(id) {
                                agent.velocity = new_vel;
                                agent.target = new_target;
                                transform.position = new_pos;
                                transform.rotation = new_rot;
                            }
                        }
                    }
                }
            }
            
            frame_count += 1;
            let elapsed = start.elapsed();
            // Target 36Hz (approx 27.77ms) for AppSW
            // We run physics/logic at 36fps and let AppSW synthesize the rest to 72Hz
            let target_frame_time = std::time::Duration::from_micros(27777); 
            if elapsed < target_frame_time {
                std::thread::sleep(target_frame_time - elapsed);
            }
        }
    });


    std::thread::spawn(move || {
        let rt = match tokio::runtime::Runtime::new() {
            Ok(rt) => rt,
            Err(e) => {
                error!("Failed to create Tokio runtime: {:?}", e);
                return;
            }
        };
        rt.block_on(async move {
            let listener = match tokio::net::TcpListener::bind("0.0.0.0:8080").await {
                Ok(l) => l,
                Err(e) => {
                    error!("Failed to bind TCP listener: {:?}", e);
                    return;
                }
            };
            info!("Listening on 0.0.0.0:8080");
            
            loop {
                if let Ok((mut socket, addr)) = listener.accept().await {
                    info!("Accepted connection from: {:?}", addr);
                    let tx = cmd_tx.clone();
                    tokio::spawn(async move {
                        use tokio::io::AsyncReadExt;
                        loop {
                            let mut len_buf = [0u8; 4];
                            if socket.read_exact(&mut len_buf).await.is_err() {
                                info!("Socket closed or read error (len)");
                                break;
                            }
                            let len = u32::from_le_bytes(len_buf) as usize;
                            let mut data = vec![0u8; len];
                            if socket.read_exact(&mut data).await.is_err() {
                                info!("Socket closed or read error (data)");
                                break;
                            }
                            
                            if let Ok(mut update) = bincode::deserialize::<world::SceneUpdate>(&data) {
                                info!("Received update (pre-decode): {:?}", update);
                                
                                // Offload texture decoding to blocking thread
                                let update = tokio::task::spawn_blocking(move || {
                                    if let world::SceneUpdate::SpawnMesh { mesh, .. } = &mut update {
                                        // Helper closure for decoding
                                        let decode = |data: &Option<Vec<u8>>| -> Option<Arc<DecodedImage>> {
                                            if let Some(bytes) = data {
                                                if let Ok(img) = image::load_from_memory(bytes) {
                                                    let rgba = img.to_rgba8();
                                                    let (width, height) = rgba.dimensions();
                                                    return Some(Arc::new(DecodedImage {
                                                        width,
                                                        height,
                                                        data: rgba.into_raw(),
                                                    }));
                                                }
                                            }
                                            None
                                        };

                                        mesh.decoded_albedo = decode(&mesh.albedo);
                                        mesh.decoded_normal = decode(&mesh.normal);
                                        mesh.decoded_mr = decode(&mesh.metallic_roughness);
                                        
                                        if mesh.decoded_albedo.is_some() {
                                            info!("Decoded albedo texture in background");
                                        }
                                    }
                                    update
                                }).await.unwrap_or_else(|e| {
                                    error!("Join error in spawn_blocking: {:?}", e);
                                    // wrapper to avoid panic, though unwrap is mostly safe here if we don't panic inside
                                    world::SceneUpdate::Spawn { id: 0, primitive: 0, position: [0.0; 3], rotation: [0.0; 4], color: [1.0, 0.0, 1.0] } // Error sentinel
                                });

                                let _ = tx.send(update).await;
                            } else {
                                error!("Failed to deserialize update");
                            }
                        }
                    });
                }
            }
        });
    });

    info!("Engine Initialized Successfully");

    let mut session_running = false;
    // activity_resumed is already defined above

    // State for AppSW (Motion Vectors)
    // Store previous frame's View Projection Matrix per eye
    let mut prev_view_projs = [glam::Mat4::IDENTITY; 2];
    
    // Previous transforms for entities (for Motion Vectors)
    let mut prev_transforms: HashMap<u64, glam::Mat4> = HashMap::new();
    
    // Cached draw data for lock-failure recovery (prevents 30-second flicker from TREX stalls)
    let mut cached_player_position = glam::Vec3::ZERO;
    let mut cached_batch_map: HashMap<usize, Vec<InstanceData>> = HashMap::new();
    let mut cached_custom_draws: Vec<(GpuMesh, InstanceData)> = Vec::new();

    while !quit {
        let timeout = if session_running {
            std::time::Duration::from_millis(0)
        } else {
            std::time::Duration::from_millis(100)
        };

        // Process Android Events from Main Thread
        while let Ok(event) = event_rx.try_recv() {
            match event {
                AndroidEvent::Destroy => {
                    info!("AndroidEvent::Destroy received in Render Thread, quitting...");
                    quit = true;
                }
                AndroidEvent::InitWindow => {
                    info!("Window Initialized (Render Thread)");
                }
                AndroidEvent::Resume => {
                    info!("AndroidEvent::Resume received");
                    activity_resumed = true;
                    info!("Activity state: resumed");
                }
                AndroidEvent::Pause => {
                     info!("AndroidEvent::Pause received");
                     activity_resumed = false;
                     info!("Activity state: paused");
                }
                AndroidEvent::WindowResized => {
                    info!("AndroidEvent::WindowResized received");
                }
            }
        }
        
        // Heartbeat log
        static mut FRAME_COUNT: u64 = 0;
        unsafe {
            FRAME_COUNT += 1;
            if FRAME_COUNT % 60 == 0 {
                info!("Heartbeat: Frame {}", FRAME_COUNT);
            }
        }

        // If not running or not resumed, throttle the loop
        if !session_running || !activity_resumed {
             info!("Frame loop paused - session_running: {}, activity_resumed: {}", session_running, activity_resumed);
             std::thread::sleep(std::time::Duration::from_millis(100));
        }

        let mut event_storage = EventDataBuffer::new();
        while let Some(event) = xr_context.instance.poll_event(&mut event_storage).unwrap_or(None) {
            match event {
                oxr::Event::SessionStateChanged(e) => {
                    info!("Session state changed to {:?}", e.state());
                    match e.state() {
                        SessionState::READY => {
                            if !session_running {
                                if let Err(e) = xr_context.session.begin(ViewConfigurationType::PRIMARY_STEREO) {
                                    error!("Failed to begin session: {:?}", e);
                                    quit = true;
                                } else {
                                    session_running = true;
                                }
                            }
                        }
                        SessionState::STOPPING => {
                            if session_running {
                                if let Err(e) = xr_context.session.end() {
                                    error!("Failed to end session: {:?}", e);
                                }
                                session_running = false;
                            }
                        }
                        SessionState::EXITING | SessionState::LOSS_PENDING => {
                            quit = true;
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        if session_running && activity_resumed {
            // Update Physics and World
            // Physics and World updates are now in a separate thread
            // physics_world.step();
            // game_world.update_streaming();

            // Mesh Upload Logic (non-blocking to prevent ANR)
            // Limit uploads per frame to avoid stalling the render thread
            const MAX_MESH_UPLOADS_PER_FRAME: usize = 1; // Strict limit to prevent ANR
            if let Ok(mut state) = game_state.try_write() {
                let mut meshes_to_upload = Vec::new();
                
                for (id, (mesh, _transform)) in state.world.ecs.query::<(&world::Mesh, &world::Transform)>().iter() {
                    if meshes_to_upload.len() >= MAX_MESH_UPLOADS_PER_FRAME {
                        break; // Throttle: defer remaining uploads to next frame
                    }
                    if state.world.ecs.get::<&GpuMesh>(id).is_ok() {
                        continue;
                    }
                    meshes_to_upload.push((id, mesh.clone()));
                }

                for (id, mesh) in meshes_to_upload {
                    let (vbo, vbo_mem) = unsafe {
                        graphics_context.create_buffer(
                            (mesh.vertices.len() * std::mem::size_of::<world::Vertex>()) as vk::DeviceSize,
                            vk::BufferUsageFlags::VERTEX_BUFFER,
                            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                        ).expect("Failed to create vertex buffer")
                    };

                    unsafe {
                        let data_ptr = graphics_context.device.map_memory(
                            vbo_mem,
                            0,
                            (mesh.vertices.len() * std::mem::size_of::<world::Vertex>()) as vk::DeviceSize,
                            vk::MemoryMapFlags::empty(),
                        ).expect("Failed to map vertex buffer");
                        std::ptr::copy_nonoverlapping(
                            mesh.vertices.as_ptr(),
                            data_ptr as *mut world::Vertex,
                            mesh.vertices.len(),
                        );
                        graphics_context.device.unmap_memory(vbo_mem);
                    }

                    let (ibo, ibo_mem) = unsafe {
                        graphics_context.create_buffer(
                            (mesh.indices.len() * std::mem::size_of::<u32>()) as vk::DeviceSize,
                            vk::BufferUsageFlags::INDEX_BUFFER,
                            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                        ).expect("Failed to create index buffer")
                    };

                    unsafe {
                        let data_ptr = graphics_context.device.map_memory(
                            ibo_mem,
                            0,
                            (mesh.indices.len() * std::mem::size_of::<u32>()) as vk::DeviceSize,
                            vk::MemoryMapFlags::empty(),
                        ).expect("Failed to map index buffer");
                        std::ptr::copy_nonoverlapping(
                            mesh.indices.as_ptr(),
                            data_ptr as *mut u32,
                            mesh.indices.len(),
                        );
                        graphics_context.device.unmap_memory(ibo_mem);
                    }

                        let mut custom_textures = Vec::new();
                        
                        // Helper to load texture or use default
                        // We need to handle potential errors gracefully or just log and use default
                        
                        // Retry strategy: Create all 3 optionals first.
                        let mut tex_albedo_opt = None;
                        let mut tex_normal_opt = None;
                        let mut tex_mr_opt = None;
                        
                        // Only use pre-decoded textures (decoded on network thread to avoid ANR)
                        if let Some(decoded) = &mesh.decoded_albedo {
                             if let Ok(tex) = graphics_context.create_texture_from_raw(decoded.width, decoded.height, &decoded.data) {
                                 tex_albedo_opt = Some(tex);
                             }
                        }
                        
                        if let Some(decoded) = &mesh.decoded_normal {
                             if let Ok(tex) = graphics_context.create_texture_from_raw(decoded.width, decoded.height, &decoded.data) {
                                 tex_normal_opt = Some(tex);
                             }
                        }
                        
                        if let Some(decoded) = &mesh.decoded_mr {
                             if let Ok(tex) = graphics_context.create_texture_from_raw(decoded.width, decoded.height, &decoded.data) {
                                 tex_mr_opt = Some(tex);
                             }
                        }
                        
                        let final_albedo = tex_albedo_opt.as_ref().unwrap_or(&albedo_tex);
                        let final_normal = tex_normal_opt.as_ref().unwrap_or(&normal_tex);
                        let final_mr = tex_mr_opt.as_ref().unwrap_or(&mr_tex);
                        
                        // Decide if we need a new descriptor set
                        let final_descriptor_set = if tex_albedo_opt.is_some() || tex_normal_opt.is_some() || tex_mr_opt.is_some() {
                             match graphics_context.create_material_descriptor_set(final_albedo, final_normal, final_mr) {
                                 Ok(set) => set,
                                 Err(e) => {
                                     error!("Failed to create custom descriptor set: {:?}", e);
                                     material_descriptor_set // Fallback
                                 }
                             }
                        } else {
                            material_descriptor_set
                        };
                        
                        // Move ownership to vector for GpuMesh
                        if let Some(t) = tex_albedo_opt { custom_textures.push(t); }
                        if let Some(t) = tex_normal_opt { custom_textures.push(t); }
                        if let Some(t) = tex_mr_opt { custom_textures.push(t); }

                        let gpu_mesh = GpuMesh {
                        vertex_buffer: vbo,
                        vertex_memory: vbo_mem,
                        index_buffer: ibo,
                        index_memory: ibo_mem,
                        index_count: mesh.indices.len() as u32,
                        material_descriptor_set: final_descriptor_set,
                        custom_textures,
                    };

                    if let Err(e) = state.world.ecs.insert_one(id, gpu_mesh) {
                        error!("Failed to insert GpuMesh for entity {:?}: {:?}", id, e);
                    } else {
                        info!("Uploaded mesh for entity {:?}", id);
                    }
                }
            }

            // Yield before blocking OpenXR call to reduce lock contention
            std::thread::yield_now();
            
            // info!("Calling xrWaitFrame");
            match xr_context.frame_waiter.wait() {
                Ok(frame_state) => {
                    // info!("xrWaitFrame returned, should_render: {}", frame_state.should_render);

                    // info!("Calling frame_stream.begin");
                    if let Err(e) = xr_context.frame_stream.begin() {
                         error!("Failed to begin frame: {:?}", e);
                         continue;
                    }
                    // info!("frame_stream.begin succeeded");

                    let mut layers: Vec<&oxr::CompositionLayerBase<oxr::Vulkan>> = Vec::new();
                    let projection_views: Vec<oxr::CompositionLayerProjectionView<oxr::Vulkan>>;
                    let projection_layer_storage: Option<oxr::CompositionLayerProjection<oxr::Vulkan>>;

                    if frame_state.should_render {
                        let (view_flags, views) = xr_context.session.locate_views(
                            ViewConfigurationType::PRIMARY_STEREO,
                            frame_state.predicted_display_time,
                            &xr_context.stage_space,
                        ).unwrap();

                        // info!("Locate views succeeded, flags: {:?}", view_flags);
                        
                        // Sync player position to game state for streaming
                        if let Ok(mut state) = game_state.try_write() {
                             let p = views[0].pose.position;
                             state.player_position = glam::Vec3::new(p.x, p.y, p.z);
                        }

                        // info!("Calling swapchain.acquire_image");
                        let stream_idx = match xr_context.swapchain.acquire_image() {
                            Ok(idx) => idx,
                            Err(e) => {
                                error!("Failed to acquire image: {:?}", e);
                                9999 // Sentinel
                            }
                        };

                        if stream_idx != 9999 {
                            // info!("swapchain.acquire_image succeeded: {}", stream_idx);
                            
                            // Acquire Depth and Motion images (assume synced)
                            let _depth_idx = xr_context.depth_swapchain.acquire_image().unwrap_or(0);
                            let _motion_idx = xr_context.motion_swapchain.acquire_image().unwrap_or(0);

                            // info!("Calling swapchain.wait_image");
                            if let Err(e) = xr_context.swapchain.wait_image(Duration::INFINITE) {
                                 error!("Failed to wait image: {:?}", e);
                            }
                            if let Err(e) = xr_context.depth_swapchain.wait_image(Duration::INFINITE) {
                                 error!("Failed to wait depth image: {:?}", e);
                            }
                            if let Err(e) = xr_context.motion_swapchain.wait_image(Duration::INFINITE) {
                                 error!("Failed to wait motion image: {:?}", e);
                            }
                            
                            // info!("swapchain.wait_image succeeded");

                                // Record Command Buffer
                                unsafe {
                                    let _ = graphics_context.device.reset_command_buffer(graphics_context.command_buffer, vk::CommandBufferResetFlags::empty());
                                    let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                                    let _ = graphics_context.device.begin_command_buffer(graphics_context.command_buffer, &begin_info);

                                    // --- 1. Prepare Instance Data ---
                                    let mut batch_map: HashMap<usize, Vec<InstanceData>> = HashMap::new();
                                    let mut custom_draws: Vec<(GpuMesh, InstanceData)> = Vec::new();
                                    
                                    if let Ok(state) = game_state.try_read() {
                                        // Update cached player position for shadow calculations
                                        cached_player_position = state.player_position;
                                        
                                        // 1. Collect Regular Instance Meshes (MeshHandle)
                                        for (id, (handle, transform)) in state.world.ecs.query::<(&world::MeshHandle, &world::Transform)>().iter() {
                                            let model = glam::Mat4::from_scale_rotation_translation(transform.scale, transform.rotation, transform.position);
                                            let entity_id = id.id() as u64; 
                                            let prev_model = *prev_transforms.get(&entity_id).unwrap_or(&model);
                                            prev_transforms.insert(entity_id, model);
                                            
                                            let color = if let Ok(mat) = state.world.ecs.get::<&world::Material>(id) {
                                                mat.color
                                            } else {
                                                [1.0, 1.0, 1.0, 1.0]
                                            };
                                            
                                            batch_map.entry(handle.0).or_default().push(InstanceData { model, prev_model, color });
                                        }

                                        // 2. Collect LOD Groups
                                        // Calculate camera position for distance check (use left eye for approx)
                                        let cam_pos = views[0].pose.position;
                                        let cam_vec3 = glam::Vec3::new(cam_pos.x, cam_pos.y, cam_pos.z);

                                        for (id, (lod_group, transform)) in state.world.ecs.query::<(&world::LODGroup, &world::Transform)>().iter() {
                                            let dist = transform.position.distance(cam_vec3);
                                            
                                            // Find suitable LOD level
                                            let mut best_handle = None;
                                            for level in &lod_group.levels {
                                                if dist < level.distance {
                                                    best_handle = Some(level.mesh);
                                                    break;
                                                }
                                            }
                                            // If None (dist > max distance), we cull it (draw nothing)

                                            if let Some(handle) = best_handle {
                                                let model = glam::Mat4::from_scale_rotation_translation(transform.scale, transform.rotation, transform.position);
                                                let entity_id = id.id() as u64;
                                                let prev_model = *prev_transforms.get(&entity_id).unwrap_or(&model);
                                                prev_transforms.insert(entity_id, model);
                                                
                                                let color = if let Ok(mat) = state.world.ecs.get::<&world::Material>(id) {
                                                    mat.color
                                                } else {
                                                    [1.0, 1.0, 1.0, 1.0]
                                                };
                                                
                                                batch_map.entry(handle.0).or_default().push(InstanceData { model, prev_model, color });
                                            }
                                        }
                                        
                                        // 3. Collect Custom GpuMesh Entities (e.g. from Editor SpawnMesh)
                                        for (id, (gpu_mesh, transform)) in state.world.ecs.query::<(&GpuMesh, &world::Transform)>().iter() {
                                            // Skip if it has MeshHandle or LODGroup (already handled)
                                            if state.world.ecs.get::<&world::MeshHandle>(id).is_ok() { continue; }
                                            if state.world.ecs.get::<&world::LODGroup>(id).is_ok() { continue; }

                                            let model = glam::Mat4::from_scale_rotation_translation(transform.scale, transform.rotation, transform.position);
                                            let entity_id = id.id() as u64; 
                                            let prev_model = *prev_transforms.get(&entity_id).unwrap_or(&model);
                                            prev_transforms.insert(entity_id, model);
                                            
                                            let color = if let Ok(mat) = state.world.ecs.get::<&world::Material>(id) {
                                                mat.color
                                            } else {
                                                [1.0, 1.0, 1.0, 1.0]
                                            };
                                            
                                            custom_draws.push((gpu_mesh.clone(), InstanceData { model, prev_model, color }));
                                        }
                                        
                                        // Update cache for next potential lock failure
                                        cached_batch_map = batch_map.clone();
                                        cached_custom_draws = custom_draws.clone();
                                    } else {
                                        // Lock failed (likely TREX stall) - use cached data to prevent flicker
                                        batch_map = cached_batch_map.clone();
                                        custom_draws = cached_custom_draws.clone();
                                    }

                                    // Upload to Instance Buffer
                                    let mut instance_offset = 0;
                                    let mut draw_calls: Vec<(usize, u32, u32)> = Vec::new(); // (mesh_idx, first_instance, instance_count)
                                    let mut custom_draw_calls: Vec<(GpuMesh, u32, u32)> = Vec::new(); // (mesh, first_instance, instance_count)

                                    // Batch Maps
                                    for (mesh_idx, instances) in batch_map {
                                        if mesh_idx >= mesh_library.len() { continue; }
                                        let count = instances.len();
                                        if instance_offset + count > 10000 {
                                            error!("Instance buffer overflow!");
                                            break;
                                        }
                                        
                                        let dest = instance_ptr.add(instance_offset);
                                        std::ptr::copy_nonoverlapping(instances.as_ptr(), dest, count);
                                        
                                        draw_calls.push((mesh_idx, instance_offset as u32, count as u32));
                                        instance_offset += count;
                                    }
                                    
                                    // Custom Draws
                                    for (mesh, instance_data) in custom_draws {
                                         if instance_offset + 1 > 10000 {
                                            error!("Instance buffer overflow (custom)!");
                                            break;
                                        }
                                        let dest = instance_ptr.add(instance_offset);
                                        std::ptr::write(dest, instance_data);
                                        
                                        custom_draw_calls.push((mesh, instance_offset as u32, 1));
                                        instance_offset += 1;
                                    }
                                    
                                    // Flush instance buffer (if not HOST_COHERENT, but we used HOST_COHERENT)

                                    // Light Data - Shadow frustum follows player position (uses cached position if lock failed earlier)
                                    let shadow_center = cached_player_position;
                                    let light_offset = glam::Vec3::new(20.0, 50.0, 20.0);
                                    let light_pos = shadow_center + light_offset;
                                    let light_proj = glam::Mat4::orthographic_rh(-50.0, 50.0, -50.0, 50.0, 1.0, 150.0);
                                    let light_view = glam::Mat4::look_at_rh(light_pos, shadow_center, glam::Vec3::Y);
                                    let light_view_proj = light_proj * light_view;

                                    // --- 2. Shadow Pass ---
                                    {
                                        let clear_values = [vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } }];
                                        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                                            .render_pass(graphics_context.shadow_render_pass)
                                            .framebuffer(graphics_context.shadow_framebuffer)
                                            .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: graphics_context.shadow_extent })
                                            .clear_values(&clear_values);

                                        graphics_context.device.cmd_begin_render_pass(graphics_context.command_buffer, &render_pass_begin_info, vk::SubpassContents::INLINE);
                                        graphics_context.device.cmd_bind_pipeline(graphics_context.command_buffer, vk::PipelineBindPoint::GRAPHICS, graphics_context.shadow_pipeline);

                                        // Bind Global Set (Set 0) - Contains Instance Buffer (Binding 1)
                                        graphics_context.device.cmd_bind_descriptor_sets(
                                            graphics_context.command_buffer,
                                            vk::PipelineBindPoint::GRAPHICS,
                                            graphics_context.shadow_pipeline_layout,
                                            0,
                                            &[global_descriptor_set],
                                            &[]
                                        );
                                        
                                        // Push Constants (Light ViewProj)
                                        let push_bytes = bytemuck::bytes_of(&light_view_proj);
                                        graphics_context.device.cmd_push_constants(
                                            graphics_context.command_buffer, 
                                            graphics_context.shadow_pipeline_layout, 
                                            vk::ShaderStageFlags::VERTEX, 
                                            0, 
                                            push_bytes
                                        );

                                        // Draw Batches
                                        for (mesh_idx, first_instance, instance_count) in &draw_calls {
                                            let gpu_mesh = &mesh_library[*mesh_idx];
                                            graphics_context.device.cmd_bind_vertex_buffers(graphics_context.command_buffer, 0, &[gpu_mesh.vertex_buffer], &[0]);
                                            graphics_context.device.cmd_bind_index_buffer(graphics_context.command_buffer, gpu_mesh.index_buffer, 0, vk::IndexType::UINT32);
                                            graphics_context.device.cmd_draw_indexed(graphics_context.command_buffer, gpu_mesh.index_count, *instance_count, 0, 0, *first_instance);
                                        }
                                        
                                        // Draw Custom Meshes
                                        for (gpu_mesh, first_instance, instance_count) in &custom_draw_calls {
                                            graphics_context.device.cmd_bind_vertex_buffers(graphics_context.command_buffer, 0, &[gpu_mesh.vertex_buffer], &[0]);
                                            graphics_context.device.cmd_bind_index_buffer(graphics_context.command_buffer, gpu_mesh.index_buffer, 0, vk::IndexType::UINT32);
                                            graphics_context.device.cmd_draw_indexed(graphics_context.command_buffer, gpu_mesh.index_count, *instance_count, 0, 0, *first_instance);
                                        }

                                        graphics_context.device.cmd_end_render_pass(graphics_context.command_buffer);
                                    }

                                    // --- 3. Main Pass (Stereo) ---
                                    for eye_index in 0..2 {
                                        let clear_values = [
                                            vk::ClearValue { color: vk::ClearColorValue { float32: [0.1, 0.1, 0.15, 1.0] } },
                                            vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
                                            vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] } } // Motion Vectors
                                        ];
                                        
                                        let view = &views[eye_index];
                                        
                                        let player_start = if let Ok(state) = game_state.read() {
                                            state.world.player_start_transform.clone()
                                        } else {
                                            world::Transform { position: glam::Vec3::ZERO, rotation: glam::Quat::IDENTITY, scale: glam::Vec3::ONE }
                                        };

                                        let view_matrix = create_view_matrix(&view.pose, &player_start);
                                        let proj_matrix = create_projection_matrix(view.fov, 0.01, 100.0);
                                        let view_proj = proj_matrix * view_matrix;

                                        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                                            .render_pass(graphics_context.render_pass)
                                            .framebuffer(xr_context.framebuffers[stream_idx as usize][eye_index])
                                            .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: xr_context.resolution })
                                            .clear_values(&clear_values);

                                        graphics_context.device.cmd_begin_render_pass(graphics_context.command_buffer, &render_pass_begin_info, vk::SubpassContents::INLINE);
                                        
                                        // Set Viewport/Scissor
                                        let viewport = vk::Viewport { x: 0.0, y: 0.0, width: xr_context.resolution.width as f32, height: xr_context.resolution.height as f32, min_depth: 0.0, max_depth: 1.0 };
                                        let scissor = vk::Rect2D { offset: vk::Offset2D::default(), extent: xr_context.resolution };
                                        graphics_context.device.cmd_set_viewport(graphics_context.command_buffer, 0, &[viewport]);
                                        graphics_context.device.cmd_set_scissor(graphics_context.command_buffer, 0, &[scissor]);

                                        // Bind Pipeline & Global Set
                                        graphics_context.device.cmd_bind_pipeline(graphics_context.command_buffer, vk::PipelineBindPoint::GRAPHICS, graphics_context.pipeline);
                                        graphics_context.device.cmd_bind_descriptor_sets(
                                            graphics_context.command_buffer,
                                            vk::PipelineBindPoint::GRAPHICS,
                                            graphics_context.pipeline_layout,
                                            0,
                                            &[global_descriptor_set],
                                            &[]
                                        );

                                        // Push Constants: ViewProj, PrevViewProj, LightSpace, CameraPos
                                        // 3 matrices + 1 vec4 = 208 bytes
                                        let mut push_data = Vec::with_capacity(208);
                                        push_data.extend_from_slice(bytemuck::bytes_of(&view_proj));
                                        push_data.extend_from_slice(bytemuck::bytes_of(&prev_view_projs[eye_index]));
                                        // LightSpaceMatrix (for shadows) - typically LightViewProj * Model. But we are instancing.
                                        // The shader expects `lightSpace` matrix to transform WorldPos to LightClipSpace.
                                        // Usually LightProj * LightView. The Vertex Shader multiplies: `lightSpace * worldPos`.
                                        // yes, just LightViewProj.
                                        push_data.extend_from_slice(bytemuck::bytes_of(&light_view_proj));
                                        // Camera world position for correct view vector in PBR lighting
                                        // Compute directly from HMD pose + player_start to avoid matrix inversion issues
                                        let hmd_pos = glam::Vec3::new(view.pose.position.x, view.pose.position.y, view.pose.position.z);
                                        let camera_world_pos = glam::Vec4::from((player_start.position + player_start.rotation * hmd_pos, 1.0));
                                        push_data.extend_from_slice(bytemuck::bytes_of(&camera_world_pos));
                                        
                                        graphics_context.device.cmd_push_constants(
                                            graphics_context.command_buffer,
                                            graphics_context.pipeline_layout,
                                            vk::ShaderStageFlags::VERTEX,
                                            0,
                                            &push_data
                                        );

                                        // Draw Batches
                                        for (mesh_idx, first_instance, instance_count) in &draw_calls {
                                            let gpu_mesh = &mesh_library[*mesh_idx];
                                            
                                            // Bind Material Set
                                            graphics_context.device.cmd_bind_descriptor_sets(
                                                graphics_context.command_buffer,
                                                vk::PipelineBindPoint::GRAPHICS,
                                                graphics_context.pipeline_layout,
                                                1,
                                                &[gpu_mesh.material_descriptor_set],
                                                &[]
                                            );
                                            
                                            graphics_context.device.cmd_bind_vertex_buffers(graphics_context.command_buffer, 0, &[gpu_mesh.vertex_buffer], &[0]);
                                            graphics_context.device.cmd_bind_index_buffer(graphics_context.command_buffer, gpu_mesh.index_buffer, 0, vk::IndexType::UINT32);
                                            graphics_context.device.cmd_draw_indexed(graphics_context.command_buffer, gpu_mesh.index_count, *instance_count, 0, 0, *first_instance);
                                        }
                                        
                                        // Draw Custom Meshes
                                        for (gpu_mesh, first_instance, instance_count) in &custom_draw_calls {
                                            // Bind Material Set
                                            graphics_context.device.cmd_bind_descriptor_sets(
                                                graphics_context.command_buffer,
                                                vk::PipelineBindPoint::GRAPHICS,
                                                graphics_context.pipeline_layout,
                                                1,
                                                &[gpu_mesh.material_descriptor_set],
                                                &[]
                                            );
                                            
                                            graphics_context.device.cmd_bind_vertex_buffers(graphics_context.command_buffer, 0, &[gpu_mesh.vertex_buffer], &[0]);
                                            graphics_context.device.cmd_bind_index_buffer(graphics_context.command_buffer, gpu_mesh.index_buffer, 0, vk::IndexType::UINT32);
                                            graphics_context.device.cmd_draw_indexed(graphics_context.command_buffer, gpu_mesh.index_count, *instance_count, 0, 0, *first_instance);
                                        }
                                        
                                        // Update Prev ViewProj
                                        prev_view_projs[eye_index] = view_proj;

                                        graphics_context.device.cmd_end_render_pass(graphics_context.command_buffer);
                                    }

                                    let _ = graphics_context.device.end_command_buffer(graphics_context.command_buffer);

                                    // Submit
                                    let cmd_buffers = [graphics_context.command_buffer];
                                    let submit_info = vk::SubmitInfo::builder().command_buffers(&cmd_buffers);
                                    let _ = graphics_context.device.reset_fences(&[graphics_context.fence]);
                                    let _ = graphics_context.device.queue_submit(graphics_context.queue, &[submit_info.build()], graphics_context.fence);
                                    let _ = graphics_context.device.wait_for_fences(&[graphics_context.fence], true, 50_000_000); // 50ms timeout to prevent ANR
                                }

                                // info!("Calling swapchain.release_image");
                                if let Err(e) = xr_context.swapchain.release_image() {
                                    error!("Failed to release image: {:?}", e);
                                }
                                if let Err(e) = xr_context.depth_swapchain.release_image() {
                                    error!("Failed to release depth image: {:?}", e);
                                }
                                if let Err(e) = xr_context.motion_swapchain.release_image() {
                                    error!("Failed to release motion image: {:?}", e);
                                }
                                // info!("swapchain.release_image succeeded");

                                // Views are already located above
                                projection_views = views.iter().enumerate().map(|(i, view)| {
                                    // Normalize orientation to avoid ERROR_POSE_INVALID
                                    let q = view.pose.orientation;
                                    let len = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w).sqrt();
                                    let normalized_pose = if len > 0.0001 {
                                        oxr::Posef {
                                            orientation: oxr::Quaternionf {
                                                x: q.x / len,
                                                y: q.y / len,
                                                z: q.z / len,
                                                w: q.w / len,
                                            },
                                            position: view.pose.position,
                                        }
                                    } else {
                                        // Fallback to identity orientation if invalid
                                        oxr::Posef {
                                            orientation: oxr::Quaternionf { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
                                            position: view.pose.position,
                                        }
                                    };

                                    oxr::CompositionLayerProjectionView::new()
                                        .pose(normalized_pose)
                                        .fov(view.fov)
                                        .sub_image(
                                            oxr::SwapchainSubImage::new()
                                                .swapchain(&xr_context.swapchain)
                                                .image_array_index(i as u32)
                                                .image_rect(oxr::Rect2Di {
                                                    offset: oxr::Offset2Di { x: 0, y: 0 },
                                                    extent: oxr::Extent2Di {
                                                        width: xr_context.resolution.width as i32,
                                                        height: xr_context.resolution.height as i32,
                                                    },
                                                })
                                        )
                                }).collect();
                                
                                 projection_layer_storage = Some(oxr::CompositionLayerProjection::new()
                                    .space(&xr_context.stage_space)
                                    .views(&projection_views));
                                
                                layers.push(projection_layer_storage.as_ref().unwrap());

                        } else {
                            projection_layer_storage = None;
                        }

                        // AppSW Layer Chaining
                        let mut space_warp_layer_info = if projection_layer_storage.is_some() {
                            // Enable AppSW by chaining XR_TYPE_COMPOSITION_LAYER_SPACE_WARP_INFO_FB
                            let motion_sub_image = oxr::sys::SwapchainSubImage {
                                swapchain: xr_context.motion_swapchain.as_raw(),
                                image_rect: oxr::sys::Rect2Di {
                                    offset: oxr::sys::Offset2Di { x: 0, y: 0 },
                                    extent: oxr::sys::Extent2Di {
                                        width: xr_context.resolution.width as i32,
                                        height: xr_context.resolution.height as i32,
                                    },
                                },
                                image_array_index: 0, 
                            };

                            let depth_sub_image = oxr::sys::SwapchainSubImage {
                                swapchain: xr_context.depth_swapchain.as_raw(),
                                image_rect: oxr::sys::Rect2Di {
                                    offset: oxr::sys::Offset2Di { x: 0, y: 0 },
                                    extent: oxr::sys::Extent2Di {
                                        width: xr_context.resolution.width as i32,
                                        height: xr_context.resolution.height as i32,
                                    },
                                },
                                image_array_index: 0,
                            };

                            let info = CompositionLayerSpaceWarpInfoFB {
                                ty: oxr::StructureType::from_raw(1000171000), // XR_TYPE_COMPOSITION_LAYER_SPACE_WARP_INFO_FB
                                next: std::ptr::null(),
                                layer_flags: 0,
                                motion_vector_sub_image: motion_sub_image,
                                app_space_delta_pose: oxr::sys::Posef {
                                    orientation: oxr::sys::Quaternionf { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
                                    position: oxr::sys::Vector3f { x: 0.0, y: 0.0, z: 0.0 },
                                },
                                depth_sub_image: depth_sub_image,
                                min_depth: 0.0,
                                max_depth: 1.0,
                                near_z: 0.01,
                                far_z: 100.0,
                            };
                            Some(info)
                        } else {
                            None
                        };
                        
                        let mut layers_ptrs: Vec<*const oxr::sys::CompositionLayerBaseHeader> = Vec::new();
                        
                        // Reconstruct CompositionLayerProjection as sys type to chain
                        let projection_layer_sys = if projection_layer_storage.is_some() {
                             
                             // Iterate over ORIGINAL views to get data
                             let sys_views: Vec<oxr::sys::CompositionLayerProjectionView> = views.iter().enumerate().map(|(i, view)| {
                                 // Re-normalize (or duplicate logic)
                                    let q = view.pose.orientation;
                                    let len = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w).sqrt();
                                    let normalized_pose = if len > 0.0001 {
                                        oxr::sys::Posef {
                                            orientation: oxr::sys::Quaternionf {
                                                x: q.x / len,
                                                y: q.y / len,
                                                z: q.z / len,
                                                w: q.w / len,
                                            },
                                            position: oxr::sys::Vector3f {
                                                x: view.pose.position.x,
                                                y: view.pose.position.y,
                                                z: view.pose.position.z,
                                            },
                                        }
                                    } else {
                                        oxr::sys::Posef {
                                            orientation: oxr::sys::Quaternionf { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
                                            position: oxr::sys::Vector3f {
                                                x: view.pose.position.x,
                                                y: view.pose.position.y,
                                                z: view.pose.position.z,
                                            },
                                        }
                                    };

                                 oxr::sys::CompositionLayerProjectionView {
                                     ty: oxr::StructureType::COMPOSITION_LAYER_PROJECTION_VIEW,
                                     next: std::ptr::null(),
                                     pose: normalized_pose,
                                     fov: oxr::sys::Fovf {
                                        angle_left: view.fov.angle_left,
                                        angle_right: view.fov.angle_right,
                                        angle_up: view.fov.angle_up,
                                        angle_down: view.fov.angle_down,
                                     },
                                     sub_image: oxr::sys::SwapchainSubImage {
                                         swapchain: xr_context.swapchain.as_raw(), 
                                         image_rect: oxr::sys::Rect2Di {
                                            offset: oxr::sys::Offset2Di { x: 0, y: 0 },
                                            extent: oxr::sys::Extent2Di {
                                                width: xr_context.resolution.width as i32,
                                                height: xr_context.resolution.height as i32,
                                            },
                                         },
                                         image_array_index: i as u32,
                                     },
                                 }
                             }).collect();
                             
                             let mut sys_layer = oxr::sys::CompositionLayerProjection {
                                 ty: oxr::StructureType::COMPOSITION_LAYER_PROJECTION,
                                 next: std::ptr::null(),
                                 layer_flags: oxr::sys::CompositionLayerFlags::EMPTY,
                                 space: xr_context.stage_space.as_raw(),
                                 view_count: sys_views.len() as u32,
                                 views: sys_views.as_ptr(),
                             };
                             
                             // Chain it!
                             if let Some(ref mut info) = space_warp_layer_info {
                                 sys_layer.next = info as *mut _ as *const _;
                             }
                             
                             Some((sys_layer, sys_views)) // Keep views alive!
                        } else {
                            None
                        };

                        if let Some((ref sys_layer, _)) = projection_layer_sys {
                            layers_ptrs.push(sys_layer as *const _ as *const oxr::sys::CompositionLayerBaseHeader);
                        }

                        // Use the blend mode selected during session creation
                        let blend_mode = xr_context.blend_mode;

                        // info!("Calling frame_stream.end (AppSW)");
                        unsafe {
                            let frame_end_info = oxr::sys::FrameEndInfo {
                                ty: oxr::StructureType::FRAME_END_INFO,
                                next: std::ptr::null(),
                                display_time: frame_state.predicted_display_time,
                                environment_blend_mode: blend_mode,
                                layer_count: layers_ptrs.len() as u32,
                                layers: layers_ptrs.as_ptr(),
                            };
                            
                            let fp = xr_context.instance.fp();
                            let res = (fp.end_frame)(xr_context.session.as_raw(), &frame_end_info);
                            if res.into_raw() < 0 {
                                error!("frame_stream.end (sys) failed: {:?}", res);
                            }
                        }
                        // info!("frame_stream.end succeeded");
                    }
                }
                Err(e) => {
                    error!("Failed to wait frame: {:?}", e);
                }
            }
        } else {
            // Frame loop paused
            unsafe {
                if FRAME_COUNT % 120 == 0 { 
                    info!("Frame loop paused - session_running: {}, activity_resumed: {}", session_running, activity_resumed);
                }
            }
        }
    }
}

#[cfg(target_os = "android")]
#[repr(C)]
#[derive(Debug)]
pub struct CompositionLayerSpaceWarpInfoFB {
    pub ty: oxr::StructureType,
    pub next: *const std::ffi::c_void,
    pub layer_flags: u64, // XrCompositionLayerSpaceWarpInfoFlagsFB
    pub motion_vector_sub_image: oxr::sys::SwapchainSubImage,
    pub app_space_delta_pose: oxr::sys::Posef,
    pub depth_sub_image: oxr::sys::SwapchainSubImage,
    pub min_depth: f32,
    pub max_depth: f32,
    pub near_z: f32,
    pub far_z: f32,
}

#[cfg(target_os = "android")]
fn create_view_matrix(pose: &openxr::Posef, world_transform: &world::Transform) -> glam::Mat4 {
    let rotation = glam::Quat::from_xyzw(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    );
    let position = glam::Vec3::new(
        pose.position.x,
        pose.position.y,
        pose.position.z,
    );
    
    // PlayerStart * HMD
    let hmd_transform = glam::Mat4::from_rotation_translation(rotation, position);
    let player_transform = glam::Mat4::from_rotation_translation(world_transform.rotation, world_transform.position);
    
    // The view matrix is the inverse of the camera's world transform
    (player_transform * hmd_transform).inverse()
}

#[cfg(target_os = "android")]
fn create_projection_matrix(fov: openxr::Fovf, near: f32, far: f32) -> glam::Mat4 {
    let tan_left = fov.angle_left.tan();
    let tan_right = fov.angle_right.tan();
    let tan_down = fov.angle_down.tan();
    let tan_up = fov.angle_up.tan();

    let width = tan_right - tan_left;
    let height = tan_up - tan_down;

    let a11 = 2.0 / width;
    let a22 = -2.0 / height; // Flip Y for Vulkan
    let a31 = (tan_right + tan_left) / width;
    let a32 = (tan_up + tan_down) / height;
    let a33 = -far / (far - near);
    let a43 = -(far * near) / (far - near);

    glam::Mat4::from_cols(
        glam::Vec4::new(a11, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, a22, 0.0, 0.0),
        glam::Vec4::new(a31, a32, a33, -1.0),
        glam::Vec4::new(0.0, 0.0, a43, 0.0),
    )
}
