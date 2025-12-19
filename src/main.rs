use stfsc_engine::graphics::{GraphicsContext, GpuMesh, InstanceData, Texture};
use stfsc_engine::world::{GameWorld, MeshHandle, Material, Transform, RigidBodyHandle, Mesh, AudioSource};
use stfsc_engine::audio::{AudioSystem, AudioBuffer, AudioSourceProperties, AttenuationModel, AudioBufferHandle};
use stfsc_engine::physics::PhysicsWorld;
use stfsc_engine::resource_loader::ResourceLoader;
use stfsc_engine::lighting::{self, LightUBO, GpuLightData};
use stfsc_engine::graphics::occlusion::OcclusionCuller;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{WindowBuilder, CursorGrabMode};
use winit::event::{Event, WindowEvent, DeviceEvent, ElementState, MouseButton, KeyEvent};
use winit::keyboard::{PhysicalKey, KeyCode, Key, NamedKey};
use log::info;
use std::sync::{Arc, RwLock};
use ash::vk;
use std::collections::{HashMap, HashSet};

/// Cached texture entry with material descriptor set
struct TextureEntry {
    #[allow(dead_code)]
    texture: Texture,
    material_descriptor_set: vk::DescriptorSet,
}

fn main() {
    env_logger::init();

    info!("STFSC Engine - Linux Desktop Target");

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("STFSC Engine - 556 Downtown")
        .with_inner_size(winit::dpi::LogicalSize::new(1920.0, 1080.0))
        .build(&event_loop)
        .unwrap();

    let graphics_context = Arc::new(GraphicsContext::new_desktop(&window).expect("Failed to create GraphicsContext"));
    
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
    let physics_world = PhysicsWorld::new();
    let game_world = GameWorld::new();
    let cmd_tx = game_world.command_sender.clone();

    struct GameState {
        physics: PhysicsWorld,
        world: GameWorld,
        player_position: glam::Vec3,
        player_velocity_y: f32, // For jumping
        player_yaw: f32,
        player_pitch: f32,
    }

    let game_state = Arc::new(RwLock::new(GameState {
        physics: physics_world,
        world: game_world,
        player_position: glam::Vec3::new(0.0, 1.7, 5.0), // Start further back to see more
        player_velocity_y: 0.0,
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
                                println!("NETWORK: Received SceneUpdate: {:?}", update);
                                info!("Received client update: {:?}", update);
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
                    let pos = state.player_position;
                    state.world.update_streaming(pos);
                    let GameState { world, physics, .. } = &mut *state;
                    world.update_logic(physics, 0.016);
                    
                    // Sync physics to ECS
                    let mut updates = Vec::new();
                    for (id, (_transform, handle)) in state.world.ecs.query::<(&Transform, &RigidBodyHandle)>().iter() {
                        if let Some(body) = state.physics.rigid_body_set.get(handle.0) {
                            let p = body.translation();
                            let r = body.rotation();
                            updates.push((id, glam::Vec3::new(p.x, p.y, p.z), glam::Quat::from_xyzw(r.i, r.j, r.k, r.w)));
                        }
                    }
                    for (id, p, r) in updates {
                        if let Ok(mut t) = state.world.ecs.get::<&mut Transform>(id) {
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
                if !cursor_captured {
                    cursor_captured = true;
                    window.set_cursor_visible(false);
                    let _ = window.set_cursor_grab(CursorGrabMode::Locked)
                        .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
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
                        // println!("KEY: {:?} (Logical: {:?})", physical_key, key_event.logical_key);

                        // Escape OR Grave (backtick/tilde key) to release cursor
                        let is_escape = physical_key == PhysicalKey::Code(KeyCode::Escape) 
                            || key_event.logical_key == Key::Named(NamedKey::Escape);
                        let is_grave = physical_key == PhysicalKey::Code(KeyCode::Backquote)
                            || key_event.logical_key == Key::Character("`".into());

                        if is_escape || is_grave {
                            if cursor_captured {
                                println!("CURSOR: Releasing mouse grab (key: {:?})", physical_key);
                                cursor_captured = false;
                                window.set_cursor_visible(true);
                                let _ = window.set_cursor_grab(CursorGrabMode::None);
                            }
                        }
                    }
                    ElementState::Released => {
                        keys_pressed.remove(&physical_key);
                    }
                }
            }
            Event::WindowEvent { event: WindowEvent::Focused(focused), .. } => {
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

                // 0. Update Player Position
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

                        if move_dir.length_squared() > 0.0 {
                            state.player_position += move_dir.normalize() * speed * dt;
                        }

                        // --- Jump & Gravity Logic ---
                        let gravity = 20.0;
                        let jump_force = 8.0;
                        let floor_y = 1.7; // Head height on ground

                        state.player_velocity_y -= gravity * dt;
                        state.player_position.y += state.player_velocity_y * dt;

                        if state.player_position.y <= floor_y {
                            state.player_position.y = floor_y;
                            state.player_velocity_y = 0.0;

                            // Allow jump if on ground
                            if keys_pressed.contains(&PhysicalKey::Code(KeyCode::Space)) {
                                state.player_velocity_y = jump_force;
                                println!("PLAYER: Jump!");
                            }
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
                        let mut audio_source_count = 0;
                        for (id, (transform, source)) in state.world.ecs.query::<(&Transform, &AudioSource)>().iter() {
                            audio_source_count += 1;
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

                // Texture Processing - Create GPU textures from pending uploads
                if let Ok(mut state) = game_state.write() {
                    let pending: Vec<_> = state.world.pending_texture_uploads.drain().collect();
                    for (texture_id, data) in pending {
                        if !texture_cache.contains_key(&texture_id) {
                            println!("TEXTURE: Creating GPU texture '{}' ({} bytes)", texture_id, data.len());
                            match graphics_context.create_texture_from_bytes(&data) {
                                Ok(texture) => {
                                    // Create descriptor set with custom albedo + default normal/mr
                                    match graphics_context.create_material_descriptor_set(&texture, &normal_tex, &mr_tex) {
                                        Ok(descriptor_set) => {
                                            println!("TEXTURE: Created texture '{}' with descriptor set", texture_id);
                                            texture_cache.insert(texture_id, TextureEntry {
                                                texture,
                                                material_descriptor_set: descriptor_set,
                                            });
                                        }
                                        Err(e) => {
                                            println!("TEXTURE ERROR: Failed to create descriptor set for '{}': {:?}", texture_id, e);
                                        }
                                    }
                                }
                                Err(e) => {
                                    println!("TEXTURE ERROR: Failed to create '{}': {:?}", texture_id, e);
                                }
                            }
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
                        resource_loader.queue_upload(id, mesh.clone());
                        pending_uploads.insert(id);
                    }
                }
                for (id, loaded_data) in resource_loader.poll_processed() {
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

                // 3. Prepare Draw Data
                let mut batch_map: HashMap<usize, Vec<InstanceData>> = HashMap::new();
                let mut custom_draws: Vec<(GpuMesh, InstanceData)> = Vec::new();
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
                    
                    // Vulkan projection: Z is [0, 1], Y is flipped
                    let mut proj_matrix = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, aspect, 0.01, 1000.0);
                    proj_matrix.col_mut(1).y *= -1.0;
                    // glam perspective_rh gives [-1, 1] Z. For Vulkan [0, 1], we adjust:
                    // new_z = (old_z + 1) * 0.5
                    let mut correction = glam::Mat4::IDENTITY;
                    correction.col_mut(2).z = 0.5;
                    correction.col_mut(3).z = 0.5;
                    proj_matrix = correction * proj_matrix;

                    view_proj = proj_matrix * view_matrix;

                    occlusion_culler.update_frustum(view_proj);

                    // Collect Regular MeshHandles (separate textured vs non-textured)
                    
                    for (id, (handle, transform)) in state.world.ecs.query::<(&MeshHandle, &Transform)>().iter() {
                        if handle.0 < mesh_library.len() {
                            let gpu_mesh = &mesh_library[handle.0];
                            let model = glam::Mat4::from_scale_rotation_translation(transform.scale, transform.rotation, transform.position);
                            if true || occlusion_culler.is_visible(&gpu_mesh.aabb.transform(model)) {
                                let entity_id = id.id() as u64;
                                let prev_model = *prev_transforms.get(&entity_id).unwrap_or(&model);
                                prev_transforms.insert(entity_id, model);
                                let material = state.world.ecs.get::<&Material>(id).ok();
                                let color = material.as_ref().map(|m| m.color).unwrap_or([1.0, 1.0, 1.0, 1.0]);
                                
                                // Check if entity has a custom texture
                                let has_custom_texture = material
                                    .as_ref()
                                    .and_then(|m| m.albedo_texture.as_ref())
                                    .and_then(|tex_id| texture_cache.get(tex_id));
                                
                                if let Some(tex_entry) = has_custom_texture {
                                    // Draw textured entities individually with their custom descriptor set
                                    textured_draws.push((handle.0, InstanceData { model, prev_model, color }, tex_entry.material_descriptor_set));
                                } else {
                                    // Batch non-textured entities
                                    batch_map.entry(handle.0).or_default().push(InstanceData { model, prev_model, color });
                                }
                            }
                        }
                    }
                    // Collect Custom GpuMeshes
                    for (id, (gpu_mesh, transform)) in state.world.ecs.query::<(&GpuMesh, &Transform)>().iter() {
                        if state.world.ecs.get::<&MeshHandle>(id).is_ok() { continue; }
                        let model = glam::Mat4::from_scale_rotation_translation(transform.scale, transform.rotation, transform.position);
                        if true || occlusion_culler.is_visible(&gpu_mesh.aabb.transform(model)) {
                            let entity_id = id.id() as u64;
                            let prev_model = *prev_transforms.get(&entity_id).unwrap_or(&model);
                            prev_transforms.insert(entity_id, model);
                            let color = state.world.ecs.get::<&Material>(id).map(|m| m.color).unwrap_or([1.0, 1.0, 1.0, 1.0]);
                            custom_draws.push((gpu_mesh.clone(), InstanceData { model, prev_model, color }));
                        }
                    }

                    // Lights
                    let mut light_ubo = LightUBO::new();
                    for (_id, (light, transform)) in state.world.ecs.query::<(&stfsc_engine::world::LightComponent, &Transform)>().iter() {
                        let direction = transform.rotation * glam::Vec3::NEG_Z;
                        let gpu_light = GpuLightData::from_light(&lighting::Light {
                            light_type: match light.light_type {
                                stfsc_engine::world::LightType::Point => lighting::LightType::Point,
                                stfsc_engine::world::LightType::Spot => lighting::LightType::Spot,
                                stfsc_engine::world::LightType::Directional => lighting::LightType::Directional,
                            },
                            color: glam::Vec3::from_array([light.color[0], light.color[1], light.color[2]]),
                            intensity: light.intensity, range: light.range,
                            inner_cone_angle: light.inner_cone_angle, outer_cone_angle: light.outer_cone_angle,
                            cast_shadows: light.cast_shadows,
                        }, transform.position, direction, -1);
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
                let mut custom_draw_calls = Vec::new();
                for (mesh, data) in custom_draws {
                    if instance_offset + 1 > max_instances { break; }
                    unsafe { std::ptr::write(instance_ptr.add(instance_offset), data); }
                    custom_draw_calls.push((mesh, instance_offset as u32, 1u32));
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

                // Shadow Matrices
                let shadow_center = player_pos;
                let light_pos = shadow_center + glam::Vec3::new(20.0, 50.0, 20.0);
                let light_view_proj = glam::Mat4::orthographic_rh(-50.0, 50.0, -50.0, 50.0, 1.0, 150.0) * glam::Mat4::look_at_rh(light_pos, shadow_center, glam::Vec3::Y);

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
                        .clear_values(&[vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } }]);
                    graphics_context.device.cmd_begin_render_pass(cmd, &shadow_rp_begin, vk::SubpassContents::INLINE);
                    graphics_context.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, graphics_context.shadow_pipeline);
                    graphics_context.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, graphics_context.shadow_pipeline_layout, 0, &[global_descriptor_set], &[]);
                    graphics_context.device.cmd_push_constants(cmd, graphics_context.shadow_pipeline_layout, vk::ShaderStageFlags::VERTEX, 0, bytemuck::bytes_of(&light_view_proj));
                    for (mesh_idx, first, count) in &draw_calls {
                        let m = &mesh_library[*mesh_idx];
                        graphics_context.device.cmd_bind_vertex_buffers(cmd, 0, &[m.vertex_buffer], &[0]);
                        graphics_context.device.cmd_bind_index_buffer(cmd, m.index_buffer, 0, vk::IndexType::UINT32);
                        graphics_context.device.cmd_draw_indexed(cmd, m.index_count, *count, 0, 0, *first);
                    }
                    graphics_context.device.cmd_end_render_pass(cmd);

                    // Main Pass
                    let clear_values = [
                        vk::ClearValue { color: vk::ClearColorValue { float32: [0.1, 0.1, 0.15, 1.0] } },
                        vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
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
                    for (m, first, count) in &custom_draw_calls {
                        graphics_context.device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, graphics_context.pipeline_layout, 1, &[m.material_descriptor_set], &[]);
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
