#[cfg(target_os = "android")]
use android_activity::{AndroidApp, MainEvent, PollEvent};
use log::{info, error, debug};
use std::sync::{Arc, RwLock};
#[cfg(target_os = "android")]
use openxr as oxr;
#[cfg(target_os = "android")]
use ash::vk;

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
use world::GameWorld;
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

    info!("STFSC Engine Starting...");

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
    
    let global_descriptor_set = match graphics_context.create_global_descriptor_set(
        graphics_context.shadow_depth_view,
        graphics_context.shadow_sampler
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

    // Load Model (Android Asset Manager)
    #[cfg(target_os = "android")]
    let mesh = {
        use std::io::Read;
        let asset_manager = app.asset_manager();
        let filename = std::ffi::CString::new("cube.obj").unwrap();
        
        let loaded_mesh = if let Some(mut asset) = asset_manager.open(&filename) {
            let mut buffer = Vec::new();
            if asset.read_to_end(&mut buffer).is_ok() {
                match world::load_obj_from_bytes(&buffer) {
                    Ok(m) => {
                        info!("Loaded cube.obj successfully from assets");
                        Some(m)
                    },
                    Err(e) => {
                        error!("Failed to parse cube.obj: {:?}", e);
                        None
                    }
                }
            } else {
                error!("Failed to read cube.obj asset");
                None
            }
        } else {
            error!("Failed to open cube.obj asset");
            None
        };

        loaded_mesh.unwrap_or_else(|| {
            info!("Using fallback cube");
            world::Mesh {
                vertices: vec![
                    world::Vertex { position: [-0.5, -0.5,  0.5], normal: [0.0, 0.0, 1.0], uv: [0.0, 0.0], color: [1.0, 0.0, 0.0], tangent: [1.0, 0.0, 0.0, 1.0] },
                    world::Vertex { position: [ 0.5, -0.5,  0.5], normal: [0.0, 0.0, 1.0], uv: [1.0, 0.0], color: [0.0, 1.0, 0.0], tangent: [1.0, 0.0, 0.0, 1.0] },
                    world::Vertex { position: [ 0.5,  0.5,  0.5], normal: [0.0, 0.0, 1.0], uv: [1.0, 1.0], color: [0.0, 0.0, 1.0], tangent: [1.0, 0.0, 0.0, 1.0] },
                    world::Vertex { position: [-0.5,  0.5,  0.5], normal: [0.0, 0.0, 1.0], uv: [0.0, 1.0], color: [1.0, 1.0, 0.0], tangent: [1.0, 0.0, 0.0, 1.0] },
                    world::Vertex { position: [-0.5, -0.5, -0.5], normal: [0.0, 0.0, -1.0], uv: [0.0, 0.0], color: [0.0, 1.0, 1.0], tangent: [1.0, 0.0, 0.0, 1.0] },
                    world::Vertex { position: [ 0.5, -0.5, -0.5], normal: [0.0, 0.0, -1.0], uv: [1.0, 0.0], color: [1.0, 0.0, 1.0], tangent: [1.0, 0.0, 0.0, 1.0] },
                    world::Vertex { position: [ 0.5,  0.5, -0.5], normal: [0.0, 0.0, -1.0], uv: [1.0, 1.0], color: [1.0, 1.0, 1.0], tangent: [1.0, 0.0, 0.0, 1.0] },
                    world::Vertex { position: [-0.5,  0.5, -0.5], normal: [0.0, 0.0, -1.0], uv: [0.0, 1.0], color: [0.0, 0.0, 0.0], tangent: [1.0, 0.0, 0.0, 1.0] },
                ],
                indices: vec![
                    0u32, 1, 2, 2, 3, 0,
                    4, 5, 6, 6, 7, 4,
                    4, 5, 1, 1, 0, 4,
                    7, 6, 2, 2, 3, 7,
                    4, 7, 3, 3, 0, 4,
                    5, 6, 2, 2, 1, 5,
                ],
                albedo: None,
                normal: None,
                metallic_roughness: None,
            }
        })
    };

    let (vertex_buffer, vertex_memory) = match graphics_context.create_buffer(
        (mesh.vertices.len() * std::mem::size_of::<world::Vertex>()) as u64,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
    ) {
        Ok(ret) => ret,
        Err(e) => {
            error!("Failed to create vertex buffer: {:?}", e);
            return;
        }
    };

    unsafe {
        if let Ok(ptr) = graphics_context.device.map_memory(vertex_memory, 0, (mesh.vertices.len() * std::mem::size_of::<world::Vertex>()) as u64, vk::MemoryMapFlags::empty()) {
            std::ptr::copy_nonoverlapping(mesh.vertices.as_ptr(), ptr as *mut world::Vertex, mesh.vertices.len());
            graphics_context.device.unmap_memory(vertex_memory);
        } else {
             error!("Failed to map vertex memory");
             return;
        }
    }

    let (index_buffer, index_memory) = match graphics_context.create_buffer(
        (mesh.indices.len() * std::mem::size_of::<u32>()) as u64,
        vk::BufferUsageFlags::INDEX_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
    ) {
        Ok(ret) => ret,
        Err(e) => {
            error!("Failed to create index buffer: {:?}", e);
            return;
        }
    };

    unsafe {
        if let Ok(ptr) = graphics_context.device.map_memory(index_memory, 0, (mesh.indices.len() * std::mem::size_of::<u32>()) as u64, vk::MemoryMapFlags::empty()) {
            std::ptr::copy_nonoverlapping(mesh.indices.as_ptr(), ptr as *mut u32, mesh.indices.len());
            graphics_context.device.unmap_memory(index_memory);
        } else {
            error!("Failed to map index memory");
            return;
        }
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
    }
    
    let game_state = Arc::new(RwLock::new(GameState {
        physics: physics_world,
        world: game_world,
    }));

    // Game Loop Thread
    let game_state_thread = game_state.clone();
    let mut game_quit = false; // TODO: Handle quit signal properly across threads
    
    std::thread::spawn(move || {
        info!("Game Loop Thread Started");
        while !game_quit {
            let start = std::time::Instant::now();
            {
                if let Ok(mut state) = game_state_thread.write() {
                    state.physics.step();
                    state.world.update_streaming();
                }
            }
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
                            
                            if let Ok(update) = bincode::deserialize::<world::SceneUpdate>(&data) {
                                info!("Received update: {:?}", update);
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
    let mut activity_resumed = false;
    
    // State for AppSW (Motion Vectors)
    // Store previous frame's View Projection Matrix per eye
    let mut prev_view_projs = [glam::Mat4::IDENTITY; 2];

    while !quit {
        let timeout = if session_running {
            std::time::Duration::from_millis(0)
        } else {
            std::time::Duration::from_millis(100)
        };

        app.poll_events(Some(timeout), |event| {
            match event {
                PollEvent::Main(MainEvent::Destroy) => {
                    info!("MainEvent::Destroy received, quitting...");
                    quit = true;
                }
                PollEvent::Main(MainEvent::InitWindow { .. }) => {
                    info!("Window Initialized");
                }
                PollEvent::Main(MainEvent::Resume { .. }) => {
                    info!("MainEvent::Resume received");
                    activity_resumed = true;
                    info!("Activity state: resumed");
                }
                PollEvent::Main(MainEvent::Pause { .. }) => {
                    info!("MainEvent::Pause received");
                    activity_resumed = false;
                    info!("Activity state: paused");
                }
                PollEvent::Main(MainEvent::ConfigChanged { .. }) => {
                    info!("MainEvent::ConfigChanged received");
                }
                PollEvent::Main(MainEvent::WindowResized { .. }) => {
                    info!("MainEvent::WindowResized received");
                }
                PollEvent::Main(MainEvent::RedrawNeeded { .. }) => {
                    // info!("MainEvent::RedrawNeeded received");
                }
                PollEvent::Main(MainEvent::InputAvailable { .. }) => {
                    info!("MainEvent::InputAvailable received");
                }
                _ => {
                    // info!("Other event: {:?}", event);
                }
            }
        });
        
        // Heartbeat log
        static mut FRAME_COUNT: u64 = 0;
        unsafe {
            FRAME_COUNT += 1;
            if FRAME_COUNT % 60 == 0 {
                info!("Heartbeat: Frame {}", FRAME_COUNT);
            }
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

            // Mesh Upload Logic
            {
                let mut state = game_state.write().unwrap();
                let mut meshes_to_upload = Vec::new();
                
                for (id, (mesh, _transform)) in state.world.ecs.query::<(&world::Mesh, &world::Transform)>().iter() {
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
                        
                        if let Some(data) = &mesh.albedo {
                             if let Ok(img) = image::load_from_memory(data) {
                                if let Ok(tex) = graphics_context.create_texture_from_image(&img.to_rgba8()) {
                                    tex_albedo_opt = Some(tex);
                                }
                             }
                        }
                         if let Some(data) = &mesh.normal {
                             if let Ok(img) = image::load_from_memory(data) {
                                if let Ok(tex) = graphics_context.create_texture_from_image(&img.to_rgba8()) {
                                    tex_normal_opt = Some(tex);
                                }
                             }
                        }
                         if let Some(data) = &mesh.metallic_roughness {
                             if let Ok(img) = image::load_from_memory(data) {
                                if let Ok(tex) = graphics_context.create_texture_from_image(&img.to_rgba8()) {
                                    tex_mr_opt = Some(tex);
                                }
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

            info!("Calling xrWaitFrame");
            match xr_context.frame_waiter.wait() {
                Ok(frame_state) => {
                    info!("xrWaitFrame returned, should_render: {}", frame_state.should_render);

                    info!("Calling frame_stream.begin");
                    if let Err(e) = xr_context.frame_stream.begin() {
                         error!("Failed to begin frame: {:?}", e);
                         continue;
                    }
                    info!("frame_stream.begin succeeded");

                    let mut layers: Vec<&oxr::CompositionLayerBase<oxr::Vulkan>> = Vec::new();
                    let projection_views: Vec<oxr::CompositionLayerProjectionView<oxr::Vulkan>>;
                    let projection_layer_storage: Option<oxr::CompositionLayerProjection<oxr::Vulkan>>;

                    if frame_state.should_render {
                        let (view_flags, views) = xr_context.session.locate_views(
                            ViewConfigurationType::PRIMARY_STEREO,
                            frame_state.predicted_display_time,
                            &xr_context.stage_space,
                        ).unwrap();

                        info!("Locate views succeeded, flags: {:?}", view_flags);

                        info!("Calling swapchain.acquire_image");
                        let stream_idx = match xr_context.swapchain.acquire_image() {
                            Ok(idx) => idx,
                            Err(e) => {
                                error!("Failed to acquire image: {:?}", e);
                                9999 // Sentinel
                            }
                        };

                        if stream_idx != 9999 {
                            info!("swapchain.acquire_image succeeded: {}", stream_idx);
                            
                            // Acquire Depth and Motion images (assume synced)
                            let _depth_idx = xr_context.depth_swapchain.acquire_image().unwrap_or(0);
                            let _motion_idx = xr_context.motion_swapchain.acquire_image().unwrap_or(0);

                            info!("Calling swapchain.wait_image");
                            if let Err(e) = xr_context.swapchain.wait_image(Duration::INFINITE) {
                                 error!("Failed to wait image: {:?}", e);
                            }
                            if let Err(e) = xr_context.depth_swapchain.wait_image(Duration::INFINITE) {
                                 error!("Failed to wait depth image: {:?}", e);
                            }
                            if let Err(e) = xr_context.motion_swapchain.wait_image(Duration::INFINITE) {
                                 error!("Failed to wait motion image: {:?}", e);
                            }
                            
                            info!("swapchain.wait_image succeeded");

                                // Record Command Buffer
                                unsafe {
                                    info!("Resetting Command Buffer: {:?}", graphics_context.command_buffer);
                                    let _ = graphics_context.device.reset_command_buffer(graphics_context.command_buffer, vk::CommandBufferResetFlags::empty());

                                    let begin_info = vk::CommandBufferBeginInfo::builder()
                                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                                    
                                    let _ = graphics_context.device.begin_command_buffer(graphics_context.command_buffer, &begin_info);

                                    // Render Loop for Stereo (2 Eyes)
                                    // Calculate Light Matrices once per frame (or here to be safe)
                                    let light_pos = glam::Vec3::new(10.0, 20.0, 4.0);
                                    let light_proj = glam::Mat4::orthographic_rh(-10.0, 10.0, -10.0, 10.0, 1.0, 50.0);
                                    let light_view = glam::Mat4::look_at_rh(light_pos, glam::Vec3::ZERO, glam::Vec3::Y);

                                    // --- Shadow Pass (Moved Outside Eye Loop) ---
                                    {
                                        let clear_values = [
                                            vk::ClearValue {
                                                depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
                                            }
                                        ];
                                        
                                        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                                            .render_pass(graphics_context.shadow_render_pass)
                                            .framebuffer(graphics_context.shadow_framebuffer)
                                            .render_area(vk::Rect2D {
                                                offset: vk::Offset2D { x: 0, y: 0 },
                                                extent: graphics_context.shadow_extent,
                                            })
                                            .clear_values(&clear_values);

                                        info!("Shadow Pass: RP={:?}, FB={:?}, CB={:?}", graphics_context.shadow_render_pass, graphics_context.shadow_framebuffer, graphics_context.command_buffer);
                                        if graphics_context.command_buffer == vk::CommandBuffer::null() {
                                            error!("CRITICAL: Command Buffer is NULL before Shadow Pass!");
                                        }

                                        unsafe {
                                            graphics_context.device.cmd_begin_render_pass(
                                                graphics_context.command_buffer,
                                                &render_pass_begin_info,
                                                vk::SubpassContents::INLINE,
                                            );
                                            
                                            graphics_context.device.cmd_bind_pipeline(
                                                graphics_context.command_buffer, 
                                                vk::PipelineBindPoint::GRAPHICS, 
                                                graphics_context.shadow_pipeline
                                            );
                                        }
                                        
                                        // Draw entities for shadow
                                        if let Ok(state) = game_state.read() {
                                            for (_id, (transform, gpu_mesh)) in state.world.ecs.query::<(&world::Transform, &GpuMesh)>().iter() {
                                                let model = glam::Mat4::from_scale_rotation_translation(transform.scale, transform.rotation, transform.position);
                                                
                                                let mut shadow_push = Vec::new();
                                                shadow_push.extend_from_slice(bytemuck::bytes_of(&model));
                                                shadow_push.extend_from_slice(bytemuck::bytes_of(&light_view));
                                                shadow_push.extend_from_slice(bytemuck::bytes_of(&light_proj));
                                                
                                                graphics_context.device.cmd_push_constants(
                                                    graphics_context.command_buffer,
                                                    graphics_context.shadow_pipeline_layout,
                                                    vk::ShaderStageFlags::VERTEX,
                                                    0,
                                                    &shadow_push
                                                );
                                                
                                                graphics_context.device.cmd_bind_vertex_buffers(graphics_context.command_buffer, 0, &[gpu_mesh.vertex_buffer], &[0]);
                                                graphics_context.device.cmd_bind_index_buffer(graphics_context.command_buffer, gpu_mesh.index_buffer, 0, vk::IndexType::UINT32);
                                                graphics_context.device.cmd_draw_indexed(graphics_context.command_buffer, gpu_mesh.index_count, 1, 0, 0, 0);
                                            }
                                        }
                                        
                                        unsafe {
                                            graphics_context.device.cmd_end_render_pass(graphics_context.command_buffer);
                                        }
                                    }

                                    for eye_index in 0..2 {


                    let clear_values = [
                                            vk::ClearValue {
                                                color: vk::ClearColorValue {
                                                    float32: [0.1, 0.1, 0.2, 1.0], // Dark Blue Opaque for Immersive VR
                                                },
                                            },
                                            vk::ClearValue {
                                                depth_stencil: vk::ClearDepthStencilValue {
                                                    depth: 1.0,
                                                    stencil: 0,
                                                },
                                            },
                                            vk::ClearValue { // Motion Vector Clear (0,0 = no motion)
                                                color: vk::ClearColorValue {
                                                    float32: [0.0, 0.0, 0.0, 0.0],
                                                },
                                            }
                                        ];
                                        
                                        let view = &views[eye_index];
                                        if eye_index == 0 && FRAME_COUNT % 120 == 0 {
                                            info!("View 0 Flags: {:?}", view_flags);
                                            info!("Active Blend Mode: {:?}", xr_context.blend_mode);
                                            info!("View 0 FOV: {:?}", view.fov);
                                            info!("View 0 Pose: Pos={:?}, Ori={:?}", view.pose.position, view.pose.orientation);
                                            info!("Resolution: {:?}", xr_context.resolution);
                                        }

                                        let view_matrix = create_view_matrix(&view.pose);
                                        let proj_matrix = create_projection_matrix(view.fov, 0.01, 100.0);

                                        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                                            .render_pass(graphics_context.render_pass)
                                            .framebuffer(xr_context.framebuffers[stream_idx as usize][eye_index])
                                            .render_area(vk::Rect2D {
                                                offset: vk::Offset2D { x: 0, y: 0 },
                                                extent: xr_context.resolution,
                                            })
                                            .clear_values(&clear_values);

                                        info!("Main Pass (Eye {}): RP={:?}, FB={:?}, CB={:?}", eye_index, graphics_context.render_pass, xr_context.framebuffers[stream_idx as usize][eye_index], graphics_context.command_buffer);
                                        graphics_context.device.cmd_begin_render_pass(graphics_context.command_buffer, &render_pass_begin_info, vk::SubpassContents::INLINE);
                                        
                                        // Render Entities
                                        graphics_context.device.cmd_bind_pipeline(graphics_context.command_buffer, vk::PipelineBindPoint::GRAPHICS, graphics_context.pipeline);
                                        
                                        // Bind Global Descriptor Set (Set 0)
                                        graphics_context.device.cmd_bind_descriptor_sets(
                                            graphics_context.command_buffer,
                                            vk::PipelineBindPoint::GRAPHICS,
                                            graphics_context.pipeline_layout,
                                            0,
                                            &[global_descriptor_set],
                                            &[],
                                        );

                                        // Bind Material Descriptor Set (Per Object)
                                        // But here we are iterating.
                                        // Wait, the previous replacement was:
                                        // graphics_context.device.cmd_bind_descriptor_sets(..., 1, &[material_descriptor_set], ...);
                                        // We should now use `gpu_mesh.material_descriptor_set`.
                                        
                                        // Viewport and Scissor
                                        let viewport = vk::Viewport {
                                            x: 0.0,
                                            y: 0.0,
                                            width: xr_context.resolution.width as f32,
                                            height: xr_context.resolution.height as f32,
                                            min_depth: 0.0,
                                            max_depth: 1.0,
                                        };
                                        let scissor = vk::Rect2D {
                                            offset: vk::Offset2D { x: 0, y: 0 },
                                            extent: xr_context.resolution,
                                        };
                                        graphics_context.device.cmd_set_viewport(graphics_context.command_buffer, 0, &[viewport]);
                                        graphics_context.device.cmd_set_scissor(graphics_context.command_buffer, 0, &[scissor]);

                                        // Update ViewProj
                                        let view_proj = proj_matrix * view_matrix;
                                        
                                        // Render ECS objects
                                        if let Ok(state) = game_state.read() {
                                            for (_id, (transform, gpu_mesh)) in state.world.ecs.query::<(&world::Transform, &GpuMesh)>().iter() {
                                                let model_matrix = glam::Mat4::from_scale_rotation_translation(
                                                    transform.scale,
                                                    transform.rotation,
                                                    transform.position
                                                );
                                                
                                                // Prepare PushConstants: Model, ViewProj, PrevViewProj, LightSpace
                                                // 64 bytes each => 256 total
                                                let mut push_data = Vec::with_capacity(256);
                                                push_data.extend_from_slice(bytemuck::bytes_of(&model_matrix));
                                                push_data.extend_from_slice(bytemuck::bytes_of(&view_proj));
                                                push_data.extend_from_slice(bytemuck::bytes_of(&prev_view_projs[eye_index]));
                                                
                                                // Light Space (Placeholder or calculated)
                                                 let light_space_matrix = light_proj * light_view * model_matrix; // Just one example
                                                push_data.extend_from_slice(bytemuck::bytes_of(&light_space_matrix));
                                                
                                                graphics_context.device.cmd_push_constants(
                                                    graphics_context.command_buffer,
                                                    graphics_context.pipeline_layout,
                                                    vk::ShaderStageFlags::VERTEX,
                                                    0,
                                                    &push_data
                                                );
                                                
                                                // Bind Material Descriptor Set (Set 1)
                                                graphics_context.device.cmd_bind_descriptor_sets(
                                                    graphics_context.command_buffer,
                                                    vk::PipelineBindPoint::GRAPHICS,
                                                    graphics_context.pipeline_layout,
                                                    1,
                                                    &[gpu_mesh.material_descriptor_set],
                                                    &[],
                                                );
                                                
                                                graphics_context.device.cmd_bind_vertex_buffers(graphics_context.command_buffer, 0, &[gpu_mesh.vertex_buffer], &[0]);
                                                graphics_context.device.cmd_bind_index_buffer(graphics_context.command_buffer, gpu_mesh.index_buffer, 0, vk::IndexType::UINT32);
                                                graphics_context.device.cmd_draw_indexed(graphics_context.command_buffer, gpu_mesh.index_count, 1, 0, 0, 0);
                                            }
                                        }

                                        // Update Prev ViewProj for next frame
                                        prev_view_projs[eye_index] = view_proj; 
                                        
                                        unsafe {
                                            graphics_context.device.cmd_end_render_pass(graphics_context.command_buffer);
                                        }
                                    
                                    }
                                    let _ = graphics_context.device.end_command_buffer(graphics_context.command_buffer);

                                    let command_buffers = [graphics_context.command_buffer];
                                    let submit_info = vk::SubmitInfo::builder()
                                        .command_buffers(&command_buffers);
                                    
                                    // Reset fence before submission
                                    let _ = graphics_context.device.reset_fences(&[graphics_context.fence]);

                                    if let Err(e) = graphics_context.device.queue_submit(graphics_context.queue, &[submit_info.build()], graphics_context.fence) {
                                        error!("Failed to submit queue: {:?}", e);
                                    }

                                    // Wait for fence to ensure GPU is done before releasing image
                                    if let Err(e) = graphics_context.device.wait_for_fences(&[graphics_context.fence], true, u64::MAX) {
                                        error!("Failed to wait for fences: {:?}", e);
                                    }
                                }

                                info!("Calling swapchain.release_image");
                                if let Err(e) = xr_context.swapchain.release_image() {
                                    error!("Failed to release image: {:?}", e);
                                }
                                if let Err(e) = xr_context.depth_swapchain.release_image() {
                                    error!("Failed to release depth image: {:?}", e);
                                }
                                if let Err(e) = xr_context.motion_swapchain.release_image() {
                                    error!("Failed to release motion image: {:?}", e);
                                }
                                info!("swapchain.release_image succeeded");

                                // Views are already located above
                                projection_views = views.into_iter().enumerate().map(|(i, view)| {
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

                    } else {
                        projection_layer_storage = None;
                    }

                    // Use the blend mode selected during session creation
                    let blend_mode = xr_context.blend_mode;

                    info!("Calling frame_stream.end");
                    if let Err(e) = xr_context.frame_stream.end(
                        frame_state.predicted_display_time,
                        blend_mode,
                        &layers,
                    ) {
                        log::error!("frame_stream.end failed: {}", e);
                    }
                    info!("frame_stream.end succeeded");
                }
                Err(e) => {
                    error!("Failed to wait frame: {:?}", e);
                }
            }
        } else {
            // Frame loop paused: session not running or activity not resumed
            unsafe {
                if FRAME_COUNT % 120 == 0 { // Log occasionally to avoid spam
                    info!("Frame loop paused - session_running: {}, activity_resumed: {}", session_running, activity_resumed);
                }
            }
        }
    }
}

#[cfg(target_os = "android")]
fn create_view_matrix(pose: &openxr::Posef) -> glam::Mat4 {
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
    glam::Mat4::from_rotation_translation(rotation, position).inverse()
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
