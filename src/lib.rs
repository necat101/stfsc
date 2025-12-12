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
        app.poll_events(None, |event| {
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
                decoded_albedo: None,
                decoded_normal: None,
                decoded_mr: None,
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
                                    world::SceneUpdate::Spawn { id: 0, position: [0.0; 3], rotation: [0.0; 4], color: [1.0, 0.0, 1.0] } // Error sentinel
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
                        
                        if let Some(decoded) = &mesh.decoded_albedo {
                             if let Ok(tex) = graphics_context.create_texture_from_raw(decoded.width, decoded.height, &decoded.data) {
                                 tex_albedo_opt = Some(tex);
                             }
                        } else if let Some(data) = &mesh.albedo {
                             if let Ok(img) = image::load_from_memory(data) {
                                if let Ok(tex) = graphics_context.create_texture_from_image(&img.to_rgba8()) {
                                    tex_albedo_opt = Some(tex);
                                }
                             }
                        }
                        
                        if let Some(decoded) = &mesh.decoded_normal {
                             if let Ok(tex) = graphics_context.create_texture_from_raw(decoded.width, decoded.height, &decoded.data) {
                                 tex_normal_opt = Some(tex);
                             }
                        } else if let Some(data) = &mesh.normal {
                             if let Ok(img) = image::load_from_memory(data) {
                                if let Ok(tex) = graphics_context.create_texture_from_image(&img.to_rgba8()) {
                                    tex_normal_opt = Some(tex);
                                }
                             }
                        }
                        
                        if let Some(decoded) = &mesh.decoded_mr {
                             if let Ok(tex) = graphics_context.create_texture_from_raw(decoded.width, decoded.height, &decoded.data) {
                                 tex_mr_opt = Some(tex);
                             }
                        } else if let Some(data) = &mesh.metallic_roughness {
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

                        info!("Calling frame_stream.end (AppSW)");
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
                        info!("frame_stream.end succeeded");
                    }
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
