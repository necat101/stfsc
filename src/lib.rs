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

// Import commonly used OpenXR types
#[cfg(target_os = "android")]
use oxr::{
    EventDataBuffer, SessionState, ViewConfigurationType, 
    ReferenceSpaceType, Posef, CompositionLayerProjectionView,
    SwapchainSubImage, Rect2Di, Offset2Di, Extent2Di,
    CompositionLayerProjection, EnvironmentBlendMode, Duration
};

#[cfg(target_os = "android")]
#[derive(Clone, Copy, Debug)]
pub struct GpuMesh {
    pub vertex_buffer: vk::Buffer,
    pub vertex_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    pub index_memory: vk::DeviceMemory,
    pub index_count: u32,
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

    // Create Mesh Buffers (Cube)
    let vertices = [
        // Front
        world::Vertex { position: [-0.5, -0.5,  0.5], color: [1.0, 0.0, 0.0] },
        world::Vertex { position: [ 0.5, -0.5,  0.5], color: [0.0, 1.0, 0.0] },
        world::Vertex { position: [ 0.5,  0.5,  0.5], color: [0.0, 0.0, 1.0] },
        world::Vertex { position: [-0.5,  0.5,  0.5], color: [1.0, 1.0, 0.0] },
        // Back
        world::Vertex { position: [-0.5, -0.5, -0.5], color: [0.0, 1.0, 1.0] },
        world::Vertex { position: [ 0.5, -0.5, -0.5], color: [1.0, 0.0, 1.0] },
        world::Vertex { position: [ 0.5,  0.5, -0.5], color: [1.0, 1.0, 1.0] },
        world::Vertex { position: [-0.5,  0.5, -0.5], color: [0.0, 0.0, 0.0] },
    ];
    let indices = [
        0u32, 1, 2, 2, 3, 0, // Front
        4, 5, 6, 6, 7, 4, // Back
        4, 5, 1, 1, 0, 4, // Bottom
        7, 6, 2, 2, 3, 7, // Top
        4, 7, 3, 3, 0, 4, // Left
        5, 6, 2, 2, 1, 5, // Right
    ];

    let (vertex_buffer, vertex_memory) = match graphics_context.create_buffer(
        (vertices.len() * std::mem::size_of::<world::Vertex>()) as u64,
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
        if let Ok(ptr) = graphics_context.device.map_memory(vertex_memory, 0, (vertices.len() * std::mem::size_of::<world::Vertex>()) as u64, vk::MemoryMapFlags::empty()) {
            std::ptr::copy_nonoverlapping(vertices.as_ptr(), ptr as *mut world::Vertex, vertices.len());
            graphics_context.device.unmap_memory(vertex_memory);
        } else {
             error!("Failed to map vertex memory");
             return;
        }
    }

    let (index_buffer, index_memory) = match graphics_context.create_buffer(
        (indices.len() * std::mem::size_of::<u32>()) as u64,
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
        if let Ok(ptr) = graphics_context.device.map_memory(index_memory, 0, (indices.len() * std::mem::size_of::<u32>()) as u64, vk::MemoryMapFlags::empty()) {
            std::ptr::copy_nonoverlapping(indices.as_ptr(), ptr as *mut u32, indices.len());
            graphics_context.device.unmap_memory(index_memory);
        } else {
            error!("Failed to map index memory");
            return;
        }
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
            if elapsed < std::time::Duration::from_millis(16) {
                std::thread::sleep(std::time::Duration::from_millis(16) - elapsed);
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
                }
                PollEvent::Main(MainEvent::Pause { .. }) => {
                    info!("MainEvent::Pause received");
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

        if session_running {
            // Update Physics and World
            // Physics and World updates are now in a separate thread
            // physics_world.step();
            // game_world.update_streaming();

            // Mesh Upload Logic
            {
                let mut state = game_state.write().unwrap();
                let mut to_upload = Vec::new();
                
                // Find entities with Mesh but without GpuMesh
                // We collect them first to avoid borrowing issues while modifying
                for (id, mesh) in state.world.ecs.query::<&world::Mesh>().without::<&GpuMesh>().iter() {
                    if !mesh.vertices.is_empty() && !mesh.indices.is_empty() {
                        to_upload.push((id, mesh.clone()));
                    }
                }

                for (id, mesh) in to_upload {
                    // Create Vertex Buffer
                    let (vertex_buffer, vertex_memory) = match graphics_context.create_buffer(
                        (mesh.vertices.len() * std::mem::size_of::<world::Vertex>()) as u64,
                        vk::BufferUsageFlags::VERTEX_BUFFER,
                        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
                    ) {
                        Ok(ret) => ret,
                        Err(e) => {
                            error!("Failed to create vertex buffer for entity {:?}: {:?}", id, e);
                            continue;
                        }
                    };

                    unsafe {
                        if let Ok(ptr) = graphics_context.device.map_memory(vertex_memory, 0, (mesh.vertices.len() * std::mem::size_of::<world::Vertex>()) as u64, vk::MemoryMapFlags::empty()) {
                            std::ptr::copy_nonoverlapping(mesh.vertices.as_ptr(), ptr as *mut world::Vertex, mesh.vertices.len());
                            graphics_context.device.unmap_memory(vertex_memory);
                        } else {
                             error!("Failed to map vertex memory for entity {:?}", id);
                             continue;
                        }
                    }

                    // Create Index Buffer
                    let (index_buffer, index_memory) = match graphics_context.create_buffer(
                        (mesh.indices.len() * std::mem::size_of::<u32>()) as u64,
                        vk::BufferUsageFlags::INDEX_BUFFER,
                        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
                    ) {
                        Ok(ret) => ret,
                        Err(e) => {
                            error!("Failed to create index buffer for entity {:?}: {:?}", id, e);
                            continue;
                        }
                    };

                    unsafe {
                        if let Ok(ptr) = graphics_context.device.map_memory(index_memory, 0, (mesh.indices.len() * std::mem::size_of::<u32>()) as u64, vk::MemoryMapFlags::empty()) {
                            std::ptr::copy_nonoverlapping(mesh.indices.as_ptr(), ptr as *mut u32, mesh.indices.len());
                            graphics_context.device.unmap_memory(index_memory);
                        } else {
                            error!("Failed to map index memory for entity {:?}", id);
                            continue;
                        }
                    }

                    let gpu_mesh = GpuMesh {
                        vertex_buffer,
                        vertex_memory,
                        index_buffer,
                        index_memory,
                        index_count: mesh.indices.len() as u32,
                    };

                    if let Err(e) = state.world.ecs.insert_one(id, gpu_mesh) {
                        error!("Failed to insert GpuMesh for entity {:?}: {:?}", id, e);
                    } else {
                        info!("Uploaded mesh for entity {:?}", id);
                    }
                }
            }

            match xr_context.frame_waiter.wait() {
                Ok(frame_state) => {
                    if let Err(e) = xr_context.frame_stream.begin() {
                         error!("Failed to begin frame: {:?}", e);
                         continue;
                    }

                    if frame_state.should_render {
                        let stream_idx = match xr_context.swapchain.acquire_image() {
                            Ok(idx) => idx,
                            Err(e) => {
                                error!("Failed to acquire image: {:?}", e);
                                continue;
                            }
                        };
                        
                        if let Err(e) = xr_context.swapchain.wait_image(Duration::INFINITE) {
                             error!("Failed to wait image: {:?}", e);
                             continue;
                        }

                            // Locate views once for both rendering and projection
                            let (_flags, views) = xr_context.session.locate_views(
                                ViewConfigurationType::PRIMARY_STEREO,
                                frame_state.predicted_display_time,
                                &xr_context.stage_space
                            ).unwrap();

                        // Record Command Buffer
                        unsafe {
                            if let Err(e) = graphics_context.device.wait_for_fences(&[graphics_context.fence], true, u64::MAX) {
                                error!("Failed to wait for fences: {:?}", e);
                            }
                            let _ = graphics_context.device.reset_fences(&[graphics_context.fence]);
                            let _ = graphics_context.device.reset_command_buffer(graphics_context.command_buffer, vk::CommandBufferResetFlags::empty());

                            let begin_info = vk::CommandBufferBeginInfo::builder()
                                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                            
                            let _ = graphics_context.device.begin_command_buffer(graphics_context.command_buffer, &begin_info);

                            // Render Loop for Stereo (2 Eyes)
                            for eye_index in 0..2 {
                                let clear_values = [
                                    vk::ClearValue {
                                        color: vk::ClearColorValue {
                                            float32: [0.0, 0.0, 0.0, 1.0], // BLACK background
                                        },
                                    },
                                    vk::ClearValue {
                                        depth_stencil: vk::ClearDepthStencilValue {
                                            depth: 1.0,
                                            stencil: 0,
                                        },
                                    }
                                ];
                                
                                // info!("Rendering Eye: {}", eye_index);

                                let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                                    .render_pass(graphics_context.render_pass)
                                    .framebuffer(xr_context.framebuffers[stream_idx as usize][eye_index])
                                    .render_area(vk::Rect2D {
                                        offset: vk::Offset2D { x: 0, y: 0 },
                                        extent: xr_context.resolution,
                                    })
                                    .clear_values(&clear_values);

                                graphics_context.device.cmd_begin_render_pass(graphics_context.command_buffer, &render_pass_begin_info, vk::SubpassContents::INLINE);
                                
                                graphics_context.device.cmd_bind_pipeline(graphics_context.command_buffer, vk::PipelineBindPoint::GRAPHICS, graphics_context.pipeline);
                                
                                graphics_context.device.cmd_bind_vertex_buffers(graphics_context.command_buffer, 0, &[vertex_buffer], &[0]);
                                graphics_context.device.cmd_bind_index_buffer(graphics_context.command_buffer, index_buffer, 0, vk::IndexType::UINT32);

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

                                // Update Push Constants
                                let view = &views[eye_index];
                                
                                let view_matrix = create_view_matrix(&view.pose);
                                let proj_matrix = create_projection_matrix(view.fov, 0.01, 100.0);
                                
                                // Render ECS Entities
                                // We iterate over all entities with Transform and Mesh
                                // For now, we use the hardcoded vertex buffer (cube) for ALL meshes
                                // But we use the Transform from the ECS
                                
                                // Debug cube removed to avoid Z-fighting/occlusion

                                // Render ECS objects
                                if let Ok(state) = game_state.read() {
                                    for (_id, (transform, gpu_mesh)) in state.world.ecs.query::<(&world::Transform, &GpuMesh)>().iter() {
                                        let model_matrix = glam::Mat4::from_scale_rotation_translation(
                                            transform.scale,
                                            transform.rotation,
                                            transform.position
                                        );

                                        let mut push_constants = Vec::new();
                                        push_constants.extend_from_slice(bytemuck::bytes_of(&model_matrix));
                                        push_constants.extend_from_slice(bytemuck::bytes_of(&view_matrix));
                                        push_constants.extend_from_slice(bytemuck::bytes_of(&proj_matrix));

                                        graphics_context.device.cmd_push_constants(
                                            graphics_context.command_buffer,
                                            graphics_context.pipeline_layout,
                                            vk::ShaderStageFlags::VERTEX,
                                            0,
                                            &push_constants,
                                        );
                                        
                                        // Bind entity buffers
                                        graphics_context.device.cmd_bind_vertex_buffers(graphics_context.command_buffer, 0, &[gpu_mesh.vertex_buffer], &[0]);
                                        graphics_context.device.cmd_bind_index_buffer(graphics_context.command_buffer, gpu_mesh.index_buffer, 0, vk::IndexType::UINT32);
                                        
                                        // debug!("Drawing entity {:?}", _id); // Uncomment for verbose logging
                                        graphics_context.device.cmd_draw_indexed(graphics_context.command_buffer, gpu_mesh.index_count, 1, 0, 0, 0);
                                    }
                                }

                                graphics_context.device.cmd_end_render_pass(graphics_context.command_buffer);
                            }
                            
                            let _ = graphics_context.device.end_command_buffer(graphics_context.command_buffer);

                            let command_buffers = [graphics_context.command_buffer];
                            let submit_info = vk::SubmitInfo::builder()
                                .command_buffers(&command_buffers);
                            
                            if let Err(e) = graphics_context.device.queue_submit(graphics_context.queue, &[submit_info.build()], graphics_context.fence) {
                                error!("Failed to submit queue: {:?}", e);
                            }
                        }

                        if let Err(e) = xr_context.swapchain.release_image() {
                            error!("Failed to release image: {:?}", e);
                        }

                        // Views are already located above
                        let projection_views: Vec<CompositionLayerProjectionView<oxr::Vulkan>> = views.into_iter().enumerate().map(|(i, view)| {
                            // Normalize orientation to avoid ERROR_POSE_INVALID
                            let q = view.pose.orientation;
                            let len = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w).sqrt();
                            let normalized_pose = if len > 0.0001 {
                                openxr::Posef {
                                    orientation: openxr::Quaternionf {
                                        x: q.x / len,
                                        y: q.y / len,
                                        z: q.z / len,
                                        w: q.w / len,
                                    },
                                    position: view.pose.position,
                                }
                            } else {
                                // Fallback to identity orientation if invalid
                                openxr::Posef {
                                    orientation: openxr::Quaternionf { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
                                    position: view.pose.position,
                                }
                            };

                            CompositionLayerProjectionView::new()
                                .pose(normalized_pose)
                                .fov(view.fov)
                                .sub_image(
                                    SwapchainSubImage::new()
                                        .swapchain(&xr_context.swapchain)
                                        .image_array_index(i as u32)
                                        .image_rect(Rect2Di {
                                            offset: Offset2Di { x: 0, y: 0 },
                                            extent: Extent2Di {
                                                width: xr_context.resolution.width as i32,
                                                height: xr_context.resolution.height as i32,
                                            },
                                        })
                                )
                        }).collect();
                        
                        let projection = CompositionLayerProjection::new()
                            .space(&xr_context.stage_space)
                            .views(&projection_views);

                        if let Err(e) = xr_context.frame_stream.end(
                            frame_state.predicted_display_time,
                            EnvironmentBlendMode::OPAQUE,
                            &[&projection],
                        ) {
                            log::error!("frame_stream.end failed: {}", e);
                        }

                    } else {
                        if let Err(e) = xr_context.frame_stream.end(
                            frame_state.predicted_display_time,
                            EnvironmentBlendMode::OPAQUE,
                            &[],
                        ) {
                            log::error!("frame_stream.end failed (no layers): {}", e);
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to wait frame: {:?}", e);
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
