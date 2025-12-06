#[cfg(target_os = "android")]
use android_activity::{AndroidApp, MainEvent, PollEvent};
use log::{info, error};
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

    // Create Mesh Buffers
    let vertices = [
        world::Vertex { position: [0.0, -0.5, -1.0], color: [1.0, 0.0, 0.0] },
        world::Vertex { position: [0.5, 0.5, -1.0], color: [0.0, 1.0, 0.0] },
        world::Vertex { position: [-0.5, 0.5, -1.0], color: [0.0, 0.0, 1.0] },
    ];
    let indices = [0u32, 1, 2];

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
    let mut physics_world = PhysicsWorld::new();
    let mut game_world = GameWorld::new();

    let cmd_tx = game_world.command_sender.clone();
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
                if let Ok((mut socket, _)) = listener.accept().await {
                    let tx = cmd_tx.clone();
                    tokio::spawn(async move {
                        use tokio::io::AsyncReadExt;
                        loop {
                            let mut len_buf = [0u8; 4];
                            if socket.read_exact(&mut len_buf).await.is_err() {
                                break;
                            }
                            let len = u32::from_le_bytes(len_buf) as usize;
                            let mut data = vec![0u8; len];
                            if socket.read_exact(&mut data).await.is_err() {
                                break;
                            }
                            
                            if let Ok(update) = bincode::deserialize::<world::SceneUpdate>(&data) {
                                info!("Received update: {:?}", update);
                                let _ = tx.send(update).await;
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
        app.poll_events(Some(std::time::Duration::from_millis(0)), |event| {
            match event {
                PollEvent::Main(MainEvent::Destroy) => {
                    quit = true;
                }
                PollEvent::Main(MainEvent::InitWindow { .. }) => {
                    info!("Window Initialized");
                }
                _ => {}
            }
        });

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
            physics_world.step();
            game_world.update_streaming();

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

                            let clear_values = [
                                vk::ClearValue {
                                    color: vk::ClearColorValue {
                                        float32: [0.1, 0.2, 0.3, 1.0], 
                                    },
                                }
                            ];

                            let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                                .render_pass(graphics_context.render_pass)
                                .framebuffer(xr_context.framebuffers[stream_idx as usize])
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

                            // TODO: Update Push Constants for each view (stereo)
                            // For now, just draw once, it will look weird in VR but proves rendering works
                            
                            graphics_context.device.cmd_draw_indexed(graphics_context.command_buffer, indices.len() as u32, 1, 0, 0, 0);

                            graphics_context.device.cmd_end_render_pass(graphics_context.command_buffer);
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

                        let (_flags, views) = xr_context.session.locate_views(
                            ViewConfigurationType::PRIMARY_STEREO,
                            frame_state.predicted_display_time,
                            &xr_context.session.create_reference_space(
                                ReferenceSpaceType::STAGE, 
                                Posef::IDENTITY
                            ).unwrap()
                        ).unwrap();

                        let projection_views: Vec<CompositionLayerProjectionView<oxr::Vulkan>> = views.into_iter().enumerate().map(|(i, view)| {
                            // Normalize orientation to avoid ERROR_POSE_INVALID
                            let q = view.pose.orientation;
                            let len = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w).sqrt();
                            let normalized_pose = if len > 0.0 {
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
                                view.pose
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
                        
                        let ref_space = xr_context.session.create_reference_space(
                            ReferenceSpaceType::STAGE, 
                            Posef::IDENTITY
                        ).unwrap();

                        let projection = CompositionLayerProjection::new()
                            .space(&ref_space)
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
