use android_activity::{AndroidApp, MainEvent, PollEvent};
use log::{info, error};
use openxr as oxr;
use ash::vk;

mod graphics;
mod physics;
mod world;
mod xr;

use graphics::GraphicsContext;
use xr::XrContext;
use physics::PhysicsWorld;
use world::GameWorld;

// Import commonly used OpenXR types
use oxr::{
    EventDataBuffer, SessionState, ViewConfigurationType, 
    ReferenceSpaceType, Posef, CompositionLayerProjectionView,
    SwapchainSubImage, Rect2Di, Offset2Di, Extent2Di,
    CompositionLayerProjection, EnvironmentBlendMode, Duration
};

#[no_mangle]
fn android_main(app: AndroidApp) {
    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Info),
    );

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
        while let Some(event) = xr_context.instance.poll_event(&mut event_storage).unwrap() {
            match event {
                oxr::Event::SessionStateChanged(e) => {
                    info!("Session state changed to {:?}", e.state());
                    match e.state() {
                        SessionState::READY => {
                            if !session_running {
                                xr_context.session.begin(ViewConfigurationType::PRIMARY_STEREO).unwrap();
                                session_running = true;
                            }
                        }
                        SessionState::STOPPING => {
                            if session_running {
                                xr_context.session.end().unwrap();
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

            let frame_state = xr_context.frame_waiter.wait().unwrap();
            xr_context.frame_stream.begin().unwrap();

            if frame_state.should_render {
                let stream_idx = xr_context.swapchain.acquire_image().unwrap();
                
                xr_context.swapchain.wait_image(Duration::INFINITE).unwrap();

                // Record Command Buffer
                unsafe {
                    graphics_context.device.wait_for_fences(&[graphics_context.fence], true, u64::MAX).unwrap();
                    graphics_context.device.reset_fences(&[graphics_context.fence]).unwrap();
                    graphics_context.device.reset_command_buffer(graphics_context.command_buffer, vk::CommandBufferResetFlags::empty()).unwrap();

                    let begin_info = vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
                    
                    graphics_context.device.begin_command_buffer(graphics_context.command_buffer, &begin_info).unwrap();

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
                    
                    // TODO: Draw entities from ECS
                    // for (id, (transform, mesh)) in game_world.ecs.query::<(&Transform, &Mesh)>().iter() { ... }

                    graphics_context.device.cmd_end_render_pass(graphics_context.command_buffer);
                    graphics_context.device.end_command_buffer(graphics_context.command_buffer).unwrap();

                    let command_buffers = [graphics_context.command_buffer];
                    let submit_info = vk::SubmitInfo::builder()
                        .command_buffers(&command_buffers);
                    
                    graphics_context.device.queue_submit(graphics_context.queue, &[submit_info.build()], graphics_context.fence).unwrap();
                }

                xr_context.swapchain.release_image().unwrap();

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
    }
}
