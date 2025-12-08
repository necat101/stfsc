use openxr as xr;
use anyhow::{Result, Context};
use ash::vk;
use ash::vk::Handle; // Import Handle trait
use crate::graphics::GraphicsContext;

pub struct XrContext {
    pub instance: xr::Instance,
    pub system: xr::SystemId,
    pub session: xr::Session<xr::Vulkan>,
    pub frame_stream: xr::FrameStream<xr::Vulkan>,
    pub frame_waiter: xr::FrameWaiter,
    pub swapchain: xr::Swapchain<xr::Vulkan>,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<Vec<vk::ImageView>>, // [swapchain_image_index][eye_index]
    pub framebuffers: Vec<Vec<vk::Framebuffer>>,       // [swapchain_image_index][eye_index]
    pub resolution: vk::Extent2D,
    pub depth_image: vk::Image,
    pub depth_image_memory: vk::DeviceMemory,
    pub depth_image_views: Vec<vk::ImageView>, // [eye_index]
    pub stage_space: xr::Space,
}

impl XrContext {
    pub fn new() -> Result<(xr::Instance, xr::SystemId)> {
        let entry = xr::Entry::linked();
        
        #[cfg(target_os = "android")]
        {
            entry.initialize_android_loader()?;
        }

        let available_extensions = entry.enumerate_extensions()?;
        
        let mut extensions = xr::ExtensionSet::default();
        
        #[cfg(target_os = "android")]
        {
            extensions.khr_android_create_instance = true;
        }

        if available_extensions.khr_vulkan_enable {
            extensions.khr_vulkan_enable = true;
        } else if available_extensions.khr_vulkan_enable2 {
            extensions.khr_vulkan_enable2 = true;
        } else {
            anyhow::bail!("No Vulkan extension supported (checked v1 and v2)");
        }

        let instance = entry.create_instance(
            &xr::ApplicationInfo {
                application_name: "STFSC Engine",
                application_version: 1,
                engine_name: "STFSC",
                engine_version: 1,
            },
            &extensions,
            &[], 
        ).context("Failed to create OpenXR instance")?;
        let system = instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)?;
        Ok((instance, system))
    }

    pub fn create_session(
        instance: &xr::Instance,
        system: xr::SystemId,
        graphics: &GraphicsContext,
    ) -> Result<Self> {
        let (session, frame_waiter, frame_stream) = unsafe {
            instance.create_session::<xr::Vulkan>(
                system,
                &xr::vulkan::SessionCreateInfo {
                    instance: graphics.instance.handle().as_raw() as _,
                    physical_device: graphics.physical_device.as_raw() as _,
                    device: graphics.device.handle().as_raw() as _,
                    queue_family_index: graphics.queue_family_index,
                    queue_index: 0,
                },
            )?
        };

        let view_configuration_views = instance.enumerate_view_configuration_views(
            system,
            xr::ViewConfigurationType::PRIMARY_STEREO,
        )?;
        let resolution = vk::Extent2D {
            width: view_configuration_views[0].recommended_image_rect_width,
            height: view_configuration_views[0].recommended_image_rect_height,
        };

        let swapchain_formats = session.enumerate_swapchain_formats()?;
        let format = swapchain_formats
            .iter()
            .cloned()
            .find(|&f| f == vk::Format::R8G8B8A8_SRGB.as_raw() as u32)
            .context("R8G8B8A8_SRGB format not supported by OpenXR runtime")?;

        let swapchain_create_info = xr::SwapchainCreateInfo {
            create_flags: xr::SwapchainCreateFlags::EMPTY,
            usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT,
            format: format,
            sample_count: 1,
            width: resolution.width,
            height: resolution.height,
            face_count: 1,
            array_size: 2, // Stereo
            mip_count: 1,
        };

        let swapchain = session.create_swapchain(&swapchain_create_info)?;
        let images = swapchain.enumerate_images()?;
        
        // images are likely openxr handles (u64)
        let swapchain_images: Vec<vk::Image> = images
            .into_iter()
            .map(|i| vk::Image::from_raw(i))
            .collect(); 

        // Create Depth Resources (Array Layers = 2)
        let (depth_image, depth_image_memory, depth_image_views) = graphics.create_depth_resources(resolution, 2)?;

        // Create Image Views and Framebuffers
        let mut swapchain_image_views = Vec::new();
        let mut framebuffers = Vec::new();

        for image in &swapchain_images {
            let mut image_views_per_eye = Vec::new();
            let mut framebuffers_per_eye = Vec::new();

            for eye_index in 0..2 {
                let create_view_info = vk::ImageViewCreateInfo::builder()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::R8G8B8A8_SRGB) // Must match swapchain format
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: eye_index, // Select layer for this eye
                        layer_count: 1,
                    })
                    .image(*image);
                
                let image_view = unsafe { graphics.device.create_image_view(&create_view_info, None)? };
                image_views_per_eye.push(image_view);

                let framebuffer_attachments = [image_view, depth_image_views[eye_index as usize]];
                let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(graphics.render_pass)
                    .attachments(&framebuffer_attachments)
                    .width(resolution.width)
                    .height(resolution.height)
                    .layers(1); 
                
                let framebuffer = unsafe { graphics.device.create_framebuffer(&framebuffer_create_info, None)? };
                framebuffers_per_eye.push(framebuffer);
            }
            swapchain_image_views.push(image_views_per_eye);
            framebuffers.push(framebuffers_per_eye);
        }

        let stage_space = session.create_reference_space(
            xr::ReferenceSpaceType::LOCAL,
            xr::Posef::IDENTITY,
        )?;

        Ok(Self {
            instance: instance.clone(),
            system,
            session,
            frame_stream,
            frame_waiter,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            framebuffers,
            resolution,
            depth_image,
            depth_image_memory,
            depth_image_views,
            stage_space,
        })
    }
}
