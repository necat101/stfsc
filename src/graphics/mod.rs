use ash::{vk, Entry, Instance, Device};
use ash::vk::Handle; // Import Handle trait
use anyhow::{Result, Context};
use std::ffi::{CString};
use openxr as xr;

pub struct GraphicsContext {
    pub entry: Entry,
    pub instance: Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: Device,
    pub queue_family_index: u32,
    pub queue: vk::Queue,
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub fence: vk::Fence,
    pub render_pass: vk::RenderPass,
}

impl GraphicsContext {
    pub fn new(xr_instance: &xr::Instance, xr_system: xr::SystemId) -> Result<Self> {
        let entry = unsafe { Entry::load()? };

        // 1. Get OpenXR Vulkan requirements
        let _xr_vulkan_reqs = xr_instance.graphics_requirements::<xr::Vulkan>(xr_system)?;
        
        // 2. Create Vulkan Instance
        let app_name = CString::new("STFSC Engine")?;
        let engine_name = CString::new("STFSC")?;
        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(0)
            .engine_name(&engine_name)
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 1, 0)); 

        // Use vulkan_legacy_instance_extensions as suggested by compiler
        let xr_extensions = xr_instance.vulkan_legacy_instance_extensions(xr_system)?;
        let xr_extensions_cstr: Vec<CString> = xr_extensions.split(' ')
            .map(|s| CString::new(s).unwrap())
            .collect();
        let xr_extension_ptrs: Vec<*const u8> = xr_extensions_cstr.iter()
            .map(|s| s.as_ptr() as *const u8)
            .collect();
        
        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&xr_extension_ptrs);

        let instance = unsafe { entry.create_instance(&create_info, None)? };

        // 3. Pick Physical Device
        // vulkan_graphics_device returns *const c_void (pointer)
        let physical_device_ptr = unsafe {
            xr_instance.vulkan_graphics_device(
                xr_system, 
                instance.handle().as_raw() as _
            )?
        };
        // vk::PhysicalDevice::from_raw expects u64.
        let physical_device = vk::PhysicalDevice::from_raw(physical_device_ptr as usize as u64);
        
        // 4. Create Logical Device
        let queue_family_index = unsafe {
            instance.get_physical_device_queue_family_properties(physical_device)
                .into_iter()
                .enumerate()
                .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .map(|(i, _)| i as u32)
                .context("No graphics queue found")?
        };

        let queue_priorities = [1.0];
        let queue_create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);

        // Use vulkan_legacy_device_extensions
        let xr_device_extensions = xr_instance.vulkan_legacy_device_extensions(xr_system)?;
        let xr_device_extensions_cstr: Vec<CString> = xr_device_extensions.split(' ')
            .map(|s| CString::new(s).unwrap())
            .collect();
        let xr_device_extension_ptrs: Vec<*const u8> = xr_device_extensions_cstr.iter()
            .map(|s| s.as_ptr() as *const u8)
            .collect();

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
            .enabled_extension_names(&xr_device_extension_ptrs);

        let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        // 5. Create Command Pool
        let pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_create_info, None)? };

        // 6. Allocate Command Buffer
        let buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&buffer_allocate_info)?[0] };

        // 7. Create Fence
        let fence_create_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED);
        let fence = unsafe { device.create_fence(&fence_create_info, None)? };

        // 8. Create Render Pass
        let render_pass_attachments = [
            vk::AttachmentDescription::builder()
                .format(vk::Format::R8G8B8A8_SRGB) // Must match swapchain format
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .build()
        ];
        let color_attachment_refs = [
            vk::AttachmentReference::builder()
                .attachment(0)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .build()
        ];
        let subpasses = [
            vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&color_attachment_refs)
                .build()
        ];
        let render_pass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&render_pass_attachments)
            .subpasses(&subpasses);
        let render_pass = unsafe { device.create_render_pass(&render_pass_create_info, None)? };

        Ok(Self {
            entry,
            instance,
            physical_device,
            device,
            queue_family_index,
            queue,
            command_pool,
            command_buffer,
            fence,
            render_pass,
        })
    }
}
