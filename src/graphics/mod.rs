use ash::{vk, Entry, Instance, Device};
use ash::vk::Handle; // Import Handle trait
use anyhow::{Result, Context};
use std::ffi::{CString};
use openxr as xr;
use glam;

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
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
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

        // 9. Create Pipeline Layout
        let push_constant_ranges = [
            vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .offset(0)
                .size(std::mem::size_of::<glam::Mat4>() as u32 * 3)
                .build()
        ];
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(&push_constant_ranges);
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None)? };

        // 10. Create Shader Modules
        let vert_code = include_bytes!(concat!(env!("OUT_DIR"), "/vert.spv"));
        let frag_code = include_bytes!(concat!(env!("OUT_DIR"), "/frag.spv"));

        let vert_module = unsafe {
            let code = ash::util::read_spv(&mut std::io::Cursor::new(&vert_code[..]))?;
            let create_info = vk::ShaderModuleCreateInfo::builder().code(&code);
            device.create_shader_module(&create_info, None)?
        };

        let frag_module = unsafe {
            let code = ash::util::read_spv(&mut std::io::Cursor::new(&frag_code[..]))?;
            let create_info = vk::ShaderModuleCreateInfo::builder().code(&code);
            device.create_shader_module(&create_info, None)?
        };

        let main_function_name = CString::new("main")?;

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(&main_function_name)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(&main_function_name)
                .build(),
        ];

        // Vertex Input
        let vertex_binding_descriptions = [
            vk::VertexInputBindingDescription {
                binding: 0,
                stride: std::mem::size_of::<crate::world::Vertex>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            }
        ];
        let vertex_attribute_descriptions = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0, // position
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 12, // color
            }
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&vertex_binding_descriptions)
            .vertex_attribute_descriptions(&vertex_attribute_descriptions);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B | vk::ColorComponentFlags::A)
            .blend_enable(false)
            .build();

        let color_blending = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR
        ];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&dynamic_states);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipeline = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None).map_err(|e| e.1).unwrap()[0] };

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }

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
            pipeline_layout,
            pipeline,
        })
    }

    pub fn create_buffer(&self, size: u64, usage: vk::BufferUsageFlags, properties: vk::MemoryPropertyFlags) -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
            
        let buffer = unsafe { self.device.create_buffer(&buffer_info, None)? };
        
        let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let memory_type_index = self.find_memory_type(mem_reqs.memory_type_bits, properties)?;
        
        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_reqs.size)
            .memory_type_index(memory_type_index);
            
        let memory = unsafe { self.device.allocate_memory(&alloc_info, None)? };
        
        unsafe { self.device.bind_buffer_memory(buffer, memory, 0)? };
        
        Ok((buffer, memory))
    }
    
    pub fn find_memory_type(&self, type_filter: u32, properties: vk::MemoryPropertyFlags) -> Result<u32> {
        let mem_properties = unsafe { self.instance.get_physical_device_memory_properties(self.physical_device) };
        for i in 0..mem_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0 && (mem_properties.memory_types[i as usize].property_flags & properties) == properties {
                return Ok(i);
            }
        }
        anyhow::bail!("Failed to find suitable memory type")
    }
}
