use ash::{vk, Entry, Instance, Device};
use ash::vk::Handle; // Import Handle trait
use anyhow::{Result, Context};
use std::ffi::{CString};
use openxr as xr;
use glam;
use std::ptr;

pub struct Texture {
    pub image: vk::Image,
    pub image_memory: vk::DeviceMemory,
    pub image_view: vk::ImageView,
    pub sampler: vk::Sampler,
}

pub struct SkyboxRenderer {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
}

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
    pub depth_format: vk::Format,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
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

        // Find Depth Format
        let depth_format = Self::find_depth_format(&instance, physical_device)?;

        // 8. Create Render Pass
        let render_pass_attachments = [
            vk::AttachmentDescription::builder()
                .format(vk::Format::R8G8B8A8_UNORM) // Color
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .build(),
            vk::AttachmentDescription::builder()
                .format(depth_format) // Depth
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .build()
        ];
        let color_attachment_refs = [
            vk::AttachmentReference::builder()
                .attachment(0)
                .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .build()
        ];
        let depth_attachment_ref = vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpasses = [
            vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&color_attachment_refs)
                .depth_stencil_attachment(&depth_attachment_ref)
                .build()
        ];
        let render_pass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&render_pass_attachments)
            .subpasses(&subpasses);
        let render_pass = unsafe { device.create_render_pass(&render_pass_create_info, None)? };

        // 8.5 Create Descriptor Set Layout
        let bindings = [
            // Binding 0: Uniform Buffer (if we used one, but we use push constants for now, so maybe just sampler?)
            // Actually, for this phase, let's just use Push Constants for matrices and Descriptor Set for Texture.
            // So Binding 1 (in shader) will be the texture. Let's make it Binding 0 in the set.
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build(),
        ];
        
        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings);
            
        let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };

        // 8.6 Create Descriptor Pool
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 100, // Enough for 100 textures
            }
        ];
        
        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(100);
            
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

        // 9. Create Pipeline Layout
        let push_constant_ranges = [
            vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .offset(0)
                .size(std::mem::size_of::<glam::Mat4>() as u32 * 3)
                .build()
        ];
        let set_layouts = [descriptor_set_layout];
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(&push_constant_ranges)
            .set_layouts(&set_layouts);
        let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None)? };

        // 10. Create Shader Modules
        let vert_code = include_bytes!(concat!(env!("OUT_DIR"), "/vert.vert.spv"));
        let frag_code = include_bytes!(concat!(env!("OUT_DIR"), "/frag.frag.spv"));

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
                offset: 12, // normal
            },
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: 24, // uv
            },
            vk::VertexInputAttributeDescription {
                location: 3,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 32, // color
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
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
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

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

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
            .depth_stencil_state(&depth_stencil_state)
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
            depth_format,
            descriptor_set_layout,
            descriptor_pool,
        })
    }

    fn find_depth_format(instance: &Instance, physical_device: vk::PhysicalDevice) -> Result<vk::Format> {
        let candidates = [
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];

        for &format in candidates.iter() {
            let props = unsafe { instance.get_physical_device_format_properties(physical_device, format) };
            if props.optimal_tiling_features.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT) {
                return Ok(format);
            }
        }
        anyhow::bail!("Failed to find supported depth format")
    }

    pub fn create_depth_resources(&self, extent: vk::Extent2D, array_layers: u32) -> Result<(vk::Image, vk::DeviceMemory, Vec<vk::ImageView>)> {
        let format = self.depth_format;
        
        let image_create_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(array_layers)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe { self.device.create_image(&image_create_info, None)? };

        let mem_reqs = unsafe { self.device.get_image_memory_requirements(image) };
        let memory_type_index = self.find_memory_type(mem_reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_reqs.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { self.device.allocate_memory(&alloc_info, None)? };

        unsafe { self.device.bind_image_memory(image, memory, 0)? };

        let mut views = Vec::new();
        for i in 0..array_layers {
            let view_create_info = vk::ImageViewCreateInfo::builder()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: i,
                    layer_count: 1,
                });

            let view = unsafe { self.device.create_image_view(&view_create_info, None)? };
            views.push(view);
        }

        Ok((image, memory, views))
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

    pub fn begin_single_time_commands(&self) -> Result<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(self.command_pool)
            .command_buffer_count(1);
        
        let command_buffer = unsafe { self.device.allocate_command_buffers(&alloc_info)?[0] };
        
        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            
        unsafe { self.device.begin_command_buffer(command_buffer, &begin_info)? };
        
        Ok(command_buffer)
    }

    pub fn end_single_time_commands(&self, command_buffer: vk::CommandBuffer) -> Result<()> {
        unsafe { self.device.end_command_buffer(command_buffer)? };
        
        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::builder()
            .command_buffers(&command_buffers);
            
        unsafe {
            self.device.queue_submit(self.queue, &[submit_info.build()], vk::Fence::null())?;
            self.device.queue_wait_idle(self.queue)?;
            self.device.free_command_buffers(self.command_pool, &command_buffers);
        }
        
        Ok(())
    }

    pub fn transition_image_layout(&self, image: vk::Image, _format: vk::Format, old_layout: vk::ImageLayout, new_layout: vk::ImageLayout) -> Result<()> {
        let command_buffer = self.begin_single_time_commands()?;
        
        let (src_access_mask, dst_access_mask, src_stage, dst_stage) = match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            _ => anyhow::bail!("Unsupported layout transition!"),
        };

        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            })
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask);
            
        unsafe {
            self.device.cmd_pipeline_barrier(
                command_buffer,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier.build()],
            );
        }

        self.end_single_time_commands(command_buffer)?;
        Ok(())
    }

    pub fn copy_buffer_to_image(&self, buffer: vk::Buffer, image: vk::Image, width: u32, height: u32) -> Result<()> {
        let command_buffer = self.begin_single_time_commands()?;
        
        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });
            
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                command_buffer,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region.build()],
            );
        }

        self.end_single_time_commands(command_buffer)?;
        Ok(())
    }

    pub fn create_default_texture(&self) -> Result<Texture> {
        let width = 256;
        let height = 256;
        let mut img = image::RgbaImage::new(width, height);
        
        for x in 0..width {
            for y in 0..height {
                let color = if (x / 32 + y / 32) % 2 == 0 {
                    image::Rgba([255, 255, 255, 255])
                } else {
                    image::Rgba([0, 0, 0, 255])
                };
                img.put_pixel(x, y, color);
            }
        }
        
        // We can reuse the logic from create_texture_from_memory but we need to refactor or duplicate.
        // Let's just duplicate the upload logic for now to avoid breaking the previous function signature 
        // or just call this "create_texture_from_image" and make the other one call it.
        
        self.create_texture_from_image(&img)
    }

    pub fn create_texture_from_image(&self, img: &image::RgbaImage) -> Result<Texture> {
        let (width, height) = img.dimensions();
        let size = (width * height * 4) as u64;
        
        // Staging Buffer
        let (staging_buffer, staging_memory) = self.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        
        unsafe {
            let ptr = self.device.map_memory(staging_memory, 0, size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(img.as_ptr(), ptr as *mut u8, size as usize);
            self.device.unmap_memory(staging_memory);
        }
        
        // Create Image
        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::R8G8B8A8_UNORM)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);
            
        let image = unsafe { self.device.create_image(&image_info, None)? };
        
        let mem_reqs = unsafe { self.device.get_image_memory_requirements(image) };
        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_reqs.size)
            .memory_type_index(self.find_memory_type(mem_reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?);
            
        let image_memory = unsafe { self.device.allocate_memory(&alloc_info, None)? };
        unsafe { self.device.bind_image_memory(image, image_memory, 0)? };
        
        // Upload
        self.transition_image_layout(image, vk::Format::R8G8B8A8_UNORM, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL)?;
        self.copy_buffer_to_image(staging_buffer, image, width, height)?;
        self.transition_image_layout(image, vk::Format::R8G8B8A8_UNORM, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;
        
        // Cleanup Staging
        unsafe {
            self.device.destroy_buffer(staging_buffer, None);
            self.device.free_memory(staging_memory, None);
        }
        
        // Create View
        let view_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
            
        let image_view = unsafe { self.device.create_image_view(&view_info, None)? };
        
        // Create Sampler
        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(false)
            .max_anisotropy(16.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0);
            
        let sampler = unsafe { self.device.create_sampler(&sampler_info, None)? };
        
        Ok(Texture {
            image,
            image_memory,
            image_view,
            sampler,
        })
    }

    pub fn create_descriptor_set(&self, texture: &Texture) -> Result<vk::DescriptorSet> {
        let layouts = [self.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layouts);
            
        let descriptor_set = unsafe { self.device.allocate_descriptor_sets(&alloc_info)?[0] };
        
        let image_info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(texture.image_view)
            .sampler(texture.sampler);
            
        let descriptor_writes = [
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&image_info))
                .build()
        ];
        
        unsafe { self.device.update_descriptor_sets(&descriptor_writes, &[]) };
        
        Ok(descriptor_set)
    }

    pub fn create_skybox_pipeline(&self, render_pass: vk::RenderPass) -> Result<SkyboxRenderer> {
        let vert_code = include_bytes!(concat!(env!("OUT_DIR"), "/skybox.vert.spv"));
        let frag_code = include_bytes!(concat!(env!("OUT_DIR"), "/skybox.frag.spv"));
        
        let vert_module = unsafe {
            let code = ash::util::read_spv(&mut std::io::Cursor::new(&vert_code[..]))?;
            let create_info = vk::ShaderModuleCreateInfo::builder().code(&code);
            self.device.create_shader_module(&create_info, None)?
        };
        let frag_module = unsafe {
            let code = ash::util::read_spv(&mut std::io::Cursor::new(&frag_code[..]))?;
            let create_info = vk::ShaderModuleCreateInfo::builder().code(&code);
            self.device.create_shader_module(&create_info, None)?
        };
        
        let main_function_name = std::ffi::CString::new("main").unwrap();
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
        
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&[]) // No vertex input, generated in shader or hardcoded cube
            .vertex_attribute_descriptions(&[]); // Actually we will use the same cube mesh for simplicity, so let's match the input
            
        // Wait, for simplicity let's just use the same vertex format as the main pipeline so we can reuse the cube mesh
        let binding_descriptions = [vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<crate::world::Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()];
            
        let attribute_descriptions = [
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT) // Position
                .offset(0)
                .build(),
        ];
        
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);
            
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
            .cull_mode(vk::CullModeFlags::NONE) // Don't cull inside
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);
            
        let multisampling = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(false) // Don't write depth
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);
            
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B | vk::ColorComponentFlags::A)
            .blend_enable(false);
            
        let color_blending = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(std::slice::from_ref(&color_blend_attachment));
            
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&dynamic_states);
            
        let push_constant_ranges = [vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(192) // 3 * mat4
            .build()];
            
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(&push_constant_ranges);
            
        let pipeline_layout = unsafe { self.device.create_pipeline_layout(&pipeline_layout_info, None)? };
        
        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);
            
        let pipeline = unsafe { self.device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None).map_err(|e| e.1)?[0] };
        
        unsafe {
            self.device.destroy_shader_module(vert_module, None);
            self.device.destroy_shader_module(frag_module, None);
        }
        
        Ok(SkyboxRenderer {
            pipeline,
            pipeline_layout,
        })
    }
}
