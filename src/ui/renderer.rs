//! UI Renderer - Vulkan pipeline for 2D UI rendering
//!
//! Renders UI elements as an overlay after the main 3D scene.

use crate::ui::{UiCanvas, UiVertex};
use crate::graphics::GraphicsContext;
use anyhow::{Context, Result};
use ash::vk;
use std::sync::Arc;

/// Maximum number of UI vertices per frame
const MAX_UI_VERTICES: usize = 16384;
/// Maximum number of UI indices per frame
const MAX_UI_INDICES: usize = 32768;

/// UI Renderer handles Vulkan resources for UI overlay rendering
pub struct UiRenderer {
    /// Reference to graphics context
    #[allow(dead_code)]
    graphics: Arc<GraphicsContext>,
    
    /// UI pipeline
    pub pipeline: vk::Pipeline,
    /// Pipeline layout
    pub pipeline_layout: vk::PipelineLayout,
    
    /// Vertex buffer for UI quads
    pub vertex_buffer: vk::Buffer,
    pub vertex_memory: vk::DeviceMemory,
    
    /// Index buffer for UI quads
    pub index_buffer: vk::Buffer,
    pub index_memory: vk::DeviceMemory,
    
    /// Mapped vertex buffer pointer
    vertex_ptr: *mut UiVertex,
    /// Mapped index buffer pointer
    index_ptr: *mut u32,
    
    /// White 1x1 texture for solid color rendering
    pub white_texture: vk::Image,
    pub white_texture_memory: vk::DeviceMemory,
    pub white_texture_view: vk::ImageView,
    pub white_sampler: vk::Sampler,
    
    /// Descriptor set layout for UI texture
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    /// Descriptor set for white texture
    pub descriptor_set: vk::DescriptorSet,
    /// Descriptor set for font atlas (optional, created when font is set)
    pub font_atlas_descriptor_set: Option<vk::DescriptorSet>,
}

/// Push constants for UI shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiPushConstants {
    pub screen_size: [f32; 2],
}

impl UiRenderer {
    /// Create a new UI renderer
    pub fn new(graphics: Arc<GraphicsContext>) -> Result<Self> {
        let device = &graphics.device;
        
        // Create vertex buffer
        let vertex_buffer_size = (MAX_UI_VERTICES * std::mem::size_of::<UiVertex>()) as u64;
        let (vertex_buffer, vertex_memory) = graphics.create_buffer(
            vertex_buffer_size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ).context("Failed to create UI vertex buffer")?;
        
        let vertex_ptr = unsafe {
            device.map_memory(vertex_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .context("Failed to map UI vertex buffer")? as *mut UiVertex
        };

        // Create index buffer
        let index_buffer_size = (MAX_UI_INDICES * std::mem::size_of::<u32>()) as u64;
        let (index_buffer, index_memory) = graphics.create_buffer(
            index_buffer_size,
            vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ).context("Failed to create UI index buffer")?;
        
        let index_ptr = unsafe {
            device.map_memory(index_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .context("Failed to map UI index buffer")? as *mut u32
        };

        // Create 1x1 white texture for solid color rendering
        let (white_texture, white_texture_memory, white_texture_view) = 
            Self::create_white_texture(&graphics)?;
        
        // Create sampler
        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .max_anisotropy(1.0)
            .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .min_lod(0.0)
            .max_lod(0.0)
            .mip_lod_bias(0.0);
        
        let white_sampler = unsafe {
            device.create_sampler(&sampler_info, None)
                .context("Failed to create UI sampler")?
        };

        // Create descriptor set layout
        let bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build(),
        ];
        
        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings);
        
        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(&layout_info, None)
                .context("Failed to create UI descriptor set layout")?
        };

        // Allocate descriptor set
        let layouts = [descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(graphics.descriptor_pool)
            .set_layouts(&layouts);
        
        let descriptor_set = unsafe {
            device.allocate_descriptor_sets(&alloc_info)
                .context("Failed to allocate UI descriptor set")?[0]
        };

        // Update descriptor set with white texture
        let image_info = [vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(white_texture_view)
            .sampler(white_sampler)
            .build()];
        
        let descriptor_writes = [vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_info)
            .build()];
        
        unsafe {
            device.update_descriptor_sets(&descriptor_writes, &[]);
        }

        // Create pipeline layout with push constants
        let push_constant_range = vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<UiPushConstants>() as u32)
            .build();
        
        let push_constant_ranges = [push_constant_range];
        let set_layouts = [descriptor_set_layout];
        
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_constant_ranges);
        
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(&pipeline_layout_info, None)
                .context("Failed to create UI pipeline layout")?
        };

        // Create pipeline
        let pipeline = Self::create_pipeline(&graphics, pipeline_layout)?;

        Ok(Self {
            graphics,
            pipeline,
            pipeline_layout,
            vertex_buffer,
            vertex_memory,
            index_buffer,
            index_memory,
            vertex_ptr,
            index_ptr,
            white_texture,
            white_texture_memory,
            white_texture_view,
            white_sampler,
            descriptor_set_layout,
            descriptor_set,
            font_atlas_descriptor_set: None,
        })
    }

    /// Create a 1x1 white texture
    fn create_white_texture(graphics: &GraphicsContext) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
        let device = &graphics.device;
        
        // Create image
        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D { width: 1, height: 1, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .format(vk::Format::R8G8B8A8_UNORM)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);
        
        let image = unsafe {
            device.create_image(&image_info, None)
                .context("Failed to create white texture image")?
        };
        
        // Allocate memory
        let mem_requirements = unsafe { device.get_image_memory_requirements(image) };
        let memory_type = graphics.find_memory_type(
            mem_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        
        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type);
        
        let memory = unsafe {
            device.allocate_memory(&alloc_info, None)
                .context("Failed to allocate white texture memory")?
        };
        
        unsafe {
            device.bind_image_memory(image, memory, 0)
                .context("Failed to bind white texture memory")?;
        }

        // Create staging buffer with white pixel
        let (staging_buffer, staging_memory) = graphics.create_buffer(
            4,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        
        unsafe {
            let ptr = device.map_memory(staging_memory, 0, 4, vk::MemoryMapFlags::empty())? as *mut u8;
            std::ptr::copy_nonoverlapping([255u8, 255, 255, 255].as_ptr(), ptr, 4);
            device.unmap_memory(staging_memory);
        }

        // Transition and copy
        let cmd = graphics.begin_single_time_commands()?;
        
        // Transition to TRANSFER_DST
        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
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
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .build();
        
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        // Copy buffer to image
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
            .image_extent(vk::Extent3D { width: 1, height: 1, depth: 1 })
            .build();
        
        unsafe {
            device.cmd_copy_buffer_to_image(
                cmd,
                staging_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }

        // Transition to SHADER_READ_ONLY
        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
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
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build();
        
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        graphics.end_single_time_commands(cmd)?;

        // Cleanup staging
        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_memory, None);
        }

        // Create image view
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
        
        let view = unsafe {
            device.create_image_view(&view_info, None)
                .context("Failed to create white texture view")?
        };

        Ok((image, memory, view))
    }

    /// Create the UI graphics pipeline
    fn create_pipeline(graphics: &GraphicsContext, layout: vk::PipelineLayout) -> Result<vk::Pipeline> {
        let device = &graphics.device;

        // Load compiled shaders (embedded at compile time via build.rs)
        let vert_code = include_bytes!("../graphics/ui_vert.spv");
        let frag_code = include_bytes!("../graphics/ui_frag.spv");

        let vert_module = Self::create_shader_module(device, vert_code)?;
        let frag_module = Self::create_shader_module(device, frag_code)?;

        let entry_name = std::ffi::CString::new("main").unwrap();
        
        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(&entry_name)
                .build(),
            vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(&entry_name)
                .build(),
        ];

        // Vertex input: position, uv, color
        let binding_descriptions = [
            vk::VertexInputBindingDescription::builder()
                .binding(0)
                .stride(std::mem::size_of::<UiVertex>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX)
                .build(),
        ];

        let attribute_descriptions = [
            // position
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(0)
                .build(),
            // uv
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(8)
                .build(),
            // color
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(16)
                .build(),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        // Dynamic viewport/scissor
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&dynamic_states);

        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::NONE) // No culling for UI
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // Alpha blending
        let color_blend_attachment = [vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .build()];

        let color_blending = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(&color_blend_attachment);

        // Depth stencil - disabled for UI (always on top)
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(false)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::ALWAYS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let pipeline_info = [vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .depth_stencil_state(&depth_stencil)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .render_pass(graphics.render_pass)
            .subpass(0)
            .build()];

        let pipeline = unsafe {
            device.create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
                .map_err(|e| anyhow::anyhow!("Failed to create UI pipeline: {:?}", e.1))?[0]
        };

        // Cleanup shader modules
        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }

        Ok(pipeline)
    }

    fn create_shader_module(device: &ash::Device, code: &[u8]) -> Result<vk::ShaderModule> {
        // Ensure alignment
        let code_aligned: Vec<u32> = code
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let create_info = vk::ShaderModuleCreateInfo::builder()
            .code(&code_aligned);

        unsafe {
            device.create_shader_module(&create_info, None)
                .context("Failed to create shader module")
        }
    }

    /// Upload canvas data to GPU buffers
    pub fn upload(&mut self, canvas: &UiCanvas) {
        let vertex_count = canvas.vertices.len().min(MAX_UI_VERTICES);
        let index_count = canvas.indices.len().min(MAX_UI_INDICES);

        if vertex_count > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    canvas.vertices.as_ptr(),
                    self.vertex_ptr,
                    vertex_count,
                );
            }
        }

        if index_count > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    canvas.indices.as_ptr(),
                    self.index_ptr,
                    index_count,
                );
            }
        }
    }

    /// Record UI render commands
    pub fn record_commands(
        &self,
        cmd: vk::CommandBuffer,
        device: &ash::Device,
        canvas: &UiCanvas,
    ) {
        if canvas.indices.is_empty() {
            return;
        }

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

            device.cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buffer], &[0]);
            device.cmd_bind_index_buffer(cmd, self.index_buffer, 0, vk::IndexType::UINT32);

            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            let push_constants = UiPushConstants {
                screen_size: [canvas.screen_size.x, canvas.screen_size.y],
            };
            
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&push_constants),
            );

            let index_count = canvas.indices.len().min(MAX_UI_INDICES) as u32;
            device.cmd_draw_indexed(cmd, index_count, 1, 0, 0, 0);
        }
    }

    /// Set font atlas for text rendering
    pub fn set_font_atlas(&mut self, font: &crate::ui::font::FontAtlas) -> Result<()> {
        let device = &self.graphics.device;
        
        // Allocate descriptor set for font atlas
        let layouts = [self.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.graphics.descriptor_pool)
            .set_layouts(&layouts);
        
        let font_descriptor_set = unsafe {
            device.allocate_descriptor_sets(&alloc_info)
                .context("Failed to allocate font atlas descriptor set")?[0]
        };

        // Update with font texture
        let image_info = [vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(font.texture.image_view)
            .sampler(self.white_sampler)
            .build()];
        
        let descriptor_writes = [vk::WriteDescriptorSet::builder()
            .dst_set(font_descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_info)
            .build()];
        
        unsafe {
            device.update_descriptor_sets(&descriptor_writes, &[]);
        }

        self.font_atlas_descriptor_set = Some(font_descriptor_set);
        Ok(())
    }

    /// Record UI render commands with font atlas for text rendering
    pub fn record_commands_with_font(
        &self,
        cmd: vk::CommandBuffer,
        device: &ash::Device,
        canvas: &UiCanvas,
    ) {
        if canvas.indices.is_empty() {
            return;
        }

        // Use font atlas descriptor if available, otherwise fallback to white texture
        let descriptor_set = self.font_atlas_descriptor_set.unwrap_or(self.descriptor_set);

        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

            device.cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buffer], &[0]);
            device.cmd_bind_index_buffer(cmd, self.index_buffer, 0, vk::IndexType::UINT32);

            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );

            let push_constants = UiPushConstants {
                screen_size: [canvas.screen_size.x, canvas.screen_size.y],
            };
            
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&push_constants),
            );

            let index_count = canvas.indices.len().min(MAX_UI_INDICES) as u32;
            device.cmd_draw_indexed(cmd, index_count, 1, 0, 0, 0);
        }
    }
}

impl Drop for UiRenderer {
    fn drop(&mut self) {
        unsafe {
            let device = &self.graphics.device;
            
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            
            device.unmap_memory(self.vertex_memory);
            device.destroy_buffer(self.vertex_buffer, None);
            device.free_memory(self.vertex_memory, None);
            
            device.unmap_memory(self.index_memory);
            device.destroy_buffer(self.index_buffer, None);
            device.free_memory(self.index_memory, None);
            
            device.destroy_sampler(self.white_sampler, None);
            device.destroy_image_view(self.white_texture_view, None);
            device.destroy_image(self.white_texture, None);
            device.free_memory(self.white_texture_memory, None);
        }
    }
}
