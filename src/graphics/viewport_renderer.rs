//! Offscreen Vulkan renderer for the editor's 3D viewport
//! 
//! This module provides GPU-accelerated rendering to a texture that can be
//! displayed in egui, replacing the software rasterizer.
//!
//! This module is only available on desktop platforms (not Android).

#![cfg(not(target_os = "android"))]

use anyhow::{Context, Result};
use ash::vk;
use glam::{Mat4, Vec3, Vec4};
use std::collections::HashMap;
use std::sync::Arc;
use std::ffi::CString;

use super::{GraphicsContext, GlobalData};
use crate::world::{Mesh, Vertex};
use eframe::egui::Color32;

/// Handle to a GPU-uploaded mesh in the viewport renderer
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ViewportMeshHandle(u64);

/// GPU mesh data (vertex/index buffers + optional material)
struct GpuMeshData {
    vertex_buffer: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_memory: vk::DeviceMemory,
    index_count: u32,
    material_descriptor_set: Option<vk::DescriptorSet>,
}

/// Instance data matching shader's InstanceData struct layout
/// struct InstanceData { mat4 model; mat4 prevModel; vec4 color; };
#[repr(C)]
#[derive(Clone, Copy)]
struct GpuInstanceData {
    model: Mat4,
    prev_model: Mat4,
    color: Vec4,
    metallic: f32,
    roughness: f32,
    _padding: [f32; 2],
}

/// Instance data for rendering a mesh at a specific transform
#[derive(Clone)]
pub struct ViewportMeshInstance {
    pub handle: ViewportMeshHandle,
    pub model_matrix: Mat4,
    pub color: Vec3,
    pub material_descriptor_set: Option<vk::DescriptorSet>,
    pub joints: Option<Vec<Mat4>>, // Added for skeletal animation
}

/// Camera and lighting data for viewport rendering
#[derive(Clone)]
pub struct ViewportRenderParams {
    pub view_proj: Mat4,
    pub light_view_proj: Mat4, // Added for shadow sampling
    pub camera_pos: Vec3,
    pub light_dir: Vec3,
    pub ambient: f32,
}

impl Default for ViewportRenderParams {
    fn default() -> Self {
        Self {
            view_proj: Mat4::IDENTITY,
            light_view_proj: Mat4::IDENTITY,
            camera_pos: Vec3::new(0.0, 5.0, 10.0),
            light_dir: Vec3::new(0.4, 0.8, 0.3).normalize(),
            ambient: 0.35,
        }
    }
}

// ViewportPushConstants replaced by GlobalData

/// Offscreen viewport renderer that renders to a texture for egui display
pub struct ViewportRenderer {
    // Vulkan resources for offscreen rendering
    color_image: vk::Image,
    color_memory: vk::DeviceMemory,
    color_view: vk::ImageView,
    
    depth_image: vk::Image,
    depth_memory: vk::DeviceMemory,
    depth_view: vk::ImageView,
    
    // Motion vectors (required by render pass)
    motion_image: vk::Image,
    motion_memory: vk::DeviceMemory,
    motion_view: vk::ImageView,
    
    framebuffer: vk::Framebuffer,
    
    // Command pool and buffers for this renderer (double-buffered for async)
    command_pool: vk::CommandPool,
    command_buffers: [vk::CommandBuffer; 2],
    
    // Double-buffered readback resources for async performance
    staging_buffers: [vk::Buffer; 2],
    staging_memories: [vk::DeviceMemory; 2],
    fences: [vk::Fence; 2],
    current_frame: usize,
    readback_pending: [bool; 2],
    frame_count: u64, // Total frames rendered, used for sync vs async decision
    
    // Dimensions
    pub width: u32,
    pub height: u32,
    
    // Persistently mapped pointers
    instance_mapped: *mut GpuInstanceData,
    bone_mapped: *mut Mat4,
    global_mapped: *mut GlobalData,
    staging_mapped: [*mut u8; 2],
    
    // Reference to graphics context
    pub graphics: Arc<GraphicsContext>,
    
    // GPU mesh cache
    mesh_cache: HashMap<ViewportMeshHandle, GpuMeshData>,
    next_mesh_handle: u64,
    
    // Per-frame scene data
    render_params: ViewportRenderParams,
    pending_instances: Vec<ViewportMeshInstance>,
    
    // Descriptor sets for rendering
    global_descriptor_set: Option<vk::DescriptorSet>,
    default_material_ds: Option<vk::DescriptorSet>,
    
    // Instance buffer for model matrices (CPU-writable, updated per frame)
    instance_buffer: vk::Buffer,
    instance_memory: vk::DeviceMemory,
    max_instances: usize,
    
    // Light uniform buffer (dummy for binding)
    light_buffer: vk::Buffer,
    light_memory: vk::DeviceMemory,
    
    // Bone matrices buffer (updated per draw call for simplicity in viewport)
    bone_buffer: vk::Buffer,
    bone_memory: vk::DeviceMemory,

    // Global data buffer (matrices, camera, light)
    global_buffer: vk::Buffer,
    global_memory: vk::DeviceMemory,

    // GPU Sharpening (RCAS) resources
    sharpening_amount: f32,
    post_color_image: vk::Image,
    post_color_memory: vk::DeviceMemory,
    post_color_view: vk::ImageView,
    post_render_pass: vk::RenderPass,
    post_framebuffer: vk::Framebuffer,
    post_pipeline_layout: vk::PipelineLayout,
    post_pipeline: vk::Pipeline,
    post_descriptor_pool: vk::DescriptorPool,
    post_ds_layout: vk::DescriptorSetLayout,
    post_descriptor_sets: [vk::DescriptorSet; 2],
    post_sampler: vk::Sampler,
}

impl ViewportRenderer {
    /// Create a new viewport renderer with the specified dimensions
    pub fn new(graphics: Arc<GraphicsContext>, width: u32, height: u32) -> Result<Self> {
        let device = &graphics.device;
        
        // 1. Create color image (RGBA8 for egui compatibility)
        let color_format = vk::Format::R8G8B8A8_UNORM;
        let (color_image, color_memory) = graphics.create_image(
            width, height, 1,
            vk::SampleCountFlags::TYPE_1,
            color_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        
        let color_view = Self::create_image_view(device, color_image, color_format, vk::ImageAspectFlags::COLOR)?;
        
        // 2. Create depth image
        let (depth_image, depth_memory) = graphics.create_image(
            width, height, 1,
            vk::SampleCountFlags::TYPE_1,
            graphics.depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        
        let depth_view = Self::create_image_view(device, depth_image, graphics.depth_format, vk::ImageAspectFlags::DEPTH)?;
        
        // 3. Create motion vector image (required by render pass)
        let motion_format = vk::Format::R16G16_SFLOAT;
        let (motion_image, motion_memory) = graphics.create_image(
            width, height, 1,
            vk::SampleCountFlags::TYPE_1,
            motion_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        
        let motion_view = Self::create_image_view(device, motion_image, motion_format, vk::ImageAspectFlags::COLOR)?;
        
        // 5. Create framebuffer
        let attachments = [color_view, depth_view, motion_view];
        let framebuffer_info = vk::FramebufferCreateInfo::builder()
            .render_pass(graphics.render_pass)
            .attachments(&attachments)
            .width(width)
            .height(height)
            .layers(1);
        let framebuffer = unsafe { device.create_framebuffer(&framebuffer_info, None)? };
        
        // 6. Create double-buffered readback resources
        let buffer_size = (width * height * 4) as u64; // RGBA
        let mut staging_buffers = [vk::Buffer::null(); 2];
        let mut staging_memories = [vk::DeviceMemory::null(); 2];
        let mut fences = [vk::Fence::null(); 2];
        
        for i in 0..2 {
            let (buf, mem) = graphics.create_buffer(
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            staging_buffers[i] = buf;
            staging_memories[i] = mem;
            
            let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
            fences[i] = unsafe { device.create_fence(&fence_info, None)? };
        }
        
        // 7. Create command pool and buffers for this renderer (double-buffered)
        let command_pool = graphics.create_command_pool()?;
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(2);
        let cmd_buffers = unsafe { device.allocate_command_buffers(&alloc_info)? };
        let command_buffers = [cmd_buffers[0], cmd_buffers[1]];
        
        let mut staging_mapped = [std::ptr::null_mut(); 2];
        for i in 0..2 {
             staging_mapped[i] = unsafe { 
                 device.map_memory(staging_memories[i], 0, buffer_size, vk::MemoryMapFlags::empty())? as *mut u8
             };
        }
        
        // 10. Create instance buffer
        let max_instances: usize = 1024;
        let instance_buffer_size = (std::mem::size_of::<GpuInstanceData>() * max_instances) as u64;
        let (instance_buffer, instance_memory) = graphics.create_buffer(
            instance_buffer_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let instance_mapped = unsafe {
            device.map_memory(instance_memory, 0, instance_buffer_size, vk::MemoryMapFlags::empty())? as *mut GpuInstanceData
        };
        
        // 11. Create light uniform buffer (dummy for binding)
        let (light_buffer, light_memory) = graphics.create_buffer(
            16416,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        // 12. Create bone matrices buffer
        // Size: 128 bones * 64 bytes = 8192 bytes
        let (bone_buffer, bone_memory) = graphics.create_buffer(
            8192,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let bone_mapped = unsafe {
            device.map_memory(bone_memory, 0, 8192, vk::MemoryMapFlags::empty())? as *mut Mat4
        };

        // 12. Create global data buffer
        let (global_buffer, global_memory) = graphics.create_buffer(
            std::mem::size_of::<GlobalData>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let global_mapped = unsafe {
            device.map_memory(global_memory, 0, std::mem::size_of::<GlobalData>() as u64, vk::MemoryMapFlags::empty())? as *mut GlobalData
        };
        
        // 13. Create descriptor sets
        let global_descriptor_set = graphics.create_global_descriptor_set(
            graphics.shadow_depth_view,
            graphics.shadow_sampler,
            instance_buffer,
            light_buffer,
            bone_buffer,
            global_buffer,
        ).ok();
        
        // Create default textures and material descriptor set
        let default_material_ds = if let Ok((albedo, normal, mr)) = graphics.create_default_pbr_textures() {
            graphics.create_material_descriptor_set(&albedo, &normal, &mr).ok()
        } else {
            None
        };

        // --- GPU Sharpening (RCAS) Setup ---
        
        // 13. Post-process Color Image
        let (post_color_image, post_color_memory) = graphics.create_image(
            width, height, 1,
            vk::SampleCountFlags::TYPE_1,
            color_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let post_color_view = Self::create_image_view(device, post_color_image, color_format, vk::ImageAspectFlags::COLOR)?;

        // 14. Post Render Pass
        let post_attachments = [vk::AttachmentDescription::builder()
            .format(color_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build()];
        
        let post_color_ref = [vk::AttachmentReference::builder().attachment(0).layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL).build()];
        let post_subpass = [vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&post_color_ref)
            .build()];
            
        let post_render_pass = unsafe { device.create_render_pass(&vk::RenderPassCreateInfo::builder().attachments(&post_attachments).subpasses(&post_subpass), None)? };
        
        let post_fb_attachments = [post_color_view];
        let post_framebuffer = unsafe { device.create_framebuffer(&vk::FramebufferCreateInfo::builder().render_pass(post_render_pass).attachments(&post_fb_attachments).width(width).height(height).layers(1), None)? };

        // 15. Post Descriptor Set Layout & Pool
        let post_bindings = [
            vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build(),
        ];
        let post_ds_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::builder().bindings(&post_bindings), None)? };
        
        let post_pool_sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 2 }];
        let post_descriptor_pool = unsafe { device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::builder().pool_sizes(&post_pool_sizes).max_sets(2), None)? };
        
        let post_layouts = [post_ds_layout, post_ds_layout];
        let post_descriptor_sets_vec = unsafe { device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::builder().descriptor_pool(post_descriptor_pool).set_layouts(&post_layouts))? };
        let post_descriptor_sets = [post_descriptor_sets_vec[0], post_descriptor_sets_vec[1]];

        // 16. Post Pipeline
        let post_push_range = [vk::PushConstantRange::builder().stage_flags(vk::ShaderStageFlags::FRAGMENT).offset(0).size(4).build()];
        let post_pipeline_layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::builder().set_layouts(&[post_ds_layout]).push_constant_ranges(&post_push_range), None)? };
        
        let post_sampler = unsafe { device.create_sampler(&vk::SamplerCreateInfo::builder().mag_filter(vk::Filter::LINEAR).min_filter(vk::Filter::LINEAR).address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE).address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE).address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE).build(), None)? };

        // Load post shaders
        let vert_code = include_bytes!(concat!(env!("OUT_DIR"), "/post.vert.spv"));
        let frag_code = include_bytes!(concat!(env!("OUT_DIR"), "/cas.frag.spv"));
        let vert_module = unsafe {
            let code = ash::util::read_spv(&mut std::io::Cursor::new(&vert_code[..]))?;
            device.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(&code), None)?
        };
        let frag_module = unsafe {
            let code = ash::util::read_spv(&mut std::io::Cursor::new(&frag_code[..]))?;
            device.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(&code), None)?
        };
        
        let main_fn = CString::new("main")?;
        let stages = [
            vk::PipelineShaderStageCreateInfo::builder().stage(vk::ShaderStageFlags::VERTEX).module(vert_module).name(&main_fn).build(),
            vk::PipelineShaderStageCreateInfo::builder().stage(vk::ShaderStageFlags::FRAGMENT).module(frag_module).name(&main_fn).build(),
        ];
        
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::builder().cull_mode(vk::CullModeFlags::NONE).polygon_mode(vk::PolygonMode::FILL).line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::builder().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend_attachment = [vk::PipelineColorBlendAttachmentState::builder().color_write_mask(vk::ColorComponentFlags::RGBA).blend_enable(false).build()];
        let color_blend = vk::PipelineColorBlendStateCreateInfo::builder().attachments(&blend_attachment);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder().depth_test_enable(false).depth_write_enable(false);
        let viewport_state = vk::PipelineViewportStateCreateInfo::builder().viewport_count(1).scissor_count(1);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(post_pipeline_layout)
            .render_pass(post_render_pass)
            .subpass(0);
            
        let post_pipeline = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info.build()], None).map_err(|e| e.1).unwrap()[0] };
        unsafe { device.destroy_shader_module(vert_module, None); device.destroy_shader_module(frag_module, None); }

        // Update descriptor sets with the scene color view
        for i in 0..2 {
            let image_info = [vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(color_view)
                .sampler(post_sampler)
                .build()];
            let write = [vk::WriteDescriptorSet::builder()
                .dst_set(post_descriptor_sets[i])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_info)
                .build()];
            unsafe { device.update_descriptor_sets(&write, &[]) };
        }

        Ok(Self {
            color_image,
            color_memory,
            color_view,
            depth_image,
            depth_memory,
            depth_view,
            motion_image,
            motion_memory,
            motion_view,
            framebuffer,
            staging_buffers,
            staging_memories,
            fences,
            current_frame: 0,
            readback_pending: [false, false],
            frame_count: 0,
            command_pool,
            command_buffers,
            width,
            height,
            graphics,
            mesh_cache: HashMap::new(),
            next_mesh_handle: 0,
            render_params: ViewportRenderParams::default(),
            pending_instances: Vec::new(),
            global_descriptor_set,
            default_material_ds,
            instance_buffer,
            instance_memory,
            instance_mapped,
            max_instances,
            light_buffer,
            light_memory,
            bone_buffer,
            bone_memory,
            bone_mapped,
            global_buffer,
            global_memory,
            global_mapped,
            staging_mapped,
            // GPU Sharpening fields
            sharpening_amount: 0.0,
            post_color_image,
            post_color_memory,
            post_color_view,
            post_render_pass,
            post_framebuffer,
            post_pipeline_layout,
            post_pipeline,
            post_descriptor_pool,
            post_ds_layout,
            post_descriptor_sets,
            post_sampler,
        })
    }
    
    /// Create an image view helper
    fn create_image_view(
        device: &ash::Device,
        image: vk::Image,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
    ) -> Result<vk::ImageView> {
        let view_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(aspect)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            );
        
        unsafe { device.create_image_view(&view_info, None).context("Failed to create image view") }
    }
    
    /// Get current dimensions
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
    
    /// Upload a mesh to GPU and return a handle for rendering
    /// Now handles material descriptor sets if they exist in the model
    pub fn upload_mesh_with_material(&mut self, mesh: &Mesh, material_ds: Option<vk::DescriptorSet>) -> Result<ViewportMeshHandle> {
        let device = &self.graphics.device;
        
        // Calculate buffer sizes
        let vertex_size = std::mem::size_of::<Vertex>() as u64;
        let vertex_buffer_size = vertex_size * mesh.vertices.len() as u64;
        let index_buffer_size = std::mem::size_of::<u32>() as u64 * mesh.indices.len() as u64;
        
        if vertex_buffer_size == 0 || index_buffer_size == 0 {
            anyhow::bail!("Cannot upload empty mesh");
        }
        
        // Create vertex buffer
        let (vertex_buffer, vertex_memory) = self.graphics.create_buffer(
            vertex_buffer_size,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        
        // Create staging buffer for vertex data
        let (vertex_staging, vertex_staging_memory) = self.graphics.create_buffer(
            vertex_buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        
        // Copy vertex data to staging
        unsafe {
            let data_ptr = device.map_memory(vertex_staging_memory, 0, vertex_buffer_size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(mesh.vertices.as_ptr(), data_ptr as *mut Vertex, mesh.vertices.len());
            device.unmap_memory(vertex_staging_memory);
        }
        
        // Create index buffer
        let (index_buffer, index_memory) = self.graphics.create_buffer(
            index_buffer_size,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        
        // Create staging buffer for index data
        let (index_staging, index_staging_memory) = self.graphics.create_buffer(
            index_buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        
        // Copy index data to staging
        unsafe {
            let data_ptr = device.map_memory(index_staging_memory, 0, index_buffer_size, vk::MemoryMapFlags::empty())?;
            std::ptr::copy_nonoverlapping(mesh.indices.as_ptr(), data_ptr as *mut u32, mesh.indices.len());
            device.unmap_memory(index_staging_memory);
        }
        
        // Copy from staging to device-local buffers
        let cmd = self.graphics.begin_single_time_commands()?;
        unsafe {
            let vertex_copy = vk::BufferCopy { src_offset: 0, dst_offset: 0, size: vertex_buffer_size };
            device.cmd_copy_buffer(cmd, vertex_staging, vertex_buffer, &[vertex_copy]);
            
            let index_copy = vk::BufferCopy { src_offset: 0, dst_offset: 0, size: index_buffer_size };
            device.cmd_copy_buffer(cmd, index_staging, index_buffer, &[index_copy]);
        }
        self.graphics.end_single_time_commands(cmd)?;
        
        // Cleanup staging buffers
        unsafe {
            device.destroy_buffer(vertex_staging, None);
            device.free_memory(vertex_staging_memory, None);
            device.destroy_buffer(index_staging, None);
            device.free_memory(index_staging_memory, None);
        }
        
        // Create handle and store
        let handle = ViewportMeshHandle(self.next_mesh_handle);
        self.next_mesh_handle += 1;
        
        self.mesh_cache.insert(handle, GpuMeshData {
            vertex_buffer,
            vertex_memory,
            index_buffer,
            index_memory,
            index_count: mesh.indices.len() as u32,
            material_descriptor_set: material_ds,
        });
        
        Ok(handle)
    }
    
    /// Backward compatibility for upload_mesh
    pub fn upload_mesh(&mut self, mesh: &Mesh) -> Result<ViewportMeshHandle> {
        self.upload_mesh_with_material(mesh, None)
    }
    
    /// Remove a mesh from GPU cache
    pub fn remove_mesh(&mut self, handle: ViewportMeshHandle) {
        if let Some(data) = self.mesh_cache.remove(&handle) {
            let device = &self.graphics.device;
            unsafe {
                device.destroy_buffer(data.vertex_buffer, None);
                device.free_memory(data.vertex_memory, None);
                device.destroy_buffer(data.index_buffer, None);
                device.free_memory(data.index_memory, None);
            }
        }
    }
    
    /// Check if a mesh handle is valid
    pub fn has_mesh(&self, handle: ViewportMeshHandle) -> bool {
        self.mesh_cache.contains_key(&handle)
    }
    
    /// Set camera and lighting parameters for the next render
    pub fn set_render_params(&mut self, params: ViewportRenderParams) {
        self.render_params = params;
    }
    
    /// Get count of pending instances to render
    pub fn pending_instance_count(&self) -> usize {
        self.pending_instances.len()
    }
    
    /// Queue a mesh instance for rendering
    pub fn queue_mesh(&mut self, instance: ViewportMeshInstance) {
        self.pending_instances.push(instance);
    }
    
    /// Clear all queued instances
    pub fn clear_queue(&mut self) {
        self.pending_instances.clear();
    }
    
        
    /// Resize the viewport textures and readback buffers
    pub fn resize(&mut self, new_width: u32, new_height: u32) -> Result<()> {
        if self.width == new_width && self.height == new_height {
            return Ok(());
        }
        
        unsafe { self.graphics.device.device_wait_idle()?; } // Wait for all frames to finish before resizing
        
        self.cleanup_framebuffer_resources();
        
        let device = &self.graphics.device;
        
        // Re-create color image
        let color_format = vk::Format::R8G8B8A8_UNORM;
        let (color_image, color_memory) = self.graphics.create_image(
            new_width, new_height, 1,
            vk::SampleCountFlags::TYPE_1,
            color_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let color_view = Self::create_image_view(device, color_image, color_format, vk::ImageAspectFlags::COLOR)?;
        
        // Re-create depth image
        let (depth_image, depth_memory) = self.graphics.create_image(
            new_width, new_height, 1,
            vk::SampleCountFlags::TYPE_1,
            self.graphics.depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let depth_view = Self::create_image_view(device, depth_image, self.graphics.depth_format, vk::ImageAspectFlags::DEPTH)?;
        
        // Re-create motion vector image
        let motion_format = vk::Format::R16G16_SFLOAT;
        let (motion_image, motion_memory) = self.graphics.create_image(
            new_width, new_height, 1,
            vk::SampleCountFlags::TYPE_1,
            motion_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let motion_view = Self::create_image_view(device, motion_image, motion_format, vk::ImageAspectFlags::COLOR)?;
        
        // Re-create framebuffer
        let attachments = [color_view, depth_view, motion_view];
        let framebuffer_info = vk::FramebufferCreateInfo::builder()
            .render_pass(self.graphics.render_pass)
            .attachments(&attachments)
            .width(new_width)
            .height(new_height)
            .layers(1);
        let framebuffer = unsafe { device.create_framebuffer(&framebuffer_info, None)? };

        // Re-create post-processing color image
        let (post_color_image, post_color_memory) = self.graphics.create_image(
            new_width, new_height, 1,
            vk::SampleCountFlags::TYPE_1,
            color_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let post_color_view = Self::create_image_view(device, post_color_image, color_format, vk::ImageAspectFlags::COLOR)?;
        
        let post_fb_attachments = [post_color_view];
        let post_framebuffer = unsafe { device.create_framebuffer(&vk::FramebufferCreateInfo::builder().render_pass(self.post_render_pass).attachments(&post_fb_attachments).width(new_width).height(new_height).layers(1), None)? };

        // Update post descriptor sets with new scene color view
        for i in 0..2 {
            let image_info = [vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(color_view)
                .sampler(self.post_sampler)
                .build()];
            let write = [vk::WriteDescriptorSet::builder()
                .dst_set(self.post_descriptor_sets[i])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_info)
                .build()];
            unsafe { device.update_descriptor_sets(&write, &[]) };
        }
        
        // Re-create double-buffered staging buffers
        let buffer_size = (new_width * new_height * 4) as u64;
        for i in 0..2 {
            let (buf, mem) = self.graphics.create_buffer(
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            self.staging_buffers[i] = buf;
            self.staging_memories[i] = mem;
            self.staging_mapped[i] = unsafe {
                device.map_memory(mem, 0, buffer_size, vk::MemoryMapFlags::empty())? as *mut u8
            };
            
            let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
            self.fences[i] = unsafe { device.create_fence(&fence_info, None)? };
        }
        
        // Update struct
        self.color_image = color_image;
        self.color_memory = color_memory;
        self.color_view = color_view;
        self.depth_image = depth_image;
        self.depth_memory = depth_memory;
        self.depth_view = depth_view;
        self.motion_image = motion_image;
        self.motion_memory = motion_memory;
        self.motion_view = motion_view;
        self.framebuffer = framebuffer;
        self.post_color_image = post_color_image;
        self.post_color_memory = post_color_memory;
        self.post_color_view = post_color_view;
        self.post_framebuffer = post_framebuffer;
        self.width = new_width;
        self.height = new_height;
        self.readback_pending = [false, false];
        self.current_frame = 0;
        self.frame_count = 0; // Reset to trigger sync readback after resize
        
        Ok(())
    }
    
    /// Clean up framebuffer-related Vulkan resources (not mesh cache)
    fn cleanup_framebuffer_resources(&mut self) {
        let device = &self.graphics.device;
        unsafe {
            let _ = device.device_wait_idle();
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_image_view(self.color_view, None);
            device.destroy_image_view(self.depth_view, None);
            device.destroy_image_view(self.motion_view, None);
            device.destroy_image(self.color_image, None);
            device.destroy_image(self.depth_image, None);
            device.destroy_image(self.motion_image, None);
            device.free_memory(self.color_memory, None);
            device.free_memory(self.depth_memory, None);
            device.free_memory(self.motion_memory, None);
            
            // Clean up post-processing resources
            device.destroy_framebuffer(self.post_framebuffer, None);
            device.destroy_image_view(self.post_color_view, None);
            device.destroy_image(self.post_color_image, None);
            device.free_memory(self.post_color_memory, None);

            for i in 0..2 {
                device.unmap_memory(self.staging_memories[i]);
                device.destroy_buffer(self.staging_buffers[i], None);
                device.free_memory(self.staging_memories[i], None);
                device.destroy_fence(self.fences[i], None);
            }
        }
    }
    
    /// Set sharpening amount for the post-processing filter (0.0 to 1.0)
    pub fn set_sharpening(&mut self, amount: f32) {
        self.sharpening_amount = amount;
    }

    /// Render the scene and copy to CPU-accessible buffer
    /// Uses double-buffered async readback for high performance
    pub fn render_and_readback(&mut self) -> Result<&[Color32]> {
        let device = &self.graphics.device;
        
        let prev_frame = (self.current_frame + 1) % 2;
        // No heavy CPU copy here anymore, we'll return a slice from mapped memory at the end.
        
        // 2. PREPARE CURRENT FRAME  
        // Get command buffer for this frame (double-buffered to avoid stalls)
        let cmd = self.command_buffers[self.current_frame];
        
        // --- Update Global UBO ---
        unsafe {
            *self.global_mapped = GlobalData {
                view_proj: self.render_params.view_proj,
                prev_view_proj: self.render_params.view_proj, // Viewport doesn't track prev frame yet
                light_space: self.render_params.light_view_proj,
                camera_pos: [self.render_params.camera_pos.x, self.render_params.camera_pos.y, self.render_params.camera_pos.z, self.render_params.ambient],
                light_dir: [self.render_params.light_dir.x, self.render_params.light_dir.y, self.render_params.light_dir.z, 0.0],
            };
        }

        // Wait for this frame's previous work to complete before reusing its command buffer
        unsafe {
            // Fences are initialized as signaled, so this will pass on first use.
            // On subsequent uses, it ensures the GPU is done with this index's previous frame.
            device.wait_for_fences(&[self.fences[self.current_frame]], true, u64::MAX)?;
            device.reset_fences(&[self.fences[self.current_frame]])?;
            
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(cmd, &begin_info)?;
        }
        
        // Upload instances
        let instance_count = self.pending_instances.len().min(self.max_instances);
        if instance_count > 0 {
            let mut instance_data = Vec::with_capacity(instance_count);
            for instance in self.pending_instances.iter().take(instance_count) {
                instance_data.push(GpuInstanceData {
                    model: instance.model_matrix,
                    prev_model: instance.model_matrix,
                    color: Vec4::new(instance.color.x, instance.color.y, instance.color.z, 1.0),
                    metallic: 0.0, // Default for viewport
                    roughness: 0.5,
                    _padding: [0.0; 2],
                });
            }
            unsafe {
                std::ptr::copy_nonoverlapping(instance_data.as_ptr(), self.instance_mapped, instance_count);
            }
        }
        
        // Use Unity-style sky blue for the viewport clear color
        let clear_values = [
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.32, 0.50, 0.69, 1.0] } }, // Sky blue
            vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 } }, // Reversed-Z (0.0 = Far)
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] } },
        ];
        
        let render_pass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.graphics.render_pass)
            .framebuffer(self.framebuffer)
            .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: self.width, height: self.height } })
            .clear_values(&clear_values);
        
        unsafe {
            device.cmd_begin_render_pass(cmd, &render_pass_info, vk::SubpassContents::INLINE);
            
            let viewport = vk::Viewport { x: 0.0, y: 0.0, width: self.width as f32, height: self.height as f32, min_depth: 0.0, max_depth: 1.0 };
            let scissor = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: self.width, height: self.height } };
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);
            
            if let Some(global_ds) = self.global_descriptor_set {
                let mut current_pipeline = self.graphics.pipeline;
                let mut current_layout = self.graphics.pipeline_layout;
                device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, current_pipeline);
                
                // Bind Global Set (Set 0)
                device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, current_layout, 0, &[global_ds], &[]);
                
                // Viewport now uses Global UBO instead of push constants
                
                let mut instance_idx: u32 = 0;
                for instance in &self.pending_instances {
                    if let Some(mesh_data) = self.mesh_cache.get(&instance.handle) {
                        // 1. Pipeline Switching (Standard vs Skinned)
                        let is_skinned = instance.joints.is_some();
                        let target_pipeline = if is_skinned { self.graphics.skinned_pipeline } else { self.graphics.pipeline };
                        let target_layout = if is_skinned { self.graphics.skinned_pipeline_layout } else { self.graphics.pipeline_layout };

                        if target_pipeline != current_pipeline {
                            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, target_pipeline);
                            current_pipeline = target_pipeline;
                            current_layout = target_layout;
                            // Re-bind descriptor sets if layout changed (though here they are identical layouts)
                            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, current_layout, 0, &[global_ds], &[]);
                        }

                        // 2. Bone Matrix Upload (Using persistent mapping)
                        if let Some(joints) = &instance.joints {
                            let bone_count = joints.len().min(128);
                            if bone_count > 0 {
                                std::ptr::copy_nonoverlapping(joints.as_ptr(), self.bone_mapped, bone_count);
                            }
                        }

                        // Bind Material Set (Set 1) - per mesh
                        let mat_ds = instance.material_descriptor_set
                            .or(mesh_data.material_descriptor_set)
                            .or(self.default_material_ds);
                            
                        if let Some(ds) = mat_ds {
                            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, current_layout, 1, &[ds], &[]);
                        }
                        
                        device.cmd_bind_vertex_buffers(cmd, 0, &[mesh_data.vertex_buffer], &[0]);
                        device.cmd_bind_index_buffer(cmd, mesh_data.index_buffer, 0, vk::IndexType::UINT32);
                        device.cmd_draw_indexed(cmd, mesh_data.index_count, 1, 0, 0, instance_idx);
                    }
                    instance_idx += 1;
                    if instance_idx as usize >= self.max_instances { break; }
                }
            }
            device.cmd_end_render_pass(cmd);
        }
        
        // --- 2.5 SHARPENING PASS (GPU RCAS) ---
        unsafe {
            // First transition scene color image to be readable by the post-processing shader
            self.transition_image_layout(cmd, self.color_image, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            
            let clear_values = [vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } }];
            let render_pass_info = vk::RenderPassBeginInfo::builder()
                .render_pass(self.post_render_pass)
                .framebuffer(self.post_framebuffer)
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: self.width, height: self.height } })
                .clear_values(&clear_values);
            
            device.cmd_begin_render_pass(cmd, &render_pass_info, vk::SubpassContents::INLINE);
            
            let viewport = vk::Viewport { x: 0.0, y: 0.0, width: self.width as f32, height: self.height as f32, min_depth: 0.0, max_depth: 1.0 };
            let scissor = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: self.width, height: self.height } };
            device.cmd_set_viewport(cmd, 0, &[viewport]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);
            
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.post_pipeline);
            device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.post_pipeline_layout, 0, &[self.post_descriptor_sets[self.current_frame]], &[]);
            
            // Push sharpening amount
            device.cmd_push_constants(cmd, self.post_pipeline_layout, vk::ShaderStageFlags::FRAGMENT, 0, std::slice::from_raw_parts(&self.sharpening_amount as *const f32 as *const u8, 4));
            
            device.cmd_draw(cmd, 3, 1, 0, 0); // Full-screen triangle
            device.cmd_end_render_pass(cmd);
            
            // Transition back for next frame
            self.transition_image_layout(cmd, self.color_image, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        }
        
        // 3. COPY TO STAGING (from post-processed image)
        self.transition_image_layout(cmd, self.post_color_image, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        
        let current_staging = self.staging_buffers[self.current_frame];
        let region = vk::BufferImageCopy::builder()
            .image_subresource(vk::ImageSubresourceLayers::builder().aspect_mask(vk::ImageAspectFlags::COLOR).mip_level(0).base_array_layer(0).layer_count(1).build())
            .image_extent(vk::Extent3D { width: self.width, height: self.height, depth: 1 })
            .build();
            
        unsafe {
            device.cmd_copy_image_to_buffer(cmd, self.post_color_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, current_staging, &[region]);
            
            // 4. TRANSITION BACK to attachment for next frame
            self.transition_image_layout(cmd, self.post_color_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
            
            device.end_command_buffer(cmd)?;
            
            // Submit
            let _lock = self.graphics.queue_mutex.lock().unwrap();
            let command_buffers_to_submit = [cmd];
            let submit_info = vk::SubmitInfo::builder()
                .command_buffers(&command_buffers_to_submit)
                .build();
            device.queue_submit(self.graphics.queue, &[submit_info], self.fences[self.current_frame])?;
        }
        
        self.readback_pending[self.current_frame] = true;
        
        self.current_frame = (self.current_frame + 1) % 2;
        self.frame_count += 1;
        
        // Clear pending instances for next frame
        self.pending_instances.clear();
        
        // Return a slice directly from the GPU-mapped staging buffer that was just filled (or is being filled)
        // For standard double-buffering, we return the frame that just finished rendering.
        // We use prev_frame here because current_frame was just incremented.
        unsafe {
            let data_ptr = self.staging_mapped[prev_frame] as *const Color32;
            Ok(std::slice::from_raw_parts(data_ptr, (self.width * self.height) as usize))
        }
    }
    
    /// Helper to transition image layout within command buffer
    fn transition_image_layout(
        &self,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let device = &self.graphics.device;
        
        let (src_access, dst_access, src_stage, dst_stage) = match (old_layout, new_layout) {
            (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL) => (
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags::TRANSFER_READ,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_READ,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ),
            (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
                vk::AccessFlags::SHADER_READ,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ),
            _ => (
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
            ),
        };
        
        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            )
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .build();
        
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
    }
}

impl Drop for ViewportRenderer {
    fn drop(&mut self) {
        self.cleanup_framebuffer_resources();
        let device = &self.graphics.device;
        unsafe {
            let _ = device.device_wait_idle();
            
            // Cleanup post-process persistent resources
            device.destroy_pipeline(self.post_pipeline, None);
            device.destroy_pipeline_layout(self.post_pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.post_ds_layout, None);
            device.destroy_descriptor_pool(self.post_descriptor_pool, None);
            device.destroy_render_pass(self.post_render_pass, None);
            device.destroy_sampler(self.post_sampler, None);

            device.destroy_command_pool(self.command_pool, None);
            device.unmap_memory(self.instance_memory);
            device.destroy_buffer(self.instance_buffer, None);
            device.free_memory(self.instance_memory, None);
            device.destroy_buffer(self.light_buffer, None);
            device.free_memory(self.light_memory, None);
            device.unmap_memory(self.bone_memory);
            device.destroy_buffer(self.bone_buffer, None);
            device.free_memory(self.bone_memory, None);

            device.unmap_memory(self.global_memory);
            device.destroy_buffer(self.global_buffer, None);
            device.free_memory(self.global_memory, None);

            for mesh_data in self.mesh_cache.values() {
                device.destroy_buffer(mesh_data.vertex_buffer, None);
                device.free_memory(mesh_data.vertex_memory, None);
                device.destroy_buffer(mesh_data.index_buffer, None);
                device.free_memory(mesh_data.index_memory, None);
            }
        }
    }
}
