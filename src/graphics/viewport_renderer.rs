//! Offscreen Vulkan renderer for the editor's 3D viewport
//! 
//! This module provides GPU-accelerated rendering to a texture that can be
//! displayed in egui, replacing the software rasterizer.

use anyhow::{Context, Result};
use ash::vk;
use std::sync::Arc;

use super::GraphicsContext;

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
    
    // Staging buffer for CPU readback (to display in egui)
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    
    // Command buffer for viewport rendering
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    
    // Dimensions
    width: u32,
    height: u32,
    
    // Pixel buffer for egui texture upload
    pixels: Vec<u8>,
    
    // Reference to graphics context
    graphics: Arc<GraphicsContext>,
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
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
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
        
        // 4. Create offscreen render pass (similar to main but outputs to texture)
        // We'll reuse the main render pass for now and create a compatible framebuffer
        
        // 5. Create framebuffer
        let attachments = [color_view, depth_view, motion_view];
        let framebuffer_info = vk::FramebufferCreateInfo::builder()
            .render_pass(graphics.render_pass)
            .attachments(&attachments)
            .width(width)
            .height(height)
            .layers(1);
        let framebuffer = unsafe { device.create_framebuffer(&framebuffer_info, None)? };
        
        // 6. Create staging buffer for CPU readback
        let buffer_size = (width * height * 4) as u64; // RGBA
        let (staging_buffer, staging_memory) = graphics.create_buffer(
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        
        // 7. Create command pool and buffer for this renderer
        let command_pool = graphics.create_command_pool()?;
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&alloc_info)?[0] };
        
        // 8. Create fence for synchronization
        let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        let fence = unsafe { device.create_fence(&fence_info, None)? };
        
        // 9. Allocate pixel buffer
        let pixels = vec![128u8; (width * height * 4) as usize]; // Gray default
        
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
            staging_buffer,
            staging_memory,
            command_pool,
            command_buffer,
            fence,
            width,
            height,
            pixels,
            graphics,
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
    
    /// Resize the viewport (recreates all resources)
    pub fn resize(&mut self, new_width: u32, new_height: u32) -> Result<()> {
        if new_width == self.width && new_height == self.height {
            return Ok(());
        }
        
        // Wait for any pending work
        unsafe {
            self.graphics.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        
        // Clean up old resources
        self.cleanup_resources();
        
        // Create new renderer with new size
        let mut new_renderer = Self::new(self.graphics.clone(), new_width, new_height)?;
        
        // Move new resources into self
        self.color_image = new_renderer.color_image;
        self.color_memory = new_renderer.color_memory;
        self.color_view = new_renderer.color_view;
        self.depth_image = new_renderer.depth_image;
        self.depth_memory = new_renderer.depth_memory;
        self.depth_view = new_renderer.depth_view;
        self.motion_image = new_renderer.motion_image;
        self.motion_memory = new_renderer.motion_memory;
        self.motion_view = new_renderer.motion_view;
        self.framebuffer = new_renderer.framebuffer;
        self.staging_buffer = new_renderer.staging_buffer;
        self.staging_memory = new_renderer.staging_memory;
        self.width = new_width;
        self.height = new_height;
        self.pixels = std::mem::take(&mut new_renderer.pixels);
        
        // Don't drop new_renderer's resources since we moved them
        std::mem::forget(new_renderer);
        
        Ok(())
    }
    
    /// Clean up Vulkan resources
    fn cleanup_resources(&mut self) {
        let device = &self.graphics.device;
        unsafe {
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
            device.destroy_buffer(self.staging_buffer, None);
            device.free_memory(self.staging_memory, None);
        }
    }
    
    /// Render the scene and copy to CPU-accessible buffer
    /// Returns the pixel data as RGBA bytes
    pub fn render_and_readback(&mut self) -> Result<&[u8]> {
        let device = &self.graphics.device;
        
        // Wait for previous frame
        unsafe {
            device.wait_for_fences(&[self.fence], true, u64::MAX)?;
            device.reset_fences(&[self.fence])?;
        }
        
        // Reset and begin command buffer
        unsafe {
            device.reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            
            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(self.command_buffer, &begin_info)?;
        }
        
        // Begin render pass
        let clear_values = [
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.2, 0.3, 0.4, 1.0] } }, // Sky blue
            vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 } }, // Reversed-Z
            vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] } }, // Motion vectors
        ];
        
        let render_pass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.graphics.render_pass)
            .framebuffer(self.framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width: self.width, height: self.height },
            })
            .clear_values(&clear_values);
        
        unsafe {
            device.cmd_begin_render_pass(self.command_buffer, &render_pass_info, vk::SubpassContents::INLINE);
            
            // Set viewport and scissor
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.width as f32,
                height: self.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width: self.width, height: self.height },
            };
            device.cmd_set_viewport(self.command_buffer, 0, &[viewport]);
            device.cmd_set_scissor(self.command_buffer, 0, &[scissor]);
            
            // TODO: Bind pipeline and draw scene geometry
            // For now, just clear to show it works
            
            device.cmd_end_render_pass(self.command_buffer);
        }
        
        // Transition color image for transfer
        self.transition_image_layout(
            self.color_image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        
        // Copy image to staging buffer
        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            )
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D { width: self.width, height: self.height, depth: 1 })
            .build();
        
        unsafe {
            device.cmd_copy_image_to_buffer(
                self.command_buffer,
                self.color_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.staging_buffer,
                &[region],
            );
            
            device.end_command_buffer(self.command_buffer)?;
        }
        
        // Submit command buffer
        let command_buffers = [self.command_buffer];
        let submit_info = vk::SubmitInfo::builder()
            .command_buffers(&command_buffers);
        
        unsafe {
            let _lock = self.graphics.queue_mutex.lock().unwrap();
            device.queue_submit(self.graphics.queue, &[submit_info.build()], self.fence)?;
            
            // Wait for completion
            device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        
        // Read pixels from staging buffer
        unsafe {
            let data_ptr = device.map_memory(
                self.staging_memory,
                0,
                (self.width * self.height * 4) as u64,
                vk::MemoryMapFlags::empty(),
            )? as *const u8;
            
            std::ptr::copy_nonoverlapping(
                data_ptr,
                self.pixels.as_mut_ptr(),
                self.pixels.len(),
            );
            
            device.unmap_memory(self.staging_memory);
        }
        
        Ok(&self.pixels)
    }
    
    /// Helper to transition image layout within command buffer
    fn transition_image_layout(
        &self,
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
            _ => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
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
                self.command_buffer,
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
        let device = &self.graphics.device;
        unsafe {
            let _ = device.device_wait_idle();
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.command_pool, None);
        }
        self.cleanup_resources();
    }
}
