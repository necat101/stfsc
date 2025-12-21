// STFSC Engine - Texture Streaming Manager
// Handles KTX2/ASTC compressed texture loading and streaming for mobile VR

use ash::vk;
use std::collections::HashMap;

/// Supported ASTC block sizes for Quest 3
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AstcBlockSize {
    Block4x4, // Highest quality (8 bpp)
    Block5x5, // Good quality (5.12 bpp)
    Block6x6, // Medium quality (3.56 bpp)
    Block8x8, // Low quality (2 bpp)
}

impl AstcBlockSize {
    /// Convert to Vulkan format (sRGB variants for color textures)
    pub fn to_vk_format_srgb(self) -> vk::Format {
        match self {
            AstcBlockSize::Block4x4 => vk::Format::ASTC_4X4_SRGB_BLOCK,
            AstcBlockSize::Block5x5 => vk::Format::ASTC_5X5_SRGB_BLOCK,
            AstcBlockSize::Block6x6 => vk::Format::ASTC_6X6_SRGB_BLOCK,
            AstcBlockSize::Block8x8 => vk::Format::ASTC_8X8_SRGB_BLOCK,
        }
    }

    /// Convert to Vulkan format (UNORM variants for data textures)
    pub fn to_vk_format_unorm(self) -> vk::Format {
        match self {
            AstcBlockSize::Block4x4 => vk::Format::ASTC_4X4_UNORM_BLOCK,
            AstcBlockSize::Block5x5 => vk::Format::ASTC_5X5_UNORM_BLOCK,
            AstcBlockSize::Block6x6 => vk::Format::ASTC_6X6_UNORM_BLOCK,
            AstcBlockSize::Block8x8 => vk::Format::ASTC_8X8_UNORM_BLOCK,
        }
    }

    /// Get block dimensions
    pub fn block_size(self) -> (u32, u32) {
        match self {
            AstcBlockSize::Block4x4 => (4, 4),
            AstcBlockSize::Block5x5 => (5, 5),
            AstcBlockSize::Block6x6 => (6, 6),
            AstcBlockSize::Block8x8 => (8, 8),
        }
    }
}

/// Texture priority levels for streaming
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TexturePriority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// A streamed texture with progressive mip loading
#[derive(Debug)]
pub struct StreamedTexture {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
    pub resident_mips: u32, // How many mips are currently loaded (from highest mip)
    pub format: vk::Format,
    pub last_used_frame: u64, // For LRU eviction
}

/// Request to load a texture or mip level
#[derive(Debug)]
pub struct TextureLoadRequest {
    pub path: String,
    pub priority: TexturePriority,
    pub distance: f32,    // Distance from camera (for priority)
    pub desired_mip: u32, // Which mip level to load (0 = full res)
}

/// Texture streaming manager
pub struct TextureStreamingManager {
    /// Loaded textures by path
    textures: HashMap<String, StreamedTexture>,
    /// Pending load requests
    pending: Vec<TextureLoadRequest>,
    /// Memory budget in bytes
    memory_budget: usize,
    /// Current memory usage
    current_memory: usize,
    /// Frame counter for LRU
    frame_count: u64,
}

impl TextureStreamingManager {
    /// Create a new texture streaming manager
    pub fn new(memory_budget_mb: usize) -> Self {
        Self {
            textures: HashMap::new(),
            pending: Vec::new(),
            memory_budget: memory_budget_mb * 1024 * 1024,
            current_memory: 0,
            frame_count: 0,
        }
    }

    /// Request a texture to be loaded
    pub fn request_texture(&mut self, path: &str, distance: f32) {
        // Don't re-request already loaded textures at full res
        if let Some(tex) = self.textures.get(path) {
            if tex.resident_mips >= tex.mip_levels {
                return; // Fully loaded
            }
        }

        // Calculate priority based on distance
        let priority = if distance < 5.0 {
            TexturePriority::Critical
        } else if distance < 20.0 {
            TexturePriority::High
        } else if distance < 50.0 {
            TexturePriority::Medium
        } else {
            TexturePriority::Low
        };

        // Check if already pending
        if self.pending.iter().any(|r| r.path == path) {
            return;
        }

        self.pending.push(TextureLoadRequest {
            path: path.to_string(),
            priority,
            distance,
            desired_mip: 0,
        });
    }

    /// Process pending texture loads (call once per frame)
    pub fn process_loads(&mut self, graphics_context: &super::GraphicsContext, max_uploads: usize) {
        self.frame_count += 1;

        // Sort by priority (highest first)
        self.pending.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Process up to max_uploads
        let to_process: Vec<_> = self
            .pending
            .drain(..max_uploads.min(self.pending.len()))
            .collect();

        for request in to_process {
            log::info!(
                "Loading texture: {} (priority: {:?})",
                request.path,
                request.priority
            );

            // Read file
            let data = match std::fs::read(&request.path) {
                Ok(d) => d,
                Err(e) => {
                    log::error!("Failed to read texture file {}: {:?}", request.path, e);
                    continue;
                }
            };

            // Create Texture from KTX2
            match graphics_context.create_texture_from_ktx2(&data) {
                Ok((image, memory, view)) => {
                    // Create default sampler
                    let sampler_info = vk::SamplerCreateInfo::builder()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                        .address_mode_u(vk::SamplerAddressMode::REPEAT)
                        .address_mode_v(vk::SamplerAddressMode::REPEAT)
                        .address_mode_w(vk::SamplerAddressMode::REPEAT)
                        .max_anisotropy(16.0)
                        .anisotropy_enable(true) // Should check device limits, but Quest 3 supports it
                        .max_lod(vk::LOD_CLAMP_NONE); // Allows access to all mips provided by ImageView

                    let sampler = unsafe {
                        graphics_context
                            .device
                            .create_sampler(&sampler_info, None)
                            .unwrap()
                    };

                    // Determine dimensions/format from image (we don't have them easily from return value without querying or parsing again)
                    // Better to have create_texture_from_ktx2 return a Texture struct or metadata.
                    // But we can just assume it worked. For width/height/mips, we could parse header again or refactor create_function.
                    // Since we parsed header in create_texture_from_ktx2, it would be nice to return it.
                    // But for now, let's just peek header again here or pass 0/fake if unused by renderer (renderer usually uses descriptor).
                    // Actually, KTX2 parsing is cheap enough to do again or we assume it's valid.

                    // Let's parse just to get metadata for StreamedTexture
                    // Use the crate rooted at ::ktx2
                    let reader = ::ktx2::Reader::new(&data).unwrap();
                    let header = reader.header();
                    let width = header.pixel_width;
                    let height = header.pixel_height;
                    let mip_levels = header.level_count;
                    // Use local helper
                    let format =
                        self::ktx2::ktx2_format_to_vk(header.format.unwrap().0.get()).unwrap();

                    let streamed = StreamedTexture {
                        image,
                        memory,
                        view,
                        sampler,
                        width,
                        height,
                        mip_levels,
                        resident_mips: mip_levels, // We loaded all mips in create_texture_from_ktx2
                        format,
                        last_used_frame: self.frame_count,
                    };

                    self.current_memory += data.len(); // Approximate usage
                    self.textures.insert(request.path.clone(), streamed);
                    log::info!("Texture loaded: {}", request.path);
                }
                Err(e) => {
                    log::error!("Failed to create KTX2 texture {}: {:?}", request.path, e);
                }
            }
        }
    }

    /// Update LRU tracking for a texture
    pub fn touch_texture(&mut self, path: &str) {
        if let Some(tex) = self.textures.get_mut(path) {
            tex.last_used_frame = self.frame_count;
        }
    }

    /// Get a loaded texture
    pub fn get_texture(&self, path: &str) -> Option<&StreamedTexture> {
        self.textures.get(path)
    }

    /// Get memory usage percentage
    pub fn memory_usage_percent(&self) -> f32 {
        if self.memory_budget > 0 {
            (self.current_memory as f32) / (self.memory_budget as f32) * 100.0
        } else {
            0.0
        }
    }

    /// Evict oldest textures to free memory
    pub fn evict_lru(&mut self, target_free: usize) {
        if self.current_memory <= self.memory_budget - target_free {
            return; // Already have enough free
        }

        // Collect textures sorted by last used (oldest first)
        let mut by_age: Vec<_> = self
            .textures
            .iter()
            .map(|(path, tex)| (path.clone(), tex.last_used_frame))
            .collect();
        by_age.sort_by_key(|(_, frame)| *frame);

        // Evict oldest until we have enough space
        for (path, _) in by_age {
            if self.current_memory <= self.memory_budget - target_free {
                break;
            }

            // TODO: Actually destroy texture resources
            if let Some(_tex) = self.textures.remove(&path) {
                log::info!("Evicted texture: {}", path);
                // self.current_memory -= tex.memory_size;
            }
        }
    }
}

/// KTX2 header parsing utilities
pub mod ktx2 {
    use anyhow::{bail, Result};

    /// KTX2 file header
    #[derive(Debug)]
    pub struct Ktx2Header {
        pub format: u32,
        pub width: u32,
        pub height: u32,
        pub depth: u32,
        pub layer_count: u32,
        pub face_count: u32,
        pub level_count: u32,
        pub supercompression_scheme: u32,
    }

    /// Mip level data
    #[derive(Debug)]
    pub struct MipLevel {
        pub level: u32,
        pub offset: u64,
        pub size: u64,
        pub uncompressed_size: u64,
    }

    /// Parse KTX2 header from bytes
    pub fn parse_header(data: &[u8]) -> Result<Ktx2Header> {
        if data.len() < 80 {
            bail!("KTX2 data too small for header");
        }

        // Check magic
        let magic = &data[0..12];
        if magic
            != [
                0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
            ]
        {
            bail!("Invalid KTX2 magic");
        }

        Ok(Ktx2Header {
            format: u32::from_le_bytes([data[12], data[13], data[14], data[15]]),
            width: u32::from_le_bytes([data[20], data[21], data[22], data[23]]),
            height: u32::from_le_bytes([data[24], data[25], data[26], data[27]]),
            depth: u32::from_le_bytes([data[28], data[29], data[30], data[31]]),
            layer_count: u32::from_le_bytes([data[32], data[33], data[34], data[35]]),
            face_count: u32::from_le_bytes([data[36], data[37], data[38], data[39]]),
            level_count: u32::from_le_bytes([data[40], data[41], data[42], data[43]]),
            supercompression_scheme: u32::from_le_bytes([data[44], data[45], data[46], data[47]]),
        })
    }

    /// Map KTX2 VkFormat to ash vk::Format
    pub fn ktx2_format_to_vk(format: u32) -> Option<ash::vk::Format> {
        // Common formats used in KTX2 for Quest 3
        use ash::vk::Format;
        match format {
            // ASTC formats (sRGB)
            178 => Some(Format::ASTC_4X4_SRGB_BLOCK),
            179 => Some(Format::ASTC_5X4_SRGB_BLOCK),
            180 => Some(Format::ASTC_5X5_SRGB_BLOCK),
            181 => Some(Format::ASTC_6X5_SRGB_BLOCK),
            182 => Some(Format::ASTC_6X6_SRGB_BLOCK),
            183 => Some(Format::ASTC_8X5_SRGB_BLOCK),
            184 => Some(Format::ASTC_8X6_SRGB_BLOCK),
            185 => Some(Format::ASTC_8X8_SRGB_BLOCK),
            // ASTC formats (UNORM)
            165 => Some(Format::ASTC_4X4_UNORM_BLOCK),
            166 => Some(Format::ASTC_5X4_UNORM_BLOCK),
            167 => Some(Format::ASTC_5X5_UNORM_BLOCK),
            168 => Some(Format::ASTC_6X5_UNORM_BLOCK),
            169 => Some(Format::ASTC_6X6_UNORM_BLOCK),
            170 => Some(Format::ASTC_8X5_UNORM_BLOCK),
            171 => Some(Format::ASTC_8X6_UNORM_BLOCK),
            172 => Some(Format::ASTC_8X8_UNORM_BLOCK),
            // Common uncompressed formats
            37 => Some(Format::R8G8B8A8_SRGB),
            44 => Some(Format::R8G8B8A8_UNORM),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_astc_formats() {
        assert_eq!(
            AstcBlockSize::Block4x4.to_vk_format_srgb(),
            vk::Format::ASTC_4X4_SRGB_BLOCK
        );
        assert_eq!(AstcBlockSize::Block8x8.block_size(), (8, 8));
    }

    #[test]
    fn test_texture_priority() {
        let mut manager = TextureStreamingManager::new(256);
        manager.request_texture("test.ktx2", 3.0); // Should be Critical
        assert_eq!(manager.pending[0].priority, TexturePriority::Critical);

        manager.request_texture("far.ktx2", 100.0); // Should be Low
        assert_eq!(manager.pending[1].priority, TexturePriority::Low);
    }
}
