//! Font rendering for UI
//!
//! Uses fontdue for lightweight glyph rasterization.
//! Creates a texture atlas for efficient GPU text rendering.

use crate::graphics::{GraphicsContext, Texture};
use anyhow::{Context, Result};
use fontdue::{Font, FontSettings};
use std::collections::HashMap;
use std::sync::Arc;

/// Glyph metrics for text layout
#[derive(Clone, Copy, Debug)]
pub struct GlyphMetrics {
    /// UV coordinates in atlas (normalized 0-1)
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    /// Size in pixels
    pub width: f32,
    pub height: f32,
    /// Offset from baseline
    pub x_offset: f32,
    pub y_offset: f32,
    /// Horizontal advance
    pub advance: f32,
}

/// Font atlas containing pre-rendered glyphs
pub struct FontAtlas {
    /// GPU texture containing the atlas
    pub texture: Texture,
    /// Metrics for each character
    pub glyphs: HashMap<char, GlyphMetrics>,
    /// Font used for rasterization
    pub font: Font,
    /// Font size this atlas was rasterized at
    pub font_size: f32,
    /// Line height
    pub line_height: f32,
}

impl FontAtlas {
    /// Create a new font atlas from embedded font data
    pub fn new(graphics: Arc<GraphicsContext>, font_size: f32) -> Result<Self> {
        // Use a simple embedded font - we'll embed a minimal one
        // For now, use fontdue's built-in handling
        let font_data = include_bytes!("../../assets/fonts/Roboto-Regular.ttf");
        Self::from_bytes(graphics, font_data, font_size)
    }

    /// Create a font atlas from raw TTF/OTF bytes
    pub fn from_bytes(graphics: Arc<GraphicsContext>, data: &[u8], font_size: f32) -> Result<Self> {
        let font = Font::from_bytes(data, FontSettings::default())
            .map_err(|e| anyhow::anyhow!("Failed to load font: {}", e))?;

        // Characters to rasterize
        let charset: Vec<char> = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
            .chars()
            .collect();

        // Calculate atlas size (power of 2 for GPU efficiency)
        let atlas_size = 512u32;
        let mut atlas_data = vec![0u8; (atlas_size * atlas_size * 4) as usize]; // RGBA

        // Add a 2x2 white block at the start for solid color rendering
        // We use 2x2 to avoid any sampling artifacts at the edges
        for y in 0..2 {
            for x in 0..2 {
                let idx = ((y * atlas_size + x) * 4) as usize;
                atlas_data[idx] = 255;
                atlas_data[idx + 1] = 255;
                atlas_data[idx + 2] = 255;
                atlas_data[idx + 3] = 255;
            }
        }

        let mut glyphs = HashMap::new();
        let mut cursor_x = 3u32; // Start after the white block
        let mut cursor_y = 0u32;
        let mut row_height = 2u32;

        for c in charset {
            let (metrics, bitmap) = font.rasterize(c, font_size);

            // Check if we need to move to next row
            if cursor_x + metrics.width as u32 + 1 >= atlas_size {
                cursor_x = 1;
                cursor_y += row_height + 1;
                row_height = 0;
            }

            // Check if we've run out of space
            if cursor_y + metrics.height as u32 + 1 >= atlas_size {
                break; // Atlas full
            }

            // Copy glyph to atlas (convert grayscale to RGBA white with alpha)
            for y in 0..metrics.height {
                for x in 0..metrics.width {
                    let src_idx = y * metrics.width + x;
                    let dst_x = cursor_x + x as u32;
                    let dst_y = cursor_y + y as u32;
                    let dst_idx = ((dst_y * atlas_size + dst_x) * 4) as usize;

                    let alpha = bitmap[src_idx];
                    atlas_data[dst_idx] = 255;     // R
                    atlas_data[dst_idx + 1] = 255; // G
                    atlas_data[dst_idx + 2] = 255; // B
                    atlas_data[dst_idx + 3] = alpha; // A
                }
            }

            // Store glyph metrics
            glyphs.insert(c, GlyphMetrics {
                uv_min: [
                    cursor_x as f32 / atlas_size as f32,
                    cursor_y as f32 / atlas_size as f32,
                ],
                uv_max: [
                    (cursor_x + metrics.width as u32) as f32 / atlas_size as f32,
                    (cursor_y + metrics.height as u32) as f32 / atlas_size as f32,
                ],
                width: metrics.width as f32,
                height: metrics.height as f32,
                x_offset: metrics.xmin as f32,
                y_offset: metrics.ymin as f32,
                advance: metrics.advance_width,
            });

            cursor_x += metrics.width as u32 + 1;
            row_height = row_height.max(metrics.height as u32);
        }

        // Create GPU texture
        let texture = graphics.create_texture_from_raw(atlas_size, atlas_size, &atlas_data)
            .context("Failed to create font atlas texture")?;

        let line_height = font_size * 1.2;

        Ok(Self {
            texture,
            glyphs,
            font,
            font_size,
            line_height,
        })
    }

    /// Get the metrics for a character (returns space metrics if not found)
    pub fn get_glyph(&self, c: char) -> Option<&GlyphMetrics> {
        self.glyphs.get(&c).or_else(|| self.glyphs.get(&' '))
    }

    /// Measure the width of a string
    pub fn measure_text(&self, text: &str) -> f32 {
        text.chars()
            .filter_map(|c| self.get_glyph(c))
            .map(|g| g.advance)
            .sum()
    }

    /// Measure text height (single line)
    pub fn text_height(&self) -> f32 {
        self.font_size
    }
}
