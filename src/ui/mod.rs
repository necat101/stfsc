//! Runtime UI System for STFSC Engine
//! 
//! Provides HUD elements and overlay menus (pause menu, settings) for the 556 game.
//! Works on both desktop and VR (Quest 3).

pub mod font;
pub mod renderer;

use glam::Vec2;
use serde::{Deserialize, Serialize};

// ============================================================================
// UI VERTEX
// ============================================================================

/// Vertex for 2D UI rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiVertex {
    /// Position in screen pixels (origin = top-left)
    pub position: [f32; 2],
    /// Texture coordinates
    pub uv: [f32; 2],
    /// RGBA color (premultiplied alpha)
    pub color: [f32; 4],
}

impl UiVertex {
    pub fn new(x: f32, y: f32, u: f32, v: f32, color: [f32; 4]) -> Self {
        Self {
            position: [x, y],
            uv: [u, v],
            color,
        }
    }
}

// ============================================================================
// UI ELEMENT TYPES
// ============================================================================

/// Anchor point for UI elements
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Anchor {
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

impl Anchor {
    /// Get normalized offset (0-1 range) for this anchor
    pub fn offset(&self) -> Vec2 {
        match self {
            Anchor::TopLeft => Vec2::new(0.0, 0.0),
            Anchor::TopCenter => Vec2::new(0.5, 0.0),
            Anchor::TopRight => Vec2::new(1.0, 0.0),
            Anchor::CenterLeft => Vec2::new(0.0, 0.5),
            Anchor::Center => Vec2::new(0.5, 0.5),
            Anchor::CenterRight => Vec2::new(1.0, 0.5),
            Anchor::BottomLeft => Vec2::new(0.0, 1.0),
            Anchor::BottomCenter => Vec2::new(0.5, 1.0),
            Anchor::BottomRight => Vec2::new(1.0, 1.0),
        }
    }
}

/// Child element that can be nested inside a Panel
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PanelChild {
    Button(Button),
    Text(Text),
}

/// A rectangular panel (solid color or textured)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Panel {
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub anchor: Anchor,
    pub color: [f32; 4],
    pub corner_radius: f32,
    pub texture_id: Option<String>,
    /// Child elements (buttons, text) nested inside this panel
    #[serde(default)]
    pub children: Vec<PanelChild>,
}

impl Panel {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            position: [x, y],
            size: [width, height],
            anchor: Anchor::TopLeft,
            color: [1.0, 1.0, 1.0, 1.0],
            corner_radius: 0.0,
            texture_id: None,
            children: Vec::new(),
        }
    }

    pub fn centered(width: f32, height: f32) -> Self {
        Self {
            position: [0.0, 0.0],
            size: [width, height],
            anchor: Anchor::Center,
            color: [1.0, 1.0, 1.0, 1.0],
            corner_radius: 0.0,
            texture_id: None,
            children: Vec::new(),
        }
    }

    pub fn with_color(mut self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.color = [r, g, b, a];
        self
    }

    pub fn with_anchor(mut self, anchor: Anchor) -> Self {
        self.anchor = anchor;
        self
    }

    pub fn with_corner_radius(mut self, radius: f32) -> Self {
        self.corner_radius = radius;
        self
    }
}

/// Text element
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Text {
    pub content: String,
    pub position: [f32; 2],
    pub anchor: Anchor,
    pub font_size: f32,
    pub color: [f32; 4],
}

impl Text {
    pub fn new(content: impl Into<String>, x: f32, y: f32) -> Self {
        Self {
            content: content.into(),
            position: [x, y],
            anchor: Anchor::TopLeft,
            font_size: 24.0,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }

    pub fn centered(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            position: [0.0, 0.0],
            anchor: Anchor::Center,
            font_size: 24.0,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }

    pub fn with_font_size(mut self, size: f32) -> Self {
        self.font_size = size;
        self
    }

    pub fn with_color(mut self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.color = [r, g, b, a];
        self
    }

    pub fn with_anchor(mut self, anchor: Anchor) -> Self {
        self.anchor = anchor;
        self
    }

    pub fn new_with_font_size(content: impl Into<String>, x: f32, y: f32, size: f32) -> Self {
        Self {
            content: content.into(),
            position: [x, y],
            anchor: Anchor::TopLeft,
            font_size: size,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

/// Interactive button
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Button {
    pub panel: Panel,
    pub label: Text,
    #[serde(skip)]
    pub hovered: bool,
    #[serde(skip)]
    pub pressed: bool,
    pub id: u32,
    /// Script callback function name
    pub on_click: Option<String>,
}

impl Button {
    pub fn new(id: u32, label: &str, x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            panel: Panel::new(x, y, width, height)
                .with_color(0.2, 0.2, 0.3, 0.9)
                .with_corner_radius(8.0),
            label: Text::new_with_font_size(label, x, y, 20.0),
            hovered: false,
            pressed: false,
            id,
            on_click: None,
        }
    }

    pub fn with_anchor(mut self, anchor: Anchor) -> Self {
        self.panel.anchor = anchor;
        self.label.anchor = anchor;
        self
    }

    pub fn with_callback(mut self, callback: &str) -> Self {
        self.on_click = Some(callback.to_string());
        self
    }

    /// Check if point (in screen coords) is inside this button
    /// Uses reference resolution scaling (1920x1080 default) for hit testing
    pub fn contains(&self, point: Vec2, screen_size: Vec2) -> bool {
        self.contains_scaled(point, screen_size, DEFAULT_REFERENCE_RESOLUTION.0, DEFAULT_REFERENCE_RESOLUTION.1)
    }
    
    /// Check if point is inside this button with explicit reference resolution
    pub fn contains_scaled(&self, point: Vec2, screen_size: Vec2, ref_width: f32, ref_height: f32) -> bool {
        // Calculate scale factor (same as UiCanvas::scale_factor)
        let scale_x = screen_size.x / ref_width;
        let scale_y = screen_size.y / ref_height;
        let scale = scale_x.min(scale_y);
        
        // Scale position and size from reference resolution
        let scaled_pos = Vec2::new(self.panel.position[0] * scale, self.panel.position[1] * scale);
        let scaled_size = Vec2::new(self.panel.size[0] * scale, self.panel.size[1] * scale);
        
        let anchor_offset = self.panel.anchor.offset();
        
        // Calculate viewport offset for centering
        let viewport_offset = Vec2::new(
            (screen_size.x - ref_width * scale) / 2.0,
            (screen_size.y - ref_height * scale) / 2.0,
        );

        // New anchor formula
        let top_left = Vec2::new(
            viewport_offset.x + (ref_width * scale * anchor_offset.x) + scaled_pos.x - (scaled_size.x * anchor_offset.x),
            viewport_offset.y + (ref_height * scale * anchor_offset.y) + scaled_pos.y - (scaled_size.y * anchor_offset.y),
        );

        point.x >= top_left.x
            && point.x <= top_left.x + scaled_size.x
            && point.y >= top_left.y
            && point.y <= top_left.y + scaled_size.y
    }
}

// ============================================================================
// UI LAYOUT
// ============================================================================

// ============================================================================
// UI LAYER TYPES
// ============================================================================

/// Defines the behavior and priority of a UI layer
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum UiLayerType {
    /// Main menu layer - loads before any scene, blocks game input
    MainMenu,
    /// Pause overlay - pauses game, captures all input
    PauseOverlay,
    /// Intermediate menu - for nested menus (settings, options), doesn't auto-pause
    IntermediateMenu,
    /// In-game overlay - always visible during gameplay (HUD, minimap)
    #[default]
    InGameOverlay,
    /// Popup overlay - transient, keybind-triggered (grenade arc, weapon wheel)
    Popup,
}

/// A serializable UI keybind for callbacks
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Keybind {
    pub key: String,
    pub callback: String,
}

/// Keybind that triggers a popup overlay
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PopupKeybind {
    /// Key code (e.g., "G", "Tab", "Q")
    pub key: String,
    /// Alias of the popup layout to show
    pub popup_alias: String,
    /// If true, popup hides when key is released
    pub hold_to_show: bool,
}

/// A complete UI layout that can be sent from the editor
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct UiLayout {
    pub buttons: Vec<Button>,
    pub panels: Vec<Panel>,
    pub texts: Vec<Text>,
    pub keybinds: Vec<Keybind>,
    pub mouse_tracking: bool,
    /// Layer type determines behavior (pause, input blocking, etc.)
    #[serde(default)]
    pub layer_type: UiLayerType,
    /// Popup keybinds that trigger transient overlays
    #[serde(default)]
    pub popup_keybinds: Vec<PopupKeybind>,
    /// Blocks game input when this layer is visible
    #[serde(default)]
    pub blocks_input: bool,
    /// Pauses game logic when this layer is visible
    #[serde(default)]
    pub pauses_game: bool,
}

/// Named UI layers for different game states
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum UiLayer {
    /// Always visible in-game (health, ammo, etc.)
    Hud,
    /// Visible when game is paused
    PauseMenu,
    /// Main menu / startup screen
    MainMenu,
    /// Custom named layer for dynamically loaded UI scenes
    Custom(String),
}

impl Default for UiLayer {
    fn default() -> Self {
        UiLayer::Hud
    }
}

/// Manages multiple UI layers with independent visibility
#[derive(Default)]
pub struct UiLayerSet {
    /// Layouts for each layer
    pub layers: std::collections::HashMap<UiLayer, UiLayout>,
    /// Which layers are currently visible
    pub visible: std::collections::HashSet<UiLayer>,
}

impl UiLayerSet {
    pub fn new() -> Self {
        Self {
            layers: std::collections::HashMap::new(),
            visible: std::collections::HashSet::new(),
        }
    }
    
    /// Set a layout for a specific layer
    pub fn set_layer(&mut self, layer: UiLayer, layout: UiLayout) {
        self.layers.insert(layer, layout);
    }
    
    /// Show a layer (make it visible)
    pub fn show(&mut self, layer: UiLayer) {
        self.visible.insert(layer);
    }
    
    /// Hide a layer
    pub fn hide(&mut self, layer: &UiLayer) {
        self.visible.remove(layer);
    }
    
    /// Check if a layer is visible
    pub fn is_visible(&self, layer: &UiLayer) -> bool {
        self.visible.contains(layer)
    }
    
    /// Get layout for a layer if it exists
    pub fn get_layer(&self, layer: &UiLayer) -> Option<&UiLayout> {
        self.layers.get(layer)
    }
    
    /// Get all visible layers in render order (Hud first, PauseMenu last)
    pub fn visible_layers(&self) -> Vec<&UiLayout> {
        let mut result = Vec::new();
        // Render order: Hud -> Custom -> MainMenu -> PauseMenu (pause on top)
        if self.is_visible(&UiLayer::Hud) {
            if let Some(layout) = self.layers.get(&UiLayer::Hud) {
                result.push(layout);
            }
        }
        // Custom layers
        for (layer, layout) in &self.layers {
            if let UiLayer::Custom(_) = layer {
                if self.is_visible(layer) {
                    result.push(layout);
                }
            }
        }
        if self.is_visible(&UiLayer::MainMenu) {
            if let Some(layout) = self.layers.get(&UiLayer::MainMenu) {
                result.push(layout);
            }
        }
        if self.is_visible(&UiLayer::PauseMenu) {
            if let Some(layout) = self.layers.get(&UiLayer::PauseMenu) {
                result.push(layout);
            }
        }
        result
    }
    
    /// Check if any visible layer should pause the game
    pub fn should_pause_game(&self) -> bool {
        for layout in self.visible_layers() {
            if layout.pauses_game || layout.layer_type == UiLayerType::PauseOverlay {
                return true;
            }
        }
        false
    }
    
    /// Check if any visible layer should block game input
    pub fn should_block_input(&self) -> bool {
        for layout in self.visible_layers() {
            if layout.blocks_input || matches!(layout.layer_type, UiLayerType::PauseOverlay | UiLayerType::MainMenu) {
                return true;
            }
        }
        false
    }
    
    /// Get mutable layout for a layer if it exists
    pub fn get_layer_mut(&mut self, layer: &UiLayer) -> Option<&mut UiLayout> {
        self.layers.get_mut(layer)
    }
}

// ============================================================================
// MENU STACK (for menu_load navigation)
// ============================================================================

/// Stack-based menu navigation for intermediate menus
#[derive(Default, Clone, Debug)]
pub struct MenuStack {
    /// Stack of menu aliases (most recent on top)
    stack: Vec<String>,
}

impl MenuStack {
    pub fn new() -> Self {
        Self { stack: Vec::new() }
    }
    
    /// Push a menu onto the stack (for menu_load())
    pub fn push(&mut self, alias: &str) {
        self.stack.push(alias.to_string());
    }
    
    /// Pop the current menu (for going back)
    pub fn pop(&mut self) -> Option<String> {
        self.stack.pop()
    }
    
    /// Get the current menu alias without removing it
    pub fn current(&self) -> Option<&str> {
        self.stack.last().map(|s| s.as_str())
    }
    
    /// Clear the entire menu stack
    pub fn clear(&mut self) {
        self.stack.clear();
    }
    
    /// Check if stack is empty
    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }
    
    /// Get stack depth (for debugging)
    pub fn depth(&self) -> usize {
        self.stack.len()
    }
}

// ============================================================================
// UI CANVAS
// ============================================================================


/// Accumulated UI draw commands for a frame
#[derive(Default)]
pub struct UiCanvas {
    /// Raw vertices to submit to GPU
    pub vertices: Vec<UiVertex>,
    /// Indices for indexed drawing
    pub indices: Vec<u32>,
    /// Actual screen size for coordinate conversion
    pub screen_size: Vec2,
    /// Reference resolution used by the editor (coordinates are authored at this size)
    pub reference_resolution: Vec2,
}

/// Default editor resolution (1920x1080)
pub const DEFAULT_REFERENCE_RESOLUTION: (f32, f32) = (1920.0, 1080.0);

impl UiCanvas {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            vertices: Vec::with_capacity(1024),
            indices: Vec::with_capacity(2048),
            screen_size: Vec2::new(width, height),
            reference_resolution: Vec2::new(DEFAULT_REFERENCE_RESOLUTION.0, DEFAULT_REFERENCE_RESOLUTION.1),
        }
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    pub fn resize(&mut self, width: f32, height: f32) {
        self.screen_size = Vec2::new(width, height);
    }
    
    /// Set the reference resolution (the resolution coordinates were authored at)
    pub fn set_reference_resolution(&mut self, width: f32, height: f32) {
        self.reference_resolution = Vec2::new(width, height);
    }
    
    /// Calculate UI scale factor based on current screen size vs reference resolution
    /// Uses the smaller scale to maintain aspect ratio and ensure everything fits
    pub fn scale_factor(&self) -> f32 {
        let scale_x = self.screen_size.x / self.reference_resolution.x;
        let scale_y = self.screen_size.y / self.reference_resolution.y;
        scale_x.min(scale_y)
    }

    /// Calculate the offset (centering) needed to align the UI within the screen
    pub fn viewport_offset(&self) -> Vec2 {
        let scale = self.scale_factor();
        Vec2::new(
            (self.screen_size.x - self.reference_resolution.x * scale) / 2.0,
            (self.screen_size.y - self.reference_resolution.y * scale) / 2.0,
        )
    }
    
    /// Scale a position from reference resolution to actual screen size (without centering)
    pub fn scale_position(&self, pos: Vec2) -> Vec2 {
        let scale = self.scale_factor();
        Vec2::new(pos.x * scale, pos.y * scale)
    }
    
    /// Scale a size from reference resolution to actual screen size
    pub fn scale_size(&self, size: Vec2) -> Vec2 {
        let scale = self.scale_factor();
        Vec2::new(size.x * scale, size.y * scale)
    }

    /// Add a quad (two triangles) to the canvas
    pub fn add_quad(&mut self, top_left: Vec2, size: Vec2, uv_min: Vec2, uv_max: Vec2, color: [f32; 4]) {
        let base_idx = self.vertices.len() as u32;
        
        let tl = top_left;
        let tr = Vec2::new(top_left.x + size.x, top_left.y);
        let bl = Vec2::new(top_left.x, top_left.y + size.y);
        let br = Vec2::new(top_left.x + size.x, top_left.y + size.y);

        self.vertices.push(UiVertex::new(tl.x, tl.y, uv_min.x, uv_min.y, color));
        self.vertices.push(UiVertex::new(tr.x, tr.y, uv_max.x, uv_min.y, color));
        self.vertices.push(UiVertex::new(br.x, br.y, uv_max.x, uv_max.y, color));
        self.vertices.push(UiVertex::new(bl.x, bl.y, uv_min.x, uv_max.y, color));

        // Two triangles: TL-TR-BR and TL-BR-BL
        self.indices.extend_from_slice(&[
            base_idx, base_idx + 1, base_idx + 2,
            base_idx, base_idx + 2, base_idx + 3,
        ]);
    }

    /// Draw a panel
    pub fn draw_panel(&mut self, panel: &Panel) {
        let scale = self.scale_factor();
        let scaled_pos = self.scale_position(Vec2::from(panel.position));
        let scaled_size = self.scale_size(Vec2::from(panel.size));
        let viewport_offset = self.viewport_offset();
        
        let anchor_offset = panel.anchor.offset();
        let top_left = Vec2::new(
            viewport_offset.x + (self.reference_resolution.x * scale * anchor_offset.x) + scaled_pos.x - (scaled_size.x * anchor_offset.x),
            viewport_offset.y + (self.reference_resolution.y * scale * anchor_offset.y) + scaled_pos.y - (scaled_size.y * anchor_offset.y),
        );
        
        // For now, ignore corner_radius (would need more complex geometry or SDF)
        // Use the white pixel in the font atlas for solid color panels
        // The pixel is at (1,1) in our 2x2 white block
        let uv_white = Vec2::new(1.0 / 512.0, 1.0 / 512.0);
        self.add_quad(top_left, scaled_size, uv_white, uv_white, panel.color);
        
        // Draw child elements relative to panel position
        for child in &panel.children {
            match child {
                PanelChild::Button(button) => {
                    // Offset button by panel position (children are in local coordinates)
                    let mut offset_btn = button.clone();
                    offset_btn.panel.position[0] += panel.position[0];
                    offset_btn.panel.position[1] += panel.position[1];
                    offset_btn.label.position[0] += panel.position[0];
                    offset_btn.label.position[1] += panel.position[1];
                    self.draw_button(&offset_btn);
                }
                PanelChild::Text(text) => {
                    // Offset text by panel position
                    let mut offset_text = text.clone();
                    offset_text.position[0] += panel.position[0];
                    offset_text.position[1] += panel.position[1];
                    self.draw_text(&offset_text);
                }
            }
        }
    }

    /// Draw a panel with font atlas for proper text rendering in children
    pub fn draw_panel_with_font(&mut self, panel: &Panel, font: &crate::ui::font::FontAtlas) {
        let scale = self.scale_factor();
        let scaled_pos = self.scale_position(Vec2::from(panel.position));
        let scaled_size = self.scale_size(Vec2::from(panel.size));
        let viewport_offset = self.viewport_offset();
        
        let anchor_offset = panel.anchor.offset();
        let top_left = Vec2::new(
            viewport_offset.x + (self.reference_resolution.x * scale * anchor_offset.x) + scaled_pos.x - (scaled_size.x * anchor_offset.x),
            viewport_offset.y + (self.reference_resolution.y * scale * anchor_offset.y) + scaled_pos.y - (scaled_size.y * anchor_offset.y),
        );
        
        // Draw panel background
        let uv_white = Vec2::new(1.0 / 512.0, 1.0 / 512.0);
        self.add_quad(top_left, scaled_size, uv_white, uv_white, panel.color);
        
        // Draw child elements relative to panel position
        for child in &panel.children {
            match child {
                PanelChild::Button(btn) => {
                    // Offset button by panel position (children are in local coordinates)
                    let mut btn = btn.clone();
                    btn.panel.position[0] += panel.position[0];
                    btn.panel.position[1] += panel.position[1];
                    btn.label.position[0] += panel.position[0];
                    btn.label.position[1] += panel.position[1];
                    self.draw_button_with_font(&btn, font);
                }
                PanelChild::Text(text) => {
                    // Offset text by panel position
                    let mut text = text.clone();
                    text.position[0] += panel.position[0];
                    text.position[1] += panel.position[1];
                    self.draw_text_with_font(&text, font);
                }
            }
        }
    }

    /// Draw text (placeholder - uses simple box for now until font system is ready)
    pub fn draw_text(&mut self, text: &Text) {
        let scale = self.scale_factor();
        let scaled_pos = self.scale_position(Vec2::from(text.position));
        let viewport_offset = self.viewport_offset();
        
        // Use a simple estimation for text size in the placeholder
        let text_size = Vec2::new(text.content.len() as f32 * 10.0 * scale, 20.0 * scale);
        
        let anchor_offset = text.anchor.offset();
        let top_left = Vec2::new(
            viewport_offset.x + (self.reference_resolution.x * scale * anchor_offset.x) + scaled_pos.x - (text_size.x * anchor_offset.x),
            viewport_offset.y + (self.reference_resolution.y * scale * anchor_offset.y) + scaled_pos.y - (text_size.y * anchor_offset.y),
        );
        
        // Draw placeholder rectangle (will be replaced with actual glyph rendering)
        // Using a slightly transparent color to indicate it's a placeholder
        let mut placeholder_color = text.color;
        placeholder_color[3] *= 0.8;
        
        let uv_white = Vec2::new(1.0 / 512.0, 1.0 / 512.0);
        self.add_quad(top_left, text_size, uv_white, uv_white, placeholder_color);
    }

    /// Draw text using font atlas for proper glyph rendering
    pub fn draw_text_with_font(&mut self, text: &Text, font: &crate::ui::font::FontAtlas) {
        let scale = self.scale_factor();
        let scaled_pos = self.scale_position(Vec2::from(text.position));
        let viewport_offset = self.viewport_offset();
        
        let font_scale = (text.font_size / font.font_size) * scale;
        
        // Calculate total text width for alignment
        let mut text_width = 0.0;
        for c in text.content.chars() {
            if let Some(glyph) = font.get_glyph(c) {
                text_width += glyph.advance * font_scale;
            }
        }
        let text_height = font.font_size * font_scale;
        
        let anchor_offset = text.anchor.offset();
        let start_x = viewport_offset.x + (self.reference_resolution.x * scale * anchor_offset.x) + scaled_pos.x - (text_width * anchor_offset.x);
        let start_y = viewport_offset.y + (self.reference_resolution.y * scale * anchor_offset.y) + scaled_pos.y - (text_height * anchor_offset.y);

        let mut cursor_x = start_x;
        
        for c in text.content.chars() {
            if let Some(glyph) = font.get_glyph(c) {
                let glyph_width = glyph.width * font_scale;
                let glyph_height = glyph.height * font_scale;
                
                // Position glyph: x_offset positions horizontally from cursor
                // For y: fontdue's ymin is the descent below baseline (usually 0 or negative for descenders)
                // We place glyphs relative to the text's top, offset by the glyph's top edge
                let x = cursor_x + glyph.x_offset * font_scale;
                // The y_offset from fontdue represents distance from baseline to glyph top
                // In our top-down screen coords, higher y = lower on screen
                // For most glyphs, ymin is positive (glyph extends above baseline)
                let y = start_y + (font.font_size - glyph.height as f32 - glyph.y_offset) * font_scale;
                
                // Only draw if glyph has actual pixels (skip space, etc)
                if glyph.width > 0.0 && glyph.height > 0.0 {
                    self.add_quad(
                        Vec2::new(x, y),
                        Vec2::new(glyph_width, glyph_height),
                        Vec2::new(glyph.uv_min[0], glyph.uv_min[1]),
                        Vec2::new(glyph.uv_max[0], glyph.uv_max[1]),
                        text.color,
                    );
                }
                
                cursor_x += glyph.advance * font_scale;
            }
        }
    }

    /// Draw a button
    pub fn draw_button(&mut self, button: &Button) {
        // Draw panel with hover/press state coloring
        let mut panel = button.panel.clone();
        if button.pressed {
            panel.color = [0.1, 0.1, 0.2, 0.95];
        } else if button.hovered {
            panel.color = [0.3, 0.3, 0.4, 0.95];
        }
        self.draw_panel(&panel);

        // Draw label centered in button
        let mut label = button.label.clone();
        let anchor_offset = button.panel.anchor.offset();
        
        // 1. Calculate the absolute center of the button in reference resolution space
        let center_x = (self.reference_resolution.x * anchor_offset.x) + button.panel.position[0] + button.panel.size[0] * (0.5 - anchor_offset.x);
        let center_y = (self.reference_resolution.y * anchor_offset.y) + button.panel.position[1] + button.panel.size[1] * (0.5 - anchor_offset.y);
        
        // 2. Position the label relative to screen center to ensure perfect centering regardless of original anchor
        label.position[0] = center_x - (self.reference_resolution.x * 0.5);
        label.position[1] = center_y - (self.reference_resolution.y * 0.5);
        label.anchor = Anchor::Center;
        self.draw_text(&label);
    }

    /// Draw a button with font atlas for proper text rendering
    pub fn draw_button_with_font(&mut self, button: &Button, font: &crate::ui::font::FontAtlas) {
        // Draw panel with hover/press state coloring
        let mut panel = button.panel.clone();
        if button.pressed {
            panel.color = [0.1, 0.1, 0.2, 0.95];
        } else if button.hovered {
            panel.color = [0.3, 0.3, 0.4, 0.95];
        }
        self.draw_panel_with_font(&panel, font);

        // Draw label centered in button
        let mut label = button.label.clone();
        let anchor_offset = button.panel.anchor.offset();
        
        // 1. Calculate the absolute center of the button in reference resolution space
        let center_x = (self.reference_resolution.x * anchor_offset.x) + button.panel.position[0] + button.panel.size[0] * (0.5 - anchor_offset.x);
        let center_y = (self.reference_resolution.y * anchor_offset.y) + button.panel.position[1] + button.panel.size[1] * (0.5 - anchor_offset.y);
        
        // 2. Position the label relative to screen center to ensure perfect centering regardless of original anchor
        label.position[0] = center_x - (self.reference_resolution.x * 0.5);
        label.position[1] = center_y - (self.reference_resolution.y * 0.5);
        label.anchor = Anchor::Center;
        self.draw_text_with_font(&label, font);
    }
}

// ============================================================================
// UI STATE
// ============================================================================

/// UI Events that can be handled by the scripting system
#[derive(Clone, Debug)]
pub enum UiEvent {
    ButtonClicked {
        id: u32,
        callback: String,
    },
}

/// Global UI state
#[derive(Clone, Debug)]
pub struct UiState {
    /// Is the game paused?
    pub paused: bool,
    /// Is the HUD visible?
    pub hud_visible: bool,
    /// Currently hovered button ID (if any)
    pub hovered_button: Option<u32>,
    /// Currently hovered button callback (if any)
    pub hovered_callback: Option<String>,
    /// Pointer position (screen coords)
    pub pointer_pos: Vec2,
    /// Is the pointer pressed?
    pub pointer_pressed: bool,
    /// Was pointer released this frame?
    pub pointer_released: bool,
    /// Pending UI events
    pub events: Vec<UiEvent>,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            paused: false,
            hud_visible: true,
            hovered_button: None,
            hovered_callback: None,
            pointer_pos: Vec2::ZERO,
            pointer_pressed: false,
            pointer_released: false,
            events: Vec::new(),
        }
    }
}

impl UiState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
        if self.paused {
            println!("UI: Game paused");
        } else {
            println!("UI: Game resumed");
        }
    }
}

// ============================================================================
// PAUSE MENU HELPER
// ============================================================================

/// Button IDs for pause menu
pub const BUTTON_RESUME: u32 = 1;
pub const BUTTON_SETTINGS: u32 = 2;
pub const BUTTON_QUIT: u32 = 3;

/// Create the default pause menu as a UiLayout
/// This can be customized by the editor or reset to default
pub fn create_default_pause_menu_layout() -> UiLayout {
    let mut layout = UiLayout::default();
    layout.layer_type = UiLayerType::PauseOverlay;
    layout.pauses_game = true;
    layout.blocks_input = true;
    
    // Background panel (semi-transparent overlay will be drawn separately)
    let panel = Panel::centered(400.0, 300.0)
        .with_color(0.1, 0.12, 0.18, 0.95)
        .with_corner_radius(16.0);
    layout.panels.push(panel);
    
    // Title text
    let mut title = Text::centered("PAUSED")
        .with_font_size(48.0)
        .with_color(1.0, 1.0, 1.0, 1.0);
    title.position[1] = -80.0;
    title.anchor = Anchor::Center;
    layout.texts.push(title);
    
    // Resume button
    let resume_btn = Button::new(BUTTON_RESUME, "Resume", 0.0, -10.0, 200.0, 50.0)
        .with_anchor(Anchor::Center)
        .with_callback("on_resume_clicked");
    layout.buttons.push(resume_btn);
    
    // Settings button
    let settings_btn = Button::new(BUTTON_SETTINGS, "Settings", 0.0, 50.0, 200.0, 50.0)
        .with_anchor(Anchor::Center)
        .with_callback("on_settings_clicked");
    layout.buttons.push(settings_btn);
    
    // Quit button
    let quit_btn = Button::new(BUTTON_QUIT, "Quit", 0.0, 110.0, 200.0, 50.0)
        .with_anchor(Anchor::Center)
        .with_callback("on_quit_clicked");
    layout.buttons.push(quit_btn);
    
    layout
}

/// Draw a simple pause menu overlay
pub fn draw_pause_menu(canvas: &mut UiCanvas, ui_state: &UiState) {
    let screen = canvas.screen_size;
    
    // Semi-transparent background overlay
    // Use the white pixel in the font atlas (at 1,1 in our 2x2 white block)
    let uv_white = Vec2::new(1.0 / 512.0, 1.0 / 512.0);
    canvas.add_quad(
        Vec2::ZERO,
        screen,
        uv_white,
        uv_white,
        [0.0, 0.0, 0.0, 0.6],
    );

    // Centered panel
    let panel_width = 400.0;
    let panel_height = 300.0;
    let panel = Panel::centered(panel_width, panel_height)
        .with_color(0.1, 0.12, 0.18, 0.95)
        .with_corner_radius(16.0);
    canvas.draw_panel(&panel);

    // Title - offset upward from center
    let title = Text::centered("PAUSED")
        .with_font_size(48.0)
        .with_color(1.0, 1.0, 1.0, 1.0);
    let mut title_positioned = title.clone();
    title_positioned.position[1] = -80.0;  // Offset from center, not absolute
    title_positioned.anchor = Anchor::Center;
    canvas.draw_text(&title_positioned);

    // Resume button - slightly above center
    let mut resume_btn = Button::new(BUTTON_RESUME, "Resume", 0.0, -10.0, 200.0, 50.0)
        .with_anchor(Anchor::Center);
    resume_btn.hovered = ui_state.hovered_button == Some(BUTTON_RESUME);
    canvas.draw_button(&resume_btn);

    // Settings button
    let mut settings_btn = Button::new(BUTTON_SETTINGS, "Settings", 0.0, 50.0, 200.0, 50.0)
        .with_anchor(Anchor::Center);
    settings_btn.hovered = ui_state.hovered_button == Some(BUTTON_SETTINGS);
    canvas.draw_button(&settings_btn);

    // Quit button
    let mut quit_btn = Button::new(BUTTON_QUIT, "Quit", 0.0, 110.0, 200.0, 50.0)
        .with_anchor(Anchor::Center);
    quit_btn.hovered = ui_state.hovered_button == Some(BUTTON_QUIT);
    canvas.draw_button(&quit_btn);
}

/// Draw pause menu with proper font rendering
pub fn draw_pause_menu_with_font(canvas: &mut UiCanvas, ui_state: &UiState, font: &crate::ui::font::FontAtlas) {
    let screen = canvas.screen_size;
    
    // Semi-transparent background overlay
    // Use the white pixel in the font atlas (at 1,1 in our 2x2 white block)
    let uv_white = Vec2::new(1.0 / 512.0, 1.0 / 512.0);
    canvas.add_quad(
        Vec2::ZERO,
        screen,
        uv_white,
        uv_white,
        [0.0, 0.0, 0.0, 0.6],
    );

    // Centered panel
    let panel_width = 400.0;
    let panel_height = 300.0;
    let panel = Panel::centered(panel_width, panel_height)
        .with_color(0.1, 0.12, 0.18, 0.95)
        .with_corner_radius(16.0);
    canvas.draw_panel(&panel);

    // Title - offset upward from center
    let title = Text::centered("PAUSED")
        .with_font_size(48.0)
        .with_color(1.0, 1.0, 1.0, 1.0);
    let mut title_positioned = title.clone();
    title_positioned.position[1] = -80.0;
    title_positioned.anchor = Anchor::Center;
    canvas.draw_text_with_font(&title_positioned, font);

    // Resume button
    let mut resume_btn = Button::new(BUTTON_RESUME, "Resume", 0.0, -10.0, 200.0, 50.0)
        .with_anchor(Anchor::Center)
        .with_callback("on_resume_clicked");
    resume_btn.hovered = ui_state.hovered_button == Some(BUTTON_RESUME);
    canvas.draw_button_with_font(&resume_btn, font);

    // Settings button
    let mut settings_btn = Button::new(BUTTON_SETTINGS, "Settings", 0.0, 50.0, 200.0, 50.0)
        .with_anchor(Anchor::Center)
        .with_callback("on_settings_clicked");
    settings_btn.hovered = ui_state.hovered_button == Some(BUTTON_SETTINGS);
    canvas.draw_button_with_font(&settings_btn, font);

    // Quit button
    let mut quit_btn = Button::new(BUTTON_QUIT, "Quit", 0.0, 110.0, 200.0, 50.0)
        .with_anchor(Anchor::Center)
        .with_callback("on_quit_clicked");
    quit_btn.hovered = ui_state.hovered_button == Some(BUTTON_QUIT);
    canvas.draw_button_with_font(&quit_btn, font);
}

/// Check which pause menu button is under the pointer, returning (id, callback_name)
pub fn get_hovered_pause_button(pointer: Vec2, screen_size: Vec2) -> Option<(u32, Option<String>)> {
    // Winit and our UI now both use top-left origin (0,0)
    let processed_pointer = pointer;
    
    // Button positions must match draw_pause_menu exactly
    let buttons = [
        Button::new(BUTTON_RESUME, "Resume", 0.0, -10.0, 200.0, 50.0)
            .with_anchor(Anchor::Center)
            .with_callback("on_resume_clicked"),
        Button::new(BUTTON_SETTINGS, "Settings", 0.0, 50.0, 200.0, 50.0)
            .with_anchor(Anchor::Center)
            .with_callback("on_settings_clicked"),
        Button::new(BUTTON_QUIT, "Quit", 0.0, 110.0, 200.0, 50.0)
            .with_anchor(Anchor::Center)
            .with_callback("on_quit_clicked"),
    ];

    for btn in &buttons {
        if btn.contains(processed_pointer, screen_size) {
            return Some((btn.id, btn.on_click.clone()));
        }
    }

    None
}
