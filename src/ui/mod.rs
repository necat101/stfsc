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

/// A rectangular panel (solid color or textured)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Panel {
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub anchor: Anchor,
    pub color: [f32; 4],
    pub corner_radius: f32,
    pub texture_id: Option<String>,
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
            label: Text::new_with_font_size(label, x, y, 20.0)
                .with_anchor(Anchor::Center), // Default to center for labels
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
    pub fn contains(&self, point: Vec2, screen_size: Vec2) -> bool {
        let anchor_offset = self.panel.anchor.offset();
        let top_left = Vec2::new(
            self.panel.position[0] + screen_size.x * anchor_offset.x - self.panel.size[0] * anchor_offset.x,
            self.panel.position[1] + screen_size.y * anchor_offset.y - self.panel.size[1] * anchor_offset.y,
        );
        point.x >= top_left.x
            && point.x <= top_left.x + self.panel.size[0]
            && point.y >= top_left.y
            && point.y <= top_left.y + self.panel.size[1]
    }
}

// ============================================================================
// UI LAYOUT
// ============================================================================

/// A serializable UI keybind
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Keybind {
    pub key: String,
    pub callback: String,
}

/// A complete UI layout that can be sent from the editor
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct UiLayout {
    pub buttons: Vec<Button>,
    pub panels: Vec<Panel>,
    pub texts: Vec<Text>,
    pub keybinds: Vec<Keybind>,
    pub mouse_tracking: bool,
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
    /// Screen size for coordinate conversion
    pub screen_size: Vec2,
}

impl UiCanvas {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            vertices: Vec::with_capacity(1024),
            indices: Vec::with_capacity(2048),
            screen_size: Vec2::new(width, height),
        }
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    pub fn resize(&mut self, width: f32, height: f32) {
        self.screen_size = Vec2::new(width, height);
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
        let anchor_offset = panel.anchor.offset();
        let top_left = Vec2::new(
            panel.position[0] + self.screen_size.x * anchor_offset.x - panel.size[0] * anchor_offset.x,
            panel.position[1] + self.screen_size.y * anchor_offset.y - panel.size[1] * anchor_offset.y,
        );
        
        // For now, ignore corner_radius (would need more complex geometry or SDF)
        // Use the white pixel in the font atlas for solid color panels
        // The pixel is at (1,1) in our 2x2 white block
        let uv_white = Vec2::new(1.0 / 512.0, 1.0 / 512.0);
        self.add_quad(top_left, Vec2::from(panel.size), uv_white, uv_white, panel.color);
    }

    /// Draw text (placeholder - uses simple box for now until font system is ready)
    pub fn draw_text(&mut self, text: &Text) {
        // Estimate text size based on content length and font size
        let char_width = text.font_size * 0.6;
        let text_width = text.content.len() as f32 * char_width;
        let text_height = text.font_size;

        let anchor_offset = text.anchor.offset();
        let top_left = Vec2::new(
            text.position[0] + self.screen_size.x * anchor_offset.x - text_width * anchor_offset.x,
            text.position[1] + self.screen_size.y * anchor_offset.y - text_height * anchor_offset.y,
        );

        // Draw placeholder rectangle (will be replaced with actual glyph rendering)
        // Using a slightly transparent color to indicate it's a placeholder
        // Using a slightly transparent color to indicate it's a placeholder
        let mut placeholder_color = text.color;
        placeholder_color[3] *= 0.8;
        
        let uv_white = Vec2::new(1.0 / 512.0, 1.0 / 512.0);
        self.add_quad(top_left, Vec2::new(text_width, text_height), uv_white, uv_white, placeholder_color);
    }

    /// Draw text using font atlas for proper glyph rendering
    pub fn draw_text_with_font(&mut self, text: &Text, font: &crate::ui::font::FontAtlas) {
        // Scale factor for different font sizes
        let scale = text.font_size / font.font_size;
        
        // Calculate total text width for anchor positioning
        let text_width: f32 = text.content.chars()
            .filter_map(|c| font.get_glyph(c))
            .map(|g| g.advance * scale)
            .sum();
        let text_height = font.font_size * scale;

        // Calculate anchor-adjusted starting position
        let anchor_offset = text.anchor.offset();
        let start_x = text.position[0] + self.screen_size.x * anchor_offset.x - text_width * anchor_offset.x;
        let start_y = text.position[1] + self.screen_size.y * anchor_offset.y - text_height * anchor_offset.y;

        let mut cursor_x = start_x;
        
        for c in text.content.chars() {
            if let Some(glyph) = font.get_glyph(c) {
                let glyph_width = glyph.width * scale;
                let glyph_height = glyph.height * scale;
                
                // Position glyph: x_offset positions horizontally from cursor
                // For y: fontdue's ymin is the descent below baseline (usually 0 or negative for descenders)
                // We place glyphs relative to the text's top, offset by the glyph's top edge
                let x = cursor_x + glyph.x_offset * scale;
                // The y_offset from fontdue represents distance from baseline to glyph top
                // In our top-down screen coords, higher y = lower on screen
                // For most glyphs, ymin is positive (glyph extends above baseline)
                let y = start_y + (font.font_size - glyph.height as f32 - glyph.y_offset) * scale;
                
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
                
                cursor_x += glyph.advance * scale;
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
        // The Button::new logic already sets the label position to (x, y) relative to anchor.
        // draw_text will already account for the anchor offset.
        let label = button.label.clone();
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
        self.draw_panel(&panel);

        // Draw label centered in button
        // The Button::new logic already sets the label position to (x, y) relative to anchor.
        // draw_text_with_font will already account for the anchor offset.
        let label = button.label.clone();
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
