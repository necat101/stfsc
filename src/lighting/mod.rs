// STFSC Engine - Dynamic Lighting System
// Supports Point, Spot, and Directional lights with PBR integration

use glam::Vec3;

/// Light types supported by the engine
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LightType {
    Point = 0,
    Spot = 1,
    Directional = 2,
}

impl Default for LightType {
    fn default() -> Self {
        LightType::Point
    }
}

/// ECS Component for lights in the world
#[derive(Clone, Debug)]
pub struct Light {
    pub light_type: LightType,
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,            // Attenuation distance (Point/Spot)
    pub inner_cone_angle: f32, // Spot light inner cone (radians)
    pub outer_cone_angle: f32, // Spot light outer cone (radians)
    pub cast_shadows: bool,    // Whether this light casts shadows
}

impl Default for Light {
    fn default() -> Self {
        Self {
            light_type: LightType::Point,
            color: Vec3::ONE,
            intensity: 1.0,
            range: 10.0,
            inner_cone_angle: 0.4, // ~23 degrees
            outer_cone_angle: 0.6, // ~34 degrees
            cast_shadows: false,
        }
    }
}

impl Light {
    /// Create a point light
    pub fn point(color: Vec3, intensity: f32, range: f32) -> Self {
        Self {
            light_type: LightType::Point,
            color,
            intensity,
            range,
            ..Default::default()
        }
    }

    /// Create a spot light
    pub fn spot(
        color: Vec3,
        intensity: f32,
        range: f32,
        inner_angle: f32,
        outer_angle: f32,
    ) -> Self {
        Self {
            light_type: LightType::Spot,
            color,
            intensity,
            range,
            inner_cone_angle: inner_angle,
            outer_cone_angle: outer_angle,
            ..Default::default()
        }
    }

    /// Create a directional light (sun-like)
    pub fn directional(color: Vec3, intensity: f32) -> Self {
        Self {
            light_type: LightType::Directional,
            color,
            intensity,
            range: f32::MAX, // Infinite range
            ..Default::default()
        }
    }
}

/// GPU-compatible light data structure
/// Packed to minimize memory and optimize cache usage
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuLightData {
    /// xyz = position (world space), w = light type (0=point, 1=spot, 2=directional)
    pub position_type: [f32; 4],
    /// xyz = direction (normalized), w = range
    pub direction_range: [f32; 4],
    /// xyz = color (linear), w = intensity
    pub color_intensity: [f32; 4],
    /// x = cos(inner_cone), y = cos(outer_cone), z = 1/(cos(inner)-cos(outer)), w = shadow_index (-1 = no shadow)
    pub cone_shadow: [f32; 4],
}

impl GpuLightData {
    /// Convert a Light component + Transform into GPU format
    pub fn from_light(light: &Light, position: Vec3, direction: Vec3, shadow_index: i32) -> Self {
        let cos_inner = light.inner_cone_angle.cos();
        let cos_outer = light.outer_cone_angle.cos();
        let cone_range_inv = if (cos_inner - cos_outer).abs() > 0.0001 {
            1.0 / (cos_inner - cos_outer)
        } else {
            0.0
        };

        Self {
            position_type: [
                position.x,
                position.y,
                position.z,
                light.light_type as u8 as f32,
            ],
            direction_range: [direction.x, direction.y, direction.z, light.range],
            color_intensity: [light.color.x, light.color.y, light.color.z, light.intensity],
            cone_shadow: [cos_inner, cos_outer, cone_range_inv, shadow_index as f32],
        }
    }
}

/// Uniform buffer object for lights (sent to GPU)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct LightUBO {
    /// Array of lights (max 16 for mobile performance)
    pub lights: [GpuLightData; MAX_LIGHTS],
    /// Number of active lights
    pub num_lights: u32,
    /// Padding to align ambient to 16-byte boundary (std140 requires vec4 to be 16-byte aligned)
    pub _padding: [u32; 3],
    /// Ambient light color and intensity
    pub ambient: [f32; 4],
}

/// Maximum lights supported (optimized for Quest 3 mobile GPU)
pub const MAX_LIGHTS: usize = 32;

impl Default for LightUBO {
    fn default() -> Self {
        Self {
            lights: [GpuLightData::default(); MAX_LIGHTS],
            num_lights: 0,
            _padding: [0; 3],
            ambient: [0.1, 0.1, 0.1, 1.0], // Default low ambient
        }
    }
}

impl LightUBO {
    /// Create a new UBO with default ambient
    pub fn new() -> Self {
        Self::default()
    }

    /// Set ambient light
    pub fn set_ambient(&mut self, color: Vec3, intensity: f32) {
        self.ambient = [
            color.x * intensity,
            color.y * intensity,
            color.z * intensity,
            1.0,
        ];
    }

    /// Clear all lights
    pub fn clear(&mut self) {
        self.num_lights = 0;
    }

    /// Add a light to the UBO
    /// Returns false if at capacity
    pub fn add_light(&mut self, light_data: GpuLightData) -> bool {
        if (self.num_lights as usize) < MAX_LIGHTS {
            self.lights[self.num_lights as usize] = light_data;
            self.num_lights += 1;
            true
        } else {
            false
        }
    }
}

/// Light culling helper - determines which lights affect a given position
pub struct LightCuller {
    // Could extend with spatial partitioning (grid, BVH) for many lights
}

impl LightCuller {
    /// Simple distance-based culling for a point
    pub fn cull_lights_for_view(
        lights: &[(GpuLightData, Vec3)], // (light_data, world_position)
        view_position: Vec3,
        max_distance: f32,
    ) -> Vec<GpuLightData> {
        let mut result: Vec<(f32, GpuLightData)> = lights
            .iter()
            .filter_map(|(data, pos)| {
                let dist = view_position.distance(*pos);
                let light_type = data.position_type[3] as u8;

                // Directional lights always pass
                if light_type == LightType::Directional as u8 {
                    return Some((0.0, *data)); // Priority 0 (always first)
                }

                // Point/Spot lights: check range
                let range = data.direction_range[3];
                if dist < range + max_distance {
                    Some((dist, *data))
                } else {
                    None
                }
            })
            .collect();

        // Sort by distance (closest first), limit to MAX_LIGHTS
        result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        result
            .into_iter()
            .take(MAX_LIGHTS)
            .map(|(_, d)| d)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_light_creation() {
        let light = Light::point(Vec3::new(1.0, 0.5, 0.2), 5.0, 15.0);
        assert_eq!(light.light_type, LightType::Point);
        assert_eq!(light.intensity, 5.0);
        assert_eq!(light.range, 15.0);
    }

    #[test]
    fn test_gpu_light_packing() {
        let light = Light::point(Vec3::ONE, 2.0, 10.0);
        let gpu_data = GpuLightData::from_light(&light, Vec3::new(1.0, 2.0, 3.0), Vec3::Z, -1);

        assert_eq!(gpu_data.position_type[0], 1.0);
        assert_eq!(gpu_data.position_type[1], 2.0);
        assert_eq!(gpu_data.position_type[2], 3.0);
        assert_eq!(gpu_data.position_type[3], 0.0); // Point = 0
    }

    #[test]
    fn test_light_ubo_capacity() {
        let mut ubo = LightUBO::new();
        for _i in 0..MAX_LIGHTS {
            assert!(ubo.add_light(GpuLightData::default()));
        }
        assert!(!ubo.add_light(GpuLightData::default())); // Should fail
        assert_eq!(ubo.num_lights as usize, MAX_LIGHTS);
    }
}
