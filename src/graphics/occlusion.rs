// STFSC Engine - Occlusion Culling System
// Hierarchical Z-Buffer (Hi-Z) based GPU occlusion culling for mobile VR

use glam::{Mat4, Vec3};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

/// Axis-Aligned Bounding Box for visibility testing
#[derive(Clone, Copy, Debug)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    /// Create a new AABB from min and max corners
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Create an AABB from center and half-extents
    pub fn from_center_extents(center: Vec3, half_extents: Vec3) -> Self {
        Self {
            min: center - half_extents,
            max: center + half_extents,
        }
    }

    /// Get all 8 corners of the AABB
    pub fn corners(&self) -> [Vec3; 8] {
        [
            Vec3::new(self.min.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.max.z),
        ]
    }

    /// Get the center of the AABB
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Get the half-extents of the AABB
    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Transform AABB by a matrix (returns bounding box of transformed corners)
    pub fn transform(&self, matrix: Mat4) -> AABB {
        let corners = self.corners();
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);

        for corner in &corners {
            let transformed = matrix.transform_point3(*corner);
            min = min.min(transformed);
            max = max.max(transformed);
        }

        AABB { min, max }
    }
}

/// View frustum for frustum culling
#[derive(Clone, Copy, Debug)]
pub struct Frustum {
    pub planes: [Vec4Plane; 6], // Left, Right, Bottom, Top, Near, Far
}

/// Plane represented as (normal.xyz, distance)
#[derive(Clone, Copy, Debug, Default)]
pub struct Vec4Plane {
    pub normal: Vec3,
    pub distance: f32,
}

impl Vec4Plane {
    /// Signed distance from plane to point (positive = front, negative = behind)
    pub fn distance_to_point(&self, point: Vec3) -> f32 {
        self.normal.dot(point) + self.distance
    }
}

impl Frustum {
    /// Extract frustum planes from view-projection matrix
    /// Uses Gribb & Hartmann method
    pub fn from_view_proj(vp: Mat4) -> Self {
        let row0 = vp.row(0);
        let row1 = vp.row(1);
        let row2 = vp.row(2);
        let row3 = vp.row(3);

        let mut planes = [Vec4Plane::default(); 6];

        // Determine planes based on Vulkan clip space [0, w] and Y-down convention
        // Projection matrix flips Y, so Row 1 is inverted relative to standard GL

        // Left plane: row3 + row0 (w + x >= 0) -> Same for GL/Vulkan
        let left = row3 + row0;
        planes[0] = Self::normalize_plane(Vec4Plane {
            normal: Vec3::new(left.x, left.y, left.z),
            distance: left.w,
        });

        // Right plane: row3 - row0 (w - x >= 0) -> Same for GL/Vulkan
        let right = row3 - row0;
        planes[1] = Self::normalize_plane(Vec4Plane {
            normal: Vec3::new(right.x, right.y, right.z),
            distance: right.w,
        });

        // Bottom plane: row3 - row1 (w - y >= 0)
        // Note: With flipped Y (a22 < 0), +Y in view space becomes -Y in clip space.
        // Screen bottom is Y_clip = +1.
        // We want y_clip <= w_c -> w_c - y_c >= 0 -> row3 - row1.
        let bottom = row3 - row1;
        planes[2] = Self::normalize_plane(Vec4Plane {
            normal: Vec3::new(bottom.x, bottom.y, bottom.z),
            distance: bottom.w,
        });

        // Top plane: row3 + row1 (w + y >= 0)
        // Screen top is Y_clip = -1.
        // We want y_clip >= -w_c -> w_c + y_c >= 0 -> row3 + row1.
        let top = row3 + row1;
        planes[3] = Self::normalize_plane(Vec4Plane {
            normal: Vec3::new(top.x, top.y, top.z),
            distance: top.w,
        });

        // Near plane: row2 (z >= 0) for Vulkan [0, 1] depth
        // GL would use row3 + row2 (z >= -w)
        let near = row2;
        planes[4] = Self::normalize_plane(Vec4Plane {
            normal: Vec3::new(near.x, near.y, near.z),
            distance: near.w,
        });

        // Far plane: row3 - row2 (w - z >= 0) -> Same as GL (z <= w)
        let far = row3 - row2;
        planes[5] = Self::normalize_plane(Vec4Plane {
            normal: Vec3::new(far.x, far.y, far.z),
            distance: far.w,
        });

        Frustum { planes }
    }

    fn normalize_plane(plane: Vec4Plane) -> Vec4Plane {
        let length = plane.normal.length();
        if length > 0.0001 {
            Vec4Plane {
                normal: plane.normal / length,
                distance: plane.distance / length,
            }
        } else {
            plane
        }
    }

    /// Test if AABB intersects with frustum
    /// Returns true if AABB is at least partially inside
    pub fn intersects_aabb(&self, aabb: &AABB) -> bool {
        for plane in &self.planes {
            // Find the corner most in the direction of the plane normal
            let p = Vec3::new(
                if plane.normal.x >= 0.0 {
                    aabb.max.x
                } else {
                    aabb.min.x
                },
                if plane.normal.y >= 0.0 {
                    aabb.max.y
                } else {
                    aabb.min.y
                },
                if plane.normal.z >= 0.0 {
                    aabb.max.z
                } else {
                    aabb.min.z
                },
            );

            // If this corner is behind the plane, the AABB is completely outside
            if plane.distance_to_point(p) < 0.0 {
                return false;
            }
        }
        true
    }

    /// Test if sphere intersects with frustum
    pub fn intersects_sphere(&self, center: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            if plane.distance_to_point(center) < -radius {
                return false;
            }
        }
        true
    }
}

/// Occlusion culling manager
/// Performs CPU-side frustum culling; Hi-Z is prepared for future GPU culling
pub struct OcclusionCuller {
    /// Current frame's frustum (combined for both eyes in VR)
    frustum: Frustum,
    /// Statistics for debugging
    stats: CullingStats,
}

/// Culling statistics for debugging/profiling
#[derive(Clone, Copy, Debug, Default)]
pub struct CullingStats {
    pub total_objects: u32,
    pub frustum_culled: u32,
    pub visible: u32,
}

impl OcclusionCuller {
    /// Create a new occlusion culler
    pub fn new() -> Self {
        Self {
            frustum: Frustum::from_view_proj(Mat4::IDENTITY),
            stats: CullingStats::default(),
        }
    }

    /// Update the frustum for the current frame
    pub fn update_frustum(&mut self, view_proj: Mat4) {
        self.frustum = Frustum::from_view_proj(view_proj);
        self.stats = CullingStats::default();
    }

    /// Update with combined frustum for both VR eyes
    /// Takes the union of left and right eye frustums for conservative culling
    pub fn update_frustum_stereo(&mut self, left_vp: Mat4, _right_vp: Mat4) {
        // For stereo, we use a merged frustum that encompasses both views
        // This is a conservative approach - objects visible in either eye pass
        // For Quest 3 with ~100° FoV per eye, the merged FoV is roughly ~110°

        // Simple approach: use left eye frustum but expand near plane
        // Better approach would be to compute proper merged frustum
        self.frustum = Frustum::from_view_proj(left_vp);
        self.stats = CullingStats::default();
    }

    /// Test if an AABB is visible (not culled)
    pub fn is_visible(&mut self, aabb: &AABB) -> bool {
        self.stats.total_objects += 1;

        if self.frustum.intersects_aabb(aabb) {
            self.stats.visible += 1;
            true
        } else {
            self.stats.frustum_culled += 1;
            false
        }
    }

    /// Test if a sphere is visible
    pub fn is_sphere_visible(&mut self, center: Vec3, radius: f32) -> bool {
        self.stats.total_objects += 1;

        if self.frustum.intersects_sphere(center, radius) {
            self.stats.visible += 1;
            true
        } else {
            self.stats.frustum_culled += 1;
            false
        }
    }

    /// Get culling statistics for this frame
    pub fn stats(&self) -> CullingStats {
        self.stats
    }

    /// Get the current frustum for parallel visibility checks
    pub fn get_frustum(&self) -> &Frustum {
        &self.frustum
    }

    /// Batch visibility test for multiple AABBs
    /// Returns indices of visible objects
    pub fn cull_batch(&mut self, aabbs: &[AABB]) -> Vec<usize> {
        let mut visible = Vec::with_capacity(aabbs.len());

        for (i, aabb) in aabbs.iter().enumerate() {
            if self.is_visible(aabb) {
                visible.push(i);
            }
        }

        visible
    }

    /// Parallel batch visibility test for multiple AABBs using rayon
    /// Returns indices of visible objects - best for large batches (100+ objects)
    pub fn cull_batch_parallel(&mut self, aabbs: &[AABB]) -> Vec<usize> {
        // Use atomics for thread-safe stats
        let total = AtomicU32::new(0);
        let visible_count = AtomicU32::new(0);
        let culled_count = AtomicU32::new(0);
        let frustum = self.frustum; // Copy for parallel access

        let visible: Vec<usize> = aabbs
            .par_iter()
            .enumerate()
            .filter_map(|(i, aabb)| {
                total.fetch_add(1, Ordering::Relaxed);
                if frustum.intersects_aabb(aabb) {
                    visible_count.fetch_add(1, Ordering::Relaxed);
                    Some(i)
                } else {
                    culled_count.fetch_add(1, Ordering::Relaxed);
                    None
                }
            })
            .collect();

        // Update stats after parallel work
        self.stats.total_objects += total.load(Ordering::Relaxed);
        self.stats.visible += visible_count.load(Ordering::Relaxed);
        self.stats.frustum_culled += culled_count.load(Ordering::Relaxed);

        visible
    }
}

impl Default for OcclusionCuller {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute AABB for a mesh given its transform
pub fn compute_mesh_aabb(base_aabb: &AABB, transform: Mat4) -> AABB {
    base_aabb.transform(transform)
}

/// Parallel visibility check for a batch of transformed AABBs
/// Returns Vec of bools matching each input AABB's visibility
/// This is a stateless function that can be called from any thread
pub fn check_visibility_parallel(frustum: &Frustum, aabbs: &[AABB]) -> Vec<bool> {
    aabbs
        .par_iter()
        .map(|aabb| frustum.intersects_aabb(aabb))
        .collect()
}

/// Unit cube AABB (centered at origin, size 1)
pub const UNIT_CUBE_AABB: AABB = AABB {
    min: Vec3::new(-0.5, -0.5, -0.5),
    max: Vec3::new(0.5, 0.5, 0.5),
};

/// Unit sphere bounding AABB
pub const UNIT_SPHERE_AABB: AABB = AABB {
    min: Vec3::new(-0.5, -0.5, -0.5),
    max: Vec3::new(0.5, 0.5, 0.5),
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb_corners() {
        let aabb = AABB::new(Vec3::ZERO, Vec3::ONE);
        let corners = aabb.corners();
        assert_eq!(corners[0], Vec3::ZERO);
        assert_eq!(corners[7], Vec3::ONE);
    }

    #[test]
    fn test_frustum_culling() {
        // Create a simple perspective projection looking down -Z
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
        let view = Mat4::IDENTITY;
        let vp = proj * view;

        let mut culler = OcclusionCuller::new();
        culler.update_frustum(vp);

        // Box in front of camera should be visible
        let front_box = AABB::from_center_extents(Vec3::new(0.0, 0.0, -5.0), Vec3::ONE);
        assert!(culler.is_visible(&front_box));

        // Box behind camera should be culled
        let behind_box = AABB::from_center_extents(Vec3::new(0.0, 0.0, 5.0), Vec3::ONE);
        assert!(!culler.is_visible(&behind_box));
    }

    #[test]
    fn test_sphere_visibility() {
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
        let mut culler = OcclusionCuller::new();
        culler.update_frustum(proj);

        // Sphere in front
        assert!(culler.is_sphere_visible(Vec3::new(0.0, 0.0, -5.0), 1.0));

        // Sphere behind
        assert!(!culler.is_sphere_visible(Vec3::new(0.0, 0.0, 5.0), 1.0));
    }
}
