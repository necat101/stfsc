//! Advanced model loader for the STFSC engine
//!
//! Provides 3D model import capabilities including:
//! - OBJ file loading (via tobj crate)
//! - Data structures for skeletal animation (bones, weights, keyframes)
//! - Multi-mesh handling
//!
//! # FBX Support
//! FBX files should be converted to OBJ or glTF format using Blender:
//! 1. Open FBX in Blender
//! 2. File → Export → Wavefront (.obj)
//! 3. Load the OBJ file using `load_obj_from_bytes` in world/mod.rs

use crate::world::Mesh;
use anyhow::Result;
use glam;
use std::collections::HashMap;

/// Represents a bone in the skeleton hierarchy
#[derive(Clone, Debug)]
pub struct Bone {
    pub name: String,
    pub parent_index: Option<usize>,
    /// Transforms from mesh space to bone space (inverse bind pose)
    pub inverse_bind_matrix: glam::Mat4,
    /// Local transform relative to parent
    pub local_transform: glam::Mat4,
}

/// Complete skeleton for skinned meshes
#[derive(Clone, Debug)]
pub struct Skeleton {
    pub bones: Vec<Bone>,
    pub bone_names: HashMap<String, usize>,
    /// Root bone indices (bones with no parent)
    pub root_bones: Vec<usize>,
}

impl Skeleton {
    pub fn new() -> Self {
        Self {
            bones: Vec::new(),
            bone_names: HashMap::new(),
            root_bones: Vec::new(),
        }
    }

    /// Get bone index by name
    pub fn get_bone_index(&self, name: &str) -> Option<usize> {
        self.bone_names.get(name).copied()
    }
    
    /// Add a bone to the skeleton
    pub fn add_bone(&mut self, name: String, parent: Option<usize>, inverse_bind: glam::Mat4, local: glam::Mat4) -> usize {
        let index = self.bones.len();
        self.bone_names.insert(name.clone(), index);
        if parent.is_none() {
            self.root_bones.push(index);
        }
        self.bones.push(Bone {
            name,
            parent_index: parent,
            inverse_bind_matrix: inverse_bind,
            local_transform: local,
        });
        index
    }
}

impl Default for Skeleton {
    fn default() -> Self {
        Self::new()
    }
}

/// Keyframe data for a single bone
#[derive(Clone, Debug)]
pub struct BoneKeyframes {
    pub bone_index: usize,
    pub position_keys: Vec<(f32, glam::Vec3)>,
    pub rotation_keys: Vec<(f32, glam::Quat)>,
    pub scale_keys: Vec<(f32, glam::Vec3)>,
}

/// Discrete event that occurs at a specific time during an animation
#[derive(Clone, Debug)]
pub struct AnimationEvent {
    /// Time in seconds from clip start when event fires
    pub time: f32,
    /// Event name/identifier (e.g., "footstep", "attack_hit")
    pub name: String,
    /// Optional parameters
    pub string_param: Option<String>,
    pub float_param: Option<f32>,
    pub int_param: Option<i32>,
}

impl AnimationEvent {
    pub fn new(time: f32, name: &str) -> Self {
        Self {
            time,
            name: name.to_string(),
            string_param: None,
            float_param: None,
            int_param: None,
        }
    }
}

/// Animation clip with keyframes for all bones
#[derive(Clone, Debug)]
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<BoneKeyframes>,
    pub events: Vec<AnimationEvent>,
}

impl AnimationClip {
    /// Create a new empty animation clip
    pub fn new(name: &str, duration: f32) -> Self {
        Self {
            name: name.to_string(),
            duration,
            channels: Vec::new(),
            events: Vec::new(),
        }
    }
    
    /// Sample a specific bone's transform at a given time
    pub fn sample_bone(&self, bone_index: usize, time: f32) -> (glam::Vec3, glam::Quat, glam::Vec3) {
        let t = if self.duration > 0.0 {
            time % self.duration
        } else {
            0.0
        };

        for channel in &self.channels {
            if channel.bone_index == bone_index {
                let pos = self.sample_vec3(&channel.position_keys, t, glam::Vec3::ZERO);
                let rot = self.sample_quat(&channel.rotation_keys, t);
                let scale = self.sample_vec3(&channel.scale_keys, t, glam::Vec3::ONE);
                return (pos, rot, scale);
            }
        }

        (glam::Vec3::ZERO, glam::Quat::IDENTITY, glam::Vec3::ONE)
    }

    pub fn sample(&self, time: f32, skeleton: &Skeleton) -> Vec<glam::Mat4> {
        let mut transforms = vec![glam::Mat4::IDENTITY; skeleton.bones.len()];
        let t = if self.duration > 0.0 {
            time % self.duration
        } else {
            0.0
        };

        for channel in &self.channels {
            if channel.bone_index >= transforms.len() {
                continue;
            }
            
            let pos = self.sample_vec3(&channel.position_keys, t, glam::Vec3::ZERO);
            let rot = self.sample_quat(&channel.rotation_keys, t);
            let scale = self.sample_vec3(&channel.scale_keys, t, glam::Vec3::ONE);

            transforms[channel.bone_index] =
                glam::Mat4::from_scale_rotation_translation(scale, rot, pos);
        }

        transforms
    }

    fn sample_vec3(&self, keys: &[(f32, glam::Vec3)], time: f32, default: glam::Vec3) -> glam::Vec3 {
        if keys.is_empty() {
            return default;
        }
        if keys.len() == 1 {
            return keys[0].1;
        }

        // Find surrounding keyframes
        let mut prev_idx = 0;
        for (i, (t, _)) in keys.iter().enumerate() {
            if *t <= time {
                prev_idx = i;
            } else {
                break;
            }
        }

        let next_idx = (prev_idx + 1).min(keys.len() - 1);
        if prev_idx == next_idx {
            return keys[prev_idx].1;
        }

        let (t0, v0) = keys[prev_idx];
        let (t1, v1) = keys[next_idx];
        let factor = if t1 - t0 > 0.0 {
            (time - t0) / (t1 - t0)
        } else {
            0.0
        };

        v0.lerp(v1, factor)
    }

    fn sample_quat(&self, keys: &[(f32, glam::Quat)], time: f32) -> glam::Quat {
        if keys.is_empty() {
            return glam::Quat::IDENTITY;
        }
        if keys.len() == 1 {
            return keys[0].1;
        }

        // Find surrounding keyframes
        let mut prev_idx = 0;
        for (i, (t, _)) in keys.iter().enumerate() {
            if *t <= time {
                prev_idx = i;
            } else {
                break;
            }
        }

        let next_idx = (prev_idx + 1).min(keys.len() - 1);
        if prev_idx == next_idx {
            return keys[prev_idx].1;
        }

        let (t0, q0) = keys[prev_idx];
        let (t1, q1) = keys[next_idx];
        let factor = if t1 - t0 > 0.0 {
            (time - t0) / (t1 - t0)
        } else {
            0.0
        };

        q0.slerp(q1, factor)
    }
}

/// Embedded texture extracted from model
#[derive(Clone, Debug)]
pub struct EmbeddedTexture {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// Extended mesh with skinning data
#[derive(Clone, Debug)]
pub struct SkinnedMesh {
    pub mesh: Mesh,
    pub bone_indices: Vec<[u32; 4]>,
    pub bone_weights: Vec<[f32; 4]>,
}

/// Complete model scene with all extracted data
#[derive(Clone, Debug)]
pub struct ModelScene {
    pub meshes: Vec<Mesh>,
    pub skinned_meshes: Vec<SkinnedMesh>,
    pub skeleton: Option<Skeleton>,
    pub animations: Vec<AnimationClip>,
    pub textures: Vec<EmbeddedTexture>,
}

impl ModelScene {
    pub fn new() -> Self {
        Self {
            meshes: Vec::new(),
            skinned_meshes: Vec::new(),
            skeleton: None,
            animations: Vec::new(),
            textures: Vec::new(),
        }
    }
    
    /// Create a ModelScene from an OBJ file (uses tobj, no skinning)
    pub fn from_obj_bytes(data: &[u8]) -> Result<Self> {
        let mesh = crate::world::load_obj_from_bytes(data)?;
        Ok(Self {
            meshes: vec![mesh],
            skinned_meshes: Vec::new(),
            skeleton: None,
            animations: Vec::new(),
            textures: Vec::new(),
        })
    }
}

impl Default for ModelScene {
    fn default() -> Self {
        Self::new()
    }
}

// Backwards compatibility aliases for the SpawnFbxMesh command
pub type FbxScene = ModelScene;
pub type GltfScene = ModelScene;

/// Load model from bytes - currently supports OBJ format
/// FBX files should be converted to OBJ using Blender
pub fn load_fbx_from_bytes(data: &[u8]) -> Result<ModelScene> {
    // Try to detect format and load accordingly
    // For now, assume OBJ format
    ModelScene::from_obj_bytes(data)
}

/// Alias for load_fbx_from_bytes  
pub fn load_gltf_from_bytes(data: &[u8]) -> Result<ModelScene> {
    load_fbx_from_bytes(data)
}

/// Merge all meshes in a scene into a single mesh
pub fn merge_fbx_meshes(scene: &ModelScene) -> Mesh {
    merge_model_meshes(scene)
}

/// Merge all meshes in a scene into a single mesh
pub fn merge_model_meshes(scene: &ModelScene) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    
    // Collect texture info from first mesh that has it
    let mut albedo_texture: Option<String> = None;

    // Merge static meshes
    for mesh in &scene.meshes {
        let base_idx = vertices.len() as u32;
        vertices.extend_from_slice(&mesh.vertices);
        indices.extend(mesh.indices.iter().map(|i| i + base_idx));
        
        // Preserve first available texture reference
        if albedo_texture.is_none() && mesh.albedo_texture.is_some() {
            albedo_texture = mesh.albedo_texture.clone();
        }
    }

    // Merge skinned meshes (without bone data for now)
    for skinned in &scene.skinned_meshes {
        let base_idx = vertices.len() as u32;
        vertices.extend_from_slice(&skinned.mesh.vertices);
        indices.extend(skinned.mesh.indices.iter().map(|i| i + base_idx));
        
        // Preserve first available texture reference
        if albedo_texture.is_none() && skinned.mesh.albedo_texture.is_some() {
            albedo_texture = skinned.mesh.albedo_texture.clone();
        }
    }

    let mut aabb_min = [0.0, 0.0, 0.0];
    let mut aabb_max = [0.0, 0.0, 0.0];
    if !vertices.is_empty() {
        let mut min = glam::Vec3::splat(f32::MAX);
        let mut max = glam::Vec3::splat(f32::MIN);
        for v in &vertices {
            let p = glam::Vec3::from(v.position);
            min = min.min(p);
            max = max.max(p);
        }
        aabb_min = min.to_array();
        aabb_max = max.to_array();
    }

    Mesh {
        vertices,
        indices,
        albedo: None,
        normal: None,
        metallic_roughness: None,
        albedo_texture,  // Now preserves texture reference from merged meshes
        aabb_min,
        aabb_max,
        decoded_albedo: None,
        decoded_normal: None,
        decoded_mr: None,
    }
}

/// Convert a ModelScene to a simple Mesh (first mesh only)
pub fn fbx_to_mesh(scene: &ModelScene) -> Option<Mesh> {
    model_to_mesh(scene)
}

/// Convert a ModelScene to a simple Mesh (first mesh only)
pub fn model_to_mesh(scene: &ModelScene) -> Option<Mesh> {
    if !scene.meshes.is_empty() {
        Some(scene.meshes[0].clone())
    } else if !scene.skinned_meshes.is_empty() {
        Some(scene.skinned_meshes[0].mesh.clone())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skeleton_new() {
        let skeleton = Skeleton::new();
        assert!(skeleton.bones.is_empty());
        assert!(skeleton.bone_names.is_empty());
    }
    
    #[test]
    fn test_skeleton_add_bone() {
        let mut skeleton = Skeleton::new();
        let idx = skeleton.add_bone(
            "root".to_string(),
            None,
            glam::Mat4::IDENTITY,
            glam::Mat4::IDENTITY,
        );
        assert_eq!(idx, 0);
        assert_eq!(skeleton.bones.len(), 1);
        assert_eq!(skeleton.root_bones.len(), 1);
        assert_eq!(skeleton.get_bone_index("root"), Some(0));
    }

    #[test]
    fn test_animation_sample_empty() {
        let clip = AnimationClip::new("test", 1.0);
        let skeleton = Skeleton::new();
        let transforms = clip.sample(0.5, &skeleton);
        assert!(transforms.is_empty());
    }

    #[test]
    fn test_animation_keyframe_interpolation() {
        let clip = AnimationClip::new("test", 1.0);

        // Test vec3 interpolation
        let keys = vec![
            (0.0, glam::Vec3::ZERO),
            (1.0, glam::Vec3::ONE),
        ];
        let result = clip.sample_vec3(&keys, 0.5, glam::Vec3::ZERO);
        assert!((result.x - 0.5).abs() < 0.001);
        assert!((result.y - 0.5).abs() < 0.001);
        assert!((result.z - 0.5).abs() < 0.001);
    }
    
    #[test]
    fn test_model_scene_default() {
        let scene = ModelScene::default();
        assert!(scene.meshes.is_empty());
        assert!(scene.skeleton.is_none());
    }
}
