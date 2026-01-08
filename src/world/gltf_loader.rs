//! Custom Animation Model Loader for STFSC Engine
//!
//! Provides skeletal animation model loading via custom binary format (.sanim)
//! and OBJ conversion for static meshes.
//!
//! # Workflow for Animated Models
//! 1. Create your animated model in Blender with skeleton + animations
//! 2. Export using the STFSC Blender addon (creates .sanim file)
//! 3. Load with `load_animated_model()` in the engine
//!
//! This avoids gltf crate dependency issues while providing full animation support.

use crate::world::{Mesh, Vertex};
use crate::world::fbx_loader::{
    AnimationClip, BoneKeyframes, ModelScene, Skeleton, SkinnedMesh,
};
use anyhow::{anyhow, Context, Result};
use glam::{Mat4, Quat, Vec3};
use std::io::{Read, Cursor};

/// Magic bytes for STFSC animation format: "SANM"
const MAGIC: [u8; 4] = [0x53, 0x41, 0x4E, 0x4D];
/// Current format version
const VERSION: u32 = 1;

/// Load a STFSC animated model from file path
pub fn load_animated_model(path: &str) -> Result<ModelScene> {
    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read animated model: {}", path))?;
    load_animated_model_from_bytes(&data)
}

/// Load a STFSC animated model from bytes
/// 
/// File format (.sanim):
/// ```text
/// [4 bytes] Magic: "SANM"
/// [4 bytes] Version (u32 LE)
/// [4 bytes] Skeleton bone count (u32 LE)
/// [N * bone_data] Bones
/// [4 bytes] Mesh count (u32 LE)  
/// [N * mesh_data] Skinned meshes
/// [4 bytes] Animation count (u32 LE)
/// [N * anim_data] Animation clips
/// ```
pub fn load_animated_model_from_bytes(data: &[u8]) -> Result<ModelScene> {
    let mut cursor = Cursor::new(data);
    
    // Read and verify magic
    let mut magic = [0u8; 4];
    cursor.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(anyhow!("Invalid SANIM file: bad magic bytes"));
    }
    
    // Read version
    let version = read_u32(&mut cursor)?;
    if version > VERSION {
        return Err(anyhow!("SANIM version {} is newer than supported ({})", version, VERSION));
    }
    
    let mut scene = ModelScene::new();
    
    // Read skeleton
    let bone_count = read_u32(&mut cursor)? as usize;
    if bone_count > 0 {
        let mut skeleton = Skeleton::new();
        for _ in 0..bone_count {
            let name = read_string(&mut cursor)?;
            let parent_index = read_i32(&mut cursor)?;
            let parent = if parent_index >= 0 { Some(parent_index as usize) } else { None };
            let inverse_bind = read_mat4(&mut cursor)?;
            let local_transform = read_mat4(&mut cursor)?;
            skeleton.add_bone(name, parent, inverse_bind, local_transform);
        }
        scene.skeleton = Some(skeleton);
    }
    
    // Read skinned meshes
    let mesh_count = read_u32(&mut cursor)? as usize;
    for _ in 0..mesh_count {
        let skinned = read_skinned_mesh(&mut cursor)?;
        scene.skinned_meshes.push(skinned);
    }
    
    // Read animations
    let anim_count = read_u32(&mut cursor)? as usize;
    for _ in 0..anim_count {
        let clip = read_animation_clip(&mut cursor)?;
        scene.animations.push(clip);
    }
    
    Ok(scene)
}

/// Create a sample animated model for testing
pub fn create_test_animated_model() -> ModelScene {
    let mut scene = ModelScene::new();
    
    // Create a simple skeleton with 3 bones: root -> spine -> head
    let mut skeleton = Skeleton::new();
    skeleton.add_bone(
        "root".to_string(),
        None,
        Mat4::IDENTITY,
        Mat4::IDENTITY,
    );
    skeleton.add_bone(
        "spine".to_string(),
        Some(0),
        Mat4::from_translation(Vec3::new(0.0, -0.5, 0.0)),
        Mat4::from_translation(Vec3::new(0.0, 0.5, 0.0)),
    );
    skeleton.add_bone(
        "head".to_string(),
        Some(1),
        Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0)),
        Mat4::from_translation(Vec3::new(0.0, 0.5, 0.0)),
    );
    scene.skeleton = Some(skeleton);
    
    // Create a simple "idle bob" animation
    let mut clip = AnimationClip::new("idle", 2.0);
    
    // Root bone: slight up/down motion
    clip.channels.push(BoneKeyframes {
        bone_index: 0,
        position_keys: vec![
            (0.0, Vec3::ZERO),
            (1.0, Vec3::new(0.0, 0.1, 0.0)),
            (2.0, Vec3::ZERO),
        ],
        rotation_keys: vec![
            (0.0, Quat::IDENTITY),
        ],
        scale_keys: vec![
            (0.0, Vec3::ONE),
        ],
    });
    
    // Spine: slight lean
    clip.channels.push(BoneKeyframes {
        bone_index: 1,
        position_keys: vec![
            (0.0, Vec3::new(0.0, 0.5, 0.0)),
        ],
        rotation_keys: vec![
            (0.0, Quat::IDENTITY),
            (0.5, Quat::from_rotation_x(0.05)),
            (1.5, Quat::from_rotation_x(-0.05)),
            (2.0, Quat::IDENTITY),
        ],
        scale_keys: vec![
            (0.0, Vec3::ONE),
        ],
    });
    
    // Head: nod
    clip.channels.push(BoneKeyframes {
        bone_index: 2,
        position_keys: vec![
            (0.0, Vec3::new(0.0, 0.5, 0.0)),
        ],
        rotation_keys: vec![
            (0.0, Quat::IDENTITY),
            (0.75, Quat::from_rotation_x(-0.1)),
            (1.25, Quat::from_rotation_x(0.1)),
            (2.0, Quat::IDENTITY),
        ],
        scale_keys: vec![
            (0.0, Vec3::ONE),
        ],
    });
    
    scene.animations.push(clip);
    
    scene
}

/// Export a ModelScene to SANIM binary format
pub fn export_to_sanim(scene: &ModelScene) -> Vec<u8> {
    let mut data = Vec::new();
    
    // Write magic and version
    data.extend_from_slice(&MAGIC);
    write_u32(&mut data, VERSION);
    
    // Write skeleton
    if let Some(ref skeleton) = scene.skeleton {
        write_u32(&mut data, skeleton.bones.len() as u32);
        for bone in &skeleton.bones {
            write_string(&mut data, &bone.name);
            write_i32(&mut data, bone.parent_index.map(|i| i as i32).unwrap_or(-1));
            write_mat4(&mut data, &bone.inverse_bind_matrix);
            write_mat4(&mut data, &bone.local_transform);
        }
    } else {
        write_u32(&mut data, 0);
    }
    
    // Write skinned meshes
    write_u32(&mut data, scene.skinned_meshes.len() as u32);
    for skinned in &scene.skinned_meshes {
        write_skinned_mesh(&mut data, skinned);
    }
    
    // Write animations
    write_u32(&mut data, scene.animations.len() as u32);
    for clip in &scene.animations {
        write_animation_clip(&mut data, clip);
    }
    
    data
}

// === Binary read helpers ===

fn read_u32(cursor: &mut Cursor<&[u8]>) -> Result<u32> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_i32(cursor: &mut Cursor<&[u8]>) -> Result<i32> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

fn read_f32(cursor: &mut Cursor<&[u8]>) -> Result<f32> {
    let mut buf = [0u8; 4];
    cursor.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_string(cursor: &mut Cursor<&[u8]>) -> Result<String> {
    let len = read_u32(cursor)? as usize;
    let mut buf = vec![0u8; len];
    cursor.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| anyhow!("Invalid UTF-8 string: {}", e))
}

fn read_vec3(cursor: &mut Cursor<&[u8]>) -> Result<Vec3> {
    Ok(Vec3::new(
        read_f32(cursor)?,
        read_f32(cursor)?,
        read_f32(cursor)?,
    ))
}

fn read_quat(cursor: &mut Cursor<&[u8]>) -> Result<Quat> {
    Ok(Quat::from_xyzw(
        read_f32(cursor)?,
        read_f32(cursor)?,
        read_f32(cursor)?,
        read_f32(cursor)?,
    ))
}

fn read_mat4(cursor: &mut Cursor<&[u8]>) -> Result<Mat4> {
    let mut cols = [[0.0f32; 4]; 4];
    for col in &mut cols {
        for val in col {
            *val = read_f32(cursor)?;
        }
    }
    Ok(Mat4::from_cols_array_2d(&cols))
}

fn read_skinned_mesh(cursor: &mut Cursor<&[u8]>) -> Result<SkinnedMesh> {
    // Read vertex count
    let vertex_count = read_u32(cursor)? as usize;
    let mut vertices = Vec::with_capacity(vertex_count);
    let mut bone_indices = Vec::with_capacity(vertex_count);
    let mut bone_weights = Vec::with_capacity(vertex_count);
    
    for _ in 0..vertex_count {
        let position = [read_f32(cursor)?, read_f32(cursor)?, read_f32(cursor)?];
        let normal = [read_f32(cursor)?, read_f32(cursor)?, read_f32(cursor)?];
        let uv = [read_f32(cursor)?, read_f32(cursor)?];
        let bi = [read_u32(cursor)?, read_u32(cursor)?, read_u32(cursor)?, read_u32(cursor)?];
        let bw = [read_f32(cursor)?, read_f32(cursor)?, read_f32(cursor)?, read_f32(cursor)?];
        
        vertices.push(Vertex {
            position,
            normal,
            uv,
            color: [1.0, 1.0, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            bone_indices: bi,
            bone_weights: bw,
        });
        bone_indices.push(bi);
        bone_weights.push(bw);
    }
    
    // Read indices
    let index_count = read_u32(cursor)? as usize;
    let mut indices = Vec::with_capacity(index_count);
    for _ in 0..index_count {
        indices.push(read_u32(cursor)?);
    }
    
    // Compute AABB
    let (aabb_min, aabb_max) = compute_aabb(&vertices);
    
    Ok(SkinnedMesh {
        mesh: Mesh {
            vertices,
            indices,
            albedo: None,
            normal: None,
            metallic_roughness: None,
            albedo_texture: None,
            aabb_min,
            aabb_max,
            decoded_albedo: None,
            decoded_normal: None,
            decoded_mr: None,
        },
        bone_indices,
        bone_weights,
    })
}

fn read_animation_clip(cursor: &mut Cursor<&[u8]>) -> Result<AnimationClip> {
    let name = read_string(cursor)?;
    let duration = read_f32(cursor)?;
    let channel_count = read_u32(cursor)? as usize;
    
    let mut channels = Vec::with_capacity(channel_count);
    for _ in 0..channel_count {
        let bone_index = read_u32(cursor)? as usize;
        
        // Position keys
        let pos_count = read_u32(cursor)? as usize;
        let mut position_keys = Vec::with_capacity(pos_count);
        for _ in 0..pos_count {
            let time = read_f32(cursor)?;
            let value = read_vec3(cursor)?;
            position_keys.push((time, value));
        }
        
        // Rotation keys
        let rot_count = read_u32(cursor)? as usize;
        let mut rotation_keys = Vec::with_capacity(rot_count);
        for _ in 0..rot_count {
            let time = read_f32(cursor)?;
            let value = read_quat(cursor)?;
            rotation_keys.push((time, value));
        }
        
        // Scale keys
        let scale_count = read_u32(cursor)? as usize;
        let mut scale_keys = Vec::with_capacity(scale_count);
        for _ in 0..scale_count {
            let time = read_f32(cursor)?;
            let value = read_vec3(cursor)?;
            scale_keys.push((time, value));
        }
        
        channels.push(BoneKeyframes {
            bone_index,
            position_keys,
            rotation_keys,
            scale_keys,
        });
    }
    
    Ok(AnimationClip {
        name,
        duration,
        channels,
    })
}

// === Binary write helpers ===

fn write_u32(data: &mut Vec<u8>, val: u32) {
    data.extend_from_slice(&val.to_le_bytes());
}

fn write_i32(data: &mut Vec<u8>, val: i32) {
    data.extend_from_slice(&val.to_le_bytes());
}

fn write_f32(data: &mut Vec<u8>, val: f32) {
    data.extend_from_slice(&val.to_le_bytes());
}

fn write_string(data: &mut Vec<u8>, s: &str) {
    write_u32(data, s.len() as u32);
    data.extend_from_slice(s.as_bytes());
}

fn write_vec3(data: &mut Vec<u8>, v: &Vec3) {
    write_f32(data, v.x);
    write_f32(data, v.y);
    write_f32(data, v.z);
}

fn write_quat(data: &mut Vec<u8>, q: &Quat) {
    write_f32(data, q.x);
    write_f32(data, q.y);
    write_f32(data, q.z);
    write_f32(data, q.w);
}

fn write_mat4(data: &mut Vec<u8>, m: &Mat4) {
    for col in m.to_cols_array_2d() {
        for val in col {
            write_f32(data, val);
        }
    }
}

fn write_skinned_mesh(data: &mut Vec<u8>, skinned: &SkinnedMesh) {
    write_u32(data, skinned.mesh.vertices.len() as u32);
    
    for (i, v) in skinned.mesh.vertices.iter().enumerate() {
        // Position
        write_f32(data, v.position[0]);
        write_f32(data, v.position[1]);
        write_f32(data, v.position[2]);
        // Normal
        write_f32(data, v.normal[0]);
        write_f32(data, v.normal[1]);
        write_f32(data, v.normal[2]);
        // UV
        write_f32(data, v.uv[0]);
        write_f32(data, v.uv[1]);
        // Bone indices
        let bi = if i < skinned.bone_indices.len() { skinned.bone_indices[i] } else { [0; 4] };
        write_u32(data, bi[0]);
        write_u32(data, bi[1]);
        write_u32(data, bi[2]);
        write_u32(data, bi[3]);
        // Bone weights
        let bw = if i < skinned.bone_weights.len() { skinned.bone_weights[i] } else { [1.0, 0.0, 0.0, 0.0] };
        write_f32(data, bw[0]);
        write_f32(data, bw[1]);
        write_f32(data, bw[2]);
        write_f32(data, bw[3]);
    }
    
    write_u32(data, skinned.mesh.indices.len() as u32);
    for idx in &skinned.mesh.indices {
        write_u32(data, *idx);
    }
}

fn write_animation_clip(data: &mut Vec<u8>, clip: &AnimationClip) {
    write_string(data, &clip.name);
    write_f32(data, clip.duration);
    write_u32(data, clip.channels.len() as u32);
    
    for channel in &clip.channels {
        write_u32(data, channel.bone_index as u32);
        
        // Position keys
        write_u32(data, channel.position_keys.len() as u32);
        for (time, val) in &channel.position_keys {
            write_f32(data, *time);
            write_vec3(data, val);
        }
        
        // Rotation keys
        write_u32(data, channel.rotation_keys.len() as u32);
        for (time, val) in &channel.rotation_keys {
            write_f32(data, *time);
            write_quat(data, val);
        }
        
        // Scale keys
        write_u32(data, channel.scale_keys.len() as u32);
        for (time, val) in &channel.scale_keys {
            write_f32(data, *time);
            write_vec3(data, val);
        }
    }
}

fn compute_aabb(vertices: &[Vertex]) -> ([f32; 3], [f32; 3]) {
    if vertices.is_empty() {
        return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    }
    
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    
    for v in vertices {
        let p = Vec3::from(v.position);
        min = min.min(p);
        max = max.max(p);
    }
    
    (min.to_array(), max.to_array())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_sanim() {
        // Create test model
        let original = create_test_animated_model();
        
        // Export to bytes
        let data = export_to_sanim(&original);
        
        // Import back
        let loaded = load_animated_model_from_bytes(&data).unwrap();
        
        // Verify skeleton
        assert!(loaded.skeleton.is_some());
        let skel = loaded.skeleton.as_ref().unwrap();
        assert_eq!(skel.bones.len(), 3);
        assert_eq!(skel.bones[0].name, "root");
        assert_eq!(skel.bones[1].name, "spine");
        assert_eq!(skel.bones[2].name, "head");
        
        // Verify animations
        assert_eq!(loaded.animations.len(), 1);
        assert_eq!(loaded.animations[0].name, "idle");
        assert!((loaded.animations[0].duration - 2.0).abs() < 0.001);
    }
    
    #[test]
    fn test_binary_helpers() {
        // Test u32
        let mut data = Vec::new();
        write_u32(&mut data, 12345);
        let mut cursor = Cursor::new(data.as_slice());
        assert_eq!(read_u32(&mut cursor).unwrap(), 12345);
        
        // Test string
        let mut data = Vec::new();
        write_string(&mut data, "hello_bone");
        let mut cursor = Cursor::new(data.as_slice());
        assert_eq!(read_string(&mut cursor).unwrap(), "hello_bone");
    }
}
