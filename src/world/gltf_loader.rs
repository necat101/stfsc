//! Custom Animation Model Loader for STFSC Engine
//!
//! Provides skeletal animation model loading via:
//! - glTF 2.0 format (.gltf, .glb) with full animation support
//! - Custom binary format (.sanim) for optimized loading
//! - OBJ conversion for static meshes
//!
//! # Workflow for Animated Models (Recommended: glTF)
//! 1. Create your animated model in Blender with skeleton + animations
//! 2. Export as glTF 2.0 Binary (.glb) - best compatibility
//! 3. Load with `load_gltf_with_animations()` in the engine
//!
//! # Alternative: SANIM Format
//! For optimized loading, use the STFSC Blender addon to export .sanim files.

use crate::world::{Mesh, Vertex};
use crate::world::fbx_loader::{
    AnimationClip, BoneKeyframes, ModelScene, Skeleton, SkinnedMesh, EmbeddedTexture,
};
use anyhow::{anyhow, Context, Result};
use glam::{Mat4, Quat, Vec3};
use std::io::{Read, Cursor};
use std::collections::HashMap;

/// Magic bytes for STFSC animation format: "SANM"
const MAGIC: [u8; 4] = [0x53, 0x41, 0x4E, 0x4D];
/// Current format version
const VERSION: u32 = 1;

// ============================================================================
// glTF 2.0 Loader with Animation Support (Manual Parser using serde_json)
// ============================================================================

use serde_json::Value;

/// Load a glTF/GLB file with full skeleton and animation extraction
/// 
/// This is the recommended way to import animated models.
/// Supports .glb (binary, all-in-one) format.
pub fn load_gltf_with_animations(data: &[u8]) -> Result<ModelScene> {
    // Detect if this is GLB (binary) or JSON glTF
    let (json_data, bin_data) = if data.len() >= 12 && &data[0..4] == b"glTF" {
        // GLB format
        parse_glb(data)?
    } else {
        // Plain JSON glTF
        (String::from_utf8_lossy(data).to_string(), Vec::new())
    };
    
    let gltf: Value = serde_json::from_str(&json_data)
        .context("Failed to parse glTF JSON")?;
    
    let mut scene = ModelScene::new();
    let mut joint_to_bone_index: HashMap<usize, usize> = HashMap::new();
    
    // Load buffers
    let buffers = load_gltf_buffers_manual(&gltf, &bin_data)?;
    
    // Extract skeleton from skins
    if let Some(skins) = gltf.get("skins").and_then(|s| s.as_array()) {
        if let Some(skin) = skins.first() {
            let mut skeleton = Skeleton::new();
            
            // Get joints array
            if let Some(joints) = skin.get("joints").and_then(|j| j.as_array()) {
                // Get inverse bind matrices if available
                let inverse_bind_matrices = if let Some(accessor_idx) = skin.get("inverseBindMatrices").and_then(|i| i.as_u64()) {
                    read_accessor_mat4_manual(&gltf, accessor_idx as usize, &buffers)?
                } else {
                    vec![Mat4::IDENTITY; joints.len()]
                };
                
                // Add each joint as a bone
                for (i, joint_idx) in joints.iter().enumerate() {
                    let joint_idx = joint_idx.as_u64().unwrap_or(0) as usize;
                    
                    // Get node for this joint
                    let nodes = gltf.get("nodes").and_then(|n| n.as_array());
                    let node = nodes.and_then(|n| n.get(joint_idx));
                    
                    let name = node
                        .and_then(|n| n.get("name"))
                        .and_then(|n| n.as_str())
                        .unwrap_or(&format!("bone_{}", joint_idx))
                        .to_string();
                    
                    // Get local transform from node
                    let local_transform = get_node_transform(node);
                    
                    // Find parent
                    let parent = find_joint_parent_manual(&gltf, joint_idx, &joint_to_bone_index);
                    
                    let inverse_bind = inverse_bind_matrices.get(i).copied().unwrap_or(Mat4::IDENTITY);
                    
                    let bone_index = skeleton.add_bone(name, parent, inverse_bind, local_transform);
                    joint_to_bone_index.insert(joint_idx, bone_index);
                }
            }
            
            scene.skeleton = Some(skeleton);
        }
    }
    
    // Extract animations
    if let Some(animations) = gltf.get("animations").and_then(|a| a.as_array()) {
        for animation in animations {
            if let Ok(clip) = extract_animation_manual(&gltf, animation, &buffers, &joint_to_bone_index) {
                scene.animations.push(clip);
            }
        }
    }
    
    // NEW: Extract images and materials
    let texture_data = extract_images_manual(&gltf, &buffers);
    scene.textures = texture_data.clone();
    let texture_names: Vec<String> = texture_data.iter().map(|t| t.name.clone()).collect();
    
    let materials = extract_materials_manual(&gltf);
    let tex_img_indices = extract_texture_source_indices(&gltf);
    
    // Re-extract meshes with material info via scene hierarchy traversal
    scene.meshes.clear();
    scene.skinned_meshes.clear();
    
    let scene_idx = gltf.get("scene").and_then(|s| s.as_u64()).unwrap_or(0) as usize;
    let mut root_nodes = Vec::new();
    
    if let Some(scenes) = gltf.get("scenes").and_then(|s| s.as_array()) {
        if let Some(default_scene) = scenes.get(scene_idx) {
            if let Some(roots) = default_scene.get("nodes").and_then(|r| r.as_array()) {
                root_nodes = roots.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect();
            }
        }
    }
    
    // If no scene roots found, try to find orphan nodes (nodes with no parents)
    if root_nodes.is_empty() {
        if let Some(nodes) = gltf.get("nodes").and_then(|n| n.as_array()) {
            for i in 0..nodes.len() {
                let mut is_child = false;
                for other_node in nodes {
                    if let Some(children) = other_node.get("children").and_then(|c| c.as_array()) {
                        if children.iter().any(|c| c.as_u64() == Some(i as u64)) {
                            is_child = true;
                            break;
                        }
                    }
                }
                if !is_child {
                    root_nodes.push(i);
                }
            }
        }
    }

    for root_idx in root_nodes {
        traverse_gltf_nodes(
            &gltf,
            root_idx,
            Mat4::IDENTITY,
            &buffers,
            &texture_names,
            &materials,
            &tex_img_indices,
            &mut scene,
        )?;
    }
    
    log::info!(
        "Loaded glTF: {} meshes, {} skinned meshes, {} bones, {} animations, {} textures",
        scene.meshes.len(),
        scene.skinned_meshes.len(),
        scene.skeleton.as_ref().map(|s| s.bones.len()).unwrap_or(0),
        scene.animations.len(),
        scene.textures.len()
    );
    
    Ok(scene)
}

/// Load glTF from file path
pub fn load_gltf_file(path: &str) -> Result<ModelScene> {
    let data = std::fs::read(path)
        .with_context(|| format!("Failed to read glTF file: {}", path))?;
    load_gltf_with_animations(&data)
}

fn traverse_gltf_nodes(
    gltf: &Value,
    node_idx: usize,
    parent_transform: Mat4,
    buffers: &[Vec<u8>],
    texture_names: &[String],
    materials: &[MaterialData],
    tex_img_indices: &[usize],
    scene: &mut ModelScene,
) -> Result<()> {
    let nodes = gltf.get("nodes").and_then(|n| n.as_array()).ok_or_else(|| anyhow!("No nodes"))?;
    let node = nodes.get(node_idx).ok_or_else(|| anyhow!("Node not found"))?;
    
    let local_transform = get_node_transform(Some(node));
    let world_transform = parent_transform * local_transform;
    
    if let Some(mesh_idx) = node.get("mesh").and_then(|m| m.as_u64()) {
        if let Some(meshes) = gltf.get("meshes").and_then(|m| m.as_array()) {
            if let Some(mesh) = meshes.get(mesh_idx as usize) {
                if let Some(primitives) = mesh.get("primitives").and_then(|p| p.as_array()) {
                    for primitive in primitives {
                        if let Ok(mut mesh_data) = extract_primitive_mesh_manual(gltf, primitive, buffers, texture_names, materials, tex_img_indices) {
                            // Check for skinning data
                            let attributes = primitive.get("attributes");
                            let has_joints = attributes.and_then(|a| a.get("JOINTS_0")).is_some();
                            let has_weights = attributes.and_then(|a| a.get("WEIGHTS_0")).is_some();
                            
                            if has_joints && has_weights {
                                if let Ok((bone_indices, bone_weights)) = extract_skin_data_manual(gltf, primitive, buffers) {
                                    // Populate Vertex data for high-res editor rendering
                                    for (i, v) in mesh_data.vertices.iter_mut().enumerate() {
                                        if let Some(bi) = bone_indices.get(i) { v.bone_indices = *bi; }
                                        if let Some(bw) = bone_weights.get(i) { v.bone_weights = *bw; }
                                    }
                                    scene.skinned_meshes.push(SkinnedMesh {
                                        mesh: mesh_data,
                                        bone_indices,
                                        bone_weights,
                                    });
                                }
                            } else {
                                // Static mesh - bake node transform for editor viewport
                                bake_mesh_transform(&mut mesh_data, world_transform);
                                scene.meshes.push(mesh_data);
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Recurse to children
    if let Some(children) = node.get("children").and_then(|c| c.as_array()) {
        for child_val in children {
            if let Some(child_idx) = child_val.as_u64() {
                traverse_gltf_nodes(gltf, child_idx as usize, world_transform, buffers, texture_names, materials, tex_img_indices, scene)?;
            }
        }
    }
    
    Ok(())
}

fn bake_mesh_transform(mesh: &mut Mesh, transform: Mat4) {
    for v in &mut mesh.vertices {
        let pos = Vec3::from(v.position);
        let norm = Vec3::from(v.normal);
        
        v.position = transform.transform_point3(pos).to_array();
        // Transform normal using the transform matrix (assuming uniform scale for simplicity)
        v.normal = transform.transform_vector3(norm).normalize().to_array();
    }
    
    let (min, max) = compute_aabb(&mesh.vertices);
    mesh.aabb_min = min;
    mesh.aabb_max = max;
}

// --- GLB Parser ---

fn parse_glb(data: &[u8]) -> Result<(String, Vec<u8>)> {
    if data.len() < 12 {
        return Err(anyhow!("GLB file too short"));
    }
    
    // GLB Header: magic (4) + version (4) + length (4)
    let _version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let _total_length = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    
    // JSON chunk: length (4) + type (4) + data
    let json_length = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let _json_type = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
    
    let json_start = 20;
    let json_end = json_start + json_length;
    
    if json_end > data.len() {
        return Err(anyhow!("Invalid GLB: JSON chunk extends past file"));
    }
    
    let json_data = String::from_utf8_lossy(&data[json_start..json_end]).to_string();
    
    // BIN chunk (optional)
    let mut bin_data = Vec::new();
    if json_end + 8 <= data.len() {
        let bin_length = u32::from_le_bytes([data[json_end], data[json_end+1], data[json_end+2], data[json_end+3]]) as usize;
        let bin_start = json_end + 8;
        let bin_end = bin_start + bin_length;
        
        if bin_end <= data.len() {
            bin_data = data[bin_start..bin_end].to_vec();
        }
    }
    
    Ok((json_data, bin_data))
}

fn load_gltf_buffers_manual(gltf: &Value, embedded_bin: &[u8]) -> Result<Vec<Vec<u8>>> {
    let mut buffers = Vec::new();
    
    if let Some(buffer_list) = gltf.get("buffers").and_then(|b| b.as_array()) {
        for buffer in buffer_list {
            if let Some(uri) = buffer.get("uri").and_then(|u| u.as_str()) {
                // Data URI or external file
                if uri.starts_with("data:") {
                    // Base64 data URI
                    if let Some(comma_pos) = uri.find(',') {
                        let base64_data = &uri[comma_pos + 1..];
                        // Simple base64 decode using standard alphabet
                        match decode_base64(base64_data) {
                            Ok(decoded) => buffers.push(decoded),
                            Err(_) => buffers.push(Vec::new()),
                        }
                    } else {
                        buffers.push(Vec::new());
                    }
                } else {
                    // External file - try to load
                    match std::fs::read(uri) {
                        Ok(data) => buffers.push(data),
                        Err(_) => buffers.push(Vec::new()),
                    }
                }
            } else {
                // No URI = embedded GLB buffer
                buffers.push(embedded_bin.to_vec());
            }
        }
    }
    
    // If no buffers defined but we have embedded data, use it
    if buffers.is_empty() && !embedded_bin.is_empty() {
        buffers.push(embedded_bin.to_vec());
    }
    
    Ok(buffers)
}

fn extract_images_manual(gltf: &Value, buffers: &[Vec<u8>]) -> Vec<EmbeddedTexture> {
    let mut textures = Vec::new();
    
    if let Some(image_list) = gltf.get("images").and_then(|i| i.as_array()) {
        for (idx, image) in image_list.iter().enumerate() {
            let name = image.get("name").and_then(|n| n.as_str())
                .unwrap_or(&format!("texture_{}", idx)).to_string();
            
            let mut data = Vec::new();
            
            if let Some(buffer_view_idx) = image.get("bufferView").and_then(|bv| bv.as_u64()) {
                // Load from bufferView
                if let Some(buffer_views) = gltf.get("bufferViews").and_then(|bv| bv.as_array()) {
                    if let Some(view) = buffer_views.get(buffer_view_idx as usize) {
                        let buffer_idx = view.get("buffer").and_then(|b| b.as_u64()).unwrap_or(0) as usize;
                        let offset = view.get("byteOffset").and_then(|o| o.as_u64()).unwrap_or(0) as usize;
                        let length = view.get("byteLength").and_then(|l| l.as_u64()).unwrap_or(0) as usize;
                        
                        if let Some(buffer) = buffers.get(buffer_idx) {
                            if offset + length <= buffer.len() {
                                data = buffer[offset..offset+length].to_vec();
                            }
                        }
                    }
                }
            } else if let Some(uri) = image.get("uri").and_then(|u| u.as_str()) {
                if uri.starts_with("data:") {
                    if let Some(comma_pos) = uri.find(',') {
                        if let Ok(decoded) = decode_base64(&uri[comma_pos+1..]) {
                            data = decoded;
                        }
                    }
                }
            }
            
            if !data.is_empty() {
                textures.push(EmbeddedTexture {
                    name,
                    width: 0, // Placeholder
                    height: 0, // Placeholder
                    data,
                });
            }
        }
    }
    
    textures
}

/// Material data extracted from glTF
struct MaterialData {
    texture_idx: Option<usize>,
    base_color: [f32; 4],
}

fn extract_materials_manual(gltf: &Value) -> Vec<MaterialData> {
    let mut materials = Vec::new();
    
    if let Some(material_list) = gltf.get("materials").and_then(|m| m.as_array()) {
        for material in material_list {
            let pbr = material.get("pbrMetallicRoughness");
            
            let tex_idx = pbr
                .and_then(|p| p.get("baseColorTexture"))
                .and_then(|t| t.get("index"))
                .and_then(|i| i.as_u64())
                .map(|i| i as usize);
            
            // Extract baseColorFactor (default is [1.0, 1.0, 1.0, 1.0])
            let base_color = if let Some(factor) = pbr.and_then(|p| p.get("baseColorFactor")).and_then(|f| f.as_array()) {
                [
                    factor.get(0).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                    factor.get(1).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                    factor.get(2).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                    factor.get(3).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                ]
            } else {
                [1.0, 1.0, 1.0, 1.0]
            };
            
            materials.push(MaterialData { texture_idx: tex_idx, base_color });
        }
    }
    
    materials
}

fn extract_texture_source_indices(gltf: &Value) -> Vec<usize> {
    let mut image_indices = Vec::new();
    
    if let Some(texture_list) = gltf.get("textures").and_then(|t| t.as_array()) {
        for texture in texture_list {
            let image_idx = texture.get("source").and_then(|s| s.as_u64()).unwrap_or(0) as usize;
            image_indices.push(image_idx);
        }
    }
    
    image_indices
}

fn decode_base64(input: &str) -> Result<Vec<u8>> {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    
    let input = input.as_bytes();
    let mut output = Vec::new();
    let mut buffer = 0u32;
    let mut bits = 0;
    
    for &byte in input {
        if byte == b'=' || byte == b'\n' || byte == b'\r' || byte == b' ' {
            continue;
        }
        
        let value = ALPHABET.iter().position(|&c| c == byte)
            .ok_or_else(|| anyhow!("Invalid base64 character"))? as u32;
        
        buffer = (buffer << 6) | value;
        bits += 6;
        
        if bits >= 8 {
            bits -= 8;
            output.push((buffer >> bits) as u8);
            buffer &= (1 << bits) - 1;
        }
    }
    
    Ok(output)
}

fn get_node_transform(node: Option<&Value>) -> Mat4 {
    let node = match node {
        Some(n) => n,
        None => return Mat4::IDENTITY,
    };
    
    // Check for matrix
    if let Some(matrix) = node.get("matrix").and_then(|m| m.as_array()) {
        if matrix.len() == 16 {
            let vals: Vec<f32> = matrix.iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();
            if vals.len() == 16 {
                return Mat4::from_cols_array(&[
                    vals[0], vals[1], vals[2], vals[3],
                    vals[4], vals[5], vals[6], vals[7],
                    vals[8], vals[9], vals[10], vals[11],
                    vals[12], vals[13], vals[14], vals[15],
                ]);
            }
        }
    }
    
    // Otherwise use TRS
    let translation = node.get("translation")
        .and_then(|t| t.as_array())
        .map(|a| Vec3::new(
            a.get(0).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
        ))
        .unwrap_or(Vec3::ZERO);
    
    let rotation = node.get("rotation")
        .and_then(|r| r.as_array())
        .map(|a| Quat::from_xyzw(
            a.get(0).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            a.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            a.get(2).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32,
            a.get(3).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
        ))
        .unwrap_or(Quat::IDENTITY);
    
    let scale = node.get("scale")
        .and_then(|s| s.as_array())
        .map(|a| Vec3::new(
            a.get(0).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
            a.get(1).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
            a.get(2).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
        ))
        .unwrap_or(Vec3::ONE);
    
    Mat4::from_scale_rotation_translation(scale, rotation, translation)
}

fn find_joint_parent_manual(gltf: &Value, joint_idx: usize, joint_map: &HashMap<usize, usize>) -> Option<usize> {
    if let Some(nodes) = gltf.get("nodes").and_then(|n| n.as_array()) {
        for (node_idx, node) in nodes.iter().enumerate() {
            if let Some(children) = node.get("children").and_then(|c| c.as_array()) {
                for child in children {
                    if child.as_u64() == Some(joint_idx as u64) {
                        return joint_map.get(&node_idx).copied();
                    }
                }
            }
        }
    }
    None
}

fn get_accessor_info(gltf: &Value, accessor_idx: usize) -> Option<(usize, usize, usize, usize, u64)> {
    let accessors = gltf.get("accessors")?.as_array()?;
    let accessor = accessors.get(accessor_idx)?;
    
    let buffer_view_idx = accessor.get("bufferView")?.as_u64()? as usize;
    let count = accessor.get("count")?.as_u64()? as usize;
    let component_type = accessor.get("componentType")?.as_u64()?;
    let accessor_offset = accessor.get("byteOffset").and_then(|o| o.as_u64()).unwrap_or(0) as usize;
    
    let buffer_views = gltf.get("bufferViews")?.as_array()?;
    let view = buffer_views.get(buffer_view_idx)?;
    
    let buffer_idx = view.get("buffer")?.as_u64()? as usize;
    let view_offset = view.get("byteOffset").and_then(|o| o.as_u64()).unwrap_or(0) as usize;
    let stride = view.get("byteStride").and_then(|s| s.as_u64()).unwrap_or(0) as usize;
    
    Some((buffer_idx, view_offset + accessor_offset, count, stride, component_type))
}

fn read_accessor_mat4_manual(gltf: &Value, accessor_idx: usize, buffers: &[Vec<u8>]) -> Result<Vec<Mat4>> {
    let (buffer_idx, offset, count, stride, _) = get_accessor_info(gltf, accessor_idx)
        .ok_or_else(|| anyhow!("Invalid accessor"))?;
    
    let buffer = buffers.get(buffer_idx).ok_or_else(|| anyhow!("Buffer not found"))?;
    let actual_stride = if stride > 0 { stride } else { 64 };
    
    let mut matrices = Vec::with_capacity(count);
    for i in 0..count {
        let start = offset + i * actual_stride;
        if start + 64 > buffer.len() { break; }
        
        let mut vals = [0.0f32; 16];
        for j in 0..16 {
            let byte_offset = start + j * 4;
            vals[j] = f32::from_le_bytes([
                buffer[byte_offset], buffer[byte_offset+1],
                buffer[byte_offset+2], buffer[byte_offset+3],
            ]);
        }
        matrices.push(Mat4::from_cols_array(&vals));
    }
    
    Ok(matrices)
}

fn read_accessor_vec3_manual(gltf: &Value, accessor_idx: usize, buffers: &[Vec<u8>]) -> Result<Vec<Vec3>> {
    let (buffer_idx, offset, count, stride, _) = get_accessor_info(gltf, accessor_idx)
        .ok_or_else(|| anyhow!("Invalid accessor"))?;
    
    let buffer = buffers.get(buffer_idx).ok_or_else(|| anyhow!("Buffer not found"))?;
    let actual_stride = if stride > 0 { stride } else { 12 };
    
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let start = offset + i * actual_stride;
        if start + 12 > buffer.len() { break; }
        values.push(Vec3::new(
            f32::from_le_bytes([buffer[start], buffer[start+1], buffer[start+2], buffer[start+3]]),
            f32::from_le_bytes([buffer[start+4], buffer[start+5], buffer[start+6], buffer[start+7]]),
            f32::from_le_bytes([buffer[start+8], buffer[start+9], buffer[start+10], buffer[start+11]]),
        ));
    }
    Ok(values)
}

fn read_accessor_vec2_manual(gltf: &Value, accessor_idx: usize, buffers: &[Vec<u8>]) -> Result<Vec<[f32; 2]>> {
    let (buffer_idx, offset, count, stride, _) = get_accessor_info(gltf, accessor_idx)
        .ok_or_else(|| anyhow!("Invalid accessor"))?;
    
    let buffer = buffers.get(buffer_idx).ok_or_else(|| anyhow!("Buffer not found"))?;
    let actual_stride = if stride > 0 { stride } else { 8 };
    
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let start = offset + i * actual_stride;
        if start + 8 > buffer.len() { break; }
        values.push([
            f32::from_le_bytes([buffer[start], buffer[start+1], buffer[start+2], buffer[start+3]]),
            f32::from_le_bytes([buffer[start+4], buffer[start+5], buffer[start+6], buffer[start+7]]),
        ]);
    }
    Ok(values)
}

fn read_accessor_quat_manual(gltf: &Value, accessor_idx: usize, buffers: &[Vec<u8>]) -> Result<Vec<Quat>> {
    let (buffer_idx, offset, count, stride, _) = get_accessor_info(gltf, accessor_idx)
        .ok_or_else(|| anyhow!("Invalid accessor"))?;
    
    let buffer = buffers.get(buffer_idx).ok_or_else(|| anyhow!("Buffer not found"))?;
    let actual_stride = if stride > 0 { stride } else { 16 };
    
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let start = offset + i * actual_stride;
        if start + 16 > buffer.len() { break; }
        values.push(Quat::from_xyzw(
            f32::from_le_bytes([buffer[start], buffer[start+1], buffer[start+2], buffer[start+3]]),
            f32::from_le_bytes([buffer[start+4], buffer[start+5], buffer[start+6], buffer[start+7]]),
            f32::from_le_bytes([buffer[start+8], buffer[start+9], buffer[start+10], buffer[start+11]]),
            f32::from_le_bytes([buffer[start+12], buffer[start+13], buffer[start+14], buffer[start+15]]),
        ));
    }
    Ok(values)
}

fn read_accessor_f32_manual(gltf: &Value, accessor_idx: usize, buffers: &[Vec<u8>]) -> Result<Vec<f32>> {
    let (buffer_idx, offset, count, stride, _) = get_accessor_info(gltf, accessor_idx)
        .ok_or_else(|| anyhow!("Invalid accessor"))?;
    
    let buffer = buffers.get(buffer_idx).ok_or_else(|| anyhow!("Buffer not found"))?;
    let actual_stride = if stride > 0 { stride } else { 4 };
    
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let start = offset + i * actual_stride;
        if start + 4 > buffer.len() { break; }
        values.push(f32::from_le_bytes([buffer[start], buffer[start+1], buffer[start+2], buffer[start+3]]));
    }
    Ok(values)
}

fn read_accessor_indices_manual(gltf: &Value, accessor_idx: usize, buffers: &[Vec<u8>]) -> Result<Vec<u32>> {
    let (buffer_idx, offset, count, stride, component_type) = get_accessor_info(gltf, accessor_idx)
        .ok_or_else(|| anyhow!("Invalid accessor"))?;
    
    let buffer = buffers.get(buffer_idx).ok_or_else(|| anyhow!("Buffer not found"))?;
    
    let mut values = Vec::with_capacity(count);
    
    match component_type {
        5121 => { // UNSIGNED_BYTE
            let actual_stride = if stride > 0 { stride } else { 1 };
            for i in 0..count {
                let start = offset + i * actual_stride;
                if start >= buffer.len() { break; }
                values.push(buffer[start] as u32);
            }
        }
        5123 => { // UNSIGNED_SHORT
            let actual_stride = if stride > 0 { stride } else { 2 };
            for i in 0..count {
                let start = offset + i * actual_stride;
                if start + 2 > buffer.len() { break; }
                values.push(u16::from_le_bytes([buffer[start], buffer[start+1]]) as u32);
            }
        }
        5125 => { // UNSIGNED_INT
            let actual_stride = if stride > 0 { stride } else { 4 };
            for i in 0..count {
                let start = offset + i * actual_stride;
                if start + 4 > buffer.len() { break; }
                values.push(u32::from_le_bytes([buffer[start], buffer[start+1], buffer[start+2], buffer[start+3]]));
            }
        }
        _ => {}
    }
    
    Ok(values)
}

fn read_accessor_u32x4_manual(gltf: &Value, accessor_idx: usize, buffers: &[Vec<u8>]) -> Result<Vec<[u32; 4]>> {
    let (buffer_idx, offset, count, stride, component_type) = get_accessor_info(gltf, accessor_idx)
        .ok_or_else(|| anyhow!("Invalid accessor"))?;
    
    let buffer = buffers.get(buffer_idx).ok_or_else(|| anyhow!("Buffer not found"))?;
    
    let mut values = Vec::with_capacity(count);
    
    match component_type {
        5121 => { // UNSIGNED_BYTE
            let actual_stride = if stride > 0 { stride } else { 4 };
            for i in 0..count {
                let start = offset + i * actual_stride;
                if start + 4 > buffer.len() { break; }
                values.push([buffer[start] as u32, buffer[start+1] as u32, buffer[start+2] as u32, buffer[start+3] as u32]);
            }
        }
        5123 => { // UNSIGNED_SHORT
            let actual_stride = if stride > 0 { stride } else { 8 };
            for i in 0..count {
                let start = offset + i * actual_stride;
                if start + 8 > buffer.len() { break; }
                values.push([
                    u16::from_le_bytes([buffer[start], buffer[start+1]]) as u32,
                    u16::from_le_bytes([buffer[start+2], buffer[start+3]]) as u32,
                    u16::from_le_bytes([buffer[start+4], buffer[start+5]]) as u32,
                    u16::from_le_bytes([buffer[start+6], buffer[start+7]]) as u32,
                ]);
            }
        }
        _ => {
            for _ in 0..count { values.push([0, 0, 0, 0]); }
        }
    }
    
    Ok(values)
}

fn read_accessor_f32x4_manual(gltf: &Value, accessor_idx: usize, buffers: &[Vec<u8>]) -> Result<Vec<[f32; 4]>> {
    let (buffer_idx, offset, count, stride, _) = get_accessor_info(gltf, accessor_idx)
        .ok_or_else(|| anyhow!("Invalid accessor"))?;
    
    let buffer = buffers.get(buffer_idx).ok_or_else(|| anyhow!("Buffer not found"))?;
    let actual_stride = if stride > 0 { stride } else { 16 };
    
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let start = offset + i * actual_stride;
        if start + 16 > buffer.len() { break; }
        values.push([
            f32::from_le_bytes([buffer[start], buffer[start+1], buffer[start+2], buffer[start+3]]),
            f32::from_le_bytes([buffer[start+4], buffer[start+5], buffer[start+6], buffer[start+7]]),
            f32::from_le_bytes([buffer[start+8], buffer[start+9], buffer[start+10], buffer[start+11]]),
            f32::from_le_bytes([buffer[start+12], buffer[start+13], buffer[start+14], buffer[start+15]]),
        ]);
    }
    Ok(values)
}

fn extract_primitive_mesh_manual(
    gltf: &Value, 
    primitive: &Value, 
    buffers: &[Vec<u8>],
    texture_names: &[String],
    materials: &[MaterialData],
    tex_img_indices: &[usize]
) -> Result<Mesh> {
    let attributes = primitive.get("attributes").ok_or_else(|| anyhow!("No attributes"))?;
    
    let positions = if let Some(idx) = attributes.get("POSITION").and_then(|p| p.as_u64()) {
        read_accessor_vec3_manual(gltf, idx as usize, buffers)?
    } else {
        Vec::new()
    };
    
    let normals = if let Some(idx) = attributes.get("NORMAL").and_then(|n| n.as_u64()) {
        read_accessor_vec3_manual(gltf, idx as usize, buffers)?
    } else {
        Vec::new()
    };
    
    let uvs = if let Some(idx) = attributes.get("TEXCOORD_0").and_then(|t| t.as_u64()) {
        read_accessor_vec2_manual(gltf, idx as usize, buffers)?
    } else {
        Vec::new()
    };
    
    // Get material for this primitive
    let material_idx = primitive.get("material").and_then(|m| m.as_u64()).map(|m| m as usize);
    
    // Get base color from material (default white)
    let base_color = if let Some(mat_idx) = material_idx {
        materials.get(mat_idx)
            .map(|m| [m.base_color[0], m.base_color[1], m.base_color[2]])
            .unwrap_or([1.0, 1.0, 1.0])
    } else {
        [1.0, 1.0, 1.0]
    };
    
    let vertex_count = positions.len();
    let mut vertices = Vec::with_capacity(vertex_count);
    
    for i in 0..vertex_count {
        let position = positions.get(i).copied().unwrap_or(Vec3::ZERO);
        let normal = normals.get(i).copied().unwrap_or(Vec3::Y);
        let uv = uvs.get(i).copied().unwrap_or([0.0, 0.0]);
        
        vertices.push(Vertex {
            position: position.to_array(),
            normal: normal.to_array(),
            uv,
            color: base_color,  // Apply material base color to vertex
            tangent: [1.0, 0.0, 0.0, 1.0],
            bone_indices: [0, 0, 0, 0],
            bone_weights: [1.0, 0.0, 0.0, 0.0],
        });
    }
    
    let indices = if let Some(idx) = primitive.get("indices").and_then(|i| i.as_u64()) {
        read_accessor_indices_manual(gltf, idx as usize, buffers)?
    } else {
        (0..vertex_count as u32).collect()
    };
    
    let (aabb_min, aabb_max) = compute_aabb(&vertices);
    
    // Get texture reference if material has one
    let mut albedo_texture = None;
    if let Some(mat_idx) = material_idx {
        if let Some(mat) = materials.get(mat_idx) {
            if let Some(tex_idx) = mat.texture_idx {
                if tex_idx < tex_img_indices.len() {
                    let img_idx = tex_img_indices[tex_idx];
                    if img_idx < texture_names.len() {
                        albedo_texture = Some(texture_names[img_idx].clone());
                    }
                }
            }
        }
    }
    
    Ok(Mesh {
        vertices,
        indices,
        albedo: None,
        normal: None,
        metallic_roughness: None,
        albedo_texture,
        aabb_min,
        aabb_max,
        decoded_albedo: None,
        decoded_normal: None,
        decoded_mr: None,
    })
}

fn extract_skin_data_manual(gltf: &Value, primitive: &Value, buffers: &[Vec<u8>]) -> Result<(Vec<[u32; 4]>, Vec<[f32; 4]>)> {
    let attributes = primitive.get("attributes").ok_or_else(|| anyhow!("No attributes"))?;
    
    let bone_indices = if let Some(idx) = attributes.get("JOINTS_0").and_then(|j| j.as_u64()) {
        read_accessor_u32x4_manual(gltf, idx as usize, buffers)?
    } else {
        Vec::new()
    };
    
    let bone_weights = if let Some(idx) = attributes.get("WEIGHTS_0").and_then(|w| w.as_u64()) {
        read_accessor_f32x4_manual(gltf, idx as usize, buffers)?
    } else {
        Vec::new()
    };
    
    Ok((bone_indices, bone_weights))
}

fn extract_animation_manual(
    gltf: &Value,
    animation: &Value,
    buffers: &[Vec<u8>],
    joint_map: &HashMap<usize, usize>,
) -> Result<AnimationClip> {
    let name = animation.get("name")
        .and_then(|n| n.as_str())
        .unwrap_or("animation")
        .to_string();
    
    let mut duration = 0.0f32;
    let mut channels_map: HashMap<usize, BoneKeyframes> = HashMap::new();
    
    let channels = animation.get("channels").and_then(|c| c.as_array());
    let samplers = animation.get("samplers").and_then(|s| s.as_array());
    
    if let (Some(channels), Some(samplers)) = (channels, samplers) {
        for channel in channels {
            let target = match channel.get("target") {
                Some(t) => t,
                None => continue,
            };
            
            let node_idx = match target.get("node").and_then(|n| n.as_u64()) {
                Some(n) => n as usize,
                None => continue,
            };
            
            let bone_index = match joint_map.get(&node_idx) {
                Some(&idx) => idx,
                None => continue,
            };
            
            let sampler_idx = match channel.get("sampler").and_then(|s| s.as_u64()) {
                Some(s) => s as usize,
                None => continue,
            };
            
            let sampler = match samplers.get(sampler_idx) {
                Some(s) => s,
                None => continue,
            };
            
            let input_idx = sampler.get("input").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
            let output_idx = sampler.get("output").and_then(|o| o.as_u64()).unwrap_or(0) as usize;
            
            let times = read_accessor_f32_manual(gltf, input_idx, buffers)?;
            
            if let Some(&max_time) = times.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
                duration = duration.max(max_time);
            }
            
            let entry = channels_map.entry(bone_index).or_insert_with(|| BoneKeyframes {
                bone_index,
                position_keys: Vec::new(),
                rotation_keys: Vec::new(),
                scale_keys: Vec::new(),
            });
            
            let path = target.get("path").and_then(|p| p.as_str()).unwrap_or("");
            
            match path {
                "translation" => {
                    let translations = read_accessor_vec3_manual(gltf, output_idx, buffers)?;
                    for (i, &t) in times.iter().enumerate() {
                        if i < translations.len() {
                            entry.position_keys.push((t, translations[i]));
                        }
                    }
                }
                "rotation" => {
                    let rotations = read_accessor_quat_manual(gltf, output_idx, buffers)?;
                    for (i, &t) in times.iter().enumerate() {
                        if i < rotations.len() {
                            entry.rotation_keys.push((t, rotations[i]));
                        }
                    }
                }
                "scale" => {
                    let scales = read_accessor_vec3_manual(gltf, output_idx, buffers)?;
                    for (i, &t) in times.iter().enumerate() {
                        if i < scales.len() {
                            entry.scale_keys.push((t, scales[i]));
                        }
                    }
                }
                _ => {}
            }
        }
    }
    
    Ok(AnimationClip {
        name,
        duration,
        channels: channels_map.into_values().collect(),
        events: Vec::new(),
    })
}


// ============================================================================
// SANIM Format Loader (Custom Binary Format)
// ============================================================================


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
        events: Vec::new(),
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
