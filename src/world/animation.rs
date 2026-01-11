//! Animation Runtime System for STFSC Engine
//!
//! Provides GPU-accelerated skeletal animation for imported 3D models:
//! - `Animator` component for animation playback state
//! - `AnimationState` component for computed bone matrices (GPU-ready)
//! - Animation blending and state machine support
//!
//! Optimized for Quest 3's 2.4 TFLOPS to render 100+ animated NPCs in 556 Downtown.

use crate::world::fbx_loader::{AnimationClip, AnimationEvent, Skeleton};
use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;
use std::sync::Arc;

/// Maximum bones per skeleton (matches shader UBO size)
pub const MAX_BONES: usize = 128;

/// Represents an animation resource that can be played by the Animator.
/// This can be a direct clip or a complex blend tree.
#[derive(Clone, Debug)]
pub enum AnimResource {
    /// Reference to an animation clip by its index in the Animator's `clips` vector.
    Clip(usize),
    /// Reference to a blend tree definition.
    BlendTree(BlendTree),
}

impl Default for AnimResource {
    fn default() -> Self {
        AnimResource::Clip(0)
    }
}

/// Represents a parameter used by blend trees.
#[derive(Clone, Debug)]
pub enum AnimParam {
    Float(f32),
    Bool(bool),
    Trigger(bool),
    Int(i32),
}

/// ECS Component: Controls animation playback for an entity
#[derive(Clone, Debug)]
pub struct Animator {
    /// Reference to the skeleton hierarchy
    pub skeleton: Arc<Skeleton>,
    /// Available animation clips for this entity
    pub clips: Vec<Arc<AnimationClip>>,
    /// Currently playing animation resource (clip or blend tree)
    pub current_resource: AnimResource,
    /// Current playback time in seconds
    pub time: f32,
    /// Playback speed multiplier (1.0 = normal)
    pub speed: f32,
    /// Whether to loop the animation
    pub looping: bool,
    /// Whether the animation is currently playing
    pub playing: bool,
    /// Blend target resource (for crossfades)
    pub blend_target: Option<AnimResource>,
    /// Blend factor (0.0 = current, 1.0 = target)
    pub blend_factor: f32,
    /// Blend duration in seconds
    pub blend_duration: f32,
    /// Extracted root motion for this frame
    pub root_motion: RootMotion,
    /// Index of the bone to use for root motion (usually 0)
    pub root_bone_index: usize,
    /// Animation layers for additive/partial blending
    pub layers: Vec<AnimationLayer>,
    /// Look-At IK solver (optional)
    pub look_at_ik: Option<LookAtIK>,
    /// Foot IK solver (optional)
    pub foot_ik: Option<FootIK>,
    /// Two-bone IK solvers (e.g., for hands reaching for objects)
    pub ik_chains: Vec<IKSolver>,
}

impl Animator {
    /// Create a new animator with a skeleton and animation clips
    pub fn new(skeleton: Arc<Skeleton>, clips: Vec<Arc<AnimationClip>>) -> Self {
        Self {
            skeleton,
            clips,
            current_resource: AnimResource::Clip(0),
            time: 0.0,
            speed: 1.0,
            looping: true,
            playing: true,
            blend_target: None,
            blend_factor: 0.0,
            blend_duration: 0.0,
            root_motion: RootMotion::default(),
            root_bone_index: 0,
            layers: Vec::new(),
            look_at_ik: None,
            foot_ik: None,
            ik_chains: Vec::new(),
        }
    }

    /// Add an animation layer for additive/partial blending
    pub fn add_layer(&mut self, layer: AnimationLayer) {
        self.layers.push(layer);
    }

    /// Set look-at IK solver
    pub fn with_look_at_ik(mut self, look_at: LookAtIK) -> Self {
        self.look_at_ik = Some(look_at);
        self
    }

    /// Set foot IK solver
    pub fn with_foot_ik(mut self, foot_ik: FootIK) -> Self {
        self.foot_ik = Some(foot_ik);
        self
    }

    /// Add a two-bone IK chain
    pub fn add_ik_chain(&mut self, ik: IKSolver) {
        self.ik_chains.push(ik);
    }


    /// Start playing a specific animation clip by index
    pub fn play(&mut self, clip_index: usize) {
        if clip_index < self.clips.len() {
            self.current_resource = AnimResource::Clip(clip_index);
            self.time = 0.0;
            self.playing = true;
            self.blend_target = None;
        }
    }

    /// Play an animation by name
    pub fn play_by_name(&mut self, name: &str) {
        for (i, clip) in self.clips.iter().enumerate() {
            if clip.name == name {
                self.play(i);
                return;
            }
        }
    }

    /// Set the playback time directly (for editor scrubbing)
    pub fn set_time(&mut self, time: f32) {
        self.time = time;
    }

    /// Crossfade to a new animation resource over a duration
    pub fn crossfade_to(&mut self, resource: AnimResource, duration: f32) {
        // Simple check to avoid redundant crossfades
        let is_same = match (&self.current_resource, &resource) {
            (AnimResource::Clip(a), AnimResource::Clip(b)) => a == b,
            _ => false, // Always allow crossfading to/from blend trees for now
        };

        if !is_same {
            self.blend_target = Some(resource);
            self.blend_factor = 0.0;
            self.blend_duration = duration.max(0.001);
        }
    }

    /// Crossfade to an animation by name
    pub fn crossfade_by_name(&mut self, name: &str, duration: f32) {
        for (i, clip) in self.clips.iter().enumerate() {
            if clip.name == name {
                self.crossfade_to(AnimResource::Clip(i), duration);
                return;
            }
        }
    }

    /// Stop the animation
    pub fn stop(&mut self) {
        self.playing = false;
    }

    /// Pause the animation (can be resumed)
    pub fn pause(&mut self) {
        self.playing = false;
    }

    /// Resume a paused animation
    pub fn resume(&mut self) {
        self.playing = true;
    }

    /// Update animation time and compute bone matrices
    pub fn update(&mut self, dt: f32) -> Vec<Mat4> {
        self.update_with_params(dt, &HashMap::new(), &mut None)
    }

    /// Update animation with parameters for blend trees and event queue
    pub fn update_with_params(
        &mut self,
        dt: f32,
        params: &HashMap<String, AnimParam>,
        queue: &mut Option<hecs::RefMut<AnimationEventQueue>>,
    ) -> Vec<Mat4> {
        // Reset root motion for this frame
        self.root_motion = RootMotion::default();

        if !self.playing || self.clips.is_empty() {
            return self.bind_pose_matrices();
        }

        let old_time = self.time;
        // Update playback time
        self.time += dt * self.speed;
        let new_time = self.time;

        // Handle looping for clips (blend trees manage their own duration/time?)
        // For now, treat blend tree as infinite or use its root clip duration if possible
        if let AnimResource::Clip(idx) = &self.current_resource {
            let current_clip = &self.clips[*idx];
            
            // Extract root motion
            let duration = current_clip.duration;
            if duration > 0.0 {
                let t_start = old_time % duration;
                let t_end = new_time % duration;

                let (pos_start, rot_start, _) = current_clip.sample_bone(self.root_bone_index, t_start);
                let (pos_end, rot_end, _) = current_clip.sample_bone(self.root_bone_index, t_end);

                if t_end >= t_start {
                    self.root_motion.delta_position = pos_end - pos_start;
                    self.root_motion.delta_rotation = rot_end * rot_start.inverse();
                } else {
                    // Wrapped (looping)
                    let (pos_max, rot_max, _) = current_clip.sample_bone(self.root_bone_index, duration);
                    let (pos_min, rot_min, _) = current_clip.sample_bone(self.root_bone_index, 0.0);
                    
                    let d1 = pos_max - pos_start;
                    let d2 = pos_end - pos_min;
                    self.root_motion.delta_position = d1 + d2;
                    
                    let r1 = rot_max * rot_start.inverse();
                    let r2 = rot_end * rot_min.inverse();
                    self.root_motion.delta_rotation = r2 * r1;
                }
                self.root_motion.has_motion = true;
            }

            // Fire events
            self.check_events(old_time, new_time, *idx, queue);

            if self.time >= duration {
                if self.looping {
                    self.time %= duration;
                } else {
                    self.time = duration;
                    self.playing = false;
                }
            }
        }

        // Sample current animation (base layer)
        let mut base_pose = self.sample_resource(&self.current_resource, self.time, params);

        // Handle crossfade blending
        if let Some(target_resource) = self.blend_target.clone() {
            self.blend_factor += dt / self.blend_duration;
            
            if self.blend_factor >= 1.0 {
                // Blend complete, switch to target
                self.current_resource = target_resource;
                self.blend_target = None;
                self.blend_factor = 0.0;
                base_pose = self.sample_resource(&self.current_resource, self.time, params);
            } else {
                let target_pose = self.sample_resource(&target_resource, self.time, params);
                base_pose = blend_poses(&base_pose, &target_pose, self.blend_factor);
            }
        }

        // Apply animation layers with avatar masks
        if !self.layers.is_empty() {
            let base_time = Some(self.time);
            for layer in &mut self.layers {
                // Update layer time and weight interpolation
                layer.update(dt, base_time);
                
                // Skip layers with zero weight
                if layer.weight <= 0.0 {
                    continue;
                }
                
                // Sample layer pose
                let layer_pose = layer.sample(&self.skeleton);
                
                // Blend layer onto base using additive or override mode
                base_pose = blend_layer_onto_base(&base_pose, layer, &layer_pose);
            }
        }

        base_pose
    }


    /// Check and fire events for a clip
    fn check_events(
        &self,
        old_time: f32,
        new_time: f32,
        clip_idx: usize,
        queue: &mut Option<hecs::RefMut<AnimationEventQueue>>,
    ) {
        if let Some(q) = queue {
            let clip = &self.clips[clip_idx];
            let duration = clip.duration;
            if duration <= 0.0 { return; }

            // Normalize times to [0, duration]
            let t_start = old_time % duration;
            let t_end = new_time % duration;

            for event in &clip.events {
                let fired = if t_start < t_end {
                    event.time > t_start && event.time <= t_end
                } else {
                    // Wrapped around (looping)
                    (event.time > t_start && event.time <= duration) || (event.time >= 0.0 && event.time <= t_end)
                };

                if fired {
                    q.push(event.clone());
                }
            }
        }
    }

    /// Sample an animation resource (clip or blend tree)
    fn sample_resource(&self, res: &AnimResource, time: f32, params: &HashMap<String, AnimParam>) -> Vec<Mat4> {
        let local_transforms = match res {
            AnimResource::Clip(idx) => {
                if *idx < self.clips.len() {
                    self.clips[*idx].sample(time, &self.skeleton)
                } else {
                    vec![Mat4::IDENTITY; self.skeleton.bones.len()]
                }
            }
            AnimResource::BlendTree(tree) => {
                tree.evaluate(params, &self.clips, &self.skeleton, time)
            }
        };

        self.compute_skinning_matrices(&local_transforms)
    }

    /// Sample a clip at a specific time, returning local bone transforms
    #[allow(dead_code)]
    fn sample_clip(&self, clip_idx: usize, time: f32) -> Vec<Mat4> {
        if clip_idx >= self.clips.len() {
            return self.bind_pose_matrices();
        }

        let clip = &self.clips[clip_idx];
        let local_transforms = clip.sample(time, &self.skeleton);

        // Convert local transforms to final skinning matrices
        self.compute_skinning_matrices(&local_transforms)
    }

    /// Compute final skinning matrices from local bone transforms
    /// 
    /// Formula: skinning_matrix[i] = global_transform[i] * inverse_bind_matrix[i]
    fn compute_skinning_matrices(&self, local_transforms: &[Mat4]) -> Vec<Mat4> {
        let bone_count = self.skeleton.bones.len().min(MAX_BONES);
        let mut global_transforms = vec![Mat4::IDENTITY; bone_count];
        let mut skinning_matrices = vec![Mat4::IDENTITY; bone_count];

        // Compute global transforms (parent-to-child order)
        for i in 0..bone_count {
            let bone = &self.skeleton.bones[i];
            let local = if i < local_transforms.len() {
                local_transforms[i]
            } else {
                bone.local_transform
            };

            global_transforms[i] = if let Some(parent_idx) = bone.parent_index {
                if parent_idx < bone_count {
                    global_transforms[parent_idx] * local
                } else {
                    local
                }
            } else {
                local
            };

            // Skinning matrix = global * inverse_bind
            skinning_matrices[i] = global_transforms[i] * bone.inverse_bind_matrix;
        }

        skinning_matrices
    }

    /// Get bind pose matrices (identity skinning)
    fn bind_pose_matrices(&self) -> Vec<Mat4> {
        vec![Mat4::IDENTITY; self.skeleton.bones.len().min(MAX_BONES)]
    }
}

/// ECS Component: Holds computed bone matrices ready for GPU upload
#[derive(Clone, Debug)]
pub struct AnimationState {
    /// Final bone matrices for GPU skinning (skinning_matrix = global * inverse_bind)
    pub bone_matrices: Vec<Mat4>,
    /// Dirty flag - set when matrices need re-upload to GPU
    pub dirty: bool,
}

impl AnimationState {
    /// Create a new animation state with identity matrices
    pub fn new(bone_count: usize) -> Self {
        Self {
            bone_matrices: vec![Mat4::IDENTITY; bone_count.min(MAX_BONES)],
            dirty: true,
        }
    }

    /// Update bone matrices from animator and mark dirty
    pub fn update_from(&mut self, matrices: Vec<Mat4>) {
        self.bone_matrices = matrices;
        self.dirty = true;
    }

    /// Get matrices as flat f32 array for GPU upload (column-major)
    pub fn as_gpu_buffer(&self) -> Vec<f32> {
        let mut buffer = Vec::with_capacity(MAX_BONES * 16);
        for mat in &self.bone_matrices {
            buffer.extend_from_slice(&mat.to_cols_array());
        }
        // Pad to MAX_BONES if needed
        while buffer.len() < MAX_BONES * 16 {
            buffer.extend_from_slice(&Mat4::IDENTITY.to_cols_array());
        }
        buffer
    }

    /// Mark as clean after GPU upload
    pub fn mark_clean(&mut self) {
        self.dirty = false;
    }
}

/// Blend two poses together using linear interpolation
/// 
/// Performs component-wise TRS blending:
/// - Position: lerp
/// - Rotation: slerp
/// - Scale: lerp
pub fn blend_poses(a: &[Mat4], b: &[Mat4], factor: f32) -> Vec<Mat4> {
    let len = a.len().min(b.len());
    let factor = factor.clamp(0.0, 1.0);

    (0..len)
        .map(|i| {
            // Decompose matrices
            let (scale_a, rot_a, trans_a) = a[i].to_scale_rotation_translation();
            let (scale_b, rot_b, trans_b) = b[i].to_scale_rotation_translation();

            // Blend components
            let scale = scale_a.lerp(scale_b, factor);
            let rot = rot_a.slerp(rot_b, factor);
            let trans = trans_a.lerp(trans_b, factor);

            // Recompose
            Mat4::from_scale_rotation_translation(scale, rot, trans)
        })
        .collect()
}

// ============================================================================
// Blend Trees - Parameter-driven animation blending (Unity-like)
// ============================================================================

/// Blend tree node types for hierarchical animation blending
#[derive(Clone, Debug)]
pub enum BlendNode {
    /// Direct animation clip reference by index
    Clip(usize),
    /// 1D blend based on single parameter (e.g., speed for idle/walk/run)
    Blend1D {
        param: String,
        /// Children with threshold values (threshold, node)
        children: Vec<(f32, BlendNode)>,
    },
    /// 2D blend based on two parameters (e.g., velocity X/Y for strafe locomotion)
    Blend2D {
        param_x: String,
        param_y: String,
        /// Children with 2D positions (x, y, node)
        children: Vec<(f32, f32, BlendNode)>,
    },
    /// Direct blend with explicit weight (for additive layers)
    Direct {
        weight: f32,
        child: Box<BlendNode>,
    },
}

impl BlendNode {
    /// Evaluate this node at current parameters, returning blended pose matrices
    pub fn evaluate(
        &self,
        params: &HashMap<String, AnimParam>,
        clips: &[Arc<AnimationClip>],
        skeleton: &Skeleton,
        time: f32,
    ) -> Vec<Mat4> {
        match self {
            BlendNode::Clip(idx) => {
                if *idx < clips.len() {
                    clips[*idx].sample(time, skeleton)
                } else {
                    vec![Mat4::IDENTITY; skeleton.bones.len()]
                }
            }
            BlendNode::Blend1D { param, children } => {
                self.evaluate_1d(param, children, params, clips, skeleton, time)
            }
            BlendNode::Blend2D { param_x, param_y, children } => {
                self.evaluate_2d(param_x, param_y, children, params, clips, skeleton, time)
            }
            BlendNode::Direct { weight, child } => {
                let pose = child.evaluate(params, clips, skeleton, time);
                // Scale by weight (for additive blending)
                pose.into_iter()
                    .map(|m| {
                        let (s, r, t) = m.to_scale_rotation_translation();
                        Mat4::from_scale_rotation_translation(
                            Vec3::ONE.lerp(s, *weight),
                            Quat::IDENTITY.slerp(r, *weight),
                            t * *weight,
                        )
                    })
                    .collect()
            }
        }
    }

    fn evaluate_1d(
        &self,
        param_name: &str,
        children: &[(f32, BlendNode)],
        params: &HashMap<String, AnimParam>,
        clips: &[Arc<AnimationClip>],
        skeleton: &Skeleton,
        time: f32,
    ) -> Vec<Mat4> {
        if children.is_empty() {
            return vec![Mat4::IDENTITY; skeleton.bones.len()];
        }
        if children.len() == 1 {
            return children[0].1.evaluate(params, clips, skeleton, time);
        }

        // Get parameter value
        let value = match params.get(param_name) {
            Some(AnimParam::Float(v)) => *v,
            _ => 0.0,
        };

        // Find the two nodes to blend between
        let mut sorted: Vec<_> = children.iter().collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Find surrounding thresholds
        let mut lower_idx = 0;
        for (i, (threshold, _)) in sorted.iter().enumerate() {
            if *threshold <= value {
                lower_idx = i;
            } else {
                break;
            }
        }
        let upper_idx = (lower_idx + 1).min(sorted.len() - 1);

        if lower_idx == upper_idx {
            return sorted[lower_idx].1.evaluate(params, clips, skeleton, time);
        }

        let lower_threshold = sorted[lower_idx].0;
        let upper_threshold = sorted[upper_idx].0;
        let range = upper_threshold - lower_threshold;
        let factor = if range > 0.0 {
            ((value - lower_threshold) / range).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let pose_a = sorted[lower_idx].1.evaluate(params, clips, skeleton, time);
        let pose_b = sorted[upper_idx].1.evaluate(params, clips, skeleton, time);

        blend_poses(&pose_a, &pose_b, factor)
    }

    fn evaluate_2d(
        &self,
        param_x: &str,
        param_y: &str,
        children: &[(f32, f32, BlendNode)],
        params: &HashMap<String, AnimParam>,
        clips: &[Arc<AnimationClip>],
        skeleton: &Skeleton,
        time: f32,
    ) -> Vec<Mat4> {
        if children.is_empty() {
            return vec![Mat4::IDENTITY; skeleton.bones.len()];
        }
        if children.len() == 1 {
            return children[0].2.evaluate(params, clips, skeleton, time);
        }

        // Get parameter values
        let x = match params.get(param_x) {
            Some(AnimParam::Float(v)) => *v,
            _ => 0.0,
        };
        let y = match params.get(param_y) {
            Some(AnimParam::Float(v)) => *v,
            _ => 0.0,
        };

        // Calculate weights using inverse distance weighting
        let point = glam::Vec2::new(x, y);
        let mut weights: Vec<f32> = Vec::with_capacity(children.len());
        let mut total_weight = 0.0f32;

        for (cx, cy, _) in children {
            let child_point = glam::Vec2::new(*cx, *cy);
            let dist = point.distance(child_point).max(0.001); // Avoid division by zero
            let weight = 1.0 / (dist * dist); // Inverse square distance
            weights.push(weight);
            total_weight += weight;
        }

        // Normalize weights
        if total_weight > 0.0 {
            for w in &mut weights {
                *w /= total_weight;
            }
        }

        // Blend all poses
        let mut result = vec![Mat4::IDENTITY; skeleton.bones.len()];
        for (i, (_, _, node)) in children.iter().enumerate() {
            let pose = node.evaluate(params, clips, skeleton, time);
            let weight = weights[i];
            
            for (j, mat) in pose.iter().enumerate() {
                if j >= result.len() { break; }
                let (s, r, t) = mat.to_scale_rotation_translation();
                let (rs, rr, rt) = result[j].to_scale_rotation_translation();
                
                result[j] = Mat4::from_scale_rotation_translation(
                    rs.lerp(s, weight),
                    rr.slerp(r, weight),
                    rt + t * weight,
                );
            }
        }

        result
    }
}

/// Complete blend tree for complex animation blending
#[derive(Clone, Debug)]
pub struct BlendTree {
    /// Name of this blend tree
    pub name: String,
    /// Root node of the blend tree
    pub root: BlendNode,
}

impl BlendTree {
    /// Create a new blend tree with a name and root node
    pub fn new(name: &str, root: BlendNode) -> Self {
        Self {
            name: name.to_string(),
            root,
        }
    }

    /// Create a simple 1D locomotion blend tree (idle -> walk -> run)
    pub fn locomotion_1d(idle_clip: usize, walk_clip: usize, run_clip: usize) -> Self {
        Self {
            name: "Locomotion".to_string(),
            root: BlendNode::Blend1D {
                param: "Speed".to_string(),
                children: vec![
                    (0.0, BlendNode::Clip(idle_clip)),   // Speed = 0: Idle
                    (1.5, BlendNode::Clip(walk_clip)),   // Speed = 1.5: Walk
                    (4.0, BlendNode::Clip(run_clip)),    // Speed = 4+: Run
                ],
            },
        }
    }

    /// Create a 2D strafe locomotion blend tree
    pub fn locomotion_2d(
        idle: usize,
        forward: usize,
        backward: usize,
        left: usize,
        right: usize,
    ) -> Self {
        Self {
            name: "Locomotion2D".to_string(),
            root: BlendNode::Blend2D {
                param_x: "VelocityX".to_string(),
                param_y: "VelocityY".to_string(),
                children: vec![
                    (0.0, 0.0, BlendNode::Clip(idle)),      // Center: Idle
                    (0.0, 1.0, BlendNode::Clip(forward)),   // Forward
                    (0.0, -1.0, BlendNode::Clip(backward)), // Backward
                    (-1.0, 0.0, BlendNode::Clip(left)),     // Strafe left
                    (1.0, 0.0, BlendNode::Clip(right)),     // Strafe right
                ],
            },
        }
    }

    /// Evaluate the blend tree at current parameters
    pub fn evaluate(
        &self,
        params: &HashMap<String, AnimParam>,
        clips: &[Arc<AnimationClip>],
        skeleton: &Skeleton,
        time: f32,
    ) -> Vec<Mat4> {
        self.root.evaluate(params, clips, skeleton, time)
    }
}

// ============================================================================
// Avatar Masks - Partial Body Animation (Unity-like)
// ============================================================================

/// Defines which bones a layer affects for partial-body animation.
/// Use this to blend upper body aiming while legs continue running, etc.
#[derive(Clone, Debug)]
pub struct AvatarMask {
    /// Human-readable name (e.g., "UpperBody", "LeftArm")
    pub name: String,
    /// Per-bone weights (0.0 = unaffected, 1.0 = fully affected)
    /// Length should match skeleton bone count
    pub bone_weights: Vec<f32>,
}

impl AvatarMask {
    /// Create a mask affecting all bones with full weight
    pub fn full(bone_count: usize) -> Self {
        Self {
            name: "Full".to_string(),
            bone_weights: vec![1.0; bone_count],
        }
    }

    /// Create a mask affecting no bones (useful as starting point)
    pub fn empty(bone_count: usize) -> Self {
        Self {
            name: "Empty".to_string(),
            bone_weights: vec![0.0; bone_count],
        }
    }

    /// Create a mask from specific bone indices with full weight
    pub fn from_bones(name: &str, bone_count: usize, active_bones: &[usize]) -> Self {
        let mut weights = vec![0.0; bone_count];
        for &idx in active_bones {
            if idx < bone_count {
                weights[idx] = 1.0;
            }
        }
        Self {
            name: name.to_string(),
            bone_weights: weights,
        }
    }

    /// Create a mask from bone indices with specified weights
    pub fn from_weights(name: &str, bone_count: usize, bones_and_weights: &[(usize, f32)]) -> Self {
        let mut weights = vec![0.0; bone_count];
        for &(idx, weight) in bones_and_weights {
            if idx < bone_count {
                weights[idx] = weight.clamp(0.0, 1.0);
            }
        }
        Self {
            name: name.to_string(),
            bone_weights: weights,
        }
    }

    /// Create upper body mask (typical humanoid: spine and above)
    /// Assumes common humanoid bone naming: bones 0-10 = lower body, 11+ = upper body
    /// For real use, pass actual bone indices based on your skeleton.
    pub fn upper_body_humanoid(bone_count: usize, spine_start_index: usize) -> Self {
        let mut weights = vec![0.0; bone_count];
        for i in spine_start_index..bone_count {
            weights[i] = 1.0;
        }
        Self {
            name: "UpperBody".to_string(),
            bone_weights: weights,
        }
    }

    /// Create lower body mask (hips and legs)
    pub fn lower_body_humanoid(bone_count: usize, spine_start_index: usize) -> Self {
        let mut weights = vec![0.0; bone_count];
        for i in 0..spine_start_index.min(bone_count) {
            weights[i] = 1.0;
        }
        Self {
            name: "LowerBody".to_string(),
            bone_weights: weights,
        }
    }

    /// Get the weight for a specific bone
    pub fn get_weight(&self, bone_index: usize) -> f32 {
        self.bone_weights.get(bone_index).copied().unwrap_or(0.0)
    }

    /// Set weight for a bone and its children (recursive in skeleton)
    pub fn set_bone_weight(&mut self, bone_index: usize, weight: f32) {
        if bone_index < self.bone_weights.len() {
            self.bone_weights[bone_index] = weight.clamp(0.0, 1.0);
        }
    }
}

/// Animation layer for additive/partial blending (enhanced Unity-like layer system)
#[derive(Clone, Debug)]
pub struct AnimationLayer {
    /// Layer name for debugging
    pub name: String,
    /// The animation clip for this layer
    pub clip: Arc<AnimationClip>,
    /// Current blend weight (0.0 = no effect, 1.0 = full)
    pub weight: f32,
    /// Target weight for smooth transitions
    pub target_weight: f32,
    /// Weight blend speed (units per second, 0 = instant)
    pub weight_blend_speed: f32,
    /// Current playback time
    pub time: f32,
    /// Playback speed
    pub speed: f32,
    /// Avatar mask for partial body animation (None = affects all bones)
    pub avatar_mask: Option<AvatarMask>,
    /// Whether this is an additive layer (adds to base pose) vs override
    pub additive: bool,
    /// Sync playback time to base layer
    pub sync_to_base: bool,
}

impl AnimationLayer {
    /// Create a new animation layer with default settings
    pub fn new(clip: Arc<AnimationClip>) -> Self {
        Self {
            name: "Layer".to_string(),
            clip,
            weight: 1.0,
            target_weight: 1.0,
            weight_blend_speed: 0.0,
            time: 0.0,
            speed: 1.0,
            avatar_mask: None,
            additive: false,
            sync_to_base: false,
        }
    }

    /// Create a named layer
    pub fn named(name: &str, clip: Arc<AnimationClip>) -> Self {
        let mut layer = Self::new(clip);
        layer.name = name.to_string();
        layer
    }

    /// Create an additive layer (blends on top of base animation)
    pub fn additive(clip: Arc<AnimationClip>) -> Self {
        let mut layer = Self::new(clip);
        layer.additive = true;
        layer
    }

    /// Set the avatar mask for partial body animation
    pub fn with_mask(mut self, mask: AvatarMask) -> Self {
        self.avatar_mask = Some(mask);
        self
    }

    /// Set layer to sync with base layer timing
    pub fn synced(mut self) -> Self {
        self.sync_to_base = true;
        self
    }

    /// Smoothly blend to a target weight
    pub fn blend_to(&mut self, weight: f32, speed: f32) {
        self.target_weight = weight.clamp(0.0, 1.0);
        self.weight_blend_speed = speed;
    }

    /// Update layer time and weight interpolation
    pub fn update(&mut self, dt: f32, base_time: Option<f32>) {
        // Interpolate weight towards target
        if self.weight_blend_speed > 0.0 {
            let delta = self.target_weight - self.weight;
            let max_change = self.weight_blend_speed * dt;
            if delta.abs() <= max_change {
                self.weight = self.target_weight;
            } else {
                self.weight += delta.signum() * max_change;
            }
        } else {
            self.weight = self.target_weight;
        }

        // Update time (sync to base or independent)
        if self.sync_to_base {
            if let Some(bt) = base_time {
                // Scale base time to this clip's duration
                let ratio = if self.clip.duration > 0.0 {
                    bt / self.clip.duration
                } else {
                    0.0
                };
                self.time = (ratio % 1.0) * self.clip.duration;
            }
        } else {
            self.time += dt * self.speed;
            if self.time >= self.clip.duration && self.clip.duration > 0.0 {
                self.time %= self.clip.duration;
            }
        }
    }

    /// Sample this layer's clip and apply mask weights
    pub fn sample(&self, skeleton: &Skeleton) -> Vec<Mat4> {
        let pose = self.clip.sample(self.time, skeleton);
        
        // Apply mask if present
        if let Some(mask) = &self.avatar_mask {
            pose.into_iter()
                .enumerate()
                .map(|(i, mat)| {
                    let mask_weight = mask.get_weight(i) * self.weight;
                    if mask_weight <= 0.0 {
                        Mat4::IDENTITY // No contribution
                    } else if mask_weight >= 1.0 {
                        mat
                    } else {
                        // Partial blend with identity
                        let (s, r, t) = mat.to_scale_rotation_translation();
                        Mat4::from_scale_rotation_translation(
                            Vec3::ONE.lerp(s, mask_weight),
                            Quat::IDENTITY.slerp(r, mask_weight),
                            t * mask_weight,
                        )
                    }
                })
                .collect()
        } else {
            pose
        }
    }
}

/// Blend a layer pose on top of a base pose using layer settings
pub fn blend_layer_onto_base(base: &[Mat4], layer: &AnimationLayer, layer_pose: &[Mat4]) -> Vec<Mat4> {
    let len = base.len().min(layer_pose.len());
    let layer_weight = layer.weight;

    (0..len)
        .map(|i| {
            let mask_weight = layer.avatar_mask
                .as_ref()
                .map(|m| m.get_weight(i))
                .unwrap_or(1.0) * layer_weight;

            if mask_weight <= 0.0 {
                base[i]
            } else if layer.additive {
                // Additive: add layer's delta to base
                let (bs, br, bt) = base[i].to_scale_rotation_translation();
                let (ls, lr, lt) = layer_pose[i].to_scale_rotation_translation();
                
                // Additive blending: base + (layer - identity) * weight
                let scale = bs + (ls - Vec3::ONE) * mask_weight;
                let rot = br * Quat::IDENTITY.slerp(lr, mask_weight);
                let trans = bt + lt * mask_weight;
                
                Mat4::from_scale_rotation_translation(scale, rot, trans)
            } else {
                // Override: lerp between base and layer
                let (bs, br, bt) = base[i].to_scale_rotation_translation();
                let (ls, lr, lt) = layer_pose[i].to_scale_rotation_translation();
                
                Mat4::from_scale_rotation_translation(
                    bs.lerp(ls, mask_weight),
                    br.slerp(lr, mask_weight),
                    bt.lerp(lt, mask_weight),
                )
            }
        })
        .collect()
}


/// Special state name for transitions that can occur from any state (like Unity's "Any State")
/// Use this as the `from` field in StateTransition to create global interrupt transitions.
pub const ANY_STATE: &str = "__ANY_STATE__";

// ============================================================================
// Sub-State Machines (Unity-like hierarchical state machines)
// ============================================================================

/// A sub-state machine for hierarchical animation organization.
/// Like Unity's sub-state machines, allows grouping related states (e.g., all combat moves).
#[derive(Clone, Debug)]
pub struct SubStateMachine {
    /// Name of this sub-state machine (e.g., "Combat", "Locomotion")
    pub name: String,
    /// Child state machine containing the sub-states
    pub state_machine: Box<AnimationStateMachine>,
    /// Entry state - which state to start in when entering this sub-machine
    pub entry_state: String,
    /// Exit state - transitioning to this state exits the sub-machine
    pub exit_state: Option<String>,
    /// Whether this sub-machine is currently active
    pub is_active: bool,
}

impl SubStateMachine {
    /// Create a new sub-state machine
    pub fn new(name: &str, entry_state: &str) -> Self {
        Self {
            name: name.to_string(),
            state_machine: Box::new(AnimationStateMachine::new()),
            entry_state: entry_state.to_string(),
            exit_state: None,
            is_active: false,
        }
    }

    /// Enter the sub-state machine, setting the current state to entry
    pub fn enter(&mut self) {
        self.state_machine.current_state = self.entry_state.clone();
        self.is_active = true;
    }

    /// Exit the sub-state machine
    pub fn exit(&mut self) {
        self.is_active = false;
    }

    /// Add a state to the sub-machine
    pub fn add_state(&mut self, name: &str, clip_index: usize) {
        self.state_machine.add_state(name, clip_index);
    }

    /// Add a blend tree state to the sub-machine
    pub fn add_blend_tree(&mut self, name: &str, blend_tree: BlendTree) {
        self.state_machine.add_blend_tree(name, blend_tree);
    }

    /// Set the exit state
    pub fn with_exit(mut self, exit_state: &str) -> Self {
        self.exit_state = Some(exit_state.to_string());
        self
    }

    /// Evaluate the sub-machine if active, returns resource and whether we hit exit
    pub fn evaluate(&mut self) -> Option<(AnimResource, f32, bool)> {
        if !self.is_active {
            return None;
        }

        // Check if we hit exit state
        let hit_exit = self.exit_state.as_ref()
            .map(|exit| self.state_machine.current_state == *exit)
            .unwrap_or(false);

        if let Some((resource, duration)) = self.state_machine.evaluate() {
            Some((resource, duration, hit_exit))
        } else {
            // No transition, but check if current state is exit
            if hit_exit {
                if let Some(resource) = self.state_machine.current_resource() {
                    return Some((resource.clone(), 0.0, true));
                }
            }
            None
        }
    }
}

// ============================================================================
// Compound Conditions (AND/OR logic for transitions)
// ============================================================================

/// Compound transition condition for complex logic
#[derive(Clone, Debug)]
pub enum CompoundCondition {
    /// Single condition (wrapper for backwards compatibility)
    Single(TransitionCondition),
    /// All conditions must be true
    And(Vec<CompoundCondition>),
    /// At least one condition must be true
    Or(Vec<CompoundCondition>),
    /// Invert the result
    Not(Box<CompoundCondition>),
}

impl CompoundCondition {
    /// Create a single condition
    pub fn single(condition: TransitionCondition) -> Self {
        CompoundCondition::Single(condition)
    }

    /// Create an AND compound
    pub fn and(conditions: Vec<CompoundCondition>) -> Self {
        CompoundCondition::And(conditions)
    }

    /// Create an OR compound
    pub fn or(conditions: Vec<CompoundCondition>) -> Self {
        CompoundCondition::Or(conditions)
    }

    /// Create a NOT wrapper
    pub fn not(condition: CompoundCondition) -> Self {
        CompoundCondition::Not(Box::new(condition))
    }
}

impl From<TransitionCondition> for CompoundCondition {
    fn from(condition: TransitionCondition) -> Self {
        CompoundCondition::Single(condition)
    }
}

// ============================================================================
// Transition Interruption (Unity-like interruption sources)
// ============================================================================

/// Controls when a transition can be interrupted
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TransitionInterruptSource {
    /// Cannot be interrupted once started
    #[default]
    None,
    /// Can be interrupted by transitions from the current state
    CurrentState,
    /// Can be interrupted by transitions from the next state
    NextState,
    /// Can be interrupted by either current or next state transitions
    CurrentOrNextState,
}

// ============================================================================
// Animation Curves (Runtime float sampling for parameter driving)
// ============================================================================

/// A keyframe in an animation curve
#[derive(Clone, Copy, Debug)]
pub struct CurveKeyframe {
    /// Time of the keyframe
    pub time: f32,
    /// Value at this keyframe
    pub value: f32,
    /// Incoming tangent for smooth interpolation (optional)
    pub in_tangent: f32,
    /// Outgoing tangent for smooth interpolation (optional)
    pub out_tangent: f32,
}

impl CurveKeyframe {
    /// Create a linear keyframe
    pub fn linear(time: f32, value: f32) -> Self {
        Self {
            time,
            value,
            in_tangent: 0.0,
            out_tangent: 0.0,
        }
    }

    /// Create a smooth keyframe with tangents
    pub fn smooth(time: f32, value: f32, in_tangent: f32, out_tangent: f32) -> Self {
        Self { time, value, in_tangent, out_tangent }
    }
}

/// Animation curve for sampling float values over time
/// Can drive parameters, material properties, IK weights, etc.
#[derive(Clone, Debug)]
pub struct AnimationCurve {
    /// Name of the curve
    pub name: String,
    /// Keyframes sorted by time
    pub keyframes: Vec<CurveKeyframe>,
    /// Wrap mode for times before first keyframe
    pub pre_wrap: WrapMode,
    /// Wrap mode for times after last keyframe
    pub post_wrap: WrapMode,
}

/// Wrap mode for animation curve evaluation
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WrapMode {
    /// Clamp to boundary value
    #[default]
    Clamp,
    /// Loop the curve
    Loop,
    /// Ping-pong back and forth
    PingPong,
}

impl AnimationCurve {
    /// Create a new empty curve
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            keyframes: Vec::new(),
            pre_wrap: WrapMode::Clamp,
            post_wrap: WrapMode::Clamp,
        }
    }

    /// Create a curve from keyframes
    pub fn from_keyframes(name: &str, keyframes: Vec<CurveKeyframe>) -> Self {
        let mut curve = Self::new(name);
        curve.keyframes = keyframes;
        curve.keyframes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
        curve
    }

    /// Add a keyframe
    pub fn add_keyframe(&mut self, keyframe: CurveKeyframe) {
        self.keyframes.push(keyframe);
        self.keyframes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// Evaluate the curve at a given time
    pub fn evaluate(&self, time: f32) -> f32 {
        if self.keyframes.is_empty() {
            return 0.0;
        }

        if self.keyframes.len() == 1 {
            return self.keyframes[0].value;
        }

        let first = &self.keyframes[0];
        let last = &self.keyframes[self.keyframes.len() - 1];
        let duration = last.time - first.time;

        // Handle wrap modes
        let wrapped_time = if time < first.time {
            match self.pre_wrap {
                WrapMode::Clamp => first.time,
                WrapMode::Loop => {
                    if duration > 0.0 {
                        last.time - ((first.time - time) % duration)
                    } else {
                        first.time
                    }
                }
                WrapMode::PingPong => {
                    if duration > 0.0 {
                        let cycles = ((first.time - time) / duration) as i32;
                        let t = (first.time - time) % duration;
                        if cycles % 2 == 0 {
                            first.time + t
                        } else {
                            last.time - t
                        }
                    } else {
                        first.time
                    }
                }
            }
        } else if time > last.time {
            match self.post_wrap {
                WrapMode::Clamp => last.time,
                WrapMode::Loop => {
                    if duration > 0.0 {
                        first.time + ((time - first.time) % duration)
                    } else {
                        first.time
                    }
                }
                WrapMode::PingPong => {
                    if duration > 0.0 {
                        let cycles = ((time - first.time) / duration) as i32;
                        let t = (time - first.time) % duration;
                        if cycles % 2 == 0 {
                            first.time + t
                        } else {
                            last.time - t
                        }
                    } else {
                        first.time
                    }
                }
            }
        } else {
            time
        };

        // Find surrounding keyframes
        let mut lower_idx = 0;
        for (i, kf) in self.keyframes.iter().enumerate() {
            if kf.time <= wrapped_time {
                lower_idx = i;
            } else {
                break;
            }
        }
        let upper_idx = (lower_idx + 1).min(self.keyframes.len() - 1);

        if lower_idx == upper_idx {
            return self.keyframes[lower_idx].value;
        }

        let kf1 = &self.keyframes[lower_idx];
        let kf2 = &self.keyframes[upper_idx];
        let t = (wrapped_time - kf1.time) / (kf2.time - kf1.time);

        // Hermite interpolation for smooth curves
        let t2 = t * t;
        let t3 = t2 * t;
        let h1 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h2 = -2.0 * t3 + 3.0 * t2;
        let h3 = t3 - 2.0 * t2 + t;
        let h4 = t3 - t2;

        let dt = kf2.time - kf1.time;
        h1 * kf1.value + h2 * kf2.value + h3 * (kf1.out_tangent * dt) + h4 * (kf2.in_tangent * dt)
    }

    /// Get the duration of the curve
    pub fn duration(&self) -> f32 {
        if self.keyframes.len() < 2 {
            return 0.0;
        }
        self.keyframes.last().unwrap().time - self.keyframes.first().unwrap().time
    }
}

#[derive(Clone, Debug)]
pub struct AnimationStateMachine {
    /// Named states mapping to animation resources
    pub states: HashMap<String, AnimationStateMachineState>,
    /// Current state name
    pub current_state: String,
    /// Transition definitions
    pub transitions: Vec<StateTransition>,
    /// Parameters that drive transitions
    pub parameters: HashMap<String, AnimParam>,
    /// Sub-state machines for hierarchical organization
    pub sub_machines: HashMap<String, SubStateMachine>,
    /// Currently active sub-machine (if any)
    pub active_sub_machine: Option<String>,
}

impl AnimationStateMachine {
    /// Get the current animation resource
    pub fn current_resource(&self) -> Option<&AnimResource> {
        // Check if we're in a sub-machine first
        if let Some(sub_name) = &self.active_sub_machine {
            if let Some(sub) = self.sub_machines.get(sub_name) {
                if sub.is_active {
                    return sub.state_machine.current_resource();
                }
            }
        }
        self.states.get(&self.current_state).map(|s| &s.resource)
    }
}

/// A transition between animation states
#[derive(Clone, Debug)]
pub struct StateTransition {
    /// Source state name
    pub from: String,
    /// Destination state name
    pub to: String,
    /// Condition to trigger transition (simple condition for backwards compat)
    pub condition: TransitionCondition,
    /// Compound condition for complex logic (optional, takes precedence)
    pub compound_condition: Option<CompoundCondition>,
    /// Crossfade duration in seconds
    pub duration: f32,
    /// Transition priority (higher = evaluated first)
    pub priority: i32,
    /// Interruption source settings
    pub interrupt_source: TransitionInterruptSource,
}

impl StateTransition {
    /// Create a simple transition
    pub fn new(from: &str, to: &str, condition: TransitionCondition, duration: f32) -> Self {
        Self {
            from: from.to_string(),
            to: to.to_string(),
            condition,
            compound_condition: None,
            duration,
            priority: 0,
            interrupt_source: TransitionInterruptSource::None,
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Set interruption source
    pub fn with_interrupt_source(mut self, source: TransitionInterruptSource) -> Self {
        self.interrupt_source = source;
        self
    }

    /// Set compound condition
    pub fn with_compound_condition(mut self, condition: CompoundCondition) -> Self {
        self.compound_condition = Some(condition);
        self
    }
}

/// Condition for triggering a state transition
#[derive(Clone, Debug)]
pub enum TransitionCondition {
    /// Trigger immediately (one-shot)
    Trigger(String),
    /// Float parameter comparison
    FloatGreater(String, f32),
    FloatLess(String, f32),
    FloatEquals(String, f32, f32), // (name, value, epsilon)
    FloatInRange(String, f32, f32), // (name, min, max)
    /// Bool parameter check
    BoolTrue(String),
    BoolFalse(String),
    /// Animation finished playing
    AnimationComplete,
    /// Integer comparison (for combo counts, etc.)
    IntEquals(String, i32),
    IntGreater(String, i32),
    IntLess(String, i32),
}

/// A state in the animation state machine
#[derive(Clone, Debug)]
pub struct AnimationStateMachineState {
    pub name: String,
    pub resource: AnimResource,
    /// Optional speed multiplier for this state
    pub speed_multiplier: f32,
    /// Animation curves that drive parameters while in this state
    pub parameter_curves: Vec<(String, AnimationCurve)>,
}

impl AnimationStateMachineState {
    /// Create a new state with a clip
    pub fn new(name: &str, clip_index: usize) -> Self {
        Self {
            name: name.to_string(),
            resource: AnimResource::Clip(clip_index),
            speed_multiplier: 1.0,
            parameter_curves: Vec::new(),
        }
    }

    /// Create a new state with a blend tree
    pub fn with_blend_tree(name: &str, blend_tree: BlendTree) -> Self {
        Self {
            name: name.to_string(),
            resource: AnimResource::BlendTree(blend_tree),
            speed_multiplier: 1.0,
            parameter_curves: Vec::new(),
        }
    }

    /// Add a parameter curve
    pub fn with_curve(mut self, param_name: &str, curve: AnimationCurve) -> Self {
        self.parameter_curves.push((param_name.to_string(), curve));
        self
    }

    /// Set speed multiplier
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed_multiplier = speed;
        self
    }
}


impl AnimationStateMachine {
    /// Create a new empty state machine
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
            current_state: String::new(),
            transitions: Vec::new(),
            parameters: HashMap::new(),
            sub_machines: HashMap::new(),
            active_sub_machine: None,
        }
    }

    /// Add a state with a clip index
    pub fn add_state(&mut self, name: &str, clip_index: usize) {
        self.states.insert(
            name.to_string(),
            AnimationStateMachineState::new(name, clip_index),
        );
        if self.current_state.is_empty() {
            self.current_state = name.to_string();
        }
    }

    /// Add a state with a blend tree
    pub fn add_blend_tree(&mut self, name: &str, blend_tree: BlendTree) {
        self.states.insert(
            name.to_string(),
            AnimationStateMachineState::with_blend_tree(name, blend_tree),
        );
        if self.current_state.is_empty() {
            self.current_state = name.to_string();
        }
    }

    /// Add a sub-state machine
    pub fn add_sub_machine(&mut self, sub_machine: SubStateMachine) {
        self.sub_machines.insert(sub_machine.name.clone(), sub_machine);
    }

    /// Enter a sub-state machine by name
    pub fn enter_sub_machine(&mut self, name: &str) -> bool {
        if let Some(sub) = self.sub_machines.get_mut(name) {
            sub.enter();
            self.active_sub_machine = Some(name.to_string());
            true
        } else {
            false
        }
    }

    /// Exit the currently active sub-state machine
    pub fn exit_sub_machine(&mut self) {
        if let Some(name) = self.active_sub_machine.take() {
            if let Some(sub) = self.sub_machines.get_mut(&name) {
                sub.exit();
            }
        }
    }

    /// Set a float parameter
    pub fn set_float(&mut self, name: &str, value: f32) {
        self.parameters.insert(name.to_string(), AnimParam::Float(value));
    }

    /// Set a bool parameter
    pub fn set_bool(&mut self, name: &str, value: bool) {
        self.parameters.insert(name.to_string(), AnimParam::Bool(value));
    }

    /// Set an integer parameter
    pub fn set_int(&mut self, name: &str, value: i32) {
        self.parameters.insert(name.to_string(), AnimParam::Int(value));
    }

    /// Fire a trigger (one-shot)
    pub fn set_trigger(&mut self, name: &str) {
        self.parameters.insert(name.to_string(), AnimParam::Trigger(true));
    }

    /// Get current clip index (only if current state is a single clip)
    pub fn current_clip(&self) -> Option<usize> {
        match self.states.get(&self.current_state)? {
            AnimationStateMachineState { resource: AnimResource::Clip(idx), .. } => Some(*idx),
            _ => None,
        }
    }

    /// Evaluate transitions and return target state resource if any.
    /// Any-state transitions are checked first with higher priority, allowing global interrupts.
    /// Transitions are sorted by priority before evaluation.
    pub fn evaluate(&mut self) -> Option<(AnimResource, f32)> {
        // Check if we're in a sub-machine
        if let Some(sub_name) = self.active_sub_machine.clone() {
            if let Some(sub) = self.sub_machines.get_mut(&sub_name) {
                if let Some((resource, duration, hit_exit)) = sub.evaluate() {
                    if hit_exit {
                        self.active_sub_machine = None;
                        sub.exit();
                        // Continue to check parent transitions
                    } else {
                        return Some((resource, duration));
                    }
                }
            }
        }

        // Phase 1: Check ANY_STATE transitions first (highest priority, like death/damage reactions)
        if let Some(result) = self.check_transitions_from(ANY_STATE) {
            return Some(result);
        }

        // Phase 2: Check transitions from current state
        self.check_transitions_from(&self.current_state.clone())
    }

    /// Check all transitions from a given source state and return the first triggered one
    fn check_transitions_from(&mut self, from_state: &str) -> Option<(AnimResource, f32)> {
        // Collect transition indices sorted by priority (descending)
        let mut transition_indices: Vec<usize> = (0..self.transitions.len())
            .filter(|&i| self.transitions[i].from == from_state)
            .collect();
        transition_indices.sort_by(|&a, &b| {
            self.transitions[b].priority.cmp(&self.transitions[a].priority)
        });

        for i in transition_indices {
            let (matches_self_transition, condition, compound, to_state, duration) = {
                let transition = &self.transitions[i];
                (
                    from_state == ANY_STATE && transition.to == self.current_state,
                    transition.condition.clone(),
                    transition.compound_condition.clone(),
                    transition.to.clone(),
                    transition.duration,
                )
            };

            // Skip self-transitions for ANY_STATE (avoid infinite loops)
            if matches_self_transition {
                continue;
            }

            // Check compound condition first if present, else simple condition
            let triggered = if let Some(compound_cond) = compound {
                self.check_compound_condition(&compound_cond)
            } else {
                self.check_condition(&condition)
            };

            if triggered {
                // Check if transitioning to a sub-machine
                if self.sub_machines.contains_key(&to_state) {
                    self.enter_sub_machine(&to_state);
                    if let Some(sub) = self.sub_machines.get(&to_state) {
                        if let Some(resource) = sub.state_machine.current_resource() {
                            return Some((resource.clone(), duration));
                        }
                    }
                }

                if let Some(state) = self.states.get(&to_state) {
                    let resource = state.resource.clone();
                    self.current_state = to_state;
                    return Some((resource, duration));
                }
            }
        }

        None
    }

    /// Check if a compound condition is satisfied
    fn check_compound_condition(&mut self, condition: &CompoundCondition) -> bool {
        match condition {
            CompoundCondition::Single(cond) => self.check_condition(cond),
            CompoundCondition::And(conditions) => {
                conditions.iter().all(|c| self.check_compound_condition(c))
            }
            CompoundCondition::Or(conditions) => {
                conditions.iter().any(|c| self.check_compound_condition(c))
            }
            CompoundCondition::Not(inner) => !self.check_compound_condition(inner),
        }
    }

    /// Check if a transition condition is satisfied
    fn check_condition(&mut self, condition: &TransitionCondition) -> bool {
        match condition {
            TransitionCondition::Trigger(name) => {
                let is_triggered = matches!(self.parameters.get(name), Some(AnimParam::Trigger(true)));
                // Consuming trigger if it fired
                if is_triggered {
                    self.parameters.insert(name.clone(), AnimParam::Trigger(false));
                }
                is_triggered
            }
            TransitionCondition::FloatGreater(name, threshold) => {
                matches!(self.parameters.get(name), Some(AnimParam::Float(v)) if *v > *threshold)
            }
            TransitionCondition::FloatLess(name, threshold) => {
                matches!(self.parameters.get(name), Some(AnimParam::Float(v)) if *v < *threshold)
            }
            TransitionCondition::FloatEquals(name, value, epsilon) => {
                matches!(self.parameters.get(name), Some(AnimParam::Float(v)) if (*v - *value).abs() <= *epsilon)
            }
            TransitionCondition::FloatInRange(name, min, max) => {
                matches!(self.parameters.get(name), Some(AnimParam::Float(v)) if *v >= *min && *v <= *max)
            }
            TransitionCondition::BoolTrue(name) => {
                matches!(self.parameters.get(name), Some(AnimParam::Bool(true)))
            }
            TransitionCondition::BoolFalse(name) => {
                matches!(self.parameters.get(name), Some(AnimParam::Bool(false)))
            }
            TransitionCondition::AnimationComplete => false, // Handled externally
            TransitionCondition::IntEquals(name, value) => {
                matches!(self.parameters.get(name), Some(AnimParam::Int(v)) if *v == *value)
            }
            TransitionCondition::IntGreater(name, threshold) => {
                matches!(self.parameters.get(name), Some(AnimParam::Int(v)) if *v > *threshold)
            }
            TransitionCondition::IntLess(name, threshold) => {
                matches!(self.parameters.get(name), Some(AnimParam::Int(v)) if *v < *threshold)
            }
        }
    }
}


impl Default for AnimationStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Unity-Like Animation Controller System
// ============================================================================

/// ECS Component: High-level animation controller that links an entity to its
/// animation state machine and manages multi-layer blending. Similar to Unity's Animator.
#[derive(Clone, Debug)]
pub struct AnimatorController {
    /// State machine controlling animation flow
    pub state_machine: AnimationStateMachine,
    /// Animation layers for blending (base layer + optional additives)
    pub layers: Vec<AnimationLayer>,
    /// Enable/disable root motion application to Transform
    pub apply_root_motion: bool,
    /// Speed multiplier for entire controller
    pub speed: f32,
    /// Whether the controller is enabled
    pub enabled: bool,
}

impl AnimatorController {
    /// Create a new controller with a state machine
    pub fn new(state_machine: AnimationStateMachine) -> Self {
        Self {
            state_machine,
            layers: Vec::new(),
            apply_root_motion: false,
            speed: 1.0,
            enabled: true,
        }
    }

    /// Create a simple controller with no state machine (plays single clip)
    pub fn simple() -> Self {
        Self {
            state_machine: AnimationStateMachine::new(),
            layers: Vec::new(),
            apply_root_motion: false,
            speed: 1.0,
            enabled: true,
        }
    }

    /// Set a float parameter on the state machine
    pub fn set_float(&mut self, name: &str, value: f32) {
        self.state_machine.set_float(name, value);
    }

    /// Set a bool parameter on the state machine
    pub fn set_bool(&mut self, name: &str, value: bool) {
        self.state_machine.set_bool(name, value);
    }

    /// Fire a trigger (consumed after transition)
    pub fn set_trigger(&mut self, name: &str) {
        self.state_machine.set_trigger(name);
    }

    /// Get current state name
    pub fn current_state(&self) -> &str {
        &self.state_machine.current_state
    }
}

impl Default for AnimatorController {
    fn default() -> Self {
        Self::simple()
    }
}

/// ECS Component: Queue of animation events fired this frame, cleared each tick
#[derive(Clone, Debug, Default)]
pub struct AnimationEventQueue {
    /// Events that fired this frame
    pub events: Vec<AnimationEvent>,
}

impl AnimationEventQueue {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    /// Add an event to the queue
    pub fn push(&mut self, event: AnimationEvent) {
        self.events.push(event);
    }

    /// Clear all events (call at end of frame)
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Check if any events with given name fired
    pub fn has_event(&self, name: &str) -> bool {
        self.events.iter().any(|e| e.name == name)
    }

    /// Get all events with given name
    pub fn get_events(&self, name: &str) -> Vec<&AnimationEvent> {
        self.events.iter().filter(|e| e.name == name).collect()
    }
}

/// Root motion delta extracted from animation for character movement
#[derive(Clone, Copy, Debug, Default)]
pub struct RootMotion {
    /// Position delta from root bone animation this frame
    pub delta_position: Vec3,
    /// Rotation delta from root bone animation this frame
    pub delta_rotation: Quat,
    /// Whether root motion was extracted this frame
    pub has_motion: bool,
}

impl RootMotion {
    pub fn new() -> Self {
        Self {
            delta_position: Vec3::ZERO,
            delta_rotation: Quat::IDENTITY,
            has_motion: false,
        }
    }

    /// Reset deltas (call after applying motion)
    pub fn reset(&mut self) {
        self.delta_position = Vec3::ZERO;
        self.delta_rotation = Quat::IDENTITY;
        self.has_motion = false;
    }
}

/// Simple Two-Bone IK Solver (e.g. for legs and arms)
#[derive(Clone, Debug)]
pub struct IKSolver {
    pub root_bone: usize,
    pub mid_bone: usize,
    pub tip_bone: usize,
    pub target: Vec3,
    pub pole_target: Vec3,
    pub enabled: bool,
    /// Weight of the IK influence (0.0 to 1.0)
    pub weight: f32,
}

impl IKSolver {
    pub fn new(root: usize, mid: usize, tip: usize) -> Self {
        Self {
            root_bone: root,
            mid_bone: mid,
            tip_bone: tip,
            target: Vec3::ZERO,
            pole_target: Vec3::new(0.0, 0.0, 1.0),
            enabled: true,
            weight: 1.0,
        }
    }
}

/// Solve two-bone IK and return local rotations for root and mid bones
pub fn solve_two_bone_ik(
    root_world_pos: Vec3,
    mid_local_pos: Vec3,
    tip_local_pos: Vec3,
    target_world_pos: Vec3,
    pole_world_pos: Vec3,
    _root_world_rot: Quat,
) -> (Quat, Quat) {
    let limb1_len = mid_local_pos.length();
    let limb2_len = tip_local_pos.length();
    let total_len = limb1_len + limb2_len;

    let to_target = target_world_pos - root_world_pos;
    let dist = to_target.length().min(total_len * 0.999);
    let to_target_norm = to_target / dist.max(0.001);

    // Law of cosines to find interior angles
    // a^2 = b^2 + c^2 - 2bc*cos(A)
    // cos(root_angle) = (limb1^2 + dist^2 - limb2^2) / (2 * limb1 * dist)
    let cos_root = (limb1_len * limb1_len + dist * dist - limb2_len * limb2_len) / (2.0 * limb1_len * dist);
    let root_angle = cos_root.clamp(-1.0, 1.0).acos();

    // cos(mid_angle) = (limb1^2 + limb2^2 - dist^2) / (2 * limb1 * limb2)
    let cos_mid = (limb1_len * limb1_len + limb2_len * limb2_len - dist * dist) / (2.0 * limb1_len * limb2_len);
    let mid_angle = PI - cos_mid.clamp(-1.0, 1.0).acos();

    // Calculate plane for the limb bending
    let to_pole = pole_world_pos - root_world_pos;
    let bend_normal = to_target_norm.cross(to_pole).normalize();
    let bend_dir = bend_normal.cross(to_target_norm).normalize();

    // Root rotation: Rotate towards target, then offset by root_angle along bend_dir
    let _base_rot = Quat::from_rotation_arc(Vec3::Z, to_target_norm); // Simplified base
    // This needs to be relative to parent... 
    // For simplicity in this implementation, we compute world rotations and the user must convert back if needed,
    // OR we assume root_world_rot is the parent's world rotation.
    
    // Final rotations (simplified for now)
    let final_root_rot = Quat::from_rotation_arc(mid_local_pos.normalize(), (to_target_norm * root_angle.cos() + bend_dir * root_angle.sin()).normalize());
    let final_mid_rot = Quat::from_rotation_y(mid_angle); // Simplified: bend around Y axis

    (final_root_rot, final_mid_rot)
}

// ============================================================================
// Advanced IK Systems (Unity-like)
// ============================================================================

/// Look-At IK for head and spine chain aiming/looking at targets
/// Like Unity's Animator.SetLookAtWeight and SetLookAtPosition
#[derive(Clone, Debug)]
pub struct LookAtIK {
    /// Target position in world space
    pub target: Vec3,
    /// Overall weight (0.0 = disabled, 1.0 = full effect)
    pub weight: f32,
    /// Body weight (how much spine participates)
    pub body_weight: f32,
    /// Head weight (how much head participates)
    pub head_weight: f32,
    /// Eyes weight (for future eye tracking)
    pub eyes_weight: f32,
    /// Maximum horizontal turn angle in radians
    pub clamp_horizontal: f32,
    /// Maximum vertical tilt angle in radians
    pub clamp_vertical: f32,
    /// Bone indices for the look-at chain: [head, neck, spine2, spine1, spine] (optional)
    /// Head is required, others are optional for distributed rotation
    pub bone_chain: Vec<usize>,
    /// Whether the solver is enabled
    pub enabled: bool,
}

impl LookAtIK {
    /// Create a new look-at IK solver with just a head bone
    pub fn new(head_bone: usize) -> Self {
        Self {
            target: Vec3::ZERO,
            weight: 1.0,
            body_weight: 0.5,
            head_weight: 1.0,
            eyes_weight: 0.0,
            clamp_horizontal: PI * 0.5, // 90 degrees
            clamp_vertical: PI * 0.25,   // 45 degrees
            bone_chain: vec![head_bone],
            enabled: true,
        }
    }

    /// Create with full humanoid chain
    pub fn humanoid(head: usize, neck: usize, spine2: usize, spine1: usize) -> Self {
        Self {
            target: Vec3::ZERO,
            weight: 1.0,
            body_weight: 0.3,
            head_weight: 1.0,
            eyes_weight: 0.0,
            clamp_horizontal: PI * 0.5,
            clamp_vertical: PI * 0.25,
            bone_chain: vec![head, neck, spine2, spine1],
            enabled: true,
        }
    }

    /// Set the look-at target
    pub fn set_target(&mut self, target: Vec3) {
        self.target = target;
    }

    /// Set overall weight (like Unity's SetLookAtWeight)
    pub fn set_weight(&mut self, weight: f32, body: f32, head: f32, eyes: f32) {
        self.weight = weight.clamp(0.0, 1.0);
        self.body_weight = body.clamp(0.0, 1.0);
        self.head_weight = head.clamp(0.0, 1.0);
        self.eyes_weight = eyes.clamp(0.0, 1.0);
    }
}

/// Solve look-at IK and return rotation adjustments for each bone in chain
pub fn solve_look_at_ik(
    look_at: &LookAtIK,
    bone_world_positions: &[Vec3],
    bone_world_rotations: &[Quat],
    bone_forward: Vec3, // Default forward direction (usually -Z or +Z)
) -> Vec<Quat> {
    if !look_at.enabled || look_at.weight <= 0.0 || look_at.bone_chain.is_empty() {
        return vec![Quat::IDENTITY; look_at.bone_chain.len()];
    }

    let chain_len = look_at.bone_chain.len();
    let mut rotations = vec![Quat::IDENTITY; chain_len];

    // Get head position (first bone in chain)
    let head_idx = look_at.bone_chain[0];
    if head_idx >= bone_world_positions.len() {
        return rotations;
    }

    let head_pos = bone_world_positions[head_idx];
    let to_target = (look_at.target - head_pos).normalize();
    
    // Calculate the desired rotation to look at target
    let current_forward = bone_world_rotations.get(head_idx)
        .map(|r| *r * bone_forward)
        .unwrap_or(bone_forward);
    
    // Full rotation needed
    let full_rotation = Quat::from_rotation_arc(current_forward, to_target);
    
    // Clamp the rotation angles
    let (axis, angle) = full_rotation.to_axis_angle();
    let clamped_angle = angle.min(look_at.clamp_horizontal).min(look_at.clamp_vertical);
    let clamped_rotation = Quat::from_axis_angle(axis, clamped_angle);

    // Distribute rotation across chain
    // Head gets most, spine bones get progressively less
    for (i, _bone_idx) in look_at.bone_chain.iter().enumerate() {
        let bone_weight = if i == 0 {
            look_at.head_weight
        } else {
            // Diminishing weight for spine bones
            look_at.body_weight * (1.0 / (i as f32 + 1.0))
        };
        
        let final_weight = look_at.weight * bone_weight;
        rotations[i] = Quat::IDENTITY.slerp(clamped_rotation, final_weight);
    }

    rotations
}

/// Foot IK for ground adaptation - places feet on uneven terrain
/// Requires physics raycast integration to function fully
#[derive(Clone, Debug)]
pub struct FootIK {
    /// Left leg bone indices: (thigh, calf, foot)
    pub left_leg: (usize, usize, usize),
    /// Right leg bone indices: (thigh, calf, foot)
    pub right_leg: (usize, usize, usize),
    /// Hips bone index for height adjustment
    pub hips_bone: usize,
    /// Overall weight (0.0 = disabled)
    pub weight: f32,
    /// Whether the solver is enabled
    pub enabled: bool,
    /// Height offset for raycasting from foot
    pub raycast_height: f32,
    /// Maximum distance feet can move from animated position
    pub max_correction: f32,
    /// Maximum hip adjustment for slope adaptation
    pub max_hip_offset: f32,
    /// Left foot ground hit (set externally from physics raycast)
    pub left_foot_target: Option<FootIKTarget>,
    /// Right foot ground hit (set externally from physics raycast)
    pub right_foot_target: Option<FootIKTarget>,
}

/// Target data for foot IK from raycast result
#[derive(Clone, Copy, Debug)]
pub struct FootIKTarget {
    /// World position where foot should be placed
    pub position: Vec3,
    /// Ground normal for foot rotation
    pub normal: Vec3,
    /// Distance from animated foot to ground
    pub height_offset: f32,
}

impl FootIK {
    /// Create foot IK with default humanoid bone indices
    /// You should override these with your actual skeleton bone indices
    pub fn new(
        left_thigh: usize, left_calf: usize, left_foot: usize,
        right_thigh: usize, right_calf: usize, right_foot: usize,
        hips: usize,
    ) -> Self {
        Self {
            left_leg: (left_thigh, left_calf, left_foot),
            right_leg: (right_thigh, right_calf, right_foot),
            hips_bone: hips,
            weight: 1.0,
            enabled: true,
            raycast_height: 0.5,
            max_correction: 0.3,
            max_hip_offset: 0.2,
            left_foot_target: None,
            right_foot_target: None,
        }
    }

    /// Set foot target from raycast result
    pub fn set_left_foot_target(&mut self, position: Vec3, normal: Vec3, height_offset: f32) {
        self.left_foot_target = Some(FootIKTarget { position, normal, height_offset });
    }

    /// Set foot target from raycast result
    pub fn set_right_foot_target(&mut self, position: Vec3, normal: Vec3, height_offset: f32) {
        self.right_foot_target = Some(FootIKTarget { position, normal, height_offset });
    }

    /// Clear foot targets (call when character is airborne)
    pub fn clear_targets(&mut self) {
        self.left_foot_target = None;
        self.right_foot_target = None;
    }
}

/// Result of foot IK solving
#[derive(Clone, Debug, Default)]
pub struct FootIKResult {
    /// Hip vertical offset adjustment
    pub hip_offset: f32,
    /// Left leg IK rotations: (thigh_rot, calf_rot)
    pub left_leg_rotations: (Quat, Quat),
    /// Right leg IK rotations
    pub right_leg_rotations: (Quat, Quat),
    /// Left foot rotation for ground alignment
    pub left_foot_rotation: Quat,
    /// Right foot rotation for ground alignment
    pub right_foot_rotation: Quat,
}

/// Solve foot IK given current bone positions
pub fn solve_foot_ik(
    foot_ik: &FootIK,
    bone_world_positions: &[Vec3],
) -> FootIKResult {
    if !foot_ik.enabled || foot_ik.weight <= 0.0 {
        return FootIKResult::default();
    }

    let mut result = FootIKResult::default();
    
    // Calculate hip offset based on foot height differences
    let mut min_offset = 0.0f32;
    
    if let Some(left_target) = &foot_ik.left_foot_target {
        let offset = left_target.height_offset.min(foot_ik.max_correction);
        min_offset = min_offset.min(-offset);
        
        // Calculate foot rotation to align with ground normal
        let up = Vec3::Y;
        if left_target.normal.dot(up) > 0.1 {
            result.left_foot_rotation = Quat::from_rotation_arc(up, left_target.normal);
        }
    }
    
    if let Some(right_target) = &foot_ik.right_foot_target {
        let offset = right_target.height_offset.min(foot_ik.max_correction);
        min_offset = min_offset.min(-offset);
        
        let up = Vec3::Y;
        if right_target.normal.dot(up) > 0.1 {
            result.right_foot_rotation = Quat::from_rotation_arc(up, right_target.normal);
        }
    }
    
    // Apply hip offset (clamped)
    result.hip_offset = min_offset.clamp(-foot_ik.max_hip_offset, foot_ik.max_hip_offset) * foot_ik.weight;
    
    // TODO: Use two-bone IK to solve leg rotations for precise foot placement
    // For now, return identity rotations and let the animation handle leg bending
    // Full implementation would:
    // 1. Adjust foot target by hip offset
    // 2. Call solve_two_bone_ik for each leg
    // 3. Return the computed rotations
    
    // Get leg positions for IK (if available)
    let _ = bone_world_positions; // Used in full implementation
    
    result
}

use std::f32::consts::PI;

// ============================================================================
// Editor Configuration Types (Serializable for SceneUpdate protocol)
// ============================================================================

/// Parameter type for editor configuration
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum AnimParamConfig {
    Float(f32),
    Bool(bool),
    Int(i32),
    Trigger,
}

impl Default for AnimParamConfig {
    fn default() -> Self {
        AnimParamConfig::Float(0.0)
    }
}

/// Condition configuration for transitions
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum TransitionConditionConfig {
    Trigger(String),
    FloatGreater(String, f32),
    FloatLess(String, f32),
    FloatEquals(String, f32, f32),
    FloatInRange(String, f32, f32),
    BoolTrue(String),
    BoolFalse(String),
    AnimationComplete,
    IntEquals(String, i32),
    IntGreater(String, i32),
    IntLess(String, i32),
}

impl Default for TransitionConditionConfig {
    fn default() -> Self {
        TransitionConditionConfig::AnimationComplete
    }
}

impl From<TransitionConditionConfig> for TransitionCondition {
    fn from(cfg: TransitionConditionConfig) -> Self {
        match cfg {
            TransitionConditionConfig::Trigger(s) => TransitionCondition::Trigger(s),
            TransitionConditionConfig::FloatGreater(s, v) => TransitionCondition::FloatGreater(s, v),
            TransitionConditionConfig::FloatLess(s, v) => TransitionCondition::FloatLess(s, v),
            TransitionConditionConfig::FloatEquals(s, v, e) => TransitionCondition::FloatEquals(s, v, e),
            TransitionConditionConfig::FloatInRange(s, min, max) => TransitionCondition::FloatInRange(s, min, max),
            TransitionConditionConfig::BoolTrue(s) => TransitionCondition::BoolTrue(s),
            TransitionConditionConfig::BoolFalse(s) => TransitionCondition::BoolFalse(s),
            TransitionConditionConfig::AnimationComplete => TransitionCondition::AnimationComplete,
            TransitionConditionConfig::IntEquals(s, v) => TransitionCondition::IntEquals(s, v),
            TransitionConditionConfig::IntGreater(s, v) => TransitionCondition::IntGreater(s, v),
            TransitionConditionConfig::IntLess(s, v) => TransitionCondition::IntLess(s, v),
        }
    }
}

/// State configuration for editor
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AnimatorStateConfig {
    pub name: String,
    pub clip_index: usize,
    pub speed_multiplier: f32,
    pub is_blend_tree: bool,
    /// For blend tree: blend tree type (0=1D, 1=2D)
    pub blend_tree_type: u8,
    /// For blend tree: parameter name(s)
    pub blend_param: String,
    pub blend_param_y: Option<String>,
    /// Optional parameter name that triggers a transition to this state from ANY
    pub trigger_param: Option<String>,
}

impl Default for AnimatorStateConfig {
    fn default() -> Self {
        Self {
            name: "State".to_string(),
            clip_index: 0,
            speed_multiplier: 1.0,
            is_blend_tree: false,
            blend_tree_type: 0,
            blend_param: String::new(),
            blend_param_y: None,
            trigger_param: None,
        }
    }
}

/// Transition configuration for editor
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct TransitionConfig {
    pub from_state: String,
    pub to_state: String,
    pub condition: TransitionConditionConfig,
    pub duration: f32,
    pub priority: i32,
}

/// Animation layer configuration for editor
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LayerConfig {
    pub name: String,
    pub clip_index: usize,
    pub weight: f32,
    pub additive: bool,
    pub sync_to_base: bool,
    /// Mask type: 0=Full, 1=UpperBody, 2=LowerBody, 3=Custom
    pub mask_type: u8,
    /// Custom bone indices if mask_type == 3
    pub mask_bones: Vec<usize>,
}

impl Default for LayerConfig {
    fn default() -> Self {
        Self {
            name: "Layer".to_string(),
            clip_index: 0,
            weight: 1.0,
            additive: false,
            sync_to_base: false,
            mask_type: 0,
            mask_bones: Vec::new(),
        }
    }
}

/// Complete animator configuration for editor-engine communication
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, Default)]
pub struct AnimatorConfig {
    /// Animation states
    pub states: Vec<AnimatorStateConfig>,
    /// State transitions
    pub transitions: Vec<TransitionConfig>,
    /// Parameters (name, config)
    pub parameters: Vec<(String, AnimParamConfig)>,
    /// Additional animation layers
    pub layers: Vec<LayerConfig>,
    /// Default state to start in
    pub default_state: String,
    /// Enable look-at IK
    pub look_at_ik_enabled: bool,
    /// Look-at IK bone chain (head, neck, upper_spine, lower_spine)
    pub look_at_bones: Vec<usize>,
    /// Enable foot IK
    pub foot_ik_enabled: bool,
    /// Foot IK bones: [left_thigh, left_calf, left_foot, right_thigh, right_calf, right_foot, hips]
    pub foot_ik_bones: Vec<usize>,
    /// Apply root motion to transform
    pub apply_root_motion: bool,
    /// Global speed multiplier
    pub speed: f32,
    /// Metadata: Animation clip names
    pub clip_names: Vec<String>,
    /// Metadata: Animation clip durations
    pub clip_durations: Vec<f32>,
    /// Metadata: Bone names for hierarchy display
    pub bone_names: Vec<String>,
    /// Metadata: Bone parent indices
    pub bone_parents: Vec<Option<usize>>,
    /// Metadata: Local bind pose transforms
    pub bone_transforms: Vec<Mat4>,
    /// Editor keyframes for custom animation editing
    pub editor_keyframes: Vec<EditorKeyframe>,
    /// Custom clip metadata for editor-created animations
    pub editor_clips: Vec<EditorClipInfo>,
}

impl AnimatorConfig {
    /// Create a new default config
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            transitions: Vec::new(),
            parameters: Vec::new(),
            layers: Vec::new(),
            default_state: String::new(),
            look_at_ik_enabled: false,
            look_at_bones: Vec::new(),
            foot_ik_enabled: false,
            foot_ik_bones: Vec::new(),
            apply_root_motion: false,
            speed: 1.0,
            clip_names: Vec::new(),
            clip_durations: Vec::new(),
            bone_names: Vec::new(),
            bone_parents: Vec::new(),
            bone_transforms: Vec::new(),
            editor_keyframes: Vec::new(),
            editor_clips: Vec::new(),
        }
    }

    /// Add a state
    pub fn add_state(&mut self, name: &str, clip_index: usize) {
        self.states.push(AnimatorStateConfig {
            name: name.to_string(),
            clip_index,
            ..Default::default()
        });
        if self.default_state.is_empty() {
            self.default_state = name.to_string();
        }
    }

    /// Add a transition
    pub fn add_transition(&mut self, from: &str, to: &str, condition: TransitionConditionConfig, duration: f32) {
        self.transitions.push(TransitionConfig {
            from_state: from.to_string(),
            to_state: to.to_string(),
            condition,
            duration,
            priority: 0,
        });
    }

    /// Add a parameter
    pub fn add_float(&mut self, name: &str, value: f32) {
        self.parameters.push((name.to_string(), AnimParamConfig::Float(value)));
    }

    pub fn add_bool(&mut self, name: &str, value: bool) {
        self.parameters.push((name.to_string(), AnimParamConfig::Bool(value)));
    }

    pub fn add_trigger(&mut self, name: &str) {
        self.parameters.push((name.to_string(), AnimParamConfig::Trigger));
    }
}

/// Editor-side keyframe representation (serializable)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EditorKeyframe {
    /// Time in seconds
    pub time: f32,
    /// Bone index this keyframe affects
    pub bone_index: usize,
    /// Channel (Position, Rotation, Scale)
    pub channel: KeyframeChannel,
    /// Value based on channel type
    pub value: KeyframeValue,
    /// Interpolation type
    pub interpolation: KeyframeInterpolation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum KeyframeChannel {
    PositionX, PositionY, PositionZ,
    RotationX, RotationY, RotationZ, RotationW,
    ScaleX, ScaleY, ScaleZ,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum KeyframeValue {
    Float(f32),
    Vec3([f32; 3]),
    Quat([f32; 4]),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum KeyframeInterpolation {
    #[default]
    Linear,
    Step,
    Bezier,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct EditorClipInfo {
    pub name: String,
    pub duration: f32,
    pub loop_mode: bool,
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_blend_poses_identity() {
        let a = vec![Mat4::IDENTITY, Mat4::IDENTITY];
        let b = vec![Mat4::IDENTITY, Mat4::IDENTITY];
        let result = blend_poses(&a, &b, 0.5);
        assert_eq!(result.len(), 2);
        // Identity blended with identity should be identity
        for mat in result {
            let (s, _r, t) = mat.to_scale_rotation_translation();
            assert!((s - Vec3::ONE).length() < 0.001);
            assert!((t - Vec3::ZERO).length() < 0.001);
        }
    }

    #[test]
    fn test_animation_state_gpu_buffer() {
        let state = AnimationState::new(4);
        let buffer = state.as_gpu_buffer();
        // Should have MAX_BONES * 16 floats
        assert_eq!(buffer.len(), MAX_BONES * 16);
    }

    #[test]
    fn test_state_machine_add_state() {
        let mut sm = AnimationStateMachine::new();
        sm.add_state("idle", 0);
        sm.add_state("walk", 1);
        assert_eq!(sm.current_state, "idle");
        assert_eq!(sm.current_clip(), Some(0));
    }

    #[test]
    fn test_blend_tree_1d_creation() {
        let tree = BlendTree::locomotion_1d(0, 1, 2);
        assert_eq!(tree.name, "Locomotion");
        match &tree.root {
            BlendNode::Blend1D { param, children } => {
                assert_eq!(param, "Speed");
                assert_eq!(children.len(), 3);
            }
            _ => panic!("Expected Blend1D node"),
        }
    }

    #[test]
    fn test_blend_tree_2d_creation() {
        let tree = BlendTree::locomotion_2d(0, 1, 2, 3, 4);
        assert_eq!(tree.name, "Locomotion2D");
        match &tree.root {
            BlendNode::Blend2D { param_x, param_y, children } => {
                assert_eq!(param_x, "VelocityX");
                assert_eq!(param_y, "VelocityY");
                assert_eq!(children.len(), 5);
            }
            _ => panic!("Expected Blend2D node"),
        }
    }

    #[test]
    fn test_animator_controller_params() {
        let mut controller = AnimatorController::default();
        controller.set_float("Speed", 2.5);
        controller.set_bool("IsGrounded", true);
        controller.set_trigger("Jump");
        
        match controller.state_machine.parameters.get("Speed") {
            Some(AnimParam::Float(v)) => assert!((v - 2.5).abs() < 0.001),
            _ => panic!("Expected float parameter"),
        }
        match controller.state_machine.parameters.get("IsGrounded") {
            Some(AnimParam::Bool(v)) => assert!(*v),
            _ => panic!("Expected bool parameter"),
        }
    }

    #[test]
    fn test_state_machine_transition() {
        let mut sm = AnimationStateMachine::new();
        sm.add_state("idle", 0);
        let tree = BlendTree::locomotion_1d(1, 2, 3);
        sm.add_blend_tree("locomotion", tree);
        
        sm.transitions.push(StateTransition::new(
            "idle",
            "locomotion",
            TransitionCondition::FloatGreater("Speed".to_string(), 0.1),
            0.2,
        ));
        
        // No transition yet
        assert!(sm.evaluate().is_none());
        
        // Trigger transition
        sm.set_float("Speed", 0.5);
        let result = sm.evaluate();
        assert!(result.is_some());
        let (resource, duration) = result.unwrap();
        assert_eq!(duration, 0.2);
        match resource {
            AnimResource::BlendTree(_) => {},
            _ => panic!("Expected blend tree resource"),
        }
        assert_eq!(sm.current_state, "locomotion");
    }

    #[test]
    fn test_animation_events() {
        let skeleton = Arc::new(Skeleton::new());
        let mut clip = Arc::new(AnimationClip::new("test", 1.0));
        
        // Add an event at 0.5s
        Arc::get_mut(&mut clip).unwrap().events.push(AnimationEvent::new(0.5, "footstep"));
        
        let mut animator = Animator::new(skeleton, vec![clip]);
        
        // Use a real world to get a RefMut
        let mut world = hecs::World::new();
        let entity = world.spawn((AnimationEventQueue::new(),));
        
        {
            let q = world.get::<&mut AnimationEventQueue>(entity).unwrap();
            let mut opt_q = Some(q);
            
            // Advance from 0.0 to 0.4 (no event)
            animator.update_with_params(0.4, &HashMap::new(), &mut opt_q);
            assert_eq!(opt_q.as_ref().unwrap().events.len(), 0);
            
            // Advance from 0.4 to 0.6 (fires event)
            animator.update_with_params(0.2, &HashMap::new(), &mut opt_q);
            assert_eq!(opt_q.as_ref().unwrap().events.len(), 1);
            assert_eq!(opt_q.as_ref().unwrap().events[0].name, "footstep");
        }
    }

    #[test]
    fn test_any_state_transition() {
        let mut sm = AnimationStateMachine::new();
        sm.add_state("idle", 0);
        sm.add_state("walk", 1);
        sm.add_state("death", 2);
        
        // Normal transition from idle to walk
        sm.transitions.push(StateTransition::new(
            "idle",
            "walk",
            TransitionCondition::FloatGreater("Speed".to_string(), 0.1),
            0.2,
        ));
        
        // ANY_STATE transition to death (global interrupt)
        sm.transitions.push(StateTransition::new(
            ANY_STATE,
            "death",
            TransitionCondition::Trigger("Die".to_string()),
            0.1,
        ));
        
        // Start in idle
        assert_eq!(sm.current_state, "idle");
        
        // Trigger death from idle - any-state should have priority
        sm.set_trigger("Die");
        let result = sm.evaluate();
        assert!(result.is_some());
        let (_, duration) = result.unwrap();
        assert_eq!(duration, 0.1);
        assert_eq!(sm.current_state, "death");
    }

    #[test]
    fn test_any_state_skips_self_transition() {
        let mut sm = AnimationStateMachine::new();
        sm.add_state("idle", 0);
        sm.add_state("walk", 1);
        
        // ANY_STATE that would transition to current state (should be skipped)
        sm.transitions.push(StateTransition::new(
            ANY_STATE,
            "idle",
            TransitionCondition::BoolTrue("ShouldIdle".to_string()),
            0.1,
        ));
        
        sm.set_bool("ShouldIdle", true);
        
        // Should not transition since we're already in idle
        let result = sm.evaluate();
        assert!(result.is_none());
        assert_eq!(sm.current_state, "idle");
    }

    #[test]
    fn test_avatar_mask_creation() {
        let mask = AvatarMask::full(10);
        assert_eq!(mask.name, "Full");
        assert_eq!(mask.bone_weights.len(), 10);
        assert!(mask.bone_weights.iter().all(|&w| w == 1.0));
        
        let empty_mask = AvatarMask::empty(10);
        assert!(empty_mask.bone_weights.iter().all(|&w| w == 0.0));
        
        let partial_mask = AvatarMask::from_bones("Partial", 10, &[2, 5, 7]);
        assert_eq!(partial_mask.get_weight(0), 0.0);
        assert_eq!(partial_mask.get_weight(2), 1.0);
        assert_eq!(partial_mask.get_weight(5), 1.0);
        assert_eq!(partial_mask.get_weight(7), 1.0);
        assert_eq!(partial_mask.get_weight(9), 0.0);
    }

    #[test]
    fn test_avatar_mask_humanoid_presets() {
        let upper = AvatarMask::upper_body_humanoid(20, 10);
        assert_eq!(upper.name, "UpperBody");
        // Bones 0-9 should have 0.0, bones 10-19 should have 1.0
        for i in 0..10 {
            assert_eq!(upper.get_weight(i), 0.0);
        }
        for i in 10..20 {
            assert_eq!(upper.get_weight(i), 1.0);
        }
        
        let lower = AvatarMask::lower_body_humanoid(20, 10);
        assert_eq!(lower.name, "LowerBody");
        for i in 0..10 {
            assert_eq!(lower.get_weight(i), 1.0);
        }
        for i in 10..20 {
            assert_eq!(lower.get_weight(i), 0.0);
        }
    }

    #[test]
    fn test_look_at_ik_creation() {
        let look_at = LookAtIK::new(10);
        assert_eq!(look_at.bone_chain.len(), 1);
        assert_eq!(look_at.bone_chain[0], 10);
        assert!(look_at.enabled);
        assert_eq!(look_at.weight, 1.0);
        
        let humanoid_look_at = LookAtIK::humanoid(15, 14, 12, 10);
        assert_eq!(humanoid_look_at.bone_chain.len(), 4);
        assert_eq!(humanoid_look_at.bone_chain[0], 15); // head
        assert_eq!(humanoid_look_at.body_weight, 0.3);
    }

    #[test]
    fn test_foot_ik_creation() {
        let foot_ik = FootIK::new(1, 2, 3, 4, 5, 6, 0);
        assert_eq!(foot_ik.left_leg, (1, 2, 3));
        assert_eq!(foot_ik.right_leg, (4, 5, 6));
        assert_eq!(foot_ik.hips_bone, 0);
        assert!(foot_ik.enabled);
        assert!(foot_ik.left_foot_target.is_none());
        assert!(foot_ik.right_foot_target.is_none());
    }

    #[test]
    fn test_animation_layer_weight_blending() {
        let skeleton = Arc::new(Skeleton::new());
        let clip = Arc::new(AnimationClip::new("test", 1.0));
        let mut layer = AnimationLayer::new(clip);
        
        // Start at weight 1.0
        assert_eq!(layer.weight, 1.0);
        
        // Blend to 0.5 at speed 2.0 (should take 0.25 seconds)
        layer.blend_to(0.5, 2.0);
        assert_eq!(layer.target_weight, 0.5);
        
        // Update for 0.1 seconds - should move 0.2 towards target
        layer.update(0.1, None);
        assert!((layer.weight - 0.8).abs() < 0.001);
        
        // Update for 0.2 more seconds - should reach target
        layer.update(0.2, None);
        assert!((layer.weight - 0.5).abs() < 0.001);
    }
}
