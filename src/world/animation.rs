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
        }
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

        // Sample current animation
        let current_pose = self.sample_resource(&self.current_resource, self.time, params);

        // Handle blending (simplified: events from target clip don't fire during crossfade yet)
        if let Some(target_resource) = self.blend_target.clone() {
            self.blend_factor += dt / self.blend_duration;
            
            if self.blend_factor >= 1.0 {
                // Blend complete, switch to target
                self.current_resource = target_resource;
                self.blend_target = None;
                self.blend_factor = 0.0;
                return self.sample_resource(&self.current_resource, self.time, params);
            }

            let target_pose = self.sample_resource(&target_resource, self.time, params);
            return blend_poses(&current_pose, &target_pose, self.blend_factor);
        }

        current_pose
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

/// Animation layer for additive/partial blending
#[derive(Clone, Debug)]
pub struct AnimationLayer {
    /// The animation clip for this layer
    pub clip: Arc<AnimationClip>,
    /// Blend weight (0.0 = no effect, 1.0 = full)
    pub weight: f32,
    /// Current playback time
    pub time: f32,
    /// Playback speed
    pub speed: f32,
    /// Optional bone mask (true = affected, false = ignored)
    pub mask: Option<Vec<bool>>,
    /// Whether this is an additive layer
    pub additive: bool,
}

impl AnimationLayer {
    /// Create a new animation layer
    pub fn new(clip: Arc<AnimationClip>) -> Self {
        Self {
            clip,
            weight: 1.0,
            time: 0.0,
            speed: 1.0,
            mask: None,
            additive: false,
        }
    }

    /// Update layer time
    pub fn update(&mut self, dt: f32) {
        self.time += dt * self.speed;
        if self.time >= self.clip.duration {
            self.time %= self.clip.duration;
        }
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
}

impl AnimationStateMachine {
    /// Get the current animation resource
    pub fn current_resource(&self) -> Option<&AnimResource> {
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
    /// Condition to trigger transition
    pub condition: TransitionCondition,
    /// Crossfade duration in seconds
    pub duration: f32,
}

/// Condition for triggering a state transition
#[derive(Clone, Debug)]
pub enum TransitionCondition {
    /// Trigger immediately (one-shot)
    Trigger(String),
    /// Float parameter comparison
    FloatGreater(String, f32),
    FloatLess(String, f32),
    /// Bool parameter check
    BoolTrue(String),
    BoolFalse(String),
    /// Animation finished playing
    AnimationComplete,
}

/// A state in the animation state machine
#[derive(Clone, Debug)]
pub struct AnimationStateMachineState {
    pub name: String,
    pub resource: AnimResource,
}

impl AnimationStateMachine {
    /// Create a new empty state machine
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
            current_state: String::new(),
            transitions: Vec::new(),
            parameters: HashMap::new(),
        }
    }

    /// Add a state with a clip index
    pub fn add_state(&mut self, name: &str, clip_index: usize) {
        self.states.insert(
            name.to_string(),
            AnimationStateMachineState {
                name: name.to_string(),
                resource: AnimResource::Clip(clip_index),
            },
        );
        if self.current_state.is_empty() {
            self.current_state = name.to_string();
        }
    }

    /// Add a state with a blend tree
    pub fn add_blend_tree(&mut self, name: &str, blend_tree: BlendTree) {
        self.states.insert(
            name.to_string(),
            AnimationStateMachineState {
                name: name.to_string(),
                resource: AnimResource::BlendTree(blend_tree),
            },
        );
        if self.current_state.is_empty() {
            self.current_state = name.to_string();
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

    /// Evaluate transitions and return target state resource if any
    pub fn evaluate(&mut self) -> Option<(AnimResource, f32)> {
        let mut result = None;

        for transition in &self.transitions {
            if transition.from != self.current_state {
                continue;
            }

            let triggered = match &transition.condition {
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
                TransitionCondition::BoolTrue(name) => {
                    matches!(self.parameters.get(name), Some(AnimParam::Bool(true)))
                }
                TransitionCondition::BoolFalse(name) => {
                    matches!(self.parameters.get(name), Some(AnimParam::Bool(false)))
                }
                TransitionCondition::AnimationComplete => false, // Handled externally
            };

            if triggered {
                if let Some(state) = self.states.get(&transition.to) {
                    let resource = state.resource.clone();
                    self.current_state = transition.to.clone();
                    result = Some((resource, transition.duration));
                    break;
                }
            }
        }


        result
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

use std::f32::consts::PI;

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
        
        sm.transitions.push(StateTransition {
            from: "idle".to_string(),
            to: "locomotion".to_string(),
            condition: TransitionCondition::FloatGreater("Speed".to_string(), 0.1),
            duration: 0.2,
        });
        
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
}
