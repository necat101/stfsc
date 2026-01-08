//! Animation Runtime System for STFSC Engine
//!
//! Provides GPU-accelerated skeletal animation for imported 3D models:
//! - `Animator` component for animation playback state
//! - `AnimationState` component for computed bone matrices (GPU-ready)
//! - Animation blending and state machine support
//!
//! Optimized for Quest 3's 2.4 TFLOPS to render 100+ animated NPCs in 556 Downtown.

use crate::world::fbx_loader::{AnimationClip, Skeleton};
use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;
use std::sync::Arc;

/// Maximum bones per skeleton (matches shader UBO size)
pub const MAX_BONES: usize = 128;

/// ECS Component: Controls animation playback for an entity
#[derive(Clone, Debug)]
pub struct Animator {
    /// Reference to the skeleton hierarchy
    pub skeleton: Arc<Skeleton>,
    /// Available animation clips for this entity
    pub clips: Vec<Arc<AnimationClip>>,
    /// Index of the currently playing clip
    pub current_clip: usize,
    /// Current playback time in seconds
    pub time: f32,
    /// Playback speed multiplier (1.0 = normal)
    pub speed: f32,
    /// Whether to loop the animation
    pub looping: bool,
    /// Whether the animation is currently playing
    pub playing: bool,
    /// Blend target clip (for crossfades)
    pub blend_target: Option<usize>,
    /// Blend factor (0.0 = current, 1.0 = target)
    pub blend_factor: f32,
    /// Blend duration in seconds
    pub blend_duration: f32,
}

impl Animator {
    /// Create a new animator with a skeleton and animation clips
    pub fn new(skeleton: Arc<Skeleton>, clips: Vec<Arc<AnimationClip>>) -> Self {
        Self {
            skeleton,
            clips,
            current_clip: 0,
            time: 0.0,
            speed: 1.0,
            looping: true,
            playing: true,
            blend_target: None,
            blend_factor: 0.0,
            blend_duration: 0.0,
        }
    }

    /// Start playing a specific animation clip by index
    pub fn play(&mut self, clip_index: usize) {
        if clip_index < self.clips.len() {
            self.current_clip = clip_index;
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

    /// Crossfade to a new animation over a duration
    pub fn crossfade_to(&mut self, clip_index: usize, duration: f32) {
        if clip_index < self.clips.len() && clip_index != self.current_clip {
            self.blend_target = Some(clip_index);
            self.blend_factor = 0.0;
            self.blend_duration = duration.max(0.001); // Avoid division by zero
        }
    }

    /// Crossfade to an animation by name
    pub fn crossfade_by_name(&mut self, name: &str, duration: f32) {
        for (i, clip) in self.clips.iter().enumerate() {
            if clip.name == name {
                self.crossfade_to(i, duration);
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
        if !self.playing || self.clips.is_empty() {
            return self.bind_pose_matrices();
        }

        // Update playback time
        self.time += dt * self.speed;

        // Handle looping
        let current_clip = &self.clips[self.current_clip];
        if self.time >= current_clip.duration {
            if self.looping {
                self.time %= current_clip.duration;
            } else {
                self.time = current_clip.duration;
                self.playing = false;
            }
        }

        // Sample current animation
        let current_pose = self.sample_clip(self.current_clip, self.time);

        // Handle blending
        if let Some(target_idx) = self.blend_target {
            self.blend_factor += dt / self.blend_duration;
            
            if self.blend_factor >= 1.0 {
                // Blend complete, switch to target
                self.current_clip = target_idx;
                self.time = self.blend_factor * self.clips[target_idx].duration; // Approximate sync
                self.blend_target = None;
                self.blend_factor = 0.0;
                return self.sample_clip(target_idx, self.time);
            }

            let target_pose = self.sample_clip(target_idx, self.time);
            return blend_poses(&current_pose, &target_pose, self.blend_factor);
        }

        current_pose
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

/// Animation state machine for complex character animation
#[derive(Clone, Debug)]
pub struct AnimationStateMachine {
    /// Named states mapping to clip indices
    pub states: HashMap<String, usize>,
    /// Current state name
    pub current_state: String,
    /// Transition definitions
    pub transitions: Vec<StateTransition>,
    /// Parameters that drive transitions
    pub parameters: HashMap<String, AnimParam>,
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

/// Animation parameter types
#[derive(Clone, Debug)]
pub enum AnimParam {
    Float(f32),
    Bool(bool),
    Trigger(bool),
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

    /// Add a state
    pub fn add_state(&mut self, name: &str, clip_index: usize) {
        self.states.insert(name.to_string(), clip_index);
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

    /// Get current clip index
    pub fn current_clip(&self) -> Option<usize> {
        self.states.get(&self.current_state).copied()
    }

    /// Evaluate transitions and return target state if any
    pub fn evaluate(&mut self) -> Option<(usize, f32)> {
        let mut result = None;

        for transition in &self.transitions {
            if transition.from != self.current_state {
                continue;
            }

            let triggered = match &transition.condition {
                TransitionCondition::Trigger(name) => {
                    matches!(self.parameters.get(name), Some(AnimParam::Trigger(true)))
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
                if let Some(&clip_idx) = self.states.get(&transition.to) {
                    self.current_state = transition.to.clone();
                    result = Some((clip_idx, transition.duration));
                    break;
                }
            }
        }

        // Clear triggers after evaluation
        for (_, param) in self.parameters.iter_mut() {
            if let AnimParam::Trigger(_) = param {
                *param = AnimParam::Trigger(false);
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
            let (s, r, t) = mat.to_scale_rotation_translation();
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
}
