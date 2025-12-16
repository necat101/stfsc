// STFSC Engine - Audio System
// 3D spatial audio for VR using native Android audio APIs

use glam::{Vec3, Quat};
use std::collections::HashMap;
use std::sync::Arc;

/// Audio source handle for tracking playing sounds
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AudioSourceHandle(pub u32);

/// Audio buffer handle for loaded sounds
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AudioBufferHandle(pub u32);

/// Spatial audio attenuation model
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AttenuationModel {
    /// No attenuation (2D sound)
    None,
    /// Linear falloff between min and max distance
    Linear { min_distance: f32, max_distance: f32 },
    /// Inverse distance (realistic falloff)
    InverseDistance { reference_distance: f32, max_distance: f32, rolloff_factor: f32 },
    /// Exponential falloff
    Exponential { reference_distance: f32, rolloff_factor: f32 },
}

impl Default for AttenuationModel {
    fn default() -> Self {
        AttenuationModel::InverseDistance {
            reference_distance: 1.0,
            max_distance: 50.0,
            rolloff_factor: 1.0,
        }
    }
}

/// Properties for a 3D audio source
#[derive(Clone, Debug)]
pub struct AudioSourceProperties {
    /// World position of the sound source
    pub position: Vec3,
    /// Velocity for doppler effect (optional)
    pub velocity: Vec3,
    /// Volume (0.0 to 1.0)
    pub volume: f32,
    /// Pitch multiplier (1.0 = normal)
    pub pitch: f32,
    /// Whether the sound loops
    pub looping: bool,
    /// Attenuation model for distance falloff
    pub attenuation: AttenuationModel,
    /// Minimum volume (prevents complete silence at distance)
    pub min_volume: f32,
}

impl Default for AudioSourceProperties {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            velocity: Vec3::ZERO,
            volume: 1.0,
            pitch: 1.0,
            looping: false,
            attenuation: AttenuationModel::default(),
            min_volume: 0.0,
        }
    }
}

/// Audio listener (usually attached to the camera/HMD)
#[derive(Clone, Debug)]
pub struct AudioListener {
    pub position: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
    pub velocity: Vec3,
}

impl Default for AudioListener {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
            velocity: Vec3::ZERO,
        }
    }
}

impl AudioListener {
    /// Update listener from HMD pose
    pub fn from_pose(position: Vec3, rotation: Quat) -> Self {
        Self {
            position,
            forward: rotation * Vec3::NEG_Z,
            up: rotation * Vec3::Y,
            velocity: Vec3::ZERO,
        }
    }
}

/// Audio buffer containing sound data
#[derive(Clone)]
pub struct AudioBuffer {
    /// Raw PCM samples (interleaved if stereo)
    pub samples: Vec<i16>,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u8,
    /// Duration in seconds
    pub duration: f32,
}

impl AudioBuffer {
    /// Create a new audio buffer from PCM data
    pub fn new(samples: Vec<i16>, sample_rate: u32, channels: u8) -> Self {
        let duration = (samples.len() as f32) / (sample_rate as f32 * channels as f32);
        Self {
            samples,
            sample_rate,
            channels,
            duration,
        }
    }

    /// Generate a simple sine wave for testing
    pub fn test_tone(frequency: f32, duration: f32, sample_rate: u32) -> Self {
        let num_samples = (duration * sample_rate as f32) as usize;
        let mut samples = Vec::with_capacity(num_samples);
        
        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let value = (2.0 * std::f32::consts::PI * frequency * t).sin();
            samples.push((value * 32767.0) as i16);
        }
        
        Self::new(samples, sample_rate, 1)
    }
}

/// Active audio source state
struct ActiveSource {
    buffer_handle: AudioBufferHandle,
    properties: AudioSourceProperties,
    /// Current playback position in samples
    playback_pos: usize,
    /// Whether currently playing
    playing: bool,
}

/// Audio system manager
/// Note: Actual audio output requires native Android integration (OpenSL ES or AAudio)
/// This provides the high-level API; platform-specific backend to be integrated
pub struct AudioSystem {
    /// Loaded audio buffers
    buffers: HashMap<AudioBufferHandle, Arc<AudioBuffer>>,
    /// Active sound sources
    sources: HashMap<AudioSourceHandle, ActiveSource>,
    /// Audio listener (HMD position)
    listener: AudioListener,
    /// Next buffer handle
    next_buffer_id: u32,
    /// Next source handle
    next_source_id: u32,
    /// Master volume (0.0 to 1.0)
    master_volume: f32,
    /// Whether audio is muted
    muted: bool,
}

impl AudioSystem {
    /// Create a new audio system
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            sources: HashMap::new(),
            listener: AudioListener::default(),
            next_buffer_id: 1,
            next_source_id: 1,
            master_volume: 1.0,
            muted: false,
        }
    }

    /// Load an audio buffer from PCM data
    pub fn load_buffer(&mut self, buffer: AudioBuffer) -> AudioBufferHandle {
        let handle = AudioBufferHandle(self.next_buffer_id);
        self.next_buffer_id += 1;
        self.buffers.insert(handle, Arc::new(buffer));
        handle
    }

    /// Unload an audio buffer
    pub fn unload_buffer(&mut self, handle: AudioBufferHandle) {
        self.buffers.remove(&handle);
    }

    /// Play a sound at a 3D position
    pub fn play_3d(&mut self, buffer: AudioBufferHandle, properties: AudioSourceProperties) -> Option<AudioSourceHandle> {
        if !self.buffers.contains_key(&buffer) {
            log::warn!("Attempted to play non-existent audio buffer: {:?}", buffer);
            return None;
        }

        let handle = AudioSourceHandle(self.next_source_id);
        self.next_source_id += 1;

        self.sources.insert(handle, ActiveSource {
            buffer_handle: buffer,
            properties,
            playback_pos: 0,
            playing: true,
        });

        Some(handle)
    }

    /// Play a 2D sound (no spatialization)
    pub fn play_2d(&mut self, buffer: AudioBufferHandle, volume: f32, looping: bool) -> Option<AudioSourceHandle> {
        let props = AudioSourceProperties {
            volume,
            looping,
            attenuation: AttenuationModel::None,
            ..Default::default()
        };
        self.play_3d(buffer, props)
    }

    /// Stop a playing sound
    pub fn stop(&mut self, handle: AudioSourceHandle) {
        if let Some(source) = self.sources.get_mut(&handle) {
            source.playing = false;
        }
    }

    /// Stop all sounds
    pub fn stop_all(&mut self) {
        for source in self.sources.values_mut() {
            source.playing = false;
        }
    }

    /// Update source position
    pub fn set_source_position(&mut self, handle: AudioSourceHandle, position: Vec3) {
        if let Some(source) = self.sources.get_mut(&handle) {
            source.properties.position = position;
        }
    }

    /// Update source volume
    pub fn set_source_volume(&mut self, handle: AudioSourceHandle, volume: f32) {
        if let Some(source) = self.sources.get_mut(&handle) {
            source.properties.volume = volume.clamp(0.0, 1.0);
        }
    }

    /// Update the listener position (call each frame with HMD pose)
    pub fn set_listener(&mut self, listener: AudioListener) {
        self.listener = listener;
    }

    /// Update listener from position and rotation
    pub fn set_listener_pose(&mut self, position: Vec3, rotation: Quat) {
        self.listener = AudioListener::from_pose(position, rotation);
    }

    /// Set master volume
    pub fn set_master_volume(&mut self, volume: f32) {
        self.master_volume = volume.clamp(0.0, 1.0);
    }

    /// Mute/unmute all audio
    pub fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
    }

    /// Calculate attenuation for a source based on distance
    fn calculate_attenuation(&self, source: &ActiveSource) -> f32 {
        if matches!(source.properties.attenuation, AttenuationModel::None) {
            return source.properties.volume;
        }

        let distance = source.properties.position.distance(self.listener.position);
        
        let gain = match source.properties.attenuation {
            AttenuationModel::None => 1.0,
            AttenuationModel::Linear { min_distance, max_distance } => {
                if distance <= min_distance {
                    1.0
                } else if distance >= max_distance {
                    0.0
                } else {
                    1.0 - (distance - min_distance) / (max_distance - min_distance)
                }
            }
            AttenuationModel::InverseDistance { reference_distance, max_distance, rolloff_factor } => {
                let clamped = distance.clamp(reference_distance, max_distance);
                reference_distance / (reference_distance + rolloff_factor * (clamped - reference_distance))
            }
            AttenuationModel::Exponential { reference_distance, rolloff_factor } => {
                (distance / reference_distance).powf(-rolloff_factor)
            }
        };

        (gain * source.properties.volume).max(source.properties.min_volume)
    }

    /// Calculate stereo panning based on source position relative to listener
    fn calculate_pan(&self, source: &ActiveSource) -> (f32, f32) {
        if matches!(source.properties.attenuation, AttenuationModel::None) {
            return (1.0, 1.0); // Centered for 2D sounds
        }

        let to_source = (source.properties.position - self.listener.position).normalize_or_zero();
        let right = self.listener.forward.cross(self.listener.up).normalize();
        
        // Dot product with right vector gives left-right balance
        let pan = to_source.dot(right);
        
        // Convert to left/right gains (-1 = left, +1 = right)
        let left_gain = ((1.0 - pan) / 2.0).clamp(0.0, 1.0);
        let right_gain = ((1.0 + pan) / 2.0).clamp(0.0, 1.0);
        
        (left_gain, right_gain)
    }

    /// Update audio system (call once per frame)
    /// This handles source cleanup and would drive the audio callback
    pub fn update(&mut self, delta_time: f32) {
        // Remove finished sources
        let finished: Vec<_> = self.sources.iter()
            .filter(|(_, s)| !s.playing)
            .map(|(h, _)| *h)
            .collect();
        
        for handle in finished {
            self.sources.remove(&handle);
        }

        // In a real implementation, this would:
        // 1. Mix active sources
        // 2. Apply spatialization
        // 3. Push to audio output buffer
        
        // For now, just log active source count periodically
        if !self.sources.is_empty() {
            log::trace!("Audio: {} active sources", self.sources.len());
        }
    }

    /// Get number of active sources
    pub fn active_source_count(&self) -> usize {
        self.sources.len()
    }

    /// Check if a source is still playing
    pub fn is_playing(&self, handle: AudioSourceHandle) -> bool {
        self.sources.get(&handle).map(|s| s.playing).unwrap_or(false)
    }
}

impl Default for AudioSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// ECS Component for audio sources
#[derive(Clone, Debug)]
pub struct AudioSource3D {
    /// Handle to the audio buffer to play
    pub buffer: AudioBufferHandle,
    /// Volume (0.0 to 1.0)
    pub volume: f32,
    /// Whether to loop
    pub looping: bool,
    /// Attenuation model
    pub attenuation: AttenuationModel,
    /// Whether currently playing
    pub playing: bool,
    /// Runtime source handle (set when playing)
    pub source_handle: Option<AudioSourceHandle>,
}

impl Default for AudioSource3D {
    fn default() -> Self {
        Self {
            buffer: AudioBufferHandle(0),
            volume: 1.0,
            looping: false,
            attenuation: AttenuationModel::default(),
            playing: false,
            source_handle: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_buffer_creation() {
        let tone = AudioBuffer::test_tone(440.0, 1.0, 44100);
        assert_eq!(tone.sample_rate, 44100);
        assert_eq!(tone.channels, 1);
        assert!((tone.duration - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_audio_system_play() {
        let mut system = AudioSystem::new();
        let buffer = system.load_buffer(AudioBuffer::test_tone(440.0, 1.0, 44100));
        
        let handle = system.play_2d(buffer, 0.5, false);
        assert!(handle.is_some());
        assert!(system.is_playing(handle.unwrap()));
        
        system.stop(handle.unwrap());
        system.update(0.016);
        assert!(!system.is_playing(handle.unwrap()));
    }

    #[test]
    fn test_attenuation() {
        let mut system = AudioSystem::new();
        system.set_listener(AudioListener::default());
        
        let buffer = system.load_buffer(AudioBuffer::test_tone(440.0, 0.1, 44100));
        
        // Play at distance
        let props = AudioSourceProperties {
            position: Vec3::new(0.0, 0.0, -10.0),
            attenuation: AttenuationModel::Linear { min_distance: 1.0, max_distance: 20.0 },
            ..Default::default()
        };
        
        let handle = system.play_3d(buffer, props);
        assert!(handle.is_some());
    }
}
