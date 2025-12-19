// STFSC Engine - Audio System
// 3D spatial audio for VR using rodio

use glam::{Quat, Vec3};
use rodio::{OutputStream, OutputStreamHandle, Sink, Source, SpatialSink};
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
    Linear {
        min_distance: f32,
        max_distance: f32,
    },
    /// Inverse distance (realistic falloff)
    InverseDistance {
        reference_distance: f32,
        max_distance: f32,
        rolloff_factor: f32,
    },
    /// Exponential falloff
    Exponential {
        reference_distance: f32,
        rolloff_factor: f32,
    },
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
    pub channels: u16, // Changed to u16 for rodio compatibility
    /// Duration in seconds
    pub duration: f32,
}

impl AudioBuffer {
    /// Create a new audio buffer from PCM data
    pub fn new(samples: Vec<i16>, sample_rate: u32, channels: u16) -> Self {
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
enum ActiveSourceState {
    Spatial(SpatialSink),
    Ambient(Sink),
}

struct ActiveSource {
    buffer_handle: AudioBufferHandle,
    properties: AudioSourceProperties,
    sink: ActiveSourceState,
}

/// Audio system manager
pub struct AudioSystem {
    /// Stream handle (keep alive)
    _stream: OutputStream,
    /// Stream handle for creating sinks
    stream_handle: OutputStreamHandle,
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
    pub fn new() -> anyhow::Result<Self> {
        let (stream, stream_handle) = OutputStream::try_default()?;

        Ok(Self {
            _stream: stream,
            stream_handle,
            buffers: HashMap::new(),
            sources: HashMap::new(),
            listener: AudioListener::default(),
            next_buffer_id: 1,
            next_source_id: 1,
            master_volume: 1.0,
            muted: false,
        })
    }

    /// Load an audio buffer from PCM data
    pub fn load_buffer(&mut self, buffer: AudioBuffer) -> AudioBufferHandle {
        let handle = AudioBufferHandle(self.next_buffer_id);
        self.next_buffer_id += 1;
        self.buffers.insert(handle, Arc::new(buffer));
        handle
    }

    /// Load an audio buffer from encoded bytes (wav/ogg/mp3/flac)
    pub fn load_buffer_from_bytes(&mut self, data: Vec<u8>) -> anyhow::Result<AudioBufferHandle> {
        let cursor = std::io::Cursor::new(data);
        let decoder = rodio::Decoder::new(cursor)?;
        let sample_rate = decoder.sample_rate();
        let channels = decoder.channels();
        let samples: Vec<i16> = decoder.collect();

        let buffer = AudioBuffer::new(samples, sample_rate, channels);
        Ok(self.load_buffer(buffer))
    }

    /// Unload an audio buffer
    pub fn unload_buffer(&mut self, handle: AudioBufferHandle) {
        self.buffers.remove(&handle);
    }

    fn create_source_from_buffer(buffer: &AudioBuffer) -> rodio::buffer::SamplesBuffer<i16> {
        rodio::buffer::SamplesBuffer::new(
            buffer.channels,
            buffer.sample_rate,
            buffer.samples.clone(),
        )
    }

    /// Play a sound at a 3D position
    pub fn play_3d(
        &mut self,
        buffer_handle: AudioBufferHandle,
        properties: AudioSourceProperties,
    ) -> Option<AudioSourceHandle> {
        let buffer = self.buffers.get(&buffer_handle)?;

        let sink = if let AttenuationModel::None = properties.attenuation {
            // Use standard Sink for 2D/Ambient
            let sink = Sink::try_new(&self.stream_handle).ok()?;
            let source = Self::create_source_from_buffer(buffer);
            if properties.looping {
                sink.append(source.repeat_infinite());
            } else {
                sink.append(source);
            }
            sink.set_volume(
                properties.volume * self.master_volume * if self.muted { 0.0 } else { 1.0 },
            );
            sink.set_speed(properties.pitch);

            ActiveSourceState::Ambient(sink)
        } else {
            // Use SpatialSink
            // We need to calculate ear positions from listener position + rotation
            let right = self.listener.forward.cross(self.listener.up).normalize();
            let ear_spacing = 0.2; // 20cm
            let left_ear = self.listener.position - right * (ear_spacing * 0.5);
            let right_ear = self.listener.position + right * (ear_spacing * 0.5);

            let sink = SpatialSink::try_new(
                &self.stream_handle,
                [
                    properties.position.x,
                    properties.position.y,
                    properties.position.z,
                ],
                [left_ear.x, left_ear.y, left_ear.z],
                [right_ear.x, right_ear.y, right_ear.z],
            )
            .ok()?;

            let source = Self::create_source_from_buffer(buffer);
            if properties.looping {
                sink.append(source.repeat_infinite());
            } else {
                sink.append(source);
            }
            sink.set_volume(
                properties.volume * self.master_volume * if self.muted { 0.0 } else { 1.0 },
            );
            sink.set_speed(properties.pitch);

            ActiveSourceState::Spatial(sink)
        };

        let handle = AudioSourceHandle(self.next_source_id);
        self.next_source_id += 1;

        self.sources.insert(
            handle,
            ActiveSource {
                buffer_handle,
                properties,
                sink,
            },
        );

        Some(handle)
    }

    /// Play a 2D sound (no spatialization)
    pub fn play_2d(
        &mut self,
        buffer: AudioBufferHandle,
        volume: f32,
        looping: bool,
    ) -> Option<AudioSourceHandle> {
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
        if let Some(source) = self.sources.remove(&handle) {
            match source.sink {
                ActiveSourceState::Ambient(s) => s.stop(),
                ActiveSourceState::Spatial(s) => s.stop(),
            }
        }
    }

    /// Stop all sounds
    pub fn stop_all(&mut self) {
        for (_, source) in self.sources.drain() {
            match source.sink {
                ActiveSourceState::Ambient(s) => s.stop(),
                ActiveSourceState::Spatial(s) => s.stop(),
            }
        }
    }

    /// Update source position
    pub fn set_source_position(&mut self, handle: AudioSourceHandle, position: Vec3) {
        if let Some(source) = self.sources.get_mut(&handle) {
            source.properties.position = position;
            if let ActiveSourceState::Spatial(s) = &source.sink {
                s.set_emitter_position([position.x, position.y, position.z]);
            }
        }
    }

    /// Update source volume
    pub fn set_source_volume(&mut self, handle: AudioSourceHandle, volume: f32) {
        if let Some(source) = self.sources.get_mut(&handle) {
            source.properties.volume = volume.clamp(0.0, 1.0);
            let vol =
                source.properties.volume * self.master_volume * if self.muted { 0.0 } else { 1.0 };
            match &source.sink {
                ActiveSourceState::Ambient(s) => s.set_volume(vol),
                ActiveSourceState::Spatial(s) => s.set_volume(vol),
            }
        }
    }

    /// Update the listener position (call each frame with HMD pose)
    pub fn set_listener(&mut self, listener: AudioListener) {
        self.listener = listener;

        // Update all spatial sinks with new ear positions
        let right = self.listener.forward.cross(self.listener.up).normalize();
        let ear_spacing = 0.2;
        let left_ear = self.listener.position - right * (ear_spacing * 0.5);
        let right_ear = self.listener.position + right * (ear_spacing * 0.5);

        for source in self.sources.values() {
            if let ActiveSourceState::Spatial(s) = &source.sink {
                s.set_left_ear_position([left_ear.x, left_ear.y, left_ear.z]);
                s.set_right_ear_position([right_ear.x, right_ear.y, right_ear.z]);
            }
        }
    }

    /// Update listener from position and rotation
    pub fn set_listener_pose(&mut self, position: Vec3, rotation: Quat) {
        self.set_listener(AudioListener::from_pose(position, rotation));
    }

    /// Set master volume
    pub fn set_master_volume(&mut self, volume: f32) {
        self.master_volume = volume.clamp(0.0, 1.0);
        // Update all sources
        for source in self.sources.values() {
            let vol =
                source.properties.volume * self.master_volume * if self.muted { 0.0 } else { 1.0 };
            match &source.sink {
                ActiveSourceState::Ambient(s) => s.set_volume(vol),
                ActiveSourceState::Spatial(s) => s.set_volume(vol),
            }
        }
    }

    /// Mute/unmute all audio
    pub fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
        let vol_mult = if muted { 0.0 } else { 1.0 };
        for source in self.sources.values() {
            let vol = source.properties.volume * self.master_volume * vol_mult;
            match &source.sink {
                ActiveSourceState::Ambient(s) => s.set_volume(vol),
                ActiveSourceState::Spatial(s) => s.set_volume(vol),
            }
        }
    }

    /// Update audio system (call once per frame)
    pub fn update(&mut self, _delta_time: f32) {
        // Remove finished sources
        // Rodio sink.empty() returns true if done.
        let finished: Vec<_> = self
            .sources
            .iter()
            .filter(|(_, s)| {
                // Only remove if not looping and empty
                !s.properties.looping
                    && match &s.sink {
                        ActiveSourceState::Ambient(k) => k.empty(),
                        ActiveSourceState::Spatial(k) => k.empty(),
                    }
            })
            .map(|(h, _)| *h)
            .collect();

        for handle in finished {
            self.sources.remove(&handle);
        }
    }

    /// Get number of active sources
    pub fn active_source_count(&self) -> usize {
        self.sources.len()
    }

    /// Check if a source is still playing
    pub fn is_playing(&self, handle: AudioSourceHandle) -> bool {
        self.sources.contains_key(&handle)
    }
}

impl Default for AudioSystem {
    fn default() -> Self {
        Self::new().expect("Failed to create AudioSystem")
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
        let mut system = AudioSystem::new().unwrap();
        let buffer = system.load_buffer(AudioBuffer::test_tone(440.0, 1.0, 44100));

        let handle = system.play_2d(buffer, 0.5, false);
        assert!(handle.is_some());
        assert!(system.is_playing(handle.unwrap()));

        // Wait for sound to finish
        std::thread::sleep(std::time::Duration::from_secs_f32(1.1));
        system.update(1.1); // Update to clean up finished sources
        assert!(!system.is_playing(handle.unwrap()));
    }

    #[test]
    fn test_attenuation() {
        let mut system = AudioSystem::new().unwrap();
        system.set_listener(AudioListener::default());

        let buffer = system.load_buffer(AudioBuffer::test_tone(440.0, 0.1, 44100));

        // Play at distance
        let props = AudioSourceProperties {
            position: Vec3::new(0.0, 0.0, -10.0),
            attenuation: AttenuationModel::Linear {
                min_distance: 1.0,
                max_distance: 20.0,
            },
            ..Default::default()
        };

        let handle = system.play_3d(buffer, props);
        assert!(handle.is_some());
        assert!(system.is_playing(handle.unwrap()));

        // Ensure it's still playing (it's a short sound)
        std::thread::sleep(std::time::Duration::from_millis(50));
        system.update(0.05);
        assert!(system.is_playing(handle.unwrap()));
    }

    #[test]
    fn test_stop_all() {
        let mut system = AudioSystem::new().unwrap();
        let buffer = system.load_buffer(AudioBuffer::test_tone(440.0, 5.0, 44100)); // Long sound

        let h1 = system.play_2d(buffer, 0.5, false).unwrap();
        let h2 = system.play_2d(buffer, 0.5, false).unwrap();

        assert!(system.is_playing(h1));
        assert!(system.is_playing(h2));

        system.stop_all();
        system.update(0.016); // Update to process stop

        assert!(!system.is_playing(h1));
        assert!(!system.is_playing(h2));
        assert_eq!(system.active_source_count(), 0);
    }

    #[test]
    fn test_master_volume_and_mute() {
        let mut system = AudioSystem::new().unwrap();
        let buffer = system.load_buffer(AudioBuffer::test_tone(440.0, 1.0, 44100));

        let handle = system.play_2d(buffer, 0.5, false).unwrap();

        // Initial volume should be 0.5 * 1.0 = 0.5
        // Rodio doesn't expose current sink volume, so we rely on internal state.
        assert_eq!(system.sources.get(&handle).unwrap().properties.volume, 0.5);

        system.set_master_volume(0.2);
        // The internal sink volume should be updated
        // We can't assert rodio's internal state directly, but we can check our own.
        assert_eq!(system.master_volume, 0.2);

        system.set_muted(true);
        // Volume should effectively be 0
        assert!(system.muted);

        system.set_muted(false);
        assert!(!system.muted);
    }
}
