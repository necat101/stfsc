pub mod tasks;

pub use tasks::{
    FrameBudget, FramePressure, TaskBudget, TaskDomain, TaskGraph, TaskGraphError,
    TaskGraphReport, TaskId, TaskPriority, TaskRunMetrics,
};

use std::io;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

static PARALLELISM: OnceLock<ParallelismConfig> = OnceLock::new();

const NANOS_PER_SECOND: u64 = 1_000_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlatformProfile {
    Desktop,
    MobileVr,
    Mobile,
}

impl PlatformProfile {
    fn detect() -> Self {
        if cfg!(target_os = "android") {
            Self::MobileVr
        } else if cfg!(any(target_os = "ios", target_os = "android")) {
            Self::Mobile
        } else {
            Self::Desktop
        }
    }

    pub fn is_mobile(self) -> bool {
        matches!(self, Self::Mobile | Self::MobileVr)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ParallelismConfig {
    pub platform: PlatformProfile,
    pub logical_cores: usize,
    pub rayon_threads: usize,
    pub background_workers: usize,
    pub blocking_threads: usize,
    pub resource_queue_capacity: usize,
}

impl ParallelismConfig {
    fn detect() -> Self {
        let platform = PlatformProfile::detect();
        let logical_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1);

        let rayon_threads = if logical_cores <= 2 {
            logical_cores
        } else if platform == PlatformProfile::MobileVr {
            logical_cores.saturating_sub(2).clamp(2, 4)
        } else {
            logical_cores.saturating_sub(1).clamp(2, 12)
        };

        let background_workers = if logical_cores <= 2 {
            1
        } else if platform.is_mobile() {
            logical_cores.saturating_sub(3).clamp(1, 3)
        } else {
            logical_cores.saturating_sub(2).clamp(2, 6)
        };

        let blocking_threads = if platform.is_mobile() {
            background_workers.clamp(1, 3)
        } else {
            background_workers.saturating_mul(2).clamp(2, 8)
        };

        let resource_queue_capacity = if platform.is_mobile() {
            background_workers.saturating_mul(24).clamp(24, 96)
        } else {
            background_workers.saturating_mul(64).clamp(64, 512)
        };

        Self {
            platform,
            logical_cores,
            rayon_threads,
            background_workers,
            blocking_threads,
            resource_queue_capacity,
        }
    }
}

pub fn parallelism_config() -> ParallelismConfig {
    *PARALLELISM.get_or_init(ParallelismConfig::detect)
}

pub fn configure_parallelism() -> ParallelismConfig {
    let config = parallelism_config();
    let result = rayon::ThreadPoolBuilder::new()
        .num_threads(config.rayon_threads)
        .thread_name(|idx| format!("stfsc-rayon-{idx}"))
        .build_global();

    match result {
        Ok(()) => log::info!(
            "Runtime: {:?}, {} logical cores, {} Rayon workers, {} background workers, queue capacity {}",
            config.platform,
            config.logical_cores,
            config.rayon_threads,
            config.background_workers,
            config.resource_queue_capacity
        ),
        Err(_) => log::debug!("Runtime: Rayon global thread pool was already configured"),
    }

    config
}

pub fn background_worker_count() -> usize {
    parallelism_config().background_workers
}

pub fn resource_queue_capacity() -> usize {
    parallelism_config().resource_queue_capacity
}

pub fn build_async_runtime(thread_name: &str) -> io::Result<tokio::runtime::Runtime> {
    let config = parallelism_config();
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(config.background_workers.max(1))
        .max_blocking_threads(config.blocking_threads.max(1))
        .thread_name(thread_name)
        .enable_all()
        .build()
}

fn duration_from_hz(hz: u32) -> Duration {
    let hz = hz.max(1) as u64;
    Duration::from_nanos(NANOS_PER_SECOND / hz)
}

#[derive(Clone, Copy, Debug)]
pub struct EngineTimingConfig {
    pub fixed_delta: Duration,
    pub target_frame_time: Duration,
    pub max_frame_delta: Duration,
    pub max_substeps: u32,
}

impl EngineTimingConfig {
    pub fn from_hz(fixed_hz: u32, target_hz: u32) -> Self {
        Self {
            fixed_delta: duration_from_hz(fixed_hz),
            target_frame_time: duration_from_hz(target_hz),
            max_frame_delta: Duration::from_millis(250),
            max_substeps: 5,
        }
        .sanitized()
    }

    pub fn desktop_60hz() -> Self {
        Self::from_hz(60, 60)
    }

    pub fn mobile_vr_36hz() -> Self {
        Self::from_hz(36, 36)
    }

    pub fn default_for_platform() -> Self {
        if parallelism_config().platform == PlatformProfile::MobileVr {
            Self::mobile_vr_36hz()
        } else {
            Self::desktop_60hz()
        }
    }

    pub fn fixed_dt_secs(self) -> f32 {
        self.fixed_delta.as_secs_f32()
    }

    pub fn sanitized(mut self) -> Self {
        if self.fixed_delta.is_zero() {
            self.fixed_delta = duration_from_hz(60);
        }
        if self.target_frame_time.is_zero() {
            self.target_frame_time = self.fixed_delta;
        }
        if self.max_frame_delta < self.fixed_delta {
            self.max_frame_delta = self.fixed_delta;
        }
        self.max_substeps = self.max_substeps.clamp(1, 16);
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FrameTick {
    pub frame_index: u64,
    pub raw_delta: Duration,
    pub clamped_delta: Duration,
    pub fixed_steps: u32,
    pub interpolation_alpha: f32,
}

pub struct FrameClock {
    config: EngineTimingConfig,
    last_instant: Instant,
    accumulator: Duration,
    frame_index: u64,
}

impl FrameClock {
    pub fn new(config: EngineTimingConfig) -> Self {
        Self::new_at(config, Instant::now())
    }

    pub fn new_at(config: EngineTimingConfig, start: Instant) -> Self {
        Self {
            config: config.sanitized(),
            last_instant: start,
            accumulator: Duration::ZERO,
            frame_index: 0,
        }
    }

    pub fn tick(&mut self) -> FrameTick {
        self.tick_at(Instant::now())
    }

    pub fn tick_at(&mut self, now: Instant) -> FrameTick {
        let raw_delta = now.saturating_duration_since(self.last_instant);
        self.last_instant = now;

        let clamped_delta = raw_delta.min(self.config.max_frame_delta);
        self.accumulator += clamped_delta;

        let mut fixed_steps = 0;
        while self.accumulator >= self.config.fixed_delta && fixed_steps < self.config.max_substeps
        {
            self.accumulator -= self.config.fixed_delta;
            fixed_steps += 1;
        }

        if fixed_steps == self.config.max_substeps && self.accumulator >= self.config.fixed_delta {
            self.accumulator = Duration::ZERO;
        }

        self.frame_index += 1;
        let interpolation_alpha =
            self.accumulator.as_secs_f32() / self.config.fixed_delta.as_secs_f32();

        FrameTick {
            frame_index: self.frame_index,
            raw_delta,
            clamped_delta,
            fixed_steps,
            interpolation_alpha: interpolation_alpha.clamp(0.0, 1.0),
        }
    }

    pub fn fixed_dt_secs(&self) -> f32 {
        self.config.fixed_dt_secs()
    }
}

pub fn sleep_remaining(frame_start: Instant, target_frame_time: Duration) {
    if let Some(remaining) = target_frame_time.checked_sub(frame_start.elapsed()) {
        std::thread::sleep(remaining);
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameMetrics {
    pub frame_index: u64,
    pub last_frame_ms: f32,
    pub average_frame_ms: f32,
    pub max_frame_ms: f32,
    pub fixed_steps: u32,
}

pub struct FrameProfiler {
    metrics: FrameMetrics,
    smoothing: f32,
}

impl FrameProfiler {
    pub fn new() -> Self {
        Self {
            metrics: FrameMetrics::default(),
            smoothing: 0.08,
        }
    }

    pub fn record(&mut self, tick: FrameTick, frame_time: Duration) -> FrameMetrics {
        let frame_ms = frame_time.as_secs_f32() * 1000.0;
        let average = if self.metrics.frame_index == 0 {
            frame_ms
        } else {
            self.metrics.average_frame_ms * (1.0 - self.smoothing) + frame_ms * self.smoothing
        };

        self.metrics = FrameMetrics {
            frame_index: tick.frame_index,
            last_frame_ms: frame_ms,
            average_frame_ms: average,
            max_frame_ms: self.metrics.max_frame_ms.max(frame_ms),
            fixed_steps: tick.fixed_steps,
        };
        self.metrics
    }

    pub fn metrics(&self) -> FrameMetrics {
        self.metrics
    }
}

impl Default for FrameProfiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timing_config_sanitizes_invalid_values() {
        let config = EngineTimingConfig {
            fixed_delta: Duration::ZERO,
            target_frame_time: Duration::ZERO,
            max_frame_delta: Duration::ZERO,
            max_substeps: 0,
        }
        .sanitized();

        assert!(config.fixed_delta > Duration::ZERO);
        assert!(config.target_frame_time > Duration::ZERO);
        assert!(config.max_frame_delta >= config.fixed_delta);
        assert_eq!(config.max_substeps, 1);
    }

    #[test]
    fn frame_clock_caps_fixed_steps_and_preserves_alpha() {
        let config = EngineTimingConfig {
            fixed_delta: Duration::from_millis(10),
            target_frame_time: Duration::from_millis(10),
            max_frame_delta: Duration::from_millis(250),
            max_substeps: 3,
        };
        let start = Instant::now();
        let mut clock = FrameClock::new_at(config, start);

        let tick = clock.tick_at(start + Duration::from_millis(35));

        assert_eq!(tick.fixed_steps, 3);
        assert!(tick.interpolation_alpha < 1.0);
    }
}
