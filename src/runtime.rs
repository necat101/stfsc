use std::sync::OnceLock;

static PARALLELISM: OnceLock<ParallelismConfig> = OnceLock::new();

#[derive(Clone, Copy, Debug)]
pub struct ParallelismConfig {
    pub logical_cores: usize,
    pub rayon_threads: usize,
    pub background_workers: usize,
}

impl ParallelismConfig {
    fn detect() -> Self {
        let logical_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1);

        let rayon_threads = if logical_cores <= 2 {
            logical_cores
        } else if cfg!(target_os = "android") {
            logical_cores.saturating_sub(2).clamp(2, 4)
        } else {
            logical_cores.saturating_sub(1).clamp(2, 12)
        };

        let background_workers = if logical_cores <= 2 {
            1
        } else if cfg!(target_os = "android") {
            logical_cores.saturating_sub(3).clamp(1, 3)
        } else {
            logical_cores.saturating_sub(2).clamp(2, 6)
        };

        Self {
            logical_cores,
            rayon_threads,
            background_workers,
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
            "Runtime: {} logical cores, {} Rayon workers, {} background loader workers",
            config.logical_cores,
            config.rayon_threads,
            config.background_workers
        ),
        Err(_) => log::debug!("Runtime: Rayon global thread pool was already configured"),
    }

    config
}

pub fn background_worker_count() -> usize {
    parallelism_config().background_workers
}
