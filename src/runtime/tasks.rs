use rayon::prelude::*;
use std::borrow::Cow;
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::{Duration, Instant};

use super::EngineTimingConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TaskId(usize);

impl TaskId {
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum TaskPriority {
    Background = 0,
    Low = 1,
    #[default]
    Normal = 2,
    High = 3,
    Critical = 4,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TaskDomain {
    #[default]
    Simulation,
    Physics,
    Scripting,
    RenderPrep,
    Streaming,
    ResourceUpload,
    Audio,
    Networking,
    Maintenance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaskBudget {
    pub target: Duration,
}

impl TaskBudget {
    pub fn new(target: Duration) -> Self {
        Self { target }
    }

    pub fn from_millis(ms: u64) -> Self {
        Self::new(Duration::from_millis(ms))
    }
}

struct TaskNode<'scope> {
    label: Cow<'scope, str>,
    domain: TaskDomain,
    priority: TaskPriority,
    budget: Option<TaskBudget>,
    dependencies: Vec<TaskId>,
    work: Box<dyn Fn() + Send + Sync + 'scope>,
}

pub struct TaskGraph<'scope> {
    nodes: Vec<TaskNode<'scope>>,
}

impl<'scope> TaskGraph<'scope> {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(capacity),
        }
    }

    pub fn add_task<F>(
        &mut self,
        label: impl Into<Cow<'scope, str>>,
        domain: TaskDomain,
        priority: TaskPriority,
        work: F,
    ) -> TaskId
    where
        F: Fn() + Send + Sync + 'scope,
    {
        self.add_task_with_budget(label, domain, priority, None, work)
    }

    pub fn add_task_with_budget<F>(
        &mut self,
        label: impl Into<Cow<'scope, str>>,
        domain: TaskDomain,
        priority: TaskPriority,
        budget: Option<TaskBudget>,
        work: F,
    ) -> TaskId
    where
        F: Fn() + Send + Sync + 'scope,
    {
        let id = TaskId(self.nodes.len());
        self.nodes.push(TaskNode {
            label: label.into(),
            domain,
            priority,
            budget,
            dependencies: Vec::new(),
            work: Box::new(work),
        });
        id
    }

    pub fn add_dependency(
        &mut self,
        task: TaskId,
        depends_on: TaskId,
    ) -> Result<(), TaskGraphError> {
        if task.index() >= self.nodes.len() {
            return Err(TaskGraphError::MissingTask { task: task.index() });
        }
        if depends_on.index() >= self.nodes.len() {
            return Err(TaskGraphError::MissingDependency {
                task: task.index(),
                dependency: depends_on.index(),
            });
        }
        self.nodes[task.index()].dependencies.push(depends_on);
        Ok(())
    }

    pub fn task_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn execute(&self) -> Result<TaskGraphReport, TaskGraphError> {
        self.validate_dependencies()?;

        let started = Instant::now();
        let task_count = self.nodes.len();
        let mut report = TaskGraphReport::default();
        let mut completed = vec![false; task_count];
        let mut completed_count = 0usize;
        let mut indegree = vec![0usize; task_count];
        let mut dependents = vec![Vec::<usize>::new(); task_count];

        for (task_index, node) in self.nodes.iter().enumerate() {
            indegree[task_index] = node.dependencies.len();
            for dependency in &node.dependencies {
                dependents[dependency.index()].push(task_index);
            }
        }

        while completed_count < task_count {
            let mut ready: Vec<usize> = (0..task_count)
                .filter(|idx| !completed[*idx] && indegree[*idx] == 0)
                .collect();

            if ready.is_empty() {
                let remaining = completed
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, done)| (!done).then_some(idx))
                    .collect();
                return Err(TaskGraphError::CyclicDependency { remaining });
            }

            ready.sort_by(|a, b| {
                self.nodes[*b]
                    .priority
                    .cmp(&self.nodes[*a].priority)
                    .then_with(|| a.cmp(b))
            });

            let wave_index = report.waves;
            let results: Vec<Result<TaskRunMetrics, TaskGraphError>> = ready
                .par_iter()
                .map(|task_index| self.run_task(*task_index, wave_index))
                .collect();

            for result in results {
                report.record(result?);
            }

            for task_index in ready {
                completed[task_index] = true;
                completed_count += 1;
                for dependent in &dependents[task_index] {
                    indegree[*dependent] = indegree[*dependent].saturating_sub(1);
                }
            }

            report.waves += 1;
        }

        report.total_time = started.elapsed();
        Ok(report)
    }

    fn validate_dependencies(&self) -> Result<(), TaskGraphError> {
        for (task_index, node) in self.nodes.iter().enumerate() {
            for dependency in &node.dependencies {
                if dependency.index() >= self.nodes.len() {
                    return Err(TaskGraphError::MissingDependency {
                        task: task_index,
                        dependency: dependency.index(),
                    });
                }
            }
        }
        Ok(())
    }

    fn run_task(
        &self,
        task_index: usize,
        wave_index: usize,
    ) -> Result<TaskRunMetrics, TaskGraphError> {
        let node = &self.nodes[task_index];
        let started = Instant::now();
        let result = catch_unwind(AssertUnwindSafe(|| (node.work)()));
        let duration = started.elapsed();

        match result {
            Ok(()) => Ok(TaskRunMetrics {
                task: TaskId(task_index),
                label: node.label.to_string(),
                domain: node.domain,
                priority: node.priority,
                wave_index,
                duration,
                budget: node.budget,
            }),
            Err(_) => Err(TaskGraphError::TaskPanicked {
                task: task_index,
                label: node.label.to_string(),
            }),
        }
    }
}

impl Default for TaskGraph<'_> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub struct TaskRunMetrics {
    pub task: TaskId,
    pub label: String,
    pub domain: TaskDomain,
    pub priority: TaskPriority,
    pub wave_index: usize,
    pub duration: Duration,
    pub budget: Option<TaskBudget>,
}

impl TaskRunMetrics {
    pub fn over_budget(&self) -> bool {
        self.budget
            .map(|budget| self.duration > budget.target)
            .unwrap_or(false)
    }
}

#[derive(Clone, Debug, Default)]
pub struct TaskGraphReport {
    pub total_time: Duration,
    pub waves: usize,
    pub tasks_executed: usize,
    pub over_budget_count: usize,
    pub longest_task: Option<TaskRunMetrics>,
}

impl TaskGraphReport {
    fn record(&mut self, metrics: TaskRunMetrics) {
        self.tasks_executed += 1;
        if metrics.over_budget() {
            self.over_budget_count += 1;
        }

        let replace_longest = self
            .longest_task
            .as_ref()
            .map(|longest| metrics.duration > longest.duration)
            .unwrap_or(true);

        if replace_longest {
            self.longest_task = Some(metrics);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TaskGraphError {
    MissingTask { task: usize },
    MissingDependency { task: usize, dependency: usize },
    CyclicDependency { remaining: Vec<usize> },
    TaskPanicked { task: usize, label: String },
}

impl fmt::Display for TaskGraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingTask { task } => write!(formatter, "task {task} does not exist"),
            Self::MissingDependency { task, dependency } => write!(
                formatter,
                "task {task} depends on missing task {dependency}"
            ),
            Self::CyclicDependency { remaining } => {
                write!(formatter, "task graph contains a cycle: {remaining:?}")
            }
            Self::TaskPanicked { task, label } => {
                write!(formatter, "task {task} ({label}) panicked")
            }
        }
    }
}

impl std::error::Error for TaskGraphError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FramePressure {
    Idle,
    Nominal,
    Tight,
    OverBudget,
}

#[derive(Clone, Debug)]
pub struct FrameBudget {
    frame_start: Instant,
    target_frame_time: Duration,
    tight_margin: Duration,
}

impl FrameBudget {
    pub fn new(target_frame_time: Duration) -> Self {
        let tight_margin = target_frame_time / 5;
        Self {
            frame_start: Instant::now(),
            target_frame_time: target_frame_time.max(Duration::from_millis(1)),
            tight_margin,
        }
    }

    pub fn for_timing(config: EngineTimingConfig) -> Self {
        Self::new(config.target_frame_time)
    }

    pub fn with_tight_margin(mut self, tight_margin: Duration) -> Self {
        self.tight_margin = tight_margin.min(self.target_frame_time);
        self
    }

    pub fn elapsed(&self) -> Duration {
        self.frame_start.elapsed()
    }

    pub fn remaining(&self) -> Duration {
        self.target_frame_time
            .checked_sub(self.elapsed())
            .unwrap_or(Duration::ZERO)
    }

    pub fn pressure(&self) -> FramePressure {
        let elapsed = self.elapsed();
        if elapsed >= self.target_frame_time {
            FramePressure::OverBudget
        } else if self.remaining() <= self.tight_margin {
            FramePressure::Tight
        } else if elapsed <= self.target_frame_time / 2 {
            FramePressure::Idle
        } else {
            FramePressure::Nominal
        }
    }

    pub fn allows_priority(&self, priority: TaskPriority) -> bool {
        match self.pressure() {
            FramePressure::Idle | FramePressure::Nominal => true,
            FramePressure::Tight => priority >= TaskPriority::Normal,
            FramePressure::OverBudget => priority >= TaskPriority::High,
        }
    }

    pub fn should_defer_background(&self) -> bool {
        !self.allows_priority(TaskPriority::Background)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn task_graph_executes_dependency_waves() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut graph = TaskGraph::new();

        let setup = {
            let order = order.clone();
            graph.add_task(
                "setup",
                TaskDomain::Simulation,
                TaskPriority::Critical,
                move || order.lock().unwrap().push("setup"),
            )
        };

        let physics = {
            let order = order.clone();
            graph.add_task(
                "physics",
                TaskDomain::Physics,
                TaskPriority::High,
                move || order.lock().unwrap().push("physics"),
            )
        };

        let render_prep = {
            let order = order.clone();
            graph.add_task(
                "render_prep",
                TaskDomain::RenderPrep,
                TaskPriority::High,
                move || order.lock().unwrap().push("render_prep"),
            )
        };

        graph.add_dependency(physics, setup).unwrap();
        graph.add_dependency(render_prep, setup).unwrap();

        let report = graph.execute().unwrap();
        let order = order.lock().unwrap();

        assert_eq!(report.waves, 2);
        assert_eq!(report.tasks_executed, 3);
        assert_eq!(order[0], "setup");
        assert!(order.contains(&"physics"));
        assert!(order.contains(&"render_prep"));
    }

    #[test]
    fn task_graph_detects_cycles() {
        let mut graph = TaskGraph::new();
        let a = graph.add_task("a", TaskDomain::Simulation, TaskPriority::Normal, || {});
        let b = graph.add_task("b", TaskDomain::Simulation, TaskPriority::Normal, || {});
        graph.add_dependency(a, b).unwrap();
        graph.add_dependency(b, a).unwrap();

        let error = graph.execute().unwrap_err();
        assert!(matches!(error, TaskGraphError::CyclicDependency { .. }));
    }

    #[test]
    fn task_graph_catches_panics() {
        let mut graph = TaskGraph::new();
        graph.add_task("bad", TaskDomain::Maintenance, TaskPriority::Low, || {
            panic!("boom");
        });

        let error = graph.execute().unwrap_err();
        assert!(matches!(error, TaskGraphError::TaskPanicked { .. }));
    }

    #[test]
    fn frame_budget_defers_background_when_over_budget() {
        let budget = FrameBudget {
            frame_start: Instant::now() - Duration::from_millis(20),
            target_frame_time: Duration::from_millis(16),
            tight_margin: Duration::from_millis(3),
        };

        assert_eq!(budget.pressure(), FramePressure::OverBudget);
        assert!(budget.allows_priority(TaskPriority::Critical));
        assert!(!budget.allows_priority(TaskPriority::Low));
        assert!(budget.should_defer_background());
    }
}
