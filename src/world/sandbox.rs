use glam::Vec3;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum GameMode {
    #[default]
    Survival,
    GodMode,
}

impl GameMode {
    pub fn all() -> Vec<Self> {
        vec![Self::Survival, Self::GodMode]
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Survival => "Survival",
            Self::GodMode => "God Mode",
        }
    }

    pub fn allows_free_building(self) -> bool {
        matches!(self, Self::GodMode)
    }

    pub fn consumes_resources(self) -> bool {
        matches!(self, Self::Survival)
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SandboxProfile {
    #[default]
    ForestSurvival,
    UrbanStreaming,
    Hybrid,
}

impl SandboxProfile {
    pub fn all() -> Vec<Self> {
        vec![Self::ForestSurvival, Self::UrbanStreaming, Self::Hybrid]
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::ForestSurvival => "Forest Survival",
            Self::UrbanStreaming => "Urban Streaming",
            Self::Hybrid => "Hybrid",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SandboxWorldSettings {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub seed: u64,
    #[serde(default)]
    pub profile: SandboxProfile,
    #[serde(default)]
    pub game_mode: GameMode,
    #[serde(default = "default_chunk_size")]
    pub chunk_size: f32,
    #[serde(default = "default_load_radius")]
    pub load_radius: i32,
    #[serde(default = "default_simulation_radius")]
    pub simulation_radius: i32,
    #[serde(default = "default_vertical_scale")]
    pub vertical_scale: f32,
    #[serde(default = "default_day_length_seconds")]
    pub day_length_seconds: f32,
    #[serde(default = "default_night_start")]
    pub night_start: f32,
    #[serde(default = "default_night_end")]
    pub night_end: f32,
    #[serde(default = "default_tree_density")]
    pub tree_density: f32,
    #[serde(default = "default_resource_density")]
    pub resource_density: f32,
    #[serde(default = "default_passive_mob_density")]
    pub passive_mob_density: f32,
    #[serde(default = "default_hostile_mob_density")]
    pub hostile_mob_density: f32,
    #[serde(default = "default_max_natural_objects_per_chunk")]
    pub max_natural_objects_per_chunk: u32,
    #[serde(default)]
    pub spawn_clear_radius: f32,
    #[serde(default = "default_max_buildables_per_chunk")]
    pub max_buildables_per_chunk: u32,
    #[serde(default = "default_max_active_mobs")]
    pub max_active_mobs: u32,
}

impl Default for SandboxWorldSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            seed: 0,
            profile: SandboxProfile::ForestSurvival,
            game_mode: GameMode::Survival,
            chunk_size: default_chunk_size(),
            load_radius: default_load_radius(),
            simulation_radius: default_simulation_radius(),
            vertical_scale: default_vertical_scale(),
            day_length_seconds: default_day_length_seconds(),
            night_start: default_night_start(),
            night_end: default_night_end(),
            tree_density: default_tree_density(),
            resource_density: default_resource_density(),
            passive_mob_density: default_passive_mob_density(),
            hostile_mob_density: default_hostile_mob_density(),
            max_natural_objects_per_chunk: default_max_natural_objects_per_chunk(),
            spawn_clear_radius: 0.0,
            max_buildables_per_chunk: default_max_buildables_per_chunk(),
            max_active_mobs: default_max_active_mobs(),
        }
    }
}

impl SandboxWorldSettings {
    pub fn forest_survival(seed: u64) -> Self {
        Self {
            enabled: true,
            seed,
            profile: SandboxProfile::ForestSurvival,
            ..Default::default()
        }
    }

    pub fn urban_streaming(seed: u64) -> Self {
        Self {
            enabled: true,
            seed,
            profile: SandboxProfile::UrbanStreaming,
            tree_density: 0.04,
            resource_density: 0.12,
            passive_mob_density: 0.02,
            hostile_mob_density: 0.03,
            max_natural_objects_per_chunk: 12,
            ..Default::default()
        }
    }

    pub fn clamp_runtime_limits(&mut self) {
        self.chunk_size = self.chunk_size.clamp(8.0, 256.0);
        self.load_radius = self.load_radius.clamp(1, 24);
        self.simulation_radius = self.simulation_radius.clamp(1, self.load_radius);
        self.vertical_scale = self.vertical_scale.clamp(0.0, 512.0);
        self.day_length_seconds = self.day_length_seconds.max(1.0);
        self.night_start = wrap01(self.night_start);
        self.night_end = wrap01(self.night_end);
        self.tree_density = self.tree_density.clamp(0.0, 1.0);
        self.resource_density = self.resource_density.clamp(0.0, 1.0);
        self.passive_mob_density = self.passive_mob_density.clamp(0.0, 1.0);
        self.hostile_mob_density = self.hostile_mob_density.clamp(0.0, 1.0);
        self.max_natural_objects_per_chunk = self.max_natural_objects_per_chunk.clamp(1, 512);
        self.spawn_clear_radius = self.spawn_clear_radius.max(0.0);
        self.max_buildables_per_chunk = self.max_buildables_per_chunk.clamp(1, 4096);
        self.max_active_mobs = self.max_active_mobs.clamp(1, 8192);
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct SandboxClock {
    pub time_of_day: f32,
    pub day_index: u32,
    pub day_length_seconds: f32,
    pub night_start: f32,
    pub night_end: f32,
}

impl Default for SandboxClock {
    fn default() -> Self {
        Self::from_settings(&SandboxWorldSettings::default())
    }
}

impl SandboxClock {
    pub fn from_settings(settings: &SandboxWorldSettings) -> Self {
        Self {
            time_of_day: 0.25,
            day_index: 0,
            day_length_seconds: settings.day_length_seconds.max(1.0),
            night_start: wrap01(settings.night_start),
            night_end: wrap01(settings.night_end),
        }
    }

    pub fn advance(&mut self, delta_seconds: f32) {
        if delta_seconds <= 0.0 {
            return;
        }

        let day_delta = delta_seconds / self.day_length_seconds.max(1.0);
        self.time_of_day += day_delta;
        while self.time_of_day >= 1.0 {
            self.time_of_day -= 1.0;
            self.day_index = self.day_index.wrapping_add(1);
        }
    }

    pub fn set_time(&mut self, time_of_day: f32, day_index: u32) {
        self.time_of_day = wrap01(time_of_day);
        self.day_index = day_index;
    }

    pub fn is_night(&self) -> bool {
        if self.night_start <= self.night_end {
            self.time_of_day >= self.night_start && self.time_of_day < self.night_end
        } else {
            self.time_of_day >= self.night_start || self.time_of_day < self.night_end
        }
    }

    pub fn sun_factor(&self) -> f32 {
        let noon_distance = (self.time_of_day - 0.5).abs() * 2.0;
        (1.0 - noon_distance).clamp(0.0, 1.0)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SandboxChunkCoord {
    pub x: i32,
    pub z: i32,
}

impl SandboxChunkCoord {
    pub fn from_world_pos(pos: Vec3, chunk_size: f32) -> Self {
        let size = chunk_size.max(1.0);
        Self {
            x: (pos.x / size).floor() as i32,
            z: (pos.z / size).floor() as i32,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SandboxChunkPlan {
    pub coord: SandboxChunkCoord,
    pub seed: u64,
    pub profile: SandboxProfile,
    pub terrain_height: f32,
    pub object_spawns: Vec<SandboxObjectSpawn>,
    pub mob_spawns: Vec<SandboxMobSpawn>,
    pub build_budget: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SandboxObjectSpawn {
    pub stable_id: u64,
    pub kind: SandboxObjectKind,
    pub position: Vec3,
    pub rotation_y: f32,
    pub scale: Vec3,
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SandboxObjectKind {
    Tree,
    Rock,
    ResourceNode(ResourceKind),
    BuildablePlaceholder,
    Landmark,
    Custom(String),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ResourceKind {
    Wood,
    Stone,
    Fiber,
    Food,
    Custom(String),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SandboxMobSpawn {
    pub stable_id: u64,
    pub archetype: String,
    pub position: Vec3,
    pub hostile: bool,
    pub night_only: bool,
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BuildableObject {
    pub archetype: String,
    pub owner_id: Option<u64>,
    pub stability: f32,
    pub can_pick_up: bool,
    pub snap_points: Vec<SnapPoint>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SnapPoint {
    pub local_position: Vec3,
    pub local_normal: Vec3,
    pub socket: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ResourceNode {
    pub kind: ResourceKind,
    pub amount: f32,
    pub respawn_seconds: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MobSpawnRule {
    pub archetype: String,
    pub hostile: bool,
    pub night_only: bool,
    pub density: f32,
    pub max_per_chunk: u32,
    pub tags: Vec<String>,
}

pub struct SandboxChunkGenerator {
    seed: u64,
}

impl SandboxChunkGenerator {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    pub fn generate_chunk(
        &self,
        coord: SandboxChunkCoord,
        settings: &SandboxWorldSettings,
    ) -> SandboxChunkPlan {
        let mut settings = settings.clone();
        settings.clamp_runtime_limits();

        let terrain_height = self.sample_signed(coord, 9) * settings.vertical_scale;
        let mut object_spawns = Vec::new();
        let max_objects = settings.max_natural_objects_per_chunk.min(512);

        for index in 0..max_objects {
            let roll = self.sample01(coord, 100 + index as u64);
            let kind = if roll < settings.tree_density {
                Some(self.tree_like_kind(coord, index, settings.profile))
            } else if roll < settings.tree_density + settings.resource_density {
                Some(SandboxObjectKind::ResourceNode(self.resource_kind(
                    coord,
                    index,
                    settings.profile,
                )))
            } else {
                None
            };

            if let Some(kind) = kind {
                let spawn = self.object_spawn(coord, index, kind, terrain_height, &settings);
                if !inside_spawn_clear_radius(spawn.position, settings.spawn_clear_radius) {
                    object_spawns.push(spawn);
                }
            }
        }

        let mut mob_spawns = Vec::new();
        for index in 0..8 {
            let passive_roll = self.sample01(coord, 700 + index);
            if passive_roll < settings.passive_mob_density {
                let spawn = self.mob_spawn(
                    coord,
                    index,
                    "passive_wildlife",
                    false,
                    false,
                    terrain_height,
                    &settings,
                );
                if !inside_spawn_clear_radius(spawn.position, settings.spawn_clear_radius) {
                    mob_spawns.push(spawn);
                }
            }

            let hostile_roll = self.sample01(coord, 900 + index);
            if hostile_roll < settings.hostile_mob_density {
                let spawn = self.mob_spawn(
                    coord,
                    100 + index,
                    "hostile_night_enemy",
                    true,
                    true,
                    terrain_height,
                    &settings,
                );
                if !inside_spawn_clear_radius(spawn.position, settings.spawn_clear_radius) {
                    mob_spawns.push(spawn);
                }
            }
        }

        SandboxChunkPlan {
            coord,
            seed: settings.seed,
            profile: settings.profile,
            terrain_height,
            object_spawns,
            mob_spawns,
            build_budget: settings.max_buildables_per_chunk,
        }
    }

    pub fn generate_window(
        &self,
        center: SandboxChunkCoord,
        radius: i32,
        settings: &SandboxWorldSettings,
    ) -> Vec<SandboxChunkPlan> {
        let radius = radius.max(0);
        let coords: Vec<_> = ((center.x - radius)..=(center.x + radius))
            .flat_map(|x| {
                ((center.z - radius)..=(center.z + radius)).map(move |z| SandboxChunkCoord { x, z })
            })
            .collect();

        coords
            .into_par_iter()
            .map(|coord| self.generate_chunk(coord, settings))
            .collect()
    }

    fn object_spawn(
        &self,
        coord: SandboxChunkCoord,
        index: u32,
        kind: SandboxObjectKind,
        terrain_height: f32,
        settings: &SandboxWorldSettings,
    ) -> SandboxObjectSpawn {
        let x = self.world_axis(coord.x, coord, index, 300, settings.chunk_size);
        let z = self.world_axis(coord.z, coord, index, 400, settings.chunk_size);
        let rotation_y = self.sample01(coord, 500 + index as u64) * std::f32::consts::TAU;
        let scale_base = 0.75 + self.sample01(coord, 600 + index as u64) * 0.75;
        let scale = match kind {
            SandboxObjectKind::Tree => Vec3::new(scale_base, scale_base * 1.8, scale_base),
            SandboxObjectKind::Rock => Vec3::splat(scale_base),
            SandboxObjectKind::ResourceNode(_) => Vec3::splat(scale_base * 0.8),
            SandboxObjectKind::BuildablePlaceholder => Vec3::ONE,
            SandboxObjectKind::Landmark => {
                Vec3::new(scale_base * 2.0, scale_base * 3.0, scale_base * 2.0)
            }
            SandboxObjectKind::Custom(_) => Vec3::ONE,
        };

        let tags = match kind {
            SandboxObjectKind::Tree => vec!["natural".to_string(), "wood".to_string()],
            SandboxObjectKind::Rock => vec!["natural".to_string(), "stone".to_string()],
            SandboxObjectKind::ResourceNode(_) => vec!["resource".to_string()],
            SandboxObjectKind::BuildablePlaceholder => vec!["buildable".to_string()],
            SandboxObjectKind::Landmark => vec!["landmark".to_string()],
            SandboxObjectKind::Custom(_) => Vec::new(),
        };

        SandboxObjectSpawn {
            stable_id: self.hash(coord, 10_000 + index as u64),
            kind,
            position: Vec3::new(x, terrain_height, z),
            rotation_y,
            scale,
            tags,
        }
    }

    fn mob_spawn(
        &self,
        coord: SandboxChunkCoord,
        index: u64,
        archetype: &str,
        hostile: bool,
        night_only: bool,
        terrain_height: f32,
        settings: &SandboxWorldSettings,
    ) -> SandboxMobSpawn {
        let x = self.world_axis(coord.x, coord, index as u32, 1100, settings.chunk_size);
        let z = self.world_axis(coord.z, coord, index as u32, 1200, settings.chunk_size);
        let mut tags = vec!["mob".to_string()];
        if hostile {
            tags.push("hostile".to_string());
        } else {
            tags.push("passive".to_string());
        }
        if night_only {
            tags.push("night".to_string());
        }

        SandboxMobSpawn {
            stable_id: self.hash(coord, 20_000 + index),
            archetype: archetype.to_string(),
            position: Vec3::new(x, terrain_height, z),
            hostile,
            night_only,
            tags,
        }
    }

    fn tree_like_kind(
        &self,
        coord: SandboxChunkCoord,
        index: u32,
        profile: SandboxProfile,
    ) -> SandboxObjectKind {
        match profile {
            SandboxProfile::ForestSurvival => {
                if self.sample01(coord, 1300 + index as u64) < 0.86 {
                    SandboxObjectKind::Tree
                } else {
                    SandboxObjectKind::Rock
                }
            }
            SandboxProfile::UrbanStreaming => SandboxObjectKind::Landmark,
            SandboxProfile::Hybrid => {
                if self.sample01(coord, 1400 + index as u64) < 0.5 {
                    SandboxObjectKind::Tree
                } else {
                    SandboxObjectKind::Landmark
                }
            }
        }
    }

    fn resource_kind(
        &self,
        coord: SandboxChunkCoord,
        index: u32,
        profile: SandboxProfile,
    ) -> ResourceKind {
        let roll = self.sample01(coord, 1500 + index as u64);
        match profile {
            SandboxProfile::ForestSurvival => {
                if roll < 0.4 {
                    ResourceKind::Wood
                } else if roll < 0.7 {
                    ResourceKind::Stone
                } else if roll < 0.9 {
                    ResourceKind::Fiber
                } else {
                    ResourceKind::Food
                }
            }
            SandboxProfile::UrbanStreaming => {
                if roll < 0.7 {
                    ResourceKind::Stone
                } else {
                    ResourceKind::Custom("salvage".to_string())
                }
            }
            SandboxProfile::Hybrid => {
                if roll < 0.5 {
                    ResourceKind::Wood
                } else {
                    ResourceKind::Custom("salvage".to_string())
                }
            }
        }
    }

    fn world_axis(
        &self,
        chunk_axis: i32,
        coord: SandboxChunkCoord,
        index: u32,
        salt: u64,
        chunk_size: f32,
    ) -> f32 {
        let local = self.sample01(coord, salt + index as u64) * chunk_size - chunk_size * 0.5;
        chunk_axis as f32 * chunk_size + local
    }

    fn sample_signed(&self, coord: SandboxChunkCoord, salt: u64) -> f32 {
        self.sample01(coord, salt) * 2.0 - 1.0
    }

    fn sample01(&self, coord: SandboxChunkCoord, salt: u64) -> f32 {
        let value = self.hash(coord, salt);
        let high = (value >> 40) as u32;
        high as f32 / 16_777_215.0
    }

    fn hash(&self, coord: SandboxChunkCoord, salt: u64) -> u64 {
        let x = coord.x as i64 as u64;
        let z = coord.z as i64 as u64;
        mix_u64(
            self.seed
                ^ x.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ z.wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
                ^ salt.wrapping_mul(0x1656_67B1_9E37_79F9),
        )
    }
}

fn default_chunk_size() -> f32 {
    32.0
}

fn default_load_radius() -> i32 {
    6
}

fn default_simulation_radius() -> i32 {
    3
}

fn default_vertical_scale() -> f32 {
    12.0
}

fn default_day_length_seconds() -> f32 {
    1200.0
}

fn default_night_start() -> f32 {
    0.75
}

fn default_night_end() -> f32 {
    0.23
}

fn default_tree_density() -> f32 {
    0.35
}

fn default_resource_density() -> f32 {
    0.18
}

fn default_passive_mob_density() -> f32 {
    0.05
}

fn default_hostile_mob_density() -> f32 {
    0.08
}

fn default_max_natural_objects_per_chunk() -> u32 {
    32
}

fn default_max_buildables_per_chunk() -> u32 {
    256
}

fn default_max_active_mobs() -> u32 {
    256
}

fn wrap01(value: f32) -> f32 {
    value.rem_euclid(1.0)
}

fn inside_spawn_clear_radius(position: Vec3, radius: f32) -> bool {
    radius > 0.0 && Vec3::new(position.x, 0.0, position.z).length_squared() < radius * radius
}

fn mix_u64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_generation_is_deterministic() {
        let settings = SandboxWorldSettings::forest_survival(42);
        let generator = SandboxChunkGenerator::new(settings.seed);
        let coord = SandboxChunkCoord { x: -3, z: 7 };

        let first = generator.generate_chunk(coord, &settings);
        let second = generator.generate_chunk(coord, &settings);

        assert_eq!(first, second);
    }

    #[test]
    fn chunk_window_uses_parallel_planning() {
        let settings = SandboxWorldSettings::forest_survival(7);
        let generator = SandboxChunkGenerator::new(settings.seed);
        let plans = generator.generate_window(SandboxChunkCoord { x: 0, z: 0 }, 2, &settings);

        assert_eq!(plans.len(), 25);
        assert!(plans
            .iter()
            .any(|plan| plan.coord == SandboxChunkCoord { x: 2, z: -2 }));
    }

    #[test]
    fn spawn_clear_radius_removes_near_origin_spawns() {
        let mut settings = SandboxWorldSettings::forest_survival(8842);
        settings.tree_density = 1.0;
        settings.resource_density = 0.0;
        settings.passive_mob_density = 1.0;
        settings.hostile_mob_density = 1.0;
        settings.spawn_clear_radius = 128.0;

        let generator = SandboxChunkGenerator::new(settings.seed);
        let center = generator.generate_chunk(SandboxChunkCoord { x: 0, z: 0 }, &settings);
        let far = generator.generate_chunk(SandboxChunkCoord { x: 8, z: 8 }, &settings);

        assert!(center.object_spawns.is_empty());
        assert!(center.mob_spawns.is_empty());
        assert!(!far.object_spawns.is_empty());
    }

    #[test]
    fn clock_wraps_and_detects_cross_midnight_night() {
        let settings = SandboxWorldSettings {
            day_length_seconds: 10.0,
            night_start: 0.75,
            night_end: 0.25,
            ..Default::default()
        };
        let mut clock = SandboxClock::from_settings(&settings);

        clock.set_time(0.8, 0);
        assert!(clock.is_night());

        clock.advance(4.0);
        assert_eq!(clock.day_index, 1);
        assert!(clock.is_night());

        clock.set_time(0.5, 1);
        assert!(!clock.is_night());
    }

    #[test]
    fn game_mode_flags_match_building_rules() {
        assert!(!GameMode::Survival.allows_free_building());
        assert!(GameMode::Survival.consumes_resources());
        assert!(GameMode::GodMode.allows_free_building());
        assert!(!GameMode::GodMode.consumes_resources());
    }
}
