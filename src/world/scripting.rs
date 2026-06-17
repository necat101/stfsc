use crate::physics::PhysicsWorld;
use crate::project::scene::MAX_SCRIPTS_PER_ENTITY;
use crate::world::sandbox::{GameMode, SandboxProfile, SandboxWorldSettings};
use crate::world::{GroundPlane, Projectile, SceneUpdate, Transform};
use hecs::{Entity, World};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

pub const BUILTIN_SCRIPT_NAMES: &[&str] = &[
    "TestBounce",
    "CrowdAgent",
    "Vehicle",
    "PoliceAgent",
    "TrafficAI",
    "EnemyTracker",
    "WeaponNPC",
    "GunWeapon",
    "BowWeapon",
    "Projectile",
    "CollisionLogger",
    "TouchToDestroy",
    "HeadAnchor",
    "LeftHandAnchor",
    "RightHandAnchor",
    "LeftAimAnchor",
    "RightAimAnchor",
    "TriggerHaptics",
];

pub const CUSTOM_FUCKSCRIPT_RUNTIME_NAME: &str = "CustomFuckScript";

static NEXT_SCRIPT_PROJECTILE_ID: AtomicU32 = AtomicU32::new(2_000_000_000);
static NEXT_SCRIPT_SCENE_OBJECT_ID: AtomicU32 = AtomicU32::new(1_500_000_000);

fn next_script_projectile_id() -> u32 {
    let id = NEXT_SCRIPT_PROJECTILE_ID.fetch_add(1, Ordering::Relaxed);
    if id == u32::MAX {
        NEXT_SCRIPT_PROJECTILE_ID.store(2_000_000_000, Ordering::Relaxed);
        2_000_000_000
    } else {
        id
    }
}

fn next_script_scene_object_id() -> u32 {
    let id = NEXT_SCRIPT_SCENE_OBJECT_ID.fetch_add(1, Ordering::Relaxed);
    if id >= 1_999_999_999 {
        NEXT_SCRIPT_SCENE_OBJECT_ID.store(1_500_000_000, Ordering::Relaxed);
        1_500_000_000
    } else {
        id
    }
}

fn script_unit_random(state: &mut u32) -> f32 {
    *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
    (*state as f32) / (u32::MAX as f32)
}

fn script_count(value: f32, max: u32) -> u32 {
    if !value.is_finite() {
        return 0;
    }
    value.round().clamp(0.0, max as f32) as u32
}

fn script_seed(value: f32) -> u64 {
    if !value.is_finite() {
        return 0;
    }
    value.max(0.0).round() as u64
}

fn script_radius(value: f32, fallback: f32) -> f32 {
    if !value.is_finite() {
        return fallback;
    }
    value.max(0.001)
}

fn script_scale(scale: glam::Vec3) -> glam::Vec3 {
    scale.max(glam::Vec3::splat(0.001))
}

fn script_color(color: glam::Vec3) -> [f32; 3] {
    color.clamp(glam::Vec3::ZERO, glam::Vec3::ONE).to_array()
}

fn script_light_cones(inner_cone: f32, outer_cone: f32) -> (f32, f32) {
    let inner = if inner_cone.is_finite() {
        inner_cone.clamp(0.0, std::f32::consts::PI)
    } else {
        0.4
    };
    let outer = if outer_cone.is_finite() {
        outer_cone.clamp(inner, std::f32::consts::PI)
    } else {
        0.6_f32.max(inner)
    };
    (inner, outer)
}

fn script_sandbox_load_radius(value: f32) -> i32 {
    if !value.is_finite() {
        return 6;
    }
    value.round().clamp(1.0, 24.0) as i32
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScriptPrimitive {
    Cube,
    Sphere,
    Cylinder,
    Plane,
    Capsule,
    Cone,
}

impl ScriptPrimitive {
    fn primitive_id(self) -> u8 {
        match self {
            Self::Cube => 0,
            Self::Sphere => 1,
            Self::Cylinder => 2,
            Self::Plane => 3,
            Self::Capsule => 4,
            Self::Cone => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScriptLightType {
    Point,
    Spot,
    Directional,
}

impl From<ScriptLightType> for crate::world::LightType {
    fn from(value: ScriptLightType) -> Self {
        match value {
            ScriptLightType::Point => Self::Point,
            ScriptLightType::Spot => Self::Spot,
            ScriptLightType::Directional => Self::Directional,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScriptGameMode {
    Survival,
    GodMode,
}

impl From<ScriptGameMode> for GameMode {
    fn from(value: ScriptGameMode) -> Self {
        match value {
            ScriptGameMode::Survival => Self::Survival,
            ScriptGameMode::GodMode => Self::GodMode,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScriptSandboxProfile {
    ForestSurvival,
    UrbanStreaming,
    Hybrid,
}

impl From<ScriptSandboxProfile> for SandboxProfile {
    fn from(value: ScriptSandboxProfile) -> Self {
        match value {
            ScriptSandboxProfile::ForestSurvival => Self::ForestSurvival,
            ScriptSandboxProfile::UrbanStreaming => Self::UrbanStreaming,
            ScriptSandboxProfile::Hybrid => Self::Hybrid,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScriptResourceKind {
    Wood,
    Stone,
    Ore,
    Crystal,
    Fiber,
    Food,
}

impl ScriptResourceKind {
    fn primitive(self) -> ScriptPrimitive {
        match self {
            Self::Wood => ScriptPrimitive::Cylinder,
            Self::Stone => ScriptPrimitive::Sphere,
            Self::Ore => ScriptPrimitive::Cube,
            Self::Crystal => ScriptPrimitive::Cone,
            Self::Fiber => ScriptPrimitive::Capsule,
            Self::Food => ScriptPrimitive::Sphere,
        }
    }

    fn min_scale(self) -> glam::Vec3 {
        match self {
            Self::Wood => glam::Vec3::new(0.35, 1.0, 0.35),
            Self::Stone => glam::Vec3::splat(0.6),
            Self::Ore => glam::Vec3::splat(0.55),
            Self::Crystal => glam::Vec3::new(0.45, 1.1, 0.45),
            Self::Fiber => glam::Vec3::new(0.35, 0.8, 0.35),
            Self::Food => glam::Vec3::splat(0.35),
        }
    }

    fn max_scale(self) -> glam::Vec3 {
        match self {
            Self::Wood => glam::Vec3::new(0.75, 2.2, 0.75),
            Self::Stone => glam::Vec3::splat(1.5),
            Self::Ore => glam::Vec3::splat(1.2),
            Self::Crystal => glam::Vec3::new(0.9, 2.4, 0.9),
            Self::Fiber => glam::Vec3::new(0.75, 1.6, 0.75),
            Self::Food => glam::Vec3::splat(0.85),
        }
    }

    fn min_color(self) -> glam::Vec3 {
        match self {
            Self::Wood => glam::Vec3::new(0.33, 0.2, 0.11),
            Self::Stone => glam::Vec3::new(0.34, 0.34, 0.36),
            Self::Ore => glam::Vec3::new(0.22, 0.22, 0.28),
            Self::Crystal => glam::Vec3::new(0.18, 0.55, 0.75),
            Self::Fiber => glam::Vec3::new(0.18, 0.45, 0.18),
            Self::Food => glam::Vec3::new(0.55, 0.16, 0.18),
        }
    }

    fn max_color(self) -> glam::Vec3 {
        match self {
            Self::Wood => glam::Vec3::new(0.55, 0.34, 0.18),
            Self::Stone => glam::Vec3::new(0.58, 0.58, 0.62),
            Self::Ore => glam::Vec3::new(0.42, 0.42, 0.52),
            Self::Crystal => glam::Vec3::new(0.48, 0.9, 1.0),
            Self::Fiber => glam::Vec3::new(0.38, 0.75, 0.32),
            Self::Food => glam::Vec3::new(0.9, 0.42, 0.24),
        }
    }

    fn salt(self) -> u32 {
        match self {
            Self::Wood => 11,
            Self::Stone => 23,
            Self::Ore => 37,
            Self::Crystal => 41,
            Self::Fiber => 53,
            Self::Food => 67,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScriptFurrySpecies {
    Fox,
    Wolf,
    Cat,
    Rabbit,
    Bear,
}

impl ScriptFurrySpecies {
    fn accent_color(self, base: glam::Vec3) -> glam::Vec3 {
        match self {
            Self::Fox => glam::Vec3::new(0.95, 0.42, 0.16),
            Self::Wolf => glam::Vec3::new(0.42, 0.46, 0.52),
            Self::Cat => glam::Vec3::new(0.72, 0.52, 0.36),
            Self::Rabbit => glam::Vec3::new(0.82, 0.76, 0.68),
            Self::Bear => glam::Vec3::new(0.36, 0.22, 0.12),
        }
        .lerp(base.clamp(glam::Vec3::ZERO, glam::Vec3::ONE), 0.35)
    }

    fn ear_scale(self) -> glam::Vec3 {
        match self {
            Self::Rabbit => glam::Vec3::new(0.16, 0.7, 0.16),
            Self::Bear => glam::Vec3::new(0.22, 0.22, 0.22),
            _ => glam::Vec3::new(0.2, 0.42, 0.2),
        }
    }

    fn tail_scale(self) -> glam::Vec3 {
        match self {
            Self::Fox => glam::Vec3::new(0.22, 0.78, 0.22),
            Self::Wolf => glam::Vec3::new(0.18, 0.62, 0.18),
            Self::Cat => glam::Vec3::new(0.12, 0.7, 0.12),
            Self::Rabbit => glam::Vec3::splat(0.22),
            Self::Bear => glam::Vec3::splat(0.18),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum XrHand {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum XrPoseSpace {
    Head,
    Grip(XrHand),
    Aim(XrHand),
}

#[derive(Clone, Copy, Debug)]
pub struct XrPose {
    pub position: glam::Vec3,
    pub rotation: glam::Quat,
    pub linear_velocity: glam::Vec3,
    pub angular_velocity: glam::Vec3,
    pub tracked: bool,
}

impl XrPose {
    pub fn new(position: glam::Vec3, rotation: glam::Quat, tracked: bool) -> Self {
        Self {
            position,
            rotation,
            linear_velocity: glam::Vec3::ZERO,
            angular_velocity: glam::Vec3::ZERO,
            tracked,
        }
    }

    pub fn forward(&self) -> glam::Vec3 {
        self.rotation * glam::Vec3::NEG_Z
    }

    pub fn up(&self) -> glam::Vec3 {
        self.rotation * glam::Vec3::Y
    }

    pub fn right(&self) -> glam::Vec3 {
        self.rotation * glam::Vec3::X
    }
}

impl Default for XrPose {
    fn default() -> Self {
        Self {
            position: glam::Vec3::ZERO,
            rotation: glam::Quat::IDENTITY,
            linear_velocity: glam::Vec3::ZERO,
            angular_velocity: glam::Vec3::ZERO,
            tracked: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct XrControllerState {
    pub grip_pose: XrPose,
    pub aim_pose: XrPose,
    pub trigger: f32,
    pub grip: f32,
    pub thumbstick: glam::Vec2,
    pub primary_button: bool,
    pub secondary_button: bool,
    pub menu_button: bool,
    pub thumbstick_clicked: bool,
    pub tracked: bool,
    pub active: bool,
}

impl Default for XrControllerState {
    fn default() -> Self {
        Self {
            grip_pose: XrPose::default(),
            aim_pose: XrPose::default(),
            trigger: 0.0,
            grip: 0.0,
            thumbstick: glam::Vec2::ZERO,
            primary_button: false,
            secondary_button: false,
            menu_button: false,
            thumbstick_clicked: false,
            tracked: false,
            active: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct XrActionState {
    pub value: f32,
    pub pressed: bool,
    pub just_pressed: bool,
    pub just_released: bool,
}

impl XrActionState {
    pub fn from_value(value: f32, threshold: f32) -> Self {
        Self {
            value,
            pressed: value >= threshold,
            just_pressed: false,
            just_released: false,
        }
    }

    pub fn from_pressed(pressed: bool) -> Self {
        Self {
            value: if pressed { 1.0 } else { 0.0 },
            pressed,
            just_pressed: false,
            just_released: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum XrInputControl {
    Trigger,
    Grip,
    PrimaryButton,
    SecondaryButton,
    MenuButton,
    ThumbstickClick,
    Thumbstick,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XrInputPhase {
    Pressed,
    Released,
    Changed,
}

#[derive(Clone, Debug)]
pub struct XrInputEvent {
    pub hand: Option<XrHand>,
    pub control: Option<XrInputControl>,
    pub action: Option<String>,
    pub phase: XrInputPhase,
    pub value: f32,
    pub axis: glam::Vec2,
}

#[derive(Clone, Debug)]
pub struct XrInputSnapshot {
    pub head: XrPose,
    pub left: XrControllerState,
    pub right: XrControllerState,
    pub actions: HashMap<String, XrActionState>,
    pub frame_index: u64,
    pub predicted_display_time_seconds: f64,
}

impl Default for XrInputSnapshot {
    fn default() -> Self {
        Self {
            head: XrPose::default(),
            left: XrControllerState::default(),
            right: XrControllerState::default(),
            actions: HashMap::new(),
            frame_index: 0,
            predicted_display_time_seconds: 0.0,
        }
    }
}

impl XrInputSnapshot {
    pub fn controller(&self, hand: XrHand) -> &XrControllerState {
        match hand {
            XrHand::Left => &self.left,
            XrHand::Right => &self.right,
        }
    }

    pub fn controller_mut(&mut self, hand: XrHand) -> &mut XrControllerState {
        match hand {
            XrHand::Left => &mut self.left,
            XrHand::Right => &mut self.right,
        }
    }

    pub fn pose(&self, space: XrPoseSpace) -> Option<XrPose> {
        match space {
            XrPoseSpace::Head => self.head.tracked.then_some(self.head),
            XrPoseSpace::Grip(hand) => {
                let pose = self.controller(hand).grip_pose;
                pose.tracked.then_some(pose)
            }
            XrPoseSpace::Aim(hand) => {
                let pose = self.controller(hand).aim_pose;
                pose.tracked.then_some(pose)
            }
        }
    }

    pub fn set_action_value(&mut self, name: impl Into<String>, value: f32, threshold: f32) {
        self.actions
            .insert(name.into(), XrActionState::from_value(value, threshold));
    }

    pub fn set_action_pressed(&mut self, name: impl Into<String>, pressed: bool) {
        self.actions
            .insert(name.into(), XrActionState::from_pressed(pressed));
    }

    pub fn action_state(&self, name: &str) -> XrActionState {
        self.actions.get(name).copied().unwrap_or_default()
    }

    pub fn populate_builtin_actions_from_controls(&mut self) {
        self.actions
            .entry("fire".to_string())
            .or_insert_with(|| XrActionState::from_value(self.right.trigger, 0.9));
        self.actions
            .entry("interact".to_string())
            .or_insert_with(|| XrActionState::from_value(self.right.trigger, 0.7));
        self.actions
            .entry("aim".to_string())
            .or_insert_with(|| XrActionState::from_value(self.left.trigger, 0.5));
        self.actions
            .entry("grab".to_string())
            .or_insert_with(|| XrActionState::from_value(self.right.grip, 0.7));
        self.actions
            .entry("grab_left".to_string())
            .or_insert_with(|| XrActionState::from_value(self.left.grip, 0.7));
        self.actions.entry("pause".to_string()).or_insert_with(|| {
            XrActionState::from_pressed(self.left.menu_button || self.right.menu_button)
        });
    }

    pub fn update_edges_from(&mut self, previous: &XrInputSnapshot) {
        for (name, state) in self.actions.iter_mut() {
            let was_pressed = previous
                .actions
                .get(name)
                .map(|old| old.pressed)
                .unwrap_or(false);
            state.just_pressed = state.pressed && !was_pressed;
            state.just_released = !state.pressed && was_pressed;
        }
    }

    pub fn diff_events(&self, previous: &XrInputSnapshot) -> Vec<XrInputEvent> {
        let mut events = Vec::new();

        for (name, state) in &self.actions {
            if state.just_pressed || state.just_released {
                events.push(XrInputEvent {
                    hand: None,
                    control: None,
                    action: Some(name.clone()),
                    phase: if state.just_pressed {
                        XrInputPhase::Pressed
                    } else {
                        XrInputPhase::Released
                    },
                    value: state.value,
                    axis: glam::Vec2::ZERO,
                });
            }
        }

        Self::push_controller_events(&mut events, XrHand::Left, &previous.left, &self.left);
        Self::push_controller_events(&mut events, XrHand::Right, &previous.right, &self.right);
        events
    }

    fn push_controller_events(
        events: &mut Vec<XrInputEvent>,
        hand: XrHand,
        previous: &XrControllerState,
        current: &XrControllerState,
    ) {
        Self::push_analog_event(
            events,
            hand,
            XrInputControl::Trigger,
            previous.trigger,
            current.trigger,
            0.5,
        );
        Self::push_analog_event(
            events,
            hand,
            XrInputControl::Grip,
            previous.grip,
            current.grip,
            0.5,
        );
        Self::push_button_event(
            events,
            hand,
            XrInputControl::PrimaryButton,
            previous.primary_button,
            current.primary_button,
        );
        Self::push_button_event(
            events,
            hand,
            XrInputControl::SecondaryButton,
            previous.secondary_button,
            current.secondary_button,
        );
        Self::push_button_event(
            events,
            hand,
            XrInputControl::MenuButton,
            previous.menu_button,
            current.menu_button,
        );
        Self::push_button_event(
            events,
            hand,
            XrInputControl::ThumbstickClick,
            previous.thumbstick_clicked,
            current.thumbstick_clicked,
        );

        if (previous.thumbstick - current.thumbstick).length_squared() > 0.0001 {
            events.push(XrInputEvent {
                hand: Some(hand),
                control: Some(XrInputControl::Thumbstick),
                action: None,
                phase: XrInputPhase::Changed,
                value: current.thumbstick.length(),
                axis: current.thumbstick,
            });
        }
    }

    fn push_button_event(
        events: &mut Vec<XrInputEvent>,
        hand: XrHand,
        control: XrInputControl,
        previous: bool,
        current: bool,
    ) {
        if previous == current {
            return;
        }

        events.push(XrInputEvent {
            hand: Some(hand),
            control: Some(control),
            action: None,
            phase: if current {
                XrInputPhase::Pressed
            } else {
                XrInputPhase::Released
            },
            value: if current { 1.0 } else { 0.0 },
            axis: glam::Vec2::ZERO,
        });
    }

    fn push_analog_event(
        events: &mut Vec<XrInputEvent>,
        hand: XrHand,
        control: XrInputControl,
        previous: f32,
        current: f32,
        threshold: f32,
    ) {
        let was_pressed = previous >= threshold;
        let is_pressed = current >= threshold;

        if was_pressed != is_pressed {
            events.push(XrInputEvent {
                hand: Some(hand),
                control: Some(control),
                action: None,
                phase: if is_pressed {
                    XrInputPhase::Pressed
                } else {
                    XrInputPhase::Released
                },
                value: current,
                axis: glam::Vec2::ZERO,
            });
        } else if (previous - current).abs() > 0.01 {
            events.push(XrInputEvent {
                hand: Some(hand),
                control: Some(control),
                action: None,
                phase: XrInputPhase::Changed,
                value: current,
                axis: glam::Vec2::ZERO,
            });
        }
    }
}

#[derive(Clone, Debug)]
pub struct XrHapticRequest {
    pub hand: XrHand,
    pub amplitude: f32,
    pub duration_seconds: f32,
    pub frequency_hz: f32,
}

impl XrHapticRequest {
    pub fn new(hand: XrHand, amplitude: f32, duration_seconds: f32) -> Self {
        Self {
            hand,
            amplitude: amplitude.clamp(0.0, 1.0),
            duration_seconds: duration_seconds.max(0.0),
            frequency_hz: 0.0,
        }
    }
}

/// The core trait for scripts in the STFSC engine.
/// Similar to Unity's MonoBehavior.
pub trait FuckScript: Send + Sync {
    /// Called when the script instance is first created by the runtime.
    fn on_awake(&mut self, _ctx: &mut ScriptContext) {}

    /// Called once when the script is attached or when the scene starts.
    fn on_start(&mut self, _ctx: &mut ScriptContext) {}

    /// Called every logic frame.
    fn on_update(&mut self, _ctx: &mut ScriptContext) {}

    /// Called on the fixed game-logic tick, before `on_update`.
    fn on_fixed_update(&mut self, _ctx: &mut ScriptContext) {}

    /// Called after `on_update`, useful for camera/hand-following scripts.
    fn on_late_update(&mut self, _ctx: &mut ScriptContext) {}

    /// Called when the script/entity is enabled.
    fn on_enable(&mut self, _ctx: &mut ScriptContext) {}

    /// Called when the script/entity is disabled or removed.
    fn on_disable(&mut self, _ctx: &mut ScriptContext) {}

    /// Called when the script instance is being destroyed.
    fn on_destroy(&mut self, _ctx: &mut ScriptContext) {}

    /// Called when a collision starts.
    fn on_collision_start(&mut self, _ctx: &mut ScriptContext, _other: Entity) {}

    /// Called while a collision persists, when the physics backend reports stay events.
    fn on_collision_stay(&mut self, _ctx: &mut ScriptContext, _other: Entity) {}

    /// Called when a collision ends.
    fn on_collision_end(&mut self, _ctx: &mut ScriptContext, _other: Entity) {}

    /// Called when a trigger overlap starts.
    fn on_trigger_start(&mut self, _ctx: &mut ScriptContext, _other: Entity) {}

    /// Called while a trigger overlap persists, when the physics backend reports stay events.
    fn on_trigger_stay(&mut self, _ctx: &mut ScriptContext, _other: Entity) {}

    /// Called when a trigger overlap ends.
    fn on_trigger_end(&mut self, _ctx: &mut ScriptContext, _other: Entity) {}

    /// Called when the player respawns (for global scripts or player-attached scripts)
    fn on_player_respawn(&mut self, _ctx: &mut ScriptContext) {}

    /// Called when a UI event occurs (e.g. button click)
    fn on_ui_event(&mut self, _ctx: &mut ScriptContext, _event: &crate::ui::UiEvent) {}

    /// Called when an animation event is triggered
    fn on_animation_event(&mut self, _ctx: &mut ScriptContext, _event_name: &str) {}

    /// Called once per logic tick with the latest XR rig snapshot.
    fn on_xr_frame(&mut self, _ctx: &mut ScriptContext, _xr: &XrInputSnapshot) {}

    /// Called for XR action/controller edge events.
    fn on_xr_input(&mut self, _ctx: &mut ScriptContext, _event: &XrInputEvent) {}

    /// Called by platforms that expose pause/resume lifecycle.
    fn on_application_pause(&mut self, _ctx: &mut ScriptContext, _paused: bool) {}

    /// Called by platforms that expose app focus lifecycle.
    fn on_application_focus(&mut self, _ctx: &mut ScriptContext, _focused: bool) {}
}

/// Context passed to scripts to allow them to interact with the engine.
pub struct ScriptContext<'a> {
    pub entity: Entity,
    pub world: &'a mut World,
    pub physics: &'a mut PhysicsWorld,
    pub dt: f32,
    pub xr: XrInputSnapshot,
    haptic_requests: Option<&'a mut Vec<XrHapticRequest>>,
    scene_commands: Option<&'a mut Vec<SceneUpdate>>,
}

impl<'a> ScriptContext<'a> {
    pub fn new(
        entity: Entity,
        world: &'a mut World,
        physics: &'a mut PhysicsWorld,
        dt: f32,
    ) -> Self {
        Self {
            entity,
            world,
            physics,
            dt,
            xr: XrInputSnapshot::default(),
            haptic_requests: None,
            scene_commands: None,
        }
    }

    pub fn new_with_xr(
        entity: Entity,
        world: &'a mut World,
        physics: &'a mut PhysicsWorld,
        dt: f32,
        xr: XrInputSnapshot,
        haptic_requests: Option<&'a mut Vec<XrHapticRequest>>,
    ) -> Self {
        Self {
            entity,
            world,
            physics,
            dt,
            xr,
            haptic_requests,
            scene_commands: None,
        }
    }

    pub fn new_with_xr_and_scene_commands(
        entity: Entity,
        world: &'a mut World,
        physics: &'a mut PhysicsWorld,
        dt: f32,
        xr: XrInputSnapshot,
        haptic_requests: Option<&'a mut Vec<XrHapticRequest>>,
        scene_commands: Option<&'a mut Vec<SceneUpdate>>,
    ) -> Self {
        Self {
            entity,
            world,
            physics,
            dt,
            xr,
            haptic_requests,
            scene_commands,
        }
    }

    /// Helper to get a component on the current entity.
    pub fn get_component<T: hecs::Component>(&self) -> Option<hecs::Ref<'_, T>> {
        self.world.get::<&T>(self.entity).ok()
    }

    /// Helper to get a mutable component on the current entity.
    pub fn get_component_mut<T: hecs::Component>(&mut self) -> Option<hecs::RefMut<'_, T>> {
        self.world.get::<&mut T>(self.entity).ok()
    }

    /// Helper to get the transform of the current entity.
    pub fn transform_mut(&mut self) -> Option<hecs::RefMut<'_, Transform>> {
        self.get_component_mut::<Transform>()
    }

    /// Helper to get the transform of the current entity (read-only).
    pub fn transform(&self) -> Option<hecs::Ref<'_, Transform>> {
        self.get_component::<Transform>()
    }

    /// Log a script-scoped message.
    pub fn log(&self, message: &str) {
        log::info!("[script {:?}] {}", self.entity, message);
    }

    /// Helper to get a component on another entity.
    pub fn get_component_on<T: hecs::Component>(&self, entity: Entity) -> Option<hecs::Ref<'_, T>> {
        self.world.get::<&T>(entity).ok()
    }

    /// Helper to get a mutable component on another entity.
    pub fn get_component_mut_on<T: hecs::Component>(
        &mut self,
        entity: Entity,
    ) -> Option<hecs::RefMut<'_, T>> {
        self.world.get::<&mut T>(entity).ok()
    }

    /// Helper to get another entity's transform.
    pub fn transform_of(&self, entity: Entity) -> Option<hecs::Ref<'_, Transform>> {
        self.get_component_on::<Transform>(entity)
    }

    /// Helper to get another entity's mutable transform.
    pub fn transform_of_mut(&mut self, entity: Entity) -> Option<hecs::RefMut<'_, Transform>> {
        self.get_component_mut_on::<Transform>(entity)
    }

    /// Find an entity by the stable editor id used in scene files.
    pub fn find_entity_by_editor_id(&self, id: u32) -> Option<Entity> {
        for (entity, editor_id) in self.world.query::<&crate::world::EditorEntityId>().iter() {
            if editor_id.0 == id {
                return Some(entity);
            }
        }
        None
    }

    /// Find the first runtime player transform in the world.
    pub fn player_position(&self) -> Option<glam::Vec3> {
        for (_entity, (transform, _player)) in self
            .world
            .query::<(&Transform, &crate::world::Player)>()
            .iter()
        {
            return Some(transform.position);
        }
        None
    }

    /// True when the world has a player marker available to scripts.
    pub fn has_player(&self) -> bool {
        self.player_position().is_some()
    }

    /// Horizontal gameplay distance from this entity to the player.
    pub fn distance_to_player(&self) -> f32 {
        let Some(player_pos) = self.player_position() else {
            return f32::INFINITY;
        };
        let Some(transform) = self.transform() else {
            return f32::INFINITY;
        };

        glam::Vec2::new(
            player_pos.x - transform.position.x,
            player_pos.z - transform.position.z,
        )
        .length()
    }

    /// Rotate toward and move toward the player while inside `radius`.
    pub fn track_player(
        &mut self,
        radius: f32,
        speed: f32,
        stop_distance: f32,
        turn_speed: f32,
    ) -> bool {
        let Some(player_pos) = self.player_position() else {
            return false;
        };
        let (position, rotation) = {
            let Some(transform) = self.transform() else {
                return false;
            };
            (transform.position, transform.rotation)
        };

        let to_player = glam::Vec3::new(player_pos.x - position.x, 0.0, player_pos.z - position.z);
        let distance = to_player.length();
        let radius = radius.max(0.0);
        if distance > radius || distance <= 0.000001 {
            return false;
        }

        let direction = to_player / distance;
        let target_rotation = glam::Quat::from_rotation_y(direction.x.atan2(direction.z));
        let turn_amount = if turn_speed <= 0.0 {
            1.0
        } else {
            (turn_speed * self.dt).clamp(0.0, 1.0)
        };
        let next_rotation = rotation.slerp(target_rotation, turn_amount);

        let stop_distance = stop_distance.max(0.0);
        let speed = speed.max(0.0);
        let move_step = if distance > stop_distance {
            (speed * self.dt).min(distance - stop_distance)
        } else {
            0.0
        };
        let next_position = position + direction * move_step;

        let rb_handle = self
            .world
            .get::<&crate::world::RigidBodyHandle>(self.entity)
            .ok()
            .map(|handle| handle.0);

        if let Some(mut transform) = self.transform_mut() {
            transform.position = next_position;
            transform.rotation = next_rotation;
        } else {
            return false;
        }

        if let Some(handle) = rb_handle {
            self.physics.set_rigid_body_transform(
                handle,
                Some(next_position.to_array()),
                Some(next_rotation.to_array()),
            );
        }

        true
    }

    /// Move the current entity by a world-space delta.
    pub fn translate(&mut self, delta: glam::Vec3) {
        if let Some(mut transform) = self.transform_mut() {
            transform.position += delta;
        }
    }

    /// Rotate the current entity around world up.
    pub fn rotate_y(&mut self, radians: f32) {
        if let Some(mut transform) = self.transform_mut() {
            transform.rotation = transform.rotation * glam::Quat::from_rotation_y(radians);
        }
    }

    /// Helper to despawn the current entity.
    pub fn despawn_self(&mut self) {
        let rb_handle = {
            self.world
                .get::<&crate::world::RigidBodyHandle>(self.entity)
                .ok()
                .map(|handle| handle.0)
        };
        if let Some(handle) = rb_handle {
            self.physics.remove_rigid_body(handle);
        }
        let _ = self.world.despawn(self.entity);
    }

    pub fn xr_pose(&self, space: XrPoseSpace) -> Option<XrPose> {
        self.xr.pose(space)
    }

    pub fn head_pose(&self) -> Option<XrPose> {
        self.xr_pose(XrPoseSpace::Head)
    }

    pub fn controller(&self, hand: XrHand) -> &XrControllerState {
        self.xr.controller(hand)
    }

    pub fn action(&self, name: &str) -> XrActionState {
        self.xr.action_state(name)
    }

    pub fn action_pressed(&self, name: &str) -> bool {
        self.action(name).pressed
    }

    pub fn action_just_pressed(&self, name: &str) -> bool {
        self.action(name).just_pressed
    }

    pub fn action_just_released(&self, name: &str) -> bool {
        self.action(name).just_released
    }

    pub fn action_value(&self, name: &str) -> f32 {
        self.action(name).value
    }

    pub fn request_haptic(&mut self, hand: XrHand, amplitude: f32, duration_seconds: f32) {
        if let Some(queue) = self.haptic_requests.as_deref_mut() {
            queue.push(XrHapticRequest::new(hand, amplitude, duration_seconds));
        }
    }

    pub fn set_procedural_generation(&mut self, enabled: bool) {
        self.queue_scene_update(SceneUpdate::SetProceduralGeneration { enabled });
    }

    pub fn set_game_mode(&mut self, mode: ScriptGameMode) {
        self.queue_scene_update(SceneUpdate::SetGameMode { mode: mode.into() });
    }

    pub fn set_sandbox_clock(&mut self, time_of_day: f32, day_index: f32) {
        self.queue_scene_update(SceneUpdate::SetSandboxClock {
            time_of_day,
            day_index: script_count(day_index, u32::MAX),
        });
    }

    pub fn configure_sandbox(
        &mut self,
        profile: ScriptSandboxProfile,
        game_mode: ScriptGameMode,
        seed: f32,
        chunk_size: f32,
        load_radius: f32,
        tree_density: f32,
        resource_density: f32,
        passive_mob_density: f32,
        hostile_mob_density: f32,
        max_natural_objects_per_chunk: f32,
        max_active_mobs: f32,
    ) {
        let mut settings = SandboxWorldSettings {
            enabled: true,
            seed: script_seed(seed),
            profile: profile.into(),
            game_mode: game_mode.into(),
            chunk_size,
            load_radius: script_sandbox_load_radius(load_radius),
            tree_density,
            resource_density,
            passive_mob_density,
            hostile_mob_density,
            max_natural_objects_per_chunk: script_count(max_natural_objects_per_chunk, 512),
            max_active_mobs: script_count(max_active_mobs, 8192),
            ..SandboxWorldSettings::default()
        };
        settings.clamp_runtime_limits();
        self.queue_scene_update(SceneUpdate::SetSandboxSettings { settings });
        self.queue_scene_update(SceneUpdate::SetProceduralGeneration { enabled: true });
    }

    pub fn configure_twilight_survival(
        &mut self,
        seed: f32,
        chunk_size: f32,
        load_radius: f32,
        tree_density: f32,
        resource_density: f32,
        passive_mob_density: f32,
        hostile_mob_density: f32,
    ) {
        let mut settings = SandboxWorldSettings {
            enabled: true,
            seed: script_seed(seed),
            profile: SandboxProfile::ForestSurvival,
            game_mode: GameMode::Survival,
            chunk_size,
            load_radius: script_sandbox_load_radius(load_radius),
            vertical_scale: 28.0,
            day_length_seconds: 900.0,
            night_start: 0.54,
            night_end: 0.24,
            tree_density,
            resource_density,
            passive_mob_density,
            hostile_mob_density,
            max_natural_objects_per_chunk: 96,
            max_buildables_per_chunk: 2048,
            max_active_mobs: 512,
            ..SandboxWorldSettings::default()
        };
        settings.clamp_runtime_limits();
        self.queue_scene_update(SceneUpdate::SetSandboxSettings { settings });
        self.queue_scene_update(SceneUpdate::SetProceduralGeneration { enabled: true });
        self.queue_scene_update(SceneUpdate::SetSandboxClock {
            time_of_day: 0.68,
            day_index: 0,
        });
    }

    fn spawn_primitive_on_layer(
        &mut self,
        primitive: ScriptPrimitive,
        position: glam::Vec3,
        rotation: glam::Quat,
        scale: glam::Vec3,
        color: glam::Vec3,
        collision_enabled: bool,
        is_static: bool,
        layer: u32,
    ) -> u32 {
        let id = next_script_scene_object_id();
        self.queue_scene_update(SceneUpdate::Spawn {
            id,
            primitive: primitive.primitive_id(),
            position: position.to_array(),
            rotation: rotation.to_array(),
            scale: script_scale(scale).to_array(),
            color: script_color(color),
            albedo_texture: None,
            collision_enabled,
            layer,
            is_static,
        });
        id
    }

    pub fn spawn_primitive(
        &mut self,
        primitive: ScriptPrimitive,
        position: glam::Vec3,
        rotation: glam::Quat,
        scale: glam::Vec3,
        color: glam::Vec3,
        collision_enabled: bool,
        is_static: bool,
    ) -> u32 {
        self.spawn_primitive_on_layer(
            primitive,
            position,
            rotation,
            scale,
            color,
            collision_enabled,
            is_static,
            crate::world::LAYER_PROP,
        )
    }

    pub fn spawn_ground(
        &mut self,
        position: glam::Vec3,
        scale: glam::Vec3,
        color: glam::Vec3,
        collision_enabled: bool,
    ) -> u32 {
        let id = next_script_scene_object_id();
        let scale = script_scale(scale);
        self.queue_scene_update(SceneUpdate::SpawnGroundPlane {
            id,
            primitive: ScriptPrimitive::Cube.primitive_id(),
            position: position.to_array(),
            scale: scale.to_array(),
            color: script_color(color),
            half_extents: [scale.x * 0.5, scale.z * 0.5],
            albedo_texture: None,
            collision_enabled,
            layer: crate::world::LAYER_ENVIRONMENT,
        });
        id
    }

    pub fn spawn_light(
        &mut self,
        light_type: ScriptLightType,
        position: glam::Vec3,
        direction: glam::Vec3,
        color: glam::Vec3,
        intensity: f32,
        range: f32,
    ) -> u32 {
        self.spawn_light_with_cones(
            light_type, position, direction, color, intensity, range, 0.4, 0.6,
        )
    }

    pub fn spawn_light_with_cones(
        &mut self,
        light_type: ScriptLightType,
        position: glam::Vec3,
        direction: glam::Vec3,
        color: glam::Vec3,
        intensity: f32,
        range: f32,
        inner_cone: f32,
        outer_cone: f32,
    ) -> u32 {
        let id = next_script_scene_object_id();
        let direction = if direction.length_squared() > 0.000001 {
            direction.normalize()
        } else {
            glam::Vec3::NEG_Y
        };
        let (inner_cone, outer_cone) = script_light_cones(inner_cone, outer_cone);
        self.queue_scene_update(SceneUpdate::SpawnLight {
            id,
            light_type: light_type.into(),
            position: position.to_array(),
            direction: direction.to_array(),
            color: script_color(color),
            intensity: intensity.max(0.0),
            range: range.max(0.0),
            inner_cone,
            outer_cone,
        });
        id
    }

    pub fn spawn_scatter(
        &mut self,
        primitive: ScriptPrimitive,
        count: f32,
        center: glam::Vec3,
        area: glam::Vec3,
        min_scale: glam::Vec3,
        max_scale: glam::Vec3,
        color: glam::Vec3,
        seed: f32,
        collision_enabled: bool,
        is_static: bool,
    ) -> u32 {
        self.spawn_scatter_range(
            primitive,
            count,
            center,
            area,
            min_scale,
            max_scale,
            color,
            color,
            seed,
            collision_enabled,
            is_static,
        )
    }

    pub fn spawn_scatter_range(
        &mut self,
        primitive: ScriptPrimitive,
        count: f32,
        center: glam::Vec3,
        area: glam::Vec3,
        min_scale: glam::Vec3,
        max_scale: glam::Vec3,
        min_color: glam::Vec3,
        max_color: glam::Vec3,
        seed: f32,
        collision_enabled: bool,
        is_static: bool,
    ) -> u32 {
        let count = script_count(count, 512);
        if count == 0 {
            return 0;
        }

        let area = area.abs();
        let min_scale = script_scale(min_scale);
        let max_scale = max_scale.max(min_scale);
        let min_color = min_color.clamp(glam::Vec3::ZERO, glam::Vec3::ONE);
        let max_color = max_color.clamp(glam::Vec3::ZERO, glam::Vec3::ONE);
        let mut state = seed.to_bits().wrapping_mul(1664525).wrapping_add(count);
        let mut spawned = 0;

        for _ in 0..count {
            let rx = script_unit_random(&mut state);
            let rz = script_unit_random(&mut state);
            let rs = script_unit_random(&mut state);
            let rc = script_unit_random(&mut state);
            let yaw = script_unit_random(&mut state) * std::f32::consts::TAU;
            let position = center + glam::Vec3::new((rx - 0.5) * area.x, 0.0, (rz - 0.5) * area.z);
            let scale = min_scale.lerp(max_scale, rs);
            let color = min_color.lerp(max_color, rc);
            self.spawn_primitive(
                primitive,
                position,
                glam::Quat::from_rotation_y(yaw),
                scale,
                color,
                collision_enabled,
                is_static,
            );
            spawned += 1;
        }

        spawned
    }

    pub fn spawn_grid(
        &mut self,
        primitive: ScriptPrimitive,
        columns: f32,
        rows: f32,
        center: glam::Vec3,
        spacing: glam::Vec3,
        scale: glam::Vec3,
        color: glam::Vec3,
        collision_enabled: bool,
        is_static: bool,
    ) -> u32 {
        let columns = script_count(columns, 128);
        let rows = script_count(rows, 128);
        if columns == 0 || rows == 0 {
            return 0;
        }

        let max_spawn = 1024;
        let x_mid = (columns.saturating_sub(1) as f32) * 0.5;
        let z_mid = (rows.saturating_sub(1) as f32) * 0.5;
        let mut spawned = 0;

        'rows: for row in 0..rows {
            for column in 0..columns {
                if spawned >= max_spawn {
                    break 'rows;
                }
                let position = center
                    + glam::Vec3::new(
                        (column as f32 - x_mid) * spacing.x,
                        0.0,
                        (row as f32 - z_mid) * spacing.z,
                    );
                self.spawn_primitive(
                    primitive,
                    position,
                    glam::Quat::IDENTITY,
                    scale,
                    color,
                    collision_enabled,
                    is_static,
                );
                spawned += 1;
            }
        }

        spawned
    }

    pub fn spawn_resource_cluster(
        &mut self,
        kind: ScriptResourceKind,
        count: f32,
        center: glam::Vec3,
        area: glam::Vec3,
        seed: f32,
    ) -> u32 {
        self.spawn_scatter_range(
            kind.primitive(),
            count,
            center,
            area,
            kind.min_scale(),
            kind.max_scale(),
            kind.min_color(),
            kind.max_color(),
            seed + kind.salt() as f32,
            true,
            true,
        )
    }

    pub fn spawn_tree_cluster(
        &mut self,
        count: f32,
        center: glam::Vec3,
        area: glam::Vec3,
        seed: f32,
    ) -> u32 {
        let count = script_count(count, 256);
        if count == 0 {
            return 0;
        }

        let area = area.abs();
        let mut state = seed.to_bits().wrapping_mul(1664525).wrapping_add(count);
        let mut spawned = 0;
        for _ in 0..count {
            let rx = script_unit_random(&mut state);
            let rz = script_unit_random(&mut state);
            let rs = script_unit_random(&mut state);
            let yaw = script_unit_random(&mut state) * std::f32::consts::TAU;
            let height = 1.4 + rs * 2.2;
            let position = center + glam::Vec3::new((rx - 0.5) * area.x, 0.0, (rz - 0.5) * area.z);
            let trunk_id = self.spawn_primitive(
                ScriptPrimitive::Cylinder,
                position + glam::Vec3::new(0.0, height * 0.45, 0.0),
                glam::Quat::from_rotation_y(yaw),
                glam::Vec3::new(0.22 + rs * 0.18, height, 0.22 + rs * 0.18),
                glam::Vec3::new(0.38, 0.23, 0.12),
                true,
                true,
            );
            let canopy_id = self.spawn_primitive(
                ScriptPrimitive::Sphere,
                glam::Vec3::new(0.0, height * 0.62, 0.0),
                glam::Quat::IDENTITY,
                glam::Vec3::splat(0.9 + rs * 0.75),
                glam::Vec3::new(0.12, 0.38 + rs * 0.18, 0.16),
                false,
                true,
            );
            self.queue_scene_update(SceneUpdate::AttachEntity {
                id: canopy_id,
                parent_id: Some(trunk_id),
            });
            spawned += 1;
        }

        spawned
    }

    pub fn spawn_campfire(
        &mut self,
        position: glam::Vec3,
        radius: f32,
        lit: bool,
    ) -> u32 {
        let radius = script_radius(radius, 1.0);
        let base_id = self.spawn_primitive(
            ScriptPrimitive::Cylinder,
            position + glam::Vec3::new(0.0, 0.08 * radius, 0.0),
            glam::Quat::IDENTITY,
            glam::Vec3::new(radius, 0.16 * radius, radius),
            glam::Vec3::new(0.18, 0.16, 0.14),
            true,
            true,
        );

        for index in 0..4 {
            let angle = index as f32 * std::f32::consts::FRAC_PI_2;
            let log_id = self.spawn_primitive(
                ScriptPrimitive::Cylinder,
                glam::Vec3::new(angle.cos() * 0.18 * radius, 0.15 * radius, angle.sin() * 0.18 * radius),
                glam::Quat::from_rotation_y(angle),
                glam::Vec3::new(0.1 * radius, 0.38 * radius, 0.1 * radius),
                glam::Vec3::new(0.42, 0.24, 0.12),
                false,
                true,
            );
            self.queue_scene_update(SceneUpdate::AttachEntity {
                id: log_id,
                parent_id: Some(base_id),
            });
        }

        if lit {
            let flame_id = self.spawn_primitive(
                ScriptPrimitive::Cone,
                glam::Vec3::new(0.0, 0.46 * radius, 0.0),
                glam::Quat::IDENTITY,
                glam::Vec3::new(0.36 * radius, 0.82 * radius, 0.36 * radius),
                glam::Vec3::new(1.0, 0.42, 0.12),
                false,
                true,
            );
            self.queue_scene_update(SceneUpdate::AttachEntity {
                id: flame_id,
                parent_id: Some(base_id),
            });
            self.spawn_light(
                ScriptLightType::Point,
                position + glam::Vec3::new(0.0, 1.8 * radius, 0.0),
                glam::Vec3::NEG_Y,
                glam::Vec3::new(1.0, 0.52, 0.22),
                2.5 * radius,
                12.0 * radius,
            );
        }

        base_id
    }

    pub fn spawn_furry_npc(
        &mut self,
        species: ScriptFurrySpecies,
        position: glam::Vec3,
        color: glam::Vec3,
        scale: f32,
        script_a: &str,
        script_b: &str,
    ) -> u32 {
        let scale = script_radius(scale, 1.0);
        let accent = species.accent_color(color);
        let body_id = self.spawn_primitive_on_layer(
            ScriptPrimitive::Capsule,
            position + glam::Vec3::new(0.0, 0.9 * scale, 0.0),
            glam::Quat::IDENTITY,
            glam::Vec3::new(0.55 * scale, 1.45 * scale, 0.55 * scale),
            accent,
            true,
            false,
            crate::world::LAYER_CHARACTER,
        );

        let head_id = self.spawn_primitive(
            ScriptPrimitive::Sphere,
            glam::Vec3::new(0.0, 0.88 * scale, 0.0),
            glam::Quat::IDENTITY,
            glam::Vec3::splat(0.46 * scale),
            accent.lerp(glam::Vec3::ONE, 0.18),
            false,
            true,
        );
        self.queue_scene_update(SceneUpdate::AttachEntity {
            id: head_id,
            parent_id: Some(body_id),
        });

        let ear_scale = species.ear_scale() * scale;
        for side in [-1.0_f32, 1.0] {
            let ear_id = self.spawn_primitive(
                ScriptPrimitive::Cone,
                glam::Vec3::new(side * 0.24 * scale, 1.26 * scale, 0.0),
                glam::Quat::IDENTITY,
                ear_scale,
                accent.lerp(glam::Vec3::ONE, 0.1),
                false,
                true,
            );
            self.queue_scene_update(SceneUpdate::AttachEntity {
                id: ear_id,
                parent_id: Some(body_id),
            });
        }

        let tail_id = self.spawn_primitive(
            ScriptPrimitive::Capsule,
            glam::Vec3::new(0.0, -0.1 * scale, 0.48 * scale),
            glam::Quat::from_rotation_y(std::f32::consts::PI),
            species.tail_scale() * scale,
            accent,
            false,
            true,
        );
        self.queue_scene_update(SceneUpdate::AttachEntity {
            id: tail_id,
            parent_id: Some(body_id),
        });

        self.queue_script_stack(body_id, &[script_a, script_b]);
        body_id
    }

    pub fn set_script(&mut self, entity_id: u32, name: &str) {
        self.queue_script_stack(entity_id, &[name]);
    }

    pub fn set_scripts(
        &mut self,
        entity_id: u32,
        first: &str,
        second: &str,
        third: &str,
        fourth: &str,
    ) {
        self.queue_script_stack(entity_id, &[first, second, third, fourth]);
    }

    fn queue_script_stack(&mut self, entity_id: u32, names: &[&str]) {
        let names: Vec<String> = names
            .iter()
            .map(|name| name.trim())
            .filter(|name| !name.is_empty())
            .take(MAX_SCRIPTS_PER_ENTITY)
            .map(ToString::to_string)
            .collect();
        if names.is_empty() {
            return;
        }
        self.queue_scene_update(SceneUpdate::SetScripts {
            id: entity_id,
            names,
        });
    }

    pub fn fire_projectile(
        &mut self,
        origin: glam::Vec3,
        direction: glam::Vec3,
        speed: f32,
        lifetime: f32,
        radius: f32,
        damage: f32,
        gravity_scale: f32,
    ) {
        self.fire_projectile_with_color(
            origin,
            direction,
            speed,
            lifetime,
            radius,
            damage,
            gravity_scale,
            glam::Vec3::new(1.0, 0.82, 0.22),
        );
    }

    pub fn fire_projectile_from_self(
        &mut self,
        speed: f32,
        lifetime: f32,
        radius: f32,
        damage: f32,
        gravity_scale: f32,
    ) {
        let shot = self.transform().map(|transform| {
            let direction = transform.rotation * glam::Vec3::NEG_Z;
            let origin = transform.position + direction * (radius.max(0.01) + 0.45);
            (origin, direction)
        });

        if let Some((origin, direction)) = shot {
            self.fire_projectile(
                origin,
                direction,
                speed,
                lifetime,
                radius,
                damage,
                gravity_scale,
            );
        }
    }

    pub fn fire_projectile_with_color(
        &mut self,
        origin: glam::Vec3,
        direction: glam::Vec3,
        speed: f32,
        lifetime: f32,
        radius: f32,
        damage: f32,
        gravity_scale: f32,
        color: glam::Vec3,
    ) {
        let direction = if direction.length_squared() > 0.000001 {
            direction.normalize()
        } else {
            glam::Vec3::NEG_Z
        };
        let speed = speed.max(0.0);
        let radius = radius.max(0.01);
        let velocity = direction * speed;
        let rotation = glam::Quat::from_rotation_arc(glam::Vec3::NEG_Z, direction);

        self.queue_scene_update(SceneUpdate::SpawnProjectile {
            id: next_script_projectile_id(),
            position: origin.to_array(),
            rotation: rotation.to_array(),
            velocity: velocity.to_array(),
            radius,
            lifetime: lifetime.max(0.01),
            damage: damage.max(0.0),
            color: color.clamp(glam::Vec3::ZERO, glam::Vec3::ONE).to_array(),
            layer: crate::world::LAYER_PROP,
            gravity_scale,
            owner: Some(self.entity.to_bits().get()),
        });
    }

    pub fn queue_scene_update(&mut self, update: SceneUpdate) {
        if let Some(queue) = self.scene_commands.as_deref_mut() {
            queue.push(update);
        } else {
            log::warn!("Script requested a scene update, but no scene command queue is attached");
        }
    }

    pub fn load_world_scene(
        &mut self,
        scene_path: impl Into<String>,
        mode: crate::project::scene::SceneTransitionMode,
    ) {
        self.queue_scene_update(SceneUpdate::CallWorldScene {
            scene_path: scene_path.into(),
            mode,
        });
    }

    pub fn load_ui_scene(
        &mut self,
        scene_path: impl Into<String>,
        layer: crate::ui::UiLayer,
        mode: crate::project::scene::SceneTransitionMode,
    ) {
        self.queue_scene_update(SceneUpdate::CallUiScene {
            scene_path: scene_path.into(),
            layer,
            mode,
        });
    }

    pub fn transition_scene(
        &mut self,
        target: crate::project::scene::SceneRef,
        mode: crate::project::scene::SceneTransitionMode,
    ) {
        self.queue_scene_update(SceneUpdate::TransitionScene { target, mode });
    }

    pub fn track_transform_to_xr_pose(
        &mut self,
        space: XrPoseSpace,
        position_offset: glam::Vec3,
        rotation_offset: glam::Quat,
    ) {
        if let Some(pose) = self.xr_pose(space) {
            if let Some(mut transform) = self.transform_mut() {
                transform.position = pose.position + pose.rotation * position_offset;
                transform.rotation = pose.rotation * rotation_offset;
            }
        }
    }
}

// --- Default Scripts ---

/// A simple test script that makes an entity bounce up and down.
pub struct TestBounceScript {
    pub time: f32,
}

impl TestBounceScript {
    pub fn new() -> Self {
        Self { time: 0.0 }
    }
}

impl FuckScript for TestBounceScript {
    fn on_update(&mut self, ctx: &mut ScriptContext) {
        self.time += ctx.dt;
        if let Ok(mut transform) = ctx.world.get::<&mut Transform>(ctx.entity) {
            transform.position.y = 1.0 + (self.time * 2.0).sin() * 0.5;
        }
    }
}

/// Script that implements the CrowdAgent behavior.
pub struct CrowdAgentScript;

impl FuckScript for CrowdAgentScript {
    fn on_start(&mut self, ctx: &mut ScriptContext) {
        use crate::world::{AgentState, CrowdAgent};
        // Ensure the entity has a CrowdAgent component to work with
        if ctx.world.get::<&CrowdAgent>(ctx.entity).is_err() {
            let mut seed = (ctx.entity.id() * 12345) as u32;
            let mut rand = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed as f32) / (u32::MAX as f32)
            };

            let state = if rand() > 0.8 {
                AgentState::Fleeing
            } else if rand() > 0.5 {
                AgentState::Running
            } else {
                AgentState::Walking
            };
            let max_speed = match state {
                AgentState::Fleeing => 8.0,
                AgentState::Running => 5.0,
                _ => 2.0,
            };

            let _ = ctx.world.insert_one(
                ctx.entity,
                CrowdAgent {
                    velocity: glam::Vec3::ZERO,
                    target: glam::Vec3::new((rand() - 0.5) * 50.0, 1.0, (rand() - 0.5) * 50.0),
                    state,
                    max_speed,
                    stuck_timer: 0.0,
                    last_pos: glam::Vec3::ZERO,
                },
            );
        }
    }

    fn on_update(&mut self, ctx: &mut ScriptContext) {
        use crate::world::{AgentState, CrowdAgent, Player};

        let mut player_pos = None;
        for (_e, (t, _p)) in ctx.world.query::<(&Transform, &Player)>().iter() {
            player_pos = Some(t.position);
            break;
        }

        let mut update = None;

        if let Ok(mut agent) = ctx.world.get::<&mut CrowdAgent>(ctx.entity) {
            if let Ok(transform) = ctx.world.get::<&Transform>(ctx.entity) {
                let pos = transform.position;

                // --- Simple RNG for the agent ---
                let mut seed =
                    (ctx.entity.id() as u32).wrapping_mul(12345) ^ (pos.x * 100.0) as u32;
                let mut rand = || {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed as f32) / (u32::MAX as f32)
                };

                // --- Stuck Detection & Avoidance ---
                if (pos - agent.last_pos).length() < 0.1 * ctx.dt {
                    agent.stuck_timer += ctx.dt;
                } else {
                    agent.stuck_timer = 0.0;
                }
                agent.last_pos = pos;

                if agent.stuck_timer > 1.5 {
                    // Try to unstuck: pick a new target immediately and random nudge
                    agent.target =
                        glam::Vec3::new((rand() - 0.5) * 60.0, pos.y, (rand() - 0.5) * 60.0);
                    agent.velocity =
                        glam::Vec3::new(rand() - 0.5, 0.0, rand() - 0.5).normalize() * 2.0;
                    agent.stuck_timer = 0.0;
                }

                // --- Behavior: Flee from Player ---
                if let Some(p_pos) = player_pos {
                    if agent.state == AgentState::Fleeing && (p_pos - pos).length() < 12.0 {
                        // Run AWAY from player
                        let away = (pos - p_pos).normalize();
                        agent.target = pos + away * 15.0;
                    }
                }

                let target = agent.target;
                let max_speed = agent.max_speed;

                let to_target = target - pos;
                let dist = to_target.length();

                // Pick a new target if reached, or if we haven't picked one yet (target is 0,0,0)
                if dist < 1.5 || target.length_squared() < 0.001 {
                    agent.target =
                        glam::Vec3::new((rand() - 0.5) * 60.0, pos.y, (rand() - 0.5) * 60.0);
                    // Also give a tiny nudge to start moving
                    agent.velocity = (agent.target - pos).normalize() * 0.1;
                } else {
                    let desired = to_target.normalize() * max_speed;
                    let steer_force = if agent.state == AgentState::Fleeing {
                        20.0
                    } else {
                        8.0
                    };
                    let steering = (desired - agent.velocity) * steer_force;

                    agent.velocity += steering * ctx.dt;
                    if agent.velocity.length() > max_speed {
                        agent.velocity = agent.velocity.normalize() * max_speed;
                    }

                    let new_pos = pos + agent.velocity * ctx.dt;
                    let new_rot = if agent.velocity.length_squared() > 0.1 {
                        let angle = agent.velocity.x.atan2(agent.velocity.z);
                        glam::Quat::from_rotation_y(angle)
                    } else {
                        transform.rotation
                    };

                    update = Some((new_pos, new_rot));
                }
            }
        }

        if let Some((pos, rot)) = update {
            if let Ok(mut transform) = ctx.world.get::<&mut Transform>(ctx.entity) {
                transform.position = pos;
                transform.rotation = rot;
            }
        }
    }
}

pub(crate) fn apply_vehicle_suspension(
    physics: &mut PhysicsWorld,
    rb_handle: rapier3d::prelude::RigidBodyHandle,
    dt: f32,
    chassis_half_height: f32,
    suspension_travel: f32,
    rest_compression: f32,
    damping_per_weight: f32,
    max_force_weight_multiplier: f32,
) -> Option<rapier3d::na::UnitQuaternion<f32>> {
    use rapier3d::prelude::*;

    let dt = dt.clamp(0.0, 1.0 / 15.0);

    let (position, velocity, mass, rotation) = {
        let body = physics.rigid_body_set.get(rb_handle)?;
        (
            *body.translation(),
            *body.linvel(),
            body.mass().max(0.1),
            *body.rotation(),
        )
    };

    let ray_length = chassis_half_height + suspension_travel;
    let ray_hit = physics
        .query_pipeline
        .cast_ray(
            &physics.rigid_body_set,
            &physics.collider_set,
            &Ray::new(
                point![position.x, position.y, position.z],
                vector![0.0, -1.0, 0.0],
            ),
            ray_length,
            true,
            QueryFilter::default().exclude_rigid_body(rb_handle),
        )
        .map(|(_, toi)| toi);

    let toi = ray_hit?;

    let ground_clearance = (toi - chassis_half_height).max(0.0);
    if ground_clearance >= suspension_travel {
        return None;
    }

    let compression = ((suspension_travel - ground_clearance) / suspension_travel).clamp(0.0, 1.0);
    let weight = (mass * physics.gravity.y.abs()).max(1.0);
    let spring_force = (weight / rest_compression.clamp(0.1, 0.95)) * compression;
    let damping_force = -velocity.y * weight * damping_per_weight;

    // If the chassis itself is contacting the ground, let Rapier's contact solver resolve it.
    // Otherwise the fake suspension adds energy exactly at impact and launches the body upward.
    let contact_blend = (ground_clearance / 0.08).clamp(0.0, 1.0);
    let force_y = ((spring_force + damping_force) * contact_blend)
        .clamp(0.0, weight * max_force_weight_multiplier);

    if force_y > 0.0 {
        if let Some(body) = physics.rigid_body_set.get_mut(rb_handle) {
            body.apply_impulse(vector![0.0, force_y * dt, 0.0], true);
        }
    }

    Some(rotation)
}

pub(crate) fn apply_vehicle_lateral_damping(
    physics: &mut PhysicsWorld,
    rb_handle: rapier3d::prelude::RigidBodyHandle,
    dt: f32,
    damping: f32,
) {
    use rapier3d::prelude::*;

    let dt = dt.clamp(0.0, 1.0 / 15.0);
    if let Some(body) = physics.rigid_body_set.get_mut(rb_handle) {
        let vel = *body.linvel();
        body.apply_impulse(
            vector![-vel.x * damping * dt, 0.0, -vel.z * damping * dt],
            true,
        );
    }
}

fn wrap_angle(mut angle: f32) -> f32 {
    while angle > std::f32::consts::PI {
        angle -= std::f32::consts::TAU;
    }
    while angle < -std::f32::consts::PI {
        angle += std::f32::consts::TAU;
    }
    angle
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct VehicleAutopilotBounds {
    min_x: f32,
    max_x: f32,
    min_z: f32,
    max_z: f32,
}

impl VehicleAutopilotBounds {
    fn from_center_half(center_x: f32, center_z: f32, half_x: f32, half_z: f32) -> Self {
        let half_x = half_x.abs().max(1.0);
        let half_z = half_z.abs().max(1.0);
        Self {
            min_x: center_x - half_x,
            max_x: center_x + half_x,
            min_z: center_z - half_z,
            max_z: center_z + half_z,
        }
    }

    fn default_test_map() -> Self {
        Self::from_center_half(0.0, 0.0, 50.0, 50.0)
    }

    fn union(self, other: Self) -> Self {
        Self {
            min_x: self.min_x.min(other.min_x),
            max_x: self.max_x.max(other.max_x),
            min_z: self.min_z.min(other.min_z),
            max_z: self.max_z.max(other.max_z),
        }
    }
}

pub(crate) fn vehicle_autopilot_bounds_from_world(world: &World) -> VehicleAutopilotBounds {
    let mut bounds: Option<VehicleAutopilotBounds> = None;

    for (_, (ground, transform)) in world.query::<(&GroundPlane, &Transform)>().iter() {
        let ground_bounds = VehicleAutopilotBounds::from_center_half(
            transform.position.x,
            transform.position.z,
            ground.half_extents.x,
            ground.half_extents.y,
        );
        bounds = Some(match bounds {
            Some(existing) => existing.union(ground_bounds),
            None => ground_bounds,
        });
    }

    bounds.unwrap_or_else(VehicleAutopilotBounds::default_test_map)
}

pub(crate) fn vehicle_autopilot_steering_with_bounds(
    position: &rapier3d::na::Vector3<f32>,
    rotation: &rapier3d::na::UnitQuaternion<f32>,
    bounds: VehicleAutopilotBounds,
) -> f32 {
    let forward = rotation.transform_vector(&rapier3d::na::Vector3::new(0.0, 0.0, -1.0));
    let current_yaw = (-forward.x).atan2(-forward.z);

    const EDGE_START_MARGIN: f32 = 3.0;
    const EDGE_LOOKAHEAD_DISTANCE: f32 = 7.0;
    let flat_forward_len = (forward.x * forward.x + forward.z * forward.z)
        .sqrt()
        .max(0.001);
    let projected_x = position.x + (forward.x / flat_forward_len) * EDGE_LOOKAHEAD_DISTANCE;
    let projected_z = position.z + (forward.z / flat_forward_len) * EDGE_LOOKAHEAD_DISTANCE;
    let margins = [
        (projected_x - bounds.min_x, 1.0, 0.0),
        (bounds.max_x - projected_x, -1.0, 0.0),
        (projected_z - bounds.min_z, 0.0, 1.0),
        (bounds.max_z - projected_z, 0.0, -1.0),
    ];

    let mut inward_x: f32 = 0.0;
    let mut inward_z: f32 = 0.0;
    let mut edge_pressure: f32 = 0.0;

    for (margin, normal_x, normal_z) in margins {
        let pressure = ((EDGE_START_MARGIN - margin) / EDGE_START_MARGIN).clamp(0.0, 1.0);
        if pressure > 0.0 {
            inward_x += normal_x * pressure;
            inward_z += normal_z * pressure;
            edge_pressure = edge_pressure.max(pressure);
        }
    }

    let inward_len = (inward_x * inward_x + inward_z * inward_z).sqrt();
    if inward_len <= 0.001 {
        return 0.0;
    }

    let inward_x = inward_x / inward_len;
    let inward_z = inward_z / inward_len;
    let tangent_x = -inward_z;
    let tangent_z = inward_x;
    let inward_weight = 0.35 + edge_pressure * 1.45;
    let tangent_weight = 1.05 - edge_pressure * 0.65;
    let desired_x = inward_x * inward_weight + tangent_x * tangent_weight;
    let desired_z = inward_z * inward_weight + tangent_z * tangent_weight;
    let desired_len = (desired_x * desired_x + desired_z * desired_z)
        .sqrt()
        .max(0.001);
    let desired_x = desired_x / desired_len;
    let desired_z = desired_z / desired_len;
    let desired_yaw = (-desired_x).atan2(-desired_z);
    let steering_strength = edge_pressure.powf(0.35);
    let steering_gain = 1.6;

    (wrap_angle(desired_yaw - current_yaw) * steering_strength * steering_gain).clamp(-5.0, 5.0)
}

/// Script that implements Vehicle behavior using physics.
pub struct VehicleScript;

impl FuckScript for VehicleScript {
    fn on_start(&mut self, ctx: &mut ScriptContext) {
        use crate::world::Vehicle;
        // Ensure entity has Vehicle component
        let has_vehicle = {
            if let Ok(mut vehicle) = ctx.world.get::<&mut Vehicle>(ctx.entity) {
                vehicle.speed = vehicle.speed.max(120.0);
                vehicle.max_speed = vehicle.max_speed.max(25.0);
                vehicle.accelerating = true;
                true
            } else {
                false
            }
        };

        if !has_vehicle {
            let _ = ctx.world.insert_one(
                ctx.entity,
                Vehicle {
                    speed: 120.0,
                    max_speed: 25.0,
                    steering: 0.0,
                    accelerating: true,
                },
            );
        }

        // Add physics if missing
        if ctx
            .world
            .get::<&crate::world::RigidBodyHandle>(ctx.entity)
            .is_err()
        {
            let pos = if let Ok(transform) = ctx.world.get::<&Transform>(ctx.entity) {
                Some(transform.position.to_array())
            } else {
                None
            };

            if let Some(position) = pos {
                let handle = ctx.physics.add_box_rigid_body(
                    ctx.entity.to_bits().get() as u128,
                    position,
                    [1.0, 0.5, 2.0], // Vehicle half-extents
                    true,
                    super::LAYER_VEHICLE,
                    u32::MAX,
                );
                let _ = ctx
                    .world
                    .insert_one(ctx.entity, crate::world::RigidBodyHandle(handle));
            }
        }
    }

    fn on_update(&mut self, ctx: &mut ScriptContext) {
        use crate::world::RigidBodyHandle;
        use crate::world::Vehicle;
        use rapier3d::prelude::*;

        if ctx.world.get::<&Vehicle>(ctx.entity).is_ok() {
            let autopilot_bounds = vehicle_autopilot_bounds_from_world(ctx.world);
            let (drive_impulse, max_speed) = {
                let mut vehicle = ctx
                    .world
                    .get::<&mut Vehicle>(ctx.entity)
                    .expect("Vehicle existed but could not be borrowed mutably");
                vehicle.speed = vehicle.speed.max(120.0);
                vehicle.max_speed = vehicle.max_speed.max(25.0);
                vehicle.accelerating = true;
                (vehicle.speed, vehicle.max_speed)
            };

            if let Ok(rb_handle) = ctx.world.get::<&RigidBodyHandle>(ctx.entity) {
                let dt = ctx.dt.clamp(0.0, 1.0 / 15.0);
                let suspension_rot = apply_vehicle_suspension(
                    ctx.physics,
                    rb_handle.0,
                    dt,
                    0.5,
                    0.75,
                    0.75,
                    0.18,
                    1.35,
                );

                if suspension_rot.is_some() {
                    apply_vehicle_lateral_damping(ctx.physics, rb_handle.0, dt, 8.0);
                }

                if let Some(body) = ctx.physics.rigid_body_set.get_mut(rb_handle.0) {
                    let steering = vehicle_autopilot_steering_with_bounds(
                        body.translation(),
                        body.rotation(),
                        autopilot_bounds,
                    );
                    let yaw_delta = steering * dt;
                    if yaw_delta.abs() > 0.0001 {
                        let turn = rapier3d::na::UnitQuaternion::from_axis_angle(
                            &rapier3d::na::Vector3::y_axis(),
                            yaw_delta,
                        );
                        let next_rot = turn * *body.rotation();
                        body.set_rotation(next_rot, true);
                    }

                    let rot = suspension_rot.unwrap_or_else(|| *body.rotation());
                    let forward_dir = rot.transform_vector(&vector![0.0, 0.0, -1.0]);
                    let forward_speed = body.linvel().dot(&forward_dir);
                    if forward_speed < max_speed {
                        body.apply_impulse(forward_dir * drive_impulse * dt, true);
                    }
                }
            }
        }
    }
}

/// Script for Police behavior - Chases the player if nearby.
pub struct PoliceAgentScript;

impl FuckScript for PoliceAgentScript {
    fn on_update(&mut self, ctx: &mut ScriptContext) {
        use crate::world::{AgentState, CrowdAgent, Player};

        // Initialize as CrowdAgent if missing
        if ctx.world.get::<&CrowdAgent>(ctx.entity).is_err() {
            let _ = ctx.world.insert_one(
                ctx.entity,
                CrowdAgent {
                    velocity: glam::Vec3::ZERO,
                    target: glam::Vec3::ZERO,
                    state: AgentState::Chasing,
                    max_speed: 10.0,
                    stuck_timer: 0.0,
                    last_pos: glam::Vec3::ZERO,
                },
            );
        }

        let mut player_pos = None;
        for (_e, (t, _p)) in ctx.world.query::<(&Transform, &Player)>().iter() {
            player_pos = Some(t.position);
            break;
        }

        if let Some(p_pos) = player_pos {
            if let Ok(mut agent) = ctx.world.get::<&mut CrowdAgent>(ctx.entity) {
                let dist = (p_pos - agent.last_pos).length();
                if dist < 30.0 {
                    agent.state = AgentState::Chasing;
                    agent.target = p_pos;
                    agent.max_speed = 10.0;
                } else {
                    agent.state = AgentState::Idle;
                    agent.max_speed = 0.0;
                }
            }
        }

        // Let the base CrowdAgentScript logic (if attached) or simple manual move handle it.
        // Actually, we'll just implement the movement here too for simplicity.
        let mut update = None;
        if let Ok(mut agent) = ctx.world.get::<&mut CrowdAgent>(ctx.entity) {
            if let Ok(transform) = ctx.world.get::<&Transform>(ctx.entity) {
                let pos = transform.position;
                agent.last_pos = pos;

                let to_target = agent.target - pos;
                if to_target.length() > 1.0 {
                    let desired = to_target.normalize() * agent.max_speed;
                    let steering = (desired - agent.velocity) * 10.0;
                    agent.velocity += steering * ctx.dt;
                    let new_pos = pos + agent.velocity * ctx.dt;
                    let new_rot = if agent.velocity.length_squared() > 0.1 {
                        let angle = agent.velocity.x.atan2(agent.velocity.z);
                        glam::Quat::from_rotation_y(angle)
                    } else {
                        transform.rotation
                    };
                    update = Some((new_pos, new_rot));
                }
            }
        }

        if let Some((pos, rot)) = update {
            if let Ok(mut transform) = ctx.world.get::<&mut Transform>(ctx.entity) {
                transform.position = pos;
                transform.rotation = rot;
            }
        }
    }
}

/// Advanced Traffic AI that stops for obstacles.
pub struct TrafficAIScript;

impl FuckScript for TrafficAIScript {
    fn on_update(&mut self, ctx: &mut ScriptContext) {
        use crate::world::RigidBodyHandle;
        use crate::world::Vehicle;
        use rapier3d::prelude::*;

        // Ensure entity has Vehicle component
        if ctx.world.get::<&Vehicle>(ctx.entity).is_err() {
            let _ = ctx.world.insert_one(
                ctx.entity,
                Vehicle {
                    speed: 0.0,
                    max_speed: 15.0,
                    steering: 0.0,
                    accelerating: true,
                },
            );
        }

        if let Ok(rb_handle) = ctx.world.get::<&RigidBodyHandle>(ctx.entity) {
            let mut obstacle_in_front = false;

            // 1. Raycast in front for other cars/players/obstacles
            let ray_hit = {
                if let Some(body) = ctx.physics.rigid_body_set.get(rb_handle.0) {
                    let position = body.translation();
                    let rot = *body.rotation();
                    let forward = rot.transform_vector(&vector![0.0, 0.0, -1.0]);

                    let ray_origin = point![
                        position.x + forward.x * 2.1,
                        position.y,
                        position.z + forward.z * 2.1
                    ];
                    let ray_dir = forward;
                    let max_dist = 5.0;

                    ctx.physics.query_pipeline.cast_ray(
                        &ctx.physics.rigid_body_set,
                        &ctx.physics.collider_set,
                        &Ray::new(ray_origin, ray_dir),
                        max_dist,
                        true,
                        QueryFilter::default().exclude_rigid_body(rb_handle.0),
                    )
                } else {
                    None
                }
            };

            if ray_hit.is_some() {
                obstacle_in_front = true;
            }

            // 2. Suspension & Drive logic
            let dt = ctx.dt.clamp(0.0, 1.0 / 15.0);
            if let Some(rot) =
                apply_vehicle_suspension(ctx.physics, rb_handle.0, dt, 0.5, 0.75, 0.75, 0.18, 1.35)
            {
                apply_vehicle_lateral_damping(ctx.physics, rb_handle.0, dt, 10.0);
                if let Some(body) = ctx.physics.rigid_body_set.get_mut(rb_handle.0) {
                    if !obstacle_in_front {
                        let forward_dir = rot.transform_vector(&vector![0.0, 0.0, -1.0]);
                        body.apply_impulse(forward_dir * 40.0 * dt, true);
                    } else {
                        // Brake
                        let vel = *body.linvel();
                        body.apply_impulse(
                            vector![-vel.x * 10.0 * dt, 0.0, -vel.z * 10.0 * dt],
                            true,
                        );
                    }
                }
            }
        }
    }
}

/// Basic NPC that aims at the player.
pub struct WeaponNPCScript;

impl FuckScript for WeaponNPCScript {
    fn on_update(&mut self, ctx: &mut ScriptContext) {
        use crate::world::Player;

        let mut player_pos = None;
        for (_e, (t, _p)) in ctx.world.query::<(&Transform, &Player)>().iter() {
            player_pos = Some(t.position);
            break;
        }

        if let Some(p_pos) = player_pos {
            if let Ok(mut transform) = ctx.world.get::<&mut Transform>(ctx.entity) {
                let to_player = (p_pos - transform.position).normalize();
                if to_player.length_squared() > 0.01 {
                    let angle = to_player.x.atan2(to_player.z);
                    let target_rot = glam::Quat::from_rotation_y(angle);
                    // Smooth rotate
                    transform.rotation = transform.rotation.slerp(target_rot, 5.0 * ctx.dt);
                }
            }
        }
    }
}

/// Radius-gated enemy movement that tracks the player position.
pub struct EnemyTrackerScript {
    pub track_radius: f32,
    pub move_speed: f32,
    pub stop_distance: f32,
    pub turn_speed: f32,
}

impl Default for EnemyTrackerScript {
    fn default() -> Self {
        Self {
            track_radius: 18.0,
            move_speed: 3.5,
            stop_distance: 1.25,
            turn_speed: 9.0,
        }
    }
}

impl FuckScript for EnemyTrackerScript {
    fn on_update(&mut self, ctx: &mut ScriptContext) {
        let _ = ctx.track_player(
            self.track_radius,
            self.move_speed,
            self.stop_distance,
            self.turn_speed,
        );
    }
}

fn muzzle_from_transform(
    ctx: &ScriptContext,
    radius: f32,
    forward_offset: f32,
) -> Option<(glam::Vec3, glam::Vec3)> {
    let transform = ctx.transform()?;
    let direction = transform.rotation * glam::Vec3::NEG_Z;
    let origin = transform.position + direction * (forward_offset + radius.max(0.01));
    Some((origin, direction))
}

#[derive(Debug)]
pub struct GunWeaponScript {
    cooldown_remaining: f32,
}

impl Default for GunWeaponScript {
    fn default() -> Self {
        Self {
            cooldown_remaining: 0.0,
        }
    }
}

impl FuckScript for GunWeaponScript {
    fn on_update(&mut self, ctx: &mut ScriptContext) {
        self.cooldown_remaining = (self.cooldown_remaining - ctx.dt).max(0.0);
        if !ctx.action_pressed("fire") || self.cooldown_remaining > 0.0 {
            return;
        }

        let Some((origin, direction)) = muzzle_from_transform(ctx, 0.055, 0.65) else {
            return;
        };

        ctx.fire_projectile_with_color(
            origin,
            direction,
            65.0,
            3.0,
            0.055,
            12.0,
            0.0,
            glam::Vec3::new(1.0, 0.76, 0.18),
        );
        ctx.request_haptic(XrHand::Right, 0.35, 0.035);
        self.cooldown_remaining = 0.12;
    }
}

#[derive(Debug)]
pub struct BowWeaponScript {
    charge_seconds: f32,
}

impl Default for BowWeaponScript {
    fn default() -> Self {
        Self {
            charge_seconds: 0.0,
        }
    }
}

impl FuckScript for BowWeaponScript {
    fn on_update(&mut self, ctx: &mut ScriptContext) {
        const MAX_CHARGE_SECONDS: f32 = 1.35;

        if ctx.action_pressed("fire") {
            self.charge_seconds = (self.charge_seconds + ctx.dt).min(MAX_CHARGE_SECONDS);
            return;
        }

        if !ctx.action_just_released("fire") {
            self.charge_seconds = 0.0;
            return;
        }

        let charge = (self.charge_seconds / MAX_CHARGE_SECONDS).clamp(0.0, 1.0);
        self.charge_seconds = 0.0;
        if charge < 0.08 {
            return;
        }

        let Some((origin, direction)) = muzzle_from_transform(ctx, 0.035, 0.75) else {
            return;
        };

        ctx.fire_projectile_with_color(
            origin,
            direction,
            18.0 + 34.0 * charge,
            6.0,
            0.035,
            8.0 + 24.0 * charge,
            1.0,
            glam::Vec3::new(0.72, 0.44, 0.16),
        );
        ctx.request_haptic(XrHand::Right, 0.2 + 0.35 * charge, 0.05);
    }
}

pub struct ProjectileScript;

impl FuckScript for ProjectileScript {
    fn on_update(&mut self, ctx: &mut ScriptContext) {
        let has_physics = ctx
            .world
            .get::<&crate::world::RigidBodyHandle>(ctx.entity)
            .is_ok();

        let (expired, fallback_velocity) = {
            if let Ok(mut projectile) = ctx.world.get::<&mut Projectile>(ctx.entity) {
                projectile.age += ctx.dt;
                let expired = projectile.age >= projectile.lifetime;

                if !has_physics {
                    if projectile.gravity_scale != 0.0 {
                        let gravity_scale = projectile.gravity_scale;
                        projectile.velocity +=
                            glam::Vec3::new(0.0, -9.81 * gravity_scale, 0.0) * ctx.dt;
                    }
                    (expired, Some(projectile.velocity))
                } else {
                    (expired, None)
                }
            } else {
                (false, None)
            }
        };

        if expired {
            ctx.despawn_self();
            return;
        }

        if let Some(velocity) = fallback_velocity {
            let dt = ctx.dt;
            if let Some(mut transform) = ctx.transform_mut() {
                transform.position += velocity * dt;
                if velocity.length_squared() > 0.000001 {
                    transform.rotation =
                        glam::Quat::from_rotation_arc(glam::Vec3::NEG_Z, velocity.normalize());
                }
            }
        }
    }

    fn on_collision_start(&mut self, ctx: &mut ScriptContext, other: Entity) {
        let (owner, damage) = {
            ctx.world
                .get::<&Projectile>(ctx.entity)
                .ok()
                .map(|projectile| (projectile.owner, projectile.damage))
                .unwrap_or((None, 0.0))
        };

        if owner == Some(other) {
            return;
        }

        log::info!(
            "Projectile {:?} hit {:?} for {:.1} damage",
            ctx.entity,
            other,
            damage
        );
        ctx.despawn_self();
    }
}

#[derive(Clone, Copy, Debug)]
pub enum XrTrackedTarget {
    Head,
    LeftGrip,
    RightGrip,
    LeftAim,
    RightAim,
}

impl XrTrackedTarget {
    fn pose_space(self) -> XrPoseSpace {
        match self {
            XrTrackedTarget::Head => XrPoseSpace::Head,
            XrTrackedTarget::LeftGrip => XrPoseSpace::Grip(XrHand::Left),
            XrTrackedTarget::RightGrip => XrPoseSpace::Grip(XrHand::Right),
            XrTrackedTarget::LeftAim => XrPoseSpace::Aim(XrHand::Left),
            XrTrackedTarget::RightAim => XrPoseSpace::Aim(XrHand::Right),
        }
    }
}

/// Anchors an entity to the HMD or controller poses exposed by the XR pipeline.
pub struct XrPoseAnchorScript {
    pub target: XrTrackedTarget,
    pub position_offset: glam::Vec3,
    pub rotation_offset: glam::Quat,
    pub smoothing: f32,
}

impl XrPoseAnchorScript {
    pub fn new(target: XrTrackedTarget) -> Self {
        Self {
            target,
            position_offset: glam::Vec3::ZERO,
            rotation_offset: glam::Quat::IDENTITY,
            smoothing: 0.0,
        }
    }
}

impl FuckScript for XrPoseAnchorScript {
    fn on_late_update(&mut self, ctx: &mut ScriptContext) {
        if let Some(pose) = ctx.xr_pose(self.target.pose_space()) {
            let dt = ctx.dt;
            if let Some(mut transform) = ctx.transform_mut() {
                let target_pos = pose.position + pose.rotation * self.position_offset;
                let target_rot = pose.rotation * self.rotation_offset;

                if self.smoothing <= 0.0 || dt <= 0.0 {
                    transform.position = target_pos;
                    transform.rotation = target_rot;
                } else {
                    let blend = (self.smoothing * dt).clamp(0.0, 1.0);
                    transform.position = transform.position.lerp(target_pos, blend);
                    transform.rotation = transform.rotation.slerp(target_rot, blend);
                }
            }
        }
    }
}

/// Small utility script for testing controller input and haptic feedback.
pub struct TriggerHapticsScript;

impl FuckScript for TriggerHapticsScript {
    fn on_xr_input(&mut self, ctx: &mut ScriptContext, event: &XrInputEvent) {
        if event.phase == XrInputPhase::Pressed && event.control == Some(XrInputControl::Trigger) {
            if let Some(hand) = event.hand {
                ctx.request_haptic(hand, 0.45, 0.04);
            }
        }
    }
}

/// A script that logs collisions for debugging.
pub struct CollisionLoggerScript;

impl FuckScript for CollisionLoggerScript {
    fn on_start(&mut self, ctx: &mut ScriptContext) {
        log::info!("CollisionLogger: Started on entity {:?}", ctx.entity);
    }

    fn on_enable(&mut self, ctx: &mut ScriptContext) {
        log::info!("CollisionLogger: Enabled on entity {:?}", ctx.entity);
    }

    fn on_collision_start(&mut self, _ctx: &mut ScriptContext, other: Entity) {
        log::info!("CollisionLogger: HIT entity {:?}", other);
    }

    fn on_collision_end(&mut self, _ctx: &mut ScriptContext, other: Entity) {
        log::info!("CollisionLogger: RELEASED entity {:?}", other);
    }
}

/// A script that despawns the entity when it touches something.
pub struct TouchToDestroyScript;

impl FuckScript for TouchToDestroyScript {
    fn on_collision_start(&mut self, ctx: &mut ScriptContext, other: Entity) {
        log::info!("TouchToDestroy: Hit {:?}, despawning self!", other);
        ctx.despawn_self();
    }
}

/// Editable script placeholder used by the editor before export-time native cache generation.
pub struct CustomFuckScript;

impl FuckScript for CustomFuckScript {
    fn on_start(&mut self, ctx: &mut ScriptContext) {
        log::info!(
            "CustomFuckScript placeholder active on {:?}; export cache will bind edited source",
            ctx.entity
        );
    }
}

/// A component wrapper for dynamic scripts. (Already in mod.rs, but keeping registry here)

/// A registry to map script names strings to constructors.
pub struct ScriptRegistry {
    builders: std::collections::HashMap<String, Box<dyn Fn() -> Box<dyn FuckScript> + Send + Sync>>,
}

impl ScriptRegistry {
    pub fn new() -> Self {
        Self {
            builders: std::collections::HashMap::new(),
        }
    }

    pub fn register<S, F>(&mut self, name: &str, builder: F)
    where
        S: FuckScript + 'static,
        F: Fn() -> S + Send + Sync + 'static,
    {
        self.builders
            .insert(name.to_string(), Box::new(move || Box::new(builder())));
    }

    pub fn create(&self, name: &str) -> Option<Box<dyn FuckScript>> {
        self.builders.get(name).map(|b| b())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.builders.contains_key(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapier3d::prelude::vector;

    fn update_queries(physics: &mut PhysicsWorld) {
        physics
            .query_pipeline
            .update(&physics.rigid_body_set, &physics.collider_set);
    }

    #[test]
    fn xr_action_edges_are_computed_from_previous_snapshot() {
        let mut previous = XrInputSnapshot::default();
        previous.set_action_pressed("fire", false);

        let mut current = XrInputSnapshot::default();
        current.set_action_pressed("fire", true);
        current.update_edges_from(&previous);

        let fire = current.action_state("fire");
        assert!(fire.pressed);
        assert!(fire.just_pressed);
        assert!(!fire.just_released);
    }

    #[test]
    fn xr_controller_edges_emit_trigger_events() {
        let previous = XrInputSnapshot::default();
        let mut current = XrInputSnapshot::default();
        current.right.trigger = 1.0;
        current.populate_builtin_actions_from_controls();
        current.update_edges_from(&previous);

        let events = current.diff_events(&previous);
        assert!(events.iter().any(|event| {
            event.hand == Some(XrHand::Right)
                && event.control == Some(XrInputControl::Trigger)
                && event.phase == XrInputPhase::Pressed
        }));
        assert!(events.iter().any(|event| {
            event.action.as_deref() == Some("fire") && event.phase == XrInputPhase::Pressed
        }));
    }

    #[test]
    fn fire_projectile_from_self_queues_projectile_spawn() {
        let mut physics = PhysicsWorld::new();
        let mut world = World::new();
        let entity = world.spawn((Transform {
            position: glam::Vec3::new(1.0, 2.0, 3.0),
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
        },));
        let mut haptics = Vec::new();
        let mut scene_commands = Vec::new();

        {
            let mut ctx = ScriptContext::new_with_xr_and_scene_commands(
                entity,
                &mut world,
                &mut physics,
                1.0 / 60.0,
                XrInputSnapshot::default(),
                Some(&mut haptics),
                Some(&mut scene_commands),
            );
            ctx.fire_projectile_from_self(30.0, 4.0, 0.05, 8.0, 1.0);
        }

        assert_eq!(scene_commands.len(), 1);
        match &scene_commands[0] {
            SceneUpdate::SpawnProjectile {
                velocity,
                radius,
                lifetime,
                damage,
                gravity_scale,
                owner,
                ..
            } => {
                assert_eq!(*velocity, [0.0, 0.0, -30.0]);
                assert_eq!(*radius, 0.05);
                assert_eq!(*lifetime, 4.0);
                assert_eq!(*damage, 8.0);
                assert_eq!(*gravity_scale, 1.0);
                assert_eq!(*owner, Some(entity.to_bits().get()));
            }
            other => panic!("expected projectile spawn, got {:?}", other),
        }
    }

    #[test]
    fn track_player_moves_toward_player_inside_radius() {
        let mut physics = PhysicsWorld::new();
        let mut world = World::new();
        let enemy = world.spawn((Transform {
            position: glam::Vec3::ZERO,
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
        },));
        let _player = world.spawn((
            Transform {
                position: glam::Vec3::new(4.0, 1.7, 0.0),
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::ONE,
            },
            crate::world::Player,
        ));

        let mut ctx = ScriptContext::new(enemy, &mut world, &mut physics, 0.5);
        assert!(ctx.track_player(10.0, 2.0, 1.0, 10.0));

        let transform = ctx.world.get::<&Transform>(enemy).unwrap();
        assert!((transform.position.x - 1.0).abs() < 0.001);
        assert!(transform.position.z.abs() < 0.001);
    }

    #[test]
    fn track_player_ignores_player_outside_radius() {
        let mut physics = PhysicsWorld::new();
        let mut world = World::new();
        let enemy = world.spawn((Transform {
            position: glam::Vec3::ZERO,
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
        },));
        let _player = world.spawn((
            Transform {
                position: glam::Vec3::new(20.0, 1.7, 0.0),
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::ONE,
            },
            crate::world::Player,
        ));

        let mut ctx = ScriptContext::new(enemy, &mut world, &mut physics, 0.5);
        assert!(!ctx.track_player(10.0, 2.0, 1.0, 10.0));

        let transform = ctx.world.get::<&Transform>(enemy).unwrap();
        assert_eq!(transform.position, glam::Vec3::ZERO);
    }

    #[test]
    fn vehicle_suspension_does_not_launch_chassis_contact() {
        let mut physics = PhysicsWorld::new();
        physics.add_box_rigid_body(
            0,
            [0.0, -0.1, 0.0],
            [10.0, 0.1, 10.0],
            false,
            crate::world::LAYER_ENVIRONMENT,
            u32::MAX,
        );
        let vehicle = physics.add_box_rigid_body(
            1,
            [0.0, 0.45, 0.0],
            [1.0, 0.5, 2.0],
            true,
            crate::world::LAYER_VEHICLE,
            u32::MAX,
        );
        update_queries(&mut physics);

        {
            let body = physics.rigid_body_set.get_mut(vehicle).unwrap();
            body.set_linvel(vector![0.0, -12.0, 0.0], true);
        }

        let before_y = physics.rigid_body_set.get(vehicle).unwrap().linvel().y;
        assert!(apply_vehicle_suspension(
            &mut physics,
            vehicle,
            1.0 / 60.0,
            0.5,
            0.75,
            0.75,
            0.18,
            1.35,
        )
        .is_some());

        let body = physics.rigid_body_set.get(vehicle).unwrap();
        assert!(body.linvel().y <= before_y + 0.001);
        assert_eq!(body.user_force(), vector![0.0, 0.0, 0.0]);
    }

    #[test]
    fn vehicle_suspension_uses_impulse_not_persistent_force() {
        let mut physics = PhysicsWorld::new();
        physics.add_box_rigid_body(
            0,
            [0.0, -0.1, 0.0],
            [10.0, 0.1, 10.0],
            false,
            crate::world::LAYER_ENVIRONMENT,
            u32::MAX,
        );
        let vehicle = physics.add_box_rigid_body(
            1,
            [0.0, 0.9, 0.0],
            [1.0, 0.5, 2.0],
            true,
            crate::world::LAYER_VEHICLE,
            u32::MAX,
        );
        update_queries(&mut physics);

        assert!(apply_vehicle_suspension(
            &mut physics,
            vehicle,
            1.0 / 60.0,
            0.5,
            0.75,
            0.75,
            0.18,
            1.35,
        )
        .is_some());

        let body = physics.rigid_body_set.get(vehicle).unwrap();
        assert_eq!(body.user_force(), vector![0.0, 0.0, 0.0]);
        assert!(body.linvel().y > 0.0);
    }

    #[test]
    fn vehicle_script_drives_even_without_suspension_contact() {
        let mut physics = PhysicsWorld::new();
        let mut world = World::new();
        let vehicle_body = physics.add_box_rigid_body(
            1,
            [0.0, 5.0, 0.0],
            [1.0, 0.5, 2.0],
            true,
            crate::world::LAYER_VEHICLE,
            u32::MAX,
        );
        update_queries(&mut physics);

        let entity = world.spawn((
            Transform {
                position: glam::Vec3::new(0.0, 5.0, 0.0),
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::ONE,
            },
            crate::world::RigidBodyHandle(vehicle_body),
            crate::world::Vehicle {
                speed: 0.0,
                max_speed: 10.0,
                steering: 0.0,
                accelerating: false,
            },
        ));

        let mut script = VehicleScript;
        let mut ctx = ScriptContext::new(entity, &mut world, &mut physics, 1.0 / 60.0);
        script.on_start(&mut ctx);
        script.on_update(&mut ctx);

        let vehicle = world.get::<&crate::world::Vehicle>(entity).unwrap();
        assert!(vehicle.accelerating);
        assert!(vehicle.speed >= 120.0);
        assert!(vehicle.max_speed >= 25.0);
        let body = physics.rigid_body_set.get(vehicle_body).unwrap();
        assert!(body.linvel().z < -0.001);
    }

    #[test]
    fn vehicle_autopilot_cruises_straight_until_near_edge() {
        let center = rapier3d::na::Vector3::new(0.0, 0.0, 0.0);
        let diagonal_inside = rapier3d::na::Vector3::new(35.0, 0.0, 35.0);
        let forward_inside = rapier3d::na::Vector3::new(0.0, 0.0, -40.0);
        let near_edge = rapier3d::na::Vector3::new(0.0, 0.0, -44.0);
        let side_edge = rapier3d::na::Vector3::new(44.0, 0.0, 0.0);
        let rotation = rapier3d::na::UnitQuaternion::identity();
        let bounds = VehicleAutopilotBounds::default_test_map();

        let center_steer = vehicle_autopilot_steering_with_bounds(&center, &rotation, bounds);
        assert_eq!(center_steer, 0.0, "center driving should stay straight");

        let diagonal_steer =
            vehicle_autopilot_steering_with_bounds(&diagonal_inside, &rotation, bounds);
        assert_eq!(
            diagonal_steer, 0.0,
            "steering should follow square map edges, not an inner circular boundary"
        );

        let forward_steer =
            vehicle_autopilot_steering_with_bounds(&forward_inside, &rotation, bounds);
        assert_eq!(
            forward_steer, 0.0,
            "vehicle should use most of the map before turning"
        );

        let edge_steer = vehicle_autopilot_steering_with_bounds(&near_edge, &rotation, bounds);
        assert!(
            edge_steer.abs() > 0.4,
            "edge steering should be strong enough to avoid the map boundary"
        );

        let side_rotation = rapier3d::na::UnitQuaternion::from_axis_angle(
            &rapier3d::na::Vector3::y_axis(),
            -std::f32::consts::FRAC_PI_2,
        );
        let side_steer = vehicle_autopilot_steering_with_bounds(&side_edge, &side_rotation, bounds);
        assert!(
            side_steer.abs() > 0.4,
            "side-edge steering should also avoid the map boundary"
        );
    }
}
