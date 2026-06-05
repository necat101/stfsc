use crate::physics::PhysicsWorld;
use crate::world::Transform;
use hecs::{Entity, World};
use std::collections::HashMap;

pub const BUILTIN_SCRIPT_NAMES: &[&str] = &[
    "TestBounce",
    "CrowdAgent",
    "Vehicle",
    "PoliceAgent",
    "TrafficAI",
    "WeaponNPC",
    "CollisionLogger",
    "TouchToDestroy",
    "HeadAnchor",
    "LeftHandAnchor",
    "RightHandAnchor",
    "LeftAimAnchor",
    "RightAimAnchor",
    "TriggerHaptics",
];

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

    /// Called when a collision starts.
    fn on_collision_start(&mut self, _ctx: &mut ScriptContext, _other: Entity) {}

    /// Called when a collision ends.
    fn on_collision_end(&mut self, _ctx: &mut ScriptContext, _other: Entity) {}

    /// Called when a trigger overlap starts.
    fn on_trigger_start(&mut self, _ctx: &mut ScriptContext, _other: Entity) {}

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
}

/// Context passed to scripts to allow them to interact with the engine.
pub struct ScriptContext<'a> {
    pub entity: Entity,
    pub world: &'a mut World,
    pub physics: &'a mut PhysicsWorld,
    pub dt: f32,
    pub xr: XrInputSnapshot,
    haptic_requests: Option<&'a mut Vec<XrHapticRequest>>,
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

    /// Helper to despawn the current entity.
    pub fn despawn_self(&mut self) {
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

/// Script that implements Vehicle behavior using physics.
pub struct VehicleScript;

impl FuckScript for VehicleScript {
    fn on_start(&mut self, ctx: &mut ScriptContext) {
        use crate::world::Vehicle;
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

        if let Ok(_vehicle) = ctx.world.get::<&Vehicle>(ctx.entity) {
            if let Ok(rb_handle) = ctx.world.get::<&RigidBodyHandle>(ctx.entity) {
                let ray_hit = {
                    if let Some(body) = ctx.physics.rigid_body_set.get(rb_handle.0) {
                        let position = body.translation();
                        let ray_origin = point![position.x, position.y, position.z];
                        let ray_dir = vector![0.0, -1.0, 0.0];
                        let max_dist = 1.5;

                        ctx.physics
                            .query_pipeline
                            .cast_ray(
                                &ctx.physics.rigid_body_set,
                                &ctx.physics.collider_set,
                                &Ray::new(ray_origin, ray_dir),
                                max_dist,
                                true,
                                QueryFilter::default().exclude_rigid_body(rb_handle.0),
                            )
                            .map(|(_, toi)| (toi, max_dist))
                    } else {
                        None
                    }
                };

                if let Some((toi, max_dist)) = ray_hit {
                    if let Some(body) = ctx.physics.rigid_body_set.get_mut(rb_handle.0) {
                        let stiffness = 300.0;
                        let damping = 15.0;
                        let compression = 1.0 - (toi / max_dist);
                        let up_force = vector![0.0, stiffness * compression, 0.0];
                        body.add_force(up_force, true);

                        let vel = *body.linvel();
                        body.add_force(-vel * damping, true);

                        let rot = *body.rotation();
                        let forward_dir = rot.transform_vector(&vector![0.0, 0.0, -1.0]);
                        body.add_force(forward_dir * 30.0, true);
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
            let ground_hit = {
                if let Some(body) = ctx.physics.rigid_body_set.get(rb_handle.0) {
                    let position = body.translation();
                    let ray_origin = point![position.x, position.y, position.z];
                    let ray_dir = vector![0.0, -1.0, 0.0];
                    let max_dist = 1.5;

                    ctx.physics
                        .query_pipeline
                        .cast_ray(
                            &ctx.physics.rigid_body_set,
                            &ctx.physics.collider_set,
                            &Ray::new(ray_origin, ray_dir),
                            max_dist,
                            true,
                            QueryFilter::default().exclude_rigid_body(rb_handle.0),
                        )
                        .map(|(_, toi)| (toi, max_dist))
                } else {
                    None
                }
            };

            if let Some((toi, max_dist)) = ground_hit {
                if let Some(body) = ctx.physics.rigid_body_set.get_mut(rb_handle.0) {
                    let stiffness = 400.0;
                    let damping = 20.0;
                    let compression = 1.0 - (toi / max_dist);
                    let up_force = vector![0.0, stiffness * compression, 0.0];
                    body.add_force(up_force, true);

                    let vel = *body.linvel();
                    body.add_force(-vel * damping, true);

                    if !obstacle_in_front {
                        let rot = *body.rotation();
                        let forward_dir = rot.transform_vector(&vector![0.0, 0.0, -1.0]);
                        body.add_force(forward_dir * 40.0, true);
                    } else {
                        // Brake
                        let vel = *body.linvel();
                        body.add_force(-vel * 10.0, true);
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
