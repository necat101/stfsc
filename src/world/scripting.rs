use hecs::{Entity, World};
use crate::world::Transform;
use crate::physics::PhysicsWorld;

/// The core trait for scripts in the STFSC engine.
/// Similar to Unity's MonoBehavior.
pub trait FuckScript: Send + Sync {
    /// Called once when the script is attached or when the scene starts.
    fn on_start(&mut self, _ctx: &mut ScriptContext) {}
    
    /// Called every logic frame.
    fn on_update(&mut self, _ctx: &mut ScriptContext) {}

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
}

/// Context passed to scripts to allow them to interact with the engine.
pub struct ScriptContext<'a> {
    pub entity: Entity,
    pub world: &'a mut World,
    pub physics: &'a mut PhysicsWorld,
    pub dt: f32,
}

impl<'a> ScriptContext<'a> {
    pub fn new(entity: Entity, world: &'a mut World, physics: &'a mut PhysicsWorld, dt: f32) -> Self {
        Self { entity, world, physics, dt }
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
        use crate::world::{CrowdAgent, AgentState};
        // Ensure the entity has a CrowdAgent component to work with
        if ctx.world.get::<&CrowdAgent>(ctx.entity).is_err() {
            let mut seed = (ctx.entity.id() * 12345) as u32;
            let mut rand = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (seed as f32) / (u32::MAX as f32)
            };
            
            let state = if rand() > 0.8 { AgentState::Fleeing } else if rand() > 0.5 { AgentState::Running } else { AgentState::Walking };
            let max_speed = match state {
                AgentState::Fleeing => 8.0,
                AgentState::Running => 5.0,
                _ => 2.0,
            };

            let _ = ctx.world.insert_one(ctx.entity, CrowdAgent {
                velocity: glam::Vec3::ZERO,
                target: glam::Vec3::new((rand() - 0.5) * 50.0, 1.0, (rand() - 0.5) * 50.0),
                state,
                max_speed,
                stuck_timer: 0.0,
                last_pos: glam::Vec3::ZERO,
            });
        }
    }

    fn on_update(&mut self, ctx: &mut ScriptContext) {
        use crate::world::{CrowdAgent, AgentState, Player};
        
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
                let mut seed = (ctx.entity.id() as u32).wrapping_mul(12345) ^ (pos.x * 100.0) as u32;
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
                    agent.target = glam::Vec3::new((rand() - 0.5) * 60.0, pos.y, (rand() - 0.5) * 60.0);
                    agent.velocity = glam::Vec3::new(rand() - 0.5, 0.0, rand() - 0.5).normalize() * 2.0;
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
                    agent.target = glam::Vec3::new((rand() - 0.5) * 60.0, pos.y, (rand() - 0.5) * 60.0);
                    // Also give a tiny nudge to start moving
                    agent.velocity = (agent.target - pos).normalize() * 0.1;
                } else {
                    let desired = to_target.normalize() * max_speed;
                    let steer_force = if agent.state == AgentState::Fleeing { 20.0 } else { 8.0 };
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
            let _ = ctx.world.insert_one(ctx.entity, Vehicle {
                speed: 0.0,
                max_speed: 15.0,
                steering: 0.0,
                accelerating: true,
            });
        }
        
        // Add physics if missing
        if ctx.world.get::<&crate::world::RigidBodyHandle>(ctx.entity).is_err() {
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
                let _ = ctx.world.insert_one(ctx.entity, crate::world::RigidBodyHandle(handle));
            }
        }
    }

    fn on_update(&mut self, ctx: &mut ScriptContext) {
        use crate::world::Vehicle;
        use crate::world::RigidBodyHandle;
        use rapier3d::prelude::*;

        if let Ok(_vehicle) = ctx.world.get::<&Vehicle>(ctx.entity) {
            if let Ok(rb_handle) = ctx.world.get::<&RigidBodyHandle>(ctx.entity) {
                let ray_hit = {
                    if let Some(body) = ctx.physics.rigid_body_set.get(rb_handle.0) {
                        let position = body.translation();
                        let ray_origin = point![position.x, position.y, position.z];
                        let ray_dir = vector![0.0, -1.0, 0.0];
                        let max_dist = 1.5;

                        ctx.physics.query_pipeline.cast_ray(
                            &ctx.physics.rigid_body_set,
                            &ctx.physics.collider_set,
                            &Ray::new(ray_origin, ray_dir),
                            max_dist,
                            true,
                            QueryFilter::default().exclude_rigid_body(rb_handle.0),
                        ).map(|(_, toi)| (toi, max_dist))
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
        use crate::world::{CrowdAgent, AgentState, Player};

        // Initialize as CrowdAgent if missing
        if ctx.world.get::<&CrowdAgent>(ctx.entity).is_err() {
            let _ = ctx.world.insert_one(ctx.entity, CrowdAgent {
                velocity: glam::Vec3::ZERO,
                target: glam::Vec3::ZERO,
                state: AgentState::Chasing,
                max_speed: 10.0,
                stuck_timer: 0.0,
                last_pos: glam::Vec3::ZERO,
            });
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
        use crate::world::Vehicle;
        use crate::world::RigidBodyHandle;
        use rapier3d::prelude::*;

        // Ensure entity has Vehicle component
        if ctx.world.get::<&Vehicle>(ctx.entity).is_err() {
            let _ = ctx.world.insert_one(ctx.entity, Vehicle {
                speed: 0.0,
                max_speed: 15.0,
                steering: 0.0,
                accelerating: true,
            });
        }

        if let Ok(rb_handle) = ctx.world.get::<&RigidBodyHandle>(ctx.entity) {
            let mut obstacle_in_front = false;
            
            // 1. Raycast in front for other cars/players/obstacles
            let ray_hit = {
                if let Some(body) = ctx.physics.rigid_body_set.get(rb_handle.0) {
                    let position = body.translation();
                    let rot = *body.rotation();
                    let forward = rot.transform_vector(&vector![0.0, 0.0, -1.0]);
                    
                    let ray_origin = point![position.x + forward.x * 2.1, position.y, position.z + forward.z * 2.1];
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

                    ctx.physics.query_pipeline.cast_ray(
                        &ctx.physics.rigid_body_set,
                        &ctx.physics.collider_set,
                        &Ray::new(ray_origin, ray_dir),
                        max_dist,
                        true,
                        QueryFilter::default().exclude_rigid_body(rb_handle.0),
                    ).map(|(_, toi)| (toi, max_dist))
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
        self.builders.insert(name.to_string(), Box::new(move || Box::new(builder())));
    }

    pub fn create(&self, name: &str) -> Option<Box<dyn FuckScript>> {
        self.builders.get(name).map(|b| b())
    }
}
