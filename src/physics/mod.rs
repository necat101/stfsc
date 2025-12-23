use rapier3d::prelude::*;

pub struct PhysicsWorld {
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    pub gravity: Vector<f32>,
    pub integration_parameters: IntegrationParameters,
    pub physics_pipeline: PhysicsPipeline,
    pub island_manager: IslandManager,
    pub broad_phase: BroadPhase,
    pub narrow_phase: NarrowPhase,
    pub impulse_joint_set: ImpulseJointSet,
    pub multibody_joint_set: MultibodyJointSet,
    pub ccd_solver: CCDSolver,
    pub query_pipeline: QueryPipeline,
    pub event_collector: EventCollector,
}

pub struct EventCollector {
    pub collision_events: std::sync::Arc<std::sync::Mutex<Vec<CollisionEvent>>>,
}

impl EventHandler for EventCollector {
    fn handle_collision_event(
        &self,
        _bodies: &RigidBodySet,
        _colliders: &ColliderSet,
        event: CollisionEvent,
        _contact_pair: Option<&ContactPair>,
    ) {
        if let Ok(mut events) = self.collision_events.lock() {
            events.push(event);
        }
    }

    fn handle_contact_force_event(
        &self,
        _dt: f32,
        _bodies: &RigidBodySet,
        _colliders: &ColliderSet,
        _contact_pair: &ContactPair,
        _total_force_magnitude: f32,
    ) {
    }
}

impl PhysicsWorld {
    pub fn new() -> Self {
        Self {
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            gravity: vector![0.0, -9.81, 0.0],
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            event_collector: EventCollector {
                collision_events: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
            },
        }
    }

    pub fn step(&mut self) {
        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            None,
            &(),
            &self.event_collector,
        );
        self.query_pipeline
            .update(&self.rigid_body_set, &self.collider_set);
    }

    pub fn add_box_rigid_body(
        &mut self,
        entity_id: u128,
        translation: [f32; 3],
        half_extents: [f32; 3],
        dynamic: bool,
    ) -> RigidBodyHandle {
        let rigid_body = if dynamic {
            RigidBodyBuilder::dynamic()
                .translation(vector![translation[0], translation[1], translation[2]])
                .user_data(entity_id)
                .ccd_enabled(true)
                .build()
        } else {
            RigidBodyBuilder::fixed()
                .translation(vector![translation[0], translation[1], translation[2]])
                .user_data(entity_id)
                .build()
        };

        let collider =
            ColliderBuilder::cuboid(half_extents[0], half_extents[1], half_extents[2]).build();
        let body_handle = self.rigid_body_set.insert(rigid_body);
        self.collider_set
            .insert_with_parent(collider, body_handle, &mut self.rigid_body_set);

        body_handle
    }

    pub fn add_sphere_rigid_body(
        &mut self,
        entity_id: u128,
        translation: [f32; 3],
        radius: f32,
        dynamic: bool,
    ) -> RigidBodyHandle {
        let rigid_body = if dynamic {
            RigidBodyBuilder::dynamic()
                .translation(vector![translation[0], translation[1], translation[2]])
                .user_data(entity_id)
                .ccd_enabled(true)
                .build()
        } else {
            RigidBodyBuilder::fixed()
                .translation(vector![translation[0], translation[1], translation[2]])
                .user_data(entity_id)
                .build()
        };

        let collider = ColliderBuilder::ball(radius).build();
        let body_handle = self.rigid_body_set.insert(rigid_body);
        self.collider_set
            .insert_with_parent(collider, body_handle, &mut self.rigid_body_set);
        body_handle
    }

    pub fn add_capsule_rigid_body(
        &mut self,
        entity_id: u128,
        translation: [f32; 3],
        half_height: f32,
        radius: f32,
        dynamic: bool,
    ) -> RigidBodyHandle {
        let rigid_body = if dynamic {
            RigidBodyBuilder::dynamic()
                .translation(vector![translation[0], translation[1], translation[2]])
                .user_data(entity_id)
                .ccd_enabled(true)
                .build()
        } else {
            RigidBodyBuilder::fixed()
                .translation(vector![translation[0], translation[1], translation[2]])
                .user_data(entity_id)
                .build()
        };

        let collider = ColliderBuilder::capsule_y(half_height, radius).build();
        let body_handle = self.rigid_body_set.insert(rigid_body);
        self.collider_set
            .insert_with_parent(collider, body_handle, &mut self.rigid_body_set);
        body_handle
    }

    pub fn add_cylinder_rigid_body(
        &mut self,
        entity_id: u128,
        translation: [f32; 3],
        half_height: f32,
        radius: f32,
        dynamic: bool,
    ) -> RigidBodyHandle {
        let rigid_body = if dynamic {
            RigidBodyBuilder::dynamic()
                .translation(vector![translation[0], translation[1], translation[2]])
                .user_data(entity_id)
                .ccd_enabled(true)
                .build()
        } else {
            RigidBodyBuilder::fixed()
                .translation(vector![translation[0], translation[1], translation[2]])
                .user_data(entity_id)
                .build()
        };

        let collider = ColliderBuilder::cylinder(half_height, radius).build();
        let body_handle = self.rigid_body_set.insert(rigid_body);
        self.collider_set
            .insert_with_parent(collider, body_handle, &mut self.rigid_body_set);
        body_handle
    }

    pub fn remove_rigid_body(&mut self, handle: RigidBodyHandle) {
        self.rigid_body_set.remove(
            handle,
            &mut self.island_manager,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            true,
        );
    }
}
