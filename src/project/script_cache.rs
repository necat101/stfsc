use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::project::scene::{
    Scene, ScriptCompileMode, MAX_SCENE_SCRIPTS_PER_SCENE, MAX_SCRIPTS_PER_ENTITY,
};
use crate::project::script_compiler::{runtime_cache_name, rust_type_name, NativeScriptSource};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScriptCacheIdentity {
    pub source_hash: u64,
    pub cache_key: String,
    pub runtime_name: String,
    pub native_symbol: String,
}

pub fn normalize_scene_key(path: &str) -> String {
    path.replace('\\', "/")
}

pub fn effective_script_source(name: &str, source: &str) -> String {
    if source.trim().is_empty() {
        format!("script {} {{}}\n", name.trim())
    } else {
        source.to_string()
    }
}

pub fn hash_script_source(
    scene: &str,
    entity_id: u32,
    slot: usize,
    name: &str,
    source: &str,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    normalize_scene_key(scene).hash(&mut hasher);
    entity_id.hash(&mut hasher);
    slot.hash(&mut hasher);
    name.hash(&mut hasher);
    source.hash(&mut hasher);
    hasher.finish()
}

pub fn hash_scene_script_source(scene: &str, slot: usize, name: &str, source: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    normalize_scene_key(scene).hash(&mut hasher);
    "scene".hash(&mut hasher);
    slot.hash(&mut hasher);
    name.hash(&mut hasher);
    source.hash(&mut hasher);
    hasher.finish()
}

pub fn script_cache_identity(
    scene: &str,
    entity_id: u32,
    slot: usize,
    name: &str,
    source: &str,
) -> ScriptCacheIdentity {
    let source_hash = hash_script_source(scene, entity_id, slot, name, source);
    let cache_key = format!("{:016x}", source_hash);
    ScriptCacheIdentity {
        source_hash,
        runtime_name: runtime_cache_name(source_hash),
        native_symbol: rust_type_name(&format!("StfscScript_{}", cache_key)),
        cache_key,
    }
}

pub fn scene_script_cache_identity(
    scene: &str,
    slot: usize,
    name: &str,
    source: &str,
) -> ScriptCacheIdentity {
    let source_hash = hash_scene_script_source(scene, slot, name, source);
    let cache_key = format!("{:016x}", source_hash);
    ScriptCacheIdentity {
        source_hash,
        runtime_name: runtime_cache_name(source_hash),
        native_symbol: rust_type_name(&format!("StfscSceneScript_{}", cache_key)),
        cache_key,
    }
}

pub fn collect_native_scripts_for_scene(
    scene_key: &str,
    scene: &mut Scene,
) -> Vec<NativeScriptSource> {
    let scene_key = normalize_scene_key(scene_key);
    let mut native_sources = Vec::new();

    {
        let components = scene.ensure_scene_script_components();
        for (slot, component) in components
            .iter_mut()
            .take(MAX_SCENE_SCRIPTS_PER_SCENE)
            .enumerate()
        {
            if matches!(component.compile_mode, ScriptCompileMode::BuiltinNative) {
                component.cache_key = None;
                continue;
            }

            let name = component.name.trim().to_string();
            if name.is_empty() {
                component.cache_key = None;
                continue;
            }

            let source = effective_script_source(&name, &component.source);
            let identity = scene_script_cache_identity(&scene_key, slot, &name, &source);
            component.cache_key = Some(identity.runtime_name.clone());
            native_sources.push(NativeScriptSource {
                runtime_name: identity.runtime_name,
                struct_name: identity.native_symbol,
                source,
            });
        }
        scene.sync_legacy_scene_script_fields();
    }

    for entity in &mut scene.entities {
        let entity_id = entity.id;
        let components = entity.ensure_script_components();

        for (slot, component) in components
            .iter_mut()
            .take(MAX_SCRIPTS_PER_ENTITY)
            .enumerate()
        {
            if matches!(component.compile_mode, ScriptCompileMode::BuiltinNative) {
                component.cache_key = None;
                continue;
            }

            let name = component.name.trim().to_string();
            if name.is_empty() {
                component.cache_key = None;
                continue;
            }

            let source = effective_script_source(&name, &component.source);
            let identity = script_cache_identity(&scene_key, entity_id, slot, &name, &source);
            component.cache_key = Some(identity.runtime_name.clone());
            native_sources.push(NativeScriptSource {
                runtime_name: identity.runtime_name,
                struct_name: identity.native_symbol,
                source,
            });
        }

        entity.sync_legacy_script_fields();
    }

    native_sources
}

pub fn rewrite_scene_script_cache_keys(scene_key: &str, scene: &mut Scene) {
    let _ = collect_native_scripts_for_scene(scene_key, scene);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::project::scene::{EntityType, ScriptComponent};
    use crate::project::script_compiler::render_native_script_module;

    #[test]
    fn native_cache_compiles_all_custom_scripts_and_keeps_vehicle_driver() {
        let mut scene = Scene::create_test_scene();
        let cube = scene
            .entities
            .iter_mut()
            .find(|entity| entity.id == 2)
            .expect("test scene should contain physics cube");
        cube.script_components = vec![ScriptComponent {
            name: "CylinderMover".to_string(),
            enabled: true,
            source: "script CylinderMover { on_update(ctx) { ctx.log(\"cylinder alive\"); } }"
                .to_string(),
            compile_mode: ScriptCompileMode::CustomNativeCache,
            cache_key: None,
        }];

        let vehicle = scene
            .entities
            .iter_mut()
            .find(|entity| matches!(entity.entity_type, EntityType::Vehicle))
            .expect("test scene should contain vehicle");
        vehicle.script_components = vec![ScriptComponent {
            name: "VehicleExtraLogic".to_string(),
            enabled: true,
            source: "script VehicleExtraLogic { on_update(ctx) { ctx.log(\"vehicle alive\"); } }"
                .to_string(),
            compile_mode: ScriptCompileMode::CustomNativeCache,
            cache_key: None,
        }];

        let native_sources = collect_native_scripts_for_scene("scenes/test.json", &mut scene);
        assert_eq!(native_sources.len(), 2);
        render_native_script_module(&native_sources).expect("native cache should compile");

        let vehicle = scene
            .entities
            .iter()
            .find(|entity| matches!(entity.entity_type, EntityType::Vehicle))
            .expect("test scene should contain vehicle");
        let runtime_names = vehicle.runtime_script_names();
        assert_eq!(runtime_names.first().map(String::as_str), Some("Vehicle"));
        assert!(
            runtime_names
                .iter()
                .any(|name| name.starts_with("__stfsc_script_")),
            "vehicle should keep its compiled custom script too"
        );
    }

    #[test]
    fn native_cache_compiles_disabled_custom_scripts_too() {
        let mut scene = Scene::create_test_scene();
        {
            let cube = scene
                .entities
                .iter_mut()
                .find(|entity| entity.id == 2)
                .expect("test scene should contain physics cube");
            cube.script_components = vec![ScriptComponent {
                name: "DisabledButCompiled".to_string(),
                enabled: false,
                source: "script DisabledButCompiled { on_update(ctx) { ctx.log(\"compiled\"); } }"
                    .to_string(),
                compile_mode: ScriptCompileMode::CustomNativeCache,
                cache_key: None,
            }];
        }

        let native_sources = collect_native_scripts_for_scene("scenes/test.json", &mut scene);
        assert_eq!(native_sources.len(), 1);
        let cube = scene
            .entities
            .iter()
            .find(|entity| entity.id == 2)
            .expect("test scene should contain physics cube");
        assert!(cube.runtime_script_names().is_empty());
        render_native_script_module(&native_sources).expect("disabled native cache should compile");
    }

    #[test]
    fn native_cache_compiles_scene_level_custom_scripts() {
        let mut scene = Scene::new("Generated Scene");
        scene.scene_script_components = vec![
            ScriptComponent {
                name: "SceneGenerator".to_string(),
                enabled: true,
                source: r#"
script SceneGenerator {
    on_start(ctx) {
        ctx.set_procedural_generation(false);
        let hero = ctx.spawn_primitive("cube", vec3(0.0, 1.0, 0.0), quat_identity(), vec3(2.0, 2.0, 2.0), vec3(0.9, 0.4, 0.2), true, true);
        ctx.set_scripts(hero, "TestBounce", "CollisionLogger", "", "");
        ctx.spawn_light_with_cones("spot", vec3(0.0, 5.0, -2.0), vec3(0.0, -1.0, 0.25), vec3(1.0, 0.9, 0.8), 5.0, 24.0, 0.25, 0.85);
        ctx.spawn_scatter_range("sphere", 16.0, vec3(0.0, 0.5, 0.0), vec3(20.0, 0.0, 20.0), vec3(0.35, 0.35, 0.35), vec3(1.2, 1.2, 1.2), vec3(0.2, 0.45, 0.3), vec3(0.6, 0.85, 0.45), 44.0, true, true);
        ctx.spawn_grid("cube", 4.0, 2.0, vec3(0.0, 0.25, 10.0), vec3(2.0, 0.0, 2.0), vec3(1.0, 0.5, 1.0), vec3(0.25, 0.35, 0.8), true, true);
    }
}
"#
                .to_string(),
                compile_mode: ScriptCompileMode::CustomNativeCache,
                cache_key: None,
            },
            ScriptComponent {
                name: "DisabledSceneValidator".to_string(),
                enabled: false,
                source:
                    "script DisabledSceneValidator { on_start(ctx) { ctx.log(\"compiled\"); } }"
                        .to_string(),
                compile_mode: ScriptCompileMode::CustomNativeCache,
                cache_key: None,
            },
        ];

        let native_sources = collect_native_scripts_for_scene("scenes/generated.json", &mut scene);
        assert_eq!(native_sources.len(), 2);
        assert_eq!(scene.scene_script_names().len(), 1);
        assert!(
            scene.scene_script_names()[0].starts_with("__stfsc_script_"),
            "enabled scene script should run by generated cache name"
        );
        render_native_script_module(&native_sources)
            .expect("scene-level native cache should compile");
    }
}
