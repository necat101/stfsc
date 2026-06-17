# STFSC Engine

**Steal This Fucking Source Code** — *Because gaming deserves better.*

<p align="center">
  <strong>v0.3.5 Alpha —</strong>
</p>

---

## Why STFSC?

The gaming industry has a problem. Bloated engines. Lazy optimization. "Good enough" performance. The result? Games that stutter, overheat devices, and frustrate players into piracy. STFSC exists because **optimization and attention to detail shouldn't be optional**—they should be the foundation.

This engine is built from the ground up for one audacious goal: **open-world gaming on standalone VR hardware**, pushing the Quest 3's 2.4 TFLOPS to its absolute limits.

> *"They said it couldn't be done. We're doing it anyway."*

---

## Target Projects

STFSC is being shaped for ambitious open-world games that need smooth simulation under pressure. The near-term engine target is **Twighlight**, a Rust/Minecraft-like survival sandbox with seeded procedural worlds, object-based building, day/night danger, survival play, and god-mode creation tools. Its project root lives at `projects/twighlight/`, with project-local `assets/`, `scenes/`, `scripts/`, and `ui/` folders.

### Project Layout
- Engine source and shared tooling stay at the repository root.
- Game projects live under `projects/<project-name>/`, like Unity-style project folders.
- Each project owns its own `project.json`, scene JSON, UI scene JSON, FuckScript source files, generated assets, and third-party asset folders.

The engine still keeps its 556-class goals: dense city streaming, vehicles, crowds, multiplayer-ready architecture, and standalone VR performance. Sandbox support is an added optimization profile, not a narrowing of the engine.

---

## ✨ Features (v0.3.5 Alpha)

### 🏎️ Turbo-Parallelization (Rayon-powered)
- **Parallel Render Prep**: Frustum culling, matrix math, and instancing batching now distributed across all CPU cores.
- **Physics-to-ECS Sync**: Multi-threaded updates from Rapier3D back to world transforms.
- **Parallel AI Logic**: High-density NPC (CrowdAgent) and Vehicle AI processed in parallel.
- **Thread-safe Streaming**: Procedural city chunk generation is multi-threaded for stutter-free travel.

### 🛠️ Professional Editor Workflow
- **Full Undo/Redo System**: All major operations (spawn, delete, move, properties) are tracked and reversible.
- **Persistent Asset Pipeline**: Automatic texture ingestion into `assets/textures/` for project portability.
- **Real-time Material Sync**: Instant inspection/tweaking of PBR properties and textures on Quest 3 without re-spawning.
- **Missing Asset Diagnostics**: Built-in warnings for missing project files with easy re-linking guidance.
- **Parallel Viewport Prep**: Smooth editor UI performance via multi-threaded viewport projection math.

### Sandbox/Open-World Foundations
- **Seeded Chunk Planning**: deterministic sandbox chunk plans for trees, resources, mobs, and build budgets.
- **Survival/God Mode Rules**: project metadata can distinguish resource-constrained survival from free-building creative play.
- **Day/Night Runtime Clock**: sandbox-aware clock hooks for hostile night spawning, lighting, and scripts.
- **Parallel Planning Windows**: chunk plan generation uses Rayon so wide sandbox worlds can stream without blocking the frame.

### Rendering
- **Vulkan-based renderer** with multiview stereo rendering
- **PBR lighting pipeline** with shadow mapping (PCF soft shadows)
- **Reversed-Z depth buffer** for maximum precision on large world scales (556 Downtown scale)
- **Application Space Warp (AppSW)** support for 36fps→72Hz upscaling
- **Dynamic Lights** (Point, Spot, Directional) with PCF Shadows and parallel UBO updates
- **Instanced rendering** with batched draw calls

### Physics
- **Rapier3D integration** for high-performance rigid body dynamics
- **Collision detection** with scriptable event callbacks
- **Static & dynamic bodies** with configurable shapes

### Audio
- **3D Spatial Audio** with distance attenuation
- **Multiple attenuation models** (Inverse, Linear, Exponential)
- **Streaming audio support** for ambient sounds

### Scripting — "Fuck Script" 🔥
- **Native Rust scripting** via `FuckScript` trait
- **Editable source export compiler**: cache-mode FuckScript sources are parsed and generated as native Rust during export; no runtime interpreter is used in the game loop
- **Lifecycle hooks**: `on_awake`, `on_start`, `on_update`, `on_enable`, `on_disable`, `on_destroy`
- **Fixed/late update hooks** for physics-step logic and post-update pose following
- **Collision callbacks**: `on_collision_start`, `on_collision_stay`, `on_collision_end`, `on_trigger_start`, `on_trigger_stay`, `on_trigger_end`
- **XR callbacks and helpers**: HMD/controller poses, abstract actions, edge events, and haptic requests through `ScriptContext`
- **Built-in scripts**: CrowdAgent, PoliceAgent, TrafficAI, EnemyTracker, VehicleAI, WeaponNPC, HeadAnchor, LeftHandAnchor, RightHandAnchor, TriggerHaptics

### Engine Architecture
- **Entity Component System (ECS)** via `hecs`
- **Runtime Task Graph** for dependency-aware parallel frame work across simulation, physics, render prep, streaming, networking, and maintenance systems
- **Frame Budget Pressure Controls** for deferring non-critical jobs when a platform is close to missing frame time
- **Platform-tuned async runtime** so desktop and standalone VR builds size networking/blocking worker pools from the same concurrency profile
- **LOD system** with distance-based mesh switching
- **Async Resource Loading** for background mesh uploads (No ANRs)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Rust 🦀 |
| Graphics API | Vulkan 1.1+ |
| XR Runtime | OpenXR |
| Physics | Rapier3D |
| Parallelism | Rayon |
| ECS | hecs |
| Target Platform | Meta Quest 3 / Quest 3S |

---

## 📦 Building

### Prerequisites
- Rust toolchain with `aarch64-linux-android` target
- Android NDK
- `cargo-apk` for APK packaging

### Build Commands
```bash
# Debug build
cargo build --target aarch64-linux-android

# Release build
cargo build --target aarch64-linux-android --release

# Package APK
cargo apk build --release
```

### Install Standalone Quest APK
```bash
adb install -r target/release/apk/stfsc_engine.apk
adb shell am start -n com.stfsc.engine/android.app.NativeActivity
```

Quest builds bundle the project scene/assets into the APK. A live editor connection is only needed when you want a real-time debug push to a connected headset.

---

## 🖥️ Linux Desktop Build

For development and testing without a Quest headset:

```bash
cargo run
```

That's it! The Linux client provides a windowed preview with WASD + mouse controls for rapid iteration.

---

## 🎛️ Editor Usage

1. Run the editor on your development machine:
   ```bash
   cargo run --bin editor
   ```
2. Use the embedded Scene viewport as the default local player.
3. For Quest real-time debugging, connect the headset over ADB, refresh the Player / Push panel, then connect Quest Push.
4. For desktop runtime debugging, start the runtime separately and use Connect Debug Push.

---

## 📊 Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Frame Rate | 72 Hz (36 fps + AppSW) | ✅ Achieved |
| GPU Utilization | <80% | ✅ ~18% idle scene |
| Memory Budget | <500 MB | ✅ ~115 MB |
| Draw Calls | Batched/Instanced | ✅ Implemented |
| Stale Frames | 0 | ✅ Achieved |

---

## 🗺️ Roadmap

### v0.2 - v0.3 ✅
- [x] Dynamic lighting & 3D Audio
- [x] Parallel Resource Loading
- [x] Physics integration (Rapier3D)
- [x] NPC AI framework (FuckScript)

### v0.3.5 (Turbo Update) ✅
- [x] **Full Parallel Core** (Rendering, Physics Sync, AI)
- [x] **Undo/Redo System** for Editor
- [x] **Robust Asset Persistence** (Local textures, Search fallbacks)
- [x] Real-time Material live-sync

### v0.4 (Sandbox & Persistence)
- [ ] Persistent sandbox save/load state
- [ ] Object-based building placement and stability hooks
- [ ] Resource inventory and crafting interfaces
- [ ] Mob spawn scheduler and night threat director
- [ ] Navmesh/pathfinding support for mobs and NPCs

### v0.5 (Connected World)
- [ ] Multiplayer netcode foundation
- [ ] GPU-side occlusion culling (Hi-Z)
- [ ] Compressed texture streaming (KTX2/ASTC)

### v1.0 (Open-World Launch)
- [ ] Complete open-world streaming for sandbox and city profiles
- [ ] Full multiplayer gameplay foundation
- [ ] Performance parity with consoles

---

## 🤝 Philosophy

**Steal this fucking source code.** Seriously.

If you're building a game and struggling with VR performance, take what you need. Learn from it. Improve on it. The goal isn't to hoard knowledge—it's to prove that **optimized, ambitious games are possible on mobile VR**.

 Piracy exists because players don't feel games are worth the price. The solution isn't more DRM—it's making games so good, so polished, so clearly crafted with care that players *want* to support them.

---

## 📄 License

This project is provided as-is for educational and development purposes. See individual file headers for specific licensing where applicable.

---

<p align="center">
  <em>Built with obsessive optimization for standalone VR gaming.</em>
</p>

<p align="center">
  <strong>STFSC Engine</strong> — <em>Steal This Fucking Source Code</em>
</p>
