# STFSC Engine

**Steal This Fucking Source Code** — *Because gaming deserves better.*

<p align="center">
  <strong>v0.3.5 Alpha — "Turbo Parallel" Update</strong>
</p>

---

## Why STFSC?

The gaming industry has a problem. Bloated engines. Lazy optimization. "Good enough" performance. The result? Games that stutter, overheat devices, and frustrate players into piracy. STFSC exists because **optimization and attention to detail shouldn't be optional**—they should be the foundation.

This engine is built from the ground up for one audacious goal: **open-world gaming on standalone VR hardware**, pushing the Quest 3's 2.4 TFLOPS to its absolute limits.

> *"They said it couldn't be done. We're doing it anyway."*

---

## 🎮 Target Project: 556

STFSC is the foundation for **556**—an upcoming open-world crime simulator aiming to be the "GTA-V of standalone VR." Featuring:

- Immersive open-world exploration
- **556 Downtown** — Online multiplayer mode
- Xbox 360-level graphical fidelity on mobile VR hardware
- Authentic crime simulator experience

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
- **Lifecycle hooks**: `on_start`, `on_update`, `on_enable`, `on_disable`
- **Collision callbacks**: `on_collision_start`, `on_collision_end`, `on_trigger_start`
- **Built-in scripts**: CrowdAgent, PoliceAgent, TrafficAI, VehicleAI, WeaponNPC

### Engine Architecture
- **Entity Component System (ECS)** via `hecs`
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

### Deploy to Quest
```bash
adb install -r target/release/apk/stfsc_engine.apk
adb shell am start -n com.stfsc.engine/android.app.NativeActivity
```

---

## 🖥️ Linux Desktop Build

For development and testing without a Quest headset:

```bash
cargo run
```

That's it! The Linux client provides a windowed preview with WASD + mouse controls for rapid iteration.

---

## 🎛️ Editor Usage

1. Start the engine on Quest (or Linux desktop)
2. Run the editor on your development machine:
   ```bash
   cargo run --bin editor
   ```
3. Connect via ADB port forwarding (Quest) or localhost (Linux):
   ```bash
   adb forward tcp:8080 tcp:8080
   ```
4. Deploy scenes in real-time!

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

### v0.4 (Combat & Persistence)
- [ ] Weapon & Ballistics system
- [ ] Global Save/Load state
- [ ] Navmesh pathfinding for NPCs
- [ ] Advanced Traffic steering behaviors

### v0.5 (Connected World)
- [ ] Multiplayer netcode foundation
- [ ] GPU-side occlusion culling (Hi-Z)
- [ ] Compressed texture streaming (KTX2/ASTC)

### v1.0 (556 Launch)
- [ ] Complete open-world streaming
- [ ] Full multiplayer gameplay
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
