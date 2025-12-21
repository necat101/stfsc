# STFSC Engine

**Steal This Fucking Source Code** — *Because gaming deserves better.*

<p align="center">
  <strong>v0.3 Alpha</strong>
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

## ✨ Features (v0.3 Alpha)

### Rendering
- **Vulkan-based renderer** with multiview stereo rendering
- **PBR lighting pipeline** with shadow mapping (PCF soft shadows)
- **Dynamic shadow resolution** — Automatic sizing based on scene scale
- **Application Space Warp (AppSW)** support for 36fps→72Hz upscaling
- **Motion vector generation** for temporal reprojection
- **Instanced rendering** with batched draw calls
- **Dynamic Lights** (Point, Spot, Directional) with PCF Shadows
- **Reversed-Z depth buffer** for maximum precision on large worlds
- **Normal Offset Shadow Mapping** for reduced shadow acne

### Physics
- **Rapier3D integration** for rigid body dynamics
- **Collision detection** with event callbacks
- **Static & dynamic bodies** with configurable shapes
- **Physics-synced transforms** for realistic movement

### Audio
- **3D Spatial Audio** with distance attenuation
- **Multiple attenuation models** (Inverse, Linear, Exponential)
- **Streaming audio support** for ambient sounds
- **Editor integration** for sound placement and preview

### Scripting — "Fuck Script" 🔥
- **Native Rust scripting** via `FuckScript` trait
- **Lifecycle hooks**: `on_start`, `on_update`, `on_enable`, `on_disable`
- **Collision callbacks**: `on_collision_start`, `on_collision_end`, `on_trigger_start`
- **Built-in scripts**: CrowdAgent, PoliceAgent, TrafficAI, VehicleAI, WeaponNPC
- **Script registry** for editor attachment

### Engine Architecture
- **Entity Component System (ECS)** via `hecs`
- **LOD system** with distance-based mesh switching
- **Crowd simulation** with agent AI and collision avoidance
- **World streaming** architecture for open-world scenes
- **Thread-safe game state** with lock-failure recovery
- **Async Resource Loading** for background mesh uploads (No ANRs)

### Editor Integration
- **Live scene deployment** over TCP/IP
- **Real-time entity manipulation** (spawn, transform, delete)
- **Scene clearing** with proper entity lifecycle management
- **Primitive mesh library** (cube, sphere, cylinder, cone, capsule, plane)
- **Custom mesh & texture streaming** from editor
- **Ground plane textures** with shadow frustum optimization
- **Script attachment** via inspector panel
- **3D Audio source placement** with live preview

### VR/XR
- **OpenXR integration** for Meta Quest devices
- **Stage space tracking** with player start positioning
- **Controller input** support
- **Passthrough ready** (OPAQUE/ALPHA_BLEND modes)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Rust 🦀 |
| Graphics API | Vulkan 1.1+ |
| XR Runtime | OpenXR |
| Physics | Rapier3D |
| ECS | hecs |
| Math | glam |
| Serialization | bincode / serde |
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

### v0.2 (Alpha Test) ✅
- [x] Dynamic lighting system
- [x] Parallel Resource Loading

### v0.2.5 ✅
- [x] 3D Audio system integration
- [ ] Occlusion culling
- [ ] Compressed texture streaming (KTX2/ASTC)

### v0.3 (Current) ✅
- [x] Physics integration (Rapier3D)
- [x] Vehicle systems
- [x] NPC AI framework (FuckScript)
- [x] Collision event system
- [x] Scene clearing with lifecycle management
- [ ] Navmesh pathfinding

### v0.4 (Planned)
- [ ] Weapon systems
- [ ] Player inventory
- [ ] Save/Load game state
- [ ] Advanced NPC behaviors

### v1.0 (556 Launch)
- [ ] Complete open-world streaming
- [ ] Multiplayer netcode
- [ ] Full game systems

---

## 🤝 Philosophy

**Steal this fucking source code.** Seriously.

If you're building a game and struggling with VR performance, take what you need. Learn from it. Improve on it. The goal isn't to hoard knowledge—it's to prove that **optimized, ambitious games are possible on mobile VR**.

Piracy exists because players don't feel games are worth the price. The solution isn't more DRM—it's making games so good, so polished, so clearly crafted with care that players *want* to support them.

STFSC is open because optimization knowledge shouldn't be gatekept.

---

## ⚠️ Alpha Disclaimer

This is **v0.3 Alpha** software. Expect:
- Incomplete features
- API changes
- The occasional crash
- Dragons 🐉

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
