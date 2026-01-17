#!/bin/bash
# STFSC Engine - Quest/XR APK Build Script
# Builds a release APK for Meta Quest VR headsets

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${PROJECT_DIR}/bin"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           STFSC Engine - Quest APK Build                      ║"
echo "║                    556 Downtown                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# Environment Setup
# ============================================================================

# Default SDK/NDK paths (adjust if needed)
export ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-$HOME/android-sdk}"
export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$ANDROID_SDK_ROOT/ndk/26.1.10909125}"
export ANDROID_NDK_ROOT="${ANDROID_NDK_HOME}"

echo "📱 Android SDK: ${ANDROID_SDK_ROOT}"
echo "📱 Android NDK: ${ANDROID_NDK_HOME}"
echo ""

# Validate environment
if [ ! -d "${ANDROID_SDK_ROOT}" ]; then
    echo "❌ Error: Android SDK not found at ${ANDROID_SDK_ROOT}"
    echo ""
    echo "To install, run: ./setup_android_sdk.sh"
    echo "Or set ANDROID_SDK_ROOT to your SDK location."
    exit 1
fi

if [ ! -d "${ANDROID_NDK_HOME}" ]; then
    echo "❌ Error: Android NDK not found at ${ANDROID_NDK_HOME}"
    echo ""
    echo "Install NDK 26.1.10909125 via Android Studio or:"
    echo "  sdkmanager 'ndk;26.1.10909125'"
    exit 1
fi

# Check for cargo-apk
if ! command -v cargo-apk &> /dev/null; then
    echo "📦 Installing cargo-apk..."
    cargo install cargo-apk
fi

# Ensure aarch64-linux-android target
if ! rustup target list --installed | grep -q "aarch64-linux-android"; then
    echo "📦 Adding aarch64-linux-android target..."
    rustup target add aarch64-linux-android
fi

# ============================================================================
# Set up NDK Toolchain for cc-rs crates (oboe-sys, etc.)
# ============================================================================
NDK_TOOLCHAIN="${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64"
export PATH="${NDK_TOOLCHAIN}/bin:${PATH}"

# Set CC and CXX for the target (cc-rs will use these)
export CC_aarch64_linux_android="${NDK_TOOLCHAIN}/bin/aarch64-linux-android23-clang"
export CXX_aarch64_linux_android="${NDK_TOOLCHAIN}/bin/aarch64-linux-android23-clang++"
export AR_aarch64_linux_android="${NDK_TOOLCHAIN}/bin/llvm-ar"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="${NDK_TOOLCHAIN}/bin/aarch64-linux-android23-clang"

echo "🔧 NDK Toolchain: ${NDK_TOOLCHAIN}"
echo "🔧 CC: ${CC_aarch64_linux_android}"

# Create bin directory
mkdir -p "${BIN_DIR}"

# ============================================================================
# Build APK
# ============================================================================

echo ""
echo "🔨 Building Quest APK (Release)..."
echo "   This may take several minutes on first build..."
echo ""

# Pre-compile shaders (downloads glslang if needed)
echo "🎨 Compiling shaders..."
python3 compile_shaders.py
if [ $? -ne 0 ]; then
    echo "❌ Shader compilation failed!"
    exit 1
fi

# Run cargo-apk build (only build the library/APK, skip desktop binaries)
cargo apk build --release --lib 2>&1

if [ $? -ne 0 ]; then
    echo "❌ APK build failed!"
    exit 1
fi

# Find the APK
APK_PATH="${PROJECT_DIR}/target/release/apk/stfsc_engine.apk"
if [ ! -f "${APK_PATH}" ]; then
    # Try alternate location
    APK_PATH=$(find "${PROJECT_DIR}/target" -name "*.apk" -path "*/release/*" 2>/dev/null | head -1)
fi

if [ -f "${APK_PATH}" ]; then
    # Copy to bin directory
    cp "${APK_PATH}" "${BIN_DIR}/stfsc_engine.apk"
    
    SIZE=$(du -h "${BIN_DIR}/stfsc_engine.apk" | cut -f1)
    
    echo ""
    echo "✅ APK build successful!"
    echo ""
    echo "📁 Output: ${BIN_DIR}/stfsc_engine.apk"
    echo "📏 Size: ${SIZE}"
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "                    INSTALLATION OPTIONS"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "Option 1: USB Cable"
    echo "  adb install ${BIN_DIR}/stfsc_engine.apk"
    echo ""
    echo "Option 2: Wi-Fi ADB (no cable needed after initial setup)"
    echo "  1. Enable Developer Mode on Quest"
    echo "  2. Settings > Developer > Wireless debugging > Pair"
    echo "  3. On PC: adb pair <quest_ip>:<pair_port> <pairing_code>"
    echo "  4. Then:  adb connect <quest_ip>:5555"
    echo "  5. Finally: adb install ${BIN_DIR}/stfsc_engine.apk"
    echo ""
    echo "Option 3: SideQuest/MQDH"
    echo "  Drag and drop the APK into SideQuest or Meta Quest Developer Hub"
    echo ""
else
    echo "❌ APK not found after build!"
    echo "   Expected at: ${APK_PATH}"
    exit 1
fi
