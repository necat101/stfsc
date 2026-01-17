#!/bin/bash
# STFSC Engine - Engine Build Script
# Builds the test client (engine runtime) for Linux or Quest

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${PROJECT_DIR}/bin"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           STFSC Engine - Test Client Build                    ║"
echo "║                    556 Downtown                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Show usage
show_usage() {
    echo "Usage: $0 <platform> [build_type]"
    echo ""
    echo "Platforms:"
    echo "  linux   - Build Linux desktop binary"
    echo "  quest   - Build Quest VR APK"
    echo ""
    echo "Build types (optional):"
    echo "  debug   - Debug build with symbols"
    echo "  release - Optimized release build (default)"
    echo ""
    echo "Examples:"
    echo "  $0 linux          # Linux release build"
    echo "  $0 quest          # Quest release APK"
    echo "  $0 linux debug    # Linux debug build"
    exit 1
}

if [ $# -lt 1 ]; then
    show_usage
fi

PLATFORM="${1,,}"  # Lowercase
BUILD_TYPE="${2:-release}"

case "$BUILD_TYPE" in
    debug|Debug|DEBUG)
        BUILD_TYPE="debug"
        CARGO_ARGS=""
        TARGET_DIR="debug"
        ;;
    release|Release|RELEASE|*)
        BUILD_TYPE="release"
        CARGO_ARGS="--release"
        TARGET_DIR="release"
        ;;
esac

# Create bin directory
mkdir -p "${BIN_DIR}"

case "$PLATFORM" in
    linux)
        echo "🐧 Platform: Linux Desktop"
        echo "📦 Build type: ${BUILD_TYPE}"
        echo ""

        # Check dependencies
        if ! command -v cargo &> /dev/null; then
            echo "❌ Error: Cargo not found. Please install Rust."
            exit 1
        fi

        missing_deps=()
        pkg-config --exists vulkan 2>/dev/null || missing_deps+=("vulkan")
        pkg-config --exists alsa 2>/dev/null || missing_deps+=("alsa")
        pkg-config --exists x11 2>/dev/null || missing_deps+=("x11")

        if [ ${#missing_deps[@]} -gt 0 ]; then
            echo "⚠️  Missing system libraries: ${missing_deps[*]}"
            echo "   Run ./run.sh first to auto-install dependencies."
            exit 1
        fi

        echo "🔨 Building Linux engine..."
        cargo build ${CARGO_ARGS} --bin stfsc_engine 2>&1

        if [ $? -ne 0 ]; then
            echo "❌ Build failed!"
            exit 1
        fi

        ENGINE_BIN="${PROJECT_DIR}/target/${TARGET_DIR}/stfsc_engine"
        if [ -f "${ENGINE_BIN}" ]; then
            cp "${ENGINE_BIN}" "${BIN_DIR}/stfsc_engine"
            chmod +x "${BIN_DIR}/stfsc_engine"
            
            SIZE=$(du -h "${BIN_DIR}/stfsc_engine" | cut -f1)
            
            echo ""
            echo "✅ Linux build successful!"
            echo ""
            echo "📁 Output: ${BIN_DIR}/stfsc_engine"
            echo "📏 Size: ${SIZE}"
            echo ""
            echo "To run:"
            echo "  ./bin/stfsc_engine"
        else
            echo "❌ Engine binary not found!"
            exit 1
        fi
        ;;

    quest)
        echo "🥽 Platform: Quest VR"
        echo "📦 Build type: ${BUILD_TYPE}"
        echo ""

        # Environment setup
        export ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-$HOME/android-sdk}"
        export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$ANDROID_SDK_ROOT/ndk/26.1.10909125}"
        export ANDROID_NDK_ROOT="${ANDROID_NDK_HOME}"

        echo "📱 Android SDK: ${ANDROID_SDK_ROOT}"
        echo "📱 Android NDK: ${ANDROID_NDK_HOME}"
        echo ""

        if [ ! -d "${ANDROID_SDK_ROOT}" ]; then
            echo "❌ Error: Android SDK not found at ${ANDROID_SDK_ROOT}"
            echo "   Run ./setup_android_sdk.sh or set ANDROID_SDK_ROOT."
            exit 1
        fi

        if [ ! -d "${ANDROID_NDK_HOME}" ]; then
            echo "❌ Error: Android NDK not found at ${ANDROID_NDK_HOME}"
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

        echo "🔨 Building Quest APK..."
        cargo apk build ${CARGO_ARGS} 2>&1

        if [ $? -ne 0 ]; then
            echo "❌ APK build failed!"
            exit 1
        fi

        # Find and copy APK
        APK_PATH="${PROJECT_DIR}/target/${TARGET_DIR}/apk/stfsc_engine.apk"
        if [ ! -f "${APK_PATH}" ]; then
            APK_PATH=$(find "${PROJECT_DIR}/target" -name "*.apk" -path "*/${TARGET_DIR}/*" 2>/dev/null | head -1)
        fi

        if [ -f "${APK_PATH}" ]; then
            cp "${APK_PATH}" "${BIN_DIR}/stfsc_engine.apk"
            
            SIZE=$(du -h "${BIN_DIR}/stfsc_engine.apk" | cut -f1)
            
            echo ""
            echo "✅ Quest APK build successful!"
            echo ""
            echo "📁 Output: ${BIN_DIR}/stfsc_engine.apk"
            echo "📏 Size: ${SIZE}"
            echo ""
            echo "Install via Wi-Fi ADB:"
            echo "  adb connect <quest_ip>:5555"
            echo "  adb install ${BIN_DIR}/stfsc_engine.apk"
        else
            echo "❌ APK not found after build!"
            exit 1
        fi
        ;;

    *)
        echo "❌ Unknown platform: ${PLATFORM}"
        show_usage
        ;;
esac
