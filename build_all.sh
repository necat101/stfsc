#!/bin/bash
# STFSC Engine - Full Distribution Build Script
# Builds both the engine (test client) and editor for GitHub release

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${PROJECT_DIR}/bin"
DIST_DIR="${PROJECT_DIR}/dist"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           STFSC Engine - Distribution Build                   ║"
echo "║                    556 Downtown                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Parse arguments
BUILD_TYPE="${1:-release}"

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

echo "📦 Build type: ${BUILD_TYPE}"
echo ""

# Check dependencies
if ! command -v cargo &> /dev/null; then
    echo "❌ Error: Cargo not found. Please install Rust."
    exit 1
fi

# Check for required system libraries
missing_deps=()
pkg-config --exists vulkan 2>/dev/null || missing_deps+=("vulkan")
pkg-config --exists alsa 2>/dev/null || missing_deps+=("alsa")
pkg-config --exists x11 2>/dev/null || missing_deps+=("x11")

if [ ${#missing_deps[@]} -gt 0 ]; then
    echo "⚠️  Missing system libraries: ${missing_deps[*]}"
    echo "   Run ./run.sh first to auto-install dependencies."
    exit 1
fi

# Create directories
mkdir -p "${BIN_DIR}"
mkdir -p "${DIST_DIR}"

# ============================================================================
# Build Engine (Test Client)
# ============================================================================
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "🎮 Building Engine (Test Client)..."
echo "═══════════════════════════════════════════════════════════════"
cargo build ${CARGO_ARGS} --bin stfsc_engine 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Engine build failed!"
    exit 1
fi

ENGINE_BIN="${PROJECT_DIR}/target/${TARGET_DIR}/stfsc_engine"
if [ -f "${ENGINE_BIN}" ]; then
    cp "${ENGINE_BIN}" "${BIN_DIR}/stfsc_engine"
    cp "${ENGINE_BIN}" "${DIST_DIR}/stfsc_engine"
    chmod +x "${BIN_DIR}/stfsc_engine"
    chmod +x "${DIST_DIR}/stfsc_engine"
    
    SIZE=$(du -h "${BIN_DIR}/stfsc_engine" | cut -f1)
    echo "✅ Engine built: ${SIZE}"
else
    echo "❌ Engine binary not found!"
    exit 1
fi

# ============================================================================
# Build Editor
# ============================================================================
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "🖥️  Building Editor..."
echo "═══════════════════════════════════════════════════════════════"
cargo build ${CARGO_ARGS} --bin editor 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Editor build failed!"
    exit 1
fi

EDITOR_BIN="${PROJECT_DIR}/target/${TARGET_DIR}/editor"
if [ -f "${EDITOR_BIN}" ]; then
    cp "${EDITOR_BIN}" "${BIN_DIR}/stfsc_editor"
    cp "${EDITOR_BIN}" "${DIST_DIR}/stfsc_editor"
    chmod +x "${BIN_DIR}/stfsc_editor"
    chmod +x "${DIST_DIR}/stfsc_editor"
    
    SIZE=$(du -h "${BIN_DIR}/stfsc_editor" | cut -f1)
    echo "✅ Editor built: ${SIZE}"
else
    echo "❌ Editor binary not found!"
    exit 1
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "                    BUILD COMPLETE"
echo "═══════════════════════════════════════════════════════════════"
echo ""

ENGINE_SIZE=$(du -h "${BIN_DIR}/stfsc_engine" | cut -f1)
EDITOR_SIZE=$(du -h "${BIN_DIR}/stfsc_editor" | cut -f1)

echo "📁 Output directory: ${BIN_DIR}/"
echo ""
echo "   stfsc_engine  │ ${ENGINE_SIZE} │ Engine/test client"
echo "   stfsc_editor  │ ${EDITOR_SIZE} │ Scene editor"
echo ""
echo "📦 Distribution directory: ${DIST_DIR}/"
echo ""
echo "To run:"
echo "  ./bin/stfsc_engine    # Run engine"
echo "  ./bin/stfsc_editor    # Run editor"
echo ""
echo "For GitHub release, upload files from: ${DIST_DIR}/"
