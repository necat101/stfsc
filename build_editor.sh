#!/bin/bash
# STFSC Engine - Editor Build Script
# Builds the STFSC Editor for scene editing and development

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${PROJECT_DIR}/bin"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           STFSC Engine - Editor Build                         ║"
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

# Create bin directory
mkdir -p "${BIN_DIR}"

# Build editor
echo "🔨 Building editor (${BUILD_TYPE})..."
cargo build ${CARGO_ARGS} --bin editor 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

# Copy binary
EDITOR_BIN="${PROJECT_DIR}/target/${TARGET_DIR}/editor"
if [ -f "${EDITOR_BIN}" ]; then
    cp "${EDITOR_BIN}" "${BIN_DIR}/stfsc_editor"
    chmod +x "${BIN_DIR}/stfsc_editor"
    
    SIZE=$(du -h "${BIN_DIR}/stfsc_editor" | cut -f1)
    
    echo ""
    echo "✅ Editor build successful!"
    echo ""
    echo "📁 Output: ${BIN_DIR}/stfsc_editor"
    echo "📏 Size: ${SIZE}"
    echo ""
    echo "To run:"
    echo "  ./bin/stfsc_editor"
    echo ""
else
    echo "❌ Editor binary not found!"
    exit 1
fi
