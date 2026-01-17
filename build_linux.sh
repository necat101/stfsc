#!/bin/bash
# STFSC Engine - Linux Release Build Script
# Builds a release-optimized binary for Linux desktop

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${PROJECT_DIR}/bin"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║           STFSC Engine - Linux Release Build                  ║"
echo "║                    556 Downtown                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Ensure dependencies
echo "📦 Checking dependencies..."
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
    echo "   Run ./run.sh first to auto-install dependencies, or install manually."
    exit 1
fi

# Create bin directory
mkdir -p "${BIN_DIR}"

# Build release
echo ""
echo "🔨 Building release binary..."
cargo build --release --bin stfsc_engine 2>&1

# Check if build succeeded
if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

# Copy binary
RELEASE_BIN="${PROJECT_DIR}/target/release/stfsc_engine"
if [ -f "${RELEASE_BIN}" ]; then
    cp "${RELEASE_BIN}" "${BIN_DIR}/stfsc_engine"
    chmod +x "${BIN_DIR}/stfsc_engine"
    
    # Get binary size
    SIZE=$(du -h "${BIN_DIR}/stfsc_engine" | cut -f1)
    
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "📁 Output: ${BIN_DIR}/stfsc_engine"
    echo "📏 Size: ${SIZE}"
    echo ""
    echo "To run:"
    echo "  ./bin/stfsc_engine"
    echo ""
else
    echo "❌ Release binary not found!"
    exit 1
fi
