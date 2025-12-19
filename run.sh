#!/bin/bash

# STFSC Engine - Auto-Setup and Run Script
# This script automatically checks for and installs required dependencies before running the engine.

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║              STFSC Engine - 556 Downtown                      ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

# Detect package manager
detect_package_manager() {
    if command -v dnf &> /dev/null; then
        echo "dnf"
    elif command -v apt &> /dev/null; then
        echo "apt"
    elif command -v pacman &> /dev/null; then
        echo "pacman"
    else
        echo "unknown"
    fi
}

PKG_MANAGER=$(detect_package_manager)

# Check if a package config file exists
check_pkg_config() {
    pkg-config --exists "$1" 2>/dev/null
}

# Install missing dependencies
install_deps() {
    local missing=()
    
    echo "Checking dependencies..."
    
    # Check for ALSA (audio)
    if ! check_pkg_config "alsa"; then
        missing+=("alsa")
    fi
    
    # Check for Vulkan
    if ! check_pkg_config "vulkan"; then
        missing+=("vulkan")
    fi
    
    # Check for X11 (windowing)
    if ! check_pkg_config "x11"; then
        missing+=("x11")
    fi
    
    if [ ${#missing[@]} -eq 0 ]; then
        echo "✓ All dependencies installed!"
        return 0
    fi
    
    echo "Missing dependencies: ${missing[*]}"
    echo ""
    
    case $PKG_MANAGER in
        dnf)
            echo "Installing dependencies via dnf..."
            sudo dnf install -y \
                alsa-lib-devel \
                vulkan-loader-devel vulkan-headers \
                libX11-devel libXcursor-devel libXrandr-devel libXi-devel \
                wayland-devel libxkbcommon-devel \
                pkg-config gcc gcc-c++
            ;;
        apt)
            echo "Installing dependencies via apt..."
            sudo apt update
            sudo apt install -y \
                libasound2-dev \
                libvulkan-dev vulkan-tools \
                libx11-dev libxcursor-dev libxrandr-dev libxi-dev \
                libwayland-dev libxkbcommon-dev \
                pkg-config build-essential
            ;;
        pacman)
            echo "Installing dependencies via pacman..."
            sudo pacman -S --noconfirm \
                alsa-lib \
                vulkan-icd-loader vulkan-headers \
                libx11 libxcursor libxrandr libxi \
                wayland libxkbcommon \
                pkg-config base-devel
            ;;
        *)
            echo "ERROR: Unknown package manager. Please install manually:"
            echo "  - ALSA development headers"
            echo "  - Vulkan development headers"
            echo "  - X11/Wayland development headers"
            exit 1
            ;;
    esac
    
    echo "✓ Dependencies installed successfully!"
}

# Main
install_deps

echo ""
echo "Starting STFSC Engine..."
echo ""

# Run the engine
cargo run "$@"
