#!/bin/bash

# STFSC Engine - Fedora Vulkan & Development Setup Script
# This script installs the necessary libraries to compile and run the engine on Fedora.

echo "--- STFSC Engine Setup (Fedora) ---"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root or with sudo:"
  echo "sudo ./setup_vulkan_fedora.sh"
  exit 1
fi

echo "Updating package repositories..."
dnf check-update

echo "Installing Vulkan development libraries..."
dnf install -y \
    vulkan-loader-devel \
    vulkan-tools \
    vulkan-headers \
    mesa-vulkan-drivers

echo "Installing windowing system dependencies (Winit/X11/Wayland)..."
dnf install -y \
    libX11-devel \
    libXcursor-devel \
    libXrandr-devel \
    libXi-devel \
    wayland-devel \
    libxkbcommon-devel

echo "Installing audio dependencies (ALSA for rodio)..."
dnf install -y \
    alsa-lib-devel

echo "Installing general build tools..."
dnf install -y \
    gcc \
    gcc-c++ \
    cmake \
    pkg-config

echo "------------------------------------------------"
echo "Setup complete! You should now be able to run:"
echo "cargo run"
echo "------------------------------------------------"
