#!/bin/bash
set -e

# Configuration
SDK_ROOT="$HOME/android-sdk"
CMDLINE_TOOLS_URL="https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip"
NDK_VERSION="26.1.10909125"
BUILD_TOOLS_VERSION="34.0.0"
# Install both 33 (Quest 3) and 30 (Common default/compat)
PLATFORMS="platforms;android-33 platforms;android-30"

# Ensure Rust target is installed
if command -v rustup &> /dev/null; then
    echo "Installing Rust Android target..."
    rustup target add aarch64-linux-android
else
    echo "Warning: rustup not found. Skipping target installation."
fi

echo "Setting up Android SDK in $SDK_ROOT..."

# Check for Java
if ! command -v java &> /dev/null; then
    echo "Error: Java (JDK) is not installed. Please install OpenJDK 17 or later."
    echo "  sudo apt install openjdk-17-jdk  # Ubuntu/Debian"
    exit 1
fi

# Create SDK directory
mkdir -p "$SDK_ROOT/cmdline-tools"

# Download Command Line Tools
if [ ! -d "$SDK_ROOT/cmdline-tools/latest" ]; then
    echo "Downloading Command Line Tools..."
    wget -q --show-progress "$CMDLINE_TOOLS_URL" -O cmdline-tools.zip
    unzip -q cmdline-tools.zip -d "$SDK_ROOT/cmdline-tools"
    mv "$SDK_ROOT/cmdline-tools/cmdline-tools" "$SDK_ROOT/cmdline-tools/latest"
    rm cmdline-tools.zip
fi

# Set Environment Variables for this script
export ANDROID_HOME="$SDK_ROOT"
export PATH="$SDK_ROOT/cmdline-tools/latest/bin:$SDK_ROOT/platform-tools:$PATH"

# Accept Licenses
echo "Accepting licenses..."
yes | sdkmanager --licenses > /dev/null

# Install Packages
echo "Installing Platform Tools, SDK Platforms (30 & 33), Build Tools, and NDK..."
# shellcheck disable=SC2086
sdkmanager "platform-tools" $PLATFORMS "build-tools;$BUILD_TOOLS_VERSION" "ndk;$NDK_VERSION"

# Automate .bashrc updates
BASHRC="$HOME/.bashrc"
echo "Updating $BASHRC..."

if ! grep -q "export ANDROID_HOME=\"$SDK_ROOT\"" "$BASHRC"; then
    echo "" >> "$BASHRC"
    echo "# Android SDK" >> "$BASHRC"
    echo "export ANDROID_HOME=\"$SDK_ROOT\"" >> "$BASHRC"
    echo "export ANDROID_NDK_HOME=\"$SDK_ROOT/ndk/$NDK_VERSION\"" >> "$BASHRC"
    echo "export PATH=\"\$ANDROID_HOME/cmdline-tools/latest/bin:\$ANDROID_HOME/platform-tools:\$PATH\"" >> "$BASHRC"
    echo "Added environment variables to $BASHRC"
else
    echo "Environment variables already present in $BASHRC"
fi

echo ""
echo "============================================================"
echo "Android SDK Setup Complete!"
echo "============================================================"
echo "Please run the following command to apply changes to your current shell:"
echo "source ~/.bashrc"
echo ""
