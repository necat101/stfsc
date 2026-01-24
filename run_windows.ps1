# STFSC Engine - Windows Auto-Setup and Run Script
# This script checks for required dependencies before running the engine.

$ErrorActionPreference = "Stop"

Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║              STFSC Engine - 556 Downtown                      ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check for Cargo (Rust)
function Test-Dependency {
    param([string]$Name, [string]$Command, [string]$InstallUrl)
    
    if (Get-Command $Command -ErrorAction SilentlyContinue) {
        Write-Host "✓ $Name is installed" -ForegroundColor Green
        return $true
    } else {
        Write-Host "❌ $Name is not installed" -ForegroundColor Red
        if ($InstallUrl) {
            Write-Host "   Install from: $InstallUrl" -ForegroundColor Yellow
        }
        return $false
    }
}

Write-Host "Checking dependencies..." -ForegroundColor Yellow
Write-Host ""

$allDepsOk = $true

# Check Rust/Cargo
if (-not (Test-Dependency "Rust (cargo)" "cargo" "https://rustup.rs/")) {
    $allDepsOk = $false
}

# Check for Vulkan SDK
$vulkanOk = $false
if (Test-Path env:VULKAN_SDK) {
    Write-Host "✓ Vulkan SDK is installed: $env:VULKAN_SDK" -ForegroundColor Green
    $vulkanOk = $true
} else {
    Write-Host "⚠️  Vulkan SDK might not be installed (VULKAN_SDK env var not set)" -ForegroundColor Yellow
    Write-Host "   If the engine fails to start, install from: https://vulkan.lunarg.com/sdk/home#windows" -ForegroundColor Yellow
    # Don't block - runtime will fail with better error if missing
}

# Check for Visual Studio Build Tools (if using MSVC toolchain)
$rustToolchain = cargo --version 2>&1 | Out-String
if ($rustToolchain -match "msvc") {
    # MSVC toolchain - check for build tools
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
        Write-Host "✓ Visual Studio Build Tools (MSVC compiler) found" -ForegroundColor Green
    } else {
        Write-Host "⚠️  MSVC compiler not found in PATH" -ForegroundColor Yellow
        Write-Host "   You may need Visual Studio Build Tools: https://visualstudio.microsoft.com/downloads/" -ForegroundColor Yellow
    }
}

Write-Host ""

if (-not $allDepsOk) {
    Write-Host "ERROR: Missing required dependencies. Please install them and try again." -ForegroundColor Red
    exit 1
}

Write-Host "All critical dependencies are installed!" -ForegroundColor Green
Write-Host ""
Write-Host "Starting STFSC Engine..." -ForegroundColor Cyan
Write-Host ""

# Run the engine with any additional arguments passed to this script
Push-Location $PSScriptRoot
try {
    cargo run --bin stfsc_engine @args
} finally {
    Pop-Location
}
