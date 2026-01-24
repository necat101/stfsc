# STFSC Engine - Windows Release Build Script
# Builds a release-optimized binary for Windows desktop

$ErrorActionPreference = "Stop"

$PROJECT_DIR = $PSScriptRoot
$BIN_DIR = Join-Path $PROJECT_DIR "bin"

Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║          STFSC Engine - Windows Release Build                 ║" -ForegroundColor Cyan
Write-Host "║                    556 Downtown                               ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Ensure dependencies
Write-Host "📦 Checking dependencies..." -ForegroundColor Yellow

# Check for Cargo (Rust)
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Error: Cargo not found. Please install Rust from https://rustup.rs/" -ForegroundColor Red
    exit 1
}

# Check for Vulkan SDK
$vulkanInstalled = $false
if (Test-Path env:VULKAN_SDK) {
    Write-Host "✓ Vulkan SDK found: $env:VULKAN_SDK" -ForegroundColor Green
    $vulkanInstalled = $true
} else {
    Write-Host "⚠️  Warning: VULKAN_SDK environment variable not set." -ForegroundColor Yellow
    Write-Host "   The build may fail if Vulkan SDK is not installed." -ForegroundColor Yellow
    Write-Host "   Download from: https://vulkan.lunarg.com/sdk/home#windows" -ForegroundColor Yellow
    Write-Host ""
    
    # Continue anyway - cargo will fail with a better error if truly missing
}

# Create bin directory
if (-not (Test-Path $BIN_DIR)) {
    New-Item -ItemType Directory -Path $BIN_DIR | Out-Null
}

# Build release
Write-Host ""
Write-Host "🔨 Building release binary..." -ForegroundColor Yellow
Push-Location $PROJECT_DIR
try {
    cargo build --release --bin stfsc_engine 2>&1 | Out-Host
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Build failed!" -ForegroundColor Red
        exit 1
    }
} finally {
    Pop-Location
}

# Copy binary
$RELEASE_BIN = Join-Path $PROJECT_DIR "target\release\stfsc_engine.exe"
if (Test-Path $RELEASE_BIN) {
    $DEST_BIN = Join-Path $BIN_DIR "stfsc_engine.exe"
    Copy-Item $RELEASE_BIN $DEST_BIN -Force
    
    # Get binary size
    $SIZE = (Get-Item $DEST_BIN).Length / 1MB
    $SIZE_STR = "{0:N2} MB" -f $SIZE
    
    Write-Host ""
    Write-Host "✅ Build successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📁 Output: $DEST_BIN" -ForegroundColor Cyan
    Write-Host "📏 Size: $SIZE_STR" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To run:" -ForegroundColor Yellow
    Write-Host "  .\bin\stfsc_engine.exe" -ForegroundColor White
    Write-Host "  OR" -ForegroundColor Yellow
    Write-Host "  .\run_windows.ps1" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "❌ Release binary not found!" -ForegroundColor Red
    exit 1
}
