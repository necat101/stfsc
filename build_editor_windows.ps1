# STFSC Engine - Windows Editor Build Script
# Builds the desktop editor for Windows

$ErrorActionPreference = "Stop"

$PROJECT_DIR = $PSScriptRoot
$BIN_DIR = Join-Path $PROJECT_DIR "bin"

Write-Host "╔═══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║            STFSC Editor - Windows Build                       ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check for Cargo
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Error: Cargo not found. Please install Rust from https://rustup.rs/" -ForegroundColor Red
    exit 1
}

# Create bin directory
if (-not (Test-Path $BIN_DIR)) {
    New-Item -ItemType Directory -Path $BIN_DIR | Out-Null
}

# Build editor
Write-Host "🔨 Building editor binary..." -ForegroundColor Yellow
Push-Location $PROJECT_DIR
try {
    cargo build --release --bin editor 2>&1 | Out-Host
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Build failed!" -ForegroundColor Red
        exit 1
    }
} finally {
    Pop-Location
}

# Copy binary
$RELEASE_BIN = Join-Path $PROJECT_DIR "target\release\editor.exe"
if (Test-Path $RELEASE_BIN) {
    $DEST_BIN = Join-Path $BIN_DIR "stfsc_editor.exe"
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
    Write-Host "  .\bin\stfsc_editor.exe" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "❌ Release binary not found!" -ForegroundColor Red
    exit 1
}
