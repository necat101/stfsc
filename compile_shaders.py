#!/usr/bin/env python3
"""
STFSC Engine - Shader Compiler Script
Cross-platform shader compilation using glslang (downloaded automatically if needed)

Usage: python3 compile_shaders.py [--force]

This script:
1. Downloads glslang if not available on PATH
2. Compiles all .vert and .frag shaders in src/graphics/
3. Outputs .spv files with underscore naming (e.g., vert_vert.spv)
"""

import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
SRC_GRAPHICS = SCRIPT_DIR / "src" / "graphics"
TOOLS_DIR = SCRIPT_DIR / ".shader-tools"

# glslang release URLs - using Vulkan SDK binaries which are more reliable
GLSLANG_VERSION = "1.3.290.0"
VULKAN_SDK_BASE = "https://sdk.lunarg.com/sdk/download"

def get_glslang_url():
    """Get the appropriate glslang download URL for the current platform."""
    system = platform.system().lower()
    
    if system == "linux":
        # Use GitHub releases for glslang directly
        return "https://github.com/KhronosGroup/glslang/releases/download/main-tot/glslang-main-linux-Release.zip"
    elif system == "darwin":
        return "https://github.com/KhronosGroup/glslang/releases/download/main-tot/glslang-main-osx-Release.zip"
    elif system == "windows":
        return "https://github.com/KhronosGroup/glslang/releases/download/main-tot/glslang-main-windows-x64-Release.zip"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

def find_glslang():
    """Find glslangValidator on PATH or in our tools directory."""
    # Check PATH first
    exe_name = "glslangValidator.exe" if platform.system() == "Windows" else "glslangValidator"
    
    result = shutil.which("glslangValidator")
    if result:
        return result
    
    # Check our tools directory
    tools_exe = TOOLS_DIR / "bin" / exe_name
    if tools_exe.exists():
        return str(tools_exe)
    
    return None

def download_glslang():
    """Download and extract glslang to the tools directory."""
    TOOLS_DIR.mkdir(exist_ok=True)
    
    url = get_glslang_url()
    zip_path = TOOLS_DIR / "glslang.zip"
    
    print(f"Downloading glslang from {url}...")
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(TOOLS_DIR)
    
    zip_path.unlink()
    
    # Make executable on Unix
    if platform.system() != "Windows":
        exe = TOOLS_DIR / "bin" / "glslangValidator"
        if exe.exists():
            exe.chmod(0o755)
    
    return find_glslang()

def compile_shader(glslang: str, shader_path: Path, output_path: Path):
    """Compile a single shader to SPIR-V."""
    cmd = [glslang, "-V", "-o", str(output_path), str(shader_path)]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR compiling {shader_path.name}:")
        print(result.stderr)
        return False
    
    print(f"  ✓ {shader_path.name} -> {output_path.name}")
    return True

def main():
    force = "--force" in sys.argv
    
    print("STFSC Engine - Shader Compiler")
    print("=" * 40)
    
    # Find or download glslang
    glslang = find_glslang()
    if not glslang:
        print("glslangValidator not found, downloading...")
        glslang = download_glslang()
        if not glslang:
            print("ERROR: Failed to get glslang compiler")
            sys.exit(1)
    
    print(f"Using compiler: {glslang}")
    
    # Find all shaders
    if not SRC_GRAPHICS.exists():
        print(f"ERROR: Shader directory not found: {SRC_GRAPHICS}")
        sys.exit(1)
    
    shaders = list(SRC_GRAPHICS.glob("*.vert")) + list(SRC_GRAPHICS.glob("*.frag"))
    
    if not shaders:
        print("No shaders found to compile")
        return
    
    print(f"\nCompiling {len(shaders)} shaders...")
    
    success = 0
    failed = 0
    skipped = 0
    
    for shader in sorted(shaders):
        # Output file uses underscore naming: vert.vert -> vert_vert.spv
        out_name = shader.stem + "_" + shader.suffix[1:] + ".spv"
        out_path = SRC_GRAPHICS / out_name
        
        # Skip if output exists and is newer (unless forced)
        if not force and out_path.exists():
            shader_mtime = shader.stat().st_mtime
            out_mtime = out_path.stat().st_mtime
            if out_mtime >= shader_mtime:
                print(f"  - {shader.name} (up to date)")
                skipped += 1
                continue
        
        if compile_shader(glslang, shader, out_path):
            success += 1
        else:
            failed += 1
    
    print(f"\nDone: {success} compiled, {skipped} skipped, {failed} failed")
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
