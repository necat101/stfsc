use std::env;
use std::path::Path;
use shaderc;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    if target_os == "android" {
        if target_arch == "aarch64" {
            // Link OpenXR Loader
            let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
            println!("cargo:rustc-link-search=native={}/libs/arm64-v8a", manifest_dir);
            println!("cargo:rustc-link-lib=openxr_loader");

            // Link Vulkan (should be in NDK, but adding explicit path helps)
            // Adjust this path if needed based on the user's NDK location
            println!("cargo:rustc-link-search=native=/home/netcat/android-sdk/ndk/26.1.10909125/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/33");
            println!("cargo:rustc-link-lib=vulkan");
        }
    }
    // Compile Shaders
    let compiler = shaderc::Compiler::new().unwrap();
    let options = shaderc::CompileOptions::new().unwrap();
    
    let src_dir = Path::new("src/graphics");
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);

    if src_dir.exists() {
        println!("cargo:rerun-if-changed=src/graphics");
        for entry in std::fs::read_dir(src_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if let Some(ext) = path.extension() {
                let shader_kind = if ext == "vert" {
                    shaderc::ShaderKind::Vertex
                } else if ext == "frag" {
                    shaderc::ShaderKind::Fragment
                } else {
                    continue;
                };

                let src = std::fs::read_to_string(&path).unwrap();
                let binary_result = compiler.compile_into_spirv(
                    &src, 
                    shader_kind, 
                    path.file_name().unwrap().to_str().unwrap(), 
                    "main", 
                    Some(&options)
                ).unwrap();

                let out_name = format!("{}.spv", path.file_stem().unwrap().to_str().unwrap());
                std::fs::write(out_path.join(out_name), binary_result.as_binary_u8()).unwrap();
            }
        }
    }
}
