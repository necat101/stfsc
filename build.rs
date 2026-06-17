// build.rs - STFSC Engine build script
use std::env;
use std::path::Path;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    if target_os == "android" {
        if target_arch == "aarch64" {
            // Link OpenXR Loader
            let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
            println!(
                "cargo:rustc-link-search=native={}/libs/arm64-v8a",
                manifest_dir
            );
            println!("cargo:rustc-link-lib=openxr_loader");

            // Link Vulkan (should be in NDK, but adding explicit path helps)
            // Adjust this path if needed based on the user's NDK location
            let ndk_path = "/home/netcat/android-sdk/ndk/26.1.10909125";
            let sysroot = format!(
                "{}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android",
                ndk_path
            );
            println!("cargo:rustc-link-search=native={}/33", sysroot);
            println!("cargo:rustc-link-lib=vulkan");

            // Link C++ shared library (required for shaderc and other C++ deps)
            println!("cargo:rustc-link-lib=c++_shared");

            // Copy libc++_shared.so to cargo-apk's temp directory so it gets bundled
            let libcpp_path = format!("{}/libc++_shared.so", sysroot);
            let target_dir = format!(
                "{}/target/cargo-apk-temp-extra-link-libraries",
                manifest_dir
            );

            if Path::new(&libcpp_path).exists() {
                // Create directory if it doesn't exist
                std::fs::create_dir_all(&target_dir).ok();
                let dest_path = format!("{}/libc++_shared.so", target_dir);
                std::fs::copy(&libcpp_path, &dest_path).ok();
                println!("cargo:rustc-link-search=native={}", target_dir);
            }
        }
    }
    // Copy pre-compiled SPIR-V shaders to OUT_DIR
    // Shaders are pre-compiled and stored as .spv files in src/graphics/
    // Naming convention in repo: shader_type.spv (e.g., vert_vert.spv)
    // Required naming for include_bytes!: shader.type.spv (e.g., vert.vert.spv)
    let src_dir = Path::new("src/graphics");
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);

    if src_dir.exists() {
        println!("cargo:rerun-if-changed=src/graphics");

        for entry in std::fs::read_dir(src_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            let filename = path.file_name().unwrap().to_str().unwrap();

            if filename.ends_with(".spv") {
                // Convert underscore naming to dot naming: vert_vert.spv -> vert.vert.spv
                let out_name = filename.replace("_", ".");
                std::fs::copy(&path, out_path.join(&out_name)).unwrap();
            }
        }
    }

    // Exported projects can provide a generated native script module. Editor
    // builds also look for the viewport play cache written by the editor so
    // custom scripts can be baked into the native registry at compile time.
    println!("cargo:rerun-if-env-changed=STFSC_SCRIPT_BINDINGS_RS");
    println!("cargo:rerun-if-changed=script_cache/generated_script_bindings.rs");
    let generated_script_dst = out_path.join("generated_fuckscript.rs");
    if let Ok(script_bindings) = env::var("STFSC_SCRIPT_BINDINGS_RS") {
        let script_bindings = Path::new(&script_bindings);
        if script_bindings.exists() {
            println!("cargo:rerun-if-changed={}", script_bindings.display());
            std::fs::copy(script_bindings, &generated_script_dst).unwrap();
        } else {
            write_empty_generated_scripts(&generated_script_dst);
        }
    } else if Path::new("script_cache/generated_script_bindings.rs").exists() {
        std::fs::copy(
            "script_cache/generated_script_bindings.rs",
            &generated_script_dst,
        )
        .unwrap();
    } else {
        write_empty_generated_scripts(&generated_script_dst);
    }
}

fn write_empty_generated_scripts(path: &Path) {
    std::fs::write(
        path,
        "use super::scripting::ScriptRegistry;\n\npub fn register_generated_scripts(_registry: &mut ScriptRegistry) {}\n",
    )
    .unwrap();
}
