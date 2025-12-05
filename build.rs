use std::env;
use std::path::Path;

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
}
