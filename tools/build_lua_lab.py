"""Build the Android port from source on Windows, or update a signed Lua Lab APK."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED
from apk_notices import notices
from mod_archive_dependency import prepare as prepare_mod_archives


RENDERER_ENTRY = "lib/arm64-v8a/libnnsdk.so"
LOADER_ENTRY = "lib/arm64-v8a/libnier.so"
SETUP_CLASSES = ("SetupActivity", "DumpScanner")


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)

def prepare_eden(source):
    eden = source / "third_party/eden"
    patch = source / "tools/patches/eden-workgroup-interface.patch"
    if not patch.is_file() or not (eden / "src/shader_recompiler").is_dir():
        raise SystemExit(f"Prepare the patched Rise checkout with tools/setup.sh and pass --renderer-source: {source}")
    # setup.sh applies patches on the first checkout. Also upgrade an already populated third_party/eden.
    apply = ["git", "-C", str(eden), "apply"]
    if subprocess.run([*apply, "--reverse", "--check", str(patch)], capture_output=True).returncode:
        run(*apply, patch)


def build_renderer(source, cmake, android_options, translator=None):
    bundled = source / "runtime/recompiler"
    provider = translator.resolve() if translator else bundled
    if provider == bundled:
        prepare_eden(source)
    build = source / "build/gmem"
    run(cmake/"cmake.exe", "-S", source/"runtime", "-B", build, *android_options,
        f"-DRISE_SHADER_TRANSLATOR_DIR={provider}")
    run(cmake/"cmake.exe", "--build", build, "--target", "nnsdk", "--parallel", "12")
    return build/"libnnsdk.so"


def replace_entries(baseline, base, replacements, extra_entries):
    with ZipFile(baseline) as original, ZipFile(base, "w") as apk:
        for name in replacements:
            original.getinfo(name)
        for entry in original.infolist():
            name = entry.filename.upper()
            if name.startswith("META-INF/") and name.count("/") == 1 and (
                name == "META-INF/MANIFEST.MF" or name.endswith((".SF", ".RSA", ".DSA", ".EC"))
                or name.startswith("META-INF/SIG-")
            ):
                continue
            if entry.filename in extra_entries:
                continue
            data = replacements[entry.filename].read_bytes() if entry.filename in replacements else original.read(entry)
            apk.writestr(entry, data)
        for name, data in extra_entries.items():
            apk.writestr(name, data, compress_type=ZIP_DEFLATED)

def build_dex(root, staging, platform, tools, jdk, release=False):
    classes = staging/"classes"
    classes.mkdir()
    prepare_mod_archives(staging)
    archive_jar = staging/"mod-archive.jar"
    run(jdk/"javac.exe", "--release", "11", "-encoding", "UTF-8", "-cp", os.pathsep.join(map(str,(platform,archive_jar))), "-d", classes,
        *sorted((root/"android/java").rglob("*.java")))
    jar = staging/"classes.jar"
    run(jdk/"jar.exe", "cf", jar, "-C", classes, ".")
    run(jdk/"java.exe", "-cp", tools/"lib/d8.jar", "com.android.tools.r8.D8", "--min-api", "30",
        *(["--release"] if release else []), "--lib", platform, "--output", staging, jar, archive_jar)
    return staging/"classes.dex"


def sign_apk(base, tools, signer, keystore, target):
    aligned = base.parent/"aligned.apk"
    signed = base.parent/"signed.apk"
    run(tools/"zipalign.exe", "-f", "-P", "16", "4", base, aligned)
    run(*signer, "sign", "--ks", keystore, "--ks-pass", "pass:android", "--out", signed, aligned)
    run(*signer, "verify", signed)
    signed.replace(target)
    signature = signed.with_suffix(".apk.idsig")
    if signature.exists():
        signature.replace(target.with_suffix(".apk.idsig"))
    print(target)


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    for opt in ("sdk", "jdk"):
        p.add_argument("--" + opt, type=Path, required=True)
    p.add_argument("--ndk", type=Path, help="Android NDK for native builds")
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source-release", action="store_true", help="Build a non-debuggable release APK from source")
    inputs.add_argument("--release-apk", type=Path, help="Original release for a full Lua Lab build")
    inputs.add_argument("--baseline-apk", type=Path, help="Signed Lua Lab APK to update")
    updates = p.add_mutually_exclusive_group()
    updates.add_argument("--touch-panel-only", action="store_true", help="Update the virtual control view in a merged baseline APK")
    updates.add_argument("--setup-only", action="store_true", help="Update first-run setup and menu language in a merged baseline APK")
    updates.add_argument("--settings-update", action="store_true", help="Update the settings UI and native Activity lifecycle in a merged baseline APK")
    p.add_argument("--keystore", type=Path, default=Path.home()/".android/debug.keystore")
    p.add_argument("--renderer-source", type=Path, default=root,
                   help="Prepared Rise renderer checkout (default: this checkout)")
    p.add_argument("--shader-translator", type=Path, help="CMake provider for the Rise shader translation contract")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    java_only = args.touch_panel_only or args.setup_only
    if (java_only or args.settings_update) and not args.baseline_apk:
        p.error("Partial updates need --baseline-apk")
    if not java_only and not args.ndk:
        p.error("--ndk is required for native builds")
    if args.release_apk and hashlib.sha256(args.release_apk.read_bytes()).hexdigest() != "7cd885bf01c1d2be814ec59b88020c947f959dd7087dbe1911bd32dba149956c":
        p.error("Expected the original mhriseport-v1.0.85.apk")
    build = root / "build/tdb-android"
    cmake = args.sdk / "cmake/3.22.1/bin"
    tools = args.sdk / "build-tools/36.0.0"
    platform = args.sdk / "platforms/android-34/android.jar"
    jdk = args.jdk / "bin"
    os.environ["JAVA_HOME"] = str(args.jdk)
    signer = [jdk/"java.exe", "-jar", tools/"lib/apksigner.jar"]
    name = "mhrise-lua-lab-renderer.apk" if args.baseline_apk else "mhrise-lua-lab.apk"
    if args.source_release:
        name = "MHRise.apk"
    elif args.touch_panel_only:
        name = "mhrise-lua-lab-touch-panel.apk"
    elif args.setup_only:
        name = "mhrise-lua-lab-setup.apk"
    elif args.settings_update:
        name = "mhrise-lua-lab-settings.apk"
    target = (args.output or root/"build"/name).resolve()
    if args.baseline_apk:
        if target == args.baseline_apk.resolve():
            p.error("Choose an output separate from the baseline APK")
        run(*signer, "verify", args.baseline_apk)
        package = subprocess.check_output([tools/"aapt2.exe", "dump", "packagename", args.baseline_apk], text=True)
        if package.strip() != "io.github.iamwilmar.mhriseport.lab":
            p.error("Expected a signed Rise Lua Lab baseline APK")
    if not java_only:
        android_options = ["-G", "Ninja",
            f"-DCMAKE_MAKE_PROGRAM={cmake/'ninja.exe'}",
            f"-DCMAKE_TOOLCHAIN_FILE={args.ndk/'build/cmake/android.toolchain.cmake'}",
            "-DCMAKE_CXX_FLAGS=--driver-mode=g++", "-DANDROID_ABI=arm64-v8a", "-DANDROID_PLATFORM=android-30",
            "-DANDROID_STL=c++_static", "-DCMAKE_BUILD_TYPE=Release"]
        renderer = build_renderer(args.renderer_source.resolve(), cmake, android_options, args.shader_translator)
    if not args.baseline_apk or args.settings_update:
        run(cmake/"cmake.exe", "-S", root/"runtime", "-B", build, *android_options,
            "-DMHRISE_LUA_DEBUG=ON", "-DMHRISE_LOADER_ONLY=ON")
        hooks = ("hook_impl", "file_redirect_hook", "main_hook", "gsl_alloc_hook")
        targets = ("nier",) if args.settings_update else ("nier", "nier_loader")
        run(cmake/"cmake.exe", "--build", build, "--target", *targets,
            *(hooks if args.source_release else ()), "--parallel", "12")
    target.parent.mkdir(parents=True, exist_ok=True)
    if args.baseline_apk:
        with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
            staging = Path(tmp)
            base = staging/"base.apk"
            replacements = {} if java_only else {RENDERER_ENTRY: renderer}
            if java_only or args.settings_update:
                from merge_android_classes import merge_classes
                families = ("TouchPad", "Controls") if args.touch_panel_only else SETUP_CLASSES
                if args.settings_update:
                    families += ("SettingsMenu", "GameActivity", "SaveFiles", "BugReport", "TextPrompt",
                                 "TouchPad", "Controls", "ModFiles", "ModUi", "DriverActivity")
                    replacements[LOADER_ENTRY] = build/"libnier.so"
                dex = build_dex(root, staging, platform, tools, jdk)
                replacements["classes.dex"] = merge_classes(root, staging, args.baseline_apk, dex, jdk, families)
            extras = {"assets/licenses/" + name: data for name, data in notices().items()}
            if args.settings_update:
                extras["assets/lua_build"] = b"1\n"
            replace_entries(args.baseline_apk, base, replacements, extras)
            sign_apk(base, tools, signer, args.keystore, target)
        return
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        staging = Path(tmp)
        build_dex(root, staging, platform, tools, jdk, release=args.source_release)
        base = staging/"base.apk"
        run(tools/"aapt2.exe", "compile", "--dir", root/"android/res", "-o", staging/"res.zip")
        run(tools/"aapt2.exe", "link", "--manifest", root/"android/AndroidManifest.xml", "-I", platform,
            "-R", staging/"res.zip", "--auto-add-overlay", *([] if args.source_release else ["--debug-mode"]), "--rename-manifest-package",
            "io.github.iamwilmar.mhriseport.lab", "-o", base)
        license_entries = {"assets/licenses/" + name: data for name, data in
                           notices(vendor=args.renderer_source / "third_party", lua_deps=build / "_deps").items()}
        with ZipFile(base, "a", ZIP_DEFLATED) as apk:
            if args.source_release:
                apk.write(root/"android/assets/citrus_curves.bin", "assets/citrus_curves.bin")
                for hook in hooks:
                    library = "lib" + hook + ".so"
                    apk.write(build/"adrenotools/src/hook"/library, "lib/arm64-v8a/" + library)
            else:
                with ZipFile(args.release_apk) as original:
                    for name in original.namelist():
                        if name.startswith(("assets/", "lib/")) and name not in (LOADER_ENTRY, RENDERER_ENTRY, "assets/lua_build") and name not in license_entries:
                            apk.writestr(name, original.read(name))
            apk.writestr("assets/lua_build", b"1\n")
            apk.write(build/"libnier.so", LOADER_ENTRY)
            apk.write(staging/"lib/arm64-v8a/libarchive-jni.so", "lib/arm64-v8a/libarchive-jni.so")
            apk.write(renderer, RENDERER_ENTRY)
            apk.write(staging/"classes.dex", "classes.dex")
            for name, data in license_entries.items():
                apk.writestr(name, data)
        sign_apk(base, tools, signer, args.keystore, target)


if __name__ == "__main__":
    main()
