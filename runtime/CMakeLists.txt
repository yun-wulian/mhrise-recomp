cmake_minimum_required(VERSION 3.22)
project(nier_runtime C CXX ASM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra)
# Release (tools/build_apk.sh RELEASE=1, la de los jugadores): cada función y dato en su sección para que el enlazador
# tire lo que nadie usa, funciones idénticas plegadas (solo las que no se comparan por dirección) y sin tabla de
# símbolos (la dinámica, la que enlaza el juego, se queda). Sin LTO: con lo que el juego enlaza por nombre, sin probar.
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-ffunction-sections -fdata-sections)
  add_link_options(-Wl,--gc-sections -Wl,--icf=safe -s)
endif()

option(MHRISE_LUA_DEBUG "Build the read-only Lua/TDB startup debugger" OFF)
option(MHRISE_LOADER_ONLY "Build the loader with a matching prebuilt renderer" OFF)
set(EXTERNAL ${CMAKE_CURRENT_LIST_DIR}/external)
if(NOT CMAKE_ANDROID_ARCH_ABI)
  set(CMAKE_ANDROID_ARCH_ABI ${ANDROID_ABI})
endif()
add_subdirectory(${EXTERNAL}/libadrenotools adrenotools)

if(NOT MHRISE_LOADER_ONLY)
# libnnsdk.so: nuestra reimplementación del SDK de Nintendo. Todo símbolo que
# define se enlaza al juego con prioridad sobre bionic y sobre los stubs.
add_library(nnsdk SHARED
  src/nnsdk/account.cpp
  src/nnsdk/cheats.cpp
  src/nnsdk/audio.cpp
  src/nnsdk/codec.cpp
  src/nnsdk/cxx.cpp
  src/nnsdk/diag.cpp
  src/nnsdk/fs.cpp
  src/nnsdk/hid.cpp
  src/nnsdk/libc.cpp
  src/nnsdk/mem.cpp
  src/nnsdk/movie.cpp
  src/nnsdk/oe.cpp
  src/nnsdk/swkbd.cpp
  src/nnsdk/time.cpp
  src/nnsdk/nvn/bcn.cpp
  src/nnsdk/nvn/astc.cpp
  src/nnsdk/nvn/driver.cpp
  src/nnsdk/nvn/perf_hint.cpp
  src/nnsdk/nvn/game_tweaks.cpp
  src/nnsdk/nvn/buffer.cpp
  src/nnsdk/nvn/device.cpp
  src/nnsdk/nvn/cmdbuf.cpp
  src/nnsdk/nvn/count_check.cpp
  src/nnsdk/nvn/formats.cpp
  src/nnsdk/nvn/executor.cpp
  src/nnsdk/nvn/face_nan_probe.cpp
  src/nnsdk/nvn/pipeline_compiler.cpp
  src/nnsdk/nvn/driver_pipeline_cache.cpp
  src/nnsdk/nvn/gpu_memory.cpp
  src/nnsdk/nvn/memory.cpp
  src/nnsdk/nvn/layout.cpp
  src/nnsdk/nvn/lsfg/chain.cpp
  src/nnsdk/nvn/lsfg/dll.cpp
  src/nnsdk/nvn/lsfg/framegen.cpp
  src/nnsdk/nvn/lsfg/motion.cpp
  src/nnsdk/nvn/state.cpp
  src/nnsdk/nvn/procs.cpp
  src/nnsdk/nvn/program.cpp
  src/nnsdk/nvn/spirv_cache.cpp
  src/nnsdk/nvn/queue.cpp
  src/nnsdk/nvn/texture.cpp
  src/nnsdk/nvn/texture_view.cpp
  src/nnsdk/nvn/texture_region.cpp
  src/nnsdk/nvn/vertex_fetch.cpp
  src/nnsdk/nvn/present_image.cpp
  src/nnsdk/nvn/present_probe.cpp
  src/nnsdk/nvn/vk.cpp
  src/nnsdk/nvn/window.cpp
  src/nnsdk/nvn/hud.cpp
  src/nnsdk/nvn/fsr.cpp
  src/nnsdk/os.cpp
  src/nnsdk/vi.cpp
  src/nnsdk/aoc.cpp
  src/nnsdk/album.cpp
  src/nnsdk/bcat.cpp
  src/nnsdk/crypto.cpp
  src/nnsdk/cxx_ios.cpp
  src/nnsdk/ec.cpp
  src/nnsdk/err.cpp
  src/nnsdk/friends.cpp
  src/nnsdk/hid_vibration.cpp
  src/nnsdk/image.cpp
  src/nnsdk/ldn.cpp
  src/nnsdk/libc_file.cpp
  src/nnsdk/libc_thread.cpp
  src/nnsdk/movie_ff.cpp
  src/nnsdk/nfp.cpp
  src/nnsdk/ngc.cpp
  src/nnsdk/nifm.cpp
  src/nnsdk/nsd.cpp
  src/nnsdk/os_fiber.cpp
  src/nnsdk/os_memory.cpp
  src/nnsdk/watch.cpp
  src/nnsdk/os_sync.cpp
  src/nnsdk/pctl.cpp
  src/nnsdk/prepo.cpp
  src/nnsdk/settings.cpp
  src/nnsdk/socket.cpp
  src/nnsdk/ssl.cpp
  src/nnsdk/util.cpp
  src/nnsdk/web.cpp
)
target_include_directories(nnsdk PRIVATE src)
# Recompilador de shaders de Eden (GPL-3): libnnsdk.so queda bajo GPL-3.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(shader_translator.cmake)
target_include_directories(nnsdk PRIVATE recompiler)
target_link_libraries(nnsdk PRIVATE rise_shader_translator)
# Vulkan por punteros (volk, desde nvn/vk.h): sin enlazar libvulkan.so, para que el driver pueda ser el
# del sistema o uno del usuario cargado con libadrenotools (nvn/driver.cpp). Las librerías de hooks de libadrenotools
# (libhook_impl.so, libmain_hook.so, libfile_redirect_hook.so, libgsl_alloc_hook.so) van en la APK junto a libnnsdk.so.
target_sources(nnsdk PRIVATE ${EXTERNAL}/volk/volk.c)
# ARM's ASTC encoder (Apache-2.0, unmodified): BC textures re-encoded as ASTC on Mali (nvn/astc.cpp). NEON only.
file(GLOB ASTCENC_SOURCES ${EXTERNAL}/astcenc/*.cpp)
add_library(astcenc STATIC ${ASTCENC_SOURCES})
target_compile_definitions(astcenc PRIVATE ASTCENC_NEON=1 ASTCENC_SVE=0 ASTCENC_SSE=0 ASTCENC_AVX=0 ASTCENC_POPCNT=0
                                           ASTCENC_F16C=0)
target_compile_options(astcenc PRIVATE -w -O2)
target_include_directories(astcenc PUBLIC ${EXTERNAL}/astcenc)
target_link_libraries(nnsdk PRIVATE astcenc)
target_include_directories(nnsdk PRIVATE ${EXTERNAL}/volk)
target_compile_definitions(nnsdk PRIVATE VK_NO_PROTOTYPES VK_USE_PLATFORM_ANDROID_KHR)
target_link_libraries(nnsdk PRIVATE adrenotools android mediandk aaudio z dl)
# FFmpeg (LGPL, solo el demuxer MP4; tools/build_ffmpeg.sh): respaldo cuando AMediaExtractor no encuentra pistas en un
# vídeo. Sin esa carpeta se compila igual, sin el respaldo.
set(FFMPEG_DIR ${CMAKE_CURRENT_LIST_DIR}/../third_party/ffmpeg-android)
if(EXISTS ${FFMPEG_DIR}/lib/libavformat.a)
  target_include_directories(nnsdk PRIVATE ${FFMPEG_DIR}/include)
  target_compile_definitions(nnsdk PRIVATE MHRISE_FFMPEG=1)
  target_link_libraries(nnsdk PRIVATE ${FFMPEG_DIR}/lib/libavformat.a ${FFMPEG_DIR}/lib/libavcodec.a
                                      ${FFMPEG_DIR}/lib/libavutil.a m)
else()
  message(WARNING "third_party/ffmpeg-android not found: building without the FFmpeg video fallback (run tools/build_ffmpeg.sh)")
endif()
# Su libc++ estática no se exporta: el juego debe usar una sola copia de libc++abi (la del loader).
target_link_options(nnsdk PRIVATE -Wl,--exclude-libs,ALL)
endif()

# Núcleo del loader: carga main, lo reloca contra libnnsdk/bionic/stubs y lo arranca.
add_library(loader_core OBJECT
  src/loader/run.cpp
  src/loader/nso.cpp
  src/loader/imports.cpp
  src/loader/crash.cpp
  src/loader/host_symbols.c
  src/loader/stub_dispatch.S
  src/loader/tls.cpp
  src/loader/tls_slow.S
  src/loader/probe.cpp
  src/loader/probe_slow.S
)
target_include_directories(loader_core PUBLIC src)
set_target_properties(loader_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MHRISE_LUA_DEBUG)
  include(FetchContent)
  FetchContent_Declare(lua54 URL https://www.lua.org/ftp/lua-5.4.9.tar.gz
    URL_HASH SHA256=2335b6c582a52654f94612bf10d2f4672805d05329aa6568b1d8cd9e5c6fb8e6)
  FetchContent_MakeAvailable(lua54)
  FetchContent_Declare(rise_imgui URL https://codeload.github.com/ocornut/imgui/tar.gz/f5befd2d29e66809cd1110a152e375a7f1981f06
    URL_HASH SHA256=85f4ce357df05bcc331b587f01976f47fb55f19fadf477a7907289686bc3f4c8)
  FetchContent_MakeAvailable(rise_imgui)
  # Android's variable Noto CJK collections use CFF2 outlines.
  foreach(lib ZLIB BZIP2 PNG HARFBUZZ BROTLI)
    set(FT_DISABLE_${lib} ON CACHE BOOL "" FORCE)
  endforeach()
  FetchContent_Declare(freetype URL https://codeload.github.com/freetype/freetype/tar.gz/refs/tags/VER-2-13-3
    URL_HASH SHA256=bc5c898e4756d373e0d991bab053036c5eb2aa7c0d5c67e8662ddc6da40c4103)
  FetchContent_MakeAvailable(freetype)
  set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
  set(JSON_Install OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(rise_json URL https://codeload.github.com/nlohmann/json/tar.gz/refs/tags/v3.11.3
    URL_HASH SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406)
  FetchContent_MakeAvailable(rise_json)
  add_library(rise_imgui_ui STATIC ${rise_imgui_SOURCE_DIR}/imgui.cpp ${rise_imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${rise_imgui_SOURCE_DIR}/imgui_draw.cpp ${rise_imgui_SOURCE_DIR}/imgui_tables.cpp
    ${rise_imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp ${rise_imgui_SOURCE_DIR}/misc/freetype/imgui_freetype.cpp)
  target_include_directories(rise_imgui_ui PUBLIC ${rise_imgui_SOURCE_DIR})
  target_compile_definitions(rise_imgui_ui PRIVATE IMGUI_IMPL_OPENGL_ES3 IMGUI_ENABLE_FREETYPE)
  target_link_libraries(rise_imgui_ui PUBLIC GLESv3 freetype)
  file(GLOB lua_sources "${lua54_SOURCE_DIR}/src/*.c")
  list(REMOVE_ITEM lua_sources "${lua54_SOURCE_DIR}/src/lua.c" "${lua54_SOURCE_DIR}/src/luac.c")
  add_library(rise_lua STATIC ${lua_sources})
  set_target_properties(rise_lua PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(rise_lua PUBLIC "${lua54_SOURCE_DIR}/src")
  target_compile_definitions(loader_core PUBLIC MHRISE_LUA_DEBUG)
  target_sources(loader_core PRIVATE src/loader/tdb_lua.cpp src/loader/lua_mod_hooks.cpp src/loader/live_read.cpp
    src/loader/application_lua.cpp src/loader/lua_frame_gc.cpp src/loader/lua_values.cpp src/loader/mod_ui.cpp src/loader/lua_data.cpp
    src/loader/lua_native_hooks.cpp src/loader/lua_explorer.cpp src/loader/native_hook.cpp src/loader/native_hook.S)
  target_link_libraries(loader_core PUBLIC rise_lua rise_imgui_ui nlohmann_json::nlohmann_json log)
endif()

# nier_loader: desde la terminal de la Flip, sin ventana.
add_executable(nier_loader src/loader/main.cpp)
target_link_libraries(nier_loader PRIVATE loader_core dl)

# libnier.so: la APK (NativeActivity), con ventana.
set(GLUE ${ANDROID_NDK}/sources/android/native_app_glue)
add_library(nier SHARED src/loader/android_main.cpp ${GLUE}/android_native_app_glue.c)
target_include_directories(nier PRIVATE ${GLUE})
target_link_libraries(nier PRIVATE loader_core adrenotools android log dl)
target_link_options(nier PRIVATE -u ANativeActivity_onCreate)
