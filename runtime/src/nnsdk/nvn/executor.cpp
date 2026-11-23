// SPDX-License-Identifier: GPL-3.0-or-later
// Ejecutor: reproduce en Vulkan los comandos que el juego envía a la cola NVN. El estado (render
// targets, programas, buffers, texturas…) es de la cola y persiste entre listas, como en NVN.
// De momento todo es síncrono: cada envío graba un VkCommandBuffer, lo manda y espera, así que los
// syncs de NVN ya están señalados al volver.
//
// Memoria: el driver no tiene VK_EXT_external_memory_host, así que los uniform buffers, vértices e
// índices se copian de la memoria del juego a un staging al dibujar. Las imágenes viven siempre en
// VK_IMAGE_LAYOUT_GENERAL (correcto y simple; las dependencias de los render passes sincronizan).

#include <pthread.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <bitset>
#include <cerrno>
#include <cmath>
#include <functional>
#include <future>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <thread>
#include <unordered_set>
#include <cstring>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <fstream>
#include <type_traits>
#include <unordered_map>

#include "nnsdk/nvn/astc.h"
#include "nnsdk/nvn/bcn.h"
#include "nnsdk/nvn/resource_access.h"
#include "nnsdk/nvn/texture_view.h"
#include "nnsdk/nvn/driver_pipeline_cache.h"
#include "nnsdk/nvn/pipeline_compiler.h"
#include "nnsdk/nvn/queue_state.h"
#include "nnsdk/nvn/count_check.h"
#include "nnsdk/nvn/face_nan_probe.h"
#include "nnsdk/watch.h"
#include "nnsdk/nvn/present_image.h"
#include "nnsdk/nvn/texture_region.h"
#include "nnsdk/nvn/hud.h"
#include "nnsdk/nvn/lsfg/framegen.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/runtime.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/formats.h"
#include "nnsdk/nvn/records_file.h"
#include "nnsdk/nvn/shaders/spirv.h"
#include "nvn_traductor_huella.h"  // NVN_TRADUCTOR_HUELLA: los registros del precalentado son de este traductor
#include "nnsdk/ini.h"

namespace nvn {

namespace {

// DIAGNÓSTICO, siempre: el SPIR-V de cada etapa de una pipeline que el driver no quiso crear (el VK_ERROR_UNKNOWN del
// compilador de Qualcomm 512.8xx en la Adreno 730 con muchos pixel shaders de la escena; con Turnip salen bien), en
// <datos>/pipelines_fallidas/<nombre>.spv, para ver fuera qué tienen. 64 como mucho por partida, cada módulo una vez.
void dump_rejected_spirv(const std::string& name, const std::vector<uint32_t>& spirv) {
    static std::mutex m;
    static std::unordered_set<std::string> done;
    std::lock_guard lock(m);
    if (spirv.empty() || done.size() >= 64 || !done.insert(name).second) return;
    const char* data = getenv("NIER_DATA");
    const std::string dir = std::string(data ? data : ".") + "/pipelines_fallidas";
    mkdir(dir.c_str(), 0755);
    if (FILE* f = fopen((dir + "/" + name + ".spv").c_str(), "wb")) {
        fwrite(spirv.data(), sizeof(uint32_t), spirv.size(), f);
        fclose(f);
        if (done.size() == 1) NVN_LOG("SPIR-V of rejected pipelines: in %s\n", dir.c_str());
    }
}

using nvnshader::Resource;
using nvnshader::Stage;

// Dónde espera el compilador de NVN cada cosa (deducido de los shaders traducidos):
constexpr uint32_t kTextureHandleBank = 2;      // handles de textura: c2[0x168 + 8·i]
constexpr uint32_t kTextureHandleBase = 0x168;
constexpr uint32_t kSamplerHandleBase = 0x568;  // samplers separados (BindSeparateSampler), en c2 tras los de textura
constexpr uint32_t kUniformBankBase = 3;        // uniform buffer i del juego = banco 3 + i
constexpr uint32_t kShaderConstantsBank = 1;   // c1: constantes que el compilador dejó tras el código
constexpr uint32_t kComputeDriverBank = 0;      // compute: c0 del driver, con los handles de textura en
                                                // 0x168 + 8·i y la tabla de storage buffers en 0x310 + 16·i
constexpr uint32_t kComputeStorageBase = 0x310;
constexpr uint32_t kComputeDriverSize = 0x400;
// Compute: Maxwell solo tiene 8 bancos de constantes en compute (c0 driver, c1 constantes, c2 handles, c3..c7 los
// uniform buffers 0..4). Los uniform buffers 5..13 ("UBE") los lee el shader como memoria global con la dirección y
// el tamaño que NVN deja en c0[0x260 + 16·j] (j = buffer − 5; como el ComputeClass de Ryujinx): Eden los ve como
// storage buffers de solo lectura. Antes salían 256 bytes a cero (tk1: "F compute SSBO leído 0+256" ×4 en la
// iluminación de efectos 314939, cuyo buffer de 2,5 MB leen los vertex shaders de los efectos). <datos>/no_ube: como antes.
constexpr uint32_t kComputeUbeBase = 0x260, kComputeUbeCount = 9, kComputeUbeFirst = 5;
// Compute: c2 es un banco de handles como el de gráficos (texturas en 0x168 + 8·i, imágenes en 0x120 + 8·i); los TEX.B
// de MH Rise cargan de ahí (17 texturas y 27 imágenes en los 139 compute volcados). Se enlaza el mismo c0 sintético.
// <datos>/no_c2_compute: como antes.
constexpr uint32_t kComputeImageHandleBase = 0x120;
constexpr uint32_t kLowTextureHandleBase = 0x20;  // la tabla baja de c2 de algunos shaders de la interfaz (resolve_texture)
// Gráficos: un c0 común con una tabla de storage buffers por etapa en 0x110 + 0x100·etapa (orden de
// nvnshader::Stage; el pixel shader los busca en 0x510).
constexpr uint32_t kGraphicsStorageBase = 0x110, kGraphicsStorageStride = 0x100, kGraphicsDriverSize = 0x700;
static_assert(kGraphicsStorageStride == kMaxStorage * 16 &&
              kGraphicsDriverSize <= kGraphicsStorageBase + kStages * kGraphicsStorageStride);
// Imágenes (SULD/SUST): su handle va en la misma tabla del driver que los de textura (c0 en compute, c2 en gráficos),
// pero no se sabe dónde empieza la suya: sin shaders de imagen de NieR y sin cabeceras de NVN. Se supone justo antes de
// las texturas (8 ranuras de 8 bytes, 0x128..0x168); si un shader pide una imagen fuera de ahí o en una ranura vacía,
// se reparten por orden (la de offset menor, la primera ranura enlazada…). resolve_image registra lo que decide.
constexpr uint32_t kImageHandleBase = 0x128;
constexpr VkDeviceSize kStagingChunk = 16 << 20;  // también lo más grande que cabe en una reserva
constexpr VkDeviceSize kFreeStagingBytes = 128 << 20;  // trozos de staging sueltos que se guardan para reutilizar
// Un teléfono con poca RAM (menos de 5 GB en total, o <datos>/low_memory): lo que se guarda por si acaso (los trozos de
// staging sueltos y los espejos de buffers), a la mitad. Lo que hace falta en cada frame (las arenas) no cambia: con
// menos, la fábrica bajaba a 1 fps.
bool low_memory() {
    static const bool low = [] {
        struct sysinfo si {};
        const uint64_t total = sysinfo(&si) == 0 ? uint64_t{si.totalram} * si.mem_unit : 0;
        const bool on = (total && total < (5ull << 30)) || NVN_FLAG("low_memory");
        if (on) NVN_LOG("poca memoria (%llu MB): reservas de staging y espejos a la mitad\n", (unsigned long long)(total >> 20));
        return on;
    }();
    return low;
}
constexpr int kLoggedDraws = 12;

std::recursive_mutex g_mutex;  // el estado del ejecutor (y lo que lee de las texturas y samplers del juego)
// Perfil de cargas: lo que los hilos del juego esperan a g_mutex en texture.cpp (el ejecutor lo tiene durante todo un envío).
std::atomic<uint64_t> g_executor_lock_waits{0}, g_executor_lock_wait_ns{0};
// La VkQueue no admite accesos concurrentes: cada vkQueueSubmit/vkQueuePresentKHR, y nada más. Aparte de g_mutex
// para que presentar (y sus esperas al fence y al swapchain) no espere a que acabe una reproducción.
std::mutex g_queue_mutex;

// NVNshaderStage: VERTEX 0, FRAGMENT 1, GEOMETRY 2, TESS_CONTROL 3, TESS_EVALUATION 4, COMPUTE 5.
Stage from_nvn_stage(int s) {
    switch (s) {
        case 0: return Stage::Vertex;
        case 1: return Stage::Fragment;
        case 2: return Stage::Geometry;
        case 3: return Stage::TessControl;
        case 4: return Stage::TessEval;
        default: return Stage::Compute;
    }
}


bool has_stencil(VkFormat f) {
    return f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_S8_UINT;
}
const char* debug_var(const char* var, uint64_t frame);  // DEPURACIÓN: más abajo
bool storage_capable(VkFormat f) {
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(vk()->gpu, f, &fp);
    return (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
}

// Posición de un tipo en el variant Command (para recorrer listas con un switch, sin std::visit).
template <typename T, typename V>
struct VariantIndex;
template <typename T, typename... Ts>
struct VariantIndex<T, std::variant<Ts...>> {
    static constexpr size_t value = [] {
        constexpr bool same[] = {std::is_same_v<T, Ts>...};
        size_t i = 0;
        while (i < sizeof...(Ts) && !same[i]) i++;
        return i;
    }();
};
template <typename T>
constexpr size_t kCommandIndex = VariantIndex<T, Command>::value;
static_assert(kCommandIndex<cmd::SetRenderTargets> == 0 && kCommandIndex<cmd::PatchSize> == std::variant_size_v<Command> - 1);

// Mismos targets con las mismas vistas (lo que hay más allá de count no cuenta).
bool same_targets(const cmd::SetRenderTargets& a, const cmd::SetRenderTargets& b) {
    if (a.count != b.count || a.depth != b.depth) return false;
    if (a.depth && !same_attachment_view(*a.depth, a.depth_view, b.depth_view)) return false;
    for (int k = 0; k < a.count; k++) {
        if (a.colors[k] != b.colors[k]) return false;
        if (a.colors[k] && !same_attachment_view(*a.colors[k], a.color_views[k], b.color_views[k])) return false;
    }
    return true;
}

// ¿Caben a y b en un mismo pase (sus adjuntos juntos)? La misma profundidad (textura y vista, o ninguna) y, en cada render
// target que tengan los dos, la misma textura con la misma vista; una textura no puede ir en dos sitios distintos. El
// tamaño y las muestras los mira quien llama.
bool targets_fit(const cmd::SetRenderTargets& a, const cmd::SetRenderTargets& b) {
    if (a.depth != b.depth) return false;
    if (a.depth && !same_attachment_view(*a.depth, a.depth_view, b.depth_view)) return false;
    for (int k = 0; k < a.count; k++) {
        if (!a.colors[k]) continue;
        for (int j = 0; j < b.count; j++) {
            if (b.colors[j] != a.colors[k]) continue;
            if (j != k || !same_attachment_view(*a.colors[k], a.color_views[k], b.color_views[k])) return false;
        }
        if (k < b.count && b.colors[k] && b.colors[k] != a.colors[k]) return false;
    }
    return true;
}

// ¿Están todos los adjuntos de a en b, en el mismo sitio y con la misma vista, con la misma profundidad? (merged_passes:
// unos targets que caben en el pase abierto sin cambiarlo.) a sin ningún adjunto, no.
bool targets_within(const cmd::SetRenderTargets& a, const cmd::SetRenderTargets& b) {
    if (a.depth != b.depth) return false;
    if (a.depth && !same_attachment_view(*a.depth, a.depth_view, b.depth_view)) return false;
    bool any = a.depth != nullptr;
    for (int k = 0; k < a.count; k++) {
        if (!a.colors[k]) continue;
        if (k >= b.count || b.colors[k] != a.colors[k] || !same_attachment_view(*a.colors[k], a.color_views[k], b.color_views[k]))
            return false;
        any = true;
    }
    return any;
}

// Los adjuntos de a y b juntos (targets_fit(a, b)): en cada sitio, el de a o, si a no tiene, el de b.
cmd::SetRenderTargets targets_union(const cmd::SetRenderTargets& a, const cmd::SetRenderTargets& b) {
    cmd::SetRenderTargets u = a;
    u.count = std::max(a.count, b.count);
    for (int k = 0; k < u.count; k++)
        if (k >= a.count || !a.colors[k]) {
            u.colors[k] = k < b.count ? b.colors[k] : nullptr;
            u.color_views[k] = k < b.count ? b.color_views[k] : TextureViewDesc{};
        }
    return u;
}

// Carga y guardado de cada adjunto de un render pass (por defecto LOAD y STORE: lo de siempre). No cuentan para la
// compatibilidad de render passes: los pipelines y framebuffers se crean con el de LOAD/STORE y valen para todos.
struct PassOps {
    VkAttachmentLoadOp color_load[8];
    VkAttachmentStoreOp color_store[8];
    VkAttachmentLoadOp depth_load, stencil_load;
    VkAttachmentStoreOp depth_store, stencil_store;
};
PassOps default_pass_ops() {
    PassOps o;
    for (int k = 0; k < 8; k++) o.color_load[k] = VK_ATTACHMENT_LOAD_OP_LOAD, o.color_store[k] = VK_ATTACHMENT_STORE_OP_STORE;
    o.depth_load = o.stencil_load = VK_ATTACHMENT_LOAD_OP_LOAD;
    o.depth_store = o.stencil_store = VK_ATTACHMENT_STORE_OP_STORE;
    return o;
}

// Intersección de dos rectángulos (vacía: tamaño 0 en el origen de b).
VkRect2D intersect(const VkRect2D& a, const VkRect2D& b) {
    const int64_t x0 = std::max<int64_t>(a.offset.x, b.offset.x), y0 = std::max<int64_t>(a.offset.y, b.offset.y);
    const int64_t x1 = std::min<int64_t>(int64_t{a.offset.x} + a.extent.width, int64_t{b.offset.x} + b.extent.width);
    const int64_t y1 = std::min<int64_t>(int64_t{a.offset.y} + a.extent.height, int64_t{b.offset.y} + b.extent.height);
    if (x1 <= x0 || y1 <= y0) return {b.offset, {0, 0}};
    return {{static_cast<int32_t>(x0), static_cast<int32_t>(y0)}, {static_cast<uint32_t>(x1 - x0), static_cast<uint32_t>(y1 - y0)}};
}

VkRect2D scissor_rect(const cmd::Scissor& v) {
    const int32_t x = std::max(v.x, 0), y = std::max(v.y, 0);
    const int64_t right = int64_t{v.x} + v.w, bottom = int64_t{v.y} + v.h;
    return {{x, y}, {static_cast<uint32_t>(std::clamp<int64_t>(right, x, INT32_MAX) - x),
                     static_cast<uint32_t>(std::clamp<int64_t>(bottom, y, INT32_MAX) - y)}};
}

uint32_t find_memory(uint32_t bits, VkMemoryPropertyFlags props) {
    const VkPhysicalDeviceMemoryProperties& m = vk()->memory;
    for (uint32_t k = 0; k < m.memoryTypeCount; k++)
        if ((bits & (1u << k)) && (m.memoryTypes[k].propertyFlags & props) == props) return k;
    return UINT32_MAX;
}

// Huellas muestreadas (content_hash, range_hash de más de 16 KB): cuánto antes se piden las muestras que vienen. 64 KB son
// 16 páginas de una textura o 32 muestras de un buffer: unos 16 fallos en vuelo (el A77 admite unos 20 en L1).
constexpr size_t kPrefetchAhead = 64 << 10;
bool prefetch_hashes() { return !NVN_FLAG("no_print_prefetch"); }

// Huella de la memoria de una textura: entera si es pequeña; si no, 64 bytes de cada página de 4 KB
// (los frames de vídeo cambian enteros; las texturas estáticas no cambian).
// Cuatro cadenas independientes, como range_hash: los mismos bytes, sin esperar cada multiplicación.
uint64_t content_hash(const Texture& t) {
    constexpr uint64_t kP1 = 0x9E3779B185EBCA87ULL, kP2 = 0xC2B2AE3D27D4EB4FULL;
    const size_t size = t.layout.total_size();
    const uint8_t* p = t.storage;
    uint64_t v[4] = {size ^ kP1, size + kP2, size, size - kP1};
    auto round = [](uint64_t acc, uint64_t w) {
        acc += w * kP2;
        acc = (acc << 31) | (acc >> 33);
        return acc * kP1;
    };
    auto mix = [&](const uint8_t* q, size_t n) {
        size_t k = 0;
        for (; k + 32 <= n; k += 32) {
            uint64_t w[4];
            memcpy(w, q + k, 32);
            v[0] = round(v[0], w[0]), v[1] = round(v[1], w[1]), v[2] = round(v[2], w[2]), v[3] = round(v[3], w[3]);
        }
        for (; k + 8 <= n; k += 8) {
            uint64_t w;
            memcpy(&w, q + k, 8);
            v[0] = round(v[0], w);
        }
    };
    if (size <= 64 * 1024) {
        mix(p, size);
    } else {
        // Cada muestra es otra página: un fallo de TLB y otro de caché, que la CPU apenas solapa (cada vuelta espera a la
        // cadena de la huella). Pedida kPrefetchAhead antes (PRFM no falla aunque la dirección no valga), la espera se
        // solapa con las siguientes. Mismos bytes y misma huella. <datos>/no_print_prefetch: como antes.
        const bool prefetch = prefetch_hashes();
        if (prefetch)
            for (size_t off = 0; off < kPrefetchAhead && off + 64 <= size; off += 4096) __builtin_prefetch(p + off);
        for (size_t off = 0; off + 64 <= size; off += 4096) {
            if (prefetch && off + kPrefetchAhead + 64 <= size) __builtin_prefetch(p + off + kPrefetchAhead);
            mix(p + off, 64);
        }
        mix(p + size - 64, 64);
    }
    uint64_t h = ((v[0] << 1) | (v[0] >> 63)) + ((v[1] << 7) | (v[1] >> 57)) + ((v[2] << 12) | (v[2] >> 52)) +
                 ((v[3] << 18) | (v[3] >> 46));
    h ^= h >> 33;
    h *= kP2;
    return h ^ (h >> 29);
}

// Claves de las cachés (pipelines, vistas, layouts): 8 bytes por paso y mezcla final. Byte a byte, la
// clave de pipeline (casi 400 bytes) costaba ~0,3 ms por frame en la fábrica.
uint64_t hash_bytes(const void* data, size_t size, uint64_t h = 1469598103934665603ULL) {
    auto* p = static_cast<const uint8_t*>(data);
    size_t k = 0;
    for (; k + 8 <= size; k += 8) {
        uint64_t w;
        memcpy(&w, p + k, 8);
        h = (h ^ w) * 0x9E3779B97F4A7C15ULL;
        h ^= h >> 29;
    }
    for (; k < size; k++) h = (h ^ p[k]) * 1099511628211ULL;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    return h ^ (h >> 33);
}

template <typename T>
std::string key_of(const T& v) {
    return std::string(reinterpret_cast<const char*>(&v), sizeof v);
}

// Claves de caché de tamaño fijo (sin construir strings en cada draw): se comparan y se hashean como
// bytes, así que se crean a cero antes de rellenarlas (el relleno entre campos también cuenta).
template <typename T>
struct BytesKey {
    T value;
    BytesKey() { memset(&value, 0, sizeof value); }
    bool operator==(const BytesKey& o) const { return memcmp(&value, &o.value, sizeof value) == 0; }
};
template <typename T>
struct BytesKeyHash {
    size_t operator()(const BytesKey<T>& k) const { return static_cast<size_t>(hash_bytes(&k.value, sizeof k.value)); }
};
struct ViewKeyData {
    uint64_t texture;  // Texture::serial
    ImageViewSpec spec;
};
struct LayoutKeyData {
    uint64_t ids[6];  // ProgramStage::spirv_id de cada etapa (0: sin ella): el layout solo depende de lo traducido
};

// --- Prioridad de NvnSubmit ---
// Al cargar, los hilos de carga del juego (ResourceManager, FileReader, CompressedFileR, JobThreads…) pueden compartir los
// núcleos grandes con NvnSubmit a la misma prioridad (nice 0, ninguno la cambia). El mismo envío del selector de personaje
// costó 627 ms en una partida y 1833 ms en otra (4-11: los mismos 47 MB de texturas, 108 MB de espejos nuevos, 21700
// mprotect y ~17000 draws): lo que varía es cuánto núcleo le toca. Con nice -10 el planificador le da ~9 veces el tiempo
// de un hilo de nice 0 en el mismo núcleo; dormido (esperando trabajo) no quita nada. El init.rc de AOSP pone RLIMIT_NICE
// 40 (hasta -20 sin permisos); si aquí no deja, sale en el log. <datos>/no_submit_priority: nice 0 como antes; se mira
// antes de cada envío (en vivo). Los hilos del deswizzle en paralelo van con la misma (son parte del envío; esos la toman
// al arrancar).
constexpr int kSubmitNice = -10;
int submit_nice() { return NVN_FLAG("no_submit_priority") ? 0 : kSubmitNice; }
void set_thread_nice(int nice, const char* who) {
    if (setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), nice) == 0) {
        NVN_LOG("prioridad de %s: nice %d\n", who, nice);
        return;
    }
    static std::atomic<int> failed{0};
    if (failed.fetch_add(1) < 4) NVN_LOG("prioridad de %s: no se pudo poner nice %d (%s)\n", who, nice, strerror(errno));
}

// --- Deswizzle en paralelo ---
// Un nivel grande se reparte por trozos de filas (~64 KB de salida cada uno) entre dos hilos propios y el que sube (el de
// NvnSubmit), que también coge trozos: si un hilo cae en un núcleo pequeño, solo se espera su último trozo. Al cargar se
// suben decenas de MB en un envío (47 MB al entrar en el selector de personaje, ~175 MB hasta la aldea) con el juego
// esperando a su sync. Vuelve con todo copiado (el recuento de bloques a cero decide el primer nivel muestreado de la
// textura antes de crear sus vistas). <datos>/single_thread_upload: todo en el hilo del envío, como antes.
constexpr VkDeviceSize kParallelDeswizzleMin = 256 << 10;
class RowPool {
public:
    void run(size_t rows, size_t chunk, const std::function<void(size_t, size_t)>& fn) {
        start();
        {
            std::lock_guard lock(m_);
            job_ = &fn;
            rows_ = rows;
            chunk_ = chunk;
            next_.store(0, std::memory_order_relaxed);
            gen_++;
        }
        cv_.notify_all();
        work(fn, rows, chunk);
        std::unique_lock lock(m_);
        job_ = nullptr;  // los que despierten tarde ya no entran
        done_cv_.wait(lock, [&] { return active_ == 0; });
    }

private:
    void work(const std::function<void(size_t, size_t)>& fn, size_t rows, size_t chunk) {
        for (;;) {
            const size_t r = next_.fetch_add(chunk, std::memory_order_relaxed);
            if (r >= rows) return;
            fn(r, std::min(rows, r + chunk));
        }
    }
    void start() {  // solo desde el hilo del envío (con g_mutex)
        if (started_) return;
        started_ = true;
        for (int k = 0; k < kThreads; k++) std::thread([this] { loop(); }).detach();  // viven lo que el proceso
    }
    void loop() {
        pthread_setname_np(pthread_self(), "Deswizzle");
        if (const int nice = submit_nice()) set_thread_nice(nice, "Deswizzle");  // la de NvnSubmit (ver kSubmitNice)
        uint64_t seen = 0;
        std::unique_lock lock(m_);
        for (;;) {
            cv_.wait(lock, [&] { return job_ && gen_ != seen; });
            seen = gen_;
            const std::function<void(size_t, size_t)>* fn = job_;
            const size_t rows = rows_, chunk = chunk_;
            active_++;
            lock.unlock();
            work(*fn, rows, chunk);
            lock.lock();
            if (--active_ == 0) done_cv_.notify_all();
        }
    }
    static constexpr int kThreads = 2;
    std::mutex m_;
    std::condition_variable cv_, done_cv_;
    const std::function<void(size_t, size_t)>* job_ = nullptr;
    size_t rows_ = 0, chunk_ = 1;
    std::atomic<size_t> next_{0};
    uint64_t gen_ = 0;
    int active_ = 0;
    bool started_ = false;
};
RowPool& row_pool() {
    static RowPool* pool = new RowPool;  // sin destructor: sus hilos siguen hasta que muere el proceso
    return *pool;
}

// --- Staging: trozos host-visible que se reciclan cuando la GPU termina ---

// Perfil de cargas: trozos de staging nuevos (vkCreateBuffer + vkAllocateMemory + vkMapMemory) y los que vuelven de la
// reserva común. Globales: Staging no conoce al ejecutor.
std::atomic<uint64_t> g_staging_new_ns{0}, g_staging_new{0}, g_staging_new_bytes{0}, g_staging_reused{0};

class Staging {
public:
    struct Alloc {
        VkBuffer buffer;
        VkDeviceSize offset;
        uint8_t* data;
    };
    // Trozos de `chunk` bytes (o más grandes, para una reserva que no cabe en uno).
    explicit Staging(VkDeviceSize chunk = kStagingChunk, bool readback = false) : chunk_(chunk), readback_(readback) {}
    VkDeviceSize bytes() const {  // perfil
        VkDeviceSize total = 0;
        for (const Chunk& c : chunks_) total += c.size;
        return total;
    }
    VkDeviceSize used() const {  // lo reservado desde el último reset() (bytes() es la capacidad: no baja)
        VkDeviceSize total = 0;
        for (const Chunk& c : chunks_) total += c.used;
        return total;
    }
    size_t used_chunks() const {  // los que tienen algo desde el último reset()
        size_t n = 0;
        for (const Chunk& c : chunks_) n += c.used > 0;
        return n;
    }
    // Más de kStagingChunk no cabe: nullptr (antes se recortaba y quien copiaba el tamaño pedido escribía fuera).
    Alloc alloc(VkDeviceSize size, VkDeviceSize alignment) {
        if (size > kStagingChunk) return {VK_NULL_HANDLE, 0, nullptr};
        return alloc_contiguous(size, alignment);
    }
    // A validated image level or guest read window may need a larger dedicated chunk.
    Alloc alloc_contiguous(VkDeviceSize size, VkDeviceSize alignment) {
        for (;; current_++) {
            if (current_ == chunks_.size() && (failed_ || !grow(size))) return {VK_NULL_HANDLE, 0, nullptr};
            Chunk& c = chunks_[current_];
            VkDeviceSize offset = (c.used + alignment - 1) / alignment * alignment;
            if (offset + size <= c.size) {
                c.used = offset + size;
                return {c.buffer, offset, c.mapped + offset};
            }
        }
    }
    void reset() {
        for (Chunk& c : chunks_) c.used = 0;
        current_ = 0;
        failed_ = false;
    }
    // Libera los trozos de más (tras un pico, p. ej. subir texturas al cargar): con varios envíos en
    // vuelo, cada uno guardando sus picos, la memoria se disparaba hasta que Android mataba el juego.
    // Deja como mucho los `keep` primeros, y menos si pasan de `max_bytes` (sin bajar de uno).
    // Los trozos que suelta van a una reserva común (hasta kFreeStagingBytes) de la que vuelve a tirar grow(): MH Rise
    // pasa de 16 MB por envío en cada frame mientras carga y, liberándolos y pidiéndolos otra vez a Vulkan cada vez
    // (cada uno es un ioctl de kgsl y un munmap), NvnSubmit gastaba un tercio del tiempo en eso.
    void trim(size_t keep, VkDeviceSize max_bytes = ~VkDeviceSize{0}) {
        while (chunks_.size() > keep || (chunks_.size() > 1 && bytes() > max_bytes)) {
            release(chunks_.back());
            chunks_.pop_back();
        }
        current_ = std::min(current_, chunks_.empty() ? size_t{0} : chunks_.size() - 1);
    }

private:
    struct Chunk {
        VkBuffer buffer;
        VkDeviceMemory memory;
        uint8_t* mapped;
        VkDeviceSize used, size;
        bool readback;
    };
    struct FreeChunks {
        std::mutex m;
        std::vector<Chunk> chunks;
        VkDeviceSize bytes = 0;
    };
    static FreeChunks& free_chunks() {
        static FreeChunks f;
        return f;
    }
    static void release(const Chunk& ch) {
        FreeChunks& f = free_chunks();
        {
            std::lock_guard lock(f.m);
            if (f.bytes + ch.size <= (low_memory() ? kFreeStagingBytes / 2 : kFreeStagingBytes)) {
                f.chunks.push_back(ch);
                f.bytes += ch.size;
                return;
            }
        }
        VkContext* c = vk();
        vkDestroyBuffer(c->device, ch.buffer, nullptr);
        vkFreeMemory(c->device, ch.memory, nullptr);
    }
    bool grow(VkDeviceSize need) {
        VkContext* c = vk();
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = (std::max(need, chunk_) + chunk_ - 1) / chunk_ * chunk_;
        {
            FreeChunks& f = free_chunks();
            std::lock_guard lock(f.m);
            for (size_t k = 0; k < f.chunks.size(); k++) {
                if (f.chunks[k].size != bi.size || f.chunks[k].readback != readback_) continue;
                Chunk reused = f.chunks[k];
                f.chunks.erase(f.chunks.begin() + static_cast<ptrdiff_t>(k));
                f.bytes -= reused.size;
                reused.used = 0;
                chunks_.push_back(reused);
                g_staging_reused.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
        }
        bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        const auto grow_t0 = std::chrono::steady_clock::now();
        struct Count {  // perfil de cargas, también si falla
            std::chrono::steady_clock::time_point t0;
            VkDeviceSize bytes;
            ~Count() {
                g_staging_new_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(),
                                           std::memory_order_relaxed);
                g_staging_new.fetch_add(1, std::memory_order_relaxed);
                g_staging_new_bytes.fetch_add(bytes, std::memory_order_relaxed);
            }
        } count{grow_t0, bi.size};
        Chunk chunk{};
        if (vkCreateBuffer(c->device, &bi, nullptr, &chunk.buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(c->device, chunk.buffer, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        const auto coherent = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        const uint32_t normal_type = find_memory(req.memoryTypeBits, coherent);
        ai.memoryTypeIndex = readback_ ? find_memory(req.memoryTypeBits, coherent | VK_MEMORY_PROPERTY_HOST_CACHED_BIT) : normal_type;
        if (ai.memoryTypeIndex == UINT32_MAX) ai.memoryTypeIndex = normal_type;
        chunk.readback = readback_;
        // Sin RAM, cada alloc siguiente volvía a intentarlo y dejaba un VkBuffer perdido: se desiste hasta reset().
        VkResult allocated = ai.memoryTypeIndex != UINT32_MAX ? vkAllocateMemory(c->device, &ai, nullptr, &chunk.memory)
                                                             : VK_ERROR_FEATURE_NOT_PRESENT;
        if ((allocated == VK_ERROR_OUT_OF_DEVICE_MEMORY || allocated == VK_ERROR_OUT_OF_HOST_MEMORY) &&
            normal_type != UINT32_MAX && normal_type != ai.memoryTypeIndex) {
            ai.memoryTypeIndex = normal_type;
            allocated = vkAllocateMemory(c->device, &ai, nullptr, &chunk.memory);
        }
        if (allocated != VK_SUCCESS) {
            vkDestroyBuffer(c->device, chunk.buffer, nullptr);
            failed_ = true;
            return false;
        }
        vkBindBufferMemory(c->device, chunk.buffer, chunk.memory, 0);
        if (vkMapMemory(c->device, chunk.memory, 0, bi.size, 0, reinterpret_cast<void**>(&chunk.mapped)) != VK_SUCCESS) {
            vkDestroyBuffer(c->device, chunk.buffer, nullptr);
            vkFreeMemory(c->device, chunk.memory, nullptr);
            failed_ = true;
            return false;
        }
        chunk.size = bi.size;
        chunks_.push_back(chunk);
        return true;
    }
    VkDeviceSize chunk_;
    bool readback_;
    std::vector<Chunk> chunks_;
    size_t current_ = 0;
    bool failed_ = false;  // grow() falló desde el último reset()

public:
    bool failed() const { return failed_; }
};

// --- Caché de vértices e índices ---
// Copiar cada vertex buffer y recorrer todos los índices en cada draw costaba ~25 ms por frame en el
// prólogo. La mayoría son mallas estáticas: cada rango se copia una vez a un arena host-visible y se
// reutiliza mientras su huella no cambie (se comprueba una vez por envío: el juego no puede tocar
// memoria que usa un envío pendiente). Lo que cambia a menudo (UI, partículas) se marca dinámico y
// vuelve al camino de siempre (staging por draw). Las entradas no se reescriben: un cambio ocupa sitio
// nuevo, así un draw anterior del mismo envío sigue leyendo lo suyo. Si el arena se llena, se empieza
// otra generación (otro arena) y lo de la vieja se vuelve a copiar al usarse; la vieja se recicla cuando
// la GPU acaba su último envío. Con <datos>/no_generations, como antes: se vacía al terminar un envío,
// con la GPU parada.

// Cuatro cadenas independientes (como xxh64): la CPU las solapa en vez de esperar cada multiplicación. Mira los
// mismos bytes que antes; se calcula cada envío para cada malla cacheada (~1 ms por frame en la fábrica).
uint64_t range_hash(const uint8_t* p, size_t size) {
    constexpr uint64_t kP1 = 0x9E3779B185EBCA87ULL, kP2 = 0xC2B2AE3D27D4EB4FULL;
    uint64_t v[4] = {size ^ kP1, size + kP2, size, size - kP1};
    auto round = [](uint64_t acc, uint64_t w) {
        acc += w * kP2;
        acc = (acc << 31) | (acc >> 33);
        return acc * kP1;
    };
    uint64_t tail = 1469598103934665603ULL;
    auto mix = [&](const uint8_t* q, size_t n) {
        size_t k = 0;
        for (; k + 32 <= n; k += 32) {
            uint64_t w[4];
            memcpy(w, q + k, 32);
            v[0] = round(v[0], w[0]), v[1] = round(v[1], w[1]), v[2] = round(v[2], w[2]), v[3] = round(v[3], w[3]);
        }
        for (; k + 8 <= n; k += 8) {
            uint64_t w;
            memcpy(&w, q + k, 8);
            v[0] = round(v[0], w);
        }
        for (; k < n; k++) tail = (tail ^ q[k]) * 1099511628211ULL;  // la cola (rangos de 4 bytes: contadores)
    };
    if (size <= 16 * 1024) {
        mix(p, size);
    } else {
        // Como content_hash: las muestras de más adelante, pedidas antes (las dos líneas: la dirección no va alineada).
        const bool prefetch = prefetch_hashes();
        if (prefetch)
            for (size_t off = 0; off < kPrefetchAhead && off + 64 <= size; off += 2048)
                __builtin_prefetch(p + off), __builtin_prefetch(p + off + 63);
        for (size_t off = 0; off + 64 <= size; off += 2048) {
            if (prefetch && off + kPrefetchAhead + 64 <= size)
                __builtin_prefetch(p + off + kPrefetchAhead), __builtin_prefetch(p + off + kPrefetchAhead + 63);
            mix(p + off, 64);
        }
        mix(p + size - 256, 256);
    }
    uint64_t h = ((v[0] << 1) | (v[0] >> 63)) + ((v[1] << 7) | (v[1] >> 57)) + ((v[2] << 12) | (v[2] >> 52)) +
                 ((v[3] << 18) | (v[3] >> 46));
    h ^= tail;
    h ^= h >> 33;
    h *= kP2;
    return h ^ (h >> 29);
}

// Huella de todos los bytes (range_hash solo mira 64 de cada 2 KB pasados 16 KB): por trozos de 16 KB, que range_hash
// mira enteros. 1 MB: 0,07 ms medido en un Mac M, ~0,2 ms estimado en la Flip; solo para lo que una escritura exacta
// marcó (Texture::level_full_hash).
constexpr size_t kDigestChunk = 16 * 1024;
uint64_t full_hash(const uint8_t* p, size_t size, std::vector<uint64_t>* chunks = nullptr) {
    uint64_t h = 0x9E3779B97F4A7C15ULL ^ size;
    if (chunks) chunks->resize((size + kDigestChunk - 1) / kDigestChunk);
    for (size_t at = 0; at < size; at += kDigestChunk) {
        const uint64_t chunk = range_hash(p + at, std::min(kDigestChunk, size - at));
        if (chunks) (*chunks)[at / kDigestChunk] = chunk;
        h = (h ^ chunk) * 0xff51afd7ed558ccdULL;
    }
    return h ^ (h >> 33);
}

// Layer-major subresource order, shared by upload and the full-content checks.
uint32_t upload_levels(const Texture& t) {
    return std::min<uint32_t>(static_cast<uint32_t>(t.layout.levels.size()), static_cast<uint32_t>(std::max(t.desc.levels, 1)));
}
uint32_t upload_layers(const Texture& t) { return std::min(t.layout.layers, t.layers); }

// ¿Se vigila la memoria de esta textura (Texture::watched)? Las que sube la CPU sin comprimir (el atlas de glifos, LUTs,
// lo que se pinta en la CPU), de más de 64 KB (hasta ahí content_hash ya las mira enteras), que caben en el staging, no
// sparse (el streaming llega por MapVirtual y por niveles enteros, que la huella sí ve) y sin no_watch. Las de un nivel
// desde su primera subida; las demás cuando ya han cambiado una vez.
bool watch_candidate(const Texture& t, bool changed_before) {
    const size_t size = t.layout.total_size();
    if (t.gpu_levels || t.no_watch || !t.storage || (t.desc.flags & 0x20) || size <= 64 * 1024 || size > kStagingChunk)
        return false;
    if (t.layout.block_w != 1 || t.layout.block_h != 1) return false;
    return changed_before || std::max(t.desc.levels, 1) == 1;
}

class BufferCache {
public:
    struct Entry {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        uint64_t hash = 0, checked = 0;
        uint32_t changes = 0, min_index = 0, max_index = 0;  // índices: el menor y el mayor (sin el de reinicio)
        bool dynamic = false;
        // <datos>/watch_vertices: revisiones seguidas sin cambios; pasadas kWatchAfter (y de más de kWatchMin), sus
        // páginas se vigilan (nnsdk/watch.h) y la revisión de cada envío es mirar si alguna se escribió, sin huella.
        uint8_t stable = 0;
        bool watched = false;
        std::vector<uint64_t> watch_keys;
    };
    // Vigilancia (en vivo, con watch_vertices; apagada por defecto): las mallas estáticas se revisaban con su huella en
    // cada frame (los streams enteros de los multi-draw: ~750 rangos y 4 MB muestreados por frame en Kamura); vigiladas,
    // la revisión es recorrer sus claves de página. Solo lo grande (lo pequeño se mira entero y barato) y lo que ya se vio
    // igual dos veces (lo dinámico nunca llega: cada escritura de la CPU sería un SIGSEGV y un mprotect).
    static constexpr size_t kWatchMin = 16 * 1024;
    static constexpr uint8_t kWatchAfter = 2;
    bool watch = false;
    // RAM: el Adreno no tiene memoria propia. Tiene que caber lo que se dibuja en un frame: con 128 MB, en la
    // fábrica se vaciaba decenas de veces por frame (todo se volvía a copiar: 1 fps).
    static constexpr VkDeviceSize kArenaLimit = 384ull << 20;  // sin generaciones
    // Con generaciones hay dos arenas que se turnan, de 256 MB: como mucho 512 MB (antes 384 fijos) y sin
    // reservar ni soltar memoria en cada cambio (cada arena guarda sus trozos para su siguiente turno). El viejo
    // solo espera a la GPU unos pocos envíos (nada nuevo lo enlaza), así que dos bastan.
    static constexpr VkDeviceSize kGenerationLimit = 256ull << 20;
    // Lleno, lo nuevo se copia en cada draw (al staging) hasta el cambio de generación (o el vaciado): como mucho
    // uno cada tantos frames, para que lo que no cabe no tire en cada frame lo que sí cabe. Sin generaciones el
    // vaciado solo puede hacerse con la GPU parada: el envío en curso se parte (al cargar la fábrica un solo envío
    // llegó a 3,4 GB).
    static constexpr uint64_t kResetEvery = 30;
    static constexpr uint32_t kDynamicAfter = 3;  // cambios vistos antes de dejar de cachear
    // Perfil fino (lo pone el ejecutor en los frames medidos, nulo en los demás): búsquedas de get() y sus huellas.
    struct Prof {
        uint64_t lookups = 0, hashes = 0, hash_bytes = 0, hash_ns = 0, watched = 0;
    };
    Prof* prof = nullptr;

    // Devuelve la entrada vigente para [address, address+size), o nullptr si el rango es dinámico o
    // no cabe. index_type >= 0: índices (los de 8 bits se amplían a 16) y se guarda el índice máximo.
    const Entry* get(uint64_t address, size_t size, int index_type, uint64_t submit) {
        if (!address || !size || size > kStagingChunk) return nullptr;
        if (prof) prof->lookups++;
        Entry& e = entry(address, (uint64_t{static_cast<uint32_t>(index_type + 1)} << 56) | size);
        if (e.dynamic) return nullptr;
        if (e.buffer && e.checked == submit) return &e;
        if (e.buffer && e.watched && watch) {  // vigilada: sin páginas escritas desde la última revisión, sigue igual
            watch_runs_.clear();
            nnsdk::watch::collect(address, size, e.watch_keys, &watch_runs_);
            if (watch_runs_.empty()) {
                e.checked = submit;
                if (prof) prof->watched++;
                return &e;
            }
        } else if (e.buffer && !e.watched && watch && e.stable >= kWatchAfter && size > kWatchMin &&
                   nnsdk::watch::watchable(address, size)) {
            // Se empieza a vigilar ANTES de la huella de abajo: lo que la CPU escriba desde aquí salta, y lo de antes lo ve
            // la huella. (Al revés, una escritura entre la huella y el primer collect no se vería nunca.)
            e.watch_keys.clear();
            nnsdk::watch::collect(address, size, e.watch_keys, nullptr);
            e.watched = true;
        }
        const auto* src = reinterpret_cast<const uint8_t*>(address);
        const auto hash_t0 = prof ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        uint64_t h = range_hash(src, size);
        if (prof) {
            prof->hashes++;
            prof->hash_bytes += size;
            prof->hash_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - hash_t0).count();
        }
        e.checked = submit;
        if (e.buffer && h == e.hash) {
            if (e.stable < 255) e.stable++;
            return &e;
        }
        e.stable = 0;
        e.watched = false;  // cambió: sin vigilar hasta que vuelva a verse igual
        if (e.buffer && ++e.changes >= kDynamicAfter) {
            e.dynamic = true;
            return nullptr;
        }
        const size_t out = index_type == 0 ? size * 2 : size;
        // Sin sitio (hasta el siguiente vaciado copia cada draw): la copia vieja ya no vale para este envío.
        Generation& g = gens_[active_];
        Staging::Alloc a = g.used + out <= limit_ ? g.arena.alloc(out, 16) : Staging::Alloc{};
        if (!a.data) {
            full_ = true;
            e.buffer = VK_NULL_HANDLE;
            return nullptr;
        }
        g.used += out;
        if (index_type == 0)
            for (size_t k = 0; k < size; k++) reinterpret_cast<uint16_t*>(a.data)[k] = src[k];
        else
            memcpy(a.data, src, size);
        if (index_type >= 0) index_range(src, size, index_type, &e.min_index, &e.max_index);
        e.buffer = a.buffer;
        e.offset = a.offset;
        e.hash = h;
        return &e;
    }
    bool needs_reset(uint64_t frame) const { return full_ && frame >= reset_frame_ + kResetEvery; }
    VkDeviceSize bytes() const { return gens_[0].arena.bytes() + gens_[1].arena.bytes(); }  // perfil
    void set_generations(bool on) { limit_ = on ? kGenerationLimit : kArenaLimit; }
    // Con generaciones, entre draws (nadie guarda punteros a entradas): si toca, lo de ahora se queda para la GPU
    // hasta el envío `last_submit` (el que se está grabando) y se sigue en la otra generación, vacía. Las entradas
    // se olvidan (también lo dinámico, como al vaciar) y se vuelven a copiar al usarse. false si la otra
    // generación aún está en vuelo.
    bool next_generation(uint64_t frame, uint64_t last_submit) {
        if (!needs_reset(frame)) return false;
        Generation& next = gens_[active_ ^ 1];
        if (next.in_flight) return false;
        Generation& old = gens_[active_];
        static int logged = 0;
        if (logged++ < 20 || logged % 100 == 0)
            NVN_LOG("caché de vértices: generación nueva (%zu entradas, %llu MB, %llu frames desde la anterior)\n",
                    entries_.size(), static_cast<unsigned long long>(old.used >> 20),
                    static_cast<unsigned long long>(frame - reset_frame_));
        old.in_flight = true;
        old.last_submit = last_submit;
        active_ ^= 1;
        next.arena.reset();
        next.used = 0;
        entries_.clear();
        front_epoch_++;  // las entradas del frente apuntaban a las de entries_
        full_ = false;
        reset_frame_ = frame;
        return true;
    }
    // La generación vieja vuelve a estar libre cuando la GPU ha terminado su último envío (completed: el último
    // envío terminado). Guarda sus trozos para su siguiente turno; sin generaciones, los suelta.
    void reap(uint64_t completed, bool generations) {
        Generation& other = gens_[active_ ^ 1];
        if (other.in_flight && other.last_submit <= completed) {
            other.in_flight = false;
            other.arena.reset();
            other.arena.trim(~size_t{0}, limit_ + 2 * kStagingChunk);  // lo de más, de cuando no había generaciones
            other.used = 0;
        }
        if (!generations && !other.in_flight && other.arena.bytes()) other.arena.trim(0);
    }
    // Sin generaciones, con la GPU parada: si toca, se empieza de cero (y se olvida lo dinámico).
    void maybe_reset(uint64_t frame) {
        if (!needs_reset(frame)) return;
        static int logged = 0;
        if (logged++ < 20 || logged % 100 == 0)
            NVN_LOG("caché de vértices vaciada (%zu entradas, %llu frames desde la anterior)\n", entries_.size(),
                    static_cast<unsigned long long>(frame - reset_frame_));
        gens_[active_].arena.reset();
        gens_[active_].used = 0;
        Generation& other = gens_[active_ ^ 1];  // de cuando había generaciones: la GPU tampoco la lee ya
        other.in_flight = false;
        other.arena.trim(0);
        other.used = 0;
        entries_.clear();
        front_epoch_++;
        full_ = false;
        reset_frame_ = frame;
    }
    // Menor y mayor índice (sin el de reinicio de primitiva); sin ninguno, lo > hi.
    static void index_range(const uint8_t* p, size_t size, int type, uint32_t* lo, uint32_t* hi) {
        uint32_t a = ~0u, b = 0;
        if (type == 0) {
            for (size_t k = 0; k < size; k++) a = std::min<uint32_t>(a, p[k]), b = std::max<uint32_t>(b, p[k]);
        } else if (type == 1) {
            const auto* q = reinterpret_cast<const uint16_t*>(p);
            for (size_t k = 0; k < size / 2; k++)
                if (q[k] != 0xFFFF) a = std::min<uint32_t>(a, q[k]), b = std::max<uint32_t>(b, q[k]);
        } else {
            const auto* q = reinterpret_cast<const uint32_t*>(p);
            for (size_t k = 0; k < size / 4; k++)
                if (q[k] != 0xFFFFFFFF) a = std::min(a, q[k]), b = std::max(b, q[k]);
        }
        *lo = a;
        *hi = b;
    }

private:
    struct KeyHash {
        size_t operator()(const std::pair<uint64_t, uint64_t>& k) const { return k.first * 31 ^ k.second; }
    };
    std::unordered_map<std::pair<uint64_t, uint64_t>, Entry, KeyHash> entries_;
    // Frente de entries_: una tabla directa pequeña (cabe en la L2) con la entrada de las últimas claves. Cada stream entero de
    // un multi-draw se busca varias veces por frame (sombras, pre-pase, color), y cada búsqueda en el unordered_map era una
    // división por el número de cubetas y dos o tres fallos de caché (cubeta, nodo anterior, nodo). Las entradas de
    // entries_ no se mueven mientras no se borren (solo clear(), que cambia front_epoch_). front = false
    // (<datos>/no_vertex_front, en vivo): siempre por el mapa, como antes.
    struct Front {
        uint64_t address = 0, key = 0, epoch = 0;
        Entry* entry = nullptr;
    };
    static constexpr size_t kFront = 2048;
    std::vector<Front> front_ = std::vector<Front>(kFront);
    uint64_t front_epoch_ = 1;
    Entry& entry(uint64_t address, uint64_t key) {
        if (!front) return entries_[{address, key}];
        Front& f = front_[static_cast<size_t>(((address ^ key * 0x9E3779B97F4A7C15ULL) * 0xff51afd7ed558ccdULL) >> 53) & (kFront - 1)];
        if (f.epoch == front_epoch_ && f.address == address && f.key == key) return *f.entry;
        Entry& e = entries_[{address, key}];
        f = {address, key, front_epoch_, &e};
        return e;
    }

public:
    bool front = true;

private:
    std::vector<std::pair<size_t, size_t>> watch_runs_;  // de collect (sin reservas por revisión)
    struct Generation {
        Staging arena;
        VkDeviceSize used = 0;
        uint64_t last_submit = 0;  // en vuelo: el último envío que puede leerla
        bool in_flight = false;    // ya no es la activa y la GPU aún puede leerla
    };
    Generation gens_[2];
    int active_ = 0;
    VkDeviceSize limit_ = kGenerationLimit;
    bool full_ = false;  // se rechazó una copia por falta de sitio
    uint64_t reset_frame_ = 0;
};



// <datos>/pipelines_estado.bin: una por pipeline creada (en el orden en que salió la primera vez), para crearlas al
// arrancar la partida siguiente en otro hilo (Executor::warm_up). Las etapas van por su clave de la caché de SPIR-V:
// base, la de la etapa (su layout); module, la del módulo que entra en la pipeline (la variante, si la hay).
struct PipelineRecord {
    uint32_t magic;  // kRecordMagic
    uint32_t kind;   // 0 gráfica, 1 compute
    PipelineKey key;
    spirv_cache::Key base[kStages], module[kStages];
    uint64_t sum;    // FNV-1a de lo anterior
};
constexpr uint32_t kRecordMagic = 0x54534550;  // "PEST"
// Se guarda tal cual (bytes): sin punteros ni nada que no se pueda copiar con memcpy.
static_assert(std::is_trivially_copyable_v<PipelineRecord> && std::is_standard_layout_v<PipelineRecord>);
// Sube si cambia el significado de un campo de PipelineKey o cómo se crea una pipeline a partir de ella.
constexpr uint32_t kRecordFormat = 1;

uint64_t record_sum(const PipelineRecord& r) {
    const auto* p = reinterpret_cast<const uint8_t*>(&r);
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t k = 0; k < offsetof(PipelineRecord, sum); k++) h = (h ^ p[k]) * 0x100000001b3ULL;
    return h;
}

// DEPURACIÓN: quién escribe los cubemaps de la GPU (como render target, con una copia de textura o de buffer, o como
// imagen de un shader), con qué vista y qué shader. El cubo global de Kamura (256² R11G11B10F, 5 niveles, id 4608) sale
// gris en el volcado (R == G en el 100 % de los texels, B entre 0,91 y 1 veces R) y el volumen de GI 6016 que se hace con
// él también, casi; ningún frame trazado lo escribe. Si no sale en ninguna de estas, lo sube la CPU (datos del juego).
// Una línea por textura, forma y origen (la primera capa y nivel que se ven; 256 como mucho: las sombras de luces
// puntuales también son cubos y se escriben cada frame).
static void note_cube_write(const Texture& t, const TextureViewDesc& v, const char* how, const Texture* src,
                            uint64_t pixel_print, uint64_t frame) {
    static std::mutex m;
    static std::set<std::tuple<const void*, const char*, const void*>> seen;
    const int layer = (v.flags & kViewLayers) ? v.base_layer : 0, level = (v.flags & kViewLevels) ? v.base_level : 0;
    {
        std::lock_guard lock(m);
        if (seen.size() >= 256 || !seen.insert({t.storage, how, src ? src->storage : nullptr}).second) return;
    }
    const uint8_t* ts = t.desc.swizzle;
    NVN_LOG("cubo escrito (%s) en el frame %llu: %dx%d formato 0x%x niveles %d storage %p swizzle %d%d%d%d | vista capa %d "
            "nivel %d formato 0x%x flags 0x%x swizzle %d%d%d%d | pixel %016llx | origen %s %dx%dx%d target %d formato 0x%x "
            "swizzle %d%d%d%d\n",
            how, static_cast<unsigned long long>(frame), t.desc.width, t.desc.height, t.desc.format, t.desc.levels,
            static_cast<const void*>(t.storage), ts[0], ts[1], ts[2], ts[3], layer, level, v.format, v.flags, v.swizzle[0],
            v.swizzle[1], v.swizzle[2], v.swizzle[3], static_cast<unsigned long long>(pixel_print), src ? "textura" : "-",
            src ? src->desc.width : 0, src ? src->desc.height : 0, src ? src->desc.depth : 0, src ? src->desc.target : 0,
            src ? src->desc.format : 0, src ? src->desc.swizzle[0] : 0, src ? src->desc.swizzle[1] : 0,
            src ? src->desc.swizzle[2] : 0, src ? src->desc.swizzle[3] : 0);
}

struct Layout {
    VkDescriptorSetLayout set = VK_NULL_HANDLE;
    VkDescriptorSetLayout bindless = VK_NULL_HANDLE;  // set 1 (nvnshader::kBindlessSet) si hay arrays bindless; compartido
    VkPipelineLayout pipeline = VK_NULL_HANDLE;
    bool count_check = false;  // its vertex shader has the draw count check: set kCountCheckSet and push constants
    // Gráficos: UBO/SSBO dinámicos (el desplazamiento de cada draw va en el bind), así un draw igual que el
    // anterior salvo por dónde están sus datos reutiliza el set (en la fábrica, ~85 % de los draws).
    bool dynamic = false;
    uint64_t memo_epoch = 0;
    VkDescriptorSet memo_set = VK_NULL_HANDLE;
    std::vector<uint64_t> memo_key;
    // <datos>/templates: los descriptores del set 0 de los draws con vkUpdateDescriptorSetWithTemplate, en el orden en que
    // los escribe draw(); con la firma (binding y tipo de cada uno) que se comprueba en cada uso. Solo el hilo del ejecutor.
    VkDescriptorUpdateTemplate update_template = VK_NULL_HANDLE;
    std::vector<uint32_t> template_signature;
    bool template_failed = false;
};

std::atomic<uint64_t> g_game_gpu_serial{0};   // sube con cada frame_gpu_ns_ (game_gpu_frame)
std::atomic<uint64_t> g_last_frame_run_ns{0};  // el ejecutor del último frame presentado (last_frame_run_ns)
std::atomic<int> g_trace_request{0};          // request_trace: 1 ligera, 2 con los render targets

class Executor {
public:
    bool run(const std::vector<const CommandList*>& lists);
    uint64_t last_submit() const { return submitted_.load(std::memory_order_acquire); }
    // true si el envío ya terminó (o termina antes del plazo). Sin g_mutex: no espera a la reproducción en curso.
    bool wait_submit(uint64_t submit, uint64_t timeout_ns) {
        const bool done = wait_submit_fence(submit, timeout_ns);
        if (done) deliver_write_backs(submit);  // lo que lea la CPU del juego tras esta espera ya está en su memoria
        return done;
    }
    bool wait_submit_fence(uint64_t submit, uint64_t timeout_ns);
    // Devoluciones pendientes (ver kWriteBackMax): las copias de lectura de envíos ya grabados. Se entregan en
    // cuanto el envío termina, desde quien lo esperaba (el hilo del juego) o desde el ejecutor; con su propio mutex.
    struct PendingWriteBack {
        uint64_t submit, address, size;
        const uint8_t* data;
        uint64_t generation;  // la del overlay al grabar la copia
        uint8_t writers;
        bool whole = false;  // Overlay::cpu_reads: the GPU copy is all of it, so all of it goes back (no three-way mix)
    };
    std::mutex wb_mutex_;
    std::vector<PendingWriteBack> pending_wb_;
    // Copias de lectura aún sin entregar (su memoria). Cada overlay tiene una sola: grabar otra encima mientras la
    // anterior espera su entrega hacía que se entregara lo de un envío posterior (la GPU ya la estaba pisando: hay
    // hasta kInFlight envíos en vuelo) o una copia a medias; con eso los contadores de los compute se disparaban y la
    // GPU se perdía. Mientras está aquí no se graba otra en ella ni se vacía la arena.
    std::unordered_map<const uint8_t*, int> busy_readbacks_;  // copia -> su arena (readback_arenas_)
    uint64_t readback_busy_[2] = {0, 0};                         // copias sin entregar en cada arena
    std::unordered_map<uint64_t, uint64_t> delivered_hash_;  // dirección -> huella de lo último entregado
    // Qué bytes de un overlay escribe de verdad la GPU: un overlay lleva el rango entero, pero la GPU solo toca una
    // parte; devolverlo entero pisaba con copias viejas lo que la CPU había actualizado (transformaciones, contadores:
    // los draws dirigidos por la GPU se disparaban a 700 ms). Por trozos de 64 bytes: los que cambian entre una copia y
    // la anterior (o respecto a lo que había al crear el overlay) son de la GPU; solo esos se devuelven.
    static constexpr size_t kOwnChunk = 64;
    struct WriteBackState {
        std::vector<uint8_t> base;   // la copia anterior (o el contenido al crear el overlay)
        std::vector<uint8_t> owned;  // 1 por trozo que alguna vez escribió la GPU
        // Sube cada vez que el overlay se rehace desde la memoria del juego (la CPU la reescribió). Una copia de una
        // generación anterior es de antes de esa escritura: entregarla pisaba lo de la CPU (contadores que crecían
        // sin fin hasta perder la GPU), así que se descarta.
        uint64_t generation = 0;
        // Con la vigilancia (watch.h): qué páginas escribió la CPU desde base, aunque escribiera lo mismo que había
        // (un contador puesto otra vez a 0): comparar valores no lo ve y la devolución pisaba esa escritura.
        bool watched = false;
        std::vector<uint64_t> keys;
        bool orphan = false;  // su overlay no sobrevivió a la compactación: se borra al entregar lo pendiente
    };
    uint64_t wb_generation_ = 0;
    bool wb_orphans_ = false;
    std::unordered_map<uint64_t, WriteBackState> wb_state_;
    uint64_t prof_deliver_calls_ = 0, prof_deliver_up_to_ = 0, prof_deliver_min_ = 0, prof_deliver_pending_ = 0;
    std::map<uint64_t, std::pair<uint64_t, uint64_t>> prof_delivered_;  // DEPURACIÓN: dirección -> (tamaño, bytes)
    std::map<uint64_t, std::pair<uint64_t, uint8_t>> prof_wb_seen_;     // DEPURACIÓN: todas las copias: (tamaño, quién)
    // Solo hasta el envío que se esperó: lo de envíos posteriores pisaría lo que la CPU escriba después de leer (en
    // la Switch la GPU ya lo había escrito antes). Entregar de más hacía crecer sin fin los contadores de los compute.
    void deliver_write_backs(uint64_t up_to);
    // DEPURACIÓN: ¿cae address en un overlay devuelto en este periodo del perfil? (dirección del overlay o 0)
    uint64_t delivered_range(uint64_t address) {
        std::lock_guard lock(wb_mutex_);
        auto it = prof_delivered_.upper_bound(address);
        if (it == prof_delivered_.begin()) return 0;
        --it;
        return address < it->first + it->second.first ? it->first : 0;
    }
    std::string describe_memory(uint64_t address, uint64_t size);
    bool sync_texture_storage(Texture& t, const Texture::TexelWrite& r);
    enum class Visibility { Storage, CpuFence, Submission, Buffer };
    bool publish_virtual_pool(void* pool, Texture* only = nullptr, Visibility visibility = Visibility::Storage);
    void wait_idle() {
        std::lock_guard lock(g_mutex);
        publish_virtual_pool(nullptr);
        wait_all();
        reap_retired();
        deliver_write_backs(~0ull);  // todo terminado (nvnQueueFinish): todo lo escrito, en la memoria del juego
    }
    void retire_texture(Texture& t);
    void retire_sampler(VkSampler s);
    PicturePath blit(Texture& t, const VkRect2D& crop, VkImage dst, VkFormat format, VkExtent2D extent, VkCommandBuffer cmd);
    void dump_frame(Texture* presented);
    void finish_loading_capture(ResourceRef<Texture> presented, const VkRect2D& crop);
    void profile_frame();

private:
    bool init();
    void begin();
    void maybe_split();
    static constexpr VkDeviceSize kStagingSoftCap = 128ull << 20;
    // Varios envíos por frame: tras tantos draws, el siguiente límite de render pass manda lo grabado a la GPU, que
    // empieza a trabajar mientras se reproduce el resto (con un envío al final, la GPU esperaba a toda la
    // reproducción). Con <datos>/single_submit, uno por envío del juego, como antes.
    static constexpr uint64_t kSplitDraws = 300;
    uint64_t submit_draws_ = 0;  // draws_ al empezar el envío de Vulkan en curso
    bool split_ = true, generations_ = true;  // interruptores, leídos una vez por envío del juego
    void finish();
    void apply(const Command& c);
    void draw(const cmd::Draw& d);
    void clear_color(const cmd::ClearColor& c);
    void clear_attachment(int slot, const VkClearAttachment& clear, const VkRect2D& rect);
    void clear_depth_stencil(const cmd::ClearDepthStencil& c);
    void clear_texture(const cmd::ClearTexture& c);
    void copy_texture(const cmd::CopyTexture& c);
    bool apply_texel_writes(Texture& t);
    bool patch_image(Texture& t, const std::vector<Texture::TexelWrite>& writes);
    bool copy_texel_box(Texture& t, const Texture::TexelWrite& w);
    bool apply_mapping_changes(Texture& t);
    void barrier();
    void full_barrier();
    void nvn_barrier(int bits);
    // full_area: renderArea entero aunque la mirada adelante diga otra cosa (borrados dentro del pase).
    bool ensure_render_pass(bool full_area = false, const cmd::SetRenderTargets* planned = nullptr);
    void end_render_pass();
    bool ensure_image(Texture& t, VkCommandBuffer cmd);
    enum class PriorContents { Preserve, Replace };
    bool prepare_contents(Texture& t, PriorContents prior, bool inspect_guest = false);
    void upload(Texture& t);
    bool decode_level(Texture& t, const LevelLayout& lv, const uint8_t* blocks, uint32_t level, uint32_t layer,
                      std::vector<std::pair<VkBuffer, VkBufferImageCopy>>& regions, VkBuffer blocks_buffer = VK_NULL_HANDLE,
                      VkDeviceSize blocks_offset = 0);
    // BC1-BC3 decoded by a compute shader (shaders/bc_decode.comp) instead of on the CPU, where the GPU can't sample them.
    // The deswizzled blocks go to staging, the shader writes RGBA8 texels to another staging range, and that is copied to
    // the image like the CPU's. 4x less to upload and no CPU decoding. <data>/no_gpu_bc (live): the CPU, as before.
    bool bc_on_gpu(const Texture& t);
    // A level of a BC texture stored as ASTC (astc.h): from the disk cache, or decoded and encoded here and cached.
    bool astc_level(Texture& t, const LevelLayout& lv, const uint8_t* blocks, uint32_t level, uint32_t layer,
                    std::vector<std::pair<VkBuffer, VkBufferImageCopy>>& regions);
    bool bc_gpu_decode(const Texture& t, const LevelLayout& lv, VkBuffer blocks, VkDeviceSize offset, uint32_t level,
                       uint32_t layer, std::vector<std::pair<VkBuffer, VkBufferImageCopy>>& regions);
    // GPU texture unswizzle (Settings → Performance, <data>/gpu_unswizzle; Eden has the same idea): the level goes to
    // staging as it is in the game's memory, one straight copy, and shaders/deswizzle.comp puts it in rows on the GPU,
    // instead of the CPU picking it apart 16 bytes at a time. For levels whose rows are whole 16-byte sectors and that
    // are uploaded as they are (not decoded or re-encoded on the CPU, not checked for missing ASTC blocks).
    bool unswizzle_on_gpu();
    bool gpu_unswizzle(const LevelLayout& lv, uint32_t bpb, const uint8_t* src, const Staging::Alloc& out);
    VkDescriptorSetLayout unswizzle_set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout unswizzle_layout_ = VK_NULL_HANDLE;
    VkPipeline unswizzle_pipeline_ = VK_NULL_HANDLE;
    bool unswizzle_setup_done_ = false;
    uint64_t prof_gpu_unswizzled_ = 0;
    uint64_t prof_count_checked_ = 0;  // multi-draws drawn with the draw count check (count_check.h)
    uint64_t prof_astc_encoded_ = 0, prof_astc_cached_ = 0, prof_astc_empty_ = 0, prof_astc_direct_ = 0;  // levels stored as ASTC: encoded here / from the disk cache
    // Deferred encodes: a big level that isn't in the ASTC cache is encoded by a background thread instead of inside the
    // submit (seconds of it, the first time a monster's or an effect's textures load: the freezes in fights on Mali).
    // Until it's back the texture is sampled from its next smaller level (min_level), as when the game hasn't streamed a
    // level in yet; the result goes to the cache too. <data>/sync_astc (live): encoded in the submit, as before.
    struct AstcJob {
        Texture* texture = nullptr;
        uint64_t serial = 0;
        VkImage image = VK_NULL_HANDLE;
        uint64_t key = 0, cache_key = 0;  // key: what astc_pending holds for it (never 0); cache_key: the disk cache's
        uint32_t level = 0, layer = 0, width = 0, height = 0, wb = 0, hb = 0;
        uint8_t kind = 0;
        bool srgb = false, ok = false;
        std::vector<uint8_t> blocks, astc;
    };
    static constexpr size_t kAstcDeferTexels = 256 * 256;  // smaller levels are quick enough to encode in the submit
    void defer_astc(AstcJob job);
    void apply_astc_results();
    void astc_worker();
    std::mutex astc_mutex_;
    std::condition_variable astc_cv_;
    std::deque<AstcJob> astc_todo_, astc_done_;
    bool astc_started_ = false;
    std::unordered_map<Texture*, int> astc_waiting_;  // textures with levels away, and how many (under g_mutex)
    uint64_t prof_astc_deferred_ = 0, prof_astc_applied_ = 0;
    VkDescriptorSetLayout bc_set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout bc_layout_ = VK_NULL_HANDLE;
    VkPipeline bc_pipeline_ = VK_NULL_HANDLE;
    bool bc_setup_done_ = false;
    // Draw count check (count_check.h): the vertex stage with it, per stage of the game (code, control, its SPIR-V);
    // null if it didn't translate. The set layout of the count and an empty one for set 1 when there's no bindless.
    std::unordered_map<uint64_t, std::unique_ptr<ProgramStage>> count_check_stages_;
    VkDescriptorSetLayout count_check_set_layout_ = VK_NULL_HANDLE, empty_set_layout_ = VK_NULL_HANDLE;
    // Its sets by buffer and offset, for the current descriptor_epoch_ (thousands of multi-draws a frame share a few)
    std::unordered_map<uint64_t, VkDescriptorSet> count_check_sets_;
    uint64_t count_check_sets_epoch_ = 0;
    const ProgramStage* count_check_stage(const ProgramStage& vs);
    // ¿Hay que volver a subir una textura que sube la CPU, antes de enlazarla? Vigilada, las páginas escritas desde la
    // última revisión (una vez por envío); si no, la huella muestreada (una vez por frame).
    bool cpu_texture_changed(Texture& t);
    // Compare written pages' subresources to their last recorded upload.
    bool collect_texture_writes(Texture& t);
    bool watch_textures_ = true;  // interruptor no_texture_watch (read_flags): las de antes, por la huella muestreada
    uint64_t prof_audit_changes_ = 0, prof_watched_textures_ = 0;
    uint64_t prof_full_hash_bytes_ = 0;  // perfil: bytes por los que pasó full_hash (~0,2 ms por MB en la Flip)
    // storage: para enlazarla como storage image (un nivel, sin swizzle, formato con STORAGE_IMAGE).
    VkImageView view(Texture& t, const TextureViewDesc& v, ViewUse use, ImageViewSpec* result = nullptr);
    VkSampler sampler(Sampler* s);
    VkRenderPass render_pass(const VkFormat* colors, uint32_t count, VkFormat depth, VkSampleCountFlagBits samples,
                             const PassOps& ops = default_pass_ops());
    Layout& layout(const ProgramStage* const* stages);
    // El último layout devuelto y las identidades con las que se pidió: los draws seguidos con los mismos programas no pasan
    // por layout_for (mutex, clave y búsqueda). Mismas identidades, mismo layout: layouts_ nunca borra.
    // <datos>/no_layout_memo (en vivo): siempre por layout_for, como antes.
    Layout* last_layout_ = nullptr;
    uint64_t last_layout_ids_[kStages] = {};
    bool memo_layout_ = true;      // interruptor no_layout_memo (read_flags)
    // La última pipeline encontrada en pipelines_ y su clave entera: un draw con la misma clave (los seguidos con el mismo
    // estado) se salta el hash de sus ~440 bytes y la búsqueda. <datos>/no_pipeline_memo (en vivo): como antes.
    VkPipeline last_pipeline_ = VK_NULL_HANDLE;
    uint64_t last_pipeline_hash_ = 0;
    PipelineKey last_pipeline_key_{};
    bool memo_pipeline_ = true;    // interruptor no_pipeline_memo (read_flags)
    // El set 1 de un draw con el layout de bindless del layout del draw (Layout::bindless, hecho con los mismos shaders) en vez
    // de volver a sacar su firma en cada uno (un vector nuevo y una búsqueda en bindless_layouts_). <datos>/no_fast_bindless.
    bool fast_bindless_ = true;    // interruptor no_fast_bindless (read_flags)
    // El de layout() a partir de lo traducido de cada etapa y su identidad (el precalentado no tiene ProgramStage).
    Layout& layout_for(const nvnshader::Translated* const* shaders, const uint64_t* ids);
    VkPipeline pipeline(const cmd::Draw& d, VkRenderPass rp, Layout& l, uint32_t color_count, VkSampleCountFlagBits samples,
                        bool probe_face = false);
    VkPipeline face_pipeline(const PipelineKey& key, uint64_t inputs, Layout& layout, VkRenderPass pass);
    bool resolve_texture(Stage s, const Resource& r, Texture*& tex, TextureViewDesc& v, Sampler*& smp);
    bool resolve_bindless(Stage s, const Resource& r, Texture*& tex, TextureViewDesc& v, Sampler*& smp);
    bool resolve_image(Stage s, const ProgramStage& ps, const Resource& r, Texture*& tex, TextureViewDesc& v);
    // Vista de storage de la textura con el tipo que declara el shader (Shader::TextureType), o la de reserva
    // del tipo del shader si no se puede enlazar así.
    VkImageView storage_view(Texture* t, const TextureViewDesc& v, uint32_t texture_type, Texture*& bound);
    VkDescriptorSet descriptor_set(VkDescriptorSetLayout layout);
    // Bindless del juego (handles en sus uniform buffers). resolve_handle: una palabra como la lee la GPU (textura en los
    // 20 bits de abajo, sampler arriba) tras la máscara y el sampler fijo del shader. bindless_set: el set 1 de los arrays
    // con índice dinámico (decals, sondas de reflexión), uno por contenido y envío.
    bool resolve_handle(uint32_t word, const Resource& r, Texture*& tex, TextureViewDesc& v, Sampler*& smp,
                        const std::vector<TexturePoolEntry>* textures = nullptr,
                        const std::vector<ResourceRef<Sampler>>* samplers = nullptr);
    VkDescriptorSetLayout bindless_layout_locked(const nvnshader::Translated* const* shaders);  // con objects_mutex_
    // known: el layout del set 1 si ya se sabe (Layout::bindless); si no, se saca de los shaders.
    VkDescriptorSet bindless_set(const ProgramStage* const* stages, VkDescriptorSetLayout known = VK_NULL_HANDLE,
                                 const cmd::SetRenderTargets* attachments = nullptr);
    Texture* sample_source(Texture& source, const TextureViewDesc& v, ViewUse use, const cmd::SetRenderTargets& attachments);
    Texture* snapshot_texture(Texture& source);
    struct FeedbackCopy {
        Texture texture;
        uint64_t version = 0;
    };
    std::unordered_map<Texture*, FeedbackCopy> feedback_copies_;
    // Muestreo: subir la textura si cambió (lo del bucle de texturas del draw), su vista con el tipo que declara el shader
    // y la de reserva de ese tipo (un 2D enlazado a un cubo o a una 3D es indefinido).
    bool prepare_shader_view(Texture& t, const TextureViewDesc& v, ViewUse use);
    struct BoundImage {
        const Resource* r;
        Stage s;
        Texture* tex = nullptr;
        TextureViewDesc view{};
        Sampler* smp = nullptr;
        Texture* storage_target = nullptr;
    };
    bool prepare_shader_image(const ProgramStage& ps, BoundImage& b);
    void publish_storage_writes(const std::vector<BoundImage>& bindings, const char* reason);
    VkImageView dummy_view(uint32_t texture_type, ViewUse use = ViewUse::Sampled);
    void log_sampler_once(uint32_t id, const Sampler* smp, const char* why);  // DEPURACIÓN (registro)
    void log_draw(const cmd::Draw& d);
    bool traced_in_list(const char* var) const;
    // DEPURACIÓN: cuántos draws del frame trazado van al log (NIER_TRAZA_MAX, fichero o entorno; 1500 si no). En Kamura
    // hay ~1470 por frame y en otras zonas más: los de la cara del cazador pueden quedar fuera.
    int trace_cap() const;
    static constexpr int kCounterSamplesPassed = 1;  // NVNcounterType
    static constexpr int kCounterTimestamp = 0;
    uint64_t samples_passed_ = 0;
    uint64_t watch_addr_ = 0;
    bool full_trace_ = false;  // DEPURACIÓN: la traza pedida vuelca también los render targets
    struct PrepassArgs {
        uint64_t indices;
        uint32_t count;
        int type;
    };
    std::unordered_map<uint64_t, PrepassArgs> prepass_args_;  // DEPURACIÓN: args de la compactación -> índices originales
    struct CulledDraw {
        cmd::Draw draw;  // el draw directo del pre-pase (malla entera)
        uint64_t frame;
    };
    std::unordered_map<uint64_t, CulledDraw> culled_draws_;  // args que escribe su geometry shader -> ese draw
    struct TracedGpuDraw {
        uint64_t args_address;
        const uint8_t* args;
        const uint8_t* indices;
        int type;
    };
    std::vector<TracedGpuDraw> traced_gpu_draws_;
    // DEPURACIÓN: con <datos>/profundidad_lequal, los draws con depth EQUAL usan LEQUAL (prueba en caliente).
    bool depth_lequal_ = false;
    // Fast depth test (<datos>/fast_depth, Graphics menu). MH Rise uses reversed Z: the pre-pass writes depth with GEQUAL
    // and the colour pass draws again with EQUAL. On Adreno, EQUAL turns off LRZ (the early depth rejection), so the
    // colour pass shades much more; GEQUAL gives the same picture for solid meshes and keeps LRZ. Not for cut-out ones
    // (grass, leaves): their pre-pass discards the see-through pixels and the colour pass relies on EQUAL to skip them,
    // so with GEQUAL the whole card shows. Those are the meshes drawn in the pre-pass with a pixel shader (solid ones
    // have none); cutout_meshes_ remembers them by vertex and index buffer, and they keep EQUAL.
    bool fast_depth_ = false;
    std::unordered_set<uint64_t> cutout_meshes_;
    static constexpr uint32_t kDepthEqual = 3, kDepthLequal = 4, kDepthGequal = 7;  // NVNdepthFunc  // DEPURACIÓN: al acabar el frame trazado, quién lee el primer contador condicional
    std::vector<std::pair<Texture*, uint32_t>> tex_dumps_;  // DEPURACIÓN: texturas por volcar (NIER_VOLCAR_TEX)
    // DEPURACIÓN (<datos>/verbose_log): estado pequeño (≤ 64 bytes) que escriben los compute (la exposición automática de MH
    // Rise y su historial): cada 30 frames su valor en la GPU y en la memoria del juego, y tras cada compactación, para
    // ver cada cuánto se mide, si converge y si se pierde.
    struct SmallState {
        uint64_t size = 0, frame = 0, prev_frame = 0, writes = 0;
    };
    std::map<uint64_t, SmallState> small_state_;
    std::unordered_map<const void*, uint64_t> state_ubo_logged_;  // compute -> frame en que se registraron sus ubo
    bool log_state_ = false;  // registro, una vez por envío del juego
    void log_small_state(const char* why);
    // DEPURACIÓN (<datos>/watch_nan, en vivo): de dónde sale lo que tiene cada overlay y qué lee cada shader. La cara del
    // cazador (traza en 'Elegir personaje') lee sus vértices de un overlay con posiciones NaN y normales 0x00800000 /
    // 0x7f7fffff, que nadie escribe en el frame trazado: se calculan al cargar el personaje. Con el interruptor, sin traza:
    // - cada overlay nuevo: quién lo crea (compute con sus uniform buffers y storage buffers, draw o copia) y lo que tenía
    //   la memoria del juego al crearlo (lo que copia el overlay);
    // - la primera vez que un draw lee posiciones (float32) de un overlay ya escrito por la GPU: si son NaN/Inf, con esa
    //   procedencia y con la memoria del juego del mismo rango (igual que el overlay: la GPU no escribió nada ahí);
    // - las lecturas (SSBO de compute o draw, streams de vértices, origen de CopyBuffer) que salen de la memoria del juego
    //   o de un espejo aunque la GPU escribió parte de ese rango en un overlay que no lo cubre entero (find_overlay exige
    //   que lo contenga): esa parte se lee vieja;
    // - los dispatch descartados (sin shader, 0 grupos, sin set, sin staging) y el código de cada compute la primera vez
    //   que se ejecuta (shaders/vigia_cs_<huella>, para shadertool).
    bool vigilar_nan_ = false;  // read_flags
    std::string writer_;        // quién escribe ahora (dispatch, draw o copia en curso), para make_overlay
    struct Provenance {
        std::string created, last;  // quién lo creó y quién lo escribió por última vez en la GPU
        uint64_t created_frame = 0, last_frame = 0, gpu_writes = 0, cpu_patches = 0, patch_frame = 0;
        uint64_t submit = 0;   // envío de Vulkan de su última escritura de la GPU (o de su creación)
        bool checked = false;  // un draw ya miró si sus posiciones son NaN
    };
    std::unordered_map<uint64_t, Provenance> provenance_;  // por dirección de inicio del overlay
    std::unordered_set<uint64_t> dumped_writers_;           // huellas de compute ya volcadas
    std::unordered_set<uint64_t> partial_logged_;           // rangos ya avisados por leerse sin lo de la GPU
    void note_partial_read(const char* who, uint64_t address, uint64_t size);
    // DEPURACIÓN (<datos>/trace_print: una huella hexadecimal por línea, y "completa" para volcar también los render
    // targets): la primera vez que se ejecuta un compute de la lista se traza lo que queda de ese frame, desde él (como
    // <datos>/trazar, pero sin esperar al frame siguiente). Para el trabajo que solo se hace al cargar, que ninguna traza
    // de un frame cualquiera ve: la cara del cazador se calcula al cargar el personaje (793657b4f1e65631 y
    // 6aaf54a71876a3f2 se traducen, es decir se ejecutan por primera vez, justo cuando aparecen los overlays de las
    // cabezas). Cada huella dispara una vez por sesión.
    bool trace_on_print_ = false;  // read_flags: existe el fichero
    bool trace_prints_loaded_ = false, trace_prints_full_ = false;
    std::vector<uint64_t> trace_prints_;
    bool trace_on_print(uint64_t print);
    void check_overlay_vertices(uint64_t address, uint64_t size, uint32_t stride, uint32_t format, uint32_t stream);
    // DEPURACIÓN (<datos>/read_after_dispatch: un tamaño en bytes por línea; 54600 es la cara del cazador): tras cada
    // compute que escribe un overlay de ese tamaño se cierra el envío, se espera a la GPU y se registra lo que quedó de
    // verdad en él (palabras NaN/Inf y a cero, la primera NaN, las primeras palabras), con lo que recibió el compute: cada
    // uniform buffer y SSBO, de dónde salió (overlay, espejo, copia del envío o staging) y sus primeras palabras tal como
    // las vio la GPU. Las líneas "F compute SSBO … en la GPU" de la traza se toman al grabar, antes de que la GPU ejecute
    // nada del envío: no dicen qué escribió ese compute (en la cara salían a cero tras el skinning y luego eran NaN). Se
    // vuelca también el código de cada uno (leer_cs<dispatch>): la huella 0 (sin constantes) junta todos los compute sin
    // c1, y el que crea la cara (dispatch 97352 en log2) nunca se había volcado.
    struct ReadInput {
        std::string label;
        uint64_t address = 0, size = 0;  // address 0: el c0 sintético (sin palabras)
        const char* from = "";
        uint32_t words = 0;  // cuántas palabras suyas se registran
        VkDescriptorBufferInfo bound{};  // lo que se enlazó (para volcar lo que ve la GPU: los espejos no están mapeados)
        const uint8_t* snapshot = nullptr;
    };
    bool read_after_dumped_ = false;  // ya se volcaron las entradas del primer dispatch que dejó NaN
    void dump_gpu_buffer(const VkDescriptorBufferInfo& b, uint64_t size, const std::string& file);
    bool read_after_ = false;  // read_flags: existe el fichero
    bool read_after_loaded_ = false;
    int read_after_logged_ = 0;
    std::vector<uint64_t> read_after_sizes_;
    bool read_after_size(uint64_t size);
    void read_after_dispatch(const ProgramStage& cs, uint32_t x, uint32_t y, uint32_t z,
                             const std::vector<std::pair<uint64_t, uint64_t>>& ranges, const std::vector<ReadInput>& inputs);

    bool ready_ = false;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    // Envíos en vuelo: cada uno con su command buffer, fence, staging y pools de descriptores. El juego
    // sigue mientras la GPU trabaja; solo se espera al reutilizar un contexto, cuando la CPU necesita un
    // resultado (write-back de compute, volcados) o cuando el juego espera un NVNsync.
    // pending, submit y waiters los cambia solo quien tiene g_mutex, y además con fence_mutex_: los hilos del juego
    // que esperan un envío (wait_submit) los leen con fence_mutex_, sin g_mutex.
    // Con varios envíos por frame (~5 en el mundo abierto) 12 cubren unos 2 frames en vuelo; más adelantada, la CPU
    // espera en begin(). Staging en trozos de 4 MB (un envío partido usa 1-2 MB) y cada contexto guarda lo que usó
    // la última vez hasta 16 MB, como antes un trozo: ~48 MB fijos (eran 6 x 16 MB).
    static constexpr VkDeviceSize kSlotStagingChunk = 4 << 20;
    static constexpr int kInFlight = 12;
    struct InFlight {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        Staging staging{kSlotStagingChunk};
        Staging texel_readback{256 << 10, true};
        std::vector<VkDescriptorPool> pools;
        size_t pool = 0;
        bool pending = false;
        uint64_t submit = 0;
        int waiters = 0;  // hilos esperando su fence fuera de los locks: hasta que salgan no se reinicia
        // Informes TIMESTAMP de este envío: su valor lo escribe la GPU (vkCmdWriteTimestamp) y se copia a la memoria
        // del juego cuando el envío termina (note_completed), antes de que el juego lo vea terminado.
        VkQueryPool queries = VK_NULL_HANDLE;
        struct Stamp {
            uint32_t query;
            uint64_t address;
            int64_t extra;  // ticks de 614,4 MHz de más (el coste de LSFG, en el informe del final del frame)
            bool end;       // el del final del frame (16 bytes detrás del del comienzo)
        };
        std::vector<Stamp> stamps;
    };
    static constexpr uint32_t kStampsPerSlot = 64;
    // Tiempos de GPU para la resolución dinámica del juego (ver ReportCounter): un tick de la GPU en ticks de 614,4 MHz
    // (las unidades de NVN), el comienzo del último frame y lo que tardó.
    double stamp_scale_ = 32.0;
    uint64_t stamp_mask_ = ~0ull, frame_start_ticks_ = 0;
    std::atomic<int64_t> frame_gpu_ns_{0};
    // Guarda del FSR al presentar (blit): con el frame del juego por encima de kFsrOffNs de GPU en 3 frames seguidos se
    // escala con el blit lineal; el FSR vuelve tras 60 frames seguidos por debajo de kFsrOnNs.
    static constexpr int64_t kFsrOffNs = 28'500'000, kFsrOnNs = 25'000'000;
    // Con <datos>/60fps el frame tiene 16,7 ms: el FSR (~2 ms en fp16) solo si el del juego deja ese sitio.
    static constexpr int64_t kFsrOffNs60 = 13'000'000, kFsrOnNs60 = 11'000'000;
    bool fsr_guard_off_ = false;
    int fsr_guard_frames_ = 0;
    uint64_t prof_fsr_guard_frames_ = 0;  // perfil: frames presentados con el blit por la guarda
    bool gpu_stamps_ = true;     // interruptor ts_cpu: los de antes (reloj de la CPU al reproducir)
    bool skip_compaction_gs_ = true;  // interruptor no_gs_skip
    int64_t stamp_extra_ = 0;    // lo que se suma al final del frame en este envío del juego
    int run_stamps_ = 0;         // informes TIMESTAMP vistos en este envío del juego
    uint64_t last_stamp_addr_ = 0;  // el del comienzo: el del final va 16 bytes detrás (main+0x9bbb08 lee +0x8 y +0x18)
    int scene_vp_w_ = 0, scene_vp_h_ = 0;  // perfil: el viewport de la escena más grande desde el último
    int scene_vp_min_w_ = 1 << 30;         // ...y el ancho más pequeño
    void write_stamps(InFlight& f);
    // DEPURACIÓN: perfil de GPU por pase (<datos>/profile_gpu, se borra solo): un timestamp antes de cada render pass y
    // de cada dispatch de un frame, y el log con lo que tardó cada grupo de pases.
    static constexpr uint32_t kProfMarks = 1024;
    struct ProfMark {
        std::string label;
        uint64_t draws = 0;
    };
    VkQueryPool prof_pool_ = VK_NULL_HANDLE;
    std::vector<ProfMark> prof_marks_;
    uint64_t prof_frame_ = ~0ull, prof_done_submit_ = 0, prof_mark_draws_ = 0;
    bool prof_closed_ = false;
    void prof_gpu_mark(const char* kind, const char* custom = nullptr);
    void prof_gpu_report();
    // DEPURACIÓN (<datos>/breadcrumbs al arrancar): migas de pan para un cuelgue de la GPU. Antes y después de cada render
    // pass y de cada dispatch la GPU escribe su número en memoria de la CPU (con barreras: todo va en serie); cuando
    // la GPU se pierde, el log dice qué empezó y no acabó, con las huellas de sus shaders, y vuelca su código.
    struct Crumb {
        uint64_t submit = 0;
        char kind = 0;  // 'p' render pass, 'c' dispatch
        uint32_t draws = 0;
        uint32_t n_stages = 0;
        const ProgramStage* stages[8] = {};
        char label[64] = {};
        // Los draws con argumentos en la GPU: dónde leerlos desde la CPU (overlay o staging) cuando se pierda.
        struct GpuDraw {
            const uint8_t* args = nullptr;
            const uint8_t* count = nullptr;  // nullptr: el número lo puso la CPU (cpu_count)
            uint64_t address = 0;
            uint32_t stride = 0, max_draws = 0, cpu_count = 0;
            int index_type = -1;
        };
        std::vector<GpuDraw> gpu_draws;
    };
    static constexpr uint32_t kCrumbs = 1 << 14;
    bool crumbs_on_ = false, crumbs_reported_ = false;
    bool crumbs_passes_ = true;  // <datos>/breadcrumbs_compute: solo los dispatch (ya llevan barreras: casi no cambia el ritmo)
    bool crumbs_barriers_ = true;  // <datos>/breadcrumbs_light: solo los dispatch y sin barreras propias (las del dispatch)
    // Adreno: una escritura de transferencia (vkCmdFillBuffer de 4 bytes) antes y después de cada dispatch. Sin ella la
    // GPU se colgaba al cargar el título de MH Rise (DEVICE_LOST en 6 de 7 arranques; ninguno en 5 con ella, ni con
    // las migas, que hacen lo mismo). Son las escrituras, no la reserva del buffer (con solo la reserva se cuelga). Las
    // barreras completas que ya rodean el dispatch no bastan, conservar las barreras condicionales de Eden tampoco, y
    // los espejos (sin SSBO recortados) tampoco; la causa exacta no se ha encontrado. <datos>/no_compute_fill.
    bool fill_quirk_ = false;
    uint32_t fill_count_ = 0;
    VkBuffer crumb_buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory crumb_memory_ = VK_NULL_HANDLE;
    volatile uint32_t* crumb_map_ = nullptr;  // [0] la última que empezó, [1] la última que acabó
    std::vector<Crumb> crumbs_;
    uint32_t crumb_id_ = 0;
    void crumb_init(bool crumbs);
    // DEPURACIÓN (<datos>/measure_changes): cuántas páginas de 64 KB de un buffer grande cambian entre una copia y la
    // siguiente (para decidir cómo dejarlo en la GPU).
    struct ChangeStats {
        std::vector<uint64_t> pages;
        uint64_t copies = 0, changed_copies = 0, changed_pages = 0, frames = 0, last_frame = 0, same_frame = 0;
    };
    std::map<uint64_t, ChangeStats> change_stats_;
    void measure_changes(uint64_t address, uint64_t size);
    void crumb_begin(char kind, const char* label);
    void crumb_end();
    void crumb_stage(const ProgramStage* s);
    void crumb_report(const char* why = "GPU perdida");
    int64_t reserve_margin_ns();
    int64_t reserve_margin_ = 2000000;
    std::chrono::steady_clock::time_point reserve_next_{};
    InFlight inflight_[kInFlight];
    std::mutex fence_mutex_;
    std::condition_variable fence_cv_;  // waiters volvió a 0
    // Recursos que el juego soltó, pendientes de que la GPU termine el último envío que pudo usarlos.
    struct Retired {
        uint64_t serial = 0;  // textura: sus vistas (y los framebuffers que las usan) caen con ella
        VkImage image = VK_NULL_HANDLE;
        GpuAllocation memory;
        uint64_t bytes = 0;
        VkSampler sampler = VK_NULL_HANDLE;
        uint64_t submit = 0;
    };
    std::vector<Retired> retired_;
    void reap_retired();
    // Caché de pipelines del driver, guardada en la SD: sin ella cada sesión recompila todos los pipelines la
    // primera vez que se ven (tirones al entrar en cada zona).
    DriverPipelineCache driver_cache_;
    VkPipeline bound_graphics_ = VK_NULL_HANDLE;  // el enlazado en cmd_
    struct DynamicState {
        VkViewport viewport;
        VkRect2D scissor;
        float blend[4];
        uint32_t ref[2], write_mask[2], compare_mask[2];
        float bias[3];
        // extended dynamic state
        uint32_t cull, front, depth_test, depth_write, depth_op, stencil_test;
        uint32_t stencil_ops[2][4];  // per face: fail, pass, depth fail, compare
    };
    DynamicState bound_dynamic_{};
    // The vertex layout last set with vkCmdSetVertexInputEXT (valid with dynamic_valid_)
    uint64_t bound_vertex_input_ = 0;  // el puesto en cmd_ (válido si dynamic_valid_)
    VkBuffer bound_vb_buf_[16] = {};  // vertex buffers enlazados en cmd_
    VkDeviceSize bound_vb_off_[16] = {};
    VkBuffer bound_index_ = VK_NULL_HANDLE;
    VkDeviceSize bound_index_offset_ = 0;
    VkIndexType bound_index_type_ = VK_INDEX_TYPE_UINT16;
    void bind_index(VkBuffer buffer, VkDeviceSize offset, VkIndexType type) {
        if (buffer == bound_index_ && offset == bound_index_offset_ && type == bound_index_type_) return;
        vkCmdBindIndexBuffer(cmd_, buffer, offset, type);
        bound_index_ = buffer;
        bound_index_offset_ = offset;
        bound_index_type_ = type;
    }
    bool dynamic_valid_ = false;
    uint64_t descriptor_epoch_ = 1;  // cambia en cada begin(): los sets recordados por los layouts caducan
    // Perfil: pipelines creadas desde el último informe y lo que tardó el driver (en NvnSubmit, con el frame parado).
    uint64_t prof_pipelines_logged_ = 0, prof_pipeline_us_ = 0;
    std::unordered_map<uint64_t, uint64_t> prof_mirror_top_;  // perfil: bytes subidos por espejo (su dirección)
    // El hilo del guardado lee la caché del driver (vkGetPipelineCacheData): una pipeline que el ejecutor crea a la vez
    // puede esperar dentro del driver a que acabe. note_pipeline_during_save lo dice (con lo que tardó).
    void note_pipeline_during_save(uint64_t us) {
        static int logged = 0;
        if (logged++ < 32)
            NVN_LOG("pipeline creada mientras el guardado leía la caché del driver: %.1f ms en NvnSubmit (%d veces)\n",
                    us / 1000.0, logged);
    }
    // Cada pool (4096 sets) cuesta unos MB de memoria de la GPU: tope por envío y lo que se guarda tras un pico.
    static constexpr size_t kMaxDescriptorPools = 32, kKeptDescriptorPools = 2;
    // Memoria que el driver tiene reservada para el proceso (incluye la interna, que no pasa por mis contadores).
    long kgsl_logged_mb_ = 0;
    std::chrono::steady_clock::time_point kgsl_checked_{};
    void watch_kgsl();
    int cur_ = 0;
    // Envíos a la cola y el último terminado (todos los anteriores también: la señal de un fence cubre lo enviado
    // antes a la cola). completed_ lo adelanta también quien espera un envío desde otro hilo.
    std::atomic<uint64_t> submitted_{0}, completed_{0};
    void note_completed(uint64_t submit) {  // con fence_mutex_
        InFlight* done[kInFlight];
        int n = 0;
        for (InFlight& f : inflight_)
            if (f.pending && f.submit <= submit && !f.stamps.empty()) done[n++] = &f;
        std::sort(done, done + n, [](const InFlight* a, const InFlight* b) { return a->submit < b->submit; });
        for (int k = 0; k < n; k++) write_stamps(*done[k]);
        uint64_t c = completed_.load(std::memory_order_relaxed);
        while (c < submit && !completed_.compare_exchange_weak(c, submit, std::memory_order_acq_rel)) {
        }
    }
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    bool recording_ = false;
    Staging* staging_ = nullptr;
    Staging::Alloc graphics_c0_{};
    VkDeviceSize graphics_c0_valid_ = 0;
    Staging::Alloc graphics_c0(VkDeviceSize size, VkDeviceSize alignment);
    void wait_slot(InFlight& f);
    void wait_all();
    void finish(bool wait);
    QueueState st_;
    bool targets_changed_ = true;
    bool in_render_pass_ = false;
    VkExtent2D rp_extent_{};
    VkRenderPass rp_ = VK_NULL_HANDLE;  // el de LOAD/STORE (compatible con el empezado): pipelines y framebuffers
    uint32_t rp_color_count_ = 0;
    // Un pase de sombras: sin color y con la profundidad en una textura de varias capas (las cascadas de MH Rise, 512x512
    // D16 con una cascada por capa; las luces puntuales, cubemaps). La de la escena es de una capa (1280x720).
    bool shadow_pass() const {
        if (!st_.rt.depth || st_.rt.depth->layers <= 1) return false;
        for (int k = 0; k < st_.rt.count && k < 8; k++)
            if (st_.rt.colors[k]) return false;
        return true;
    }
    bool shadow_skip_logged_ = false;
    VkSampleCountFlagBits rp_samples_ = VK_SAMPLE_COUNT_1_BIT;
    VkRect2D rp_area_{};          // renderArea del pase abierto
    bool rp_area_full_ = true;    // ...y si es el framebuffer entero
    Texture* rp_targets_[9] = {};  // texturas adjuntas al pase abierto
    uint32_t rp_target_count_ = 0;
    // Todas las del pase se redibujan cada frame (Texture::rt_recurring): solo entonces una pipeline nueva puede ir al hilo
    // de segundo plano y sus draws esperarla sin dibujarse. Lo que el juego dibuja una vez (cubemaps de una zona al
    // entrar, retratos, iconos) se queda como estaba: en NvnSubmit.
    bool rp_recurring_ = false;

    // --- Optimizaciones del tiler (Adreno) ---
    // Cada una con su interruptor en vivo (flags.h): por defecto activas, <datos>/sin_<nombre> la apaga. Se leen
    // una vez por envío (read_flags, al empezar run()); todas pueden cambiar entre envíos: nada de su estado pasa
    // de un envío al siguiente, salvo no_ubwc, que solo cuenta para las imágenes que se creen después.
    struct TilerOptions {
        bool clear_diferido = true;    // borrado completo -> loadOp CLEAR del siguiente pase con ese target
        bool rt_igual = true;          // volver a enlazar los mismos targets no corta el pase
        bool barrera_perezosa = true;  // una barrera dentro del pase solo lo corta si hace falta
        bool solo_escritos = true;     // muestrear un adjunto solo corta el pase si se escribió en él (rp_written_)
        bool store_ops = true;         // STORE_OP_NONE para lo que el pase no escribe
        bool area = true;              // renderArea = lo que dibujan los draws del pase
        bool ubwc = true;              // profundidad/stencil sin MUTABLE_FORMAT (compresión UBWC)
    };
    TilerOptions opt_;
    void read_flags();
    // Lo que se está reproduciendo en run() (cada run() con su número) y el comando en curso: la mirada adelante de
    // los pases recorre lo que queda.
    const std::vector<const CommandList*>* run_lists_ = nullptr;
    const Command* cur_command_ = nullptr;
    size_t cur_list_ = 0, cur_index_ = 0;  // posición en run(): la misma lista puede ir dos veces en un envío
    uint64_t run_serial_ = 0;
    // Borrados diferidos: un ClearColor/ClearDepthStencil de todo el target antes de que empiece su pase no abre
    // un pase (LOAD + borrar + STORE) sino que espera: el siguiente pase que lo use como adjunto (misma vista) lo
    // hace con loadOp CLEAR, sin leer nada. Si antes se muestrea, se copia, se presenta o acaba el envío, se hace
    // con vkCmdClearColorImage/DepthStencilImage fuera del pase. Uno por nivel y capa.
    struct PendingClear {
        Texture* texture;
        VkImageSubresourceRange range;
        VkClearValue value;
    };
    std::vector<PendingClear> pending_clears_;
    bool defer_clear(Texture* t, const TextureViewDesc& v, VkImageAspectFlags aspects, const VkClearValue& value);
    void flush_clears(const Texture* only, const VkImageSubresourceRange* range = nullptr);
    void touch_targets();
    static VkExtent2D target_extent(const cmd::SetRenderTargets& targets);
    // Barrera de NVN aplazada al final del pase abierto (nvn_barrier), y si en el pase ya se escribió storage
    // desde la última barrera (entonces no se aplaza).
    bool pending_barrier_ = false, storage_since_barrier_ = false;
    // Barreras completas sin repetir. Una barrera completa (full_barrier: de todo a todo, escrituras a lecturas y
    // escrituras) justo detrás de otra, sin nada grabado en medio, no ordena nada más. gpu_work_: se grabó algo desde la
    // última (lo pone note_gpu_work() delante de cada comando de Vulkan que lee o escribe memoria fuera de un pase: copias,
    // borrados, dispatch, barreras de imagen, el principio de un pase; los draws van dentro de un pase, que ya lo puso).
    // Los rellenos de 4 bytes de fill_quirk_ no cuentan: solo escriben crumb_buffer_, que no lee ningún comando de la GPU,
    // y la barrera de detrás de cada dispatch, que siempre va (el dispatch es trabajo), los deja en orden entre ellos.
    // work_after_pass_: se grabó algo desde el final del último pase (o la última barrera). Su dependencia externa (todos
    // los gráficos del pase -> todos los comandos, escrituras -> lecturas y escrituras) encadena con la del principio (todos
    // los comandos de antes -> el pase): ordena lo grabado antes del final del pase con todo lo que venga, como una barrera
    // completa. <datos>/tight_barriers (en vivo): se salta la que no ordena nada; <datos>/tight_barriers_pass: también
    // la que solo tiene detrás el final de un pase (cortes por barreras de NVN, dispatch tras un pase). Apagadas por
    // defecto; el perfil cuenta las que se saltarían aunque no se salten.
    bool gpu_work_ = true, work_after_pass_ = true;
    void note_gpu_work() { gpu_work_ = work_after_pass_ = true; }
    bool skip_redundant_barriers_ = false, pass_end_is_barrier_ = false;  // interruptores (read_flags)
    uint64_t prof_full_barriers_ = 0, prof_barriers_redundant_ = 0, prof_barriers_after_pass_ = 0, prof_barriers_skipped_ = 0;
    bool splitting_ = false;  // finish() de un envío partido a mitad (maybe_split): sin vaciar overlays
    // Adjuntos del pase abierto que pudo escribir algo desde que empezó (borrados, draws con máscara de color o con
    // escritura de profundidad/stencil): solo esos difieren de su memoria. Muestrear uno que no se ha escrito no
    // obliga a cortar el pase (MH Rise muestrea la profundidad tras cada barrera de sus partículas y agua: ~100 cortes
    // por frame en la aldea).
    VkImageAspectFlags rp_written_[9]{}; // aspects written in the current attachment view
    VkImageSubresourceRange rp_ranges_[9]{};
    int rp_slots_[9] = {};     // el slot de cada adjunto: color (>= 0) o profundidad (-1)
    bool pass_overlaps(const BoundImage& b, bool written_only) const {
        for (uint32_t k = 0; k < rp_target_count_; ++k) {
            if (b.tex != rp_targets_[k] || (written_only && !rp_written_[k])) continue;
            const auto use = b.r->kind == Resource::Image ? ViewUse::Storage : ViewUse::Sampled;
            auto range = rp_ranges_[k];
            if (written_only) range.aspectMask &= rp_written_[k];
            if (ranges_overlap(view_range(*b.tex, b.view, use), range)) return true;
        }
        return false;
    }
    void note_pass_writes() {
        const auto* fs = st_.stages[static_cast<int>(Stage::Fragment)];
        const auto access = draw_attachment_access(st_.masks.masks & rp_keep_masks_, fs ? fs->shader.color_outputs : 0,
                                                   st_.depth_stencil, st_.stencil_mask);
        for (uint32_t k = 0; k < rp_target_count_; k++) {
            if (auto written = access.writes(rp_slots_[k]) & rp_targets_[k]->aspect) {
                rp_written_[k] |= written;
                record_texture_write(*rp_targets_[k], rp_ranges_[k], "render target");
            }
        }
    }
    // Mirada adelante desde el comando en curso hasta el siguiente SetRenderTargets con otros targets (el pase no
    // puede seguir más allá): qué adjuntos escriben sus draws y borrados, si hay borrados y dónde pueden dibujar.
    // Un pase con esos targets que empiece antes de `end` en el mismo envío solo ve una parte de eso: le vale
    // (un pase que se corta por una subida o una barrera reutiliza la del primero).
    struct PassScan {
        uint64_t run = 0;  // run_serial_ del envío (0: ninguna)
        cmd::SetRenderTargets rt{};
        cmd::SetRenderTargets pass{};  // los adjuntos del pase (con merged_passes, más que rt; si no, los mismos)
        size_t start_list = 0, start_cmd = 0;  // desde dónde se miró (inspect_per_pass)
        size_t end_list = 0, end_cmd = 0;
        uint32_t colors_written = 0;  // bit por render target
        VkImageAspectFlags ds_read = 0, ds_written = 0;
        bool cleared = false, drawn = false;
        int64_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;  // unión de lo que pueden tocar sus draws (sin recortar)
    };
    PassScan scan_;
    const PassScan& scan_pass(const cmd::SetRenderTargets& pass);
    // Pases unidos (<datos>/merged_passes, en vivo; apagado por defecto). MH Rise cambia en mitad de la escena entre unos
    // targets y un subconjunto suyo (color + velocidad + profundidad y solo color + profundidad, las mismas texturas y
    // vistas): cada cambio cerraba el pase (guardar y volver a cargar ~4 MB a 960x540) y abría otro. Con esto el pase se
    // empieza con los adjuntos de todos los targets que vienen y caben juntos (merge_targets: hasta unos que no caben, un
    // dispatch o una copia de texturas), y los draws de un subconjunto van en él con la máscara de color de los que no son
    // suyos a cero (rp_keep_masks_). Lo que el pase no tenía antes se añade solo si ya es de la GPU (gpu_written, con
    // imagen), del mismo tamaño y muestras. Muestrear o escribir como imagen un adjunto del pase lo corta como siempre.
    bool merge_passes_ = false;      // interruptor merged_passes (read_flags)
    // <datos>/inspect_per_pass (en vivo; apagado por defecto): cada pase mira desde su primer comando, no reutiliza la mirada
    // del primero de sus targets. Tras un corte (barrera, subida, espejo…) lo que queda puede no escribir la profundidad o
    // un color (STORE_OP_NONE en vez de guardarlo) y tocar menos pantalla (área recortada: menos bins que cargar y guardar).
    // Cuesta un recorrido de lo que queda de esos targets por cada corte.
    bool scan_per_pass_ = false;
    uint32_t rp_keep_masks_ = ~0u;   // máscaras de color del pase que valen para los targets del juego (4 bits por sitio)
    bool merge_targets(cmd::SetRenderTargets* u);
    uint32_t keep_masks_for(const cmd::SetRenderTargets& v) const {
        uint32_t keep = ~0u;
        for (int k = 0; k < cur_pass_rt_.count && k < 8; k++)
            if (cur_pass_rt_.colors[k] && !(k < v.count && v.colors[k])) keep &= ~(0xFu << (4 * k));
        return keep;
    }
    uint64_t prof_rt_merged_ = 0, prof_passes_merged_ = 0;  // perfil: cambios de targets sin cortar, pases con adjuntos de más
    // Perfil del tiler (por 300 frames).
    uint64_t prof_clear_loads_ = 0, prof_clear_flushes_ = 0, prof_store_none_ = 0, prof_area_cut_ = 0;
    uint64_t prof_barriers_deferred_ = 0, prof_barrier_splits_ = 0, prof_rt_kept_ = 0;
    uint64_t prof_split_storage_ = 0, prof_split_sample_ = 0;  // perfil: por qué se cortó (escribe storage, lee un adjunto)
    // Perfil de cortes (con registro): por qué se cerró cada pase y qué tiene que ver con él el siguiente. Lo dice quien
    // cierra el pase por su cuenta (split_why_: espejo, textura, barrera, unión de overlays…); si nadie lo dijo, el comando
    // del juego en curso (kCommandNames). Al empezar el siguiente (ensure_render_pass): si vuelve a abrir el mismo
    // framebuffer es un corte de verdad (el tiler guarda los adjuntos y los vuelve a cargar); si sus targets caben con los
    // del anterior en un pase (mismas texturas y vistas donde coinciden, la misma profundidad, el mismo tamaño) es un
    // cambio a un subconjunto o superconjunto, lo que junta merged_passes.
    const char* split_why_ = nullptr;
    const char* last_end_why_ = nullptr;
    const char* prof_pass_suffix_ = nullptr;  // profile_gpu: lo de antes del pase, en su etiqueta
    VkFramebuffer cur_fb_ = VK_NULL_HANDLE, last_end_fb_ = VK_NULL_HANDLE;
    cmd::SetRenderTargets cur_pass_rt_{}, last_end_rt_{};  // los adjuntos del pase abierto y los del último cerrado
    VkExtent2D last_end_extent_{};
    VkSampleCountFlagBits last_end_samples_ = VK_SAMPLE_COUNT_1_BIT;
    struct CutStat {
        uint64_t same = 0, fits = 0;  // vuelven a abrir el mismo framebuffer / caben con el anterior
    };
    std::map<std::string, CutStat> prof_cuts_;  // "WxH motivo" -> cortes desde el último perfil
    uint64_t prof_passes_same_ = 0, prof_passes_fit_ = 0;
    // Perfil de la escena: draws y anchos del viewport por tamaño del adjunto de profundidad (si la resolución dinámica
    // de MH Rise baja, baja el ancho mínimo del viewport de la de 960x540).
    struct DepthStat {
        uint64_t draws = 0;
        int vp_min = 1 << 30, vp_max = 0;
    };
    std::map<uint32_t, DepthStat> prof_depth_rts_;
    std::unordered_map<std::string, VkRenderPass> render_passes_;
    std::unordered_map<std::string, VkFramebuffer> framebuffers_;
    std::unordered_map<BytesKey<ViewKeyData>, VkImageView, BytesKeyHash<ViewKeyData>> views_;
    std::unordered_map<BytesKey<LayoutKeyData>, Layout, BytesKeyHash<LayoutKeyData>> layouts_;
    std::unordered_map<uint64_t, VkPipeline> pipelines_;
    std::unordered_map<uint64_t, VkPipeline> face_probe_pipelines_;
    unsigned face_probe_matches_ = 0, face_probe_saved_ = 0;
    struct DefaultImages { Texture one_d, two_d, three_d, cube; };
    DefaultImages sampled_defaults_, storage_defaults_;
    std::vector<uint64_t> bindless_key_;  // la clave del draw en curso (sin reservas por draw)
    struct HandleRun { uint32_t first, word; };
    struct HandleTable {
        uintptr_t address = 0;
        uint32_t count = 0, shift = 0, mask = 0, fixed = 0, left = 0;
        uint64_t pool_epoch = 0;
        std::vector<uint64_t> pages;
        std::vector<HandleRun> runs;
    };
    std::array<HandleTable, 32> handle_tables_;
    std::shared_ptr<const std::vector<TexturePoolEntry>> handle_textures_;
    std::shared_ptr<const std::vector<ResourceRef<Sampler>>> handle_samplers_;
    uint64_t handle_pool_epoch_ = 0;
    std::vector<std::pair<size_t, size_t>> handle_changes_;
    const std::vector<HandleRun>* handle_runs(uintptr_t address, uint32_t count, const Resource& r);
    std::map<std::vector<uint64_t>, VkDescriptorSetLayout> bindless_layouts_;  // (binding, count, etapa) de cada array
    bool bindless_ = true;        // <datos>/no_bindless: los handles de los uniform buffers con la de reserva, como antes
    bool sampler_compare_af_ = false, sampler_game_af_ = false, sampler_no_shadows_ = false;
    bool fixed_samplers_ = true;  // <datos>/no_fixed_sampler: el sampler de la ranura aunque el shader ORee uno fijo
    bool image_base_120_ = true;  // <datos>/image_base_128: las imágenes de gráficos desde c2[0x128], como antes
    VkSampler default_sampler_ = VK_NULL_HANDLE;
    uint32_t allocations_ = 0;
    BufferCache buffer_cache_;
    // Uniform/storage buffers ya copiados en el envío en curso: el juego no puede cambiar esa memoria
    // hasta que la GPU lo ejecute, así que varios draws que enlazan el mismo rango comparten la copia.
    // Se vacía al empezar cada envío y con CopyBuffer (que sí escribe memoria a mitad de envío).
    // Tabla plana (sin un nodo por copia: miles de malloc/free por frame) que se vacía cambiando de época.
    // Llena, no se apunta más: esos rangos se vuelven a copiar.
    // UpdateUniformBuffer también escribe a mitad de envío, pero en cada draw y unos pocos bytes: en vez de vaciarla,
    // invalidate() sube un contador por cada página de 4 KB escrita, y una copia guarda la suma de los de sus páginas al
    // hacerse: solo vale mientras esa suma no cambie (con la primera comparación barata: si no se ha invalidado nada
    // desde entonces, ni se suma). Una colisión de páginas en la tabla de contadores solo hace copiar de más.
    class CopiedTable {
    public:
        const std::pair<VkBuffer, VkDeviceSize>* find(uint64_t address, uint64_t size) {
            for (size_t i = slot(address, size), n = 0; n < kProbe; i = (i + 1) & (kSize - 1), n++) {
                Slot& e = slots_[i];
                if (e.epoch != epoch_) return nullptr;
                if (e.address == address && e.size == size) {
                    if (e.invalidations != invalidations_) {
                        if (stamp(address, size) != e.stamp) return nullptr;
                        e.invalidations = invalidations_;
                    }
                    return &e.value;
                }
            }
            return nullptr;
        }
        void insert(uint64_t address, uint64_t size, VkBuffer buffer, VkDeviceSize offset) {
            for (size_t i = slot(address, size), n = 0; n < kProbe; i = (i + 1) & (kSize - 1), n++) {
                Slot& e = slots_[i];
                if (e.epoch != epoch_ || (e.address == address && e.size == size)) {
                    e = {address, size, epoch_, stamp(address, size), invalidations_, {buffer, offset}};
                    return;
                }
            }
        }
        void clear() { epoch_++; }
        // La CPU (el ejecutor, por un comando) reescribió [address, address + size): las copias que lo tocan, fuera.
        void invalidate(uint64_t address, uint64_t size) {
            for (uint64_t p = address >> kPageShift, end = (address + std::max<uint64_t>(size, 1) - 1) >> kPageShift; p <= end; p++)
                writes_[page_slot(p)]++;
            invalidations_++;
        }

    private:
        static constexpr size_t kSize = 8192, kProbe = 32, kPageShift = 12, kPageSlots = 1 << 14;
        struct Slot {
            uint64_t address = 0, size = 0, epoch = 0, stamp = 0, invalidations = 0;
            std::pair<VkBuffer, VkDeviceSize> value{};
        };
        static size_t slot(uint64_t address, uint64_t size) {
            uint64_t h = (address ^ (size * 0x9E3779B97F4A7C15ULL)) * 0xff51afd7ed558ccdULL;
            return static_cast<size_t>(h >> 51) & (kSize - 1);
        }
        static size_t page_slot(uint64_t page) { return static_cast<size_t>((page * 0x9E3779B97F4A7C15ULL) >> 50); }
        uint64_t stamp(uint64_t address, uint64_t size) const {
            uint64_t sum = 0;
            for (uint64_t p = address >> kPageShift, end = (address + std::max<uint64_t>(size, 1) - 1) >> kPageShift; p <= end; p++)
                sum += writes_[page_slot(p)];
            return sum;
        }
        std::vector<Slot> slots_ = std::vector<Slot>(kSize);
        std::vector<uint32_t> writes_ = std::vector<uint32_t>(kPageSlots);  // escrituras por página (por hash)
        uint64_t epoch_ = 1, invalidations_ = 0;
    };
    CopiedTable copied_;
    // Sets 0 de gráficos ya escritos en este envío de Vulkan, por contenido: un draw con el mismo layout y lo mismo enlazado
    // (buffers y tamaños, vistas y samplers: lo que guarda Layout::memo_key; los desplazamientos van en el bind) reutiliza
    // el set aunque no sea el del draw anterior con ese layout: cascadas de sombras, el pre-pase y el pase de color con el
    // mismo material… (en la traza de Kamura, 173 de los 729 draws que escribían un set nuevo). Tabla plana que se vacía
    // cambiando de época (descriptor_epoch_: la de los pools, que begin() vacía); las claves van seguidas en keys_. Llena, no
    // se apunta más. <datos>/no_set_cache (en vivo): solo el del draw anterior, como antes.
    class SetTable {
    public:
        VkDescriptorSet find(uint64_t epoch, VkDescriptorSetLayout layout, const std::vector<uint64_t>& key, uint64_t h) const {
            if (epoch != keys_epoch_) return VK_NULL_HANDLE;
            for (size_t i = h & (kSize - 1), n = 0; n < kProbe; i = (i + 1) & (kSize - 1), n++) {
                const Slot& s = slots_[i];
                if (s.epoch != epoch) return VK_NULL_HANDLE;
                if (s.hash == h && s.layout == layout && s.len == key.size() &&
                    std::equal(key.begin(), key.end(), keys_.begin() + s.at))
                    return s.set;
            }
            return VK_NULL_HANDLE;
        }
        void insert(uint64_t epoch, VkDescriptorSetLayout layout, const std::vector<uint64_t>& key, uint64_t h,
                    VkDescriptorSet set) {
            if (epoch != keys_epoch_) keys_.clear(), keys_epoch_ = epoch;
            if (keys_.size() + key.size() > kMaxKeys) return;
            for (size_t i = h & (kSize - 1), n = 0; n < kProbe; i = (i + 1) & (kSize - 1), n++) {
                Slot& s = slots_[i];
                if (s.epoch == epoch) continue;
                s = {epoch, h, layout, static_cast<uint32_t>(keys_.size()), static_cast<uint32_t>(key.size()), set};
                keys_.insert(keys_.end(), key.begin(), key.end());
                return;
            }
        }

    private:
        static constexpr size_t kSize = 8192, kProbe = 16, kMaxKeys = size_t{1} << 20;
        struct Slot {
            uint64_t epoch = 0, hash = 0;
            VkDescriptorSetLayout layout = VK_NULL_HANDLE;
            uint32_t at = 0, len = 0;
            VkDescriptorSet set = VK_NULL_HANDLE;
        };
        std::vector<Slot> slots_ = std::vector<Slot>(kSize);
        std::vector<uint64_t> keys_;
        uint64_t keys_epoch_ = 0;
    };
    SetTable sets_cache_;
    bool cache_sets_ = true;   // interruptor no_set_cache (read_flags)
    bool templates_ = false;   // interruptor plantillas (read_flags; apagado por defecto)
    bool skip_set1_ = true;    // interruptor no_skip_set1 (read_flags)
    // Escribe el set con la plantilla del layout (la crea con la primera escritura): false si no se puede (sin la función,
    // sin plantilla o con otra firma), y entonces escribe quien llama con vkUpdateDescriptorSets.
    bool update_with_template(Layout& l, VkDescriptorSet set, const std::vector<VkWriteDescriptorSet>& writes);
    std::vector<uint32_t> template_signature_;  // la del draw en curso (sin reservas por draw)
    std::vector<uint8_t> template_data_;
    // Set 1 enlazado en cmd_ (y con qué layout) y el layout del último set 0: enlazar el set 0 con el mismo layout que el
    // anterior no toca el 1 (reglas de compatibilidad de Vulkan), así que el mismo set 1 no se vuelve a enlazar. begin() lo
    // olvida (command buffer nuevo).
    VkDescriptorSet bound_set1_ = VK_NULL_HANDLE;
    VkPipelineLayout bound_set1_layout_ = VK_NULL_HANDLE, bound_set0_layout_ = VK_NULL_HANDLE;
    uint64_t prof_copied_bytes_ = 0;
    // DEPURACIÓN (perfil): bytes copiados por (dirección, tamaño, si es SSBO y la etapa), para ver qué rangos pesan.
    std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> prof_copied_ranges_;  // clave -> (bytes, veces)
    void note_copy(uint64_t address, uint64_t size, bool ssbo, int stage) {
        if (size < 256 * 1024) return;
        auto& e = prof_copied_ranges_[address * 31 + size * 7 + ssbo * 2 + stage];
        e.first += size, e.second++;
        if (e.second == 1) prof_copied_keys_[address * 31 + size * 7 + ssbo * 2 + stage] = {address, size, ssbo, stage};
    }
    struct CopyKey { uint64_t address, size; bool ssbo; int stage; };
    std::unordered_map<uint64_t, CopyKey> prof_copied_keys_;
    // Lo que escriben los compute en storage buffers se queda en la GPU: cada rango escrito tiene una
    // copia persistente (overlay) que se enlaza tal cual en los SSBO/UBO que lo lean después, en vez de
    // esperar a la GPU y copiarlo de vuelta a la memoria del juego (5 vaciados de la GPU por frame en el
    // prólogo). La copia se crea con lo que hay en la memoria del juego y se descarta si la CPU la
    // reescribe (su huella deja de coincidir). Se alinea como la dirección original (módulo 256) para
    // que un buffer enlazado en una dirección alineada lo siga estando.
    struct Overlay {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;  // dónde está la dirección de inicio dentro de buffer
        uint8_t* data = nullptr;  // la misma memoria, mapeada (DEPURACIÓN: leer lo que escribió la GPU)
        uint64_t size = 0, cpu_hash = 0, checked = 0;
        // Envío de Vulkan que lo escribió por última vez; el de la última copia de lectura (al final de un envío que
        // lo escribió, la GPU lo copia a readback) y el último que se devolvió a la memoria del juego.
        uint64_t written = 0, readback_submit = 0, written_back = 0;
        VkBuffer readback = VK_NULL_HANDLE;
        VkDeviceSize readback_offset = 0;
        uint8_t* readback_data = nullptr;
        uint8_t writers = 0;  // kWrittenByGraphics | kWrittenByCompute | kWrittenByCopy
        int8_t readback_arena = -1;  // de cuál de readback_arenas_ es la copia de lectura
        // Su hueco en la arena (sin el desfase de la dirección): al tirarlo vuelve a overlay_free_ (ver make_overlay).
        VkDeviceSize slot_offset = 0, slot_size = 0;
        uint8_t* slot_data = nullptr;
        bool cpu_reads = false;  // destination of a GPU copy the game reads back on the CPU (copy_from_overlay)
        bool merged = false;  // hecho por merge_overlays (une rangos: que la CPU lo tire es lo que la unión puede empeorar)
        // <datos>/watch_overlays: sus páginas vigiladas (nnsdk/watch.h); overlay_unchanged mira si alguna se escribió
        // antes de pasar a la huella.
        bool watched = false;
        std::vector<uint64_t> watch_keys;
    };
    // Vigilancia de overlays (en vivo, con watch_overlays; apagada por defecto): cada overlay que se usa en un frame se
    // revisaba con la huella de su rango entero (los que escriben los compute miden megas). Vigilado, sin páginas escritas
    // por la CPU desde la última revisión, sigue igual sin huella; con alguna, la huella decide como antes. Solo los de más
    // de kOverlayWatchMin (los pequeños comparten página con lo que la CPU escribe en cada frame: saltaría cada vez).
    static constexpr uint64_t kOverlayWatchMin = 64 * 1024;
    bool watch_overlays_ = false;  // interruptor watch_overlays (read_flags)
    std::vector<std::pair<size_t, size_t>> overlay_watch_runs_;
    // kWrittenByCopy: lo llenó una copia en la GPU desde otro overlay (copy_from_overlay): tampoco está en la memoria del
    // juego. Solo para uniones y compactaciones (la devolución sigue con `written`, como antes).
    static constexpr uint8_t kWrittenByGraphics = 1, kWrittenByCompute = 2, kWrittenByCopy = 4;
    // Conservar lo de la GPU (ver merge_overlays y cpu_wrote_partial). Interruptores en vivo (read_flags):
    // <datos>/no_keep_overlays: todo como antes (sustituir y tirar); <datos>/no_merge_reads: solo las uniones
    // por una lectura (un SSBO o el origen de una copia que cae en parte en lo de la GPU), que meten en la unión bytes que
    // solo escribe la CPU: si la CPU los reescribe sin un comando, la unión entera se tira.
    bool keep_gpu_overlays_ = true, merge_reads_ = true;
    // Overlays de la GPU que un rango nuevo solapa solo en parte: se parten en vez de tirarse (merge_overlays).
    // <datos>/no_split_overlays (en vivo): como antes, se tiran y lo suyo sale de la memoria del juego.
    bool split_overlays_ = true;
    static constexpr uint64_t kMergeReadMax = 1ull << 20;  // unión por una lectura: hasta esto
    // Perfil (con registro, en "perfil overlays"): overlays escritos por la GPU tirados, por motivo (erase_reason; su
    // texto, el mismo que el de '[vigía] se tira el overlay'), y uniones y parches parciales. Siempre se cuentan, sin el
    // tope del log de watch_nan.
    struct LostCount {
        const char* why;
        uint64_t count, bytes;
    };
    std::vector<LostCount> prof_lost_;
    uint64_t prof_merged_ = 0, prof_merged_parts_ = 0, prof_merged_bytes_ = 0, prof_partial_patched_ = 0, prof_split_ = 0;
    void note_lost(const Overlay& o) {
        if (!o.writers) return;  // solo la copia de la memoria del juego: no se pierde nada
        for (LostCount& l : prof_lost_)
            if (l.why == erase_why_ || strcmp(l.why, erase_why_) == 0) {
                l.count++, l.bytes += o.size;
                return;
            }
        prof_lost_.push_back({erase_why_, 1, o.size});
    }
    // Motivos de erase_reason que comparten varios sitios (el perfil los suma por texto).
    static constexpr const char* kWhyReplaced = "lo sustituye un overlay nuevo que lo solapa (sale de la memoria del juego)";
    static constexpr const char* kWhyStraddled =
        "lo solapa solo en parte un rango escrito nuevo (¿memoria reutilizada?): se sustituye (sale de la memoria del juego)";
    static constexpr const char* kWhyPassOpen =
        "lo sustituye un overlay nuevo que lo solapa, con el pase abierto (sin unión; sale de la memoria del juego)";
    static constexpr const char* kWhyMergedRewrote =
        "la CPU reescribió parte de una unión de overlays (su huella cambió: lo de la GPU de dentro se va con ella)";
    const char* rewrote_why(const Overlay& o, const char* plain) const { return o.merged ? kWhyMergedRewrote : plain; }
    // La CPU del juego también lee lo que escribe la GPU (en la Switch es la misma memoria): MH Rise decide qué mips
    // carga por streaming con lo que sus shaders escriben en buffers, y sin devolverlo nunca pedía los mips grandes
    // (las texturas salían a cero: la ropa magenta). Al empezar cada envío del juego, los overlays pequeños que ya
    // escribió la GPU vuelven a su memoria. Solo con <datos>/use_return: estable (3/3 arranques), pero no arregla las caras y
    // el título baja de 30 a 18 fps (compara y huellas de overlays de megas en cada entrega, en el hilo del juego).
    static constexpr uint64_t kWriteBackMax = 16ull << 20;
    // El overlay que contiene address (make_overlay puede devolver uno mayor que ya lo cubría).
    void note_overlay_written(uint64_t address, uint8_t writer) {
        auto it = overlays_.upper_bound(address);
        if (it == overlays_.begin()) return;
        --it;
        if (address >= it->first + it->second.size) return;
        it->second.written = submitted_.load(std::memory_order_relaxed) + 1;
        it->second.writers |= writer;
        note_gpu_write_vs_mirrors(it->first, it->second.size, writer == kWrittenByCompute ? "SSBO de compute" : "SSBO de un draw");
        if (vigilar_nan_) {  // DEPURACIÓN
            Provenance& p = provenance_[it->first];
            p.last = writer_;
            p.last_frame = frame_;
            p.gpu_writes++;
            p.submit = it->second.written;
        }
    }
    bool record_overlay_readbacks();  // al empezar un envío, lo del anterior; true si grabó alguna copia
    bool readback_pending_ = false;
    // Las copias de lectura, en dos arenas que se alternan: cuando la actual pasa de 64 MB (overlays rehechos), se pasa
    // a la otra si no le quedan copias sin entregar, y se vacía (las de los overlays que aún la usaban se rehacen).
    Staging readback_arenas_[2];
    int readback_cur_ = 0;
    std::atomic<uint64_t> prof_written_back_{0};
    std::map<uint64_t, Overlay> overlays_;
    // Espejos en la GPU de buffers grandes que los shaders solo leen. MH Rise lee por frame SSBO de 24 y 18 MB que casi
    // no cambian (una página de 64 KB por frame en uno, nada en el otro) y se copiaban enteros al staging en cada draw
    // o compute que los enlazaba (~10 veces por frame, 270 MB); además, pasados los 16 MB de un trozo de staging se
    // recortaban y el shader leía fuera (la GPU se colgaba con DEVICE_LOST según lo que hubiera mapeado detrás). Cada
    // espejo se revisa por páginas una vez por envío del juego y solo se sube lo que cambió (en el command buffer, en
    // orden con lo que ya lo leía).
    struct Mirror {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        uint64_t size = 0;
        std::vector<uint64_t> pages;  // hash de cada página (vacío: sin subir), o las claves de la vigilancia
        bool watched = false;         // con vigilancia de escrituras (nnsdk/watch.h): sin hashes
        uint64_t checked = 0;         // submit_index_ de la última revisión
        uint64_t used_frame = 0, used_submit = 0;  // último frame y envío de Vulkan que lo usaron (para soltarlo)
        bool unread = false;  // revisado por preload_mirrors y sin leer aún en este envío (ver cpu_wrote_mirrors)
        // Direct mirrors (see promote_mirror): one the game rewrites whole frame after frame (MH Rise's 4 MB rings of
        // per-frame data, ~8 of them after a quest) gets three copies in memory the CPU writes and the GPU reads, used in
        // turn: each change is a memcpy into the copy no submit in flight is reading, and that one is bound. No staging,
        // no vkCmdCopyBuffer, no barriers, no render pass cut. buffer is the copy in use.
        static constexpr int kVersions = 3;
        bool direct = false;
        int hot = 0;  // checks in a row that rewrote (nearly) all of it
        int cur = 0;
        VkBuffer versions[kVersions]{};
        VkDeviceMemory version_memory[kVersions]{};
        uint8_t* version_data[kVersions]{};
        uint64_t version_submit[kVersions]{};  // the last submit that read each copy
    };
    static constexpr uint64_t kMirrorMin = 1ull << 20, kMirrorPage = 64ull << 10, kMirrorBudget = 384ull << 20;
    std::map<uint64_t, Mirror> mirrors_;  // por dirección de inicio
    std::vector<Mirror> retired_mirrors_;
    // DIAGNÓSTICO (sombras a cuadros de las Ruinas): lo que escribe la GPU (SSBO, CopyBuffer, render target, imagen) dentro
    // de un espejo. El espejo copia la memoria del juego, y lo de la GPU se queda en overlays e imágenes: quien lee el
    // espejo entero no lo ve. Sospechosa: la caché de sombras estáticas (0x5db12d8000+113 MB en la traza del 4-11, la lee
    // el primer draw de cada cascada, píxel 08ab44f0a529a808): en la cascada 0 los árboles solo salían en rectángulos.
    // Una línea por espejo y tipo de escritura, siempre (es una búsqueda en un map de ~50); de los de menos de 16 MB, 32.
    void note_gpu_write_vs_mirrors(uint64_t address, uint64_t size, const char* what);
    // A GPU-only mapping keeps its latest contents in the image until another backing or a
    // CPU visibility operation needs them. gpu_dirty pins the old pages until publication.
    using DirtyLevels = std::map<std::pair<uint32_t, uint32_t>, uint64_t>; // (mip, layer) -> last write
    uint64_t texture_write_order_ = 0;
    struct TexelReadback {
        Texture* texture;
        Texture::TexelWrite region;
        uint64_t order = 0;
        size_t row, slice, layer_bytes;
        uint8_t* data = nullptr;
        TexelReadback(Texture& t, const Texture::TexelWrite& r)
            : texture(&t), region(r), row(size_t{r.box.width} * t.layout.bytes_per_block),
              slice(row * r.box.height), layer_bytes(slice * r.box.depth) {}
        void record(VkCommandBuffer cmd, const Staging::Alloc& allocation);
        void publish() const;
    };
    bool prepare_texel_readback(Texture& t);
    bool readback_texels(std::vector<TexelReadback>& copies);
    std::map<void*, std::map<Texture*, DirtyLevels>> virtual_writes_;
    std::atomic<bool> virtual_pending_{false};
    bool cpu_publication_ok_ = true;
    void record_texture_write(Texture& t, const VkImageSubresourceRange& range, const char* what) {
        const uint32_t levels = upload_levels(t), layers = upload_layers(t);
        const size_t slots = size_t{levels} * layers;
        DirtyLevels* dirty = nullptr;
        const auto* format = t.virtual_storage ? format_info(t.desc.format) : nullptr;
        if (format && t.desc.samples <= 1 && t.aspect && !(t.aspect & (t.aspect - 1)) &&
            t.vk_format == format->vk && t.layout.bytes_per_block == format->bytes) {
            auto [it, inserted] = virtual_writes_.try_emplace(t.desc.pool);
            if (inserted) {
                auto* pool = g_memory_pools.get(t.desc.pool);
                std::lock_guard lock(pool->mapping_mutex);
                pool->gpu_dirty = true;
                virtual_pending_.store(true, std::memory_order_release);
            }
            dirty = &it->second[&t];
        }
        if (dirty || t.gpu_levels < slots || !t.astc_pending.empty()) {
            const uint64_t order = dirty ? ++texture_write_order_ : 0;
            for (uint32_t l = range.baseArrayLayer; l < layers && l - range.baseArrayLayer < range.layerCount; ++l)
                for (uint32_t m = range.baseMipLevel; m < levels && m - range.baseMipLevel < range.levelCount; ++m) {
                    const size_t slot = size_t{l} * levels + m;
                    t.set_level_owner(slot, slots, true);
                    if (dirty) (*dirty)[{m, l}] = order;
                    if (slot < t.astc_pending.size()) t.astc_pending[slot] = 0;
                }
        }
        ++t.content_version;
        if (t.storage && !mirrors_.empty())
            note_gpu_write_vs_mirrors(reinterpret_cast<uintptr_t>(t.storage), t.layout.total_size(), what);
        t.gpu_write_frame = frame_;
        if (!dirty && t.layout.linear && t.storage && t.returned < kReturnLinearMax &&
            std::find(linear_written_.begin(), linear_written_.end(), &t) == linear_written_.end()) {
            linear_written_.push_back(&t);
            static int logged = 0;
            if (logged++ < 16)
                NVN_LOG("linear texture %dx%d format 0x%x written by the GPU (%s): goes back to its memory at %p\n",
                        t.desc.width, t.desc.height, t.desc.format, what, static_cast<void*>(t.storage));
        }
    }
    // Linear textures the GPU wrote in this submit. On the Switch a linear texture is how the CPU reads a picture back
    // (the photo mode copies the frame into one and encodes it from its memory); here what the GPU writes stays in the
    // Vulkan image, so the photos came out black. Their level 0 goes back to the memory at the end of the submit. One that
    // keeps being written (more than kReturnLinearMax times) stops: that one isn't for the CPU.
    std::vector<Texture*> linear_written_;
    static constexpr uint32_t kReturnLinearMax = 600;
    void return_linear_textures();
    uint64_t mirror_bytes_ = 0, prof_mirror_bytes_ = 0, prof_mirrors_direct_ = 0, prof_direct_bytes_ = 0;
    // may_upload = false (con el render pass ya empezado): solo uno revisado en este envío.
    const Mirror* mirror(uint64_t address, uint64_t size, VkDeviceSize* offset, bool may_upload = true);
    // Los espejos del último frame, revisados al empezar cada envío (ver preload_mirrors); <datos>/no_mirrors_at_start: no.
    void preload_mirrors();
    // La CPU escribe [address, address + size) al reproducir (ClearBuffer, UpdateBuffer, CopyBuffer): un espejo que
    // preload_mirrors revisó y que nadie ha leído aún en este envío se vuelve a revisar al leerse, como si no se hubiera
    // adelantado (el primer draw que lo lee ve lo que había entonces, igual que antes).
    void cpu_wrote_mirrors(uint64_t address, uint64_t size) {
        if (mirrors_.empty() || !size) return;
        auto it = mirrors_.upper_bound(address);
        if (it != mirrors_.begin()) --it;
        for (; it != mirrors_.end() && it->first < address + size; ++it)
            if (it->second.unread && it->first + it->second.size > address) it->second.checked = 0, it->second.unread = false;
    }
    bool mirror_batch_ = false, mirror_batch_barrier_ = false;  // dentro de preload_mirrors: una barrera para todos
    uint64_t prof_mirror_preloads_ = 0;  // perfil: espejos con páginas subidas al empezar el envío
    void destroy_mirror(Mirror& m);
    bool promote_mirror(Mirror& m);  // the three copies of a direct mirror; false: it stays as it was
    // Multi-draw: no se sabe qué índices lee cada draw; su espejo es una ventana del memory pool que empieza en el
    // bloque de 8 MB de la dirección y sigue 24 MB (o hasta el final del pool): todos los de un bloque la comparten.
    static constexpr uint64_t kIndexWindowAlign = 8ull << 20, kIndexWindow = 24ull << 20;
    bool index_window(uint64_t address, uint64_t* base, uint64_t* size);
    // El memory pool que contiene address: [start, start + size). Cada multi-draw lo buscaba dos veces recorriendo
    // g_memory_pools entero (con su mutex; MH Rise reparte sus mallas en decenas de pools de 4 MB, y casi todos sus draws
    // son multi-draw). Los últimos encontrados se recuerdan mientras no se cree ni se destruya ningún pool
    // (Objects::generation). <datos>/no_pool_memo (en vivo): se recorre siempre, como antes.
    bool pool_range(uint64_t address, uint64_t* start, uint64_t* size);
    struct PoolMemo {
        uint64_t start = 0, size = 0;
    };
    static constexpr int kPoolMemo = 8;
    PoolMemo pool_memo_[kPoolMemo];
    uint64_t pool_memo_generation_ = ~0ull;
    int pool_memo_next_ = 0;
    void reap_mirrors();
    // Cuántos overlays tocan cada bloque de 64 KB (por hash: una colisión solo hace mirar el mapa). Casi todas las
    // búsquedas fallan, y así se descartan sin recorrer el mapa. Todo cambio de overlays_ pasa por aquí.
    uint32_t overlay_filter_[1 << 16] = {};
    static size_t overlay_slot(uint64_t granule) { return static_cast<size_t>((granule * 0x9E3779B97F4A7C15ULL) >> 48); }
    void filter_overlay(uint64_t address, uint64_t size, int delta) {
        for (uint64_t g = address >> 16, end = (address + std::max<uint64_t>(size, 1) - 1) >> 16; g <= end; g++)
            overlay_filter_[overlay_slot(g)] += delta;
    }
    // DEPURACIÓN (watch_nan): por qué se tira el overlay que va a tirar erase_overlay y qué lo pidió (lo pone quien
    // llama). Si la GPU lo había escrito, note_erase lo dice antes de borrar su procedencia: lo suyo se pierde y quien lo
    // lea después ve la memoria del juego (o el overlay nuevo que lo sustituye, hecho con ella). Las tiradas sin log de
    // HEAD (huella cambiada, comandos de la CPU que no caben enteros en un overlay) quedan así a la vista.
    const char* erase_why_ = "?";
    uint64_t erase_by_ = 0, erase_by_size_ = 0;
    void erase_reason(const char* why, uint64_t by = 0, uint64_t by_size = 0) {
        erase_why_ = why;
        erase_by_ = by;
        erase_by_size_ = by_size;
    }
    void note_erase(uint64_t start, const Overlay& o);
    // lost = false: lo suyo ya está en otro overlay (una unión lo copió): ni se cuenta ni se dice que se pierde.
    std::map<uint64_t, Overlay>::iterator erase_overlay(std::map<uint64_t, Overlay>::iterator it, bool lost = true) {
        {
            std::lock_guard lock(wb_mutex_);
            wb_state_.erase(it->first);
            delivered_hash_.erase(it->first);
        }
        filter_overlay(it->first, it->second.size, -1);
        if (lost) note_lost(it->second);
        if (vigilar_nan_ && lost) note_erase(it->first, it->second);  // DEPURACIÓN
        if (!provenance_.empty()) provenance_.erase(it->first);  // DEPURACIÓN (watch_nan)
        // Su hueco, libre cuando la GPU acabe el envío que se graba ahora (puede que ya lo use).
        if (const Overlay& o = it->second; o.slot_size && recycle_overlays_) {
            if (overlay_free_.size() >= kMaxFreeSlots) overlay_free_.erase(overlay_free_.begin());  // el más viejo se pierde
            overlay_free_.push_back({o.buffer, o.slot_offset, o.slot_size, o.slot_data, submitted_.load(std::memory_order_relaxed) + 1});
        }
        return overlays_.erase(it);
    }
    void compact_overlays();
    // Los overlays que tocan [address, address + size), fuera (la CPU acaba de escribir ahí: manda la memoria del juego).
    // Casi nunca hay ninguno: el filtro lo descarta sin mirar el mapa. Los overlays no se solapan entre sí.
    void drop_overlays(uint64_t address, uint64_t size) {
        if (!overlay_maybe_in(address, size)) return;
        for (auto it = first_overlay_touching(address); it != overlays_.end() && it->first < address + size;)
            it = erase_overlay(it);
    }
    // ¿Puede haber un overlay en [address, address + size)? false: seguro que no (el filtro, sin mirar el mapa).
    bool overlay_maybe_in(uint64_t address, uint64_t size) const {
        if (overlays_.empty()) return false;
        for (uint64_t g = address >> 16, end = (address + std::max<uint64_t>(size, 1) - 1) >> 16; g <= end; g++)
            if (overlay_filter_[overlay_slot(g)]) return true;
        return false;
    }
    // El primer overlay que puede tocar [address, …): el que empieza antes y llega a address, o el siguiente.
    std::map<uint64_t, Overlay>::iterator first_overlay_touching(uint64_t address) {
        auto it = overlays_.upper_bound(address);
        if (it != overlays_.begin() && std::prev(it)->first + std::prev(it)->second.size > address) --it;
        return it;
    }
    // Hueco para un overlay de need bytes (ver make_overlay); false sin sitio. size: para el perfil.
    bool alloc_overlay_slot(VkDeviceSize need, Staging::Alloc* a, VkDeviceSize* slot_size, uint64_t size);
    void clear_overlays() {
        overlays_.clear();
        overlay_free_.clear();  // sus huecos son de la arena que se vacía
        memset(overlay_filter_, 0, sizeof overlay_filter_);
    }
    Staging overlay_arena_;
    VkDeviceSize overlay_used_ = 0;
    // Huecos de overlays tirados (la CPU reescribió su rango, un rango nuevo los solapó…) para los overlays nuevos. Con la
    // arena solo lineal, en una cacería MH Rise rehacía ~16 MB de overlays por frame: la arena (64 MB) se llenaba cada 2-3
    // frames y cada compactación esperaba a que la GPU acabara todo y leía ~26 MB de memoria no cacheable (cap1.log:
    // 12,8 ms/frame esperando a la GPU, 50 ms/frame en el ejecutor, 14-20 fps). Un hueco se reutiliza cuando la GPU ha
    // terminado el envío en el que se tiró: nada de lo grabado hasta entonces lo vuelve a leer.
    // <datos>/no_overlay_gaps: solo la arena, como antes.
    struct FreeSlot {
        VkBuffer buffer;
        VkDeviceSize offset, size;
        uint8_t* data;
        uint64_t free_after;  // envío de Vulkan tras el que ya no lo lee nadie
    };
    static constexpr size_t kMaxFreeSlots = 1024;
    std::vector<FreeSlot> overlay_free_;
    bool recycle_overlays_ = true;  // interruptor no_overlay_gaps (read_flags)
    bool patch_overlays_ = true;    // interruptor no_patch_overlays (read_flags; ver overlay_containing)
    bool write_back_on_ = false;    // use_return (read_flags): sin él no hace falta la referencia de cada overlay
    // Without use_return, still return the destinations of GPU copies out of GPU-written buffers (Overlay::cpu_reads):
    // that's how the game reads results back. MH Rise's occlusion test writes which objects it saw from a pixel shader,
    // copies that to another buffer and checks it on the CPU the next frame; without it the CPU saw zeros, and endemic
    // life and wirebugs were never drawn (you could still gather them). <datos>/no_readback_copies: as before.
    bool readback_copies_ = true, any_cpu_reads_ = false;
    // Perfil: bytes de overlays nuevos (de la arena) y en huecos reutilizados; compactaciones y lo que costaron.
    uint64_t prof_overlay_new_bytes_ = 0, prof_overlay_reused_bytes_ = 0;
    uint64_t prof_compactions_ = 0, prof_compact_wait_us_ = 0, prof_compact_copy_us_ = 0;
    // Overlay que contiene [address, address+size), con el desplazamiento de address en su buffer.
    const Overlay* find_overlay(uint64_t address, uint64_t size, VkDeviceSize* offset);
    // DEPURACIÓN (watch_nan): el overlay que toca [address, address + size) sin contenerlo entero (llamar si
    // find_overlay falló), o nullptr.
    const std::pair<const uint64_t, Overlay>* partial_overlay(uint64_t address, uint64_t size);
    // ¿Sigue mandando la GPU en el overlay de address? No si la CPU reescribió su rango (su huella cambió). Si la memoria
    // es justo lo que le entregamos, esa pasa a ser su huella. Una entrega a medias (desde el hilo del juego, mientras
    // aquí se calcula la huella) daba una huella mezclada: parecía una escritura de la CPU, el overlay se tiraba y lo
    // escrito por la GPU se perdía (VK_ERROR_DEVICE_LOST al poco). Con wb_mutex_, que la entrega tiene mientras copia,
    // se vuelve a mirar.
    bool overlay_unchanged(uint64_t address, Overlay& o);
    // El overlay de un rango que escribe un shader (lo crea si hace falta). may_copy = false con un render pass abierto
    // o con descriptores ya grabados: sin copias en la GPU, así que no se une con los que solapa (se sustituyen, como antes).
    const Overlay* make_overlay(uint64_t address, uint64_t size, VkDeviceSize* offset, bool may_copy = true);
    // Un overlay nuevo de [address, address+size) con lo de los overlays de la GPU que caen enteros dentro copiado encima,
    // en la GPU (cierra el render pass). nullptr si no hace falta (ninguno que conservar) o no se puede (tope, sin sitio,
    // no_keep_overlays). for_read: lo pide una lectura (hasta kMergeReadMax, y sin tirar nada de la GPU).
    const Overlay* merge_overlays(uint64_t address, uint64_t size, VkDeviceSize* offset, bool for_read);
    // Antes del render pass de un draw: los overlays de lo que escribe (y de los SSBO que lee y caen en parte en lo de la
    // GPU), unidos ya, para que dentro del pase make_overlay/find_overlay los encuentren sin copiar nada.
    void prepare_draw_overlays();
    // Lo mismo antes de los descriptores de un dispatch: una unión en su bucle tiraría el overlay al que apunta el
    // descriptor ya grabado de otro SSBO del mismo dispatch (lo que escribiera ahí se perdería).
    uint32_t prepare_dispatch_overlays();
    void wb_reference(uint64_t address, uint64_t size);  // con wb_mutex_
    // CopyBuffer/ClearBuffer/UpdateBuffer de la CPU: el overlay que contiene entero [address, address+size) y sigue al
    // día con la memoria del juego (mirado ANTES de escribirla), o overlays_.end().
    std::map<uint64_t, Overlay>::iterator overlay_containing(uint64_t address, uint64_t size);
    // Tras escribir la CPU ese rango en la memoria del juego: lo mismo en el overlay, en la GPU y en orden.
    void patch_overlay(std::map<uint64_t, Overlay>::iterator it, uint64_t address, uint64_t size);
    // La CPU (un comando) escribe [address, address+size) y no cae entero en un overlay: antes de escribir, los que toca,
    // con lo de la GPU y aún al día (sus direcciones); después, cpu_wrote_partial les copia la parte escrita y conserva el
    // resto de lo de la GPU (antes se tiraban enteros). Los demás que toca, fuera, como antes: la CPU ya los había
    // reescrito (manda la CPU) o no tienen nada de la GPU.
    std::vector<uint64_t> overlays_before_cpu_write(uint64_t address, uint64_t size);
    void cpu_wrote_partial(const std::vector<uint64_t>& kept, uint64_t address, uint64_t size);
    // CopyBuffer cuyo origen está en un overlay: copia en la GPU a un overlay del destino.
    enum class BufferCopySource { Guest, Snapshot, Copied, Error };
    BufferCopySource copy_from_overlay(uint64_t src, uint64_t dst, uint64_t size, std::vector<uint8_t>& snapshot);
    Staging::Alloc alloc_read_snapshot(uint64_t address, uint64_t size);
    bool read_vertex_stream(BufferRange range, bool allow_mirror, VkBuffer& buffer, VkDeviceSize& offset, bool& gpu_vertices);
    // Called outside the pass. A null buffer on success means the CPU owns the entire range.
    bool gather_overlays(uint64_t address, uint64_t size, VkBuffer* buffer, VkDeviceSize* offset, const uint8_t** host = nullptr);
    uint64_t submit_index_ = 0;  // envíos ejecutados: la caché de vértices revisa una vez por envío
    std::vector<Texture*> frame_targets_, frame_uploads_;  // para los volcados de depuración
    void dump(Texture& t, const std::string& name, uint32_t layer = 0, int forced_mip = -1);
    // DEPURACIÓN (<datos>/check_astc): tras subir una ASTC, cada mip (hasta 6) pasa a RGBA8 en la GPU y se cuentan los
    // píxeles exactamente magenta (el color de error de un bloque ASTC inválido): datos mal desentrelazados.
    void check_astc(Texture& t);
    bool dump_raw(Texture& t, const std::string& name, const std::string& dir = "/storage/emulated/0/NieR-volcados");
    void dump_volume(Texture& t, const std::string& name);  // DEPURACIÓN: nivel 0 entero de una 3D/array/cubemap
    // layer < 0: las primeras capas de un array (hasta 4: las cascadas de las sombras), cada una en <name>_c<capa>.
    void dump_depth(Texture& t, const std::string& name, int layer = -1);
    void trace_draw(const cmd::Draw& d);  // DEPURACIÓN: estado completo de cada draw del frame trazado
    void dispatch(const cmd::Dispatch& v);
    VkPipeline compute_pipeline(const ProgramStage& cs, Layout& l);
    std::unordered_map<uint64_t, VkPipeline> compute_pipelines_;  // por la clave (como pipelines_)
    // Claim each key once between the warm-up coordinator and the serialized submitter.
    struct PipelineBuild {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkResult status = VK_SUCCESS;
        uint64_t driver_ns = 0;
        spirv_cache::Key modules[kStages]{};  // on failure, identify the variants already attempted
    };
    using PipelineFuture = std::shared_future<PipelineBuild>;
    std::mutex warm_mutex_;
    std::unordered_map<uint64_t, PipelineFuture> warm_builds_;
    std::unordered_set<uint64_t> warm_done_;
    bool warm_started_ = false;             // solo el hilo del ejecutor
    uint64_t prof_warm_taken_ = 0, prof_warm_waits_ = 0;
    std::optional<PipelineBuild> warm_take(uint64_t h);
    PipelineBuild await_pipeline(const PipelineFuture& build);
    void warm_up();

public:
    // Precarga (program.cpp): calienta la caché del driver para un compute recién traducido.
    void prefetch_compute(const ProgramStage& cs);

private:
    // The submitter keeps the modules, layout and pass alive until the worker publishes its VkResult.
    using PipelineTask = std::packaged_task<PipelineBuild()>;
    std::mutex async_mutex_;
    std::condition_variable async_cv_;
    std::deque<PipelineTask> async_queue_;
    std::once_flag pipeline_workers_once_;
    PipelineFuture enqueue_pipeline(PipelineTask task, bool first_use);
    bool async_pipelines_ = true;                           // read_flags
    bool serial_pipelines_ = false;                         // init: PowerVR (o single_thread_pipelines), todo en este hilo
    bool low_table_ = true;                                 // read_flags: la tabla baja de c2 (resolve_texture)
    bool separate_tables_ = true;  // read_flags (<datos>/no_split_tables: un array, como antes): StageBindings::combined_textures
    void async_loop();
    std::vector<PipelineRecord> load_pipeline_records();
    // Registros de pipelines nuevas para <datos>/pipelines_estado.bin: se añaden desde otro hilo (save_pipeline_records).
    std::mutex records_mutex_;
    std::vector<PipelineRecord> records_pending_;
    std::unordered_set<uint64_t> records_known_;  // claves (hash) que ya están en el fichero
    bool records_loaded_ = false;                 // leído (y rehecho si no valía): ya se puede añadir
    bool records_writing_ = false;
    void record_pipeline(const PipelineRecord& r);
    void save_pipeline_records();
    // Pipelines already in pipelines_estado.bin that the warm-up didn't make (it stops at kWarmMax) went to the
    // background threads as if they were new, and their draws were skipped for 2-4 frames even though the driver had
    // them cached (~0.25 ms each): objects blinked after loading a save or reaching an area. They're made here in
    // NvnSubmit now, up to kKnownBudgetUs a frame (past that, to the threads as before). One that takes as long as a
    // real compile means the driver's cache didn't have it; after three of those, this session goes back to the threads.
    static constexpr uint64_t kKnownBudgetUs = 8000, kKnownSlowUs = 20000;
    uint64_t known_frame_ = UINT64_MAX, known_us_ = 0;  // (executor thread only)
    int known_slow_ = 0;
    bool known_off_ = false;
    bool pipeline_known(uint64_t h) {
        std::lock_guard lock(records_mutex_);
        return records_loaded_ && records_known_.count(h) != 0;
    }
    // The keys of the pipelines the last session used (<data>/pipelines_recent.bin), so the warm-up starts with what
    // the player will see first instead of the title screen and the first areas. Rewritten when there are new ones.
    size_t recent_written_ = 0;  // pipelines_.size() at the last write
    std::atomic<bool> recent_writing_{false};
    std::vector<uint64_t> recent_prev_;  // the file's at start-up (records_mutex_): kept behind this session's
    void save_recent_pipelines();
    std::vector<uint64_t> load_recent_pipelines();
    // Lo que comparte el hilo del precalentado con el ejecutor: layouts (y los del set 1) y render passes.
    std::mutex objects_mutex_;
    // Perfil (con registro): claves por etapa (dirección + generación, como antes) de los draws: las pipelines que se
    // habrían creado sin compartirlas entre programas con el mismo SPIR-V.
    std::unordered_set<uint64_t> prof_stage_keys_;
    bool prof_count_stage_keys_ = false;
    bool per_stage_keys_ = false;  // interruptor pipelines_per_stage: las claves de antes (sin compartir entre etapas)
    VkFormat rp_formats_[8] = {}, rp_depth_ = VK_FORMAT_UNDEFINED;  // los del pase empezado (clave de pipeline)
    // Frame trazado: el que sigue a la aparición de <datos>/trazar (se borra al verlo); ninguno si no.
    uint64_t trace_frame_ = UINT64_MAX;
    int traced_draws_ = 0;
    // Armed by a one-shot report request. Do not split the render pass to observe a draw.
    bool loading_armed_ = false;
    struct AtlasBinding {
        uint32_t stage, set, binding, element;
        const Texture* texture;
        const Sampler* sampler;
        TextureViewDesc view;
        VkImageView image_view;
    };
    std::vector<AtlasBinding> loading_matches_;
    FILE* loading_log_ = nullptr;
    std::string loading_dir_;
    std::set<uint64_t> loading_stages_;
    uint64_t loading_pipeline_key_ = 0;
    uint8_t loading_blend_enable_ = 0;
    void check_loading_binding(const Texture& t, const TextureViewDesc& view, VkImageView image_view,
                               uint32_t stage, const Resource& r, uint32_t element, const Sampler* smp);
    void record_loading_draw(const cmd::Draw& d, VkPipeline pipeline, const cmd::SetRenderTargets& attachments);
    // Perfil: tiempo total en run() (incluye la espera a la GPU) y en las esperas a la GPU.
    uint64_t prof_run_us_ = 0, prof_gpu_wait_us_ = 0, prof_submits_ = 0, prof_draws_ = 0;
    // Los tramos de cada draw (~12 relojes por draw) solo se miden en uno de cada kProfileEvery frames; el log
    // los da por frame medido, con el mismo sentido que antes.
    static constexpr uint64_t kProfileEvery = 8;
    bool prof_timing_ = false;                   // el frame en curso se mide
    uint64_t prof_timed_frames_ = 0, prof_timed_frame_ = UINT64_MAX;
    std::chrono::steady_clock::time_point prof_now() const {
        return prof_timing_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    }
    uint64_t prof_section_us_[6] = {};  // en ns (en µs, cada tramo perdía ~0,5 µs por draw)
    uint64_t prof_sub_us_[4] = {};  // descriptores: buffers, texturas, actualizar+enlazar, reservar set
    uint64_t prof_dispatch_us_ = 0, prof_draw_total_us_ = 0;
    void prof_mark_sub(int k, std::chrono::steady_clock::time_point& t) {
        if (!prof_timing_) return;
        auto now = std::chrono::steady_clock::now();
        prof_sub_us_[k] += std::chrono::duration_cast<std::chrono::nanoseconds>(now - t).count();
        t = now;
    }
    uint64_t prof_render_passes_ = 0, prof_writebacks_ = 0, prof_dispatches_ = 0;
    uint64_t prof_bindless_sets_ = 0, prof_bindless_textures_ = 0, prof_bindless_us_ = 0;  // set 1 armados, sus texturas
    uint64_t prof_bindless_reused_ = 0;
    // Perfil fino (con registro, en los frames medidos como prof_mark; por frame medido en "perfil fino"): lo que los tramos
    // juntan. Huellas de las texturas que sube la CPU (cpu_texture_changed), de la caché de vértices y de los overlays;
    // búsquedas de memory pools (index_window: cada multi-draw); sets 0 de los draws (reutilizados por el del draw anterior,
    // por la tabla de sets_cache_ o nuevos, y lo que cuesta reservarlos, escribirlos y enlazarlos); pipelines y layouts.
    struct FineProf {
        uint64_t tex_checks = 0, tex_hashes = 0, tex_hash_bytes = 0, tex_hash_ns = 0;
        uint64_t ov_hashes = 0, ov_hash_bytes = 0, ov_hash_ns = 0, ov_watched = 0;
        uint64_t pool_scans = 0, pool_visits = 0, pool_memo = 0;
        uint64_t sets_memo = 0, sets_map = 0, sets_new = 0, set_writes = 0, set_templates = 0;
        uint64_t set_alloc_ns = 0, set_update_ns = 0, set_bind_ns = 0, set1_skipped = 0;
        uint64_t pipe_lookups = 0, pipe_memo = 0, pipe_binds = 0, layout_memo = 0, bindless_fast = 0;
        BufferCache::Prof vtx;
    };
    FineProf fine_;
    // Tramos de "pipeline" (overlays y espejos, set 1, render pass, layout y pipeline) y de "vértices" (índices, streams), ns.
    uint64_t prof_split_ns_[6] = {};
    void prof_split(int k, std::chrono::steady_clock::time_point& t) {
        if (!prof_timing_) return;
        auto now = std::chrono::steady_clock::now();
        prof_split_ns_[k] += std::chrono::duration_cast<std::chrono::nanoseconds>(now - t).count();
        t = now;
    }
    static uint64_t ns_since(std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    }
    std::map<uint64_t, uint64_t> gpu_written_ranges_;  // DEPURACIÓN: [inicio, fin) escritos por compute
    uint64_t prof_reads_[6] = {};                       // lecturas de eso: vértices, índices, ubo, ssbo, indirecto, cs-ssbo
    void note_read(int kind, uint64_t a, uint64_t size) {
        auto it = gpu_written_ranges_.upper_bound(a);
        if (it == gpu_written_ranges_.begin()) return;
        --it;
        if (a < it->second && a + size > it->first) {
            prof_reads_[kind]++;
            if (a < it->first || a + size > it->second) prof_partial_++;
        }
    }
    uint64_t prof_partial_ = 0;
    uint64_t prof_texture_bytes_ = 0;
    uint64_t prof_disabled_draws_ = 0;  // DEPURACIÓN: draws descartados por SetRenderEnable(false) en la traza
    void prof_mark(int section, std::chrono::steady_clock::time_point& t) {
        if (!prof_timing_) return;
        auto now = std::chrono::steady_clock::now();
        prof_section_us_[section] += std::chrono::duration_cast<std::chrono::nanoseconds>(now - t).count();
        t = now;
    }
    uint32_t trace_counts_[std::variant_size_v<Command>] = {};

    // --- Perfil de cargas ---
    // Lo que solo pasa al cargar (subir texturas, crear imágenes, vistas, espejos, pipelines…) se mide siempre: son pocas
    // veces por frame. El perfil por draw (prof_mark) solo mide 1 de cada kProfileEvery frames y el tirón del selector de
    // personaje (1,5 s en un envío del 4-11) cayó en uno sin medir: no se sabía en qué se iba. Cada LoadTimer suma su
    // tiempo exclusivo (sin el de los que se anidan dentro) en load_run_; al acabar cada envío del juego (run), si pasa de
    // kSlowRunMs y hay registro, sale "envío lento" con el desglose ordenado y el resto sin medir; profile_frame da lo del
    // periodo ("perfil cargas").
    enum LoadKind {
        kLdUpload, kLdDeswizzle, kLdImage, kLdView, kLdMirrorNew, kLdMirrorCheck, kLdMirrorCopy, kLdShader, kLdPipeline,
        kLdWarm, kLdBindless, kLdOverlay, kLdFramebuffer, kLdSampler, kLdReap, kLdGpuWait, kLdSubmit, kLdKinds
    };
    static constexpr const char* kLoadNames[kLdKinds] = {
        "subidas de texturas (sin el deswizzle)", "deswizzle", "imágenes nuevas", "vistas nuevas", "espejos nuevos",
        "revisión de espejos", "copia a espejos", "shaders traducidos", "pipelines del driver", "precalentado (tomar)",
        "bindless", "overlays", "framebuffers", "samplers", "liberar lo retirado", "espera a la GPU", "vkQueueSubmit"};
    static constexpr double kSlowRunMs = 50;
    struct LoadProf {
        uint64_t ns[kLdKinds] = {}, n[kLdKinds] = {}, bytes[kLdKinds] = {};
        uint64_t protect_ns = 0, protects = 0, staging_ns = 0, staging_new = 0, staging_bytes = 0, staging_reused = 0;
        void add(const LoadProf& o) {
            for (int k = 0; k < kLdKinds; k++) ns[k] += o.ns[k], n[k] += o.n[k], bytes[k] += o.bytes[k];
            protect_ns += o.protect_ns, protects += o.protects, staging_ns += o.staging_ns, staging_new += o.staging_new;
            staging_bytes += o.staging_bytes, staging_reused += o.staging_reused;
        }
    };
    LoadProf load_run_, load_period_;
    uint64_t load_period_runs_ = 0, load_period_slow_ = 0, load_period_run_ns_ = 0, load_period_worst_ns_ = 0;
    struct LoadTimer {
        Executor& e;
        LoadKind kind;
        std::chrono::steady_clock::time_point t0;
        LoadTimer* parent;
        uint64_t child_ns = 0;
        LoadTimer(Executor& ex, LoadKind k, uint64_t bytes = 0)
            : e(ex), kind(k), t0(std::chrono::steady_clock::now()), parent(ex.load_top_) {
            e.load_top_ = this;
            e.load_run_.n[k]++;
            e.load_run_.bytes[k] += bytes;
        }
        ~LoadTimer() {
            const uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
            e.load_run_.ns[kind] += ns > child_ns ? ns - child_ns : 0;
            if (parent) parent->child_ns += ns;
            e.load_top_ = parent;
        }
        LoadTimer(const LoadTimer&) = delete;
        LoadTimer& operator=(const LoadTimer&) = delete;
    };
    LoadTimer* load_top_ = nullptr;
    std::string load_summary(const LoadProf& p, uint64_t total_ns) const;

public:
    uint64_t frame_ = 0;
    uint64_t frame_run_ns_ = 0;  // los envíos de este frame (para g_drop_stats al presentar)
    int64_t game_gpu_ns() const { return frame_gpu_ns_.load(std::memory_order_relaxed); }

private:
    uint64_t draws_ = 0, skipped_draws_ = 0, dispatches_ = 0, gpu_dispatches_ = 0;
    // Skipped draws by why (the verbose log's draw line lists them): tess, multi, absurd_ind, absurd_dir, no_vs,
    // bindless, pass, pipeline, set, count_set, gpu_data, indices
    uint64_t skipped_why_[12] = {};
    void skip_draw(int why) {
        skipped_draws_++;
        skipped_why_[why]++;
    }
};

// El desglose ordenado de más a menos (lo de menos de 0,5 ms no sale) y el resto sin medir (draws, dispatches, estado…).
std::string Executor::load_summary(const LoadProf& p, uint64_t total_ns) const {
    int order[kLdKinds];
    for (int k = 0; k < kLdKinds; k++) order[k] = k;
    std::sort(order, order + kLdKinds, [&](int a, int b) { return p.ns[a] > p.ns[b]; });
    std::string out;
    uint64_t measured = 0;
    char b[192];
    for (int k : order) {
        measured += p.ns[k];
        if (p.ns[k] < 500000) continue;
        if (p.bytes[k])
            snprintf(b, sizeof b, "%s%s %.0f ms (%llu, %.1f MB)", out.empty() ? "" : ", ", kLoadNames[k], p.ns[k] / 1e6,
                     static_cast<unsigned long long>(p.n[k]), p.bytes[k] / 1048576.0);
        else
            snprintf(b, sizeof b, "%s%s %.0f ms (%llu)", out.empty() ? "" : ", ", kLoadNames[k], p.ns[k] / 1e6,
                     static_cast<unsigned long long>(p.n[k]));
        out += b;
    }
    snprintf(b, sizeof b, "; resto sin medir %.0f ms. De todo eso: mprotect %.0f ms en %llu llamadas (todos los hilos), "
             "staging nuevo %.0f ms (%llu trozos, %llu MB; %llu de la reserva)",
             total_ns > measured ? (total_ns - measured) / 1e6 : 0.0, p.protect_ns / 1e6,
             static_cast<unsigned long long>(p.protects), p.staging_ns / 1e6, static_cast<unsigned long long>(p.staging_new),
             static_cast<unsigned long long>(p.staging_bytes >> 20), static_cast<unsigned long long>(p.staging_reused));
    out += b;
    return out;
}

// Nombres de los comandos (el orden de Command en commands.h): el perfil de cortes dice cuál cerró un pase.
constexpr const char* kCommandNames[] = {
    "SetRenderTargets", "ClearColor", "ClearDepthStencil", "Viewport", "Scissor", "DepthRange", "BindProgram",
    "BindVertexBuffers", "BindUniformBuffers", "BindStorageBuffer", "BindTexture", "BindSampler", "BlendState",
    "ColorState", "ChannelMaskState", "DepthStencilState", "PolygonState", "MultisampleState", "BindVertexAttribs",
    "BindVertexStreams", "SetPools", "Stencil", "Draw", "Dispatch", "CopyBuffer", "CopyTexture", "CopyBufferToTexture",
    "Barrier", "ReportCounter", "BlendColor", "RenderEnable", "ReportValue", "ClearBuffer", "BindImage", "ClearTexture",
    "UpdateBuffer", "PrimitiveRestart", "FenceSync", "PolygonOffset", "PatchSize"};
static_assert(std::size(kCommandNames) == std::variant_size_v<Command>);

bool Executor::init() {
    VkContext* c = vk();
    if (!c) return false;
    driver_cache_.load();
    // PowerVR: su compilador de shaders se cae creando pipelines en varios hilos a la vez (lo vieron los ports de MHGU y
    // MH4U: el juego se cerraba, y en cada arranque siguiente al precalentar los guardados). Ahí todo en el hilo del
    // ejecutor, cuando hace falta: sin precalentado y sin los hilos de pipelines async. <datos>/single_thread_pipelines hace
    // lo mismo en cualquier GPU; <datos>/parallel_pipelines lo quita también en una PowerVR.
    serial_pipelines_ = NVN_FLAG("single_thread_pipelines") ||
                        (strstr(c->props.deviceName, "PowerVR") && !NVN_FLAG("parallel_pipelines"));
    if (serial_pipelines_) NVN_LOG("pipelines en un solo hilo (%s): sin precalentado ni async\n", c->props.deviceName);
    warm_started_ = !serial_pipelines_;
    if (warm_started_) std::thread([this] { warm_up(); }).detach();  // vive lo que el proceso (como el de NvnSubmit)
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = c->queue_family;
    vkCreateCommandPool(c->device, &pci, nullptr, &pool_);
    for (InFlight& f : inflight_) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool_;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        vkAllocateCommandBuffers(c->device, &ai, &f.cmd);
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        vkCreateFence(c->device, &fi, nullptr, &f.fence);
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = kStampsPerSlot;
        if (vkCreateQueryPool(c->device, &qi, nullptr, &f.queries) != VK_SUCCESS) f.queries = VK_NULL_HANDLE;
    }
    {
        uint32_t n = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(c->gpu, &n, nullptr);
        std::vector<VkQueueFamilyProperties> families(n);
        vkGetPhysicalDeviceQueueFamilyProperties(c->gpu, &n, families.data());
        const uint32_t bits = c->queue_family < n ? families[c->queue_family].timestampValidBits : 0;
        stamp_mask_ = bits >= 64 ? ~0ull : (1ull << bits) - 1;
        stamp_scale_ = c->props.limits.timestampPeriod * 0.6144;  // ns por tick x ticks de 614,4 MHz por ns
        if (!bits) for (InFlight& f : inflight_) f.queries = VK_NULL_HANDLE;
        VkQueryPoolCreateInfo pq{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        pq.queryType = VK_QUERY_TYPE_TIMESTAMP;
        pq.queryCount = kProfMarks;
        if (!bits || vkCreateQueryPool(c->device, &pq, nullptr, &prof_pool_) != VK_SUCCESS) prof_pool_ = VK_NULL_HANDLE;
        NVN_LOG("timestamps de la GPU: %u bits, %.3f ns por tick (x%.2f a ticks de NVN)\n", bits,
                c->props.limits.timestampPeriod, stamp_scale_);
    }
    staging_ = &inflight_[0].staging;

    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.maxLod = VK_LOD_CLAMP_NONE;
    vkCreateSampler(c->device, &si, nullptr, &default_sampler_);

    // Textura de reserva (1x1 negra) para lo que el shader pide y no está enlazado.
    for (DefaultImages* group : {&sampled_defaults_, &storage_defaults_}) {
        auto init_default = [&](Texture& t, int target) {
            t.desc.format = 0x25;
            t.desc.width = t.desc.height = t.desc.levels = 1;
            t.desc.depth = target == kTargetCube ? 6 : 1;
            t.desc.target = target;
            t.desc.flags = group == &storage_defaults_ ? kTextureFlagImage : 0;
            t.layout = compute_layout(1, 1, 1, t.desc.depth, 1, 4, 1, 1, true, 4, false);
            t.serial = next_texture_serial();
        };
        init_default(group->one_d, kTarget1D);
        init_default(group->two_d, kTarget2D);
        init_default(group->three_d, kTarget3D);
        init_default(group->cube, kTargetCube);
    }
    const bool crumbs = NVN_FLAG("breadcrumbs") || NVN_FLAG("breadcrumbs_compute") || NVN_FLAG("breadcrumbs_light");
    if (crumbs || !NVN_FLAG("no_compute_fill")) crumb_init(crumbs);
    fill_quirk_ = crumb_map_ && !NVN_FLAG("no_compute_fill");
    crumbs_passes_ = !NVN_FLAG("breadcrumbs_compute") && !NVN_FLAG("breadcrumbs_light");
    crumbs_barriers_ = !NVN_FLAG("breadcrumbs_light");
    ready_ = true;
    return true;
}

void Executor::wait_slot(InFlight& f) {
    if (!f.pending) return;
    auto t0 = std::chrono::steady_clock::now();
    if (vkWaitForFences(vk()->device, 1, &f.fence, VK_TRUE, UINT64_MAX) == VK_ERROR_DEVICE_LOST) crumb_report();
    // Con migas: el último envío ya acabó y aun así una empezó y no acabó. El driver abortó ese trabajo sin perder el
    // dispositivo (la Mali-G615 de un Dimensity 8300: cada envío 700 ms, el vigilante de la GPU matando un trabajo colgado,
    // y la pantalla en negro). Mismo informe que con la GPU perdida, una vez.
    else if (crumbs_on_ && crumb_map_ && f.submit == submitted_.load(std::memory_order_relaxed) && crumb_map_[0] != crumb_map_[1])
        crumb_report("work aborted by the driver");
    const auto waited_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    prof_gpu_wait_us_ += waited_ns / 1000;
    g_drop_stats.wait_ns.fetch_add(static_cast<uint64_t>(waited_ns), std::memory_order_relaxed);
    std::lock_guard lock(fence_mutex_);
    note_completed(f.submit);  // antes de dejar de estar en vuelo: quien no lo encuentre ya lo ve terminado
    f.pending = false;
}

void Executor::wait_all() {
    for (int k = 1; k <= kInFlight; k++) wait_slot(inflight_[(cur_ + k) % kInFlight]);  // del más antiguo
}

void Executor::begin() {
    if (recording_) return;
    InFlight& f = inflight_[cur_];
    {
        LoadTimer timer(*this, kLdGpuWait);
        wait_slot(f);  // su staging y sus descriptores ya no los lee la GPU
    }
    const size_t used_chunks = f.staging.used_chunks();
    f.staging.reset();
    f.staging.trim(std::max<size_t>(used_chunks, 1), kStagingChunk);
    f.texel_readback.reset();
    f.texel_readback.trim(f.texel_readback.bytes() > kStagingChunk ? 0 : 1);
    // El tope de staging es por envío: los demás contextos guardaban su pico hasta reutilizarse (6 x 128 MB).
    // Los que la GPU ya terminó lo devuelven ahora (nadie en la CPU lee su staging tras un begin()).
    // Se miran también los demás envíos terminados, del más antiguo al más nuevo (acaban en orden): completed_
    // avanza y lo retirado se libera antes.
    bool running = false;
    for (int k = 1; k < kInFlight; k++) {
        InFlight& o = inflight_[(cur_ + k) % kInFlight];
        if (o.pending && !running) {
            if (vkGetFenceStatus(vk()->device, o.fence) == VK_SUCCESS)
                wait_slot(o);  // no bloquea: ya acabó
            else
                running = true;  // los posteriores tampoco
        }
        if (!o.pending && o.staging.bytes() > kStagingChunk) {
            o.staging.reset();
            o.staging.trim(1);
        }
    }
    {
        LoadTimer timer(*this, kLdReap);
        buffer_cache_.reap(completed_.load(std::memory_order_acquire), generations_);
        reap_retired();
        reap_mirrors();
    }
    while (f.pools.size() > kKeptDescriptorPools) {  // como el staging: no guardar el pico de un envío enorme
        vkDestroyDescriptorPool(vk()->device, f.pools.back(), nullptr);
        f.pools.pop_back();
    }
    copied_.clear();
    for (VkDescriptorPool p : f.pools) vkResetDescriptorPool(vk()->device, p, 0);
    f.pool = 0;
    cmd_ = f.cmd;
    submit_draws_ = draws_;
    bound_graphics_ = VK_NULL_HANDLE;  // un command buffer nuevo no tiene nada enlazado
    bound_set1_ = VK_NULL_HANDLE;
    bound_set1_layout_ = bound_set0_layout_ = VK_NULL_HANDLE;
    dynamic_valid_ = false;
    memset(bound_vb_buf_, 0, sizeof bound_vb_buf_);
    bound_index_ = VK_NULL_HANDLE;
    descriptor_epoch_++;
    graphics_c0_valid_ = 0;
    staging_ = &f.staging;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(cmd_, 0);
    vkBeginCommandBuffer(cmd_, &bi);
    recording_ = true;
    // Copias de lectura de lo que escribió el envío anterior, al empezar este (al final de cada command buffer
    // coincidían con pérdidas de la GPU). Se entregan cuando termine este envío.
    if (readback_pending_) {
        readback_pending_ = false;
        if (record_overlay_readbacks()) {
            VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            hb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 0, 1, &hb, 0, nullptr, 0, nullptr);
        }
    }
    f.stamps.clear();  // ya escritos (wait_slot) o de un envío que no se mandó
    if (f.queries) vkCmdResetQueryPool(cmd_, f.queries, 0, kStampsPerSlot);
    // Lo de este envío va después de todo lo anterior (subidas que pisan texturas que otro envío aún
    // lee, render targets que se muestrean…): antes lo garantizaba esperar a la GPU en cada envío.
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0,
                         nullptr, 0, nullptr);
    for (DefaultImages* group : {&sampled_defaults_, &storage_defaults_}) {
        for (Texture* d : {&group->one_d, &group->two_d, &group->three_d, &group->cube}) {
            if (d->image || !ensure_image(*d, cmd_)) continue;
            VkClearColorValue black{};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, d->layers};
            vkCmdClearColorImage(cmd_, d->image, VK_IMAGE_LAYOUT_GENERAL, &black, 1, &range);
        }
    }
    note_gpu_work();  // un command buffer nuevo: sin saber qué hubo antes (y lo de arriba)
}

void Executor::finish() { finish(false); }

void Executor::finish(bool wait) {
    if (!recording_) return;
    end_render_pass();
    flush_clears(nullptr);  // borrados diferidos: presentar (otro command buffer) o la CPU pueden leer la textura
    readback_pending_ = NVN_FLAG("use_return") || any_cpu_reads_;  // las copias de lectura de este envío, al principio del siguiente
    vkEndCommandBuffer(cmd_);
    InFlight& f = inflight_[cur_];
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_;
    VkContext* c = vk();
    {
        // Reiniciar un fence sí necesita que nadie lo use a la vez (esperarlo, no). Ya está señalado (begin()
        // esperó este contexto) y no está en vuelo, así que no entra nadie nuevo y quien quede sale enseguida.
        std::unique_lock lock(fence_mutex_);
        fence_cv_.wait(lock, [&] { return f.waiters == 0; });
        vkResetFences(c->device, 1, &f.fence);
    }
    VkResult r;
    {
        LoadTimer timer(*this, kLdSubmit);
        std::lock_guard qlock(vk_queue_mutex());
        r = vkQueueSubmit(c->queue, 1, &si, f.fence);
    }
    recording_ = false;
    if (r != VK_SUCCESS) {
        NVN_LOG("vkQueueSubmit: %s\n", vk_result_name(r));
        crumb_report();
    } else {
        {
            std::lock_guard lock(fence_mutex_);
            f.submit = submitted_.load(std::memory_order_relaxed) + 1;
            f.pending = true;
        }
        submitted_.store(f.submit, std::memory_order_release);
        prof_submits_++;
        if (wait) wait_slot(f);
    }
    cur_ = (cur_ + 1) % kInFlight;
    if (NVN_FLAG("verbose_log")) watch_kgsl();  // DEPURACIÓN: lee /sys de la GPU cada medio segundo
    // Sin sitio para overlays nuevos (grow() falló) también se vacía: overlay_used_ ya no subiría y los overlays
    // quedarían desactivados hasta reiniciar.
    // Solo al acabar el envío del juego (no al partirlo a mitad de frame): lo que la GPU escribió antes en este
    // frame vive en los overlays, y vaciarlos entre ese draw y el que lo lee le daba datos viejos (índices y
    // argumentos fuera de rango: la GPU se colgaba al cargar las Ruinas).
    // 128 MB (antes 64): con los huecos reutilizados la arena se estabiliza en lo vivo más lo de los envíos en vuelo
    // (~26 + ~32 MB en una cacería), demasiado cerca de 64.
    const bool overlays_full = !splitting_ && (overlay_used_ > (128ull << 20) || overlay_arena_.failed());
    const bool cache_reset = !generations_ && buffer_cache_.needs_reset(frame_);  // con generaciones no se espera
    if (cache_reset || overlays_full) {
        const auto t0 = std::chrono::steady_clock::now();
        wait_all();
        const auto t1 = std::chrono::steady_clock::now();
        if (!generations_) buffer_cache_.maybe_reset(frame_);
        if (overlays_full) {
            compact_overlays();
            prof_compactions_++;
            prof_compact_wait_us_ += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            prof_compact_copy_us_ +=
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t1).count();
        }
    }
}

// Arena de overlays llena (con la GPU parada). Antes se tiraban todos y se rehacían desde la memoria del juego, que la
// GPU nunca actualiza: lo que la GPU guarda de un frame al siguiente (la exposición automática, partículas…) volvía a
// un valor viejo y la pantalla bajaba de brillo de golpe al mover la cámara (al moverla aparecen rangos nuevos y la
// arena se llena antes). Ahora los usados hace poco se copian a la arena vacía; los demás se rehacen si vuelven.
void Executor::compact_overlays() {
    constexpr uint64_t kKeepSubmits = 8;       // usados en los últimos envíos del juego
    constexpr uint64_t kKeepBytes = 32ull << 20;  // y como mucho esto: lo que pase se rehace como antes
    // Estado pequeño que escribe la GPU y que solo se enlaza de vez en cuando: se conserva aunque no se haya usado en los
    // últimos envíos. MH Rise guarda la exposición automática y su historial de 8 medidas en 36 bytes que solo enlaza el
    // compute que mide (no en todos los frames; ~3 envíos por frame): rehecho desde la memoria del juego (ceros), la
    // exposición siguiente salía de 1/max(0, 1e-8). <datos>/no_save_state: como antes.
    constexpr uint64_t kKeepSmall = 64ull << 10, kKeepSmallBytes = 4ull << 20;
    const bool keep_small = !NVN_FLAG("no_save_state");
    struct Kept {
        uint64_t address, cpu_hash, checked;
        uint8_t writers;
        std::vector<uint8_t> bytes;
    };
    std::vector<Kept> kept;
    std::vector<std::pair<uint64_t, uint64_t>> recent;  // (último uso, dirección), los más recientes primero
    std::vector<std::pair<uint64_t, uint64_t>> small;   // los pequeños que escribió la GPU, igual
    for (const auto& [address, o] : overlays_) {
        if (!o.data) continue;
        if (keep_small && o.writers && o.size <= kKeepSmall)
            small.push_back({o.checked, address});
        else if (submit_index_ - o.checked <= kKeepSubmits)
            recent.push_back({o.checked, address});
    }
    std::sort(small.begin(), small.end(), std::greater<>());
    uint64_t small_bytes = 0;
    size_t small_kept = 0;
    for (const auto& [checked, address] : small) {
        const Overlay& o = overlays_[address];
        if (small_bytes + o.size > kKeepSmallBytes) {  // no caben: compiten con los demás, como antes
            if (submit_index_ - o.checked <= kKeepSubmits) recent.push_back({o.checked, address});
            continue;
        }
        small_bytes += o.size;
        small_kept++;
        kept.push_back({address, o.cpu_hash, o.checked, o.writers, std::vector<uint8_t>(o.data, o.data + o.size)});
    }
    std::sort(recent.begin(), recent.end(), std::greater<>());
    uint64_t bytes = 0;
    for (const auto& [checked, address] : recent) {
        const Overlay& o = overlays_[address];
        if (bytes + o.size > kKeepBytes) continue;
        bytes += o.size;
        kept.push_back({address, o.cpu_hash, o.checked, o.writers, std::vector<uint8_t>(o.data, o.data + o.size)});  // memoria coherente
    }
    // Lo escrito por la GPU que no está en ningún otro sitio y no se usó en los últimos envíos: rehecho, sale de la memoria
    // del juego y quien lo calculó una sola vez no lo vuelve a escribir (las mallas que un compute calcula al cargar y que
    // solo se dibujan cuando se ven). Hasta kKeepGpu cada uno y kKeepGpuBytes en total, los más recientes primero; lo
    // demás se pierde (se cuenta y, con watch_nan, se dice). <datos>/no_keep_overlays: como antes.
    constexpr uint64_t kKeepGpu = 1ull << 20, kKeepGpuBytes = 16ull << 20;
    uint64_t gpu_bytes = 0;
    size_t gpu_kept = 0, gpu_lost = 0;
    {
        std::unordered_set<uint64_t> kept_at;
        for (const Kept& k : kept) kept_at.insert(k.address);
        std::vector<std::pair<uint64_t, uint64_t>> gpu;  // (último uso, dirección) de lo de la GPU que no entró
        for (const auto& [address, o] : overlays_)
            if (o.data && o.writers && !kept_at.count(address)) gpu.push_back({o.checked, address});
        std::sort(gpu.begin(), gpu.end(), std::greater<>());
        erase_reason("compactación de la arena (no entra en lo que se conserva)");  // DEPURACIÓN
        for (const auto& [checked, address] : gpu) {
            const Overlay& o = overlays_[address];
            if (keep_gpu_overlays_ && o.size <= kKeepGpu && gpu_bytes + o.size <= kKeepGpuBytes) {
                gpu_bytes += o.size;
                gpu_kept++;
                kept.push_back({address, o.cpu_hash, o.checked, o.writers, std::vector<uint8_t>(o.data, o.data + o.size)});
                continue;
            }
            gpu_lost++;
            note_lost(o);
            if (vigilar_nan_) note_erase(address, o);  // DEPURACIÓN
        }
    }
    const size_t before = overlays_.size();
    clear_overlays();
    overlay_arena_.reset();
    overlay_used_ = 0;
    size_t restored = 0;
    for (const Kept& k : kept) {
        Staging::Alloc a = overlay_arena_.alloc(k.bytes.size() + 256, 256);
        if (!a.data) {  // sin sitio ni en la arena vacía: lo de la GPU de estos se pierde
            erase_reason("compactación de la arena (sin sitio al rehacerlo)");  // DEPURACIÓN
            Overlay lost;
            for (size_t j = restored; j < kept.size(); j++)
                if (kept[j].writers) lost.writers = kept[j].writers, lost.size = kept[j].bytes.size(), note_lost(lost);
            break;
        }
        overlay_used_ += k.bytes.size() + 256;
        const VkDeviceSize skew = k.address & 255;
        memcpy(a.data + skew, k.bytes.data(), k.bytes.size());
        Overlay& o = overlays_[k.address];
        o.buffer = a.buffer;
        o.offset = a.offset + skew;
        o.data = a.data + skew;
        o.size = k.bytes.size();
        o.cpu_hash = k.cpu_hash;
        o.checked = k.checked;
        o.writers = k.writers;  // sigue siendo estado de la GPU en la próxima compactación
        o.slot_offset = a.offset, o.slot_size = k.bytes.size() + 256, o.slot_data = a.data;
        filter_overlay(k.address, o.size, +1);
        restored++;
    }
    for (auto it = provenance_.begin(); it != provenance_.end();)  // DEPURACIÓN (watch_nan)
        it = overlays_.count(it->first) ? std::next(it) : provenance_.erase(it);
    {  // el estado de devolución de los que no se conservaron: hasta entregar lo que tengan pendiente
        std::lock_guard lock(wb_mutex_);
        for (auto& [address, w] : wb_state_)
            if (!overlays_.count(address)) w.orphan = wb_orphans_ = true;
    }
    static int logged = 0;
    if (logged++ < 10 || logged % 100 == 0)
        NVN_LOG("overlays: arena llena (%zu); se conservan %zu (%zu de estado de la GPU, %.1f KB; %.1f MB usados hace poco; "
                "%zu más escritos por la GPU, %.1f KB; %zu escritos por la GPU se pierden)\n",
                before, restored, small_kept, small_bytes / 1024.0, bytes / 1048576.0, gpu_kept, gpu_bytes / 1024.0, gpu_lost);
    if (log_state_ && !small_state_.empty()) log_small_state("tras compactar");
}

// DEPURACIÓN (registro): el estado pequeño que escriben los compute, como lo ve la GPU (su overlay, memoria coherente: lo
// último que acabó) y como lo ve la CPU del juego (su memoria, que la GPU no actualiza sin use_return).
void Executor::log_small_state(const char* why) {
    for (const auto& [address, s] : small_state_) {
        std::string gpu, cpu;
        char b[24];
        auto it = overlays_.upper_bound(address);
        const Overlay* o = nullptr;
        uint64_t start = 0;
        if (it != overlays_.begin() && std::prev(it)->first + std::prev(it)->second.size >= address + s.size)
            start = std::prev(it)->first, o = &std::prev(it)->second;
        for (uint64_t i = 0; i < s.size / 4; i++) {
            if (o && o->data) snprintf(b, sizeof b, " %.5g", reinterpret_cast<const float*>(o->data + (address - start))[i]), gpu += b;
            snprintf(b, sizeof b, " %.5g", reinterpret_cast<const float*>(address)[i]), cpu += b;
        }
        NVN_LOG("[estado] %s, frame %llu: %#llx+%llu GPU:%s | juego:%s | %llu escrituras, la última en el frame %llu (la anterior %llu)\n",
                why, (unsigned long long)frame_, (unsigned long long)address, (unsigned long long)s.size,
                o ? gpu.c_str() : " SIN OVERLAY (se rehará con la memoria del juego)", cpu.c_str(),
                (unsigned long long)s.writes, (unsigned long long)s.frame, (unsigned long long)s.prev_frame);
    }
}

// DEPURACIÓN: marca de tiempo antes de un render pass o un dispatch (fuera de un pase) en el frame que se perfila.
void Executor::prof_gpu_mark(const char* kind, const char* custom) {
    if (!prof_pool_) return;
    if (prof_frame_ == ~0ull) {  // ¿empezar? se mira al empezar el primer pase de cada frame, no en cada pase
        static uint64_t checked = ~0ull;
        if (checked == frame_) return;
        checked = frame_;
        const char* data = getenv("NIER_DATA");
        const std::string file = std::string(data ? data : ".") + "/profile_gpu";
        if (access(file.c_str(), F_OK) != 0) return;
        unlink(file.c_str());
        prof_frame_ = frame_;
        prof_marks_.clear();
        prof_closed_ = false;
        vkCmdResetQueryPool(cmd_, prof_pool_, 0, kProfMarks);
    }
    if (prof_closed_) return;
    if (!prof_marks_.empty()) prof_marks_.back().draws = draws_ - prof_mark_draws_;
    prof_mark_draws_ = draws_;
    const bool end = kind[0] == 'f' || frame_ != prof_frame_ || prof_marks_.size() + 1 >= kProfMarks;
    char label[160];
    if (end) {
        snprintf(label, sizeof label, "fin");
    } else if (custom) {
        snprintf(label, sizeof label, "%s", custom);
    } else if (kind[0] == 'c') {
        snprintf(label, sizeof label, "compute");
    } else {
        // Los adjuntos del pase empezado (cur_pass_rt_; con merged_passes pueden ser más que los del juego): el primer color,
        // cuántos más (+N), la profundidad, y lo que cerró el de antes si era el mismo framebuffer o cabía con él.
        const cmd::SetRenderTargets& prt = cur_pass_rt_;
        const Texture* c0 = prt.count > 0 ? prt.colors[0] : nullptr;
        const Texture* d = prt.depth;
        const Texture* any = c0 ? c0 : d;
        int more = 0;
        for (int k = 1; k < prt.count; k++) more += prt.colors[k] != nullptr;
        char extra[8] = "";
        if (more) snprintf(extra, sizeof extra, "+%d", more);
        snprintf(label, sizeof label, "pase %dx%d %s%s%s%s", any ? any->desc.width : 0, any ? any->desc.height : 0,
                 c0 ? (format_info(c0->desc.format) ? format_info(c0->desc.format)->name : "?") : "-", extra, d ? " +z" : "",
                 prof_pass_suffix_ ? prof_pass_suffix_ : "");
    }
    vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, prof_pool_, static_cast<uint32_t>(prof_marks_.size()));
    prof_marks_.push_back({label, 0});
    if (end) {
        prof_closed_ = true;
        prof_done_submit_ = submitted_.load(std::memory_order_relaxed) + 1;  // el envío que se está grabando
    }
}

void Executor::measure_changes(uint64_t address, uint64_t size) {
    ChangeStats& c = change_stats_[address];
    const size_t n = static_cast<size_t>((size + 65535) >> 16);
    std::vector<uint64_t> pages(n);
    for (size_t k = 0; k < n; k++) {
        const uint64_t at = uint64_t{k} << 16;
        pages[k] = range_hash(reinterpret_cast<const uint8_t*>(address + at), static_cast<size_t>(std::min<uint64_t>(65536, size - at)));
    }
    if (c.pages.size() == n) {
        uint64_t changed = 0;
        for (size_t k = 0; k < n; k++) changed += pages[k] != c.pages[k];
        c.copies++;
        if (changed) c.changed_copies++, c.changed_pages += changed;
        if (c.last_frame == frame_) c.same_frame++;
    }
    if (c.last_frame != frame_) c.frames++;
    c.last_frame = frame_;
    c.pages = std::move(pages);
    if (c.copies && c.copies % 200 == 0)
        NVN_LOG("cambios %#llx+%llu KB: %llu copias (%llu en el mismo frame que la anterior), %llu con cambios, %.1f "
                "páginas cambiadas de %zu de media en esas\n", static_cast<unsigned long long>(address),
                static_cast<unsigned long long>(size >> 10), static_cast<unsigned long long>(c.copies),
                static_cast<unsigned long long>(c.same_frame), static_cast<unsigned long long>(c.changed_copies),
                c.changed_copies ? double(c.changed_pages) / double(c.changed_copies) : 0.0, n);
}

// DEPURACIÓN: las migas (ver Crumb).
// El buffer también es el del relleno de los dispatch (fill_quirk_).
void Executor::crumb_init(bool crumbs) {
    VkContext* c = vk();
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = 64;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (vkCreateBuffer(c->device, &bi, nullptr, &crumb_buffer_) != VK_SUCCESS) return;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(c->device, crumb_buffer_, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* mapped = nullptr;
    if (vkAllocateMemory(c->device, &ai, nullptr, &crumb_memory_) != VK_SUCCESS ||
        vkBindBufferMemory(c->device, crumb_buffer_, crumb_memory_, 0) != VK_SUCCESS ||
        vkMapMemory(c->device, crumb_memory_, 0, bi.size, 0, &mapped) != VK_SUCCESS)
        return;
    crumb_map_ = static_cast<volatile uint32_t*>(mapped);
    crumb_map_[0] = crumb_map_[1] = 0;
    if (!crumbs) return;
    crumbs_.resize(kCrumbs);
    crumbs_on_ = true;
    NVN_LOG("breadcrumbs: on (all GPU work serialized)\n");
}

void Executor::crumb_begin(char kind, const char* label) {
    if (!crumbs_on_) return;
    note_gpu_work();  // rellenos y barreras propias (ver gpu_work_)
    const uint32_t id = ++crumb_id_;
    Crumb& c = crumbs_[id % kCrumbs];
    c = Crumb{};
    c.submit = submitted_.load(std::memory_order_relaxed) + 1;
    c.kind = kind;
    snprintf(c.label, sizeof c.label, "%s", label);
    if (!crumbs_barriers_) {  // las barreras del dispatch ordenan lo de antes y lo de después
        vkCmdFillBuffer(cmd_, crumb_buffer_, 0, 4, id);
        return;
    }
    // Lo anterior, acabado; la miga, escrita antes de que empiece lo siguiente.
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
    vkCmdFillBuffer(cmd_, crumb_buffer_, 0, 4, id);
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}

void Executor::crumb_end() {
    if (!crumbs_on_) return;
    note_gpu_work();
    if (!crumbs_barriers_) {
        vkCmdFillBuffer(cmd_, crumb_buffer_, 4, 4, crumb_id_);
        return;
    }
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
    vkCmdFillBuffer(cmd_, crumb_buffer_, 4, 4, crumb_id_);
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}

// Un shader de la miga abierta (los distintos, hasta 8).
void Executor::crumb_stage(const ProgramStage* s) {
    if (!crumbs_on_ || !crumb_id_ || !s) return;
    Crumb& c = crumbs_[crumb_id_ % kCrumbs];
    for (uint32_t i = 0; i < c.n_stages; i++)
        if (c.stages[i] == s) return;
    if (c.n_stages < 8) c.stages[c.n_stages++] = s;
}

void Executor::crumb_report(const char* why) {
    if (!crumbs_on_ || crumbs_reported_) return;
    crumbs_reported_ = true;
    const uint32_t started = crumb_map_[0], ended = crumb_map_[1];
    NVN_LOG("breadcrumbs: %s; last started %u, last finished %u (%u recorded)\n", why, started,
            ended, crumb_id_);
    const uint32_t first = ended > 4 ? ended - 4 : 1, last = std::min(crumb_id_, started + 3);
    for (uint32_t id = first; id <= last && crumb_id_ - id < kCrumbs; id++) {
        const Crumb& c = crumbs_[id % kCrumbs];
        std::string prints;
        for (uint32_t i = 0; i < c.n_stages; i++) {
            char b[40];
            snprintf(b, sizeof b, " %016llx", static_cast<unsigned long long>(c.stages[i]->print));
            prints += b;
        }
        // Con solo los dispatch, una que acabó y no tiene siguiente: lo colgado son los draws de después (van en ella).
        const uint32_t culprit = started != ended ? started : ended;
        NVN_LOG("breadcrumbs: %s%u submit %llu %s, %u draws, shaders%s\n", id == culprit ? "-> " : "   ", id,
                static_cast<unsigned long long>(c.submit), c.label, c.draws, prints.c_str());
        if (id == culprit)
            for (const Crumb::GpuDraw& g : c.gpu_draws) {
                const uint32_t n = g.count ? *reinterpret_cast<const uint32_t*>(g.count) : g.cpu_count ? g.cpu_count : 1;
                std::string args;
                char b[80];
                const uint32_t stride = g.stride ? g.stride : g.index_type >= 0 ? 20 : 16;
                for (uint32_t k = 0; g.args && k < std::min<uint32_t>(std::min(n, g.max_draws), 6); k++) {
                    const auto* e = reinterpret_cast<const uint32_t*>(g.args + uint64_t(k) * stride);
                    snprintf(b, sizeof b, " [%u %u %u %d %u]", e[0], e[1], e[2], static_cast<int>(e[3]), e[4]);
                    args += b;
                }
                NVN_LOG("breadcrumbs:     GPU args %#llx: %u of %u draws%s (indices %d)%s\n", static_cast<unsigned long long>(g.address), n,
                        g.max_draws, g.count ? " (count from the GPU)" : "", g.index_type, args.c_str());
            }
        if (id == culprit)
            for (uint32_t i = 0; i < c.n_stages; i++) {
                char name[48];
                snprintf(name, sizeof name, "migas_%016llx", static_cast<unsigned long long>(c.stages[i]->print));
                dump_stage_code(*c.stages[i], name);
            }
    }
}

// DEPURACIÓN: con el frame perfilado ya terminado en la GPU, el log: cada grupo (tipo de pase o compute), su tiempo y
// sus draws, de más a menos.
void Executor::prof_gpu_report() {
    if (!prof_closed_ || prof_frame_ == ~0ull || completed_.load(std::memory_order_acquire) < prof_done_submit_) return;
    const uint32_t n = static_cast<uint32_t>(prof_marks_.size());
    std::vector<uint64_t> t(n);
    const VkResult r = vkGetQueryPoolResults(vk()->device, prof_pool_, 0, n, n * sizeof(uint64_t), t.data(), sizeof(uint64_t),
                                             VK_QUERY_RESULT_64_BIT);
    prof_frame_ = ~0ull;
    if (r != VK_SUCCESS || n < 2) return;
    struct Group {
        double ms = 0;
        uint64_t draws = 0, count = 0;
    };
    std::map<std::string, Group> groups;
    double total = 0;
    for (uint32_t k = 0; k + 1 < n; k++) {
        const double ms = static_cast<double>((t[k + 1] - t[k]) & stamp_mask_) * vk()->props.limits.timestampPeriod / 1e6;
        Group& g = groups[prof_marks_[k].label];
        g.ms += ms, g.draws += prof_marks_[k].draws, g.count++;
        total += ms;
    }
    std::vector<std::pair<std::string, Group>> sorted(groups.begin(), groups.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
    NVN_LOG("perfil GPU del frame %llu: %.2f ms en %u marcas\n", static_cast<unsigned long long>(frame_), total, n);
    for (size_t k = 0; k < sorted.size() && k < 40; k++)  // 40: con el motivo de cada corte hay más grupos
        NVN_LOG("perfil GPU:  %6.2f ms  %4llu x  %5llu draws  %s\n", sorted[k].second.ms,
                static_cast<unsigned long long>(sorted[k].second.count), static_cast<unsigned long long>(sorted[k].second.draws),
                sorted[k].first.c_str());
}

// Margen que se reserva además del coste de LSFG (ns): <datos>/lsfg_reserva con milisegundos, leído como mucho una vez
// por segundo; sin fichero, 2 ms.
int64_t Executor::reserve_margin_ns() {
    const auto now = std::chrono::steady_clock::now();
    if (now >= reserve_next_) {
        reserve_next_ = now + std::chrono::seconds(1);
        const char* data = getenv("NIER_DATA");
        double ms = 2.0;
        if (FILE* f = fopen((std::string(data ? data : ".") + "/lsfg_reserva").c_str(), "r")) {
            if (fscanf(f, "%lf", &ms) != 1) ms = 2.0;
            fclose(f);
        }
        const int64_t v = static_cast<int64_t>(ms * 1e6);
        if (v != reserve_margin_) NVN_LOG("resolución dinámica: margen para LSFG %.1f ms\n", ms);
        reserve_margin_ = v;
    }
    return reserve_margin_;
}

// El envío de f terminó (con fence_mutex_): copia sus timestamps de la GPU a los informes del juego.
void Executor::write_stamps(InFlight& f) {
    const uint32_t n = static_cast<uint32_t>(f.stamps.size());
    uint64_t ticks[kStampsPerSlot];
    if (vkGetQueryPoolResults(vk()->device, f.queries, 0, n, sizeof ticks, ticks, sizeof ticks[0], VK_QUERY_RESULT_64_BIT) !=
        VK_SUCCESS) {
        f.stamps.clear();
        return;
    }
    for (uint32_t k = 0; k < n; k++) {
        const InFlight::Stamp& s = f.stamps[k];
        const uint64_t t = ticks[s.query] & stamp_mask_;
        const uint64_t report[2] = {0, static_cast<uint64_t>(static_cast<double>(t) * stamp_scale_) + s.extra};
        memcpy(reinterpret_cast<void*>(s.address), report, sizeof report);
        if (!s.end) {
            frame_start_ticks_ = t;
        } else if (frame_start_ticks_ && t > frame_start_ticks_) {
            const auto gpu_ns = static_cast<int64_t>((t - frame_start_ticks_) * vk()->props.limits.timestampPeriod);
            frame_gpu_ns_.store(gpu_ns, std::memory_order_relaxed);
            g_drop_stats.gpu_sum_ns.fetch_add(static_cast<uint64_t>(gpu_ns), std::memory_order_relaxed);
            g_drop_stats.gpu_frames.fetch_add(1, std::memory_order_relaxed);
            note_max(g_drop_stats.gpu_max_ns, static_cast<uint64_t>(gpu_ns));
            g_game_gpu_serial.fetch_add(1, std::memory_order_release);
            frame_start_ticks_ = 0;
        }
    }
    f.stamps.clear();
}

void Executor::retire_texture(Texture& t) {
    std::lock_guard lock(g_mutex);
    if (auto it = feedback_copies_.find(&t); it != feedback_copies_.end()) {
        retire_texture(it->second.texture);
        feedback_copies_.erase(it);
    }
    astc_waiting_.erase(&t);  // (an encode still away for it is dropped when it's back)
    t.astc_pending.clear();
    std::erase(frame_targets_, &t);
    std::erase(frame_uploads_, &t);
    std::erase(linear_written_, &t);
    std::erase_if(tex_dumps_, [&](const auto& item) { return item.first == &t; });
    std::erase_if(loading_matches_, [&](const auto& b) { return b.texture == &t; });
    // Fuera de run() no queda ninguno (finish los hace); por si acaso, ninguno con una imagen que ya no está.
    pending_clears_.erase(std::remove_if(pending_clears_.begin(), pending_clears_.end(),
                                         [&](const PendingClear& p) { return p.texture == &t; }),
                          pending_clears_.end());
    if (t.image || t.memory.memory) {
        Retired r;
        r.serial = t.serial;
        r.image = t.image;
        r.memory = t.memory;
        r.bytes = t.memory.memory ? t.gpu_bytes : 0;
        // El siguiente envío, no el último: el blit de presentar (window.cpp) va a la misma cola con su propio
        // fence, fuera de mi cuenta, y una señal de fence cubre todo lo enviado antes en la cola. (Presentar graba
        // el blit y lo envía sin soltar g_mutex: no puede quedar un blit de esta imagen por enviar.)
        r.submit = submitted_.load(std::memory_order_relaxed) + 1;
        retired_.push_back(r);
    }
    t.image = VK_NULL_HANDLE;
    t.memory = {};
    t.gpu_bytes = 0;
}

void Executor::retire_sampler(VkSampler s) {
    std::lock_guard lock(g_mutex);
    Retired r;
    r.sampler = s;
    r.submit = submitted_.load(std::memory_order_relaxed) + 1;  // como las texturas
    retired_.push_back(r);
}

// Destruye lo retirado cuyo último envío ya terminó. Las vistas de una textura van por su serial; los
// framebuffers, por los handles de las vistas que llevan en la clave (el pase de render, las vistas y el
// tamaño): un handle de vista destruido puede volver a salir en una vista nueva.
void Executor::reap_retired() {
    if (retired_.empty()) return;
    VkContext* c = vk();
    std::vector<uint64_t> serials;
    size_t keep = 0;
    const uint64_t completed = completed_.load(std::memory_order_acquire);
    for (const Retired& r : retired_) {
        if (r.submit > completed) {
            retired_[keep++] = r;
            continue;
        }
        if (r.image) vkDestroyImage(c->device, r.image, nullptr);
        if (r.memory.memory) {
            GpuAllocation memory = r.memory;
            gpu_free(memory);
            g_texture_gpu_bytes -= r.bytes;
            g_texture_gpu_images--;
        }
        if (r.sampler) vkDestroySampler(c->device, r.sampler, nullptr);
        if (r.serial) serials.push_back(r.serial);
    }
    retired_.resize(keep);
    if (serials.empty()) return;
    std::sort(serials.begin(), serials.end());
    std::unordered_set<VkImageView> dead;
    for (auto it = views_.begin(); it != views_.end();) {
        if (!std::binary_search(serials.begin(), serials.end(), it->first.value.texture)) {
            ++it;
            continue;
        }
        dead.insert(it->second);
        vkDestroyImageView(c->device, it->second, nullptr);
        it = views_.erase(it);
    }
    if (dead.empty()) return;
    for (auto it = framebuffers_.begin(); it != framebuffers_.end();) {
        const std::string& key = it->first;
        bool uses = false;
        for (size_t o = sizeof(VkRenderPass); !uses && o + sizeof(VkImageView) + 2 * sizeof(uint32_t) <= key.size();
             o += sizeof(VkImageView)) {
            VkImageView v;
            memcpy(&v, key.data() + o, sizeof v);
            uses = dead.count(v) != 0;
        }
        if (!uses) {
            ++it;
            continue;
        }
        vkDestroyFramebuffer(c->device, it->second, nullptr);
        it = framebuffers_.erase(it);
    }
}

// Sin g_mutex: con él, un hilo del juego que esperaba la GPU (aunque fuera un envío viejo) se quedaba detrás de la
// reproducción en curso, que lo tiene 10-30 ms. Se busca el fence con fence_mutex_ y se espera fuera de todo lock;
// mientras alguien lo espera, el ejecutor no lo reinicia (finish()). Si ya no está en vuelo, terminó.
void Executor::deliver_write_backs(uint64_t up_to) {
    const uint64_t done = std::min(completed_.load(std::memory_order_acquire), up_to);
    std::lock_guard lock(wb_mutex_);
    prof_deliver_calls_++;  // DEPURACIÓN
    prof_deliver_up_to_ = up_to;
    prof_deliver_min_ = pending_wb_.empty() ? 0 : pending_wb_.front().submit;
    prof_deliver_pending_ = pending_wb_.size();
    size_t keep = 0;
    for (const PendingWriteBack& w : pending_wb_) {
        if (w.submit > done) {
            pending_wb_[keep++] = w;
            continue;
        }
        if (auto b = busy_readbacks_.find(w.data); b != busy_readbacks_.end()) {  // se entregue o no, queda libre
            readback_busy_[b->second]--;
            busy_readbacks_.erase(b);
        }
        prof_wb_seen_[w.address] = {w.size, w.writers};
        auto st = wb_state_.find(w.address);
        if (st == wb_state_.end() || st->second.base.size() != w.size) continue;  // overlay rehecho: sin referencia
        if (st->second.generation != w.generation) continue;  // de antes de que la CPU lo reescribiera
        WriteBackState& ws = st->second;
        auto* game = reinterpret_cast<uint8_t*>(w.address);
        // A buffer the game fills with a GPU copy to read back: the copy is what it wants, whatever the CPU did to it
        // since (MH Rise rewrites its occlusion results on the CPU after reading them; mixing kept most of the GPU's
        // out, and endemic life flickered).
        if (w.whole) {
            nnsdk::watch::before_kernel_write(game, w.size);
            memcpy(game, w.data, w.size);
            memcpy(ws.base.data(), w.data, w.size);
            if (ws.watched) nnsdk::watch::collect(w.address, w.size, ws.keys, nullptr);
            prof_written_back_ += w.size;
            auto& pd = prof_delivered_[w.address];
            pd.first = w.size, pd.second += w.size;
            delivered_hash_[w.address] = range_hash(game, w.size);
            continue;
        }
        bool opened = false, cpu_rewrote = false;
        std::vector<std::pair<size_t, size_t>> cpu_runs;  // tramos de páginas que escribió la CPU desde base
        if (ws.watched) nnsdk::watch::collect(w.address, w.size, ws.keys, &cpu_runs);
        size_t run = 0;
        // Mezcla a tres por trozo: base (lo que ambos vieron), la GPU (la copia) y la CPU (la memoria ahora). Lo que
        // cambió la GPU va a la memoria solo si la CPU no tocó ese trozo (ni su página) desde base. La base pasa a ser
        // lo que queda en la memoria.
        for (size_t at = 0; at < w.size; at += kOwnChunk) {
            const size_t n = std::min(kOwnChunk, w.size - at);
            uint8_t* base = ws.base.data() + at;
            const bool gpu_changed = memcmp(w.data + at, base, n) != 0;
            while (run < cpu_runs.size() && cpu_runs[run].first + cpu_runs[run].second <= at) run++;
            const bool cpu_wrote = run < cpu_runs.size() && cpu_runs[run].first < at + n;
            const bool cpu_changed = memcmp(game + at, base, n) != 0;
            cpu_rewrote |= cpu_changed;
            if (gpu_changed && !cpu_changed && !cpu_wrote) {
                if (!opened) {
                    nnsdk::watch::before_kernel_write(game, w.size);  // por si está vigilada
                    opened = true;
                }
                memcpy(game + at, w.data + at, n);
                prof_written_back_ += n;
                auto& pd = prof_delivered_[w.address];
                pd.first = w.size, pd.second += n;
            }
            if (gpu_changed || cpu_changed) memcpy(base, game + at, n);
        }
        if (opened && ws.watched) nnsdk::watch::collect(w.address, w.size, ws.keys, nullptr);  // lo nuestro no es de la CPU
        // La huella de la memoria pasa a ser "nuestra" (el overlay sigue mandando) solo si la CPU no cambió nada: si
        // lo cambió, esa huella también es suya y registrarla escondía su escritura (un contador puesto otra vez a 0
        // que la GPU nunca veía: crecía sin fin y la GPU se perdía). Así el overlay se tira y la GPU lee la memoria,
        // con lo de la CPU y lo entregado, como cuando la CPU reescribe sin devolución.
        if (cpu_rewrote)
            delivered_hash_.erase(w.address);
        else if (opened)
            delivered_hash_[w.address] = range_hash(game, w.size);
    }
    pending_wb_.resize(keep);
    if (wb_orphans_) {  // los de overlays que la compactación no conservó, cuando ya no les queda nada que entregar
        std::unordered_set<uint64_t> still;
        for (const PendingWriteBack& w : pending_wb_) still.insert(w.address);
        wb_orphans_ = false;
        for (auto it = wb_state_.begin(); it != wb_state_.end();) {
            if (it->second.orphan && !still.count(it->first)) {
                delivered_hash_.erase(it->first);
                it = wb_state_.erase(it);
            } else {
                wb_orphans_ |= it->second.orphan;
                ++it;
            }
        }
    }
}

bool Executor::wait_submit_fence(uint64_t submit, uint64_t timeout_ns) {
    if (submit <= completed_.load(std::memory_order_acquire)) return true;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(std::min<uint64_t>(timeout_ns, 1ull << 62));
    // Con <datos>/no_free_wait, como antes (para comparar): se mira el fence con g_mutex cada 200 µs.
    if (NVN_FLAG("no_free_wait")) {
        for (;;) {
            {
                std::lock_guard lock(g_mutex);
                std::lock_guard fence_lock(fence_mutex_);
                if (submit <= completed_.load(std::memory_order_acquire)) return true;
                InFlight* slot = nullptr;
                for (InFlight& f : inflight_)
                    if (f.pending && f.submit == submit) slot = &f;
                if (!slot) return true;  // ya no está en vuelo: terminó
                if (vkGetFenceStatus(vk()->device, slot->fence) == VK_SUCCESS) {
                    note_completed(submit);
                    return true;
                }
            }
            if (std::chrono::steady_clock::now() >= deadline) return false;
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }
    InFlight* slot = nullptr;
    VkFence fence = VK_NULL_HANDLE;
    {
        std::lock_guard lock(fence_mutex_);
        if (submit <= completed_.load(std::memory_order_acquire)) return true;
        for (InFlight& f : inflight_)
            if (f.pending && f.submit == submit) slot = &f;
        if (!slot) return true;  // ya no está en vuelo: terminó
        slot->waiters++;
        fence = slot->fence;
    }
    const auto now = std::chrono::steady_clock::now();
    const uint64_t left = now >= deadline ? 0 : std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now).count();
    const VkResult r = left ? vkWaitForFences(vk()->device, 1, &fence, VK_TRUE, left) : vkGetFenceStatus(vk()->device, fence);
    {
        std::lock_guard lock(fence_mutex_);
        if (r == VK_SUCCESS) note_completed(submit);
        if (--slot->waiters == 0) fence_cv_.notify_all();
    }
    return r == VK_SUCCESS;
}

bool Executor::record_overlay_readbacks() {
    const uint64_t current = submitted_.load(std::memory_order_relaxed) + 1;  // el que se graba ahora
    bool barrier_done = false;
    std::lock_guard lock(wb_mutex_);
    for (auto& [address, o] : overlays_) {
        // Escrito desde la última copia (que se grabó al empezar el envío readback_submit: lo de ese envío va después).
        if (!o.written || o.written < o.readback_submit || o.size > kWriteBackMax) continue;
        if (!write_back_on_ && !o.cpu_reads) continue;  // (without use_return, only what the game copies to read back)
        if (o.readback && busy_readbacks_.count(o.readback_data)) continue;  // la anterior aún sin entregar
        if (!o.readback) {
            const int other = 1 - readback_cur_;
            if (readback_arenas_[readback_cur_].used() > (64ull << 20) && !readback_busy_[other]) {
                for (auto& [a2, o2] : overlays_)
                    if (o2.readback_arena == other) o2.readback = VK_NULL_HANDLE, o2.readback_data = nullptr, o2.readback_arena = -1;
                readback_arenas_[other].reset();
                readback_cur_ = other;
            }
            Staging::Alloc a = readback_arenas_[readback_cur_].alloc(o.size, 16);
            if (!a.data) continue;
            o.readback = a.buffer, o.readback_offset = a.offset, o.readback_data = a.data;
            o.readback_arena = static_cast<int8_t>(readback_cur_);
        }
        if (!barrier_done) {
            VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
            barrier_done = true;
        }
        const VkBufferCopy region{o.offset, o.readback_offset, o.size};
        note_gpu_work();
        vkCmdCopyBuffer(cmd_, o.buffer, o.readback, 1, &region);
        o.readback_submit = current;
        busy_readbacks_[o.readback_data] = o.readback_arena;
        readback_busy_[o.readback_arena]++;
        const auto st = wb_state_.find(address);
        pending_wb_.push_back({current, address, o.size, o.readback_data, st != wb_state_.end() ? st->second.generation : 0, o.writers,
                               o.cpu_reads && !write_back_on_});
    }
    return barrier_done;
}

bool Executor::run(const std::vector<const CommandList*>& lists) {
    std::lock_guard lock(g_mutex);
    if (!ready_ && !init()) return false;
    cpu_publication_ok_ = true;
    auto t0 = std::chrono::steady_clock::now();
    // Perfil de cargas: lo de este envío (lo de fuera de run, p. ej. presentar, también cae aquí).
    const uint64_t protect_ns0 = nnsdk::watch::protect_ns(), protects0 = nnsdk::watch::protects();
    const uint64_t staging_ns0 = g_staging_new_ns.load(std::memory_order_relaxed), staging_new0 = g_staging_new.load(std::memory_order_relaxed),
                   staging_bytes0 = g_staging_new_bytes.load(std::memory_order_relaxed),
                   staging_reused0 = g_staging_reused.load(std::memory_order_relaxed);
    const uint64_t draws0 = draws_, dispatches0 = dispatches_;
    // CPU de este hilo y cambios de contexto: si el envío dura mucho más que su CPU, estuvo esperando (cerrojos, el
    // kernel) o le quitaron el núcleo (involuntarios: otros hilos en los núcleos grandes, p. ej. los de carga del juego).
    timespec cpu0{};
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu0);
    rusage ru0{};
    getrusage(RUSAGE_THREAD, &ru0);
    submit_index_++;
    // Interruptores, una vez por envío del juego. no_generations cambia la caché en el siguiente punto seguro
    // (entre draws o al terminar un envío de Vulkan).
    split_ = !NVN_FLAG("single_submit");
    gpu_stamps_ = !NVN_FLAG("ts_cpu");
    skip_compaction_gs_ = !NVN_FLAG("no_gs_skip");
    log_state_ = NVN_FLAG("verbose_log");
    fast_depth_ = NVN_FLAG("fast_depth");
    prof_gpu_report();
    run_stamps_ = 0;
    // A la resolución dinámica se le dice que el frame cuesta lo suyo más LSFG más 2 ms de margen (con todo justo a 32 ms
    // los frames reales llegaban tarde y la guarda de LSFG saltaba). ts_no_lsfg: sin sumar nada.
    {
        const int64_t lsfg = NVN_FLAG("ts_no_lsfg") ? 0 : lsfg::reserved_ns();
        stamp_extra_ = lsfg ? static_cast<int64_t>((lsfg + reserve_margin_ns()) * 0.6144) : 0;
    }
    generations_ = !NVN_FLAG("no_generations");
    buffer_cache_.set_generations(generations_);
    prof_count_stage_keys_ = NVN_FLAG("verbose_log");
    per_stage_keys_ = NVN_FLAG("pipelines_per_stage");
    prof_timing_ = frame_ % kProfileEvery == 0 && NVN_FLAG("verbose_log");  // los tiempos por draw, solo depurando
    if (prof_timing_ && prof_timed_frame_ != frame_) prof_timed_frame_ = frame_, prof_timed_frames_++;
    buffer_cache_.prof = prof_timing_ ? &fine_.vtx : nullptr;
    read_flags();
    begin();
    apply_astc_results();
    if (!NVN_FLAG("no_mirrors_at_start")) preload_mirrors();
    run_lists_ = &lists;  // la mirada adelante de los render passes (scan_pass) recorre lo que queda
    run_serial_++;
    for (cur_list_ = 0; cur_list_ < lists.size(); cur_list_++)
        for (cur_index_ = 0; cur_index_ < lists[cur_list_]->commands.size(); cur_index_++)
            apply(lists[cur_list_]->commands[cur_index_]);
    run_lists_ = nullptr;
    if (!virtual_writes_.empty() && !publish_virtual_pool(nullptr, nullptr, Visibility::Submission)) cpu_publication_ok_ = false;
    finish();
    if (!linear_written_.empty()) return_linear_textures();
    const uint64_t run_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    prof_run_us_ += run_ns / 1000;
    frame_run_ns_ += run_ns;
    LoadProf& p = load_run_;
    p.protect_ns = nnsdk::watch::protect_ns() - protect_ns0;
    p.protects = nnsdk::watch::protects() - protects0;
    p.staging_ns = g_staging_new_ns.load(std::memory_order_relaxed) - staging_ns0;
    p.staging_new = g_staging_new.load(std::memory_order_relaxed) - staging_new0;
    p.staging_bytes = g_staging_new_bytes.load(std::memory_order_relaxed) - staging_bytes0;
    p.staging_reused = g_staging_reused.load(std::memory_order_relaxed) - staging_reused0;
    const bool slow = run_ns >= kSlowRunMs * 1e6;
    if (slow && log_state_) {
        timespec cpu1{};
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu1);
        rusage ru1{};
        getrusage(RUSAGE_THREAD, &ru1);
        const double cpu_ms = (cpu1.tv_sec - cpu0.tv_sec) * 1e3 + (cpu1.tv_nsec - cpu0.tv_nsec) / 1e6;
        int nice_now = getpriority(PRIO_PROCESS, static_cast<id_t>(gettid()));
        NVN_LOG("envío lento: %.0f ms (CPU del hilo %.0f ms, %ld cambios de contexto voluntarios y %ld involuntarios, "
                "nice %d; frame %llu, %zu listas, %llu draws, %llu dispatches): %s\n",
                run_ns / 1e6, cpu_ms, ru1.ru_nvcsw - ru0.ru_nvcsw, ru1.ru_nivcsw - ru0.ru_nivcsw, nice_now,
                static_cast<unsigned long long>(frame_), lists.size(), static_cast<unsigned long long>(draws_ - draws0),
                static_cast<unsigned long long>(dispatches_ - dispatches0), load_summary(p, run_ns).c_str());
    }
    if (run_ns >= 40'000'000) {  // para el log de ritmo (window.cpp), siempre
        std::lock_guard lock(g_drop_stats.slow_mutex);
        if (run_ns > g_drop_stats.slow_ns) {
            g_drop_stats.slow_ns = run_ns;
            g_drop_stats.slow_summary = load_summary(p, run_ns);
        }
    }
    load_period_.add(p);
    load_period_runs_++;
    load_period_slow_ += slow;
    load_period_run_ns_ += run_ns;
    load_period_worst_ns_ = std::max(load_period_worst_ns_, run_ns);
    load_run_ = LoadProf{};
    return cpu_publication_ok_;
}

void Executor::read_flags() {
    opt_.clear_diferido = !NVN_FLAG("no_deferred_clear");
    opt_.rt_igual = !NVN_FLAG("no_same_rt");
    opt_.barrera_perezosa = !NVN_FLAG("no_lazy_barrier");
    opt_.solo_escritos = !NVN_FLAG("no_written_only");
    opt_.store_ops = !NVN_FLAG("no_store_ops");
    opt_.area = !NVN_FLAG("no_area");
    opt_.ubwc = !NVN_FLAG("no_ubwc");  // solo para las imágenes que se creen a partir de ahora
    pass_end_is_barrier_ = NVN_FLAG("tight_barriers_pass");
    merge_passes_ = NVN_FLAG("merged_passes");
    async_pipelines_ = !NVN_FLAG("no_async_pipelines") && !serial_pipelines_;
    low_table_ = !NVN_FLAG("no_low_table");
    separate_tables_ = !NVN_FLAG("no_split_tables");
    scan_per_pass_ = NVN_FLAG("inspect_per_pass");
    // A full barrier with nothing recorded since the last one orders nothing; the driver still drains the GPU and
    // flushes its caches for it (on Turnip, a wait for idle each). About half of the ~250 a frame in MH Rise are those.
    // Off by default (Settings → Performance → Skip empty GPU barriers: <datos>/tight_barriers).
    skip_redundant_barriers_ = pass_end_is_barrier_ || NVN_FLAG("tight_barriers");
    recycle_overlays_ = !NVN_FLAG("no_overlay_gaps");
    patch_overlays_ = !NVN_FLAG("no_patch_overlays");
    keep_gpu_overlays_ = !NVN_FLAG("no_keep_overlays");
    merge_reads_ = !NVN_FLAG("no_merge_reads");
    split_overlays_ = !NVN_FLAG("no_split_overlays");
    vigilar_nan_ = NVN_FLAG("watch_nan");  // DEPURACIÓN
    trace_on_print_ = NVN_FLAG("trace_print");  // DEPURACIÓN
    read_after_ = NVN_FLAG("read_after_dispatch");  // DEPURACIÓN
    if (!vigilar_nan_ && !provenance_.empty()) provenance_.clear();
    watch_textures_ = !NVN_FLAG("no_texture_watch");
    bindless_ = !NVN_FLAG("no_bindless");
    sampler_compare_af_ = NVN_FLAG("af_compare");
    sampler_game_af_ = NVN_FLAG("game_af");
    sampler_no_shadows_ = NVN_FLAG("no_shadows");
    fixed_samplers_ = !NVN_FLAG("no_fixed_sampler");
    image_base_120_ = !NVN_FLAG("image_base_128");
    cache_sets_ = !NVN_FLAG("no_set_cache");
    templates_ = NVN_FLAG("templates");
    skip_set1_ = !NVN_FLAG("no_skip_set1");
    memo_layout_ = !NVN_FLAG("no_layout_memo");
    buffer_cache_.watch = NVN_FLAG("watch_vertices");
    buffer_cache_.front = !NVN_FLAG("no_vertex_front");
    watch_overlays_ = NVN_FLAG("watch_overlays");
    memo_pipeline_ = !NVN_FLAG("no_pipeline_memo");
    fast_bindless_ = !NVN_FLAG("no_fast_bindless");
    readback_copies_ = !NVN_FLAG("no_readback_copies");
    const bool write_back = NVN_FLAG("use_return");
    if (write_back && !write_back_on_) {
        // Se enciende en vivo: los overlays de antes no tienen referencia (make_overlay solo la hace con la devolución) y
        // no se devolverían hasta rehacerse. La memoria del juego es aún lo de la CPU (sin devolución la GPU no la toca):
        // esa es su referencia.
        std::lock_guard lock(wb_mutex_);
        size_t n = 0;
        for (const auto& [address, o] : overlays_)
            if (o.size <= kWriteBackMax && !wb_state_.count(address)) wb_reference(address, o.size), n++;
        NVN_LOG("devolución encendida: %zu overlays con referencia nueva\n", n);
    }
    write_back_on_ = write_back;
}

// --- Estado ---

void Executor::apply(const Command& c) {
    if (!recording_) begin(); // a CPU visibility fence may have submitted the preceding commands
    cur_command_ = &c;
    if (frame_ == trace_frame_) trace_counts_[c.index()]++;
    std::visit(
        [&](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, cmd::SetRenderTargets>) {
                // Los mismos targets con las mismas vistas: el pase abierto sigue (NieR vuelve a enlazar los suyos).
                const bool same = opt_.rt_igual && same_targets(v, st_.rt);
                if (same && in_render_pass_ && !targets_changed_) prof_rt_kept_++;
                // merged_passes: otros targets cuyos adjuntos están todos en el pase abierto: sigue, con las máscaras de los
                // que no son suyos a cero.
                bool within = !same && merge_passes_ && in_render_pass_ && !targets_changed_ && targets_within(v, cur_pass_rt_);
                if (within) {
                    const auto extent = target_extent(v);
                    within = extent.width == rp_extent_.width && extent.height == rp_extent_.height;
                }
                st_.rt = v;
                if (within) {
                    rp_keep_masks_ = keep_masks_for(v);
                    prof_rt_merged_++;
                } else if (!same) {
                    targets_changed_ = true;
                }
            } else if constexpr (std::is_same_v<T, cmd::ClearColor>) {
                clear_color(v);
            } else if constexpr (std::is_same_v<T, cmd::ClearDepthStencil>) {
                clear_depth_stencil(v);
            } else if constexpr (std::is_same_v<T, cmd::Viewport>) {
                if (v.index == 0) st_.viewport = {v.x, v.y, v.w, v.h, st_.viewport.minDepth, st_.viewport.maxDepth};
            } else if constexpr (std::is_same_v<T, cmd::Scissor>) {
                if (v.index == 0) {
                    st_.scissor = scissor_rect(v);
                    st_.scissor_set = true;
                }
            } else if constexpr (std::is_same_v<T, cmd::DepthRange>) {
                st_.viewport.minDepth = v.n;
                st_.viewport.maxDepth = v.f;
            } else if constexpr (std::is_same_v<T, cmd::BindProgram>) {
                // Bits de etapa: VERTEX 1, FRAGMENT 2, GEOMETRY 4, TESS_CONTROL 8, TESS_EVALUATION 16, COMPUTE 32.
                static const Stage kBits[] = {Stage::Vertex, Stage::Fragment, Stage::Geometry,
                                              Stage::TessControl, Stage::TessEval, Stage::Compute};
                for (int b = 0; b < 6; b++)
                    if (v.stages & (1 << b)) st_.stages[static_cast<int>(kBits[b])] = nullptr;
                if (v.stages & 32) st_.compute = nullptr;
                st_.pending_stages &= ~static_cast<uint32_t>(v.stages);
                // El compute va aparte: st_.stages solo lleva las etapas del pipeline gráfico.
                if (v.program)
                    for (ProgramStage& ps : v.program->stages)
                        if (ps.tried ? ps.module != VK_NULL_HANDLE : [&] {
                                LoadTimer timer(*this, kLdShader);  // solo la primera vez (traducir o de la caché)
                                if (translate_or_queue(ps) == 1) {
                                    st_.pending_stages |= static_cast<uint32_t>(v.stages) & 31u;
                                    return false;
                                }
                                return ps.module != VK_NULL_HANDLE;
                            }()) {
                            if (ps.shader.stage == Stage::Compute) {
                                st_.compute = &ps;
                                st_.stage_generation[static_cast<int>(Stage::Compute)] = v.program->generation;
                                continue;
                            }
                            st_.stages[static_cast<int>(ps.shader.stage)] = &ps;
                            st_.stage_generation[static_cast<int>(ps.shader.stage)] = v.program->generation;
                        }
            } else if constexpr (std::is_same_v<T, cmd::BindVertexBuffers>) {
                for (size_t k = 0; k < v.ranges.size() && v.first + k < kMaxVertexBuffers; k++)
                    st_.vertex_buffers[v.first + k] = v.ranges[k];
            } else if constexpr (std::is_same_v<T, cmd::BindUniformBuffers>) {
                StageBindings& b = st_.bindings[static_cast<int>(from_nvn_stage(v.stage))];
                for (size_t k = 0; k < v.ranges.size() && v.first + k < kMaxUniforms; k++)
                    b.uniforms[v.first + k] = v.ranges[k];
            } else if constexpr (std::is_same_v<T, cmd::BindStorageBuffer>) {
                if (v.index < kMaxStorage) {
                    const auto stage = static_cast<int>(from_nvn_stage(v.stage));
                    auto& bound = st_.bindings[stage].storage[v.index];
                    if (bound.address != v.range.address || bound.size != v.range.size) {
                        bound = v.range;
                        const VkDeviceSize at = kGraphicsStorageBase + stage * kGraphicsStorageStride + v.index * 16;
                        graphics_c0_valid_ = std::min(graphics_c0_valid_, at);
                    }
                }
            } else if constexpr (std::is_same_v<T, cmd::BindTexture>) {
                StageBindings& b = st_.bindings[static_cast<int>(from_nvn_stage(v.stage))];
                if (v.index >= kMaxTextures) {
                } else if (v.combined && separate_tables_) {
                    b.combined_textures[v.index] = v.handle;  // la tabla baja de c2 (StageBindings::combined_textures)
                } else {
                    b.textures[v.index] = v.handle, b.combined[v.index] = v.combined;
                }
            } else if constexpr (std::is_same_v<T, cmd::BindSampler>) {
                StageBindings& b = st_.bindings[static_cast<int>(from_nvn_stage(v.stage))];
                if (v.index < kMaxTextures) b.samplers[v.index] = v.handle;
            } else if constexpr (std::is_same_v<T, BlendState>) {
                st_.blend[v.target & 7] = v;
            } else if constexpr (std::is_same_v<T, ColorState>) {
                st_.color = v;
            } else if constexpr (std::is_same_v<T, ChannelMaskState>) {
                st_.masks = v;
            } else if constexpr (std::is_same_v<T, DepthStencilState>) {
                st_.depth_stencil = v;
            } else if constexpr (std::is_same_v<T, PolygonState>) {
                st_.polygon = v;
            } else if constexpr (std::is_same_v<T, MultisampleState>) {
                st_.multisample = v;
            } else if constexpr (std::is_same_v<T, cmd::BindVertexAttribs>) {
                st_.attribs.assign(v.attribs.begin(), v.attribs.end());
            } else if constexpr (std::is_same_v<T, cmd::BindVertexStreams>) {
                st_.streams.assign(v.streams.begin(), v.streams.end());
            } else if constexpr (std::is_same_v<T, cmd::SetPools>) {
                if (v.textures.assigned()) st_.texture_pool = v.textures;
                if (v.samplers.assigned()) st_.sampler_pool = v.samplers;
            } else if constexpr (std::is_same_v<T, cmd::PolygonOffset>) {
                st_.depth_bias[0] = v.units, st_.depth_bias[1] = v.clamp, st_.depth_bias[2] = v.factor;
            } else if constexpr (std::is_same_v<T, cmd::Stencil>) {
                uint32_t* target = v.what == cmd::Stencil::Ref ? st_.stencil_ref
                                   : v.what == cmd::Stencil::Mask ? st_.stencil_mask
                                                                   : st_.stencil_value_mask;
                if (v.faces & 1) target[0] = v.value;
                if (v.faces & 2) target[1] = v.value;
            } else if constexpr (std::is_same_v<T, cmd::Draw>) {
                const auto t0 = prof_now();
                draw(v);
                if (prof_timing_)
                    prof_draw_total_us_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
            } else if constexpr (std::is_same_v<T, cmd::Dispatch>) {
                const auto t0 = prof_now();
                dispatch(v);
                if (prof_timing_)
                    prof_dispatch_us_ += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
            } else if constexpr (std::is_same_v<T, cmd::BlendColor>) {
                memcpy(st_.blend_color, v.color, sizeof v.color);
            } else if constexpr (std::is_same_v<T, cmd::RenderEnable>) {
                if (frame_ == trace_frame_ && v.mode >= 0 && !watch_addr_) watch_addr_ = v.counters;
                if (frame_ == trace_frame_ && v.mode >= 0) {
                    const auto* c = reinterpret_cast<const uint64_t*>(v.counters);
                    NVN_LOG("F render enable condicional modo %d, contadores %#llx: %llx %llx\n", v.mode,
                            static_cast<unsigned long long>(v.counters), static_cast<unsigned long long>(c ? c[0] : 0),
                            static_cast<unsigned long long>(c ? c[1] : 0));
                } else if (frame_ == trace_frame_) {
                    NVN_LOG("F render enable %d\n", v.enable);
                }
                st_.render_enabled = v.enable;
            } else if constexpr (std::is_same_v<T, cmd::ClearBuffer>) {
                // Como CopyBuffer: memoria del juego, se rellena al ejecutar. Si el rango vive entero en un overlay,
                // el relleno va también a él (patch_overlay); si no, los overlays que lo tocan dejan de valer.
                auto* p = reinterpret_cast<uint32_t*>(v.address);
                auto keep = overlay_containing(v.address, v.size);  // antes de escribir: ¿sigue valiendo?
                const std::vector<uint64_t> partial =
                    keep == overlays_.end() ? overlays_before_cpu_write(v.address, v.size) : std::vector<uint64_t>{};
                for (uint64_t k = 0; k < v.size / 4; k++) p[k] = v.value;
                cpu_wrote_mirrors(v.address, v.size);
                if (keep != overlays_.end()) {
                    patch_overlay(keep, v.address, v.size);
                } else if (keep_gpu_overlays_ && patch_overlays_) {
                    cpu_wrote_partial(partial, v.address, v.size);
                } else {
                    erase_reason("ClearBuffer de la CPU que lo toca sin caber entero en él", v.address, v.size);  // DEPURACIÓN
                    for (auto it = overlays_.begin(); it != overlays_.end();) {
                        bool overlap = it->first < v.address + v.size && v.address < it->first + it->second.size;
                        it = overlap ? erase_overlay(it) : std::next(it);
                    }
                }
                copied_.clear();
            } else if constexpr (std::is_same_v<T, cmd::UpdateBuffer>) {
                // UpdateUniformBuffer (las constantes de cada draw en MH Rise): como ClearBuffer, se escribe la memoria
                // del juego al reproducir, en su sitio del orden. Lo que la GPU tuviera de ese rango deja de valer, y de
                // las copias de este envío solo las que tocan esas páginas (los draws de antes ya tienen la suya).
                auto keep = overlay_containing(v.address, v.size);
                const std::vector<uint64_t> partial =
                    keep == overlays_.end() ? overlays_before_cpu_write(v.address, v.size) : std::vector<uint64_t>{};
                memcpy(reinterpret_cast<void*>(v.address), v.data, v.size);
                cpu_wrote_mirrors(v.address, v.size);
                if (keep != overlays_.end()) {
                    patch_overlay(keep, v.address, v.size);
                } else if (keep_gpu_overlays_ && patch_overlays_) {
                    cpu_wrote_partial(partial, v.address, v.size);
                } else {
                    erase_reason("UpdateBuffer de la CPU que lo toca sin caber entero en él", v.address, v.size);  // DEPURACIÓN
                    drop_overlays(v.address, v.size);
                }
                copied_.invalidate(v.address, v.size);
            } else if constexpr (std::is_same_v<T, cmd::BindImage>) {
                StageBindings& b = st_.bindings[static_cast<int>(from_nvn_stage(v.stage))];
                if (v.index >= 0 && v.index < kMaxImages) b.images[v.index] = v.handle;
            } else if constexpr (std::is_same_v<T, cmd::ClearTexture>) {
                clear_texture(v);
            } else if constexpr (std::is_same_v<T, cmd::PrimitiveRestart>) {
                st_.primitive_restart = v.enable;
                st_.restart_index = v.index;
            } else if constexpr (std::is_same_v<T, cmd::PatchSize>) {
                st_.patch_size = v.size;
            } else if constexpr (std::is_same_v<T, cmd::FenceSync>) {
                if (v.flags && !virtual_writes_.empty()) {
                    if (!publish_virtual_pool(nullptr, nullptr, Visibility::CpuFence)) cpu_publication_ok_ = false;
                    targets_changed_ = true;
                }
            } else if constexpr (std::is_same_v<T, cmd::ReportValue>) {
                // Ejecutamos en orden y de forma síncrona: el trabajo anterior ya está hecho.
                if (frame_ == trace_frame_ && v.address)
                    NVN_LOG("F ReportValue %u -> %#llx (había %08x)\n", v.value, static_cast<unsigned long long>(v.address),
                            *reinterpret_cast<const uint32_t*>(v.address));
                if (v.address) *reinterpret_cast<volatile uint32_t*>(v.address) = v.value;
            } else if constexpr (std::is_same_v<T, cmd::CopyBuffer>) {
                if (frame_ == trace_frame_)
                    NVN_LOG("F CopyBuffer %#llx -> %#llx (%llu bytes)\n", (unsigned long long)v.src, (unsigned long long)v.dst,
                            (unsigned long long)v.size);
                // Si el origen lo escribió la GPU (overlay), la copia se hace en la GPU: p. ej. NieR copia el número
                // de balas que cuenta un compute al instanceCount de un draw indirecto.
                if (!v.size || v.src == v.dst) return;
                std::vector<uint8_t> snapshot;
                const auto source = copy_from_overlay(v.src, v.dst, v.size, snapshot);
                if (source == BufferCopySource::Error) {
                    NVN_LOG("CopyBuffer failed: %#llx -> %#llx (%llu bytes)\n", (unsigned long long)v.src,
                            (unsigned long long)v.dst, (unsigned long long)v.size);
                    cpu_publication_ok_ = false;
                    return;
                }
                if (source == BufferCopySource::Copied) {
                    copied_.clear();
                    return;
                }
                // Los buffers son memoria del juego: la copia de la GPU es una copia de memoria aquí (y, si cae entera
                // dentro de un overlay, también en él: patch_overlay).
                if (vigilar_nan_ && source == BufferCopySource::Guest) {  // DEPURACIÓN
                    char b[96];
                    snprintf(b, sizeof b, "CopyBuffer de la CPU a %#llx (%llu bytes)", static_cast<unsigned long long>(v.dst),
                             static_cast<unsigned long long>(v.size));
                    writer_ = b;
                    note_partial_read("CopyBuffer", v.src, v.size);
                }
                auto keep = overlay_containing(v.dst, v.size);  // antes de escribir: ¿sigue valiendo?
                const std::vector<uint64_t> partial =
                    keep == overlays_.end() ? overlays_before_cpu_write(v.dst, v.size) : std::vector<uint64_t>{};
                const void* bytes = source == BufferCopySource::Snapshot ? snapshot.data() : reinterpret_cast<const void*>(v.src);
                memmove(reinterpret_cast<void*>(v.dst), bytes, v.size);
                cpu_wrote_mirrors(v.dst, v.size);
                if (keep != overlays_.end()) {
                    patch_overlay(keep, v.dst, v.size);
                } else if (keep_gpu_overlays_ && patch_overlays_) {
                    cpu_wrote_partial(partial, v.dst, v.size);
                } else {
                    erase_reason("CopyBuffer de la CPU cuyo destino lo toca sin caber entero en él", v.dst, v.size);  // DEPURACIÓN
                    for (auto it = overlays_.begin(); it != overlays_.end();) {  // como ClearBuffer
                        bool overlap = it->first < v.dst + v.size && v.dst < it->first + it->second.size;
                        it = overlap ? erase_overlay(it) : std::next(it);
                    }
                }
                copied_.clear();
            } else if constexpr (std::is_same_v<T, cmd::CopyTexture>) {
                copy_texture(v);
            } else if constexpr (std::is_same_v<T, cmd::CopyBufferToTexture>) {
                // En la Switch la GPU copia el buffer (lineal) a la memoria de la textura. Aquí, la CPU, al ejecutar:
                // así la memoria de la textura queda como en la consola y la caja se sube al usarla. MH Rise carga así
                // los mips grandes de sus texturas por streaming; antes no se copiaba nada y se veían a cero (ASTC
                // inválido: magenta en la ropa).
                if (v.dst && v.src && (v.dst->desc.target == kTargetCube || v.dst->desc.target == kTargetCubeArray)) {
                    VkDeviceSize cube_off = 0;  // DEPURACIÓN: los cubos que se llenan así (note_cube_write)
                    note_cube_write(*v.dst, v.dst_view,
                                    find_overlay(v.src, 16, &cube_off) ? "copia de un buffer de la GPU" : "copia de un buffer",
                                    nullptr, 0, frame_);
                }
                {  // DEPURACIÓN: las primeras copias
                    static int logged = 0;
                    if (logged++ < 40 && v.dst)
                        NVN_LOG("CopyBufferToTexture %#llx -> textura %llu %dx%d formato 0x%x flags 0x%x nivel %d región %d,%d,%d %dx%dx%d\n",
                                static_cast<unsigned long long>(v.src), static_cast<unsigned long long>(v.dst->serial),
                                v.dst->desc.width, v.dst->desc.height, v.dst->desc.format, v.dst->desc.flags,
                                (v.dst_view.flags & kViewLevels) ? v.dst_view.base_level : 0, v.region.x, v.region.y,
                                v.region.z, v.region.width, v.region.height, v.region.depth);
                }
                if (v.dst && v.src) {
                    VkDeviceSize off = 0;
                    if (find_overlay(v.src, 16, &off)) {
                        static int logged = 0;
                        if (logged++ < 4) NVN_LOG("CopyBufferToTexture desde un buffer que escribió la GPU: se copia la memoria\n");
                    }
                    write_texels_to(*v.dst, &v.dst_view, &v.region, reinterpret_cast<const void*>(v.src), 0, 0);
                }
            } else if constexpr (std::is_same_v<T, cmd::Barrier>) {
                nvn_barrier(v.bits);
            } else if constexpr (std::is_same_v<T, cmd::ReportCounter>) {
                // Informe de 16 bytes: {valor, timestamp}. Ejecutamos al momento: el trabajo previo ya está.
                // SAMPLES_PASSED (1): NieR lo pide por objeto en el pre-pase de profundidad y solo dibuja en color
                // lo que vio pasar muestras. Sin consultas de oclusión de verdad, todo cuenta como visible (con un
                // contador creciente, por si compara dos informes).
                uint64_t report[2] = {0, 0};
                if (v.type == kCounterSamplesPassed) report[0] = samples_passed_ += uint64_t{1} << 20;
                if (frame_ == trace_frame_) {
                    const auto* old = reinterpret_cast<const uint32_t*>(v.address);
                    NVN_LOG("F ReportCounter había %08x %08x %08x %08x | %08x %08x %08x %08x\n", old[0], old[1], old[2], old[3],
                            old[4], old[5], old[6], old[7]);
                }
                // TIMESTAMP (0): NieR escribe uno al empezar y otro al acabar cada frame, espera a su sync y con la resta
                // (en ticks de 614,4 MHz, x625/384 a ns en main+0x9bbb08) decide la resolución dinámica: más de 32 ms
                // la baja, menos de 30 la sube. Con el reloj de la CPU al reproducir veía 32 veces menos y nunca bajaba.
                // Ahora lo escribe la GPU, y al del final se le suma lo que cuesta el frame generado de LSFG, para que la
                // resolución deje sitio a los dos (ver write_stamps).
                InFlight& slot = inflight_[cur_];
                if (v.type == kCounterTimestamp && gpu_stamps_ && slot.queries && recording_ &&
                    slot.stamps.size() < kStampsPerSlot) {
                    const uint32_t q = static_cast<uint32_t>(slot.stamps.size());
                    vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, slot.queries, q);
                    const bool end = last_stamp_addr_ && v.address == last_stamp_addr_ + 16;
                    if (!end) last_stamp_addr_ = v.address;
                    if (end && frame_ == prof_frame_ && !prof_closed_ && !in_render_pass_) prof_gpu_mark("fin");
                    run_stamps_++;
                    slot.stamps.push_back({q, v.address, end ? stamp_extra_ : 0, end});
                } else {
                    asm volatile("mrs %0, cntvct_el0" : "=r"(report[1]));
                    memcpy(reinterpret_cast<void*>(v.address), report, sizeof report);
                }
                if (frame_ == trace_frame_)
                    NVN_LOG("F ReportCounter tipo %d -> %#llx\n", v.type, static_cast<unsigned long long>(v.address));
            }
        },
        c);
}

// --- Imágenes ---

bool Executor::ensure_image(Texture& t, VkCommandBuffer cmd) {
    if (t.image) return true;
    if (t.vk_failed || frame_ < t.retry_frame) return false;
    LoadTimer timer(*this, kLdImage);
    VkContext* c = vk();
    const FormatInfo* f = format_info(t.desc.format);
    if (!f || f->vk == VK_FORMAT_UNDEFINED) {
        t.vk_failed = true;
        return false;
    }
    VkFormat format = f->vk;
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(c->gpu, format, &fp);
    // BCn en una GPU que no lo lee (Mali, PowerVR…): la imagen, sin comprimir, y cada nivel se descomprime al subirlo
    // (bcn.h). <datos>/bc_on_cpu hace lo mismo en cualquier GPU (para probar esa ruta en un Adreno).
    uint8_t bcn_kind = bcn::kNone;
    if (bcn::kind_of(format) != bcn::kNone &&
        (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) || NVN_FLAG("bc_on_cpu"))) {
        bcn_kind = bcn::kind_of(format);
        static std::set<VkFormat> told;
        if (told.insert(format).second)
            NVN_LOG("formato %s: la GPU no lo lee, se descomprime en la CPU al subirlo\n", f->name);
        format = bcn::decoded_format(format);
        // Stored as ASTC 4x4 instead (astc.h): the RGBA8 kinds only (BC4/BC5 are one or two channels and BC6H is HDR).
        // The blocks line up with the BC ones, 4x4 each.
        const bool rgba_kind = bcn_kind == bcn::kBC1 || bcn_kind == bcn::kBC2 || bcn_kind == bcn::kBC3 || bcn_kind == bcn::kBC7;
        if (rgba_kind && astc::wanted())
            format = format == VK_FORMAT_R8G8B8A8_SRGB ? VK_FORMAT_ASTC_4x4_SRGB_BLOCK : VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
        vkGetPhysicalDeviceFormatProperties(c->gpu, format, &fp);
    }
    // 24-bit depth on a GPU without it (AMD's, and so Samsung's Xclipse: no D24S8 at all): 32-bit float depth instead,
    // which holds every value the game writes. Without this, every pass with the game's DEPTH24_STENCIL8 depth (the depth
    // pre-pass, the shadows, the main view) failed to start and its draws were skipped: a third of the frame, and the
    // scenery came out as flat, see-through shapes on an Xclipse 530.
    if (is_depth_format(format) && !(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
        const VkFormat wider = format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D16_UNORM_S8_UINT
                                   ? VK_FORMAT_D32_SFLOAT_S8_UINT
                               : format == VK_FORMAT_X8_D24_UNORM_PACK32 || format == VK_FORMAT_D16_UNORM ? VK_FORMAT_D32_SFLOAT
                                                                                                         : format;
        VkFormatProperties wp;
        vkGetPhysicalDeviceFormatProperties(c->gpu, wider, &wp);
        if (wider != format && (wp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
            static std::set<VkFormat> told;
            if (told.insert(format).second) NVN_LOG("depth format %s: the GPU doesn't have it, 32-bit float depth instead\n", f->name);
            format = wider;
            fp = wp;
        }
    }
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) &&
        !(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
        NVN_LOG("formato %s sin soporte de imagen en la GPU\n", f->name);
        t.vk_failed = true;
        return false;
    }
    bool depth = is_depth_format(format);
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    int target = t.desc.target;
    bool cube = target == kTargetCube || target == kTargetCubeArray;
    ii.imageType = target == kTarget3D ? VK_IMAGE_TYPE_3D
                   : target == kTarget1D || target == kTarget1DArray ? VK_IMAGE_TYPE_1D
                                                                       : VK_IMAGE_TYPE_2D;
    ii.format = format;
    ii.extent = {static_cast<uint32_t>(std::max(t.desc.width, 1)), static_cast<uint32_t>(std::max(t.desc.height, 1)),
                 target == kTarget3D ? static_cast<uint32_t>(std::max(t.desc.depth, 1)) : 1};
    t.layers = target == kTarget1DArray || target == kTarget2DArray || target == kTarget2DMultisampleArray || cube
                   ? std::max(t.desc.depth, cube ? 6 : 1)
                   : 1;
    ii.mipLevels = std::max(t.desc.levels, 1);
    ii.arrayLayers = t.layers;
    ii.samples = static_cast<VkSampleCountFlagBits>(std::max(t.desc.samples, 1));
    ii.tiling = VK_IMAGE_TILING_OPTIMAL;
    ii.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (depth && (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
        ii.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    if (!depth && (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT))
        ii.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    // Imágenes (flag IMAGE): STORAGE si el formato lo admite o, si es sRGB, su UNORM (EXTENDED_USAGE: el uso vale para
    // las vistas con otro formato; la imagen ya es MUTABLE).
    t.storage_image = false;
    bool extended_usage = false;
    if ((t.desc.flags & kTextureFlagImage) && !depth) {
        if (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)
            t.storage_image = true;
        else if (storage_format(format) != format && storage_capable(storage_format(format)))
            t.storage_image = extended_usage = true;
        if (t.storage_image) ii.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }
    if (cube) ii.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    // MUTABLE_FORMAT (vistas con otro formato) probablemente le quita al Adreno la compresión UBWC. En Vulkan cada
    // formato de profundidad/stencil es su propia clase de compatibilidad: sin MUTABLE no se pierde ninguna vista
    // válida. Las de color sí se reinterpretan (NVN lo permite y no sabemos con qué formatos), y las comprimidas no
    // llevan UBWC: esas se quedan como estaban.
    t.mutable_format = !(depth && opt_.ubwc);
    if (t.mutable_format) ii.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    if (extended_usage) ii.flags |= VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
    ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (VkResult r = vkCreateImage(c->device, &ii, nullptr, &t.image); r != VK_SUCCESS) {
        NVN_LOG("vkCreateImage %dx%d %s: %s\n", t.desc.width, t.desc.height, f->name, vk_result_name(r));
        t.vk_failed = true;
        t.storage_image = false;
        return false;
    }
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(c->device, t.image, &req);
    if (!gpu_alloc(req, find_memory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), t.memory)) {
        NVN_LOG("sin memoria para una textura %dx%d (%u creadas; vivas %llu, %llu MB)\n", t.desc.width, t.desc.height,
                allocations_, (unsigned long long)g_texture_gpu_images.load(), (unsigned long long)(g_texture_gpu_bytes.load() >> 20));
        vkDestroyImage(c->device, t.image, nullptr);
        t.image = VK_NULL_HANDLE;
        t.retry_frame = frame_ + 30;  // sin memoria ahora: se reintenta en un segundo, no se da por perdida
        return false;
    }
    t.gpu_bytes = req.size;
    g_texture_gpu_bytes += req.size;
    g_texture_gpu_images++;
    if (++allocations_ % 500 == 0)
        NVN_LOG("%u imágenes creadas; vivas %llu, %llu MB\n", allocations_, (unsigned long long)g_texture_gpu_images.load(),
                (unsigned long long)(g_texture_gpu_bytes.load() >> 20));
    vkBindImageMemory(c->device, t.image, t.memory.memory, t.memory.offset);
    t.vk_format = format;
    t.bcn = bcn_kind;
    t.astc = format == VK_FORMAT_ASTC_4x4_SRGB_BLOCK || format == VK_FORMAT_ASTC_4x4_UNORM_BLOCK ? bcn_kind != bcn::kNone : false;
    t.aspect = depth ? (VK_IMAGE_ASPECT_DEPTH_BIT | (has_stencil(format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0))
                     : VK_IMAGE_ASPECT_COLOR_BIT;
    if (format == VK_FORMAT_S8_UINT) t.aspect = VK_IMAGE_ASPECT_STENCIL_BIT;

    if (cmd == cmd_ && in_render_pass_) {  // una barrera de imagen dentro de un render pass no es válida
        split_why_ = "imagen nueva (barrera de imagen)";
        end_render_pass();
        targets_changed_ = true;
    }
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = t.image;
    b.subresourceRange = {t.aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    if (cmd == cmd_) note_gpu_work();
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
    return true;
}

// Sube los texels que el juego dejó en su memoria (disposición Tegra) a la imagen. Fuera de render pass.
// Un nivel de una textura descomprimida en la CPU (Texture::bcn): `blocks`, sus bloques ya en orden lineal, salen como
// texels en trozos de staging (cada uno de las filas que quepan en kStagingChunk), descomprimidos por filas de bloques
// entre los hilos de row_pool, y una copia por trozo. false: el staging se acabó (se reintenta como cualquier subida).
bool Executor::astc_level(Texture& t, const LevelLayout& lv, const uint8_t* blocks, uint32_t level, uint32_t layer,
                          std::vector<std::pair<VkBuffer, VkBufferImageCopy>>& regions) {
    const bcn::Kind kind = static_cast<bcn::Kind>(t.bcn);
    const bool srgb = t.vk_format == VK_FORMAT_ASTC_4x4_SRGB_BLOCK;
    const size_t block_row = size_t{lv.width_blocks} * bcn::block_bytes(kind);
    const size_t slice_bc = block_row * lv.height_blocks, slice_astc = size_t{lv.width_blocks} * lv.height_blocks * 16;
    static thread_local std::vector<uint8_t> texels;
    // (whatever this upload does with the level, an encode still away for it is old now)
    const size_t slot = size_t{layer} * upload_levels(t) + level;
    if (slot < t.astc_pending.size()) t.astc_pending[slot] = 0;
    for (uint32_t z = 0; z < std::max(lv.depth, 1u); z++) {
        const uint8_t* slice = blocks + size_t{z} * slice_bc;
        Staging::Alloc a = staging_->alloc_contiguous(slice_astc, 16);
        if (!a.data) return false;
        // A level the game hasn't streamed in yet is all zeros (it creates its textures with every mip and fills the big
        // ones later). That decodes to one colour: ASTC has a block for that (void extent), so nothing to encode or cache.
        if (std::all_of(slice, slice + slice_bc, [](uint8_t b) { return b == 0; })) {
            uint8_t texel[16 * 4];
            bcn::decode(kind, slice, 1, 1, texel, 16, 4, 4);  // (a zero block: the colour every texel gets)
            uint8_t block[16];
            const uint64_t mode = 0xFFFFFFFFFFFFFDFCull;  // void extent, LDR, no extent coordinates
            memcpy(block, &mode, 8);
            for (int c = 0; c < 4; c++) {
                const uint16_t v = static_cast<uint16_t>(texel[c] * 257);
                memcpy(block + 8 + 2 * c, &v, 2);
            }
            for (size_t b = 0; b < slice_astc; b += 16) memcpy(a.data + b, block, 16);
            prof_astc_empty_++;
        } else if (kind == bcn::kBC1) {
            // Block for block, no encoder (astc::from_bc1): about as fast as a copy, so nothing goes to the cache
            LoadTimer timer(*this, kLdDeswizzle, slice_astc);
            astc::from_bc1(slice, a.data, size_t{lv.width_blocks} * lv.height_blocks);
            prof_astc_direct_++;
        } else {
        // The key: the BC blocks, their kind, size and colour space, and the encoder's version (1 here: change it if the
        // encoder settings change, or old files would be used).
        uint64_t key = full_hash(slice, slice_bc);
        key ^= (uint64_t{lv.width} << 40) ^ (uint64_t{lv.height} << 16) ^ (uint64_t{kind} << 8) ^ (srgb ? 0x80 : 0) ^ 1;
        key *= 0x9E3779B97F4A7C15ull;
        bool cached;
        {
            LoadTimer timer(*this, kLdDeswizzle, slice_astc);
            cached = astc::load(key, slice_astc, a.data);
        }
        if (!cached && lv.depth <= 1 && size_t{lv.width} * lv.height >= kAstcDeferTexels && !NVN_FLAG("sync_astc")) {
            AstcJob job;
            job.texture = &t;
            job.serial = t.serial;
            job.image = t.image;
            job.key = key | 1;
            job.cache_key = key;
            job.level = level;
            job.layer = layer;
            job.width = lv.width;
            job.height = lv.height;
            job.wb = lv.width_blocks;
            job.hb = lv.height_blocks;
            job.kind = static_cast<uint8_t>(kind);
            job.srgb = srgb;
            job.blocks.assign(slice, slice + slice_bc);
            if (t.astc_pending.size() <= slot) t.astc_pending.resize(slot + 1, 0);
            t.astc_pending[slot] = job.key;
            defer_astc(std::move(job));
            prof_astc_deferred_++;
            continue;  // (nothing for this level now: upload() keeps it out of what's sampled)
        }
        if (!cached) {
            LoadTimer timer(*this, kLdDeswizzle, size_t{lv.width} * lv.height * 4);
            // Whole blocks (the edge ones too: the encoder wants them filled), then the encode.
            const uint32_t w = lv.width_blocks * 4, h = lv.height_blocks * 4;
            texels.resize(size_t{w} * h * 4);
            bcn::decode(kind, slice, lv.width_blocks, lv.height_blocks, texels.data(), size_t{w} * 4, w, h);
            if (!astc::encode(texels.data(), w, h, srgb, a.data)) return false;
            astc::store(key, a.data, slice_astc);
            prof_astc_encoded_++;
        } else {
            prof_astc_cached_++;
        }
        }
        VkBufferImageCopy r{};
        r.bufferOffset = a.offset;
        r.bufferRowLength = lv.width_blocks * 4;
        r.bufferImageHeight = lv.height_blocks * 4;
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, layer, 1};
        r.imageOffset = {0, 0, static_cast<int32_t>(z)};
        r.imageExtent = {lv.width, lv.height, 1};
        regions.emplace_back(a.buffer, r);
    }
    return true;
}

const ProgramStage* Executor::count_check_stage(const ProgramStage& vs) {
    const uint64_t key = vs.code ^ (reinterpret_cast<uint64_t>(vs.control) * 0x9E3779B97F4A7C15ull) ^
                         (vs.spirv_id * 0xC2B2AE3D27D4EB4Full);
    if (auto it = count_check_stages_.find(key); it != count_check_stages_.end()) return it->second.get();
    auto stage = std::make_unique<ProgramStage>();
    stage->code = vs.code;
    stage->control = vs.control;
    stage->constants = vs.constants;
    stage->constants_size = vs.constants_size;
    stage->count_check = true;
    ensure_translated(*stage);
    if (!stage->module || stage->shader.stage != Stage::Vertex) stage.reset();
    static int logged = 0;
    if (logged++ < 4)
        NVN_LOG("vertex %016llx with the draw count check%s\n", static_cast<unsigned long long>(vs.print),
                stage ? "" : ": didn't translate, drawn without it");
    return (count_check_stages_[key] = std::move(stage)).get();
}

// ---- ASTC encodes in the background (see AstcJob) ---------------------------------------------------------------------

void Executor::defer_astc(AstcJob job) {
    astc_waiting_[job.texture]++;
    {
        std::lock_guard lock(astc_mutex_);
        astc_todo_.push_back(std::move(job));
    }
    if (!astc_started_) {
        astc_started_ = true;
        std::thread([this] { astc_worker(); }).detach();  // lives as long as the process (the executor is global)
    }
    astc_cv_.notify_one();
}

void Executor::astc_worker() {
    pthread_setname_np(pthread_self(), "AstcEncode");
    // Below the game's threads: this is catch-up work (the encoder's helper threads inherit it)
    setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), 10);
    std::vector<uint8_t> texels;
    for (;;) {
        AstcJob job;
        {
            std::unique_lock lock(astc_mutex_);
            astc_cv_.wait(lock, [&] { return !astc_todo_.empty(); });
            job = std::move(astc_todo_.front());
            astc_todo_.pop_front();
        }
        const uint32_t w = job.wb * 4, h = job.hb * 4;
        texels.resize(size_t{w} * h * 4);
        bcn::decode(static_cast<bcn::Kind>(job.kind), job.blocks.data(), job.wb, job.hb, texels.data(), size_t{w} * 4, w, h);
        job.astc.resize(size_t{job.wb} * job.hb * 16);
        job.ok = astc::encode(texels.data(), w, h, job.srgb, job.astc.data());
        if (job.ok) astc::store(job.cache_key, job.astc.data(), job.astc.size());  // (the next time, a file read)
        job.blocks = {};
        std::lock_guard lock(astc_mutex_);
        astc_done_.push_back(std::move(job));
    }
}

// At the start of a submit (no render pass yet): the levels that came back, into their images, for the textures that
// still have them (same image, same blocks as when they left).
void Executor::apply_astc_results() {
    std::deque<AstcJob> done;
    {
        std::lock_guard lock(astc_mutex_);
        if (astc_done_.empty()) return;
        done.swap(astc_done_);
    }
    bool open = false;  // the barrier before the copies, once
    for (AstcJob& j : done) {
        auto it = astc_waiting_.find(j.texture);
        if (it == astc_waiting_.end()) continue;  // retired meanwhile
        Texture& t = *j.texture;
        if (t.serial != j.serial) continue;
        if (--it->second <= 0) astc_waiting_.erase(it);
        const size_t slot = size_t{j.layer} * upload_levels(t) + j.level;
        if (t.image != j.image || slot >= t.astc_pending.size() || t.astc_pending[slot] != j.key) continue;  // old
        if (!j.ok) continue;  // (stays out of what's sampled; the next upload of the level tries again)
        Staging::Alloc a = staging_->alloc_contiguous(j.astc.size(), 16);
        if (!a.data) continue;
        memcpy(a.data, j.astc.data(), j.astc.size());
        if (!open) {
            open = true;
            VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            mb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0,
                                 nullptr, 0, nullptr);
        }
        VkBufferImageCopy r{};
        r.bufferOffset = a.offset;
        r.bufferRowLength = j.wb * 4;
        r.bufferImageHeight = j.hb * 4;
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, j.level, j.layer, 1};
        r.imageExtent = {j.width, j.height, 1};
        note_gpu_work();
        vkCmdCopyBufferToImage(cmd_, a.buffer, t.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
        t.astc_pending[slot] = 0;
        ++t.content_version;
        t.residency.encoded(j.layer, j.level);
        prof_astc_applied_++;
    }
    if (open) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
    }
}

bool Executor::bc_on_gpu(const Texture& t) {
#ifdef HAVE_bc_decode_SPV
    const bcn::Kind kind = static_cast<bcn::Kind>(t.bcn);
    if (kind < bcn::kBC1 || kind > bcn::kBC5S || NVN_FLAG("no_gpu_bc")) return false;
    if (!bc_setup_done_) {
        bc_setup_done_ = true;
        VkContext* c = vk();
        const VkDescriptorSetLayoutBinding b[2] = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dl.bindingCount = 2;
        dl.pBindings = b;
        const VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, 5 * sizeof(uint32_t)};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &bc_set_layout_;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &pc;
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        mi.codeSize = sizeof k_bc_decode_spv;
        mi.pCode = k_bc_decode_spv;
        VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateDescriptorSetLayout(c->device, &dl, nullptr, &bc_set_layout_) == VK_SUCCESS &&
            vkCreatePipelineLayout(c->device, &pl, nullptr, &bc_layout_) == VK_SUCCESS &&
            vkCreateShaderModule(c->device, &mi, nullptr, &module) == VK_SUCCESS) {
            VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            ci.stage.module = module;
            ci.stage.pName = "main";
            ci.layout = bc_layout_;
            if (vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, nullptr, &bc_pipeline_) != VK_SUCCESS)
                bc_pipeline_ = VK_NULL_HANDLE;
        }
        if (module) vkDestroyShaderModule(c->device, module, nullptr);
        NVN_LOG("BC1-BC5 textures: %s\n", bc_pipeline_ ? "decoded on the GPU" : "the GPU decoder didn't build, decoded on the CPU");
    }
    return bc_pipeline_ != VK_NULL_HANDLE;
#else
    (void)t;
    return false;  // (spirv.h from before bc_decode.comp: tools/shaders.sh adds it)
#endif
}

bool Executor::unswizzle_on_gpu() {
#ifdef HAVE_deswizzle_SPV
    if (!NVN_FLAG("gpu_unswizzle")) return false;
    if (!unswizzle_setup_done_) {
        unswizzle_setup_done_ = true;
        VkContext* c = vk();
        const VkDescriptorSetLayoutBinding b[2] = {
            {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
            {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dl.bindingCount = 2;
        dl.pBindings = b;
        const VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT, 0, 9 * sizeof(uint32_t)};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 1;
        pl.pSetLayouts = &unswizzle_set_layout_;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &pc;
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        mi.codeSize = sizeof k_deswizzle_spv;
        mi.pCode = k_deswizzle_spv;
        VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateDescriptorSetLayout(c->device, &dl, nullptr, &unswizzle_set_layout_) == VK_SUCCESS &&
            vkCreatePipelineLayout(c->device, &pl, nullptr, &unswizzle_layout_) == VK_SUCCESS &&
            vkCreateShaderModule(c->device, &mi, nullptr, &module) == VK_SUCCESS) {
            VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            ci.stage.module = module;
            ci.stage.pName = "main";
            ci.layout = unswizzle_layout_;
            if (vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, nullptr, &unswizzle_pipeline_) != VK_SUCCESS)
                unswizzle_pipeline_ = VK_NULL_HANDLE;
        }
        if (module) vkDestroyShaderModule(c->device, module, nullptr);
        NVN_LOG("GPU texture unswizzle: %s\n", unswizzle_pipeline_ ? "on" : "the shader didn't build, unswizzled on the CPU");
    }
    return unswizzle_pipeline_ != VK_NULL_HANDLE;
#else
    return false;  // (spirv.h from before deswizzle.comp: tools/shaders.sh adds it)
#endif
}

// The level as it is (src, lv.size bytes) into staging, and a dispatch that writes its rows to out. false: nothing was
// recorded (no staging or set), and the caller unswizzles on the CPU.
bool Executor::gpu_unswizzle(const LevelLayout& lv, uint32_t bpb, const uint8_t* src, const Staging::Alloc& out) {
    const VkDeviceSize align = std::max<VkDeviceSize>(vk()->props.limits.minStorageBufferOffsetAlignment, 16);
    Staging::Alloc tiled = staging_->alloc(lv.size, align);
    const VkDescriptorSet set = tiled.data ? descriptor_set(unswizzle_set_layout_) : VK_NULL_HANDLE;
    if (!set) return false;
    memcpy(tiled.data, src, lv.size);
    const uint32_t row_bytes = lv.width_blocks * bpb, rows = std::max(lv.depth, 1u) * lv.height_blocks;
    const uint32_t gobs_per_row = lv.row_gobs ? lv.row_gobs : (row_bytes + 63) / 64;
    const uint32_t bh = lv.block_height_gobs, bd = std::max(lv.block_depth, 1u);
    const uint32_t block_size = 512 * bh * bd, block_row_size = block_size * gobs_per_row;
    const uint32_t slice_size = block_row_size * ((lv.height_blocks + 8 * bh - 1) / (8 * bh));
    const size_t linear_size = size_t{row_bytes} * rows;
    const VkDescriptorBufferInfo in{tiled.buffer, tiled.offset, lv.size}, to{out.buffer, out.offset, linear_size};
    VkWriteDescriptorSet w[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
    for (int k = 0; k < 2; k++) {
        w[k].dstSet = set;
        w[k].dstBinding = static_cast<uint32_t>(k);
        w[k].descriptorCount = 1;
        w[k].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[k].pBufferInfo = k ? &to : &in;
    }
    vkUpdateDescriptorSets(vk()->device, 2, w, 0, nullptr);
    const uint32_t groups = static_cast<uint32_t>((linear_size / 16 + 63) / 64);
    const uint32_t gx = std::min<uint32_t>(groups, 65535), gy = (groups + gx - 1) / gx;
    const uint32_t params[9] = {row_bytes / 16, rows, lv.height_blocks, bh, bd, block_size, block_row_size, slice_size, gx};
    note_gpu_work();
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, unswizzle_pipeline_);
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, unswizzle_layout_, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(cmd_, unswizzle_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof params, params);
    vkCmdDispatch(cmd_, gx, gy, 1);
    prof_gpu_unswizzled_++;
    return true;
}

bool Executor::bc_gpu_decode(const Texture& t, const LevelLayout& lv, VkBuffer blocks, VkDeviceSize offset, uint32_t level,
                             uint32_t layer, std::vector<std::pair<VkBuffer, VkBufferImageCopy>>& regions) {
    const bcn::Kind kind = static_cast<bcn::Kind>(t.bcn);
    const VkDeviceSize align = std::max<VkDeviceSize>(vk()->props.limits.minStorageBufferOffsetAlignment, 16);
    // Texels per output row: BC4 and BC5 rounded up to whole 32-bit words (4 R8 or 2 RG8 texels; bc_decode.comp writes
    // a word per invocation). The copy to the image skips the extra ones (bufferRowLength).
    const uint32_t tb = bcn::texel_bytes(kind), row_texels = (lv.width + 4 / tb - 1) / (4 / tb) * (4 / tb);
    const size_t pitch = size_t{row_texels} * tb, block_row = size_t{lv.width_blocks} * bcn::block_bytes(kind);
    const uint32_t per_chunk = static_cast<uint32_t>(std::max<size_t>(1, kStagingChunk / std::max<size_t>(pitch * 4, 1)));
    LoadTimer timer(*this, kLdDeswizzle, size_t{lv.width} * lv.height * 4);
    note_gpu_work();
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, bc_pipeline_);
    bool any = false;
    for (uint32_t z = 0; z < std::max(lv.depth, 1u); z++) {
        for (uint32_t by = 0; by < lv.height_blocks; by += per_chunk) {
            const uint32_t hb = std::min(per_chunk, lv.height_blocks - by);
            const uint32_t rows = std::min(hb * 4, lv.height - std::min(lv.height, by * 4));
            if (!rows) break;
            Staging::Alloc out = staging_->alloc(pitch * rows, align);
            const VkDescriptorSet set = out.data ? descriptor_set(bc_set_layout_) : VK_NULL_HANDLE;
            if (!set) return false;
            // The blocks of this slice and these rows, bound from an aligned offset; the rest goes in src_word.
            const VkDeviceSize first = offset + (VkDeviceSize{z} * lv.height_blocks + by) * block_row;
            const VkDeviceSize base = first / align * align;
            const VkDescriptorBufferInfo src{blocks, base, VK_WHOLE_SIZE}, dst{out.buffer, out.offset, pitch * rows};
            VkWriteDescriptorSet w[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
            for (int k = 0; k < 2; k++) {
                w[k].dstSet = set;
                w[k].dstBinding = static_cast<uint32_t>(k);
                w[k].descriptorCount = 1;
                w[k].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w[k].pBufferInfo = k ? &dst : &src;
            }
            vkUpdateDescriptorSets(vk()->device, 2, w, 0, nullptr);
            const uint32_t params[5] = {static_cast<uint32_t>((first - base) / 4), row_texels, rows, lv.width_blocks,
                                        static_cast<uint32_t>(kind)};
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, bc_layout_, 0, 1, &set, 0, nullptr);
            vkCmdPushConstants(cmd_, bc_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof params, params);
            note_gpu_work();
            vkCmdDispatch(cmd_, (static_cast<uint32_t>(pitch / 4) + 7) / 8, (rows + 7) / 8, 1);
            any = true;
            VkBufferImageCopy r{};
            r.bufferOffset = out.offset;
            r.bufferRowLength = row_texels;
            r.bufferImageHeight = rows;
            r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, layer, 1};
            r.imageOffset = {0, static_cast<int32_t>(by * 4), static_cast<int32_t>(z)};
            r.imageExtent = {lv.width, rows, 1};
            regions.emplace_back(out.buffer, r);
        }
    }
    if (any) {  // the texels, before the copies to the image (recorded by upload after this)
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
    }
    return true;
}

bool Executor::decode_level(Texture& t, const LevelLayout& lv, const uint8_t* blocks, uint32_t level, uint32_t layer,
                            std::vector<std::pair<VkBuffer, VkBufferImageCopy>>& regions, VkBuffer blocks_buffer,
                            VkDeviceSize blocks_offset) {
    if (t.astc) return astc_level(t, lv, blocks, level, layer, regions);
    if (blocks_buffer && bc_pipeline_) return bc_gpu_decode(t, lv, blocks_buffer, blocks_offset, level, layer, regions);
    const bcn::Kind kind = static_cast<bcn::Kind>(t.bcn);
    const uint32_t bb = bcn::block_bytes(kind), tb = bcn::texel_bytes(kind);
    const size_t pitch = size_t{lv.width} * tb, block_row = size_t{lv.width_blocks} * bb;
    const uint32_t per_chunk = static_cast<uint32_t>(std::max<size_t>(1, kStagingChunk / std::max<size_t>(pitch * 4, 1)));
    LoadTimer timer(*this, kLdDeswizzle, size_t{lv.width} * lv.height * tb);
    for (uint32_t z = 0; z < std::max(lv.depth, 1u); z++) {
        const uint8_t* slice = blocks + size_t{z} * lv.height_blocks * block_row;
        for (uint32_t by = 0; by < lv.height_blocks; by += per_chunk) {
            const uint32_t hb = std::min(per_chunk, lv.height_blocks - by);
            const uint32_t rows = std::min(hb * 4, lv.height - std::min(lv.height, by * 4));
            if (!rows) break;
            Staging::Alloc a = staging_->alloc(pitch * rows, 16);
            if (!a.data) return false;
            const std::function<void(size_t, size_t)> fn = [&](size_t r0, size_t r1) {
                const uint32_t first = by + static_cast<uint32_t>(r0), count = static_cast<uint32_t>(r1 - r0);
                const uint32_t y0 = first * 4 - by * 4;
                bcn::decode(kind, slice + size_t{first} * block_row, lv.width_blocks, count, a.data + y0 * pitch, pitch, lv.width,
                            rows - y0);
            };
            if (hb > 4 && !NVN_FLAG("single_thread_upload")) row_pool().run(hb, std::max<size_t>(1, hb / 8), fn);
            else fn(0, hb);
            VkBufferImageCopy r{};
            r.bufferOffset = a.offset;
            r.bufferRowLength = lv.width;
            r.bufferImageHeight = rows;
            r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, layer, 1};
            r.imageOffset = {0, static_cast<int32_t>(by * 4), static_cast<int32_t>(z)};
            r.imageExtent = {lv.width, rows, 1};
            regions.emplace_back(a.buffer, r);
        }
    }
    return true;
}

void Executor::upload(Texture& t) {
    if (!t.storage || !t.image) return;
    LoadTimer timer(*this, kLdUpload, t.layout.total_size());
    if (std::find(frame_uploads_.begin(), frame_uploads_.end(), &t) == frame_uploads_.end()) frame_uploads_.push_back(&t);
    if (!t.uploaded) prof_texture_bytes_ += t.layout.total_size();  // DEPURACIÓN: memoria de texturas subidas
    // Vigilancia (ver watch_candidate), antes de leer la memoria: lo que la CPU escriba desde aquí salta y lo ve la próxima
    // revisión. Empezar a vigilar da todas las páginas por escritas: se sube entera una vez.
    if (watch_textures_ && !t.watched && watch_candidate(t, t.level_hash_image == t.image && !t.level_full_hash.empty()) &&
        nnsdk::watch::watchable(reinterpret_cast<uintptr_t>(t.storage), t.layout.total_size())) {
        t.watched = true;
        t.watch_keys.clear();
        prof_watched_textures_++;
        static int logged = 0;  // DEPURACIÓN
        if (logged++ < 20)
            NVN_LOG("textura vigilada %llu: %dx%dx%d formato 0x%x niveles %d (%zu KB) en %p\n", static_cast<unsigned long long>(t.serial),
                    t.desc.width, t.desc.height, t.desc.depth, t.desc.format, t.desc.levels, t.layout.total_size() >> 10,
                    static_cast<void*>(t.storage));
    }
    t.uploaded = false;  // a failed staging allocation must retry all unrecorded copies
    t.content_hash = content_hash(t);
    t.checked_frame = frame_;
    const TextureLayout& L = t.layout;
    const MemoryPool* pool = g_memory_pools.get(t.desc.pool);
    const uint32_t bpb = L.bytes_per_block;
    // Cada nivel puede caer en un trozo de staging distinto: una copia por trozo. (Antes se cortaba al
    // cambiar de trozo y las mips restantes se quedaban sin subir: basura en lo lejano, según lo lleno
    // que estuviera el staging.)
    std::vector<std::pair<VkBuffer, VkBufferImageCopy>> regions;
    const bool astc = t.vk_format >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && t.vk_format <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK;
    const uint32_t level_count = std::min<uint32_t>(L.levels.size(), std::max(t.desc.levels, 1));
    const size_t slots = size_t{std::min(L.layers, t.layers)} * level_count;
    const bool partial = t.level_full_hash.size() == slots && t.level_hash_image == t.image;
    t.level_hash_image = VK_NULL_HANDLE;
    if (t.level_full_hash.size() != slots) t.level_full_hash.assign(slots, 0);
    const bool native_astc = astc && !t.bcn && bpb == 16;
    std::vector<uint8_t> issues = (partial || t.gpu_levels) ? t.residency.issues : std::vector<uint8_t>{};
    const bool virtual_pool = pool && pool->reserved;
    if (virtual_pool || native_astc || t.astc) issues.resize(slots);
    if (t.desc.flags & 0x20) t.mapped_levels.resize(slots);
    const bool upload_v1 = NVN_FLAG("upload_v1");
    for (uint32_t layer = 0; layer < std::min(L.layers, t.layers); layer++) {
        for (uint32_t level = 0; level < L.levels.size() && level < static_cast<uint32_t>(std::max(t.desc.levels, 1)); level++) {
            const LevelLayout& lv = L.levels[level];
            const size_t linear_size = size_t{lv.width_blocks} * bpb * lv.height_blocks * lv.depth;
            const uint8_t* src = t.storage + layer * L.layer_size + lv.offset;
            const size_t index = size_t{layer} * level_count + level;
            if (t.owns_level(index)) continue;
            const bool backed = !virtual_pool || pool->covers(size_t(t.desc.offset) + size_t{layer} * L.layer_size + lv.offset, lv.size);
            if (!t.mapped_levels.empty()) t.mapped_levels[index].backed = backed;
            if (virtual_pool) {
                if (backed) issues[index] &= ~MipResidency::Unmapped;
                else issues[index] |= MipResidency::Unmapped;
            }
            {
                // A fully remapped slot must upload even if its guest bytes match the digest before a GPU write.
                const bool comparable = partial &&
                    (t.level_owners.empty() || t.level_owners[index] != Texture::LevelOwner::Remapped);
                if (comparable && full_hash(src, lv.size) == t.level_full_hash[index]) continue;
                // The streamer can write while deswizzle workers are running. Record the digest of the
                // exact bytes they consume, including changes outside the fast page samples.
                static thread_local std::vector<uint8_t> level_bytes;
                level_bytes.assign(src, src + lv.size);
                src = level_bytes.data();
                const uint64_t hash = full_hash(src, lv.size, t.mapped_levels.empty() ? nullptr : &t.mapped_levels[index].hashes);
                prof_full_hash_bytes_ += lv.size;
                if (comparable && hash == t.level_full_hash[index]) continue;
                t.level_full_hash[index] = hash;
                if (partial) t.full_check_until = frame_ + 8;
            }
            // Descomprimida en la CPU (t.bcn): los bloques van primero a una memoria de paso, y lo que llega a la GPU son
            // los texels (decode_level, más abajo)
            Staging::Alloc a{};
            static thread_local std::vector<uint8_t> bc_scratch;
            const bool bc_gpu = linear_size <= kStagingChunk && t.bcn && !t.astc && bc_on_gpu(t);
            const bool count_zeros = native_astc;
            // Unswizzled by the GPU: only what's copied to the image as it is
            const bool gpu_rows = !L.linear && !t.bcn && !count_zeros && !upload_v1 && (lv.width_blocks * bpb) % 16 == 0 &&
                                  lv.size <= kStagingChunk && linear_size >= 16 && unswizzle_on_gpu();
            if (bc_gpu || gpu_rows) {
                a = staging_->alloc(linear_size, std::max<VkDeviceSize>(vk()->props.limits.minStorageBufferOffsetAlignment, 16));
                if (!a.data) return;
            } else if (t.bcn) {
                bc_scratch.resize(linear_size);
                a.data = bc_scratch.data();
            } else {
                a = staging_->alloc_contiguous(linear_size, 16);
                if (!a.data) return;
            }
            // ASTC: un bloque a cero no lo produce ningún codificador (es inválido: magenta). Un nivel con más del 1%
            // así no está cargado aún (MH Rise crea sus texturas con todos los mips y trae los grandes por streaming);
            // lo muestreado empieza después (Texture::min_level). Los bloques a cero se cuentan al copiarlos, en la
            // memoria del juego: antes se releían del staging (HOST_COHERENT sin HOST_CACHED: sin caché de la CPU,
            // cada carga va a la RAM), 2 cargas de 8 bytes por bloque de cada nivel ASTC subido.
            // <datos>/upload_v1: como antes (copia trozo a trozo y bloques a cero releídos del staging).
            size_t zeros = 0;
            const bool gpu_done = gpu_rows && gpu_unswizzle(lv, bpb, src, a);
            if (upload_v1) {
                if (L.linear) {
                    for (uint32_t y = 0; y < lv.height_blocks; y++)
                        memcpy(a.data + size_t{y} * lv.width_blocks * bpb, src + size_t{y} * L.stride, lv.width_blocks * bpb);
                } else {
                    deswizzle_level_v1(lv, bpb, src, a.data);
                }
                if (count_zeros) {
                    static const uint8_t zero[16] = {};
                    for (size_t b = 0; b < linear_size / 16; b++) zeros += memcmp(a.data + b * 16, zero, 16) == 0;
                }
            } else if (L.linear) {
                for (uint32_t y = 0; y < lv.height_blocks; y++) {
                    const uint8_t* row = src + size_t{y} * L.stride;
                    memcpy(a.data + size_t{y} * lv.width_blocks * bpb, row, lv.width_blocks * bpb);
                    if (count_zeros) zeros += count_zero_chunks(row, size_t{lv.width_blocks} * bpb);
                }
            } else if (!gpu_done) {
                LoadTimer deswizzle_timer(*this, kLdDeswizzle, linear_size);
                const size_t rows = size_t{std::max(lv.depth, 1u)} * lv.height_blocks;
                const size_t row_bytes = std::max<size_t>(size_t{lv.width_blocks} * bpb, 1);
                if (linear_size >= kParallelDeswizzleMin && rows > 1 && !NVN_FLAG("single_thread_upload")) {
                    std::atomic<size_t> zero_total{0};
                    const std::function<void(size_t, size_t)> fn = [&](size_t r0, size_t r1) {
                        size_t z = 0;
                        deswizzle_rows(lv, bpb, src, a.data, r0, r1, count_zeros ? &z : nullptr);
                        if (z) zero_total.fetch_add(z, std::memory_order_relaxed);
                    };
                    row_pool().run(rows, std::max<size_t>(1, (64 << 10) / row_bytes), fn);
                    zeros += zero_total.load(std::memory_order_relaxed);
                } else {
                    deswizzle_level(lv, bpb, src, a.data, count_zeros ? &zeros : nullptr);
                }
            }
            if (count_zeros) {
                if (zeros * 100 > linear_size / 16) issues[index] |= MipResidency::InvalidAstc;
                else issues[index] &= ~MipResidency::InvalidAstc;
            }
            if (t.bcn) {
                if (!decode_level(t, lv, a.data, level, layer, regions, bc_gpu ? a.buffer : VK_NULL_HANDLE, a.offset)) return;
                if (t.astc) {
                    if (index < t.astc_pending.size() && t.astc_pending[index]) issues[index] |= MipResidency::Encoding;
                    else issues[index] &= ~MipResidency::Encoding;
                }
                continue;
            }
            VkBufferImageCopy r{};
            r.bufferOffset = a.offset;
            r.bufferRowLength = lv.width_blocks * L.block_w;
            r.bufferImageHeight = lv.height_blocks * L.block_h;
            r.imageSubresource = {t.aspect & ~VK_IMAGE_ASPECT_STENCIL_BIT, level, layer, 1};
            if (!r.imageSubresource.aspectMask) r.imageSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
            r.imageExtent = {lv.width, lv.height, std::max(lv.depth, 1u)};
            regions.emplace_back(a.buffer, r);
        }
    }
    if (!regions.empty()) {
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0,
                             nullptr, 0, nullptr);
    }
    for (size_t k = 0; k < regions.size();) {
        size_t end = k;
        std::vector<VkBufferImageCopy> batch;
        while (end < regions.size() && regions[end].first == regions[k].first) batch.push_back(regions[end++].second);
        note_gpu_work();
        vkCmdCopyBufferToImage(cmd_, regions[k].first, t.image, VK_IMAGE_LAYOUT_GENERAL,
                               static_cast<uint32_t>(batch.size()), batch.data());
        k = end;
    }
    if (!regions.empty()) {
        VkMemoryBarrier ready{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        ready.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        ready.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                             1, &ready, 0, nullptr, 0, nullptr);
    }
    t.level_hash_image = t.image;
    std::replace(t.level_owners.begin(), t.level_owners.end(), Texture::LevelOwner::Remapped, Texture::LevelOwner::Guest);
    t.residency.publish(level_count, std::move(issues));
    t.uploaded = true;
    ++t.content_version;
    if (!t.next_full_check) t.next_full_check = frame_ + 1 + t.serial % 120;
}

bool Executor::collect_texture_writes(Texture& t) {
    t.watch_submit = submit_index_;
    std::vector<std::pair<size_t, size_t>> runs;  // (offset, bytes) desde t.storage
    nnsdk::watch::collect(reinterpret_cast<uintptr_t>(t.storage), t.layout.total_size(), t.watch_keys, &runs);
    if (runs.empty()) return false;
    const TextureLayout& L = t.layout;
    const uint32_t levels = upload_levels(t), layers = upload_layers(t);
    const size_t slots = size_t{layers} * levels;
    if (t.level_full_hash.size() != slots) t.level_full_hash.assign(slots, 0);
    bool any = false;
    for (uint32_t layer = 0; layer < layers; layer++)
        for (uint32_t level = 0; level < levels; level++) {
            const size_t begin = size_t{layer} * L.layer_size + L.levels[level].offset, end = begin + L.levels[level].size;
            bool hit = false;
            for (auto [at, len] : runs) hit = hit || (at < end && begin < at + len);
            const size_t index = size_t{layer} * levels + level;
            if (!hit || t.owns_level(index)) continue;
            // Páginas escritas: ¿cambió de verdad? (con lo mismo, o vistas siempre por escritas porque un pool virtual las ve
            // en otra dirección, no se sube ni se corta el pase.) Se cierran antes de leer: lo de después, a la próxima.
            const uint64_t full = full_hash(t.storage + begin, L.levels[level].size);
            prof_full_hash_bytes_ += L.levels[level].size;
            if (full == t.level_full_hash[index]) continue;
            any = true;
        }
    // Cambiada entera las 8 últimas veces que cambió (vídeo, algo que la CPU rehace cada frame): fuera de la vigilancia,
    // que saltaría en cada página de cada frame; la huella muestreada ya ve esos cambios. Lo de ahora sí se sube. Un cambio
    // pequeño (un glifo) vuelve a contar desde cero; páginas dadas por escritas sin cambiar (alias) no cuentan.
    if (any) {
        size_t written = 0;
        for (auto [at, len] : runs) written += len;
        t.watch_rewrites = written * 2 >= t.layout.total_size() ? t.watch_rewrites + 1 : 0;
        if (t.watch_rewrites >= 8 && t.uploaded) {
            t.watched = false;
            t.no_watch = true;
            static int logged = 0;  // DEPURACIÓN
            if (logged++ < 20)
                NVN_LOG("textura %llu %dx%d formato 0x%x: cambia entera cada vez, deja de vigilarse\n",
                        static_cast<unsigned long long>(t.serial), t.desc.width, t.desc.height, t.desc.format);
        }
    }
    return any;
}

// Verify the notified parts against the same bytes that were encoded into the image.
// A repeated mapping can also publish new bytes in the very same physical pages.
static bool remap_matches_image(const Texture& t, const std::vector<Texture::MappingRange>& ranges, uint64_t& compared_bytes) {
    const uint32_t levels = upload_levels(t);
    if (t.level_hash_image != t.image || t.mapped_levels.size() != size_t{upload_layers(t)} * levels) return false;
    const MemoryPool* pool = g_memory_pools.get(t.desc.pool);
    for (uint32_t layer = 0; layer < upload_layers(t); ++layer) {
        for (uint32_t level = 0; level < levels; ++level) {
            const auto& lv = t.layout.levels[level];
            const size_t begin = size_t{layer} * t.layout.layer_size + lv.offset;
            const auto& cached = t.mapped_levels[size_t{layer} * levels + level];
            bool checked_backing = false;
            for (auto [first, last] : ranges) {
                if (first >= begin + lv.size) break;
                if (last <= begin) continue;
                const auto* bytes = t.storage + begin;
                if (!checked_backing && pool && cached.backed != pool->covers(size_t(t.desc.offset) + begin, lv.size))
                    return false;
                checked_backing = true;
                const size_t start = (std::max(first, begin) - begin) / kDigestChunk;
                const size_t stop = (std::min(last - begin, lv.size) - 1) / kDigestChunk;
                if (stop >= cached.hashes.size()) return false;
                for (size_t i = start; i <= stop; ++i) {
                    const size_t at = i * kDigestChunk;
                    const size_t size = std::min(kDigestChunk, lv.size - at);
                    compared_bytes += size;
                    if (range_hash(bytes + at, size) != cached.hashes[i]) return false;
                }
            }
        }
    }
    return true;
}

bool Executor::cpu_texture_changed(Texture& t) {
    if (prof_timing_) fine_.tex_checks++;
    if (!t.uploaded) return true;
    const bool watched = t.watched && watch_textures_;
    if (watched && t.watch_submit != submit_index_ && collect_texture_writes(t)) return true;
    if (watched && frame_ < t.next_full_check) return false;
    if (t.checked_frame == frame_) return false;
    // <datos>/print_every_2 (en vivo, apagado por defecto): una textura revisada en el frame anterior no se vuelve a mirar en
    // este (la huella de las ~600 que sube la CPU, cada frame, es casi todo el tramo "texturas"). Lo que la CPU reescriba
    // se ve como mucho un frame más tarde; WriteTexels y CopyBufferToTexture no pasan por aquí (apply_texel_writes) y las
    // vigiladas tampoco.
    if (t.checked_frame + 1 == frame_ && t.content_hash && NVN_FLAG("print_every_2")) return false;
    t.checked_frame = frame_;
    if (frame_ <= t.full_check_until || frame_ >= t.next_full_check) {
        t.next_full_check = frame_ + 120;
        const uint32_t levels = upload_levels(t);
        for (uint32_t layer = 0; layer < upload_layers(t); ++layer) {
            for (uint32_t level = 0; level < levels; ++level) {
                if (t.owns_level(size_t{layer} * levels + level)) continue;
                const LevelLayout& lv = t.layout.levels[level];
                const uint8_t* src = t.storage + size_t{layer} * t.layout.layer_size + lv.offset;
                prof_full_hash_bytes_ += lv.size;
                if (full_hash(src, lv.size) != t.level_full_hash[size_t{layer} * levels + level]) {
                    ++prof_audit_changes_;
                    return true;
                }
            }
        }
        return false;
    }
    if (!prof_timing_) return content_hash(t) != t.content_hash;
    const auto t0 = std::chrono::steady_clock::now();
    const bool changed = content_hash(t) != t.content_hash;
    fine_.tex_hashes++;
    fine_.tex_hash_bytes += t.layout.total_size();
    fine_.tex_hash_ns += ns_since(t0);
    return changed;
}

VkImageView Executor::view(Texture& t, const TextureViewDesc& v, ViewUse use, ImageViewSpec* result) {
    const bool sampled = use == ViewUse::Sampled;
    if (sampled && !result && t.view_memo && t.view_memo_serial == t.serial &&
        t.view_memo_revision == t.residency.revision && memcmp(&v, &t.view_memo_desc, sizeof v) == 0)
        return t.view_memo;
    const ImageViewSpec spec = describe_view(t, v, use);
    if (result) *result = spec;
    if (!spec.range.aspectMask) {
        NVN_LOG("invalid image view range on texture %llu: mip %d layer %d\n",
                (unsigned long long)t.serial, v.base_level, v.base_layer);
        return VK_NULL_HANDLE;
    }
    if (!spec.range.levelCount) return VK_NULL_HANDLE; // no resident mip in the requested view
    BytesKey<ViewKeyData> key;
    key.value.texture = t.serial;
    key.value.spec = spec;
    auto it = views_.find(key);
    VkImageView iv = VK_NULL_HANDLE;
    if (it != views_.end()) {
        iv = it->second;
    } else {
        if (use == ViewUse::Storage && !storage_capable(spec.format)) return VK_NULL_HANDLE;
        VkImageViewUsageCreateInfo usage{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
        usage.usage = spec.usage;
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.pNext = &usage;
        vi.image = t.image;
        vi.format = spec.format;
        vi.viewType = spec.type;
        vi.subresourceRange = spec.range;
        vi.components = spec.components;
        LoadTimer timer(*this, kLdView);
        if (VkResult r = vkCreateImageView(vk()->device, &vi, nullptr, &iv); r != VK_SUCCESS) {
            NVN_LOG("vkCreateImageView (texture %llu, format %d, type %d): %s\n",
                    (unsigned long long)t.serial, spec.format, spec.type, vk_result_name(r));
            return VK_NULL_HANDLE;
        }
        views_.emplace(key, iv);
    }
    if (sampled) {
        t.view_memo = iv;
        t.view_memo_serial = t.serial;
        t.view_memo_revision = t.residency.revision;
        t.view_memo_desc = v;
    }
    return iv;
}

// Filtros de NVN: min NEAREST 0, LINEAR 1, NEAREST_MIPMAP_NEAREST 2, LINEAR_MIPMAP_NEAREST 3,
// NEAREST_MIPMAP_LINEAR 4, LINEAR_MIPMAP_LINEAR 5; mag NEAREST 0, LINEAR 1. Wrap: CLAMP 0, REPEAT 1,
// MIRROR_CLAMP 2, MIRROR_CLAMP_TO_EDGE 3, MIRROR_CLAMP_TO_BORDER 4, CLAMP_TO_BORDER 5, MIRRORED_REPEAT 6,
// CLAMP_TO_EDGE 7.
VkSampler Executor::sampler(Sampler* s) {
    if (!s) return default_sampler_;
    const SamplerDesc& d = s->desc;
    // Comparación (las sombras: el 268 de los mapas de sombra de MH Rise, LINEAR_MIPMAP_NEAREST con 1x): sin el 4x mínimo
    // de abajo, con el anisotrópico que pida el juego, como Eden (TSCEntry::MaxAnisotropy: 1x si depth_compare_enabled y
    // el juego no pide más). Cómo filtra el Adreno una comparación con anisotrópico es cosa del driver: sospechoso de las
    // sombras de árboles a cuadros de las Ruinas (sin probar aún en la Flip). <datos>/af_compare en vivo: el de
    // antes, en su propio VkSampler (vk_af).
    const bool legacy_af = d.compare_mode != 0 && sampler_compare_af_;
    // <datos>/game_af (en vivo, para medir): sin el 4x mínimo de abajo, solo el anisotrópico que pide el juego, en el
    // otro VkSampler (vk_af, que en los que no son de comparación no se usaba). Cuesta GPU en los pases con muchas texturas
    // (el de la escena a 960x540); la nitidez del suelo en ángulo baja a la de la consola.
    const bool game_af = d.compare_mode == 0 && sampler_game_af_;
    // Sombras apagadas (<datos>/no_shadows, en vivo; el menú de opciones): la comparación siempre pasa, así que toda
    // lectura de un mapa de sombras da "iluminado" se haya dibujado o no (los draws de las cascadas se saltan en draw()).
    // En su propio VkSampler, para poder volver al momento.
    const bool lit = d.compare_mode != 0 && sampler_no_shadows_;
    VkSampler& out = lit ? s->vk_lit : legacy_af || game_af ? s->vk_af : s->vk;
    if (out) return out;
    auto wrap = [](int w) {
        switch (w) {
            case 1: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
            case 2:
            case 3: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
            case 5: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
            case 6: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
            default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        }
    };
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = d.mag_filter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    si.minFilter = (d.min_filter == 1 || d.min_filter == 3 || d.min_filter == 5) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    si.mipmapMode = (d.min_filter == 4 || d.min_filter == 5) ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = wrap(d.wrap[0]);
    si.addressModeV = wrap(d.wrap[1]);
    si.addressModeW = wrap(d.wrap[2]);
    si.mipLodBias = d.lod_bias;
    // Anisotrópico: el que pide el juego (antes se ignoraba y el suelo y las paredes se emborronaban en ángulo), y como
    // mínimo 4x en las texturas con mipmaps y filtro lineal. no_af al crear el sampler: como antes.
    {
        const bool mip_linear = si.minFilter == VK_FILTER_LINEAR && d.min_filter >= 2;
        float af = d.max_anisotropy;
        if (mip_linear && !game_af && (!d.compare_mode || legacy_af)) af = std::max(af, 4.0f);
        af = std::min(af, vk()->props.limits.maxSamplerAnisotropy);
        si.anisotropyEnable = af > 1.0f && !NVN_FLAG("no_af") ? VK_TRUE : VK_FALSE;
        si.maxAnisotropy = si.anisotropyEnable ? af : 1.0f;
    }
    si.compareEnable = d.compare_mode != 0;
    si.compareOp = lit ? VK_COMPARE_OP_ALWAYS : vk_compare_op(d.compare_func);
    si.minLod = d.min_filter >= 2 ? std::max(d.lod_min, 0.0f) : 0.0f;
    si.maxLod = d.min_filter >= 2 ? std::min(d.lod_max, 16.0f) : 0.25f;
    si.borderColor = d.border[3] > 0.5f ? (d.border[0] > 0.5f ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE
                                                                : VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK)
                                        : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    // Borde exacto (VK_EXT_custom_border_color) si no es uno de los tres de Vulkan: con CLAMP_TO_BORDER el borde de una
    // textura de comparación es la profundidad con la que se compara fuera del mapa de sombras. <datos>/no_exact_border
    // al crear el sampler: la aproximación de antes.
    VkSamplerCustomBorderColorCreateInfoEXT custom{VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT};
    const bool border_used = si.addressModeU == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
                             si.addressModeV == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
                             si.addressModeW == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    const float* bc = d.border;
    const bool preset = (bc[0] == 0 && bc[1] == 0 && bc[2] == 0 && (bc[3] == 0 || bc[3] == 1)) ||
                        (bc[0] == 1 && bc[1] == 1 && bc[2] == 1 && bc[3] == 1);
    static std::atomic<uint32_t> custom_borders{0};
    if (border_used && !preset && vk()->custom_border_color && custom_borders.load() < vk()->max_custom_border_samplers &&
        !NVN_FLAG("no_exact_border")) {
        custom_borders++;
        si.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
        memcpy(custom.customBorderColor.float32, bc, sizeof custom.customBorderColor.float32);
        custom.format = VK_FORMAT_UNDEFINED;  // customBorderColorWithoutFormat
        custom.pNext = si.pNext;
        si.pNext = &custom;
    }
    // Reducción MIN/MAX (NVN: 0 media, 1 mínimo, 2 máximo; VK_EXT_sampler_filter_minmax). Vulkan no la admite con
    // comparación. <datos>/no_reduction al crear el sampler: siempre la media, como antes.
    VkSamplerReductionModeCreateInfoEXT reduction{VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO_EXT};
    if ((d.reduction == 1 || d.reduction == 2) && !si.compareEnable && vk()->filter_minmax && !NVN_FLAG("no_reduction")) {
        reduction.reductionMode = d.reduction == 1 ? VK_SAMPLER_REDUCTION_MODE_MIN_EXT : VK_SAMPLER_REDUCTION_MODE_MAX_EXT;
        reduction.pNext = si.pNext;
        si.pNext = &reduction;
    }
    if (border_used || d.reduction || d.compare_mode) {  // DEPURACIÓN: los que no son un sampler normal, una vez
        static std::atomic<int> logged{0};
        if (logged++ < 64)
            NVN_LOG("sampler nuevo: wrap %d/%d/%d borde %.3g %.3g %.3g %.3g (%s) compare %d func %d reducción %d (%s) filtros %d/%d "
                    "aniso %.0f%s\n",
                    d.wrap[0], d.wrap[1], d.wrap[2], bc[0], bc[1], bc[2], bc[3],
                    si.borderColor == VK_BORDER_COLOR_FLOAT_CUSTOM_EXT ? "exacto" : preset ? "de Vulkan" : "aproximado",
                    d.compare_mode, d.compare_func, d.reduction, si.pNext == &reduction ? "sí" : "no", d.min_filter, d.mag_filter,
                    si.maxAnisotropy, legacy_af ? " (af_compare)" : "");
    }
    LoadTimer timer(*this, kLdSampler);
    if (vkCreateSampler(vk()->device, &si, nullptr, &out) != VK_SUCCESS) {
        out = VK_NULL_HANDLE;
        return default_sampler_;
    }
    return out;
}

// --- Render passes ---
// En el Adreno cada pase pasa sus adjuntos por la memoria del tile (GMEM): carga al empezar (LOAD), guarda al acabar
// (STORE), en cada bin. Lo que sigue intenta no mover lo que no hace falta: borrados como loadOp CLEAR, nada de
// guardar lo que el pase no escribe, área de render ajustada a lo que se dibuja y menos cortes de pase.

VkRenderPass Executor::render_pass(const VkFormat* colors, uint32_t count, VkFormat depth, VkSampleCountFlagBits samples,
                                   const PassOps& ops) {
    std::string key = std::string(reinterpret_cast<const char*>(colors), count * sizeof(VkFormat)) + key_of(depth) +
                      key_of(samples) + key_of(ops);
    std::lock_guard lock(objects_mutex_);  // el hilo del precalentado también los pide
    if (auto it = render_passes_.find(key); it != render_passes_.end()) return it->second;
    std::vector<VkAttachmentDescription> atts;
    std::vector<VkAttachmentReference> color_refs;
    for (uint32_t k = 0; k < count; k++) {
        if (colors[k] == VK_FORMAT_UNDEFINED) {
            color_refs.push_back({VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_GENERAL});
            continue;
        }
        color_refs.push_back({static_cast<uint32_t>(atts.size()), VK_IMAGE_LAYOUT_GENERAL});
        atts.push_back({0, colors[k], samples, ops.color_load[k], ops.color_store[k], VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                        VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL});
    }
    VkAttachmentReference depth_ref{VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_GENERAL};
    if (depth != VK_FORMAT_UNDEFINED) {
        depth_ref.attachment = static_cast<uint32_t>(atts.size());
        atts.push_back({0, depth, samples, ops.depth_load, ops.depth_store, ops.stencil_load, ops.stencil_store,
                        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL});
    }
    VkSubpassDescription sub{};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = count;
    sub.pColorAttachments = color_refs.data();
    sub.pDepthStencilAttachment = depth != VK_FORMAT_UNDEFINED ? &depth_ref : nullptr;
    // Dependencias completas con lo de antes y lo de después: las imágenes no cambian de layout.
    VkSubpassDependency deps[2] = {
        {VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
         VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, 0},
        {0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
         VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, 0}};
    VkRenderPassCreateInfo ri{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    ri.attachmentCount = static_cast<uint32_t>(atts.size());
    ri.pAttachments = atts.data();
    ri.subpassCount = 1;
    ri.pSubpasses = &sub;
    ri.dependencyCount = 2;
    ri.pDependencies = deps;
    VkRenderPass rp = VK_NULL_HANDLE;
    if (VkResult r = vkCreateRenderPass(vk()->device, &ri, nullptr, &rp); r != VK_SUCCESS)
        NVN_LOG("vkCreateRenderPass: %s\n", vk_result_name(r));
    render_passes_[key] = rp;
    return rp;
}

// Mirada adelante (ver PassScan): recorre los comandos desde el que pide el pase con una copia del estado que importa.
// pass: los adjuntos del pase (st_.rt, o más con merged_passes: entonces sigue por los targets que caben en él, y lo que
// escribe cada draw o borrado es de los targets de su tramo).
const Executor::PassScan& Executor::scan_pass(const cmd::SetRenderTargets& pass) {
    // Dónde está el comando en curso (lista, índice): lo lleva run(), no la dirección (una lista puede repetirse).
    const std::vector<const CommandList*>& lists = *run_lists_;
    size_t l = cur_list_, k = cur_index_;
    const bool inside = l < scan_.end_list || (l == scan_.end_list && k < scan_.end_cmd);  // antes de donde acabó
    const bool here = !scan_per_pass_ || (scan_.start_list == l && scan_.start_cmd == k);  // inspect_per_pass: la de este pase
    if (scan_.run == run_serial_ && inside && here && same_targets(scan_.rt, st_.rt) && same_targets(scan_.pass, pass))
        return scan_;
    PassScan s;
    s.run = run_serial_;
    s.rt = st_.rt;
    s.pass = pass;
    s.start_list = l, s.start_cmd = k;
    cmd::SetRenderTargets cur = st_.rt;  // los targets del juego en el tramo en curso (sin merged_passes, siempre s.rt)
    if (l == lists.size()) {  // no está en las listas (no debería pasar): todo escrito, área entera
        s.colors_written = ~0u;
        s.ds_read = s.ds_written = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
        s.cleared = true;
        scan_ = s;
        scan_.run = 0;
        return scan_;
    }
    s.x0 = s.y0 = INT64_MAX;
    s.x1 = s.y1 = INT64_MIN;
    uint32_t masks = st_.masks.masks;
    DepthStencilState ds = st_.depth_stencil;
    uint32_t stencil_mask[]{st_.stencil_mask[0], st_.stencil_mask[1]};
    const auto* fs = st_.stages[static_cast<int>(Stage::Fragment)];
    uint32_t exports = fs ? fs->shader.color_outputs : 0;
    uint8_t polygon_mode = st_.polygon.polygon_mode;
    VkViewport vp = st_.viewport;
    VkRect2D sc = st_.scissor;
    bool sc_set = st_.scissor_set;
    // Etapas que pueden convertir triángulos en puntos o líneas (geometry 4, teselación 8 y 16: bits de NVN).
    uint32_t geo = (st_.stages[static_cast<int>(Stage::Geometry)] ? 4u : 0u) |
                   (st_.stages[static_cast<int>(Stage::TessControl)] ? 8u : 0u) |
                   (st_.stages[static_cast<int>(Stage::TessEval)] ? 16u : 0u);
    while (l < lists.size()) {
        const std::vector<Command>& cmds = lists[l]->commands;
        if (k >= cmds.size()) {
            l++, k = 0;
            continue;
        }
        const Command& c = cmds[k];
        bool stop = false;
        switch (c.index()) {
            case kCommandIndex<cmd::SetRenderTargets>: {
                const cmd::SetRenderTargets& v = *std::get_if<cmd::SetRenderTargets>(&c);
                if (same_targets(v, cur)) break;
                if (merge_passes_ && targets_within(v, s.pass)) {  // sigue en el mismo pase (ver apply)
                    cur = v;
                    break;
                }
                stop = true;
                break;
            }
            case kCommandIndex<cmd::ClearColor>: {
                const cmd::ClearColor& v = *std::get_if<cmd::ClearColor>(&c);
                if (v.index >= 0 && v.index < cur.count && cur.colors[v.index]) s.colors_written |= 1u << v.index, s.cleared = true;
                break;
            }
            case kCommandIndex<cmd::ClearDepthStencil>: {
                const auto& clear = std::get<cmd::ClearDepthStencil>(c);
                if (cur.depth && (clear.depth_mask || clear.stencil_mask)) {
                    if (clear.depth_mask) s.ds_written |= VK_IMAGE_ASPECT_DEPTH_BIT;
                    if (clear.stencil_mask) s.ds_written |= VK_IMAGE_ASPECT_STENCIL_BIT;
                    s.cleared = true;
                }
                break;
            }
            case kCommandIndex<cmd::Viewport>: {
                const cmd::Viewport& v = *std::get_if<cmd::Viewport>(&c);
                if (v.index == 0) vp.x = v.x, vp.y = v.y, vp.width = v.w, vp.height = v.h;
                break;
            }
            case kCommandIndex<cmd::Scissor>: {
                const cmd::Scissor& v = *std::get_if<cmd::Scissor>(&c);
                if (v.index == 0) {  // como apply()
                    sc = scissor_rect(v);
                    sc_set = true;
                }
                break;
            }
            case kCommandIndex<ChannelMaskState>: masks = std::get_if<ChannelMaskState>(&c)->masks; break;
            case kCommandIndex<DepthStencilState>: ds = *std::get_if<DepthStencilState>(&c); break;
            case kCommandIndex<cmd::Stencil>: {
                const auto& v = std::get<cmd::Stencil>(c);
                if (v.what == cmd::Stencil::Mask) {
                    if (v.faces & 1) stencil_mask[0] = v.value;
                    if (v.faces & 2) stencil_mask[1] = v.value;
                }
                break;
            }
            case kCommandIndex<PolygonState>: polygon_mode = std::get_if<PolygonState>(&c)->polygon_mode; break;
            case kCommandIndex<cmd::BindProgram>: {
                const cmd::BindProgram& v = *std::get_if<cmd::BindProgram>(&c);
                geo &= ~(static_cast<uint32_t>(v.stages) & 28u);
                if (v.stages & 2) exports = 0;
                bool unknown = false;
                if (v.program)
                    for (const ProgramStage& ps : v.program->stages) {
                        if (!ps.tried) { geo |= 28u; unknown = true; }
                        else if (ps.module && ps.shader.stage == Stage::Fragment) exports = ps.shader.color_outputs;
                        else if (ps.module && ps.shader.stage == Stage::Geometry) geo |= 4u;
                        else if (ps.module && ps.shader.stage == Stage::TessControl) geo |= 8u;
                        else if (ps.module && ps.shader.stage == Stage::TessEval) geo |= 16u;
                    }
                if (unknown) exports = 0xff; // retain attachments until reflection is available
                break;
            }
            case kCommandIndex<cmd::Draw>: {
                const int p = std::get_if<cmd::Draw>(&c)->primitive;
                const auto access = draw_attachment_access(masks, exports, ds, stencil_mask);
                for (int t = 0; t < cur.count; t++)
                    if (cur.colors[t] && access.writes(t)) s.colors_written |= 1u << t;
                if (cur.depth) {
                    s.ds_read |= access.ds_reads;
                    s.ds_written |= access.ds_writes;
                }
                // Dónde puede dibujar: dentro del scissor (sin él, en todo el framebuffer); los triángulos rellenos,
                // además, dentro del viewport (Vulkan los recorta a él). Los puntos (tamaño del shader) y las líneas
                // pueden salirse del viewport, y un geometry shader o la teselación pueden convertir en eso lo que sea.
                int64_t x0 = 0, y0 = 0, x1 = INT32_MAX, y1 = INT32_MAX;
                if (sc_set)
                    x0 = sc.offset.x, y0 = sc.offset.y, x1 = x0 + sc.extent.width, y1 = y0 + sc.extent.height;
                const bool points_or_lines = p == 0 || p == 1 || p == 3 || p == 10 || p == 11 || p == 14;
                if (!points_or_lines && !geo && polygon_mode >= 2) {  // NVN: POINT 0, LINE 1, FILL 2
                    x0 = std::max<int64_t>(x0, static_cast<int64_t>(std::floor(std::min(vp.x, vp.x + vp.width))));
                    x1 = std::min<int64_t>(x1, static_cast<int64_t>(std::ceil(std::max(vp.x, vp.x + vp.width))));
                    y0 = std::max<int64_t>(y0, static_cast<int64_t>(std::floor(std::min(vp.y, vp.y + vp.height))));
                    y1 = std::min<int64_t>(y1, static_cast<int64_t>(std::ceil(std::max(vp.y, vp.y + vp.height))));
                }
                if (x1 > x0 && y1 > y0) {
                    s.drawn = true;
                    s.x0 = std::min(s.x0, x0), s.y0 = std::min(s.y0, y0), s.x1 = std::max(s.x1, x1), s.y1 = std::max(s.y1, y1);
                }
                break;
            }
            default: break;
        }
        if (stop) break;
        k++;
    }
    s.end_list = l;
    s.end_cmd = k;
    scan_ = s;
    return scan_;
}

// merged_passes: los adjuntos con los que empezar el pase de st_.rt. Desde el comando en curso, los targets que vienen y
// caben con los de hasta ahora (targets_fit), hasta unos que no caben, un dispatch, una copia o un borrado de textura (que
// cierran el pase de todas formas) o kMergeLookahead comandos. Lo que se añade tiene que ser ya de la GPU (con imagen y
// gpu_written: tenerlo antes como adjunto no cambia lo que se subiría de la memoria del juego), del mismo tamaño en su
// nivel y con las mismas muestras que los targets del juego. true si el pase lleva algo más que st_.rt.
bool Executor::merge_targets(cmd::SetRenderTargets* u) {
    constexpr int kMergeLookahead = 2048;
    const cmd::SetRenderTargets& rt = st_.rt;
    uint32_t width = 0, height = 0;
    int samples = 0;
    bool any = false, mixed = false;
    auto note = [&](const Texture* t, const TextureViewDesc& v) {
        if (!t) return;
        const auto [w,h] = mip_extent(*t, (v.flags & kViewLevels) ? v.base_level : 0);
        const int n = std::max(t->desc.samples, 1);
        if (!any) width = w, height = h, samples = n, any = true;
        else if (w != width || h != height || n != samples) mixed = true;
    };
    for (int k = 0; k < rt.count; k++) note(rt.colors[k], rt.color_views[k]);
    note(rt.depth, rt.depth_view);
    if (!any || mixed) return false;
    *u = rt;
    cmd::SetRenderTargets cur = rt;
    bool grew = false;
    const std::vector<const CommandList*>& lists = *run_lists_;
    size_t l = cur_list_, k = cur_index_;
    for (int n = 0; l < lists.size() && n < kMergeLookahead;) {
        const std::vector<Command>& cmds = lists[l]->commands;
        if (k >= cmds.size()) {
            l++, k = 0;
            continue;
        }
        const Command& c = cmds[k++];
        n++;
        const size_t i = c.index();
        if (i == kCommandIndex<cmd::Dispatch> || i == kCommandIndex<cmd::CopyTexture> || i == kCommandIndex<cmd::ClearTexture>) break;
        if (i != kCommandIndex<cmd::SetRenderTargets>) continue;
        const cmd::SetRenderTargets& v = *std::get_if<cmd::SetRenderTargets>(&c);
        if (same_targets(v, cur)) continue;
        if (!targets_fit(v, *u)) break;
        bool ok = true, adds = false;
        for (int j = 0; j < v.count && ok; j++) {
            const Texture* t = v.colors[j];
            if (!t || (j < u->count && u->colors[j])) continue;
            const auto& av = v.color_views[j];
            const auto [w,h] = mip_extent(*t, (av.flags & kViewLevels) ? av.base_level : 0);
            ok = t->image && t->gpu_levels == size_t{upload_levels(*t)} * upload_layers(*t) &&
                 w == width && h == height && std::max(t->desc.samples, 1) == samples;
            adds = true;
        }
        if (!ok) break;
        *u = targets_union(*u, v);
        cur = v;
        grew |= adds;
    }
    return grew;
}

bool Executor::ensure_render_pass(bool full_area, const cmd::SetRenderTargets* planned) {
    if (in_render_pass_ && !targets_changed_) return true;
    if (!split_why_) split_why_ = "targets nuevos";
    end_render_pass();
    targets_changed_ = false;
    // merged_passes: con los adjuntos de los targets que vienen y caben con estos (merge_targets).
    cmd::SetRenderTargets merged;
    const bool merging = planned ? !same_targets(*planned, st_.rt) : merge_passes_ && run_lists_ && merge_targets(&merged);
    const cmd::SetRenderTargets& rt = planned ? *planned : merging ? merged : st_.rt;
    VkFormat formats[8] = {};
    std::vector<VkImageView> views;
    // Adjunto i (en el orden de views, que es el del render pass): textura, vista y render target (-1: profundidad).
    Texture* targets[9] = {};
    ImageViewSpec target_specs[9]{};
    int slots[9] = {};
    int slot = 0;
    uint32_t width = UINT32_MAX, height = UINT32_MAX;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    auto add = [&](Texture* t, const TextureViewDesc& v) -> VkFormat {
        if (!t || !ensure_image(*t, cmd_)) return VK_FORMAT_UNDEFINED;
        if (t->desc.target == kTargetCube || t->desc.target == kTargetCubeArray) {
            const ProgramStage* ps = st_.stages[static_cast<int>(Stage::Fragment)];
            note_cube_write(*t, v, "render target", nullptr, ps ? ps->print : 0, frame_);
        }
        ImageViewSpec spec;
        VkImageView iv = view(*t, v, ViewUse::Attachment, &spec);
        if (!iv) return VK_FORMAT_UNDEFINED;
        const auto count = static_cast<VkSampleCountFlagBits>(std::max(t->desc.samples, 1));
        if (!views.empty() && samples != count) {
            NVN_LOG("incompatible framebuffer sample counts: %u and %u\n", unsigned(samples), unsigned(count));
            return VK_FORMAT_UNDEFINED;
        }
        targets[views.size()] = t, target_specs[views.size()] = spec, slots[views.size()] = slot;
        views.push_back(iv);
        const auto extent = mip_extent(*t, spec.range.baseMipLevel);
        width = std::min(width, extent.width);
        height = std::min(height, extent.height);
        samples = count;
        return spec.format;
    };
    for (int k = 0; k < rt.count; k++) {
        slot = k;
        formats[k] = add(rt.colors[k], rt.color_views[k]);
        if (rt.colors[k] && formats[k] == VK_FORMAT_UNDEFINED) return false;
        if (rt.colors[k] && std::find(frame_targets_.begin(), frame_targets_.end(), rt.colors[k]) == frame_targets_.end())
            frame_targets_.push_back(rt.colors[k]);
    }
    slot = -1;
    VkFormat depth = add(rt.depth, rt.depth_view);
    if (rt.depth && depth == VK_FORMAT_UNDEFINED) return false;
    if (rt.depth && std::find(frame_targets_.begin(), frame_targets_.end(), rt.depth) == frame_targets_.end())
        frame_targets_.push_back(rt.depth);
    if (views.empty()) return false;
    // El de LOAD/STORE: con él se crean los framebuffers y los pipelines (las cargas y guardados no cuentan para la
    // compatibilidad, así que valen para el que se empiece de verdad).
    VkRenderPass rp = render_pass(formats, static_cast<uint32_t>(rt.count), depth, samples);
    if (!rp) return false;
    std::string fb_key = key_of(rp) + std::string(reinterpret_cast<const char*>(views.data()), views.size() * sizeof(VkImageView)) +
                         key_of(width) + key_of(height);
    VkFramebuffer fb;
    if (auto it = framebuffers_.find(fb_key); it != framebuffers_.end()) {
        fb = it->second;
    } else {
        VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fi.renderPass = rp;
        fi.attachmentCount = static_cast<uint32_t>(views.size());
        fi.pAttachments = views.data();
        fi.width = width;
        fi.height = height;
        fi.layers = 1;
        VkResult fb_result;
        {
            LoadTimer timer(*this, kLdFramebuffer);
            fb_result = vkCreateFramebuffer(vk()->device, &fi, nullptr, &fb);
        }
        if (VkResult r = fb_result; r != VK_SUCCESS) {
            NVN_LOG("vkCreateFramebuffer: %s\n", vk_result_name(r));
            return false;
        }
        framebuffers_[fb_key] = fb;
    }
    const uint32_t n = static_cast<uint32_t>(views.size());
    // A framebuffer binding alone transfers no ownership. Color attachment LOAD and
    // partial clears must see guest memory even on the image's first use.
    for (uint32_t i = 0; i < n; ++i) {
        Texture& t = *targets[i];
        if (slots[i] >= 0 && t.desc.samples <= 1 && !prepare_contents(t, PriorContents::Preserve)) return false;
    }
    // Cargas: el borrado diferido de un adjunto se hace con loadOp CLEAR si es de esa misma vista y el pase cubre el
    // adjunto entero (el área será todo el framebuffer); si no, ahora, antes de empezar.
    PassOps ops = default_pass_ops();
    VkClearValue clears[9] = {};
    VkImageAspectFlags clear_aspects[9] = {};
    bool load_clear = false;
    for (uint32_t i = 0; i < n && !pending_clears_.empty(); i++) {
        Texture* t = targets[i];
        const auto& spec = target_specs[i];
        auto it = std::find_if(pending_clears_.begin(), pending_clears_.end(), [&](const PendingClear& p) {
            return p.texture == t && ranges_overlap(p.range, spec.range);
        });
        if (it == pending_clears_.end()) continue;
        const auto extent = mip_extent(*t, spec.range.baseMipLevel);
        bool usable = spec.format == t->vk_format && extent.width == width && extent.height == height;
        for (uint32_t j = 0; j < n; ++j)
            if (j != i && targets[j] == t && ranges_overlap(spec.range, target_specs[j].range)) usable = false;
        if (!usable) {
            flush_clears(t, &spec.range);
            continue;
        }
        if (slots[i] >= 0) {
            ops.color_load[slots[i]] = VK_ATTACHMENT_LOAD_OP_CLEAR;
        } else {
            if (it->range.aspectMask & VK_IMAGE_ASPECT_DEPTH_BIT) ops.depth_load = VK_ATTACHMENT_LOAD_OP_CLEAR;
            if (it->range.aspectMask & VK_IMAGE_ASPECT_STENCIL_BIT) ops.stencil_load = VK_ATTACHMENT_LOAD_OP_CLEAR;
        }
        clears[i] = it->value;
        clear_aspects[i] = it->range.aspectMask;
        pending_clears_.erase(it);
        load_clear = true;
        prof_clear_loads_++;
    }
    // Guardados y área, de la mirada adelante. Lo que ningún draw ni borrado del pase puede escribir no se guarda: con
    // STORE_OP_NONE la memoria se queda como estaba (nunca DONT_CARE: se puede leer después). El área es lo que pueden
    // tocar sus draws; con borrados, todo (loadOp CLEAR y vkCmdClearAttachments no pasan del área).
    const PassScan* scan = run_lists_ && (opt_.store_ops || opt_.area) ? &scan_pass(rt) : nullptr;
    if (scan && opt_.store_ops && vk()->store_op_none)
        for (uint32_t i = 0; i < n; i++) {
            if (slots[i] >= 0 && !((scan->colors_written >> slots[i]) & 1) && ops.color_load[slots[i]] != VK_ATTACHMENT_LOAD_OP_CLEAR) {
                ops.color_store[slots[i]] = VK_ATTACHMENT_STORE_OP_NONE_QCOM;
                if (vk()->load_op_none) ops.color_load[slots[i]] = VK_ATTACHMENT_LOAD_OP_NONE_EXT;
                prof_store_none_++;
            } else if (slots[i] < 0) {
                const auto written = scan->ds_written | clear_aspects[i];
                if (!(written & VK_IMAGE_ASPECT_DEPTH_BIT)) ops.depth_store = VK_ATTACHMENT_STORE_OP_NONE_QCOM;
                if (!(written & VK_IMAGE_ASPECT_STENCIL_BIT)) ops.stencil_store = VK_ATTACHMENT_STORE_OP_NONE_QCOM;
                if (vk()->load_op_none) {
                    if (!((written | scan->ds_read) & VK_IMAGE_ASPECT_DEPTH_BIT)) ops.depth_load = VK_ATTACHMENT_LOAD_OP_NONE_EXT;
                    if (!((written | scan->ds_read) & VK_IMAGE_ASPECT_STENCIL_BIT)) ops.stencil_load = VK_ATTACHMENT_LOAD_OP_NONE_EXT;
                }
                prof_store_none_ += (targets[i]->aspect & ~written) != 0;
            }
        }
    const VkRect2D full{{0, 0}, {width, height}};
    VkRect2D area = full;
    if (scan && opt_.area && !full_area && !load_clear && !scan->cleared && scan->drawn) {
        const int64_t x0 = std::clamp<int64_t>(scan->x0, 0, width), x1 = std::clamp<int64_t>(scan->x1, 0, width);
        const int64_t y0 = std::clamp<int64_t>(scan->y0, 0, height), y1 = std::clamp<int64_t>(scan->y1, 0, height);
        if (x1 > x0 && y1 > y0)
            area = {{static_cast<int32_t>(x0), static_cast<int32_t>(y0)}, {static_cast<uint32_t>(x1 - x0), static_cast<uint32_t>(y1 - y0)}};
    }
    const PassOps defaults = default_pass_ops();
    VkRenderPass begun = memcmp(&ops, &defaults, sizeof ops) == 0 ? rp : render_pass(formats, static_cast<uint32_t>(rt.count), depth, samples, ops);
    const bool fallback = !begun;  // sin esa variante: el de siempre, y los borrados dentro del pase
    VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    bi.renderPass = fallback ? rp : begun;
    bi.framebuffer = fb;
    bi.renderArea = area;
    bi.clearValueCount = load_clear && !fallback ? n : 0;
    bi.pClearValues = load_clear && !fallback ? clears : nullptr;
    {  // perfil de cortes (ver split_why_): este pase y el que se acaba de cerrar
        const bool same = last_end_why_ && last_end_fb_ && fb == last_end_fb_;
        const bool fits = last_end_why_ && last_end_fb_ && !same && width == last_end_extent_.width &&
                          height == last_end_extent_.height && samples == last_end_samples_ && targets_fit(rt, last_end_rt_);
        const char* kind = same ? "corte" : fits ? "cabía" : nullptr;
        if (kind && log_state_) {
            char key[112];
            snprintf(key, sizeof key, "%ux%u %s", width, height, last_end_why_);
            CutStat& cs = prof_cuts_[key];
            if (same) cs.same++, prof_passes_same_++;
            else cs.fits++, prof_passes_fit_++;
        }
        static char suffix[128];
        if (kind) snprintf(suffix, sizeof suffix, " tras %s: %s", kind, last_end_why_);
        prof_pass_suffix_ = kind ? suffix : nullptr;
        last_end_why_ = nullptr;
        last_end_fb_ = VK_NULL_HANDLE;
        cur_fb_ = fb;
        cur_pass_rt_ = rt;
    }
    rp_keep_masks_ = merging ? keep_masks_for(st_.rt) : ~0u;  // tras cur_pass_rt_
    prof_passes_merged_ += merging;
    prof_gpu_mark("pase");
    prof_pass_suffix_ = nullptr;
    if (crumbs_on_ && crumbs_passes_) {
        const Texture* c0 = rt.count > 0 ? rt.colors[0] : nullptr;
        char label[64];
        snprintf(label, sizeof label, "pase %ux%u %s%s", width, height,
                 c0 ? (format_info(c0->desc.format) ? format_info(c0->desc.format)->name : "?") : "-", rt.depth ? " +z" : "");
        crumb_begin('p', label);
    }
    note_gpu_work();
    vkCmdBeginRenderPass(cmd_, &bi, VK_SUBPASS_CONTENTS_INLINE);
    targets_changed_ = false;
    if (fallback && load_clear)
        for (uint32_t i = 0; i < n; i++)
            if (clear_aspects[i]) {
                VkClearAttachment a{clear_aspects[i], slots[i] >= 0 ? static_cast<uint32_t>(slots[i]) : 0u, clears[i]};
                VkClearRect r{full, 0, 1};
                vkCmdClearAttachments(cmd_, 1, &a, 1, &r);
            }
    prof_render_passes_++;
    in_render_pass_ = true;
    rp_ = rp;
    rp_extent_ = {width, height};
    rp_area_ = area;
    rp_area_full_ = area.offset.x == 0 && area.offset.y == 0 && area.extent.width == width && area.extent.height == height;
    if (!rp_area_full_) prof_area_cut_++;
    rp_color_count_ = static_cast<uint32_t>(rt.count);
    rp_samples_ = samples;
    std::copy(formats, formats + 8, rp_formats_);
    rp_depth_ = depth;
    rp_target_count_ = n;
    std::copy(targets, targets + n, rp_targets_);
    rp_recurring_ = n > 0;
    for (uint32_t i = 0; i < n; i++) {
        Texture* t = targets[i];
        if (!t) continue;
        if (t->rt_frame != frame_) {
            t->rt_recurring = t->rt_frame && frame_ - t->rt_frame <= 2;
            t->rt_frame = frame_;
        }
        rp_recurring_ &= t->rt_recurring;
    }
    std::copy(slots, slots + n, rp_slots_);
    for (uint32_t i = 0; i < n; ++i) rp_ranges_[i] = target_specs[i].range;
    std::copy_n(clear_aspects, n, rp_written_);
    storage_since_barrier_ = false;  // lo de antes del pase ya va ordenado por su dependencia externa
    if (frame_ == trace_frame_) {  // DEPURACIÓN: cargas (L/C) y guardados (s/n) de cada adjunto
        std::string a;
        for (uint32_t i = 0; i < n; i++) {
            const auto& actual = fallback ? defaults : ops;
            auto log_ops = [&](std::string label, VkAttachmentLoadOp load, VkAttachmentStoreOp store) {
                a += " " + label + ":" + (load == VK_ATTACHMENT_LOAD_OP_CLEAR ? "C" : load == VK_ATTACHMENT_LOAD_OP_NONE_EXT ? "n" : "L") +
                     (store == VK_ATTACHMENT_STORE_OP_STORE ? "s" : "n");
            };
            if (slots[i] >= 0) log_ops(std::to_string(slots[i]), actual.color_load[slots[i]], actual.color_store[slots[i]]);
            if (targets[i]->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) log_ops("d", actual.depth_load, actual.depth_store);
            if (targets[i]->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) log_ops("s", actual.stencil_load, actual.stencil_store);
        }
        NVN_LOG("F pase %ux%u área %d,%d %ux%u adjuntos%s\n", width, height, area.offset.x, area.offset.y, area.extent.width,
                area.extent.height, a.c_str());
    }
    return true;
}

void Executor::end_render_pass() {
    if (!in_render_pass_) {
        split_why_ = nullptr;  // lo dijo alguien que no cerró nada: no vale para el siguiente
        return;
    }
    // Perfil de cortes: quién lo cierra (el que lo dijo o el comando del juego en curso) y con qué adjuntos.
    last_end_why_ = split_why_                 ? split_why_
                    : run_lists_ && cur_command_ ? kCommandNames[cur_command_->index()]
                                                 : "fin del envío";
    split_why_ = nullptr;
    last_end_fb_ = cur_fb_;
    last_end_rt_ = cur_pass_rt_;
    last_end_extent_ = rp_extent_;
    last_end_samples_ = rp_samples_;
    vkCmdEndRenderPass(cmd_);
    // The next pass may be in another submit. Attribute transfers and queue gaps
    // after this boundary separately from work on the attachments.
    if (prof_frame_ == frame_ && !prof_closed_) prof_gpu_mark("outside", "fuera de pase");
    in_render_pass_ = false;
    work_after_pass_ = false;  // lo grabado hasta aquí, ordenado con lo que venga por la dependencia externa del pase
    if (crumbs_passes_) crumb_end();
    if (pending_barrier_) {  // la barrera de NVN aplazada (nvn_barrier)
        pending_barrier_ = false;
        full_barrier();
    }
}

void Executor::full_barrier() {
    storage_since_barrier_ = false;
    // Perfil y tight_barriers (ver gpu_work_): sin nada grabado desde la última, o (_pase) desde el final de un pase.
    prof_full_barriers_++;
    const bool redundant = !gpu_work_, after_pass = gpu_work_ && !work_after_pass_;
    prof_barriers_redundant_ += redundant;
    prof_barriers_after_pass_ += after_pass;
    if ((redundant && skip_redundant_barriers_) || (after_pass && pass_end_is_barrier_)) {
        prof_barriers_skipped_++;
        gpu_work_ = work_after_pass_ = false;
        return;
    }
    gpu_work_ = work_after_pass_ = false;
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0,
                         nullptr, 0, nullptr);
}

void Executor::barrier() {
    pending_barrier_ = false;  // esta la cubre
    if (!split_why_) split_why_ = "barrera";
    end_render_pass();
    targets_changed_ = true;
    full_barrier();
}

// Barrera de NVN. Todo lo de antes del pase abierto ya va ordenado con él por su dependencia externa: dentro del pase
// solo importan sus propios draws. Si en el pase ya se escribieron storage buffers o imágenes desde la última barrera,
// lo que venga puede leerlos: se corta como siempre. Si no, se aplaza al final del pase (end_render_pass); draw() lo
// corta antes si un draw muestrea un adjunto del pase o escribe storage.
void Executor::nvn_barrier(int bits) {
    const bool lazy = opt_.barrera_perezosa && in_render_pass_ && !targets_changed_ && !storage_since_barrier_;
    if (frame_ == trace_frame_) NVN_LOG("F barrera 0x%x%s\n", bits, lazy ? " aplazada" : "");
    if (!lazy) {
        // Perfil de cortes: por qué no se aplazó (si hay pase abierto). rt_dirty: el juego ya cambió de targets.
        if (in_render_pass_)
            split_why_ = !opt_.barrera_perezosa ? "barrera (no_lazy_barrier)"
                         : targets_changed_         ? "barrera con targets nuevos"
                                                : "barrera tras escribir storage en el pase";
        barrier();
        return;
    }
    pending_barrier_ = true;
    prof_barriers_deferred_++;
}

// --- Borrados y copias ---

// Resolve the framebuffer dimensions for a pending clear without claiming sibling attachments.
void Executor::touch_targets() {
    auto touch = [&](Texture* t) {
        if (!t) return;
        ensure_image(*t, cmd_);
        if (std::find(frame_targets_.begin(), frame_targets_.end(), t) == frame_targets_.end()) frame_targets_.push_back(t);
    };
    for (int k = 0; k < st_.rt.count; k++) touch(st_.rt.colors[k]);
    touch(st_.rt.depth);
}

// Tamaño del framebuffer de los targets enlazados (como en ensure_render_pass: el menor de todos).
VkExtent2D Executor::target_extent(const cmd::SetRenderTargets& targets) {
    uint32_t width = UINT32_MAX, height = UINT32_MAX;
    auto add = [&](const Texture* t, const TextureViewDesc& v) {
        if (!t) return;
        uint32_t level = (v.flags & kViewLevels) ? v.base_level : 0;
        const auto extent = mip_extent(*t, level);
        width = std::min(width, extent.width);
        height = std::min(height, extent.height);
    };
    for (int k = 0; k < targets.count; k++) add(targets.colors[k], targets.color_views[k]);
    add(targets.depth, targets.depth_view);
    return {width, height};
}

// Un borrado de todo el adjunto antes de que empiece el pase de estos targets: se apunta para el siguiente pase que lo
// use (loadOp CLEAR) en vez de abrir uno solo para borrar. false si no se puede (se borra como siempre): textura sin
// imagen, multisample, vista con otro formato o de otro tipo, el adjunto no es todo el framebuffer o scissor parcial.
bool Executor::defer_clear(Texture* t, const TextureViewDesc& v, VkImageAspectFlags aspects, const VkClearValue& value) {
    split_why_ = "borrado de otros targets";
    end_render_pass();  // el de antes, de otros targets: se iba a cerrar en el siguiente draw
    touch_targets();
    if (!t->image || t->desc.samples > 1) return false;
    const int target = (v.flags & kViewTarget) ? v.target : t->desc.target;
    if (target != kTarget2D && target != kTarget2DArray && target != kTargetRectangle) return false;
    const auto spec = describe_view(*t, v, ViewUse::Attachment);
    if (!spec.range.aspectMask || spec.format != t->vk_format) return false;
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(vk()->gpu, t->vk_format, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_DST_BIT)) return false;  // para vkCmdClear*Image
    const VkExtent2D fb = target_extent(st_.rt);
    const auto extent = mip_extent(*t, spec.range.baseMipLevel);
    if (extent.width != fb.width || extent.height != fb.height) return false;
    if (st_.scissor_set && (st_.scissor.offset.x > 0 || st_.scissor.offset.y > 0 ||
                            int64_t{st_.scissor.offset.x} + st_.scissor.extent.width < fb.width ||
                            int64_t{st_.scissor.offset.y} + st_.scissor.extent.height < fb.height))
        return false;
    // The delayed clear replaces only this view. Retain guest texels outside it.
    const bool entire = t->layers == 1 && t->desc.levels == 1 && aspects == t->aspect;
    if (!prepare_contents(*t, entire ? PriorContents::Replace : PriorContents::Preserve)) return false;
    for (PendingClear& p : pending_clears_) {
        if (p.texture != t || p.range.baseMipLevel != spec.range.baseMipLevel ||
            p.range.baseArrayLayer != spec.range.baseArrayLayer) continue;
        // Otro sobre lo mismo: manda el último (de profundidad+stencil, por aspecto).
        if (aspects & VK_IMAGE_ASPECT_COLOR_BIT) p.value.color = value.color;
        if (aspects & VK_IMAGE_ASPECT_DEPTH_BIT) p.value.depthStencil.depth = value.depthStencil.depth;
        if (aspects & VK_IMAGE_ASPECT_STENCIL_BIT) p.value.depthStencil.stencil = value.depthStencil.stencil;
        p.range.aspectMask |= aspects;
        record_texture_write(*t, spec.range, "deferred clear");
        return true;
    }
    auto range = spec.range;
    range.aspectMask = aspects;
    pending_clears_.push_back({t, range, value});
    record_texture_write(*t, range, "deferred clear");
    return true;
}

// Hace ya los borrados diferidos de esa textura (de todas si es nula), fuera del render pass: la van a leer sin
// empezar un pase con ella (muestrearla, copiarla, presentarla, volcarla) o se acaba el envío.
void Executor::flush_clears(const Texture* only, const VkImageSubresourceRange* needed) {
    if (pending_clears_.empty()) return;
    size_t keep = 0;
    for (size_t i = 0; i < pending_clears_.size(); i++) {
        const PendingClear p = pending_clears_[i];
        if ((only && p.texture != only) || (needed && !ranges_overlap(p.range, *needed))) {
            pending_clears_[keep++] = p;
            continue;
        }
        if (!p.texture->image) continue;
        if (in_render_pass_) {
            split_why_ = "borrado diferido hecho aparte";
            end_render_pass();
            targets_changed_ = true;
        }
        const auto& range = p.range;
        note_gpu_work();
        // Packed depth/stencil uses a single layout on Vulkan 1.1.
        auto hazard = range;
        hazard.aspectMask = p.texture->aspect;
        sync_image(cmd_, p.texture->image, hazard, prior_writes, transfer_write);
        if (range.aspectMask & VK_IMAGE_ASPECT_COLOR_BIT)
            vkCmdClearColorImage(cmd_, p.texture->image, VK_IMAGE_LAYOUT_GENERAL, &p.value.color, 1, &range);
        else
            vkCmdClearDepthStencilImage(cmd_, p.texture->image, VK_IMAGE_LAYOUT_GENERAL, &p.value.depthStencil, 1, &range);
        sync_image(cmd_, p.texture->image, hazard, transfer_write, later_access);
        prof_clear_flushes_++;
    }
    pending_clears_.resize(keep);
}

void Executor::clear_attachment(int slot, const VkClearAttachment& clear, const VkRect2D& rect) {
    if (!rect.extent.width || !rect.extent.height) return;
    for (uint32_t i = 0; i < rp_target_count_; ++i) {
        if (rp_slots_[i] != slot) continue;
        const VkClearRect region{rect, 0, 1};
        vkCmdClearAttachments(cmd_, 1, &clear, 1, &region);
        rp_written_[i] |= clear.aspectMask;
        record_texture_write(*rp_targets_[i], rp_ranges_[i], "attachment clear");
        return;
    }
}

void Executor::clear_color(const cmd::ClearColor& c) {
    if (c.index < 0 || c.index >= st_.rt.count) return;
    if (frame_ == trace_frame_)
        NVN_LOG("F clear rt%d (%p) %.3f %.3f %.3f %.3f mask 0x%x\n", c.index,
                static_cast<void*>(st_.rt.colors[c.index]), c.color[0], c.color[1],
                c.color[2], c.color[3], c.mask);
    if (!st_.rt.colors[c.index]) return;
    if (loading_log_)
        fprintf(loading_log_, "clear_request after_draw=%llu rt=%d serial=%llu mask=%x rgba=%g,%g,%g,%g\n",
                (unsigned long long)draws_,c.index,(unsigned long long)st_.rt.colors[c.index]->serial,c.mask,
                c.color[0],c.color[1],c.color[2],c.color[3]);
    if (Texture* t = st_.rt.colors[c.index]) {
        memcpy(t->clear_color, c.color, sizeof c.color);
        t->cleared = true;
    }
    VkClearValue value{};
    memcpy(value.color.float32, c.color, sizeof c.color);
    // Todo el target (máscara completa) antes de que empiece su pase: diferido (ver PendingClear).
    if (opt_.clear_diferido && (!in_render_pass_ || targets_changed_) && (c.mask & 0xF) == 0xF &&
        defer_clear(st_.rt.colors[c.index], st_.rt.color_views[c.index], VK_IMAGE_ASPECT_COLOR_BIT, value))
        return;
    if (!ensure_render_pass()) return;
    if (!rp_area_full_) {  // la mirada adelante no lo vio (no debería pasar): un pase con todo el área
        split_why_ = "borrado en un pase con área recortada";
        end_render_pass();
        targets_changed_ = true;
        if (!ensure_render_pass(true)) return;
    }
    VkClearAttachment a{VK_IMAGE_ASPECT_COLOR_BIT, static_cast<uint32_t>(c.index), value};
    clear_attachment(c.index, a, st_.scissor_set ? intersect(st_.scissor, rp_area_) : rp_area_);
}

void Executor::clear_depth_stencil(const cmd::ClearDepthStencil& c) {
    Texture* t = st_.rt.depth;
    if (!t) return;
    const bool deferrable = opt_.clear_diferido && (!in_render_pass_ || targets_changed_);
    if (deferrable) {
        split_why_ = "borrado de otros targets";
        end_render_pass();  // el de antes, de otros targets
        touch_targets();    // para t->aspect (y lo que haría el pase)
        if (!t->image) return;
    } else if (!ensure_render_pass() || !t->image) {
        return;
    }
    VkImageAspectFlags aspect = 0;
    if (c.depth_mask) aspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
    if (c.stencil_mask && (t->aspect & VK_IMAGE_ASPECT_STENCIL_BIT)) aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
    if (!aspect) return;
    VkClearValue value{};
    value.depthStencil = {c.depth, static_cast<uint32_t>(c.stencil)};
    if (deferrable && (aspect & t->aspect) && defer_clear(t, st_.rt.depth_view, aspect & t->aspect, value)) return;
    if (!ensure_render_pass() || !t->image) return;
    if (!rp_area_full_) {  // como en clear_color
        split_why_ = "borrado en un pase con área recortada";
        end_render_pass();
        targets_changed_ = true;
        if (!ensure_render_pass(true)) return;
    }
    VkClearAttachment a{aspect, 0, value};
    clear_attachment(-1, a, st_.scissor_set ? intersect(st_.scissor, rp_area_) : rp_area_);
}

void Executor::copy_texture(const cmd::CopyTexture& c) {
    if (!c.src || !c.dst) return;
    if (c.src == c.dst && c.src_region == c.dst_region && transfer_level(c.src_view) == transfer_level(c.dst_view) &&
        transfer_base_layer(c.src_view) == transfer_base_layer(c.dst_view)) return;
    end_render_pass();
    targets_changed_ = true;
    if (!ensure_image(*c.src, cmd_) || !ensure_image(*c.dst, cmd_)) return;
    TextureCopyPlan plan;
    if (!plan_texture_copy(c, plan) || (c.src == c.dst && plan.source == plan.destination)) return;
    const auto sr = plan.source.range(plan.aspects), dr = plan.destination.range(plan.aspects);
    flush_clears(c.src, &sr);
    flush_clears(c.dst, &dr);
    if (!prepare_contents(*c.src, PriorContents::Preserve, true)) return;
    if (c.dst != c.src && !prepare_contents(*c.dst, plan.replaces_destination ? PriorContents::Replace : PriorContents::Preserve, true)) return;
    Texture* src = c.src == c.dst && ranges_overlap(sr, dr) ? snapshot_texture(*c.src) : c.src.get();
    if (!src) return;
    if (c.dst->desc.target == kTargetCube || c.dst->desc.target == kTargetCubeArray)
        note_cube_write(*c.dst, c.dst_view, "copia", c.src, 0, frame_);
    note_gpu_work();
    plan.record(cmd_, src->image, c.dst->image);
    record_texture_write(*c.dst, dr, "CopyTexture");
    if (loading_log_)
        fprintf(loading_log_, "image_copy after_draw=%llu source=%llu destination=%llu src_mip=%u dst_mip=%u\n",
                (unsigned long long)draws_,(unsigned long long)c.src->serial,(unsigned long long)c.dst->serial,
                plan.source.level,plan.destination.level);
}

// WriteTexels ya dejó los texels en la memoria de la textura. Si la imagen no la ha escrito la GPU, basta con subirla
// entera otra vez (uploaded = false: quien llama la sube enseguida). Si sí (render target, copia, borrado, imagen), se
// suben solo esas cajas, para no pisar lo demás con la memoria del juego, que no tiene lo de la GPU.
bool Executor::apply_texel_writes(Texture& t) {
    if (!g_texel_writes.load(std::memory_order_acquire)) return true;
    std::vector<Texture::TexelWrite> writes;
    {
        std::lock_guard lock(texel_writes_mutex());
        if (t.texel_writes.empty()) return true;
        writes.swap(t.texel_writes);
        g_texel_writes -= static_cast<uint32_t>(writes.size());
    }
    // Sus cajas ya dicen qué cambia: sin vigilancia (cada WriteTexels saltaría en cada página; un frame de vídeo, en todas).
    if (!t.gpu_levels) t.no_watch = true, t.watched = false;
    if (!t.image || !t.storage) return true;  // la subida al crearla ya lo lleva
    if (!t.gpu_levels) {
        t.uploaded = false;
        return true;
    }
    if (patch_image(t, writes)) return true;
    std::lock_guard lock(texel_writes_mutex());
    t.texel_writes.insert(t.texel_writes.begin(), writes.begin(), writes.end());
    g_texel_writes += static_cast<uint32_t>(writes.size());
    return false;
}

bool Executor::patch_image(Texture& t, const std::vector<Texture::TexelWrite>& writes) {
    if (writes.empty()) return true;
    split_why_ = "escrituras de la CPU en una textura (WriteTexels)";
    end_render_pass();
    targets_changed_ = true;
    flush_clears(&t);  // un borrado pendiente es de antes que estas escrituras
    ++t.content_version;
    for (const auto& w : writes) {
        const VkImageSubresourceRange range{t.aspect, w.level, 1, w.layer, w.layers};
        note_gpu_work();
        sync_image(cmd_, t.image, range, later_access, transfer_write);
        const bool copied = copy_texel_box(t, w);
        sync_image(cmd_, t.image, range, transfer_write, later_access);
        if (!copied) return false;
    }
    return true;
}

bool Executor::copy_texel_box(Texture& t, const Texture::TexelWrite& w) {
    const TextureLayout& L = t.layout;
    const uint32_t bpb = L.bytes_per_block;
    VkImageAspectFlags aspect = t.aspect & ~VK_IMAGE_ASPECT_STENCIL_BIT;
    if (!aspect) aspect = VK_IMAGE_ASPECT_STENCIL_BIT;
    const LevelLayout& lv = L.levels[w.level];
    const size_t row = size_t{w.box.width} * bpb, bytes = row * w.box.height * w.box.depth;
    for (uint32_t k = 0; k < w.layers && w.layer + k < std::min(L.layers, t.layers); k++) {
        // Descomprimida en la CPU (t.bcn): los bloques a una memoria de paso y, de ahí, los texels al staging
        static thread_local std::vector<uint8_t> bc_scratch;
        Staging::Alloc a{};
        if (t.bcn) {
            bc_scratch.resize(bytes);
            a.data = bc_scratch.data();
        } else {
            a = staging_->alloc(bytes, 16);
            if (!a.data) return false;
        }
        const uint8_t* level = t.storage + size_t{w.layer + k} * L.layer_size + lv.offset;
        if (L.linear) {
            for (uint32_t y = 0; y < w.box.height; y++)
                memcpy(a.data + y * row, level + size_t{w.box.y + y} * L.stride + size_t{w.box.x} * bpb, row);
        } else {
            deswizzle_region(lv, bpb, w.box, level, a.data, row, row * w.box.height);
        }
        if (t.astc) {  // the box's blocks decoded and encoded again (no cache: these are small and change)
            const bcn::Kind kind = static_cast<bcn::Kind>(t.bcn);
            const uint32_t tw = w.box.width * 4, th = w.box.height * 4;
            const uint32_t x0 = w.box.x * 4, y0 = w.box.y * 4;
            const uint32_t cw = std::min(tw, lv.width - std::min(lv.width, x0)), ch = std::min(th, lv.height - std::min(lv.height, y0));
            if (!cw || !ch) continue;
            static thread_local std::vector<uint8_t> texels;
            texels.resize(size_t{tw} * th * 4);
            for (uint32_t z = 0; z < std::max(w.box.depth, 1u); z++) {
                Staging::Alloc d = staging_->alloc(size_t{w.box.width} * w.box.height * 16, 16);
                if (!d.data) return false;
                bcn::decode(kind, a.data + size_t{z} * row * w.box.height, w.box.width, w.box.height, texels.data(),
                            size_t{tw} * 4, tw, th);
                if (!astc::encode(texels.data(), tw, th, t.vk_format == VK_FORMAT_ASTC_4x4_SRGB_BLOCK, d.data)) return false;
                VkBufferImageCopy r{};
                r.bufferOffset = d.offset;
                r.bufferRowLength = tw;
                r.bufferImageHeight = th;
                r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, w.level, w.layer + k, 1};
                r.imageOffset = {static_cast<int32_t>(x0), static_cast<int32_t>(y0), static_cast<int32_t>(w.box.z + z)};
                r.imageExtent = {cw, ch, 1};
                note_gpu_work();
                vkCmdCopyBufferToImage(cmd_, d.buffer, t.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
            }
            continue;
        }
        if (t.bcn) {
            const bcn::Kind kind = static_cast<bcn::Kind>(t.bcn);
            const uint32_t tw = w.box.width * 4, th = w.box.height * 4, tb = bcn::texel_bytes(kind);
            const uint32_t x0 = w.box.x * 4, y0 = w.box.y * 4;
            const uint32_t cw = std::min(tw, lv.width - std::min(lv.width, x0)), ch = std::min(th, lv.height - std::min(lv.height, y0));
            if (!cw || !ch) continue;
            for (uint32_t z = 0; z < std::max(w.box.depth, 1u); z++) {
                Staging::Alloc d = staging_->alloc(size_t{cw} * ch * tb, 16);
                if (!d.data) return false;
                bcn::decode(kind, a.data + size_t{z} * row * w.box.height, w.box.width, w.box.height, d.data, size_t{cw} * tb,
                            cw, ch);
                VkBufferImageCopy r{};
                r.bufferOffset = d.offset;
                r.bufferRowLength = cw;
                r.bufferImageHeight = ch;
                r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, w.level, w.layer + k, 1};
                r.imageOffset = {static_cast<int32_t>(x0), static_cast<int32_t>(y0), static_cast<int32_t>(w.box.z + z)};
                r.imageExtent = {cw, ch, 1};
                note_gpu_work();
                vkCmdCopyBufferToImage(cmd_, d.buffer, t.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
            }
            continue;
        }
        VkBufferImageCopy r{};
        r.bufferOffset = a.offset;
        r.bufferRowLength = w.box.width * L.block_w;
        r.bufferImageHeight = w.box.height * L.block_h;
        r.imageSubresource = {aspect, w.level, w.layer + k, 1};
        r.imageOffset = {static_cast<int32_t>(w.box.x * L.block_w), static_cast<int32_t>(w.box.y * L.block_h),
                         static_cast<int32_t>(w.box.z)};
        // Hasta el borde del nivel como mucho (los bloques comprimidos del borde pueden pasarse en texels).
        r.imageExtent = {std::min(w.box.width * L.block_w, lv.width - std::min(lv.width, w.box.x * L.block_w)),
                         std::min(w.box.height * L.block_h, lv.height - std::min(lv.height, w.box.y * L.block_h)),
                         w.box.depth};
        note_gpu_work();
        if (r.imageExtent.width && r.imageExtent.height)
            vkCmdCopyBufferToImage(cmd_, a.buffer, t.image, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    }
    return true;
}

// nvnCommandBufferClearTexture{,i,ui}: una región de un nivel (y unas capas) a un color. Toda la región que cubre el
// nivel: vkCmdClearColorImage. Una parte: un pase con esa textura de único adjunto y vkCmdClearAttachments en el
// rectángulo (las 3D, las comprimidas y los formatos que no pueden ser adjunto no: se borra el nivel entero, con aviso;
// las comprimidas, nada, que vkCmdClearColorImage no las admite). La máscara de canales no se
// puede respetar con ninguno de los dos: se borra entero (con aviso).
void Executor::clear_texture(const cmd::ClearTexture& c) {
    Texture* t = c.texture;
    if (!t || !(c.mask & 0xF)) return;
    end_render_pass();
    targets_changed_ = true;
    if (!ensure_image(*t, cmd_) || !(t->aspect & VK_IMAGE_ASPECT_COLOR_BIT)) return;
    flush_clears(t);
    const TextureViewDesc& v = c.view;
    const uint32_t level = (v.flags & kViewLevels) ? static_cast<uint32_t>(v.base_level) : 0u;
    if (level >= static_cast<uint32_t>(std::max(t->desc.levels, 1))) return;
    const int target = t->desc.target;
    const bool layered = target == kTarget2DArray || target == kTarget2DMultisampleArray || target == kTargetCube ||
                         target == kTargetCubeArray;
    const bool is_3d = target == kTarget3D;
    const uint32_t lw = std::max(t->desc.width >> level, 1), lh = std::max(t->desc.height >> level, 1);
    const uint32_t ld = is_3d ? std::max(t->desc.depth >> level, 1) : 1;
    uint32_t layer = (v.flags & kViewLayers) ? static_cast<uint32_t>(v.base_layer) : 0u, layers = 1;
    if (layered) layer += static_cast<uint32_t>(std::max(c.region.z, 0)), layers = static_cast<uint32_t>(std::max(c.region.depth, 1));
    if (layer >= t->layers) return;
    layers = std::min(layers, t->layers - layer);
    const int64_t x0 = std::max(c.region.x, 0), y0 = std::max(c.region.y, 0);
    const int64_t x1 = std::min<int64_t>(int64_t{c.region.x} + c.region.width, lw), y1 = std::min<int64_t>(int64_t{c.region.y} + c.region.height, lh);
    if (x1 <= x0 || y1 <= y0) return;
    const bool whole = x0 == 0 && y0 == 0 && x1 == lw && y1 == lh &&
                       (!is_3d || (c.region.z <= 0 && int64_t{c.region.z} + std::max(c.region.depth, 1) >= ld));
    const FormatInfo* f = format_info(t->desc.format);
    const bool compressed = f && f->block > 1;
    if (compressed) return;
    const bool replaces_all = whole && level == 0 && t->desc.levels == 1 && layers == t->layers;
    if (!prepare_contents(*t, replaces_all ? PriorContents::Replace : PriorContents::Preserve)) return;
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(vk()->gpu, t->vk_format, &fp);
    const bool attachable = (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) != 0;
    static int logged = 0;
    if ((c.mask & 0xF) != 0xF && logged++ < 4) NVN_LOG("ClearTexture con máscara 0x%x: se borran todos los canales\n", c.mask);
    VkClearValue value{};
    memcpy(value.color.uint32, c.color, sizeof c.color);
    if (whole || is_3d || !attachable) {
        if (!whole && logged++ < 4)
            NVN_LOG("ClearTexture de una parte de una textura 3D, comprimida o que no es adjunto (formato 0x%x): se borra el nivel "
                    "entero\n", t->desc.format);
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, level, 1, layer, layers};
        note_gpu_work();
        sync_image(cmd_, t->image, range, prior_writes, transfer_write);
        vkCmdClearColorImage(cmd_, t->image, VK_IMAGE_LAYOUT_GENERAL, &value.color, 1, &range);
        record_texture_write(*t, range, "ClearTexture");
        sync_image(cmd_, t->image, range, transfer_write, later_access);
        return;
    }
    // Una parte: cada capa como único adjunto (vista 2D de un nivel y una capa). El pase lo abre ensure_render_pass con
    // los targets cambiados un momento; sin STORE_OP_NONE, que la mirada adelante no ve este borrado.
    const cmd::SetRenderTargets saved = st_.rt;
    const bool store_ops = opt_.store_ops;
    opt_.store_ops = false;
    for (uint32_t k = 0; k < layers; k++) {
        cmd::SetRenderTargets rt{};
        rt.count = 1;
        rt.colors[0] = t;
        TextureViewDesc& av = rt.color_views[0];
        av = v;
        av.flags |= kViewLevels | kViewLayers | kViewTarget;
        av.flags &= ~kViewSwizzle;
        av.base_level = static_cast<int16_t>(level), av.levels = 1;
        av.base_layer = static_cast<int32_t>(layer + k), av.layers = 1;
        av.target = kTarget2D;
        st_.rt = rt;
        targets_changed_ = true;
        if (ensure_render_pass(true)) {
            VkClearAttachment a{VK_IMAGE_ASPECT_COLOR_BIT, 0, value};
            clear_attachment(0, a, {{static_cast<int32_t>(x0), static_cast<int32_t>(y0)},
                                    {static_cast<uint32_t>(x1 - x0), static_cast<uint32_t>(y1 - y0)}});
        }
        end_render_pass();
    }
    opt_.store_ops = store_ops;
    st_.rt = saved;
    targets_changed_ = true;
}

// --- Draws ---

bool Executor::resolve_texture(Stage s, const Resource& r, Texture*& tex, TextureViewDesc& v, Sampler*& smp) {
    tex = nullptr;
    smp = nullptr;
    v = {};
    // Handle en un uniform buffer del juego (bindless de verdad): resolve_bindless.
    if (r.cbuf_index >= kUniformBankBase) return resolve_bindless(s, r, tex, v, smp);
    // Tabla de handles: c2 en gráficos; en compute c0 (los TEX con índice inmediato, ver texture_bound_buffer) y c2
    // (los TEX.B de MH Rise cargan de c2[0x168 + 8·i]: la iluminación de efectos 314939, el volumen de niebla 4x2x64…;
    // antes la textura negra de reserva). Misma tabla, mismo índice.
    const bool bank_ok = s == Stage::Compute ? r.cbuf_index == kComputeDriverBank ||
                                                   (r.cbuf_index == kTextureHandleBank && !NVN_FLAG("no_c2_compute"))
                                             : r.cbuf_index == kTextureHandleBank;
    // Tabla baja de c2 (0x20 + 8·i, hasta las imágenes en 0x120): la de las texturas combinadas (BindTexture; ver
    // StageBindings::combined_textures). 9 pixel shaders de la interfaz (de 1842 en la caché de SPIR-V) leen ahí el handle
    // entero de sus ranuras 0 y 1: los iconos de la tabla de recompensas al acabar una misión (148e89d349436450, 5-11).
    // Antes no se resolvían (la de reserva, transparente). <datos>/no_low_table (en vivo): como antes.
    const bool low_table = low_table_ && s != Stage::Compute && bank_ok && r.cbuf_offset >= kLowTextureHandleBase &&
                           r.cbuf_offset < kComputeImageHandleBase && (r.cbuf_offset - kLowTextureHandleBase) % 8 == 0;
    if (!bank_ok || (r.cbuf_offset < kTextureHandleBase && !low_table)) return false;
    static int logged31 = 0;  // DEPURACIÓN: ranuras altas
    if (!low_table && (r.cbuf_offset - kTextureHandleBase) / 8 >= 24 && logged31++ < 10)
        NVN_LOG("textura en c%u[0x%x] (%s), ranura %u: %s\n", r.cbuf_index, r.cbuf_offset, nvnshader::stage_name(s),
                (r.cbuf_offset - kTextureHandleBase) / 8,
                st_.bindings[static_cast<int>(s)].textures[std::min<uint32_t>((r.cbuf_offset - kTextureHandleBase) / 8, kMaxTextures - 1)] ? "enlazada" : "vacía");
    uint32_t index = (r.cbuf_offset - (low_table ? kLowTextureHandleBase : kTextureHandleBase)) / 8;
    if (index >= kMaxTextures) return false;
    const StageBindings& b = st_.bindings[static_cast<int>(s)];
    // La tabla de cada sitio (StageBindings::combined_textures); si en la suya no hay nada, la otra, como cuando eran una.
    const uint64_t separate = b.textures[index], combined_handle = separate_tables_ ? b.combined_textures[index] : 0;
    const bool from_combined = low_table ? combined_handle != 0 : !separate && combined_handle;
    uint64_t handle = from_combined ? combined_handle : separate;
    const bool combined = from_combined || b.combined[index];
    if (low_table) {  // qué sale de ahí (para comprobar en la Flip que son los iconos), unas pocas veces
        static int logged_low = 0;
        if (logged_low++ < 24) {
            const uint32_t id = handle_texture_id(handle);
            const Texture* t = st_.texture_pool ? st_.texture_pool->texture_at(id).texture : nullptr;
            NVN_LOG("tabla baja c2[0x%x] (%s %016llx): ranura %u, handle 0x%llx (%s) -> textura %u %s %dx%d formato 0x%x\n",
                    r.cbuf_offset, nvnshader::stage_name(s),
                    static_cast<unsigned long long>(st_.stages[static_cast<int>(s)] ? st_.stages[static_cast<int>(s)]->print : 0),
                    index, static_cast<unsigned long long>(handle), combined ? "combinada" : "separada", id,
                    t ? "ok" : "NO HAY", t ? t->desc.width : 0, t ? t->desc.height : 0, t ? t->desc.format : 0);
        }
    }
    uint32_t texture_id = handle_texture_id(handle);
    // DEPURACIÓN: NIER_TEX_SUST="a:b,c:d" usa la textura b donde se pide a. Se lee una vez: con la variable
    // vacía, cada textura de cada draw construía dos strings.
    static const std::vector<std::pair<uint32_t, uint32_t>> substitutions = [] {
        std::vector<std::pair<uint32_t, uint32_t>> out;
        const char* list = getenv("NIER_TEX_SUST");
        for (const char* q = list; q && *q;) {
            unsigned from, to;
            if (sscanf(q, "%u:%u", &from, &to) == 2) out.emplace_back(from, to);
            q = strchr(q, ',');
            if (q) q++;
        }
        return out;
    }();
    for (auto [from, to] : substitutions)
        if (from == texture_id) {
            texture_id = to;
            break;
        }
    // Textura y sampler separados: el sampler va por su propia ranura (c2[0x568 + 8·j]), que no tiene por
    // qué coincidir con la de la textura.
    uint32_t sampler_slot = index;
    if (r.has_secondary && r.secondary_cbuf_offset >= kSamplerHandleBase)
        sampler_slot = std::min<uint32_t>((r.secondary_cbuf_offset - kSamplerHandleBase) / 8, kMaxTextures - 1);
    uint32_t sampler_id = combined ? handle_sampler_id(handle) : static_cast<uint32_t>(b.samplers[sampler_slot]);
    // El shader ORea un sampler fijo al handle que lee de c2 (TEX.B con handle | 0x10c00000: el 268 de comparación de las
    // sombras, el 267 del volumen de sondas y del cubemap…; Eden lo perdía, ver handle_or). La GPU usa ese (los 12 bits de
    // arriba del handle) y no el de la ranura, que el juego ni enlaza: salía el sampler 0, sin comparación ni mipmaps.
    // En el censo, 158 de 202 pixel shaders; los 111 que leen sombras. <datos>/no_fixed_sampler: como antes.
    const bool fixed = fixed_samplers_ && (r.handle_or >> 20) != 0;
    if (fixed) sampler_id = (combined ? handle_sampler_id(handle) : 0) | (r.handle_or >> 20);
    if (st_.texture_pool) {
        const auto e = st_.texture_pool->texture_at(texture_id);
        tex = e.texture;
        v = e.view;
    }
    if (st_.sampler_pool) smp = st_.sampler_pool->sampler_at(sampler_id);
    if (fixed) log_sampler_once(sampler_id, smp, "fijo del shader");
    // DEPURACIÓN: qué texturas muestrean los primeros draws.
    static int logged = 0;
    if (logged < 16 || (frame_ == trace_frame_ && traced_draws_ < trace_cap())) {
        logged++;
        std::string hex;
        char buf[4];
        if (tex && tex->storage)
            for (int k = 0; k < 32; k++) snprintf(buf, sizeof buf, "%02x", tex->storage[k]), hex += buf;
        if (tex && tex->storage && frame_ == trace_frame_) {  // estadística de toda la memoria de la textura
            size_t n = size_t(tex->desc.width) * tex->desc.height, nz = 0, sum = 0;
            uint8_t mx = 0;
            for (size_t k = 0; k < n; k++) nz += tex->storage[k] != 0, sum += tex->storage[k], mx = std::max(mx, tex->storage[k]);
            hex += " | fin " + std::to_string(reinterpret_cast<uintptr_t>(tex->storage + n)) + " no-cero " + std::to_string(nz) + "/" + std::to_string(n) + " media " + std::to_string(n ? sum / n : 0) +
                   " max " + std::to_string(mx);
        }
        const char* to_dump = frame_ == trace_frame_ ? debug_var("NIER_VOLCAR_TEX", trace_frame_) : nullptr;  // fichero o entorno
        if (tex && to_dump && *to_dump &&  // DEPURACIÓN: volcar esas texturas
            (std::string(",") + to_dump + ",").find("," + std::to_string(texture_id) + ",") != std::string::npos &&
            std::find(tex_dumps_.begin(), tex_dumps_.end(), std::make_pair(tex, texture_id)) == tex_dumps_.end())
            tex_dumps_.emplace_back(tex, texture_id);
        NVN_LOG("  textura id %u (handle 0x%llx, sampler %u): %s %dx%dx%d target %d niveles %d capas %u formato 0x%x flags 0x%x storage %p subida %d gpu %d %s\n",
                texture_id, static_cast<unsigned long long>(handle), sampler_id, tex ? "ok" : "NO HAY",
                tex ? tex->desc.width : 0, tex ? tex->desc.height : 0, tex ? tex->desc.depth : 0, tex ? tex->desc.target : 0,
                tex ? tex->desc.levels : 0, tex ? tex->layers : 0, tex ? tex->desc.format : 0,
                tex ? tex->desc.flags : 0, tex ? static_cast<void*>(tex->storage) : nullptr, tex ? tex->uploaded : 0,
                tex && tex->gpu_levels, hex.c_str());
        if (smp && frame_ == trace_frame_) {  // DEPURACIÓN: cada sampler de la traza, una vez (borde, comparación, reducción)
            static uint64_t samplers_frame = ~0ull;
            static std::set<uint32_t> samplers_logged;
            if (samplers_frame != trace_frame_) samplers_frame = trace_frame_, samplers_logged.clear();
            if (samplers_logged.insert(sampler_id).second) {
                const SamplerDesc& sd = smp->desc;
                NVN_LOG("  sampler %u: filtros %d/%d wrap %d/%d/%d compare %d func %d reducción %d aniso %.0f lod %.2f..%.2f "
                        "bias %.2f borde %.3g %.3g %.3g %.3g\n",
                        sampler_id, sd.min_filter, sd.mag_filter, sd.wrap[0], sd.wrap[1], sd.wrap[2], sd.compare_mode,
                        sd.compare_func, sd.reduction, sd.max_anisotropy, sd.lod_min, sd.lod_max, sd.lod_bias, sd.border[0],
                        sd.border[1], sd.border[2], sd.border[3]);
            }
        }
    }
    return tex != nullptr;
}

// Bindless de verdad: el shader carga el handle de un uniform buffer del juego (banco c3 + i). Aquí los de un solo
// descriptor (c5[0xa0] en 84 pixel shaders: 0xffffffff, "sin textura", en todo lo visto); los arrays con índice dinámico
// (sondas, decals) van al set 1 (bindless_set). La palabra se lee como la GPU: resolve_handle. Siempre se registran las
// dos palabras que hay (las primeras 48 posiciones y todas en la traza). <datos>/no_bindless: la de reserva, como antes.
bool Executor::resolve_bindless(Stage s, const Resource& r, Texture*& tex, TextureViewDesc& v, Sampler*& smp) {
    const StageBindings& b = st_.bindings[static_cast<int>(s)];
    auto read_u32 = [&](uint32_t bank, uint32_t offset, uint32_t* out) {
        const int index = static_cast<int>(bank) - static_cast<int>(kUniformBankBase);
        if (index < 0 || index >= kMaxUniforms) return false;
        const BufferRange& u = b.uniforms[index];
        if (!u.address || uint64_t{offset} + 4 > u.size) return false;
        memcpy(out, reinterpret_cast<const void*>(u.address + offset), 4);
        return true;
    };
    uint32_t w0 = 0, w1 = 0;
    const bool have0 = read_u32(r.cbuf_index, r.cbuf_offset, &w0), have1 = read_u32(r.cbuf_index, r.cbuf_offset + 4, &w1);
    {  // DEPURACIÓN: qué hay ahí (una vez por sitio, y en la traza)
        static std::set<std::tuple<int, uint32_t, uint32_t>> seen;
        const bool fresh = seen.size() < 48 && seen.insert({static_cast<int>(s), r.cbuf_index, r.cbuf_offset}).second;
        if (fresh || frame_ == trace_frame_)
            NVN_LOG("%sbindless %s c%u[0x%x] tipo %u x%u (& 0x%x | 0x%x): palabras %s%08x %s%08x\n", frame_ == trace_frame_ ? "F " : "",
                    nvnshader::stage_name(s), r.cbuf_index, r.cbuf_offset, r.texture_type, r.count, r.handle_and_mask,
                    r.handle_or, have0 ? "" : "(no) ", w0, have1 ? "" : "(no) ", w1);
    }
    if (!have0 || r.count > 1 || !bindless_) return false;
    uint32_t word = w0;
    if (r.has_secondary) {  // sampler separado en otro sitio: en c2 nuestro handle de sampler es el id (la GPU lo ve << 20)
        uint32_t other = 0;
        if (r.secondary_cbuf_index == kTextureHandleBank && r.secondary_cbuf_offset >= kSamplerHandleBase)
            word |= static_cast<uint32_t>(
                        b.samplers[std::min<uint32_t>((r.secondary_cbuf_offset - kSamplerHandleBase) / 8, kMaxTextures - 1)])
                    << 20;
        else if (read_u32(r.secondary_cbuf_index, r.secondary_cbuf_offset, &other))
            word |= r.secondary_shift_left < 32 ? other << r.secondary_shift_left : 0;
    }
    const bool ok = resolve_handle(word, r, tex, v, smp);
    static int logged = 0;  // DEPURACIÓN
    if (logged++ < 24 || frame_ == trace_frame_)
        NVN_LOG("%sbindless %s c%u[0x%x]: palabra %08x -> %s\n", frame_ == trace_frame_ ? "F " : "", nvnshader::stage_name(s),
                r.cbuf_index, r.cbuf_offset, word, ok ? "textura" : "la de reserva");
    return ok;
}

// Handle de Maxwell leído de la memoria del juego: textura en los 20 bits de abajo y sampler en los 12 de arriba (el
// formato de NVN; deko3d: dkMakeTextureHandle = imagen | sampler << 20). El shader puede quedarse con una parte y añadir
// un sampler fijo: (palabra & handle_and_mask) | handle_or. Las sondas de reflexión de MH Rise guardan
// medio float << 16 | índice y el shader hace (v & 0x7fff) | 0x10b00000 (sampler 267); los decals guardan el índice y
// el shader ORea 0x10800000 (264). 0xffffffff es "sin textura": el shader lo comprueba antes de muestrear.
static uint32_t shader_handle(uint32_t word, const Resource& r) {
    uint32_t h = word & r.handle_and_mask;
    if (r.shift_left < 32) h <<= r.shift_left;
    return h | r.handle_or;
}

bool Executor::resolve_handle(uint32_t word, const Resource& r, Texture*& tex, TextureViewDesc& v, Sampler*& smp,
                              const std::vector<TexturePoolEntry>* textures, const std::vector<ResourceRef<Sampler>>* samplers) {
    tex = nullptr;
    smp = nullptr;
    v = {};
    if (word == 0xFFFFFFFFu) return false;
    const uint32_t h = shader_handle(word, r);
    const uint32_t texture_id = h & 0xFFFFF;
    const uint32_t sampler_id = h >> 20;  // sin sampler en la palabra ni en el shader: el 0, como la GPU
    TexturePoolEntry entry{};
    if (textures) {
        if (texture_id < textures->size()) entry = (*textures)[texture_id];
    } else if (st_.texture_pool) entry = st_.texture_pool->texture_at(texture_id);
    tex = entry.texture;
    v = entry.view;
    if (tex && !select_shader_target(*tex, r.texture_type, v)) tex = nullptr;
    if (samplers) {
        if (sampler_id < samplers->size()) smp = (*samplers)[sampler_id];
    } else if (st_.sampler_pool) smp = st_.sampler_pool->sampler_at(sampler_id);
    if (tex && (r.handle_or >> 20)) log_sampler_once(sampler_id, smp, "fijo del shader (bindless)");
    return tex != nullptr;
}

bool Executor::apply_mapping_changes(Texture& t) {
    auto ranges = g_textures.take_mapping_changes(t);
    if (!t.gpu_levels) {
        if (t.uploaded && !remap_matches_image(t, ranges, prof_full_hash_bytes_)) t.uploaded = false;
        return true;
    }
    const uint32_t levels = upload_levels(t), layers = upload_layers(t);
    const size_t slots = size_t{levels} * layers;
    std::vector<Texture::TexelWrite> patches;
    std::vector<size_t> released;
    std::vector<BlockBox> boxes;
    auto issues = t.residency.issues;
    const auto* pool = g_memory_pools.get(t.desc.pool);
    for (uint32_t layer = 0; layer < layers; ++layer) {
        for (uint32_t mip = 0; mip < levels; ++mip) {
            const size_t slot = size_t{layer} * levels + mip;
            const auto& lv = t.layout.levels[mip];
            const size_t start = size_t{layer} * t.layout.layer_size + lv.offset;
            for (auto [first, last] : ranges) {
                if (first >= start + lv.size) break;
                if (last <= start) continue;
                if (!t.owns_level(slot)) {
                    t.uploaded = false;
                    break;
                }
                if (t.desc.samples > 1 || (t.aspect & (t.aspect - 1))) {
                    NVN_LOG("cannot remap GPU texture %llu with samples %d and aspects %#x\n",
                            (unsigned long long)t.serial, t.desc.samples, t.aspect);
                    g_textures.restore_mapping_changes(t, ranges);
                    return false;
                }
                if (first <= start && last >= start + lv.size) {
                    released.push_back(slot);
                    break;
                }
                boxes.clear();
                if (!mapped_boxes(t.layout, lv, std::max(first, start) - start, std::min(last - start, lv.size), boxes)) {
                    NVN_LOG("texture mapping splits a format block in layer %u mip %u\n", layer, mip);
                    g_textures.restore_mapping_changes(t, ranges);
                    return false;
                }
                issues.resize(slots);
                if (pool && pool->covers(size_t(t.desc.offset) + start, lv.size)) issues[slot] &= ~MipResidency::Unmapped;
                else issues[slot] |= MipResidency::Unmapped;
                for (const auto& box : boxes) patches.push_back({mip, layer, 1, box});
            }
        }
    }
    if (!patch_image(t, patches)) {
        g_textures.restore_mapping_changes(t, ranges);
        return false;
    }
    t.residency.publish(levels, std::move(issues));
    if (!released.empty()) {
        flush_clears(&t);
        for (size_t slot : released) t.set_level_owner(slot, slots, false);
        t.uploaded = false;
    }
    return true;
}

bool Executor::prepare_contents(Texture& t, PriorContents prior, bool inspect_guest) {
    if (!t.storage) return true;
    if (__atomic_load_n(&t.mapping_changed, __ATOMIC_ACQUIRE) && !apply_mapping_changes(t)) return false;
    if (!apply_texel_writes(t)) return false;
    if (prior == PriorContents::Replace || t.gpu_levels == size_t{upload_levels(t)} * upload_layers(t)) return true;
    if (inspect_guest || cpu_texture_changed(t)) {
        split_why_ = "subida de textura";
        end_render_pass();
        targets_changed_ = true;
        upload(t);
        if (!t.uploaded) return false;
    }
    return true;
}

bool Executor::prepare_shader_view(Texture& t, const TextureViewDesc& v, ViewUse use) {
    if (!ensure_image(t, cmd_)) return true; // the descriptor uses its typed default
    if (!pending_clears_.empty()) {
        const auto range = view_range(t, v, use);
        flush_clears(&t, &range);
    }
    return prepare_contents(t, PriorContents::Preserve);
}

bool Executor::prepare_shader_image(const ProgramStage& ps, BoundImage& b) {
    if (b.r->kind == Resource::Image) resolve_image(b.s, ps, *b.r, b.tex, b.view);
    else resolve_texture(b.s, *b.r, b.tex, b.view, b.smp);
    if (!b.tex) return true;
    if (!select_shader_target(*b.tex, b.r->texture_type, b.view)) {
        b.tex = nullptr;
        return true;
    }
    const uint64_t old_version = b.tex->content_version;
    if (!prepare_shader_view(*b.tex, b.view, b.r->kind == Resource::Image ? ViewUse::Storage : ViewUse::Sampled)) return false;
    if (b.s != Stage::Compute && b.tex->content_version != old_version &&
        b.tex->vk_format >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && b.tex->vk_format <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK &&
        NVN_FLAG("check_astc")) {
        check_astc(*b.tex);
        begin();
    }
    return true;
}

void Executor::publish_storage_writes(const std::vector<BoundImage>& bindings, const char* reason) {
    for (const BoundImage& b : bindings)
        if (b.storage_target && b.r->is_written)
            record_texture_write(*b.storage_target, view_range(*b.storage_target, b.view, ViewUse::Storage), reason);
}

// A sampled attachment must keep its pre-draw contents while the framebuffer is writable.
// Copy the entire image so that the original sampler's mip, layer, aspect and swizzle remain valid.
Texture* Executor::sample_source(Texture& source, const TextureViewDesc& v, ViewUse use, const cmd::SetRenderTargets& attachments) {
    auto overlaps = [&](Texture* target, const TextureViewDesc& av) {
        return target == &source && ranges_overlap(view_range(source, v, use), view_range(source, av, ViewUse::Attachment));
    };
    if (overlaps(attachments.depth, attachments.depth_view)) return snapshot_texture(source);
    for (int i = 0; i < attachments.count; ++i)
        if (overlaps(attachments.colors[i], attachments.color_views[i])) return snapshot_texture(source);
    return &source;
}

Texture* Executor::snapshot_texture(Texture& source) {
    if (!source.image) return nullptr;
    FeedbackCopy& copy = feedback_copies_[&source];
    Texture& dst = copy.texture;
    if (dst.image && copy.version == source.content_version) return &dst;
    flush_clears(&source); // the snapshot copies every mip and layer
    end_render_pass();
    targets_changed_ = true;
    if (!dst.serial) {
        dst.serial = next_texture_serial();
        dst.desc = source.desc;
    }
    if (!ensure_image(dst, cmd_) || dst.vk_format != source.vk_format) {
        NVN_LOG("attachment snapshot unavailable: texture %llu format %d\n",
                static_cast<unsigned long long>(source.serial), int(source.vk_format));
        return nullptr;
    }
    dst.residency = source.residency;
    VkMemoryBarrier order{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    order.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    order.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    note_gpu_work();
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         1, &order, 0, nullptr, 0, nullptr);
    std::vector<VkImageCopy> regions;
    for (int mip = 0; mip < std::max(source.desc.levels, 1); ++mip) {
        for (const auto aspect : {VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_ASPECT_STENCIL_BIT}) {
            if (!(source.aspect & aspect)) continue;
            VkImageCopy r{};
            r.srcSubresource = r.dstSubresource = {static_cast<VkImageAspectFlags>(aspect), static_cast<uint32_t>(mip), 0, source.layers};
            r.extent = {static_cast<uint32_t>(std::max(source.desc.width >> mip, 1)),
                        static_cast<uint32_t>(std::max(source.desc.height >> mip, 1)),
                        source.desc.target == kTarget3D ? static_cast<uint32_t>(std::max(source.desc.depth >> mip, 1)) : 1u};
            regions.push_back(r);
        }
    }
    vkCmdCopyImage(cmd_, source.image, VK_IMAGE_LAYOUT_GENERAL, dst.image, VK_IMAGE_LAYOUT_GENERAL,
                   static_cast<uint32_t>(regions.size()), regions.data());
    order.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    order.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                         1, &order, 0, nullptr, 0, nullptr);
    copy.version = source.content_version;
    return &dst;
}

VkImageView Executor::dummy_view(uint32_t texture_type, ViewUse use) {
    auto& group = use == ViewUse::Storage ? storage_defaults_ : sampled_defaults_;
    Texture* t = &group.two_d;
    TextureViewDesc v{};
    if (texture_type <= 1) t = &group.one_d;
    if (texture_type == 4) t = &group.three_d;
    if (texture_type == 5 || texture_type == 6) t = &group.cube;
    if (!select_shader_target(*t, texture_type, v)) return VK_NULL_HANDLE;
    return t->image ? view(*t, v, use) : VK_NULL_HANDLE;
}

void Executor::log_sampler_once(uint32_t id, const Sampler* smp, const char* why) {
    static std::vector<bool> logged(4096);  // se llama por textura y draw: sin búsquedas en un set
    static int count = 0;
    if (id >= logged.size() || logged[id] || count >= 48) return;
    logged[id] = true;
    count++;
    if (!smp) {
        NVN_LOG("sampler %u (%s): no está en el pool, el de por defecto\n", id, why);
        return;
    }
    const SamplerDesc& sd = smp->desc;
    NVN_LOG("sampler %u (%s): filtros %d/%d wrap %d/%d/%d compare %d func %d reducción %d aniso %.0f lod %.2f..%.2f bias %.2f "
            "borde %.3g %.3g %.3g %.3g\n",
            id, why, sd.min_filter, sd.mag_filter, sd.wrap[0], sd.wrap[1], sd.wrap[2], sd.compare_mode, sd.compare_func,
            sd.reduction, sd.max_anisotropy, sd.lod_min, sd.lod_max, sd.lod_bias, sd.border[0], sd.border[1], sd.border[2],
            sd.border[3]);
}

// El layout del set 1: un binding por array (los pone nvnshader: etapa · 16 + j), con su count. Se comparte entre todos los
// shaders con la misma firma (en el censo de MH Rise hay dos: las sondas, y las sondas más tres arrays de decals), así el
// set armado para un draw sirve para los demás que leen los mismos handles.
VkDescriptorSetLayout Executor::bindless_layout_locked(const nvnshader::Translated* const* shaders) {
    std::vector<uint64_t> signature;
    for (int s = 0; s < kStages; s++)
        if (shaders[s] && shaders[s]->has_bindless_set)
            for (const Resource& r : shaders[s]->resources)
                if (r.set == nvnshader::kBindlessSet)
                    signature.push_back(uint64_t{r.binding} | uint64_t{r.count} << 16 |
                                        uint64_t{static_cast<uint32_t>(vk_stage(static_cast<Stage>(s)))} << 40);
    if (signature.empty()) return VK_NULL_HANDLE;
    if (auto it = bindless_layouts_.find(signature); it != bindless_layouts_.end()) return it->second;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    std::string counts;
    for (uint64_t e : signature) {
        bindings.push_back({static_cast<uint32_t>(e & 0xFFFF), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            static_cast<uint32_t>(e >> 16 & 0xFFFFFF), static_cast<VkShaderStageFlags>(e >> 40), nullptr});
        counts += " " + std::to_string(e & 0xFFFF) + "x" + std::to_string(e >> 16 & 0xFFFFFF);
    }
    VkDescriptorSetLayoutCreateInfo si{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    si.bindingCount = static_cast<uint32_t>(bindings.size());
    si.pBindings = bindings.data();
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    if (VkResult res = vkCreateDescriptorSetLayout(vk()->device, &si, nullptr, &layout); res != VK_SUCCESS) {
        NVN_LOG("bindless: vkCreateDescriptorSetLayout del set 1 (%s): %s\n", counts.c_str(), vk_result_name(res));
        layout = VK_NULL_HANDLE;
    } else {
        NVN_LOG("bindless: layout del set 1 nuevo (binding x count:%s)\n", counts.c_str());
    }
    return bindless_layouts_[signature] = layout;
}

// Unassigned slots in an immutable pool snapshot all select the same default.
// Keep the sampler independent: a missing image can still have a registered sampler.
// Assigned slots are resolved live, including retirement, residency changes and feedback.
const std::vector<Executor::HandleRun>* Executor::handle_runs(uintptr_t address, uint32_t count, const Resource& r) {
    const uint32_t shift = r.size_shift;
    if (count < 32 || shift >= 16) return nullptr;
    const uint64_t span = (uint64_t{count - 1} << shift) + sizeof(uint32_t);
    if (span > 65536 || !nnsdk::watch::watchable(address, span)) return nullptr;
    auto& table = handle_tables_[((address >> 2) ^ (address >> 12) ^ (r.handle_or >> 20)) % handle_tables_.size()];
    if (table.address != address || table.count != count || table.shift != shift || table.pool_epoch != handle_pool_epoch_ ||
        table.mask != r.handle_and_mask || table.fixed != r.handle_or || table.left != r.shift_left) {
        table.address = address;
        table.count = count;
        table.shift = shift;
        table.pool_epoch = handle_pool_epoch_;
        table.mask = r.handle_and_mask;
        table.fixed = r.handle_or;
        table.left = r.shift_left;
        table.pages.clear();
    }
    handle_changes_.clear();
    nnsdk::watch::collect(address, span, table.pages, &handle_changes_);
    if (!handle_changes_.empty()) {
        table.runs.clear();
        uint64_t previous = ~uint64_t{0};
        for (uint32_t k = 0; k < count; ++k) {
            uint32_t word;
            memcpy(&word, reinterpret_cast<const void*>(address + (size_t{k} << shift)), sizeof(word));
            uint64_t identity = 0;
            if (word != UINT32_MAX) {
                const uint32_t h = shader_handle(word, r), image = h & 0xfffff, sampler = h >> 20;
                if (handle_textures_ && image < handle_textures_->size() && (*handle_textures_)[image].texture.assigned())
                    identity = image + 1;
                if (handle_samplers_ && sampler < handle_samplers_->size() && (*handle_samplers_)[sampler].assigned())
                    identity |= uint64_t{sampler + 1} << 32;
            }
            if (identity != previous) table.runs.push_back({k, identity ? word : UINT32_MAX});
            previous = identity;
        }
    }
    return &table.runs;
}

// El set 1 de este draw o dispatch: cada elemento i de cada array con la textura cuyo handle está en el uniform buffer del
// juego, en cbuf_offset + (i << size_shift), y la de reserva de su tipo donde no hay nada (fuera del buffer, 0xffffffff, un
// índice que no está en el pool o de otro tipo). Cache the actual views and samplers: a pool entry, a mip range or
// an attachment snapshot can change without the uniform buffer moving.
VkDescriptorSet Executor::bindless_set(const ProgramStage* const* stages, VkDescriptorSetLayout known,
                                      const cmd::SetRenderTargets* attachments) {
    VkDescriptorSetLayout layout = known;
    if (!layout) {
        const nvnshader::Translated* shaders[kStages] = {};
        for (int s = 0; s < kStages; s++)
            if (stages[s]) shaders[s] = &stages[s]->shader;
        std::lock_guard lock(objects_mutex_);
        layout = bindless_layout_locked(shaders);
    }
    if (!layout) return VK_NULL_HANDLE;
    std::vector<uint64_t>& key = bindless_key_;
    key.clear();
    auto ubo_of = [&](int s, const Resource& r) {
        const int index = static_cast<int>(r.cbuf_index) - static_cast<int>(kUniformBankBase);
        return index >= 0 && index < kMaxUniforms ? st_.bindings[s].uniforms[index] : BufferRange{};
    };
    size_t total = 0;
    const auto t0 = std::chrono::steady_clock::now();
    const bool trace = frame_ == trace_frame_;
    const auto textures = st_.texture_pool ? st_.texture_pool->texture_snapshot() : nullptr;
    const auto samplers = st_.sampler_pool ? st_.sampler_pool->sampler_snapshot() : nullptr;
    if (textures != handle_textures_ || samplers != handle_samplers_) {
        handle_textures_ = textures;
        handle_samplers_ = samplers;
        ++handle_pool_epoch_;
    }
    uint64_t resolved = 0;
    for (int s = 0; s < kStages; s++) {
        if (!stages[s] || !stages[s]->shader.has_bindless_set) continue;
        for (const Resource& r : stages[s]->shader.resources) {
            if (r.set != nvnshader::kBindlessSet) continue;
            const BufferRange u = ubo_of(s, r);
            total += r.count;
            uint32_t found = 0;
            std::string shown;  // DEPURACIÓN (traza): los primeros elementos resueltos
            const VkImageView missing = dummy_view(r.texture_type);
            if (!missing) return VK_NULL_HANDLE;
            const VkSampler default_sample = sampler(nullptr);
            // Key each binding by runs of identical resolved descriptors. A run ends at an
            // exclusive array index. Expand the complete Vulkan array only on a cache miss.
            key.push_back((uint64_t{r.binding} << 32) | r.count);
            VkImageView run_view = missing;
            VkSampler run_sampler = default_sample;
            uint32_t run_start = 0;
            auto finish_run = [&](uint32_t end) {
                if (end == run_start) return;
                key.push_back(end);
                key.push_back(reinterpret_cast<uint64_t>(run_view));
                key.push_back(reinterpret_cast<uint64_t>(run_sampler));
                run_start = end;
            };
            auto descriptor = [&](uint32_t index, VkSampler sample, VkImageView iv) {
                if (sample == run_sampler && iv == run_view) return;
                finish_run(index);
                run_sampler = sample;
                run_view = iv;
            };
            // The same masked handle can occur hundreds of times in one reflected array.
            // Reuse its resolved descriptor only within this binding of this draw.
            struct Element { uint32_t handle; VkSampler sampler; VkImageView view; bool found; };
            std::array<Element, 32> elements{};
            const uint64_t first_end = uint64_t{r.cbuf_offset} + sizeof(uint32_t);
            uint32_t readable = 0;
            if (bindless_ && u.address && u.size >= first_end) {
                const uint64_t last = r.size_shift < 64 ? (u.size - first_end) >> r.size_shift : 0;
                readable = static_cast<uint32_t>(std::min<uint64_t>(r.count, last + 1));
            }
            const auto* runs = !trace && !loading_armed_ ? handle_runs(u.address + r.cbuf_offset, readable, r) : nullptr;
            const uint32_t entries = runs ? static_cast<uint32_t>(runs->size()) : readable;
            for (uint32_t i = 0; i < entries; ++i) {
                const uint32_t k = runs ? (*runs)[i].first : i;
                const uint32_t repeats = runs ? (i + 1 < entries ? (*runs)[i + 1].first : readable) - k : 1;
                Texture* t = nullptr;
                TextureViewDesc v{};
                Sampler* smp = nullptr;
                const uint64_t off = uint64_t{r.cbuf_offset} + (uint64_t{k} << (r.size_shift & 63));
                uint32_t word;
                if (runs) word = (*runs)[i].word;
                else memcpy(&word, reinterpret_cast<const void*>(u.address + off), sizeof(word));
                if (word == 0xFFFFFFFFu) {
                    descriptor(k, default_sample, missing);
                    continue;
                }
                const uint32_t masked = word & r.handle_and_mask;
                auto& entry = elements[(masked ^ (masked >> 16)) % elements.size()];
                if (!trace && !loading_armed_ && entry.view && entry.handle == masked) {
                    if (entry.found) found += repeats;
                    descriptor(k, entry.sampler, entry.view);
                    continue;
                }
                if (resolve_handle(word, r, t, v, smp, textures.get(), samplers.get())) {
                    if (!prepare_shader_view(*t, v, ViewUse::Sampled)) return VK_NULL_HANDLE;
                    if (attachments && t->image) {
                        t = sample_source(*t, v, ViewUse::Sampled, *attachments);
                        if (!t) return VK_NULL_HANDLE;
                    }
                    found += repeats;
                    if (trace && found <= 6) {
                        char b[96];
                        snprintf(b, sizeof b, " %u:%u/%u(%dx%d t%d f0x%x)", k, (word & r.handle_and_mask) & 0xFFFFF,
                                 ((word & r.handle_and_mask) | r.handle_or) >> 20, t->desc.width, t->desc.height, t->desc.target,
                                 t->desc.format);
                        shown += b;
                    }
                }
                const VkImageView resolved_view = t && t->image ? view(*t, v, ViewUse::Sampled) : VK_NULL_HANDLE;
                const VkImageView iv = resolved_view ? resolved_view : missing;
                if (loading_armed_ && attachments && t && iv && iv != missing)
                    check_loading_binding(*t, v, iv, s, r, k, smp);
                const VkSampler sample = sampler(smp);
                entry = {masked, sample, iv, t != nullptr};
                descriptor(k, sample, iv);
            }
            if (readable < r.count) descriptor(readable, default_sample, missing);
            finish_run(r.count);
            resolved += found;
            if (trace)
                NVN_LOG("F bindless %s c%u[0x%x] x%u paso %u (& 0x%x | 0x%x) de %#llx+%llu: %u texturas%s\n",
                        nvnshader::stage_name(static_cast<Stage>(s)), r.cbuf_index, r.cbuf_offset, r.count, 1u << r.size_shift,
                        r.handle_and_mask, r.handle_or, static_cast<unsigned long long>(u.address),
                        static_cast<unsigned long long>(u.size), found, shown.c_str());
        }
    }
    const uint64_t hash = hash_bytes(key.data(), key.size() * sizeof(uint64_t), reinterpret_cast<uint64_t>(layout));
    if (VkDescriptorSet cached = sets_cache_.find(descriptor_epoch_, layout, key, hash)) {
        prof_bindless_reused_++;
        return cached;
    }
    VkDescriptorSet set = descriptor_set(layout);
    if (!set) return VK_NULL_HANDLE;
    static std::vector<VkDescriptorImageInfo> infos;
    static std::vector<VkWriteDescriptorSet> writes;
    infos.clear();
    infos.reserve(total);  // keep pImageInfo stable across subsequent bindings
    writes.clear();
    for (size_t cursor = 0; cursor < key.size();) {
        const uint64_t binding = key[cursor++];
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set;
        w.dstBinding = static_cast<uint32_t>(binding >> 32);
        w.descriptorCount = static_cast<uint32_t>(binding);
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        const size_t first = infos.size();
        for (uint32_t i = 0; i < w.descriptorCount;) {
            const auto end = static_cast<uint32_t>(key[cursor++]);
            const auto iv = reinterpret_cast<VkImageView>(key[cursor++]);
            const auto sample = reinterpret_cast<VkSampler>(key[cursor++]);
            infos.insert(infos.end(), end - i, {sample, iv, VK_IMAGE_LAYOUT_GENERAL});
            i = end;
        }
        w.pImageInfo = infos.data() + first;
        writes.push_back(w);
    }
    vkUpdateDescriptorSets(vk()->device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    sets_cache_.insert(descriptor_epoch_, layout, key, hash, set);
    prof_bindless_sets_++;
    prof_bindless_textures_ += resolved;
    prof_bindless_us_ += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    static int logged = 0;
    if (logged++ < 12)
        NVN_LOG("bindless: set 1 armado (%zu arrays, %zu descriptores, %llu texturas del juego)\n", writes.size(), infos.size(),
                static_cast<unsigned long long>(resolved));
    return set;
}

// Imagen de un shader (SULD/SUST): la ranura de BindImage por el offset de su handle en la tabla del driver (ver
// kImageHandleBase) y, si ahí no hay nada enlazado o el offset cae fuera, por orden: la k-ésima imagen del shader (por
// offset) es la k-ésima ranura enlazada. Sirve tal cual con una sola imagen (el caso de MH Rise: compute, ranura 0)
// aunque la base de la tabla no fuera la supuesta. El handle es el id del pool de texturas (RegisterImage lo guarda
// ahí, como RegisterTexture).
bool Executor::resolve_image(Stage s, const ProgramStage& ps, const Resource& r, Texture*& tex, TextureViewDesc& v) {
    tex = nullptr;
    v = {};
    const StageBindings& b = st_.bindings[static_cast<int>(s)];
    const uint32_t bank = s == Stage::Compute ? kComputeDriverBank : kTextureHandleBank;
    // La tabla empieza en 0x120 (como en compute: kComputeImageHandleBase): un pixel shader de MH Rise escribe c2[0x120] y
    // c2[0x128] con las ranuras 0 y 1 enlazadas (5024 y 5088); con la base supuesta 0x128 las dos iban a la ranura 0 y la 1
    // no se escribía nunca. En compute vale también c2. <datos>/image_base_128: como antes.
    const uint32_t base = image_base_120_ ? kComputeImageHandleBase : kImageHandleBase;
    const bool bank_ok = r.cbuf_index == bank || (image_base_120_ && s == Stage::Compute && r.cbuf_index == kTextureHandleBank);
    int slot = -1;
    bool by_offset = false;
    if (bank_ok && r.cbuf_offset >= base && (r.cbuf_offset - base) % 8 == 0 &&
        (r.cbuf_offset - base) / 8 < static_cast<uint32_t>(kMaxImages) && b.images[(r.cbuf_offset - base) / 8]) {
        slot = static_cast<int>((r.cbuf_offset - base) / 8);
        by_offset = true;
    } else {
        int rank = 0;
        for (const Resource& o : ps.shader.resources)
            if (o.kind == Resource::Image &&
                (o.cbuf_index < r.cbuf_index || (o.cbuf_index == r.cbuf_index && o.cbuf_offset < r.cbuf_offset)))
                rank++;
        for (int k = 0, seen = 0; k < kMaxImages && slot < 0; k++)
            if (b.images[k] && seen++ == rank) slot = k;
    }
    // Qué se decidió para cada offset (la primera vez): con varias imágenes, si el orden no cuadra, aquí se ve la base.
    static std::set<std::tuple<int, uint32_t, uint32_t, int>> seen;
    if (seen.size() < 32 && seen.insert({static_cast<int>(s), r.cbuf_index, r.cbuf_offset, slot}).second) {
        std::string bound;
        for (int k = 0; k < kMaxImages; k++)
            if (b.images[k]) bound += " " + std::to_string(k) + ":" + std::to_string(handle_texture_id(b.images[k]));
        NVN_LOG("imagen %s en c%u[0x%x] (%s): ranura %d por %s; enlazadas (ranura:id):%s\n", nvnshader::stage_name(s),
                r.cbuf_index, r.cbuf_offset, r.is_written ? "escrita" : "leída", slot, by_offset ? "offset" : "orden",
                bound.empty() ? " ninguna" : bound.c_str());
    }
    if (slot < 0) return false;
    const uint32_t id = handle_texture_id(b.images[slot]);
    if (st_.texture_pool) {
        const auto e = st_.texture_pool->texture_at(id);
        tex = e.texture;
        v = e.view;
    }
    if (tex && r.is_written && (tex->desc.target == kTargetCube || tex->desc.target == kTargetCubeArray))
        note_cube_write(*tex, v, "imagen de un shader", nullptr, ps.print, frame_);
    if (tex && r.is_written) {  // DEPURACIÓN: qué shader escribe qué textura (los volúmenes de luz que hace la GPU)
        static std::set<std::pair<uint64_t, uint32_t>> written;
        if (written.size() < 96 && written.insert({ps.print, id}).second)
            NVN_LOG("imagen escrita: %s %016llx -> textura id %u (%dx%dx%d target %d formato 0x%x) en c%u[0x%x], frame %llu\n",
                    nvnshader::stage_name(s), static_cast<unsigned long long>(ps.print), id, tex->desc.width, tex->desc.height,
                    tex->desc.depth, tex->desc.target, tex->desc.format, r.cbuf_index, r.cbuf_offset,
                    static_cast<unsigned long long>(frame_));
    }
    return tex != nullptr;
}

VkImageView Executor::storage_view(Texture* t, const TextureViewDesc& v, uint32_t texture_type, Texture*& bound) {
    // El tipo de vista tiene que ser el de la imagen del shader (Shader::TextureType: 1D, 1D array, 2D, 2D array, 3D,
    // cubo, array de cubos, buffer, rect), si es de la misma familia que la textura (un 2D puede verse como 2D array,
    // un cubo como 2D array; un 3D solo como 3D).
    if (t && t->image && t->storage_image)
        if (VkImageView iv = view(*t, v, ViewUse::Storage)) {
            bound = t;
            return iv;
        }
    if (t) {
        static std::set<std::pair<int, int>> logged;  // (formato, flags)
        if (logged.size() < 16 && logged.insert({t->desc.format, t->desc.flags}).second)
            NVN_LOG("imagen sin STORAGE (formato 0x%x, flags 0x%x, %dx%d%s): se enlaza la de reserva\n", t->desc.format,
                    t->desc.flags, t->desc.width, t->desc.height, t->image ? "" : ", sin imagen");
    }
    return dummy_view(texture_type, ViewUse::Storage);
}

// Por la identidad de lo traducido de cada etapa (no por la dirección de la etapa): los programas con el mismo código
// comparten layout, y el del precalentado es el mismo objeto que usará el ejecutor.
Layout& Executor::layout(const ProgramStage* const* stages) {
    const nvnshader::Translated* shaders[kStages] = {};
    uint64_t ids[kStages] = {};
    for (int s = 0; s < kStages; s++) {
        if (!stages[s]) continue;
        shaders[s] = &stages[s]->shader;
        ids[s] = stages[s]->spirv_id;
        if (per_stage_keys_) {  // pipelines_per_stage: uno por etapa del juego, como antes
            const uint64_t mine[3] = {reinterpret_cast<uint64_t>(stages[s]), st_.stage_generation[s], ids[s]};
            ids[s] = hash_bytes(mine, sizeof mine) | 1;
        }
    }
    if (memo_layout_ && last_layout_ && memcmp(ids, last_layout_ids_, sizeof ids) == 0) {
        if (prof_timing_) fine_.layout_memo++;
        return *last_layout_;
    }
    Layout& l = layout_for(shaders, ids);
    last_layout_ = &l;
    memcpy(last_layout_ids_, ids, sizeof ids);
    return l;
}

// Desde el ejecutor y desde el precalentado: con objects_mutex_ (las referencias a layouts_ siguen valiendo aunque crezca).
Layout& Executor::layout_for(const nvnshader::Translated* const* shaders, const uint64_t* ids) {
    BytesKey<LayoutKeyData> key;
    memcpy(key.value.ids, ids, sizeof key.value.ids);
    std::lock_guard lock(objects_mutex_);
    if (auto it = layouts_.find(key); it != layouts_.end()) return it->second;
    // Dinámicos solo en gráficos y si caben en los límites del dispositivo (si no, el camino de siempre).
    uint32_t ubos = 0, ssbos = 0;
    for (int s = 0; s < kStages; s++)
        if (shaders[s])
            for (const Resource& r : shaders[s]->resources)
                ubos += r.kind == Resource::UniformBuffer, ssbos += r.kind == Resource::StorageBuffer;
    const VkPhysicalDeviceLimits& limits = vk()->props.limits;
    const bool dynamic = !shaders[static_cast<int>(Stage::Compute)] && ubos <= limits.maxDescriptorSetUniformBuffersDynamic &&
                         ssbos <= limits.maxDescriptorSetStorageBuffersDynamic;
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (int s = 0; s < kStages; s++) {
        if (!shaders[s]) continue;
        for (const Resource& r : shaders[s]->resources) {
            if (r.set) continue;  // arrays bindless: set 1 (bindless_layout)
            VkDescriptorType type;
            switch (r.kind) {
                case Resource::UniformBuffer:
                    type = dynamic ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    break;
                case Resource::StorageBuffer:
                    type = dynamic ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    break;
                case Resource::TextureBuffer: type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER; break;
                case Resource::ImageBuffer: type = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER; break;
                case Resource::Texture: type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; break;
                default: type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; break;
            }
            bindings.push_back({r.binding, type, 1, static_cast<VkShaderStageFlags>(vk_stage(static_cast<Stage>(s))), nullptr});
        }
    }
    Layout l;
    l.dynamic = dynamic;
    static bool logged = false;
    if (!logged) {
        logged = true;
        NVN_LOG("descriptores dinámicos: hasta %u UBO y %u SSBO por set\n", limits.maxDescriptorSetUniformBuffersDynamic,
                limits.maxDescriptorSetStorageBuffersDynamic);
    }
    VkDescriptorSetLayoutCreateInfo si{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    si.bindingCount = static_cast<uint32_t>(bindings.size());
    si.pBindings = bindings.data();
    if (VkResult r = vkCreateDescriptorSetLayout(vk()->device, &si, nullptr, &l.set); r != VK_SUCCESS) {
        l.set = VK_NULL_HANDLE;
        NVN_LOG("vkCreateDescriptorSetLayout (%zu bindings): %s\n", bindings.size(), vk_result_name(r));
    }
    l.bindless = bindless_layout_locked(shaders);
    VkDescriptorSetLayout sets[3] = {l.set, l.bindless, VK_NULL_HANDLE};
    VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pi.setLayoutCount = l.bindless ? 2 : 1;
    pi.pSetLayouts = sets;
    // The draw count check (count_check.h): the count in set 2 (set 1 empty without bindless) and two push constants
    const VkPushConstantRange push{VK_SHADER_STAGE_VERTEX_BIT, 0, kCountCheckPushSize};
    if (shaders[static_cast<int>(Stage::Vertex)] && shaders[static_cast<int>(Stage::Vertex)]->count_check) {
        if (!count_check_set_layout_) {
            const VkDescriptorSetLayoutBinding b{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
            VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            ci.bindingCount = 1;
            ci.pBindings = &b;
            vkCreateDescriptorSetLayout(vk()->device, &ci, nullptr, &count_check_set_layout_);
            ci.bindingCount = 0;
            vkCreateDescriptorSetLayout(vk()->device, &ci, nullptr, &empty_set_layout_);
        }
        if (count_check_set_layout_ && empty_set_layout_) {
            if (!sets[1]) sets[1] = empty_set_layout_;
            sets[kCountCheckSet] = count_check_set_layout_;
            pi.setLayoutCount = kCountCheckSet + 1;
            pi.pushConstantRangeCount = 1;
            pi.pPushConstantRanges = &push;
            l.count_check = true;
        }
    }
    vkCreatePipelineLayout(vk()->device, &pi, nullptr, &l.pipeline);
    return layouts_[key] = l;
}

// Crea la pipeline gráfica de la clave con solo lo que hay en ella: la usan pipeline() y el hilo del precalentado, así
// lo que este deja creado es lo mismo que habría creado el ejecutor. modules: el de cada etapa que entra (VK_NULL_HANDLE:
// no entra; el geometry de la compactación saltado ya viene a nulo). vs_inputs: genéricos que lee el vertex shader.
// DIAGNÓSTICO: los formatos de los atributos de una pipeline que el driver rechazó (" 0:RGBA8_UNORM 2:RG16_I2F…"), con
// los escalados (USCALED / SSCALED: el formato "_UI2F" / "_I2F" de NVN) marcados: Eden los emula en el driver de Qualcomm
// porque no los compila bien.
std::string attrib_formats(const PipelineKey& key, uint64_t vs_inputs) {
    std::string out;
    bool scaled = false;
    for (uint32_t k = 0; k < 16; k++) {
        if (!(vs_inputs & (1ull << k))) continue;
        const FormatInfo* f = format_info(key.attribs[k].format);
        char b[48];
        snprintf(b, sizeof b, " %u:%s", k, f && f->name ? f->name : "?");
        out += b;
        if (f && f->name && (strstr(f->name, "_UI2F") || strstr(f->name, "_I2F"))) scaled = true;
    }
    return " atributos" + out + (scaled ? " (con escalados)" : "");
}


// Consume the lowest active vertex location, preserving Vulkan binding order.
static uint32_t take_location(uint32_t& used) {
    const uint32_t location = static_cast<uint32_t>(__builtin_ctz(used));
    used &= used - 1;
    return location;
}

static void apply_fragment_exports(PipelineKey& key, const nvnshader::Translated* fs) {
    const uint32_t exports = fs ? fs->color_outputs : 0;
    key.masks.masks = color_write_channels(key.masks.masks, exports);
    key.color.blend_enable &= exports;
}

VkPipeline Executor::face_pipeline(const PipelineKey& key, uint64_t inputs, Layout& layout, VkRenderPass pass) {
    // The captured face draw has vertex and fragment stages only. Keep its fetch specialization.
    constexpr int vertex = static_cast<int>(Stage::Vertex), fragment = static_cast<int>(Stage::Fragment);
    for (int i = 0; i < kStages; ++i)
        if (i != vertex && i != fragment && i != static_cast<int>(Stage::Compute) && st_.stages[i] &&
            !(i == static_cast<int>(Stage::Geometry) && key.skip_gs)) return VK_NULL_HANDLE;
    const auto* vs = st_.stages[vertex];
    const auto* fs = st_.stages[fragment];
    if (!vs) return VK_NULL_HANDLE;
    nvnshader::Translated variant;
    const auto* translated = &fs->shader;
    if (fs->shader.input_generics & ~vs->shader.output_generics) {
        const auto v = geometry_variant(*const_cast<ProgramStage*>(fs), vs->shader.output_generics, 0);
        if (!v.module || !cached_translation(v.cache_key, variant)) return VK_NULL_HANDLE;
        translated = &variant;
    }
    const std::string dir = "/storage/emulated/0/NieR-volcados";
    mkdir(dir.c_str(), 0775);
    const std::string name = dir + "/face-f" + std::to_string(frame_) + "-d" + std::to_string(traced_draws_);
    const auto& source = translated->spirv;
    std::ofstream original(name + ".spv", std::ios::binary);
    original.write(reinterpret_cast<const char*>(source.data()), source.size() * sizeof(uint32_t));
    original.close();
    if (!original) return VK_NULL_HANDLE;
    const uint64_t h = hash_bytes(&key, sizeof key);
    if (auto it = face_probe_pipelines_.find(h); it != face_probe_pipelines_.end()) return it->second;
    VkShaderModule modules[kStages]{};
    const auto fetch = vertex_input_types(key, inputs);
    modules[vertex] = fetch ? vertex_variant(*const_cast<ProgramStage*>(vs), fetch).module : vs->module;
    if (!modules[vertex]) return VK_NULL_HANDLE;
    const auto patched = instrument_face_nan(source);
    if (patched.empty() || create_shader_module(vk()->device, patched, &modules[fragment]) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    VkPipeline diagnostic = VK_NULL_HANDLE;
    const auto status = make_graphics_pipeline(key, modules, inputs, layout.pipeline, pass, VK_NULL_HANDLE, diagnostic);
    vkDestroyShaderModule(vk()->device, modules[fragment], nullptr);
    if (status != VK_SUCCESS) {
        NVN_LOG("face probe: graphics pipeline %s\n", vk_result_name(status));
        return VK_NULL_HANDLE;
    }
    face_probe_pipelines_.emplace(h, diagnostic);
    return diagnostic;
}

VkPipeline Executor::pipeline(const cmd::Draw& d, VkRenderPass rp, Layout& l, uint32_t color_count,
                              VkSampleCountFlagBits samples, bool probe_face) {
    if (prof_timing_) fine_.pipe_lookups++;
    const VkContext& context = *vk();
    PipelineKey key;
    memset(&key, 0, sizeof key);
    // Por lo traducido, no por la etapa: los programas del juego con el mismo código comparten pipelines (antes cada
    // ProgramStage llevaba las suyas: 2200 etapas de 496 huellas, 9354 pipelines en una partida de 21 min).
    for (int s = 0; s < kStages; s++) key.stages[s] = st_.stages[s] ? st_.stages[s]->spirv_id : 0;
    for (int k = 0; k < 8; k++) key.pass_colors[k] = static_cast<uint32_t>(rp_formats_[k]);
    key.pass_depth = static_cast<uint32_t>(rp_depth_);
    key.topology = static_cast<uint32_t>(d.primitive);
    key.polygon = st_.polygon;
    key.depth_stencil = st_.depth_stencil;
    if (context.extended_dynamic_state) normalize_dynamic(key);  // (set per draw: see the dynamic state below)
    key.color = st_.color;
    key.masks = st_.masks;
    key.masks.masks &= rp_keep_masks_;  // merged_passes: los adjuntos del pase que no son de los targets del juego, sin escribir
    memcpy(key.blend, st_.blend, sizeof key.blend);
    key.color_count = color_count;
    key.samples = samples;
    // El geometry shader de la compactación cuesta mucho en el Adreno (traducido por Eden como passthrough emulado: el
    // pre-pase de profundidad se llevaba ~5,5 ms de GPU en las Ruinas) y lo que produce ya no se usa: sin él, el vertex
    // shader va directo al fragment (mismas posiciones, la misma profundidad). no_gs_skip: con él, como antes.
    const ProgramStage* gs = st_.stages[static_cast<int>(Stage::Geometry)];
    const ProgramStage* vs_stage = st_.stages[static_cast<int>(Stage::Vertex)];
    const ProgramStage* fs_stage = st_.stages[static_cast<int>(Stage::Fragment)];
    apply_fragment_exports(key, fs_stage ? &fs_stage->shader : nullptr);
    key.skip_gs = gs && gs->compaction_gs && skip_compaction_gs_ && vs_stage &&
                  (!fs_stage || (fs_stage->shader.input_generics & ~vs_stage->shader.output_generics) == 0);
    key.alpha_to_coverage = st_.multisample.alpha_to_coverage;
    // Reinicio de primitiva: Vulkan 1.1 no lo admite en listas (y en ellas no hace nada visible: el índice de todo unos
    // no se dibuja igual) y siempre usa el índice de todo unos.
    {
        const VkPrimitiveTopology topology = d.primitive == 7 ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST : vk_topology(d.primitive);
        const bool strip = topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP || topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP ||
                           topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN ||
                           topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY ||
                           topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
        key.primitive_restart = st_.primitive_restart && strip && d.index_type >= 0;
        key.patch_size = static_cast<uint8_t>(std::min<uint32_t>(st_.patch_size, 255));
        const uint32_t all_ones = d.index_type == 0 ? 0xFFu : d.index_type == 1 ? 0xFFFFu : 0xFFFFFFFFu;
        static int logged = 0;
        if (key.primitive_restart && st_.restart_index != all_ones && logged++ < 4)
            NVN_LOG("reinicio de primitiva con índice 0x%x (índices de %u bytes): Vulkan solo reinicia con 0x%x\n",
                    st_.restart_index, index_size(d.index_type), all_ones);
    }
    const ProgramStage* vs = st_.stages[static_cast<int>(Stage::Vertex)];
    uint64_t inputs = vs ? vs->shader.input_generics : 0;
    for (uint32_t used = inputs & 0xffff; used;) {
        const auto k = take_location(used);
        VertexAttribState a = k < st_.attribs.size() ? st_.attribs[k] : VertexAttribState{};
        key.attribs[k] = a;
        if (a.stream < st_.streams.size()) {
            key.strides[k] = st_.streams[a.stream].stride;
            key.divisors[k] = st_.streams[a.stream].divisor;
        }
    }
    // (the attribute check below still sees the layout: diagnostics)
    PipelineKey layout_key;
    if (context.vertex_input_dynamic_state) {
        memcpy(layout_key.attribs, key.attribs, sizeof key.attribs);
        memcpy(layout_key.strides, key.strides, sizeof key.strides);
        memcpy(layout_key.divisors, key.divisors, sizeof key.divisors);
        normalize_vertex_input(key);  // (set per draw: see the dynamic state)
    }
    // La misma clave que la última encontrada: su hash y su pipeline, sin volver a hashear ni buscar.
    if (probe_face) return face_pipeline(key, inputs, l, rp);
    const bool memo = memo_pipeline_ && !per_stage_keys_ && last_pipeline_ && memcmp(&key, &last_pipeline_key_, sizeof key) == 0;
    uint64_t h = memo ? last_pipeline_hash_ : hash_bytes(&key, sizeof key);
    if (prof_count_stage_keys_ || per_stage_keys_) {  // la clave de antes (por etapa) de este draw
        struct {
            const ProgramStage* stages[kStages];
            uint64_t generation[kStages], h;
        } old{};
        memcpy(old.stages, st_.stages, sizeof old.stages);
        memcpy(old.generation, st_.stage_generation, sizeof old.generation);
        old.h = h;
        const uint64_t old_h = hash_bytes(&old, sizeof old);
        if (prof_count_stage_keys_) prof_stage_keys_.insert(old_h);
        // <datos>/pipelines_per_stage (en vivo): como antes, una pipeline por etapa del juego, sin compartirlas (ni
        // precalentado: el hilo las deja por lo traducido).
        if (per_stage_keys_) h = old_h;
    }
    if (loading_armed_) {
        loading_pipeline_key_ = h;
        loading_blend_enable_ = key.color.blend_enable;
    }
    if (memo) {
        if (prof_timing_) fine_.pipe_memo++;
        return last_pipeline_;
    }
    if (auto it = pipelines_.find(h); it != pipelines_.end()) {
        // Compartida con otras etapas del mismo SPIR-V: si sin el geometry de la compactación no se pudo crear (la de esa
        // clave quedó nula), esta etapa también vuelve a usarlo (antes cada etapa fallaba la suya y se marcaba).
        if (!it->second && key.skip_gs) const_cast<ProgramStage*>(gs)->compaction_gs = false;
        if (it->second && !per_stage_keys_) {  // la del memo (las nulas no: su efecto de arriba tiene que repetirse)
            last_pipeline_ = it->second;
            last_pipeline_hash_ = h;
            memcpy(&last_pipeline_key_, &key, sizeof key);
        }
        return it->second;
    }
    std::optional<PipelineBuild> warm;
    if (!per_stage_keys_) {
        LoadTimer timer(*this, kLdWarm);
        warm = warm_take(h);
        if (warm && warm->pipeline) {
            ++prof_warm_taken_;
            return pipelines_[h] = warm->pipeline;
        }
    }
    bool async = async_pipelines_ && rp_recurring_ && !per_stage_keys_ && !key.skip_gs;
    bool known = false;
    if (async && !known_off_) {
        if (known_frame_ != frame_) {
            known_frame_ = frame_;
            known_us_ = 0;
        }
        if (known_us_ < kKnownBudgetUs && pipeline_known(h)) {
            known = true;
            async = false;
        }
    }
    // Un atributo que no cabe en la zancada de su stream (offset + tamaño > stride) lee parte del vértice siguiente. En
    // t6 (05:46, antes de 3d8197d) las cabezas con skinning en compute (la del cazador) se dibujaban en el pase de
    // color con las zancadas 12/8/4 del cuerpo anterior en vez de 12/12/8: nvnVertexStreamStateGetStride/GetDivisor
    // eran stubs a 0 y la caché de estado del juego (main+0x77178b0, compara layouts con esos getters) daba por iguales
    // todos los de mismo número de streams y no volvía a enlazar el estado: normales y tangentes de otro vértice (cara
    // oscura y a facetas). Se vigila al crear cada pipeline (una línea por caso, como mucho 16).
    const PipelineKey& lk = context.vertex_input_dynamic_state ? layout_key : key;
    for (int k = 0; k < 16; k++) {
        if (!(inputs & (1ull << k)) || lk.divisors[k] || !lk.strides[k]) continue;
        const FormatInfo* f = format_info(lk.attribs[k].format);
        if (!f || f->block != 1 || uint32_t{lk.attribs[k].offset} + f->bytes <= lk.strides[k]) continue;
        static int logged_stride = 0;
        const ProgramStage* ps = st_.stages[static_cast<int>(Stage::Fragment)];
        if (logged_stride++ < 16)
            NVN_LOG("atributo %d (formato %s, %u bytes desde %u) más ancho que la zancada %u de su stream %u: vertex %016llx "
                    "pixel %016llx (¿estado de streams de otro draw?)\n", k, f->name, f->bytes, lk.attribs[k].offset,
                    lk.strides[k], lk.attribs[k].stream, static_cast<unsigned long long>(vs ? vs->print : 0),
                    static_cast<unsigned long long>(ps ? ps->print : 0));
    }

    // El módulo de cada etapa (con su clave de la caché de SPIR-V, para el registro del precalentado).
    VkShaderModule modules[kStages] = {};
    uint64_t previous_for[kStages];  // (el previous_stores de la variante de cada etapa: ~0, la etapa tal cual)
    std::fill(std::begin(previous_for), std::end(previous_for), ~0ull);
    const uint64_t input_types = vertex_input_types(key, inputs);
    PipelineRecord rec;
    memset(&rec, 0, sizeof rec);
    // (not with the draw count check: the warm-up builds from the SPIR-V cache, which keeps the stage without it)
    bool recordable = !(st_.stages[static_cast<int>(Stage::Vertex)] && st_.stages[static_cast<int>(Stage::Vertex)]->count_check);
    for (int s = 0; s < kStages; s++) {
        if (!st_.stages[s] || s == static_cast<int>(Stage::Compute)) continue;
        rec.base[s] = st_.stages[s]->cache_key;
        recordable &= (st_.stages[s]->cache_key.a | st_.stages[s]->cache_key.b) != 0;
        if (key.skip_gs && s == static_cast<int>(Stage::Geometry)) continue;
        ProgramStage::Variant v{st_.stages[s]->module, st_.stages[s]->cache_key};
        if (s == static_cast<int>(Stage::Vertex) && input_types) {
            v = vertex_variant(*const_cast<ProgramStage*>(st_.stages[s]), input_types);
            if (!v.module) return VK_NULL_HANDLE;
        }
        if (s == static_cast<int>(Stage::Geometry) && vs) {
            // Topología de entrada según la primitiva de NVN (orden de OpenGL).
            int topology = d.primitive == 0                       ? 0
                           : d.primitive >= 1 && d.primitive <= 3  ? 1
                           : d.primitive == 10 || d.primitive == 11 ? 2
                           : d.primitive == 12 || d.primitive == 13 ? 4
                                                                    : 3;
            v = geometry_variant(*const_cast<ProgramStage*>(st_.stages[s]), vs->shader.output_generics, topology);
            if (!v.module) return pipelines_[h] = VK_NULL_HANDLE;
        }
        // Un pixel shader que lee genéricos que la etapa anterior no escribe: el Adreno no enlaza ("input in_attrN not
        // declared in output from previous stage"). Su variante solo declara los que llegan; los demás se leen a cero.
        if (s == static_cast<int>(Stage::Fragment) && vs) {
            const ProgramStage* gs = key.skip_gs ? nullptr : st_.stages[static_cast<int>(Stage::Geometry)];
            const uint64_t previous = gs ? gs->shader.output_generics : vs->shader.output_generics;
            if (st_.stages[s]->shader.input_generics & ~previous) {
                previous_for[s] = previous;
                v = geometry_variant(*const_cast<ProgramStage*>(st_.stages[s]), previous, 0);
                if (!v.module) return pipelines_[h] = VK_NULL_HANDLE;
                static int logged = 0;
                if (logged++ < 8)
                    NVN_LOG("pixel %016llx lee genéricos 0x%llx que no escribe la etapa anterior (0x%llx): variante\n",
                            static_cast<unsigned long long>(st_.stages[s]->print),
                            static_cast<unsigned long long>(st_.stages[s]->shader.input_generics),
                            static_cast<unsigned long long>(previous));
            }
        }
        modules[s] = v.module;
        rec.module[s] = v.cache_key;
        recordable &= (v.cache_key.a | v.cache_key.b) != 0;
    }

    VkPipeline p = VK_NULL_HANDLE;
    const auto created_at = std::chrono::steady_clock::now();
    const bool during_save = driver_cache_.reading();
    const auto build = [&] {
        PipelineBuild result;
        result.status = driver_cache_.compile([&](VkPipelineCache cache) {
            return make_graphics_pipeline(key, modules, inputs, l.pipeline, rp, cache, result.pipeline);
        });
        return result;
    };
    VkResult created;
    const bool rejected_at_warmup = warm && std::equal(std::begin(rec.module), std::end(rec.module), warm->modules);
    const bool worker = async && recordable && !rejected_at_warmup;
    {
        LoadTimer timer(*this, kLdPipeline);
        if (rejected_at_warmup) {
            created = warm->status;
        } else if (worker) {
            hud::shader_queued();
            g_drop_stats.pipelines_async.fetch_add(1, std::memory_order_relaxed);
            const auto result = await_pipeline(enqueue_pipeline(PipelineTask(build), true));
            hud::shader_done();
            p = result.pipeline;
            created = result.status;
        } else {
            const auto result = build();
            p = result.pipeline;
            created = result.status;
        }
    }
    const uint64_t created_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - created_at).count();
    prof_pipeline_us_ += created_us;
    if (known) {
        known_us_ += created_us;
        g_drop_stats.pipelines_known.fetch_add(1, std::memory_order_relaxed);
        g_drop_stats.pipelines_known_ns.fetch_add(created_us * 1000, std::memory_order_relaxed);
        if (created_us > kKnownSlowUs && ++known_slow_ >= 3) {
            known_off_ = true;
            NVN_LOG("recorded pipelines: 3 took as long as a compile (the last %llu ms; pipelines.bin without them?), "
                    "back to the background threads for this session\n", static_cast<unsigned long long>(created_us / 1000));
        }
    } else if (!worker) {
        g_drop_stats.pipelines_live.fetch_add(1, std::memory_order_relaxed);
        g_drop_stats.pipelines_live_ns.fetch_add(created_us * 1000, std::memory_order_relaxed);
    }
    if (during_save || driver_cache_.reading()) note_pipeline_during_save(created_us);
    // El driver no la quiere (el VK_ERROR_UNKNOWN del compilador de Qualcomm 512.8xx, el de otros): sus etapas vertex y
    // pixel traducidas con más prudencia, por niveles (safe_variant), hasta que una valga. La que vale queda como la de
    // esta pipeline y en el registro del precalentado, así que la partida siguiente ya la crea sin fallar antes.
    // <datos>/no_safe_translation: sin reintentos, como antes.
    if (created != VK_SUCCESS && !NVN_FLAG("no_safe_translation")) {
        const int stages_to_redo[] = {static_cast<int>(Stage::Vertex), static_cast<int>(Stage::Fragment)};
        for (int level = 1; level <= kSafeLevels && created != VK_SUCCESS; level++) {
            VkShaderModule safe[kStages];
            std::copy(std::begin(modules), std::end(modules), safe);
            PipelineRecord safe_rec = rec;
            bool usable = true;
            for (int s : stages_to_redo) {
                if (!st_.stages[s] || !modules[s]) continue;
                const ProgramStage::Variant v =
                    safe_variant(*const_cast<ProgramStage*>(st_.stages[s]), previous_for[s], level,
                                 s == static_cast<int>(Stage::Vertex) ? input_types : 0);
                if (!v.module) {
                    usable = false;
                    break;
                }
                safe[s] = v.module;
                safe_rec.module[s] = v.cache_key;
            }
            if (!usable) continue;
            {
                LoadTimer timer(*this, kLdPipeline);
                created = driver_cache_.compile([&](VkPipelineCache cache) {
                    return make_graphics_pipeline(key, safe, inputs, l.pipeline, rp, cache, p);
                });
            }
            static std::atomic<int> logged{0};
            if (created == VK_SUCCESS) {
                rec = safe_rec;
                for (int s : stages_to_redo) recordable &= !st_.stages[s] || (rec.module[s].a | rec.module[s].b) != 0;
                if (logged++ < 24)
                    NVN_LOG("pipeline rejected by the driver: safe translation level %d works (vertex %016llx, "
                            "pixel %016llx)\n", level,
                            static_cast<unsigned long long>(vs ? vs->print : 0),
                            static_cast<unsigned long long>(st_.stages[static_cast<int>(Stage::Fragment)]
                                                                ? st_.stages[static_cast<int>(Stage::Fragment)]->print : 0));
            }
        }
    }
    // DIAGNÓSTICO (las primeras 6 que no salen ni así): la misma pipeline cambiando una parte del estado cada vez, para
    // saber qué es lo que el driver no quiere (en la Adreno 730 con el 512.863 no era ningún shader: los mismos pasaban en
    // otras pipelines). Las de prueba se destruyen; solo es el log.
    if (created != VK_SUCCESS) {
        static std::atomic<int> probed{0};
        if (probed++ < 6) {
            struct Probe {
                const char* what;
                void (*change)(PipelineKey&);
            };
            static const Probe probes[] = {
                {"no blending", [](PipelineKey& k) { k.color.blend_enable = 0; }},
                {"no depth or stencil", [](PipelineKey& k) { memset(&k.depth_stencil, 0, sizeof k.depth_stencil); }},
                {"no alpha to coverage", [](PipelineKey& k) { k.alpha_to_coverage = 0; }},
                {"default rasterization", [](PipelineKey& k) { memset(&k.polygon, 0, sizeof k.polygon); }},
                {"no primitive restart", [](PipelineKey& k) { k.primitive_restart = 0; }},
                {"all of the above", [](PipelineKey& k) {
                     k.color.blend_enable = 0, k.alpha_to_coverage = 0, k.primitive_restart = 0;
                     memset(&k.depth_stencil, 0, sizeof k.depth_stencil);
                     memset(&k.polygon, 0, sizeof k.polygon);
                 }},
            };
            std::string passes;
            for (const Probe& pr : probes) {
                PipelineKey changed = key;
                pr.change(changed);
                VkPipeline test = VK_NULL_HANDLE;
                if (make_graphics_pipeline(changed, modules, inputs, l.pipeline, rp, VK_NULL_HANDLE, test) == VK_SUCCESS) {
                    passes += std::string(passes.empty() ? "" : ", ") + pr.what;
                    vkDestroyPipeline(context.device, test, nullptr);
                }
            }
            char formats[160] = "";
            for (uint32_t k = 0; k < key.color_count && k < 8; k++) {
                const size_t at = strlen(formats);
                snprintf(formats + at, sizeof formats - at, "%s%u", k ? "/" : "", key.pass_colors[k]);
            }
            NVN_LOG("rejected pipeline, state: %u colors (formats %s, depth %u), blend 0x%x, %u samples, "
                    "depth %d/%d, stencil %d; works with %s\n", key.color_count, formats, key.pass_depth,
                    static_cast<unsigned>(key.color.blend_enable), key.samples, key.depth_stencil.depth_test,
                    key.depth_stencil.depth_write, key.depth_stencil.stencil_test,
                    passes.empty() ? "none of the changes (not the state: the shaders or how they link)" : passes.c_str());
        }
    }
    if (VkResult r = created; r != VK_SUCCESS)
    {
        std::string stages;
        for (int k = 0; k < kStages; k++)
            if (st_.stages[k]) {
                const ProgramStage& ps = *st_.stages[k];
                char b[64];
                snprintf(b, sizeof b, "(entradas 0x%llx, %016llx)", (unsigned long long)ps.shader.input_generics,
                         (unsigned long long)ps.print);
                stages += std::string(" ") + nvnshader::stage_name(static_cast<Stage>(k)) + b;
                snprintf(b, sizeof b, "%016llx_%s", (unsigned long long)ps.print, nvnshader::stage_name(static_cast<Stage>(k)));
                dump_rejected_spirv(b, ps.shader.spirv);
            }
        NVN_LOG("vkCreateGraphicsPipelines: %s (%d), primitiva %d, etapas:%s;%s%s\n", vk_result_name(r), static_cast<int>(r),
                d.primitive, stages.c_str(), attrib_formats(key, inputs).c_str(),
                key.skip_gs ? " (sin el geometry shader: se vuelve a usar)" : "");
        if (key.skip_gs) const_cast<ProgramStage*>(gs)->compaction_gs = false;  // el siguiente draw, con él
    }
    if (p && recordable && !per_stage_keys_) {
        rec.kind = 0;
        rec.key = key;
        record_pipeline(rec);
    }
    if (pipelines_.size() % 100 == 99) NVN_LOG("%zu pipelines\n", pipelines_.size() + 1);
    if (p) driver_cache_.note_created(frame_);
    return pipelines_[h] = p;
}

// Solo un pool lleno justifica abrir otro. Antes cualquier fallo (layout inválido, sin memoria…) abría un
// pool nuevo en cada vuelta, unos MB de memoria de la GPU cada uno, y al cargar la fábrica se tragó 2,5 GB
// en segundos hasta que el sistema se quedó sin RAM.
VkDescriptorSet Executor::descriptor_set(VkDescriptorSetLayout layout) {
    VkContext* c = vk();
    InFlight& f = inflight_[cur_];
    if (!layout) return VK_NULL_HANDLE;
    for (;; f.pool++) {
        bool fresh = false;
        if (f.pool == f.pools.size()) {
            if (f.pools.size() >= kMaxDescriptorPools) {
                static int logged = 0;
                if (logged++ < 5) NVN_LOG("%zu pools de descriptores en un envío: draw sin dibujar\n", f.pools.size());
                f.pool = f.pools.size() - 1;
                return VK_NULL_HANDLE;
            }
            fresh = true;
            VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16384},
                                            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096},
                                            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 16384},
                                            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 4096},
                                            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16384},
                                            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024},
                                            {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1024},
                                            {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1024}};
            VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
            pi.maxSets = 4096;
            pi.poolSizeCount = static_cast<uint32_t>(std::size(sizes));
            pi.pPoolSizes = sizes;
            VkDescriptorPool pool;
            if (vkCreateDescriptorPool(c->device, &pi, nullptr, &pool) != VK_SUCCESS) return VK_NULL_HANDLE;
            f.pools.push_back(pool);
        }
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = f.pools[f.pool];
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &layout;
        VkDescriptorSet set;
        VkResult r = vkAllocateDescriptorSets(c->device, &ai, &set);
        if (r == VK_SUCCESS) return set;
        if (fresh || (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL)) {
            static int logged = 0;
            if (logged++ < 8) NVN_LOG("vkAllocateDescriptorSets: %s (pool %zu de %zu)\n", vk_result_name(r), f.pool, f.pools.size());
            return VK_NULL_HANDLE;
        }
    }
}

// Los descriptores de un set con su plantilla: los datos van seguidos, cada uno en 24 bytes (VkDescriptorBufferInfo y
// VkDescriptorImageInfo miden lo mismo), en el orden de writes. El driver no tiene que recorrer un VkWriteDescriptorSet por
// descriptor (~15 por draw). Funciones de Vulkan 1.1 (o de VK_KHR_descriptor_update_template), por vkGetDeviceProcAddr.
bool Executor::update_with_template(Layout& l, VkDescriptorSet set, const std::vector<VkWriteDescriptorSet>& writes) {
    static_assert(sizeof(VkDescriptorBufferInfo) == 24 && sizeof(VkDescriptorImageInfo) == 24);
    constexpr size_t kEntry = 24;
    auto proc = [](const char* core, const char* khr) {
        const PFN_vkVoidFunction f = vkGetDeviceProcAddr(vk()->device, core);
        return f ? f : vkGetDeviceProcAddr(vk()->device, khr);
    };
    static const auto create = reinterpret_cast<PFN_vkCreateDescriptorUpdateTemplate>(
        proc("vkCreateDescriptorUpdateTemplate", "vkCreateDescriptorUpdateTemplateKHR"));
    static const auto update = reinterpret_cast<PFN_vkUpdateDescriptorSetWithTemplate>(
        proc("vkUpdateDescriptorSetWithTemplate", "vkUpdateDescriptorSetWithTemplateKHR"));
    if (!create || !update || l.template_failed || writes.empty()) return false;
    std::vector<uint32_t>& signature = template_signature_;
    signature.clear();
    for (const VkWriteDescriptorSet& w : writes) {
        if (w.descriptorCount != 1 || (!w.pBufferInfo && !w.pImageInfo)) return false;
        signature.push_back(w.dstBinding << 8 | static_cast<uint32_t>(w.descriptorType));
    }
    if (!l.update_template) {
        std::vector<VkDescriptorUpdateTemplateEntry> entries;
        for (size_t k = 0; k < writes.size(); k++)
            entries.push_back({writes[k].dstBinding, 0, 1, writes[k].descriptorType, k * kEntry, kEntry});
        VkDescriptorUpdateTemplateCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO};
        ci.descriptorUpdateEntryCount = static_cast<uint32_t>(entries.size());
        ci.pDescriptorUpdateEntries = entries.data();
        ci.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;
        ci.descriptorSetLayout = l.set;
        if (VkResult r = create(vk()->device, &ci, nullptr, &l.update_template); r != VK_SUCCESS) {
            static int logged = 0;
            if (logged++ < 4) NVN_LOG("vkCreateDescriptorUpdateTemplate (%zu descriptores): %s\n", entries.size(), vk_result_name(r));
            l.update_template = VK_NULL_HANDLE;
            l.template_failed = true;
            return false;
        }
        l.template_signature = signature;
    }
    if (signature != l.template_signature) return false;  // otros descriptores que la primera vez (una imagen sin vista)
    template_data_.resize(writes.size() * kEntry);
    for (size_t k = 0; k < writes.size(); k++) {
        const VkWriteDescriptorSet& w = writes[k];
        memcpy(template_data_.data() + k * kEntry,
               w.pBufferInfo ? static_cast<const void*>(w.pBufferInfo) : static_cast<const void*>(w.pImageInfo), kEntry);
    }
    update(vk()->device, set, l.update_template, template_data_.data());
    return true;
}

void Executor::log_draw(const cmd::Draw& d) {
    NVN_LOG("draw %llu: primitiva %d, %d %s\n", static_cast<unsigned long long>(draws_), d.primitive, d.count,
            d.index_type >= 0 ? "índices" : "vértices");
    for (int s = 0; s < kStages; s++) {
        const ProgramStage* ps = st_.stages[s];
        if (!ps) continue;
        std::string want, bound;
        for (const Resource& r : ps->shader.resources)
            want += r.kind == Resource::UniformBuffer
                        ? " c" + std::to_string(r.cbuf_index) + "(" + std::to_string(ps->shader.constant_buffer_used[r.cbuf_index]) + ")"
                        : " tex@" + std::to_string(r.cbuf_offset);
        const StageBindings& b = st_.bindings[s];
        for (int k = 0; k < kMaxUniforms; k++)
            if (b.uniforms[k].size) bound += " ubo" + std::to_string(k) + "(" + std::to_string(b.uniforms[k].size) + ")";
        for (int k = 0; k < kMaxTextures; k++) {
            if (b.textures[k] || b.samplers[k]) bound += " tex" + std::to_string(k);
            if (b.combined_textures[k]) bound += " comb" + std::to_string(k);
        }
        for (int k = 0; k < kMaxStorage; k++)
            if (b.storage[k].size) {
                char sb[64];
                snprintf(sb, sizeof sb, " s%d %#llx+%llu", k, static_cast<unsigned long long>(b.storage[k].address),
                         static_cast<unsigned long long>(b.storage[k].size));
                bound += sb;
            }
        NVN_LOG("  %s %016llx pide:%s | enlazado:%s\n", nvnshader::stage_name(static_cast<Stage>(s)),
                static_cast<unsigned long long>(ps->print), want.c_str(), bound.c_str());
    }
}

void Executor::trace_draw(const cmd::Draw& d) {
    std::string rts;
    for (int k = 0; k < st_.rt.count; k++)
        if (Texture* t = st_.rt.colors[k])
            rts += " " + std::to_string(k) + ":" + std::to_string(t->desc.width) + "x" + std::to_string(t->desc.height) +
                   "/0x" + [&] { char b[32]; snprintf(b, sizeof b, "%x@%llx", t->desc.format, (unsigned long long)(uintptr_t)t->storage); return std::string(b); }();
    if (st_.rt.depth)
        rts += " d:" + [&] { char b[48]; snprintf(b, sizeof b, "%dx%d/0x%x@%llx", st_.rt.depth->desc.width, st_.rt.depth->desc.height,
                                                  st_.rt.depth->desc.format, (unsigned long long)(uintptr_t)st_.rt.depth->storage); return std::string(b); }();
    // La capa de la vista de profundidad (las cascadas de las sombras van a capas de un array: sin esto no se ve cuál).
    if (st_.rt.depth && (st_.rt.depth_view.flags & kViewLayers))
        rts += " capa " + std::to_string(st_.rt.depth_view.base_layer);
    const auto& v = st_.viewport;
    const auto& ds = st_.depth_stencil;
    NVN_LOG("F draw prim %d n %d inst %d idx %d | rt[%d]%s depth %p | vp %.0f,%.0f %.0fx%.0f z %.2f-%.2f | sc %s %d,%d %ux%u\n",
            d.primitive, d.count, d.instances, d.index_type, st_.rt.count, rts.c_str(), static_cast<void*>(st_.rt.depth),
            v.x, v.y, v.width, v.height, v.minDepth, v.maxDepth, st_.scissor_set ? "on" : "off", st_.scissor.offset.x,
            st_.scissor.offset.y, st_.scissor.extent.width, st_.scissor.extent.height);
    NVN_LOG("F   masks 0x%08x blend_en 0x%x b0 %d/%d/%d/%d eq %d | depth t%d w%d f%d sten %d | cull %d front %d mode %d | attribs %zu streams %zu | reinicio %d 0x%x | ms %d a2c %d atest %d\n",
            st_.masks.masks, st_.color.blend_enable, st_.blend[0].src_rgb, st_.blend[0].dst_rgb, st_.blend[0].src_alpha,
            st_.blend[0].dst_alpha, st_.blend[0].eq_rgb, int(ds.depth_test), int(ds.depth_write), int(ds.depth_func),
            int(ds.stencil_test), st_.polygon.cull_face, st_.polygon.front_face, st_.polygon.polygon_mode,
            st_.attribs.size(), st_.streams.size(), int(st_.primitive_restart), st_.restart_index,
            int(st_.multisample.samples), int(st_.multisample.alpha_to_coverage), int(st_.color.alpha_test));
    for (size_t k = 0; k < st_.attribs.size(); k++)
        NVN_LOG("F   attrib %zu: formato 0x%x stream %d offset %d\n", k, st_.attribs[k].format, st_.attribs[k].stream,
                st_.attribs[k].offset);
    for (size_t k = 0; k < st_.streams.size(); k++) {
        const BufferRange& vb = st_.vertex_buffers[k];
        std::string vals;
        char b[24];
        if (vb.address)
            for (size_t i = 0; i < std::min<size_t>(vb.size / 4, 16); i++)
                snprintf(b, sizeof b, " %.3g", reinterpret_cast<const float*>(vb.address)[i]), vals += b;
        // ¿Lo escribió la GPU? Los vértices que calcula un compute (las cabezas con skinning en compute: la del cazador,
        // la de Hinoa) viven en un overlay y la memoria del juego sigue a cero: entonces se muestra lo de la GPU. Se mira
        // sin tocarlo (find_overlay puede tirarlo si la CPU reescribió el rango; eso lo decide el draw).
        std::string from = "memoria";
        if (vb.address && vb.size) {
            auto it = overlays_.upper_bound(vb.address);
            if (it != overlays_.begin() && std::prev(it)->first + std::prev(it)->second.size >= vb.address + vb.size) {
                const uint64_t start = std::prev(it)->first;
                const Overlay& o = std::prev(it)->second;
                char ob[64];
                snprintf(ob, sizeof ob, "overlay %#llx+%llu", static_cast<unsigned long long>(start),
                         static_cast<unsigned long long>(o.size));
                from = ob;
                if (o.data) {
                    vals.clear();
                    const auto* f = reinterpret_cast<const float*>(o.data + (vb.address - start));
                    for (size_t i = 0; i < std::min<size_t>(vb.size / 4, 16); i++) snprintf(b, sizeof b, " %.3g", f[i]), vals += b;
                }
            }
        }
        NVN_LOG("F   stream %zu: stride %u divisor %u | vb %#llx (%llu bytes, %s):%s\n", k, st_.streams[k].stride,
                st_.streams[k].divisor, static_cast<unsigned long long>(vb.address),
                static_cast<unsigned long long>(vb.size), from.c_str(), vals.c_str());
    }
    if (d.index_type >= 0 && d.index_address) {
        std::string vals;
        for (int i = 0; i < std::min(d.count, 8); i++) {
            uint32_t v = d.index_type == 0   ? reinterpret_cast<const uint8_t*>(d.index_address)[i]
                         : d.index_type == 1 ? reinterpret_cast<const uint16_t*>(d.index_address)[i]
                                             : reinterpret_cast<const uint32_t*>(d.index_address)[i];
            vals += " " + std::to_string(v);
        }
        NVN_LOG("F   índices (%#llx):%s first %d base_vertex %d\n", static_cast<unsigned long long>(d.index_address),
                vals.c_str(), d.first, d.base_vertex);
    }
}

// DEPURACIÓN: el valor de una variable de traza. Un fichero <datos>/<VARIABLE> con el valor la pone en vivo, sin
// reiniciar (se lee una vez por frame trazado); si no hay fichero, la del entorno.
const char* debug_var(const char* var, uint64_t frame) {
    struct Cached {
        uint64_t frame = ~0ull;
        bool from_file = false;
        std::string value;
    };
    static std::unordered_map<std::string, Cached> cache;
    Cached& c = cache[var];
    if (c.frame != frame) {
        c.frame = frame;
        const char* data = getenv("NIER_DATA");
        std::ifstream in(std::string(data ? data : ".") + "/" + var);
        c.from_file = static_cast<bool>(in);
        if (c.from_file) {
            std::getline(in, c.value);
            while (!c.value.empty() && (c.value.back() == '\r' || c.value.back() == ' ')) c.value.pop_back();
        }
    }
    return c.from_file ? c.value.c_str() : getenv(var);
}

int Executor::trace_cap() const {
    const char* v = debug_var("NIER_TRAZA_MAX", trace_frame_);
    return v && *v ? atoi(v) : 1500;
}

// DEPURACIÓN: ¿el draw trazado actual está en la lista "a,b,c" de esa variable de entorno?
bool Executor::traced_in_list(const char* var) const {
    const char* v = debug_var(var, trace_frame_);
    if (!v) return false;
    std::string list = std::string(",") + v + ",";
    return list.find("," + std::to_string(traced_draws_) + ",") != std::string::npos;
}

// Al empezar un draw o un dispatch (nadie guarda entradas de la caché ni datos del envío a medias).
void Executor::maybe_split() {
    if (!recording_ || !staging_) return;
    // Caché de vértices llena: con generaciones se sigue en otra, sin partir el envío ni esperar a la GPU.
    if (generations_) buffer_cache_.next_generation(frame_, submitted_.load(std::memory_order_relaxed) + 1);
    // Tope de RAM por envío: si el staging del envío actual pasa de esto, se manda a la GPU y se sigue en otro,
    // en vez de seguir creciendo (con mucho que copiar en un frame llegó a agotar la RAM del sistema). Lo
    // mismo con la caché de vértices sin generaciones: solo se vacía entre envíos.
    const bool staging_full = staging_->bytes() >= kStagingSoftCap;
    const bool cache_full = !generations_ && buffer_cache_.needs_reset(frame_);
    // Bastantes draws desde el último envío y el render pass cerrado (o a punto de cerrarse: cambian los render
    // targets): se manda lo grabado y la GPU empieza. Partir con el pass abierto obligaría a guardarlo y cargarlo.
    const bool many_draws = split_ && draws_ - submit_draws_ >= kSplitDraws && (!in_render_pass_ || targets_changed_);
    if (!staging_full && !cache_full && !many_draws) return;
    static int logged = 0;
    if ((staging_full || cache_full) && logged++ < 5)
        NVN_LOG("%s en un envío: se parte\n", staging_full ? "staging lleno" : "caché de vértices llena");
    split_why_ = many_draws && !staging_full && !cache_full ? "envío partido (draws)" : "envío partido (staging o caché)";
    end_render_pass();
    targets_changed_ = true;
    splitting_ = true;
    finish(false);
    splitting_ = false;
    begin();
}

// Un draw de más de 2^28 vértices (o 2^24 por instancia, o 2^20 instancias) no lo hace ningún juego: la GPU estaría
// segundos con él y el kernel la daría por colgada (kgsl: gpu timeout, VK_ERROR_DEVICE_LOST).
static bool absurd_draw(uint32_t count, uint32_t instances) {
    return count > (1u << 24) || instances > (1u << 20) || uint64_t{count} * instances > (1ull << 28);
}

// Earlier draws can still reference the old staging allocation. Only reuse its
// unchanged prefix; a longer or changed range gets an immutable snapshot.
Staging::Alloc Executor::graphics_c0(VkDeviceSize size, VkDeviceSize alignment) {
    if (size <= graphics_c0_valid_) return graphics_c0_;
    auto alloc = staging_->alloc(size, alignment);
    if (!alloc.data) return alloc;
    memset(alloc.data, 0, size);
    for (uint32_t at = kGraphicsStorageBase; at + 12 <= size; at += 16) {
        const auto slot = at - kGraphicsStorageBase;
        const auto& range = st_.bindings[slot / kGraphicsStorageStride].storage[(slot % kGraphicsStorageStride) / 16];
        const auto bytes = static_cast<uint32_t>(range.size);
        memcpy(alloc.data + at, &range.address, 8);
        memcpy(alloc.data + at + 8, &bytes, 4);
    }
    graphics_c0_ = alloc;
    graphics_c0_valid_ = size;
    prof_copied_bytes_ += size;
    return alloc;
}

void Executor::draw(const cmd::Draw& in) {
    if (!st_.render_enabled) {
        if (frame_ == trace_frame_) prof_disabled_draws_++;
        return;
    }
    if (st_.pending_stages) {  // a shader of this program is still translating in the background (translate_or_queue)
        g_drop_stats.draws_skipped.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    maybe_split();
    const VkContext& context = *vk();
    if (shadow_pass() && NVN_FLAG("no_shadows")) {
        if (!shadow_skip_logged_) {
            shadow_skip_logged_ = true;
            NVN_LOG("shadows off (<datos>/no_shadows): skipping shadow map draws (%ux%u, %u layers)\n",
                    st_.rt.depth->desc.width, st_.rt.depth->desc.height, st_.rt.depth->layers);
        }
        return;
    }
    cmd::Draw d = in;
    if (d.indirect) note_read(4, d.indirect, 20);
    // Argumentos escritos por la GPU (p. ej. el geometry shader del pre-pase de NieR compacta los triángulos
    // visibles y suma su número aquí): el draw se hace con vkCmdDraw*Indirect desde ese buffer.
    // Descarte de triángulos de NieR: el geometry shader del pre-pase compacta los triángulos visibles con
    // operaciones de warp (ballot, shuffle) que el Adreno no tiene en geometría; la emulación de Eden (cada
    // hilo como si estuviera solo) no sirve para un algoritmo cooperativo y la lista salía con triángulos
    // repetidos, índices fuera de rango y huecos (piezas negras). Se dibuja la malla original del pre-pase:
    // la rasterización descarta lo mismo (caras traseras, fuera de pantalla, sin píxeles).
    if (d.indirect && d.index_type >= 0 && !d.draw_count)
        if (auto it = culled_draws_.find(d.indirect); it != culled_draws_.end() && frame_ - it->second.frame <= 1) {
            const cmd::Draw& o = it->second.draw;
            d.index_address = o.index_address;
            d.index_type = o.index_type;
            d.count = o.count;
            d.first = o.first;
            d.base_vertex = o.base_vertex;
            d.base_instance = 0;
            d.instances = 1;
            d.indirect = 0;
        }
    VkBuffer gpu_args = VK_NULL_HANDLE;
    VkDeviceSize gpu_args_offset = 0;
    // Multi-draw: los argumentos de max_draws draws (en la GPU o, si no, copiados al staging) y su número (en la GPU,
    // o leído aquí). Los índices y vértices de cada draw no se conocen: los streams van enteros.
    const bool multi = d.draw_count != 0;
    VkBuffer count_buf = VK_NULL_HANDLE;
    VkDeviceSize count_offset = 0;
    uint32_t multi_draws = 0;  // con el número en la CPU
    const uint64_t one_args = d.index_type >= 0 ? 20 : 16;
    const uint64_t args_size = multi ? uint64_t(d.stride) * uint64_t(d.max_draws - 1) + one_args : one_args;
    // Without VK_KHR_draw_indirect_count (ARM's Vulkan 1.1 drivers, such as r32 on a Dimensity 720) the count can't be
    // read on the GPU. MH Rise draws almost its whole main view this way, so skipping them left only the sky: the count
    // is taken on the CPU instead (see below). <data>/no_count_fallback: skipped, as before.
    // Teselación: Vulkan la quiere con las dos etapas, parches como primitiva y la feature del driver. Un draw que no
    // cumple no se dibuja (un pipeline así no es válido, y un driver puede caerse al crearlo).
    if (const bool tcs = st_.stages[static_cast<int>(Stage::TessControl)], tes = st_.stages[static_cast<int>(Stage::TessEval)];
        tcs || tes || d.primitive == 14) {
        if (!tcs || !tes || d.primitive != 14 || !context.tessellation) {
            static int logged = 0;
            if (logged++ < 4)
                NVN_LOG("draw con teselación sin dibujar: control %d, evaluación %d, primitiva %d, feature %d\n", tcs, tes,
                        d.primitive, context.tessellation);
            skip_draw(0);  // (tess)
            return;
        }
    }
    const bool count_fallback = multi && !context.draw_indexed_indirect_count && !NVN_FLAG("no_count_fallback");
    if (multi && ((!context.draw_indexed_indirect_count && !count_fallback) || d.stride < static_cast<int>(one_args) ||
                  d.stride % 4 || d.indirect % 4 || d.draw_count % 4)) {
        static int logged = 0;
        if (logged++ < 4)
            NVN_LOG("multi-draw no soportado (stride %d, args %#llx, número %#llx)\n", d.stride,
                    static_cast<unsigned long long>(d.indirect), static_cast<unsigned long long>(d.draw_count));
        skip_draw(1);  // (multi)
        return;
    }
    const uint8_t* args_host = nullptr;   // DEPURACIÓN (migas): los argumentos vistos desde la CPU
    const uint8_t* count_host = nullptr;
    if (d.indirect) {
        if (const Overlay* o = find_overlay(d.indirect, args_size, &gpu_args_offset)) {
            gpu_args = o->buffer;
            if (o->data) args_host = o->data + (gpu_args_offset - o->offset);
        } else if (!gather_overlays(d.indirect, args_size, &gpu_args, &gpu_args_offset, &args_host)) {
            skip_draw(10);
            return;
        }
    }
    // Draw count check (count_check.h): the vertex stage with it stands in for the game's until draw() returns
    struct VertexSwap {
        QueueState& st;
        const ProgramStage* saved = nullptr;
        ~VertexSwap() {
            if (saved) st.stages[static_cast<int>(Stage::Vertex)] = saved;
        }
    } vertex_swap{st_};
    VkBuffer check_buf = VK_NULL_HANDLE;  // the count it reads, and where
    VkDeviceSize check_offset = 0;
    if (multi) {
        if (const Overlay* o = find_overlay(d.draw_count, 4, &count_offset)) {
            // Without draw-indirect-count, a count no GPU work wrote is the game's own, set on the CPU: read from its
            // memory (below, exactly) rather than drawing every draw up to max_draws
            if (o->writers || !count_fallback) {
                count_buf = o->buffer;
                if (o->data) count_host = o->data + (count_offset - o->offset);
            }
        } else if (!gather_overlays(d.draw_count, 4, &count_buf, &count_offset, &count_host)) {
            skip_draw(10);
            return;
        }
        // The count stays on the GPU and the vertex shader drops the draws past it (count_check.h): one call per draw
        // (no multiDrawIndirect, so each knows its index), a plain vertex stage (a geometry or tessellation one could
        // make vertices of its own). On by default there; <data>/no_count_check turns it off, <data>/count_check forces it
        // (Graphics → Exact draw counts).
        const ProgramStage* game_vs = st_.stages[static_cast<int>(Stage::Vertex)];
        if (count_fallback && count_buf && (NVN_FLAG("count_check") || !NVN_FLAG("no_count_check")) && !context.multi_draw_indirect && game_vs &&
            !st_.stages[static_cast<int>(Stage::Geometry)] && !st_.stages[static_cast<int>(Stage::TessControl)] &&
            !st_.stages[static_cast<int>(Stage::TessEval)])
            if (const ProgramStage* checked = count_check_stage(*game_vs)) {
                vertex_swap.saved = game_vs;
                st_.stages[static_cast<int>(Stage::Vertex)] = checked;
                check_buf = count_buf;
                check_offset = count_offset;
            }
        if (count_fallback && count_buf) {
            // The count is on the GPU and can't be used there: every draw is drawn. The count is the game's GPU culling
            // (about 7 % of these come back 0 in the village), and the arguments of a culled draw are still a valid
            // mesh, so this only draws a few hidden things. Taking last frame's count instead flickered: it lags the
            // arguments by a frame.
            multi_draws = static_cast<uint32_t>(d.max_draws);
            count_buf = VK_NULL_HANDLE;
            static bool logged = false;
            if (!logged) {
                logged = true;
                NVN_LOG("multi-draw with the count on the GPU and no VK_KHR_draw_indirect_count: every draw is drawn\n");
            }
        } else if (!count_buf) {
            multi_draws = std::min<uint32_t>(*reinterpret_cast<const uint32_t*>(d.draw_count), static_cast<uint32_t>(d.max_draws));
            if (!multi_draws) return;
        }
        const bool cpu_args = !gpu_args;
        if (cpu_args) {
            Staging::Alloc al = staging_->alloc(args_size, 16);
            if (!al.data) return;
            memcpy(al.data, reinterpret_cast<const void*>(d.indirect), args_size);
            gpu_args = al.buffer;
            gpu_args_offset = al.offset;
            args_host = al.data;
            // Conteos imposibles (memoria reutilizada, argumentos a medio escribir): colgarían la GPU segundos. Fuera.
            for (int k = 0; k < d.max_draws; k++) {
                auto* e = reinterpret_cast<uint32_t*>(al.data + uint64_t(k) * d.stride);
                if (!absurd_draw(e[0], e[1])) continue;
                static int logged_multi = 0;
                if (logged_multi++ < 16)
                    NVN_LOG("multi-draw con argumentos imposibles (draw %d de %d en %#llx): %u x %u, devuelto %#llx\n", k,
                            d.max_draws, static_cast<unsigned long long>(d.indirect), e[0], e[1],
                            static_cast<unsigned long long>(delivered_range(d.indirect + uint64_t(k) * d.stride)));
                e[0] = e[1] = 0;
            }
        }
        if (frame_ == trace_frame_) {  // DEPURACIÓN: lo que dejó el frame anterior en los argumentos y el número
            VkDeviceSize ao = 0, co = 0;
            const Overlay* aa = cpu_args ? nullptr : find_overlay(d.indirect, args_size, &ao);
            const Overlay* ca = count_buf ? find_overlay(d.draw_count, 4, &co) : nullptr;
            const auto* a = reinterpret_cast<const uint32_t*>(aa && aa->data ? aa->data + (ao - aa->offset)
                                                                            : reinterpret_cast<const uint8_t*>(d.indirect));
            std::string args;
            char b[64];
            for (int k = 0; k < std::min(d.max_draws, 3); k++) {
                const uint32_t* e = a + k * d.stride / 4;
                snprintf(b, sizeof b, " [%u %u %u %d %u]", e[0], e[1], e[2], static_cast<int>(e[3]), e[4]);
                args += b;
            }
            NVN_LOG("F   multi-draw: número antes %u, argumentos antes%s\n",
                    ca && ca->data ? *reinterpret_cast<const uint32_t*>(ca->data + (co - ca->offset)) : multi_draws, args.c_str());
        }
        if (frame_ == trace_frame_)  // DEPURACIÓN
            NVN_LOG("F multi-draw %d draws máx. (stride %d), argumentos %s %#llx, número %s %#llx%s, índices %d en %#llx\n",
                    d.max_draws, d.stride, cpu_args ? "de la CPU" : "de la GPU", static_cast<unsigned long long>(d.indirect),
                    count_buf ? "de la GPU" : "de la CPU", static_cast<unsigned long long>(d.draw_count),
                    count_buf ? "" : (" = " + std::to_string(multi_draws)).c_str(), d.index_type,
                    static_cast<unsigned long long>(d.index_address));
        d.count = 0;
        d.indirect = 0;
    }
    if (gpu_args && !multi) {
        if (frame_ == trace_frame_) {
            NVN_LOG("F indirecto en la GPU %#llx (índices %d en %#llx)\n", static_cast<unsigned long long>(d.indirect),
                    d.index_type, static_cast<unsigned long long>(d.index_address));
            VkDeviceSize io = 0, ao = 0;
            const Overlay* ia = find_overlay(d.index_address, 4, &io);
            const Overlay* aa = find_overlay(d.indirect, 20, &ao);
            if (ia && aa && ia->data && aa->data)
                traced_gpu_draws_.push_back({d.indirect, aa->data + (ao - aa->offset), ia->data + (io - ia->offset), d.index_type});
        }
        d.count = 0;  // no se sabe en la CPU
        d.indirect = 0;
    } else if (d.indirect) {  // {count, instances, first(índice), base_vertex, base_instance} o {count, instances, first, base_instance}
        const auto* a = reinterpret_cast<const uint32_t*>(d.indirect);
        if (frame_ == trace_frame_)  // DEPURACIÓN
            NVN_LOG("F indirecto %#llx: %u %u %u %u %u (índices %d en %#llx)\n", static_cast<unsigned long long>(d.indirect), a[0],
                    a[1], a[2], a[3], a[4], d.index_type, static_cast<unsigned long long>(d.index_address));
        d.count = static_cast<int>(a[0]);
        d.instances = static_cast<int>(a[1]);
        if (d.index_type >= 0) {
            d.index_address += uint64_t{a[2]} * index_size(d.index_type);
            d.base_vertex = static_cast<int32_t>(a[3]);
            d.base_instance = static_cast<int>(a[4]);
        } else {
            d.first = static_cast<int>(a[2]);
            d.base_instance = static_cast<int>(a[3]);
        }
        d.indirect = 0;
        if (!d.count || !d.instances) return;
        if (absurd_draw(a[0], a[1])) {
            static int logged_absurd = 0;
            if (logged_absurd++ < 16)
                NVN_LOG("indirecto con argumentos imposibles en %#llx: %u x %u, devuelto %#llx\n",
                        static_cast<unsigned long long>(in.indirect), a[0], a[1],
                        static_cast<unsigned long long>(delivered_range(in.indirect)));
            skip_draw(2);  // (absurd_ind)
            return;
        }
    } else if (!gpu_args && !multi && absurd_draw(static_cast<uint32_t>(d.count), static_cast<uint32_t>(d.instances))) {
        static int logged_direct = 0;
        if (logged_direct++ < 16) NVN_LOG("draw directo imposible: %d x %d\n", d.count, d.instances);
        skip_draw(3);  // (absurd_dir)
        return;
    }
    draws_++;
    if (crumbs_on_ && crumb_id_) {
        Crumb& cr = crumbs_[crumb_id_ % kCrumbs];
        if (gpu_args && cr.gpu_draws.size() < 32)
            cr.gpu_draws.push_back({args_host, count_buf ? count_host : nullptr, in.indirect, multi ? uint32_t(d.stride) : 0,
                                    multi ? uint32_t(d.max_draws) : 1, multi_draws, d.index_type});
        crumbs_[crumb_id_ % kCrumbs].draws++;
        crumb_stage(st_.stages[static_cast<int>(Stage::Vertex)]);
        crumb_stage(st_.stages[static_cast<int>(Stage::Fragment)]);
    }
    if (draws_ <= kLoggedDraws) log_draw(d);
    // DEPURACIÓN: los draws con argumentos de la GPU (balas, partículas) se trazan siempre, con sus shaders.
    const bool gpu_counted = gpu_args != VK_NULL_HANDLE;
    if (frame_ == trace_frame_ && (traced_draws_++ < trace_cap() || gpu_counted)) {
        trace_draw(d);
        log_draw(d);
        const bool half_res = st_.rt.count > 0 && st_.rt.colors[0] && st_.rt.colors[0]->desc.width == 640;
        const char* from = debug_var("NIER_TRAZA_DESDE", trace_frame_);
        const bool dump_code = gpu_counted || traced_draws_ <= 40 || half_res ||
                               (from && *from && traced_draws_ >= atoi(from) && traced_draws_ < atoi(from) + 40) ||
                               traced_in_list("NIER_TRAZA_SHADERS");  // DEPURACIÓN: "a,b,c": los shaders de esos draws
        // <datos>/full_trace: esta vez con shaders y render targets aunque NIER_TRAZA_LIGERA esté puesta.
        const char* light = NVN_FLAG("full_trace") ? nullptr : getenv("NIER_TRAZA_LIGERA");
        for (int st = 0; st < kStages && dump_code && (gpu_counted || !(light && *light)); st++)  // escribir en la SD es lento: solo unos pocos
            if (st_.stages[st])
                dump_stage_code(*st_.stages[st], "traza" + std::to_string(traced_draws_) + "_" +
                                                     nvnshader::stage_name(static_cast<Stage>(st)));
        // DEPURACIÓN: NIER_TRAZA_OBJ=<zancada>: el registro del objeto que lee el vertex shader (SSBO 0, índice en
        // el primer u32 de su uniform buffer 1), p. ej. las esferas de las balas (160 bytes por objeto).
        if (const char* obj = getenv("NIER_TRAZA_OBJ"); obj && *obj) {
            const StageBindings& vb = st_.bindings[static_cast<int>(Stage::Vertex)];
            const uint32_t stride = static_cast<uint32_t>(atoi(obj));
            if (vb.uniforms[1].address && vb.storage[0].address && stride) {
                const uint32_t index = *reinterpret_cast<const uint32_t*>(vb.uniforms[1].address);
                const uint64_t at = uint64_t{index} * stride;
                if (at + stride <= vb.storage[0].size) {
                    VkDeviceSize ov_offset = 0;
                    const Overlay* ov = find_overlay(vb.storage[0].address + at, stride, &ov_offset);
                    const auto* f = reinterpret_cast<const float*>(ov && ov->data ? ov->data + (ov_offset - ov->offset)
                                                                                   : reinterpret_cast<const uint8_t*>(vb.storage[0].address + at));
                    std::string vals;
                    char b[24];
                    for (uint32_t i = 0; i < stride / 4 && i < 40; i++) snprintf(b, sizeof b, " %.4g", f[i]), vals += b;
                    NVN_LOG("F   objeto %u (%s):%s\n", index, ov ? "GPU" : "memoria", vals.c_str());
                }
            }
        }
        for (int st = 0; st < kStages; st++) {  // primeros floats de cada uniform buffer enlazado
            if (!st_.stages[st]) continue;
            for (int k = 0; k < kMaxUniforms; k++) {
                const BufferRange& u = st_.bindings[st].uniforms[k];
                if (!u.size) continue;
                const float* f = reinterpret_cast<const float*>(u.address);
                std::string vals;
                char b[24];
                for (size_t i = 0; i < std::min<size_t>(u.size / 4, 16); i++) snprintf(b, sizeof b, " %.3g", f[i]), vals += b;
                NVN_LOG("F   %s ubo%d (%llu bytes):%s\n", nvnshader::stage_name(static_cast<Stage>(st)), k,
                        static_cast<unsigned long long>(u.size), vals.c_str());
            }
        }
    }
    const ProgramStage* vs = st_.stages[static_cast<int>(Stage::Vertex)];
    if (!vs) {
        skip_draw(4);  // (no_vs)
        return;
    }
    auto prof_t = prof_now();  // perfil por secciones (solo en los frames medidos)
    const uint32_t vertex_inputs = vs->shader.input_generics & 0xffff;
    // 1. Texturas: crear y subir lo que falte (las subidas van fuera del render pass).
    static std::vector<BoundImage> textures;
    textures.clear();
    if (loading_armed_) loading_matches_.clear();
    // Para la barrera aplazada (nvn_barrier): ¿escribe storage este draw? ¿muestrea un adjunto del pase abierto?
    // writes_target (merged_passes): escribe como imagen un adjunto del pase (uno de los añadidos: el juego no lo tiene de
    // render target aquí); el pase se corta antes, si no lo que guardara al final pisaría lo escrito.
    bool writes_storage = false, samples_target = false, writes_target = false;
    for (int s = 0; s < kStages; s++) {
        if (!st_.stages[s]) continue;
        for (const Resource& r : st_.stages[s]->shader.resources) {
            writes_storage |= shader_writes_storage(r);
            if (r.kind != Resource::Texture && r.kind != Resource::Image) continue;
            if (r.set) continue;  // arrays bindless: bindless_set
            BoundImage b{&r, static_cast<Stage>(s)};
            if (!prepare_shader_image(*st_.stages[s], b)) { skip_draw(8); return; }
            // Muestrear un adjunto del pase abierto: con una barrera aplazada o con el pase que sigue tras volver a
            // enlazar los mismos targets (rt_igual), el pase se corta, como cortaba antes el SetRenderTargets.
            // merged_passes: el pase sigue tras cambiar de targets, como con rt_igual.
            if (b.tex && in_render_pass_ && (pending_barrier_ || opt_.rt_igual || merge_passes_))
                samples_target = samples_target || pass_overlaps(b, opt_.solo_escritos);
            if (merge_passes_ && b.tex && in_render_pass_ && r.kind == Resource::Image && r.is_written && pass_overlaps(b, false))
                writes_target = true;
            textures.push_back(b);
        }
    }
    prof_mark(0, prof_t);
    auto prof_s = prof_t;  // los tramos de "pipeline" (prof_split)
    for (auto& [tex, id] : tex_dumps_)  // DEPURACIÓN (NIER_VOLCAR_TEX), ya subidas
        if (tex) {
            split_why_ = "volcado (depuración)";
            end_render_pass();
            targets_changed_ = true;
            if (tex->desc.target == kTarget3D || tex->layers > 1) dump_volume(*tex, "vol" + std::to_string(id));
            if (tex->layers > 1) {  // cubemaps y arrays: cada capa, en el mip 0 y en uno intermedio
                for (uint32_t l = 0; l < std::min<uint32_t>(tex->layers, 6); l++) {
                    dump(*tex, "tex" + std::to_string(id) + "_c" + std::to_string(l) + "_m0", l, 0);
                    dump(*tex, "tex" + std::to_string(id) + "_c" + std::to_string(l) + "_m" + std::to_string(tex->desc.levels / 2), l,
                         tex->desc.levels / 2);
                }
            } else {
                dump(*tex, "tex" + std::to_string(id));
            }
            begin();
            tex = nullptr;
        }
    // Overlays de lo que escribe el draw (y de los SSBO que lee y caen solo en parte en lo de la GPU), unidos antes del
    // pase: dentro, make_overlay no puede copiar nada en la GPU.
    prepare_draw_overlays();
    // Espejos de los SSBO grandes que solo se leen: se revisan (y suben) antes de empezar el pase. Con un multi-draw,
    // también la ventana de sus índices.
    if (in.draw_count && (context.draw_indexed_indirect_count || !NVN_FLAG("no_count_fallback"))) {
        VkDeviceSize off = 0;
        uint64_t base = 0, size = 0;
        if (in.index_type >= 1 && in.index_address && index_window(in.index_address, &base, &size)) mirror(base, size, &off);
    }
    for (int s = 0; s < kStages; s++) {
        if (!st_.stages[s]) continue;
        const uint32_t base = kGraphicsStorageBase + kGraphicsStorageStride * s;
        for (const Resource& r : st_.stages[s]->shader.resources) {
            if (r.kind != Resource::StorageBuffer || r.is_written || r.cbuf_index != 0 || r.cbuf_offset < base ||
                r.cbuf_offset >= base + 16 * kMaxStorage)
                continue;
            const BufferRange& range = st_.bindings[s].storage[(r.cbuf_offset - base) / 16];
            VkDeviceSize off = 0;
            if (range.size >= kMirrorMin && !find_overlay(range.address, std::min<uint64_t>(range.size, kStagingChunk), &off))
                mirror(range.address, range.size, &off);
        }
    }
    prof_split(0, prof_s);
    prof_mark(1, prof_t);
    auto prof_v = prof_t;
    const uint8_t* indices = d.index_type >= 0 ? reinterpret_cast<const uint8_t*>(d.index_address) : nullptr;
    const uint32_t isize = indices ? index_size(d.index_type) : 0;
    VkBuffer gpu_index = VK_NULL_HANDLE;  // índices que escribió la GPU
    VkDeviceSize gpu_index_offset = 0;
    if (indices) {
        const uint64_t bytes = uint64_t{isize} * std::max(d.count, 1);
        if (const Overlay* o = find_overlay(d.index_address, bytes, &gpu_index_offset)) {
            gpu_index = o->buffer;
        } else if (!gpu_args && !gather_overlays(d.index_address, bytes, &gpu_index, &gpu_index_offset)) {
            skip_draw(11);
            return;
        }
    }
    if ((gpu_index || gpu_args) && (d.primitive == 7 || d.index_type == 0)) {
        static int logged = 0;
        if (logged++ < 8)
            NVN_LOG("draw con datos de la GPU no soportado (primitiva %d, índices %d, args %d, índices en la GPU %d)\n",
                    d.primitive, d.index_type, gpu_args != VK_NULL_HANDLE, gpu_index != VK_NULL_HANDLE);
        skip_draw(10);  // (gpu_data)
        return;
    }
    if (indices && !gpu_index) note_read(1, d.index_address, uint64_t{isize} * d.count);
    // Multi-draw con los índices en la memoria del juego: de su ventana en espejo (revisada antes del pase).
    VkBuffer mirror_index = VK_NULL_HANDLE;
    VkDeviceSize mirror_index_offset = 0;
    if (multi && indices && !gpu_index && d.index_type >= 1) {
        uint64_t base = 0, size = 0;
        VkDeviceSize off = 0;
        if (index_window(d.index_address, &base, &size))
            if (const Mirror* m = mirror(base, size, &off, false)) {
                mirror_index = m->buffer;
                mirror_index_offset = off + (d.index_address - base);
            }
    }
    // Cuántos índices leer: los del draw; con el número en la GPU y los índices en la CPU (p. ej. quads de
    // balas o partículas, que la GPU cuenta) no se sabe: un rango generoso, hasta el final del memory pool.
    size_t index_bytes = size_t{isize} * d.count;
    if (gpu_args && indices && !gpu_index && !mirror_index) {
        size_t avail = 0;
        uint64_t pool_start = 0, pool_size = 0;
        if (pool_range(d.index_address, &pool_start, &pool_size)) avail = pool_start + pool_size - d.index_address;
        // Multi-draw: todas las mallas suelen compartir un buffer de índices (hasta lo que cabe en el staging).
        index_bytes = std::min<size_t>(avail, multi ? kStagingChunk : size_t{1} << 20) / isize * isize;
        if (frame_ == trace_frame_)
            NVN_LOG("F indirecto en la GPU con índices de la CPU: %zu bytes desde %#llx\n", index_bytes,
                    static_cast<unsigned long long>(d.index_address));
    }
    const BufferCache::Entry* cached_indices =
        indices && !gpu_index && d.primitive != 7 && index_bytes
            ? buffer_cache_.get(d.index_address, index_bytes, d.index_type, submit_index_)
            : nullptr;
    VkBuffer host_index = VK_NULL_HANDLE;  // ese rango, si cambia demasiado para la caché: copia directa
    VkDeviceSize host_index_offset = 0;
    if (gpu_args && indices && !gpu_index && !cached_indices && index_bytes && !mirror_index) {
        Staging::Alloc al = staging_->alloc(index_bytes, 16);
        if (!al.data) return;
        memcpy(al.data, indices, index_bytes);
        host_index = al.buffer;
        host_index_offset = al.offset;
    }
    if (gpu_args && indices && !gpu_index && !cached_indices && !host_index && !mirror_index) {
        skip_draw(11);  // (indices)
        return;
    }
    const VkBuffer cached_index_buffer = cached_indices ? cached_indices->buffer : VK_NULL_HANDLE;
    const VkDeviceSize cached_index_offset = cached_indices ? cached_indices->offset : 0;
    // GPU-generated arguments can supply both the first vertex and the base instance.
    const bool known_vertices = !gpu_args && !gpu_index;
    uint32_t first_vertex = 0;
    uint64_t vertex_count = uint32_t(d.count);
    int32_t vertex_offset = d.base_vertex;
    if (known_vertices && indices) {
        uint32_t lo, hi;
        if (cached_indices) {
            lo = cached_indices->min_index;
            hi = cached_indices->max_index;
        } else {
            BufferCache::index_range(indices, index_bytes, d.index_type, &lo, &hi);
        }
        if (lo > hi) lo = hi = 0;
        if (d.base_vertex >= 0) {
            first_vertex = lo + uint32_t(d.base_vertex);
            vertex_count = uint64_t{hi} - lo + 1;
            vertex_offset = static_cast<int32_t>(0u - lo);
        } else {
            vertex_count = uint64_t(std::max<int64_t>(0, int64_t{hi} + d.base_vertex + 1));
        }
    } else if (known_vertices) {
        first_vertex = uint32_t(d.first);
    }
    prof_split(4, prof_v);

    // A stream is shared by all its attributes. Its last fetch ends at the last
    // attribute's end, which may be on either side of the stride boundary.
    uint32_t extent[kMaxVertexBuffers]{};
    for (uint32_t used = vertex_inputs; used;) {
        const auto k = take_location(used);
        const auto a = k < st_.attribs.size() ? st_.attribs[k] : VertexAttribState{};
        if (a.stream >= kMaxVertexBuffers) continue;
        const FormatInfo* f = format_info(a.format);
        const uint32_t bytes = f && f->vk != VK_FORMAT_UNDEFINED ? f->bytes : 16;
        extent[a.stream] = std::max(extent[a.stream], a.offset + bytes);
    }
    struct VertexSource {
        VkBuffer buffer{};
        VkDeviceSize offset{};
    };
    VertexSource streams[kMaxVertexBuffers]{};
    VkBuffer bind_buf[16]{};
    VkDeviceSize bind_off[16]{};
    VkBuffer zero_buffer{};
    VkDeviceSize zero_offset{};
    bool gpu_vertices = false;
    for (uint32_t used = vertex_inputs; used;) {
        const auto k = take_location(used);
        const auto a = k < st_.attribs.size() ? st_.attribs[k] : VertexAttribState{};
        BufferRange range = a.stream < kMaxVertexBuffers ? st_.vertex_buffers[a.stream] : BufferRange{};
        const auto stream = a.stream < st_.streams.size() ? st_.streams[a.stream] : VertexStreamState{};
        const uint64_t skip = stream.divisor ? 0 : uint64_t{first_vertex} * stream.stride;
        if (!range.address || skip >= range.size || a.offset >= range.size - skip) {
            if (!zero_buffer) {
                auto empty = staging_->alloc(64, 16);
                if (!empty.data) return;
                memset(empty.data, 0, 64);
                zero_buffer = empty.buffer;
                zero_offset = empty.offset;
            }
            bind_buf[k] = zero_buffer;
            bind_off[k] = zero_offset;
            continue;
        }
        auto& source = streams[a.stream];
        if (!source.buffer) {
            range.address += skip;
            range.size -= skip;
            const bool known = stream.divisor ? !gpu_args : known_vertices;
            const uint64_t elements = stream.divisor ? uint64_t(uint32_t(d.base_instance)) + std::max(d.instances, 1) : vertex_count;
            if (known || !stream.stride) {
                // Compare before multiplying: a guest base instance plus count can exceed 32 bits.
                const uint64_t last = elements ? elements - 1 : 0;
                const uint64_t tail = std::min<uint64_t>(range.size, extent[a.stream]);
                if (!stream.stride || last <= (range.size - tail) / stream.stride)
                    range.size = last * stream.stride + tail;
            }
            bool from_gpu = false;
            if (!read_vertex_stream(range, !known, source.buffer, source.offset, from_gpu)) {
                skip_draw(10);
                return;
            }
            gpu_vertices |= from_gpu;
            if (vigilar_nan_ && from_gpu)
                check_overlay_vertices(range.address + a.offset, std::min<uint64_t>(range.size - a.offset,
                    uint64_t{std::max(stream.stride, 4u)} * 16), stream.stride, a.format, a.stream);
        }
        bind_buf[k] = source.buffer;
        bind_off[k] = source.offset + a.offset;
    }
    prof_split(5, prof_v);
    prof_mark(4, prof_t);
    prof_s = prof_t;
    // Arrays bindless (set 1): aquí, antes del pase, porque pueden subir texturas; después de los volcados de depuración
    // (que empiezan otro command buffer: un set de los pools de antes no vale en el nuevo).
    cmd::SetRenderTargets attachments = st_.rt;
    if (merge_passes_ && run_lists_) {
        if (in_render_pass_ && !targets_changed_) attachments = cur_pass_rt_;
        else {
            cmd::SetRenderTargets merged;
            if (merge_targets(&merged)) attachments = merged;
        }
    }
    for (BoundImage& b : textures) {
        if (!b.tex || (b.r->kind != Resource::Texture && b.r->is_written) || !b.tex->image) continue;
        b.tex = sample_source(*b.tex, b.view, b.r->kind == Resource::Image ? ViewUse::Storage : ViewUse::Sampled, attachments);
        if (!b.tex) { skip_draw(8); return; }
    }
    bool wants_bindless = false;
    for (int s = 0; s < kStages; s++) wants_bindless |= st_.stages[s] && st_.stages[s]->shader.has_bindless_set;
    Layout* early_layout = wants_bindless && fast_bindless_ ? &layout(st_.stages) : nullptr;  // su set 1 ya está en él
    if (early_layout && prof_timing_) fine_.bindless_fast++;
    const VkDescriptorSet bindless = wants_bindless ? [&] {
        LoadTimer timer(*this, kLdBindless);
        return bindless_set(st_.stages, early_layout ? early_layout->bindless : VK_NULL_HANDLE, &attachments);
    }() : VK_NULL_HANDLE;
    if (wants_bindless && !bindless) {
        skip_draw(5);  // (bindless)
        return;
    }
    prof_split(1, prof_s);
    // 2. Render pass, pipeline. Con una barrera aplazada, el pase se corta aquí si este draw la necesita.
    if (in_render_pass_ && ((pending_barrier_ && writes_storage) || samples_target || writes_target)) {
        split_why_ = writes_target    ? "escribe como imagen un adjunto del pase"
                     : samples_target ? "muestrea un adjunto escrito en el pase"
                                      : "storage tras una barrera aplazada";
        end_render_pass();
        targets_changed_ = true;
        prof_barrier_splits_++;
        if (samples_target) prof_split_sample_++;
        else prof_split_storage_++;
    }
    if (!ensure_render_pass(false, &attachments)) {
        skip_draw(6);  // (pass)
        return;
    }
    if (log_state_ && st_.rt.depth) {  // perfil de la escena (prof_depth_rts_): draws y anchos de viewport por profundidad
        DepthStat& ds = prof_depth_rts_[(static_cast<uint32_t>(st_.rt.depth->desc.width) << 16) |
                                        (static_cast<uint32_t>(st_.rt.depth->desc.height) & 0xFFFF)];
        const int w = static_cast<int>(st_.viewport.width);
        ds.draws++, ds.vp_min = std::min(ds.vp_min, w), ds.vp_max = std::max(ds.vp_max, w);
    }
    // Resolución dinámica (perfil): el viewport de la escena, la que tiene profundidad de 1280x720.
    if (st_.rt.depth && st_.rt.depth->desc.width == 1280 && st_.rt.depth->desc.height == 720) {
        scene_vp_w_ = std::max(scene_vp_w_, static_cast<int>(st_.viewport.width));
        scene_vp_h_ = std::max(scene_vp_h_, static_cast<int>(st_.viewport.height));
        if (st_.viewport.width >= 320) scene_vp_min_w_ = std::min(scene_vp_min_w_, static_cast<int>(st_.viewport.width));
    }
    prof_split(2, prof_s);
    Layout& l = early_layout ? *early_layout : layout(st_.stages);
    // DEPURACIÓN: NIER_SIN_BLEND="a,b": esos draws del frame trazado se dibujan sin mezcla (se ve su color tal cual).
    // NIER_SIN_DEPTH: esos, sin depth test.
    const auto saved_blend = st_.color.blend_enable;
    const auto saved_depth = st_.depth_stencil;
    if (frame_ == trace_frame_ && traced_in_list("NIER_SIN_BLEND")) st_.color.blend_enable = 0;
    if (frame_ == trace_frame_ && traced_in_list("NIER_SIN_DEPTH")) st_.depth_stencil.depth_test = false;
    if (depth_lequal_ && st_.depth_stencil.depth_func == kDepthEqual) st_.depth_stencil.depth_func = kDepthLequal;
    if (fast_depth_ && st_.depth_stencil.depth_test) {
        const uint64_t mesh = st_.vertex_buffers[0].address * 0x9E3779B97F4A7C15ull ^ d.index_address;
        if (rp_color_count_ == 0 && st_.depth_stencil.depth_write) {
            if (st_.stages[static_cast<int>(Stage::Fragment)]) {
                if (cutout_meshes_.size() > 65536) cutout_meshes_.clear();
                cutout_meshes_.insert(mesh);
            }
        } else if (st_.depth_stencil.depth_func == kDepthEqual && !cutout_meshes_.count(mesh)) {
            st_.depth_stencil.depth_func = kDepthGequal;
        }
    }
    VkPipeline p = VK_NULL_HANDLE;
    if (frame_ == trace_frame_ && full_trace_) {
        const auto* fs = st_.stages[static_cast<int>(Stage::Fragment)];
        if (fs && fs->print == kFacePixelShader && st_.rt.count && st_.rt.colors[0] &&
            st_.rt.colors[0]->vk_format == VK_FORMAT_B10G11R11_UFLOAT_PACK32) {
            ++face_probe_matches_;
            p = pipeline(d, rp_, l, rp_color_count_, rp_samples_, true);
            if (!p) NVN_LOG("face probe: unavailable for draw %d; using original pipeline\n", traced_draws_);
        }
    }
    const bool face_probed = p != VK_NULL_HANDLE;
    if (!p) p = pipeline(d, rp_, l, rp_color_count_, rp_samples_);
    st_.color.blend_enable = saved_blend;
    st_.depth_stencil = saved_depth;
    if (!p) {
        skip_draw(7);  // (pipeline)
        return;
    }
    if (p != bound_graphics_) {  // seguidos con el mismo estado: sin volver a enlazarlo
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
        bound_graphics_ = p;
        if (prof_timing_) fine_.pipe_binds++;
    }
    prof_split(3, prof_s);

    prof_mark(1, prof_t);
    // 3. Descriptores: uniform buffers copiados al staging y texturas resueltas por los pools.
    auto prof_sub = prof_now();
    // Dynamic offsets are draw state. The buffer handles, ranges, views and
    // samplers in contents identify the reusable set independently of them.
    static std::vector<VkWriteDescriptorSet> writes;
    static std::vector<VkDescriptorBufferInfo> buffers;
    static std::vector<VkDescriptorImageInfo> images;
    static std::vector<uint32_t> dynamic_offsets;
    static std::vector<uint64_t> contents;  // lo que define el set: buffers y tamaños, vistas y samplers
    writes.clear();
    buffers.clear();
    images.clear();
    dynamic_offsets.clear();
    contents.clear();
    const bool dynamic = l.dynamic;
    auto push_buffer = [&](VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size) {
        if (dynamic) {
            dynamic_offsets.push_back(static_cast<uint32_t>(offset));
            contents.push_back(reinterpret_cast<uint64_t>(buffer));
            contents.push_back(size);
        } else {
            buffers.push_back({buffer, offset, size});
        }
    };
    buffers.reserve(kStages * (kMaxUniforms + kMaxStorage + 2));
    images.reserve(std::max<size_t>(textures.size(), kStages * (kMaxTextures + kMaxImages)) + kStages * 8);
    const VkDeviceSize ubo_align = std::max<VkDeviceSize>(context.props.limits.minUniformBufferOffsetAlignment, 16);
    const VkDeviceSize ssbo_align = std::max<VkDeviceSize>(context.props.limits.minStorageBufferOffsetAlignment, 16);
    uint8_t handles_cbuf[0x700];
    for (int s = 0; s < kStages; s++) {
        if (!st_.stages[s]) continue;
        for (const Resource& r : st_.stages[s]->shader.resources) {
            if (r.kind != Resource::UniformBuffer && r.kind != Resource::StorageBuffer) continue;
            BufferRange range{};
            const void* src = nullptr;
            if (r.kind == Resource::StorageBuffer) {
                const uint32_t base = kGraphicsStorageBase + kGraphicsStorageStride * s;
                if (r.cbuf_index == 0 && r.cbuf_offset >= base && r.cbuf_offset < base + 16 * kMaxStorage)
                    range = st_.bindings[s].storage[(r.cbuf_offset - base) / 16];
                if (r.is_written && range.address) {
                    static int logged = 0;
                    if (frame_ == trace_frame_ && range.size == 20 && d.index_type >= 0)  // DEPURACIÓN: args de la compactación
                        prepass_args_[range.address] = {d.index_address, static_cast<uint32_t>(d.count), d.index_type};
                    if (s == static_cast<int>(Stage::Geometry) && range.size == 20 && d.index_type >= 0 && !d.indirect) {
                        culled_draws_[range.address] = {d, frame_};
                        const_cast<ProgramStage*>(st_.stages[s])->compaction_gs = true;
                    }
                    if (frame_ == trace_frame_)
                        NVN_LOG("F SSBO escrito %#llx+%llu\n", static_cast<unsigned long long>(range.address),
                                static_cast<unsigned long long>(range.size));
                    if (logged++ < 4)
                        NVN_LOG("SSBO escrito desde un shader %s: %#llx+%llu, se queda en la GPU\n",
                                nvnshader::stage_name(static_cast<Stage>(s)), static_cast<unsigned long long>(range.address),
                                static_cast<unsigned long long>(range.size));
                }
            }
            if (r.kind == Resource::UniformBuffer) {
                int index = static_cast<int>(r.cbuf_index) - static_cast<int>(kUniformBankBase);
                if (index >= 0 && index < kMaxUniforms) range = st_.bindings[s].uniforms[index];
                if (r.cbuf_index == kShaderConstantsBank)
                    range = {st_.stages[s]->constants, st_.stages[s]->constants_size};
                if (r.cbuf_index == kTextureHandleBank) {
                    // c2 leído como datos: algunos shaders comprueban o combinan los handles (textura en
                    // 0x168 + 8·i, sampler separado en 0x568 + 8·i, y los combinan con OR). Con ceros
                    // tomaban el camino "sin textura" y el material salía negro.
                    memset(handles_cbuf, 0, sizeof handles_cbuf);
                    const StageBindings& sbind = st_.bindings[s];
                    for (int k = 0; k < kMaxTextures; k++) {
                        const uint32_t tex = static_cast<uint32_t>(sbind.textures[k]);
                        const uint32_t smp = static_cast<uint32_t>(sbind.samplers[k]);
                        if (kTextureHandleBase + 8 * k + 4 <= sizeof handles_cbuf)
                            memcpy(handles_cbuf + kTextureHandleBase + 8 * k, &tex, 4);
                        if (kSamplerHandleBase + 8 * k + 4 <= sizeof handles_cbuf)
                            memcpy(handles_cbuf + kSamplerHandleBase + 8 * k, &smp, 4);
                        if (const uint64_t c = sbind.combined_textures[k]; c && separate_tables_) {  // la tabla baja
                            const uint32_t word = handle_texture_id(c) | handle_sampler_id(c) << 20;
                            memcpy(handles_cbuf + kLowTextureHandleBase + 8 * k, &word, 4);
                        }
                    }
                    src = handles_cbuf;
                    range.size = sizeof handles_cbuf;
                }
                if (r.cbuf_index == 0) range.size = kGraphicsDriverSize;
            }
            if (!src && range.address) src = reinterpret_cast<const void*>(range.address);
            if (range.address) note_read(r.kind == Resource::UniformBuffer ? 2 : 3, range.address, range.size);
            VkDeviceSize size = std::min<VkDeviceSize>(range.size ? range.size : 256,
                                                       r.kind == Resource::UniformBuffer ? 65536 : kStagingChunk);
            if (r.kind == Resource::UniformBuffer && r.cbuf_index < 18) {  // solo lo que lee el shader
                uint32_t used = st_.stages[s]->shader.constant_buffer_used[r.cbuf_index];
                if (used) size = std::min<VkDeviceSize>(size, (used + 15) & ~15u);
            }
            VkDeviceSize ov_offset = 0;
            const bool driver_c0 = r.kind == Resource::UniformBuffer && r.cbuf_index == 0;
            const bool from_game = !driver_c0 && src == reinterpret_cast<const void*>(range.address) && range.size;
            if (vigilar_nan_ && from_game && r.kind == Resource::StorageBuffer && r.is_written) {  // DEPURACIÓN
                const ProgramStage* vsp = st_.stages[static_cast<int>(Stage::Vertex)];
                const ProgramStage* fsp = st_.stages[static_cast<int>(Stage::Fragment)];
                char b[128];
                snprintf(b, sizeof b, "draw (vs %016llx ps %016llx, etapa %s), frame %llu",
                         static_cast<unsigned long long>(vsp ? vsp->print : 0), static_cast<unsigned long long>(fsp ? fsp->print : 0),
                         nvnshader::stage_name(static_cast<Stage>(s)), static_cast<unsigned long long>(frame_));
                writer_ = b;
            }
            // Con el pase abierto make_overlay no copia: las uniones ya las hizo prepare_draw_overlays.
            const Overlay* ov = !from_game ? nullptr
                                : r.kind == Resource::StorageBuffer && r.is_written
                                    ? make_overlay(range.address, size, &ov_offset, false)
                                    : find_overlay(range.address, size, &ov_offset);
            if (ov && r.kind == Resource::StorageBuffer && r.is_written) note_overlay_written(range.address, kWrittenByGraphics);
            if (vigilar_nan_ && from_game && !ov && r.kind == Resource::StorageBuffer) note_partial_read("un draw", range.address, size);
            if (frame_ == trace_frame_ && r.kind == Resource::StorageBuffer && !r.is_written) {  // DEPURACIÓN
                std::string vals;
                char b[24];
                if (ov && ov->data)
                    for (size_t i = 0; i < std::min<size_t>(size / 4, 8); i++)
                        snprintf(b, sizeof b, " %.4g", reinterpret_cast<const float*>(ov->data + (ov_offset - ov->offset))[i]), vals += b;
                NVN_LOG("F SSBO leído %#llx+%llu: %s%s\n", static_cast<unsigned long long>(range.address),
                        static_cast<unsigned long long>(size), ov ? "de la GPU, último valor:" : "de la memoria", vals.c_str());
            }
            VkDeviceSize mirror_offset = 0;
            const Mirror* mir = !ov && from_game && r.kind == Resource::StorageBuffer && !r.is_written
                                    ? mirror(range.address, range.size, &mirror_offset, false)
                                    : nullptr;
            if (driver_c0) {
                auto a = graphics_c0(size, ubo_align);
                if (!a.data) return;
                push_buffer(a.buffer, a.offset, size);
            } else if (ov) {
                push_buffer(ov->buffer, ov_offset, size);
            } else if (mir) {
                push_buffer(mir->buffer, mirror_offset, range.size);
            } else if (const auto* hit = from_game ? copied_.find(range.address, size) : nullptr) {
                push_buffer(hit->first, hit->second, size);  // ya copiado en este envío
            } else {
                if (from_game && size >= (4u << 20) && NVN_FLAG("measure_changes")) measure_changes(range.address, range.size);
                Staging::Alloc a = staging_->alloc(size, r.kind == Resource::UniformBuffer ? ubo_align : ssbo_align);
                if (!a.data) return;
                if (src && range.size)
                    memcpy(a.data, src, size);
                else
                    memset(a.data, 0, size);
                prof_copied_bytes_ += size;
                note_copy(range.address, size, r.kind == Resource::StorageBuffer, s);
                if (from_game) copied_.insert(range.address, size, a.buffer, a.offset);
                push_buffer(a.buffer, a.offset, size);
            }
        }
    }
    prof_mark_sub(0, prof_sub);
    for (BoundImage& b : textures) {
        if (b.r->kind == Resource::Image) {  // storage image: sin sampler
            VkImageView iv = storage_view(b.tex, b.view, b.r->texture_type, b.storage_target);
            if (!iv) { skip_draw(8); return; }
            if (dynamic) {
                contents.push_back(reinterpret_cast<uint64_t>(iv));
                contents.push_back(0);
            } else images.push_back({VK_NULL_HANDLE, iv, VK_IMAGE_LAYOUT_GENERAL});
            continue;
        }
        Texture* t = b.tex;
        if ((!t || !t->image) && frame_ == trace_frame_)  // DEPURACIÓN
            NVN_LOG("F textura sin resolver: c%u[0x%x] (%s)\n", b.r->cbuf_index, b.r->cbuf_offset,
                    b.tex ? "sin imagen" : "no está en el pool");
        VkImageView iv = t && t->image ? view(*t, b.view, ViewUse::Sampled) : VK_NULL_HANDLE;
        if (loading_armed_ && iv)
            check_loading_binding(*t, b.view, iv, static_cast<uint32_t>(b.s), *b.r, 0, b.smp);
        if (!iv) iv = dummy_view(b.r->texture_type);
        if (!iv) { skip_draw(8); return; }
        const VkSampler sample = sampler(b.smp);
        if (dynamic) {
            contents.push_back(reinterpret_cast<uint64_t>(iv));
            contents.push_back(reinterpret_cast<uint64_t>(sample));
        } else images.push_back({sample, iv, VK_IMAGE_LAYOUT_GENERAL});
    }
    prof_mark_sub(1, prof_sub);
    // El set del draw anterior con este layout sirve si apunta a lo mismo (solo cambian los desplazamientos,
    // que van en el bind). Vale hasta el siguiente begin(): ahí se vacían los pools y se liberan buffers y vistas.
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (dynamic && l.memo_set && l.memo_epoch == descriptor_epoch_ && l.memo_key == contents) {
        set = l.memo_set;
        if (prof_timing_) fine_.sets_memo++;
    } else {
        // Uno de este envío con el mismo contenido (ver SetTable): pasa a ser el del draw anterior de este layout.
        const bool cache = dynamic && cache_sets_;
        const uint64_t contents_hash =
            cache ? hash_bytes(contents.data(), contents.size() * sizeof(uint64_t), reinterpret_cast<uint64_t>(l.set)) : 0;
        if (cache) set = sets_cache_.find(descriptor_epoch_, l.set, contents, contents_hash);
        if (set) {
            if (prof_timing_) fine_.sets_map++;
        } else {
            const auto alloc_t0 = prof_now();
            set = descriptor_set(l.set);
            if (prof_timing_) fine_.set_alloc_ns += ns_since(alloc_t0);
            prof_mark_sub(3, prof_sub);
            if (!set) {
                skip_draw(8);  // (set)
                return;
            }
            const auto update_t0 = prof_now();
            // Only a new set needs Vulkan update structures. Materialize from the same
            // resolved contents used for the lookup, including feedback views.
            if (dynamic) {
                size_t k = 0;
                for (size_t i = 0; i < dynamic_offsets.size(); ++i, k += 2)
                    buffers.push_back({reinterpret_cast<VkBuffer>(contents[k]), 0, contents[k + 1]});
                for (size_t i = 0; i < textures.size(); ++i, k += 2)
                    images.push_back({reinterpret_cast<VkSampler>(contents[k + 1]),
                                      reinterpret_cast<VkImageView>(contents[k]), VK_IMAGE_LAYOUT_GENERAL});
            }
            size_t buffer_index = 0;
            for (const ProgramStage* stage : st_.stages) {
                if (!stage) continue;
                for (const Resource& r : stage->shader.resources) {
                    if (r.kind != Resource::UniformBuffer && r.kind != Resource::StorageBuffer) continue;
                    const auto type = r.kind == Resource::UniformBuffer
                        ? (dynamic ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                        : (dynamic ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
                    writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, r.binding, 0, 1, type,
                                      nullptr, &buffers[buffer_index++], nullptr});
                }
            }
            for (size_t i = 0; i < textures.size(); ++i) {
                const Resource& r = *textures[i].r;
                const auto type = r.kind == Resource::Image ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes.push_back({VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, set, r.binding, 0, 1, type,
                                  &images[i], nullptr, nullptr});
            }
            if (templates_ && update_with_template(l, set, writes)) {
                if (prof_timing_) fine_.set_templates++;
            } else {
                vkUpdateDescriptorSets(context.device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
            }
            if (prof_timing_) {
                fine_.set_update_ns += ns_since(update_t0);
                fine_.sets_new++;
                fine_.set_writes += writes.size();
            }
            if (cache) sets_cache_.insert(descriptor_epoch_, l.set, contents, contents_hash, set);
        }
        if (dynamic) {
            l.memo_set = set;
            l.memo_epoch = descriptor_epoch_;
            l.memo_key = contents;
        }
    }
    const auto bind_t0 = prof_now();
    // Con otro layout que el del set 0 anterior, el set 1 enlazado puede quedar invalidado: se vuelve a enlazar.
    if (l.pipeline != bound_set0_layout_) bound_set1_ = VK_NULL_HANDLE;
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, l.pipeline, 0, 1, &set,
                            static_cast<uint32_t>(dynamic_offsets.size()), dynamic_offsets.data());
    bound_set0_layout_ = l.pipeline;
    if (bindless) {  // tras el set 0: enlazar el 0 con otro layout invalida el 1
        if (skip_set1_ && bindless == bound_set1_ && l.pipeline == bound_set1_layout_) {
            if (prof_timing_) fine_.set1_skipped++;
        } else {
            vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, l.pipeline, nvnshader::kBindlessSet, 1, &bindless, 0, nullptr);
            bound_set1_ = bindless;
            bound_set1_layout_ = l.pipeline;
        }
    }
    if (prof_timing_) fine_.set_bind_ns += ns_since(bind_t0);
    // The draw count check: the count's word, bound from an aligned offset (the rest goes in the push constants)
    uint32_t check_word = 0;
    if (check_buf) {
        const VkDeviceSize align = std::max<VkDeviceSize>(context.props.limits.minStorageBufferOffsetAlignment, 16);
        const VkDeviceSize base = check_offset / align * align;
        if (count_check_sets_epoch_ != descriptor_epoch_) {  // (the pools they came from were reset)
            count_check_sets_.clear();
            count_check_sets_epoch_ = descriptor_epoch_;
        }
        // One set per count (buffer and offset): the multi-draws of a frame use a few counts many times over
        const uint64_t key = (reinterpret_cast<uint64_t>(check_buf) * 0x9E3779B97F4A7C15ull) ^ check_offset;
        VkDescriptorSet cs = VK_NULL_HANDLE;
        if (auto it = count_check_sets_.find(key); it != count_check_sets_.end()) cs = it->second;
        else if (l.count_check && (cs = descriptor_set(count_check_set_layout_))) {
            const VkDescriptorBufferInfo info{check_buf, base, check_offset - base + 4};
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = cs;
            w.dstBinding = 0;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.pBufferInfo = &info;
            vkUpdateDescriptorSets(context.device, 1, &w, 0, nullptr);
            count_check_sets_[key] = cs;
        }
        if (!cs || !l.count_check) {
            skip_draw(9);  // (count_set)
            return;
        }
        vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, l.pipeline, kCountCheckSet, 1, &cs, 0, nullptr);
        check_word = static_cast<uint32_t>((check_offset - base) / 4);
        prof_count_checked_++;
    }
    prof_mark_sub(2, prof_sub);

    prof_mark(2, prof_t);
    // 4. Estado dinámico.
    // NVN mapea la Y de NDC como OpenGL (+1 arriba) aunque la ventana tenga el origen arriba a la
    // izquierda (viewport, scissor y gl_FragCoord desde arriba); Vulkan pone -1 arriba. Viewport con
    // altura negativa (Vulkan 1.1): misma caja, Y invertida.
    // Persiste en el command buffer (todos los pipelines declaran los mismos estados dinámicos): solo lo que cambia.
    DynamicState ds{};
    ds.viewport = st_.viewport;
    ds.viewport.y += ds.viewport.height;
    ds.viewport.height = -ds.viewport.height;
    ds.scissor = st_.scissor_set ? st_.scissor : VkRect2D{{0, 0}, rp_extent_};
    if (!rp_area_full_) ds.scissor = intersect(ds.scissor, rp_area_);  // nada fuera del renderArea, pase lo que pase
    memcpy(ds.blend, st_.blend_color, sizeof ds.blend);
    for (int f = 0; f < 2; f++)
        ds.ref[f] = st_.stencil_ref[f], ds.write_mask[f] = st_.stencil_mask[f], ds.compare_mask[f] = st_.stencil_value_mask[f];
    memcpy(ds.bias, st_.depth_bias, sizeof ds.bias);
    if (!context.depth_bias_clamp) ds.bias[1] = 0.0f;
    const bool all = !dynamic_valid_;
    if (all || memcmp(&ds.viewport, &bound_dynamic_.viewport, sizeof ds.viewport)) vkCmdSetViewport(cmd_, 0, 1, &ds.viewport);
    if (all || memcmp(&ds.scissor, &bound_dynamic_.scissor, sizeof ds.scissor)) vkCmdSetScissor(cmd_, 0, 1, &ds.scissor);
    if (all || memcmp(ds.blend, bound_dynamic_.blend, sizeof ds.blend)) vkCmdSetBlendConstants(cmd_, ds.blend);
    if (all || memcmp(ds.bias, bound_dynamic_.bias, sizeof ds.bias)) vkCmdSetDepthBias(cmd_, ds.bias[0], ds.bias[1], ds.bias[2]);
    constexpr VkStencilFaceFlags kFace[2] = {VK_STENCIL_FACE_FRONT_BIT, VK_STENCIL_FACE_BACK_BIT};
    for (int f = 0; f < 2; f++) {
        if (all || ds.ref[f] != bound_dynamic_.ref[f]) vkCmdSetStencilReference(cmd_, kFace[f], ds.ref[f]);
        if (all || ds.write_mask[f] != bound_dynamic_.write_mask[f]) vkCmdSetStencilWriteMask(cmd_, kFace[f], ds.write_mask[f]);
        if (all || ds.compare_mask[f] != bound_dynamic_.compare_mask[f])
            vkCmdSetStencilCompareMask(cmd_, kFace[f], ds.compare_mask[f]);
    }
    if (context.extended_dynamic_state) {
        const DepthStencilState& z = st_.depth_stencil;
        ds.cull = vk_cull_mode(st_.polygon.cull_face);
        ds.front = st_.polygon.front_face ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
        ds.depth_test = z.depth_test;
        ds.depth_write = z.depth_write;
        ds.depth_op = vk_compare_op(z.depth_func);
        ds.stencil_test = z.stencil_test;
        const uint32_t ops[2][4] = {{static_cast<uint32_t>(vk_stencil_op(z.front_fail)), static_cast<uint32_t>(vk_stencil_op(z.front_zpass)),
                                     static_cast<uint32_t>(vk_stencil_op(z.front_zfail)), static_cast<uint32_t>(vk_compare_op(z.front_func))},
                                    {static_cast<uint32_t>(vk_stencil_op(z.back_fail)), static_cast<uint32_t>(vk_stencil_op(z.back_zpass)),
                                     static_cast<uint32_t>(vk_stencil_op(z.back_zfail)), static_cast<uint32_t>(vk_compare_op(z.back_func))}};
        memcpy(ds.stencil_ops, ops, sizeof ops);
        if (all || ds.cull != bound_dynamic_.cull) vkCmdSetCullModeEXT(cmd_, ds.cull);
        if (all || ds.front != bound_dynamic_.front) vkCmdSetFrontFaceEXT(cmd_, static_cast<VkFrontFace>(ds.front));
        if (all || ds.depth_test != bound_dynamic_.depth_test) vkCmdSetDepthTestEnableEXT(cmd_, ds.depth_test);
        if (all || ds.depth_write != bound_dynamic_.depth_write) vkCmdSetDepthWriteEnableEXT(cmd_, ds.depth_write);
        if (all || ds.depth_op != bound_dynamic_.depth_op) vkCmdSetDepthCompareOpEXT(cmd_, static_cast<VkCompareOp>(ds.depth_op));
        if (all || ds.stencil_test != bound_dynamic_.stencil_test) vkCmdSetStencilTestEnableEXT(cmd_, ds.stencil_test);
        for (int f = 0; f < 2; f++)
            if (all || memcmp(ds.stencil_ops[f], bound_dynamic_.stencil_ops[f], sizeof ds.stencil_ops[f]))
                vkCmdSetStencilOpEXT(cmd_, kFace[f], static_cast<VkStencilOp>(ds.stencil_ops[f][0]),
                                     static_cast<VkStencilOp>(ds.stencil_ops[f][1]), static_cast<VkStencilOp>(ds.stencil_ops[f][2]),
                                     static_cast<VkCompareOp>(ds.stencil_ops[f][3]));
    }
    if (context.vertex_input_dynamic_state) {
        // The vertex layout, as make_graphics_pipeline would bake it: each generic the vertex shader reads at its
        // location, a binding of its own (its own buffer: see the vertices below), offset 0
        VkVertexInputBindingDescription2EXT binds[16];
        VkVertexInputAttributeDescription2EXT attrs[16];
        uint32_t n = 0;
        uint64_t print = 0xcbf29ce484222325ull;
        for (uint32_t used = vertex_inputs; used;) {
            const auto k = take_location(used);
            const VertexAttribState a = k < st_.attribs.size() ? st_.attribs[k] : VertexAttribState{};
            const uint32_t stride = a.stream < st_.streams.size() ? st_.streams[a.stream].stride : 0;
            const bool instanced = a.stream < st_.streams.size() && st_.streams[a.stream].divisor;
            const VkFormat format = vertex_format(a.format);
            binds[n] = {VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, nullptr, k, stride,
                        instanced ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX, 1};
            attrs[n] = {VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, nullptr, k, k, format, 0};
            for (uint64_t v : {uint64_t{k}, uint64_t{stride}, uint64_t{instanced}, uint64_t{static_cast<uint32_t>(format)}})
                print = (print ^ v) * 0x100000001b3ull;
            n++;
        }
        if (all || print != bound_vertex_input_) {
            vkCmdSetVertexInputEXT(cmd_, n, binds, n, attrs);
            bound_vertex_input_ = print;
        }
    }
    bound_dynamic_ = ds;
    dynamic_valid_ = true;

    prof_mark(3, prof_t);
    // The inputs were resolved before opening the pass so mixed CPU/GPU streams could be copied.
    const uint32_t bind_used = vertex_inputs;
    if (bind_used) {
        uint32_t lo = 16, hi = 0;
        for (uint32_t used = bind_used; used;) {
            const auto k = take_location(used);
            if (bind_buf[k] != bound_vb_buf_[k] || bind_off[k] != bound_vb_off_[k])
                lo = std::min(lo, k), hi = k;
        }
        if (lo <= hi) {
            const uint32_t first_used = static_cast<uint32_t>(__builtin_ctz(bind_used));
            for (uint32_t k = lo; k <= hi; k++)
                if (!(bind_used >> k & 1)) {  // hueco: lo que ya había, o cualquier buffer válido
                    bind_buf[k] = bound_vb_buf_[k] ? bound_vb_buf_[k] : bind_buf[first_used];
                    bind_off[k] = bound_vb_buf_[k] ? bound_vb_off_[k] : 0;
                }
            vkCmdBindVertexBuffers(cmd_, lo, hi - lo + 1, bind_buf + lo, bind_off + lo);
            for (uint32_t k = lo; k <= hi; k++) bound_vb_buf_[k] = bind_buf[k], bound_vb_off_[k] = bind_off[k];
        }
    }
    prof_mark(4, prof_t);
    // DEPURACIÓN (<datos>/profile_gpu_draws con profile_gpu): en los pases sin color (sombras), una marca por draw con sus
    // shaders, para ver cuál cuesta.
    if (prof_pool_ && prof_frame_ == frame_ && !prof_closed_ && st_.rt.count == 0 && NVN_FLAG("profile_gpu_draws")) {
        const ProgramStage* vsp = st_.stages[static_cast<int>(Stage::Vertex)];
        const ProgramStage* fsp = st_.stages[static_cast<int>(Stage::Fragment)];
        char label[96];
        snprintf(label, sizeof label, "draw%s vs %016llx fs %016llx", multi ? " multi" : "",
                 static_cast<unsigned long long>(vsp ? vsp->print : 0), static_cast<unsigned long long>(fsp ? fsp->print : 0));
        prof_gpu_mark("draw", label);
    }
    // 6. Dibujar. Los quads (NVN 7) se convierten a triángulos.
    if (d.primitive == 7) {
        uint32_t quads = static_cast<uint32_t>(d.count) / 4;
        Staging::Alloc al = staging_->alloc(quads * 6 * sizeof(uint32_t), 16);
        if (!al.data) return;
        auto* out = reinterpret_cast<uint32_t*>(al.data);
        for (uint32_t q = 0; q < quads; q++) {
            uint32_t v[4];
            for (int j = 0; j < 4; j++) {
                uint32_t k = q * 4 + j;
                v[j] = !indices ? d.first + k - first_vertex : d.index_type == 0 ? indices[k] : d.index_type == 1 ? reinterpret_cast<const uint16_t*>(indices)[k]
                                                                                                     : reinterpret_cast<const uint32_t*>(indices)[k];
            }
            uint32_t tri[6] = {v[0], v[1], v[2], v[0], v[2], v[3]};
            memcpy(out + q * 6, tri, sizeof tri);
        }
        bind_index(al.buffer, al.offset, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd_, quads * 6, std::max(d.instances, 1), 0, indices ? vertex_offset : 0, d.base_instance);
    } else if (gpu_args || gpu_index) {
        const VkIndexType type = d.index_type == 2 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
        if (gpu_index)
            bind_index(gpu_index, gpu_index_offset, type);
        else if (mirror_index)
            bind_index(mirror_index, mirror_index_offset, type);
        else if (cached_index_buffer)
            bind_index(cached_index_buffer, cached_index_offset, type);
        else if (host_index)
            bind_index(host_index, host_index_offset, type);
        const uint32_t stride = static_cast<uint32_t>(d.stride);
        if (multi && count_buf && indices)
            context.draw_indexed_indirect_count(cmd_, gpu_args, gpu_args_offset, count_buf, count_offset,
                                              static_cast<uint32_t>(d.max_draws), stride);
        else if (multi && count_buf)
            context.draw_indirect_count(cmd_, gpu_args, gpu_args_offset, count_buf, count_offset, static_cast<uint32_t>(d.max_draws), stride);
        else if (multi && (context.multi_draw_indirect || multi_draws == 1)) {
            if (check_buf) {  // (one draw: index 0)
                const uint32_t push[2] = {0, check_word};
                vkCmdPushConstants(cmd_, l.pipeline, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof push, push);
            }
            if (indices) vkCmdDrawIndexedIndirect(cmd_, gpu_args, gpu_args_offset, multi_draws, stride);
            else vkCmdDrawIndirect(cmd_, gpu_args, gpu_args_offset, multi_draws, stride);
        } else if (multi) {  // without the multiDrawIndirect feature: one call per draw
            for (uint32_t k = 0; k < multi_draws; k++) {
                const VkDeviceSize at = gpu_args_offset + VkDeviceSize{k} * stride;
                if (check_buf) {  // this draw's index, for the count check
                    const uint32_t push[2] = {k, check_word};
                    vkCmdPushConstants(cmd_, l.pipeline, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof push, push);
                }
                if (indices) vkCmdDrawIndexedIndirect(cmd_, gpu_args, at, 1, 0);
                else vkCmdDrawIndirect(cmd_, gpu_args, at, 1, 0);
            }
        }
        else if (gpu_args && indices)
            vkCmdDrawIndexedIndirect(cmd_, gpu_args, gpu_args_offset, 1, 20);
        else if (gpu_args)
            vkCmdDrawIndirect(cmd_, gpu_args, gpu_args_offset, 1, 16);
        else
            vkCmdDrawIndexed(cmd_, static_cast<uint32_t>(d.count), std::max(d.instances, 1), 0, vertex_offset, d.base_instance);
    } else if (cached_index_buffer) {
        bind_index(cached_index_buffer, cached_index_offset,
                   d.index_type == 2 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16);
        vkCmdDrawIndexed(cmd_, static_cast<uint32_t>(d.count), std::max(d.instances, 1), 0, vertex_offset, d.base_instance);
    } else if (indices) {
        // Vulkan 1.1 sin VK_EXT_index_type_uint8 activo: los índices de 8 bits se amplían a 16.
        uint32_t out_size = d.index_type == 0 ? 2 : isize;
        Staging::Alloc al = staging_->alloc(VkDeviceSize{out_size} * d.count, 16);
        if (!al.data) return;
        if (d.index_type == 0)
            for (int k = 0; k < d.count; k++) reinterpret_cast<uint16_t*>(al.data)[k] = indices[k];
        else
            memcpy(al.data, indices, VkDeviceSize{isize} * d.count);
        bind_index(al.buffer, al.offset, d.index_type == 2 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16);
        vkCmdDrawIndexed(cmd_, static_cast<uint32_t>(d.count), std::max(d.instances, 1), 0, vertex_offset, d.base_instance);
    } else {
        vkCmdDraw(cmd_, static_cast<uint32_t>(d.count), std::max(d.instances, 1), static_cast<uint32_t>(d.first) - first_vertex,
                  static_cast<uint32_t>(d.base_instance));
    }
    if (writes_storage) storage_since_barrier_ = true;
    note_pass_writes();
    publish_storage_writes(textures, "imagen de un draw");
    if (face_probed) {
        end_render_pass();
        targets_changed_ = true;
        const auto name = "face-f" + std::to_string(frame_) + "-d" + std::to_string(traced_draws_);
        const bool saved = dump_raw(*st_.rt.colors[0], name);
        face_probe_saved_ += saved;
        NVN_LOG("face probe: %s raw %s\n", name.c_str(), saved ? "saved" : "failed");
        begin();
    }
    if (loading_armed_ && (loading_log_ || !loading_matches_.empty())) {
        record_loading_draw(d, p, attachments);
        if (loading_log_)
            for (const BoundImage& b : textures)
                if (b.tex)
                    fprintf(loading_log_, "  source stage=%u binding=%u kind=%u write=%u serial=%llu content=%llu\n",
                            unsigned(b.s), b.r->binding, unsigned(b.r->kind), unsigned(b.r->is_written),
                            (unsigned long long)b.tex->serial, (unsigned long long)b.tex->content_version);
    }
    prof_mark(5, prof_t);
    // DEPURACIÓN: NIER_VOLCAR_TRAS="a,b,c": vuelca el render target 0 tras esos draws del frame trazado.
    if (frame_ == trace_frame_ && st_.rt.count > 0 && st_.rt.colors[0]) {
        const Texture* rt0 = st_.rt.colors[0];
        const char* by_size = debug_var("NIER_VOLCAR_TAM", trace_frame_);  // DEPURACIÓN: "640x360": tras cada draw a un RT de ese tamaño
        const bool size_match = by_size && *by_size &&
                                std::to_string(rt0->desc.width) + "x" + std::to_string(rt0->desc.height) == by_size;
        // NIER_VOLCAR_PRECALC (no vacía): también tras cada draw con vértices de la GPU (las cabezas con skinning en compute:
        // cara, ojos y pestañas del cazador), sin saber antes sus números.
        const char* precalc = gpu_vertices ? debug_var("NIER_VOLCAR_PRECALC", trace_frame_) : nullptr;
        // NIER_VOLCAR_HUELLAS="a,b" (huellas de pixel shader en hexadecimal): tras cada draw con uno de esos. Los números de
        // draw cambian de un frame a otro y la huella no: el destello del Aknosom (5-11) sale en una traza del detector.
        bool by_print = false;
        const ProgramStage* fs = st_.stages[static_cast<int>(Stage::Fragment)];
        char fs_hex[20] = {};
        if (const char* prints = debug_var("NIER_VOLCAR_HUELLAS", trace_frame_); prints && *prints && fs) {
            snprintf(fs_hex, sizeof fs_hex, "%016llx", static_cast<unsigned long long>(fs->print));
            by_print = (std::string(",") + prints + ",").find(std::string(",") + fs_hex + ",") != std::string::npos;
        }
        if (traced_in_list("NIER_VOLCAR_TRAS") || size_match || by_print || (precalc && *precalc)) {
            split_why_ = "volcado (depuración)";
            end_render_pass();
            targets_changed_ = true;
            dump(*st_.rt.colors[0], "tras" + std::to_string(traced_draws_) + (by_print ? std::string("_") + fs_hex : ""));
            begin();  // dump() envió lo grabado hasta aquí: se sigue en un command buffer nuevo
        }
    } else if (frame_ == trace_frame_ && st_.rt.depth && st_.rt.depth->image && traced_in_list("NIER_VOLCAR_TRAS")) {
        // Pases solo de profundidad (las cascadas de las sombras): la capa de la vista enlazada, con toda la precisión.
        split_why_ = "volcado (depuración)";
        end_render_pass();
        targets_changed_ = true;
        const TextureViewDesc& dv = st_.rt.depth_view;
        const int layer = (dv.flags & kViewLayers) ? std::max<int>(dv.base_layer, 0) : 0;
        dump_depth(*st_.rt.depth, "tras" + std::to_string(traced_draws_) + "_capa" + std::to_string(layer), layer);
        begin();  // dump_depth envió lo grabado hasta aquí
    }
    if (draws_ % 5000 == 0 && NVN_FLAG("verbose_log"))
    {
        static const char* const kWhy[12] = {"tess", "multi", "absurd_ind", "absurd_dir", "no_vs", "bindless", "pass",
                                             "pipeline", "set", "count_set", "gpu_data", "indices"};
        std::string why;
        for (int k = 0; k < 12; k++)
            if (skipped_why_[k]) why += std::string(why.empty() ? "" : ", ") + kWhy[k] + " " + std::to_string(skipped_why_[k]);
        NVN_LOG("%llu draws (%llu sin dibujar%s%s), %zu pipelines\n", static_cast<unsigned long long>(draws_),
                static_cast<unsigned long long>(skipped_draws_), why.empty() ? "" : ": ", why.c_str(), pipelines_.size());
    }
}

VkPipeline Executor::compute_pipeline(const ProgramStage& cs, Layout& l) {
    constexpr int S = static_cast<int>(Stage::Compute);
    // Los compute a los que se les dejaron las barreras de sus bucles uniformes (la exposición automática) vuelven en vivo
    // a la traducción sin ellas con su interruptor (use_stripped_module): otra pipeline, con otra clave, que ni se toma del
    // precalentado ni se apunta para él (sus registros llevan la traducción con las barreras).
    const bool stripped = use_stripped_module(cs);
    PipelineKey key;
    memset(&key, 0, sizeof key);
    key.stages[S] = cs.spirv_id;
    key.topology = kComputeTopology;
    uint64_t h = hash_bytes(&key, sizeof key);
    if (per_stage_keys_) {  // pipelines_per_stage: como antes (etapa, generación)
        const uint64_t mine[3] = {reinterpret_cast<uint64_t>(&cs), st_.stage_generation[S], h};
        h = hash_bytes(mine, sizeof mine);
    }
    if (stripped) {
        const uint64_t mine[2] = {h, 0x5354524950504544ull};  // "STRIPPED"
        h = hash_bytes(mine, sizeof mine);
    }
    if (auto it = compute_pipelines_.find(h); it != compute_pipelines_.end()) return it->second;
    std::optional<PipelineBuild> warm;
    if (!per_stage_keys_ && !stripped) {
        LoadTimer timer(*this, kLdWarm);
        warm = warm_take(h);
        if (warm && warm->pipeline) {
            ++prof_warm_taken_;
            return compute_pipelines_[h] = warm->pipeline;
        }
    }
    VkPipeline p = VK_NULL_HANDLE;
    const auto created_at = std::chrono::steady_clock::now();
    const bool during_save = driver_cache_.reading();
    VkResult r;
    if (warm && warm->modules[S] == cs.cache_key) {
        r = warm->status;
    } else {
        LoadTimer timer(*this, kLdPipeline);
        r = driver_cache_.compile([&](VkPipelineCache cache) {
            return make_compute_pipeline(stripped ? cs.stripped_module : cs.module, l.pipeline, cache, p);
        });
    }
    if (r != VK_SUCCESS) NVN_LOG("vkCreateComputePipelines: %s\n", vk_result_name(r));
    const uint64_t created_us =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - created_at).count();
    prof_pipeline_us_ += created_us;
    g_drop_stats.pipelines_live.fetch_add(1, std::memory_order_relaxed);
    g_drop_stats.pipelines_live_ns.fetch_add(created_us * 1000, std::memory_order_relaxed);
    if (during_save || driver_cache_.reading()) note_pipeline_during_save(created_us);
    if (p) driver_cache_.note_created(frame_);
    if (p && (cs.cache_key.a | cs.cache_key.b) != 0 && !per_stage_keys_ && !stripped) {
        PipelineRecord rec;
        memset(&rec, 0, sizeof rec);
        rec.kind = 1;
        rec.key = key;
        rec.base[S] = rec.module[S] = cs.cache_key;
        record_pipeline(rec);
    }
    return compute_pipelines_[h] = p;
}

bool Executor::overlay_unchanged(uint64_t address, Overlay& o) {
    if (watch_overlays_ && o.watched) {  // vigilado: sin páginas escritas por la CPU desde la última revisión, sigue igual
        overlay_watch_runs_.clear();
        nnsdk::watch::collect(address, o.size, o.watch_keys, &overlay_watch_runs_);
        if (overlay_watch_runs_.empty()) {
            if (prof_timing_) fine_.ov_watched++;
            return true;
        }
    } else if (watch_overlays_ && o.size > kOverlayWatchMin && nnsdk::watch::watchable(address, o.size)) {
        // Antes de la huella: lo que la CPU escriba desde aquí salta; lo de antes lo ve la huella de abajo.
        o.watch_keys.clear();
        nnsdk::watch::collect(address, o.size, o.watch_keys, nullptr);
        o.watched = true;
    }
    const auto prof_t0 = prof_now();
    const bool same = range_hash(reinterpret_cast<const uint8_t*>(address), o.size) == o.cpu_hash;
    if (prof_timing_) {
        fine_.ov_hashes++;
        fine_.ov_hash_bytes += o.size;
        fine_.ov_hash_ns += ns_since(prof_t0);
    }
    if (same) return true;
    std::lock_guard lock(wb_mutex_);
    const uint64_t now = range_hash(reinterpret_cast<const uint8_t*>(address), o.size);
    if (now == o.cpu_hash) return true;
    auto it = delivered_hash_.find(address);
    if (it == delivered_hash_.end() || it->second != now) return false;
    o.cpu_hash = now;  // lo entregamos nosotros: la GPU sigue mandando
    return true;
}

const Executor::Overlay* Executor::find_overlay(uint64_t address, uint64_t size, VkDeviceSize* offset) {
    if (overlays_.empty() || !address || !overlay_filter_[overlay_slot(address >> 16)]) return nullptr;
    auto it = overlays_.upper_bound(address);
    if (it == overlays_.begin()) return nullptr;
    --it;
    Overlay& o = it->second;
    if (address + size > it->first + o.size) return nullptr;
    if (o.checked != submit_index_) {  // ¿la CPU ha reescrito el rango? entonces manda la memoria del juego
        o.checked = submit_index_;
        if (!overlay_unchanged(it->first, o)) {
            erase_reason(rewrote_why(o, "la CPU reescribió su rango (su huella cambió), al buscarlo"), address, size);  // DEPURACIÓN
            erase_overlay(it);
            return nullptr;
        }
    }
    *offset = o.offset + (address - it->first);
    return &o;
}

Executor::BufferCopySource Executor::copy_from_overlay(uint64_t src, uint64_t dst, uint64_t size,
                                                      std::vector<uint8_t>& snapshot) {
    VkDeviceSize src_offset = 0;
    VkBuffer src_buffer = VK_NULL_HANDLE;
    const uint8_t* host = nullptr;
    const bool overlap = (src < dst ? dst - src : src - dst) < size;
    const Overlay* so = overlap ? nullptr : find_overlay(src, size, &src_offset);
    if (so) {
        src_buffer = so->buffer;
        host = so->data + (src_offset - so->offset);
    } else {
        // Intersecting overlays can extend beyond either end of the source. Snapshot the
        // exact bytes before allocating the destination, including GPU writes in this submit.
        if (!gather_overlays(src, size, &src_buffer, &src_offset, &host)) return BufferCopySource::Error;
        if (!src_buffer) return BufferCopySource::Guest;
    }
    VkDeviceSize dst_offset = 0;
    if (vigilar_nan_) {  // DEPURACIÓN
        char b[96];
        snprintf(b, sizeof b, "CopyBuffer en la GPU desde el overlay de %#llx, frame %llu", static_cast<unsigned long long>(src),
                 static_cast<unsigned long long>(frame_));
        writer_ = b;
    }
    const Overlay* d = make_overlay(dst, size, &dst_offset);  // puede invalidar `so`: ya no se usa
    if (!d) {
        // A large destination may not fit in the overlay arena. The CPU copy must still
        // read the resolved source, whose allocation lives until this submission completes.
        full_barrier();
        VkMemoryBarrier visibility{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        visibility.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        visibility.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                             1, &visibility, 0, nullptr, 0, nullptr);
        const auto previous = submitted_.load();
        const VkFence fence = inflight_[cur_].fence;
        splitting_ = true;
        finish();
        const bool ok = submitted_.load() != previous &&
                        vkWaitForFences(vk()->device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
        if (ok) snapshot.assign(host, host + size);
        splitting_ = false;
        begin();
        return ok ? BufferCopySource::Snapshot : BufferCopySource::Error;
    }
    note_gpu_write_vs_mirrors(dst, size, "CopyBuffer");
    if (vigilar_nan_) {  // DEPURACIÓN: el destino lo escribe la GPU (written/writers no se marcan aquí)
        auto it = overlays_.upper_bound(dst);
        if (it != overlays_.begin()) {
            Provenance& p = provenance_[std::prev(it)->first];
            p.last = writer_;
            p.last_frame = frame_;
            p.gpu_writes++;
            p.submit = submitted_.load(std::memory_order_relaxed) + 1;
        }
    }
    split_why_ = "copia en la GPU desde un overlay (CopyBuffer)";
    barrier();
    VkBufferCopy region{src_offset, dst_offset, size};
    note_gpu_work();
    vkCmdCopyBuffer(cmd_, src_buffer, d->buffer, 1, &region);
    barrier();
    // Lo de dst ya no está en la memoria del juego: para uniones y compactaciones es de la GPU (sin `written`: la
    // devolución sigue como antes).
    if (auto it = overlays_.upper_bound(dst);
        keep_gpu_overlays_ && it != overlays_.begin() && dst < std::prev(it)->first + std::prev(it)->second.size)
        std::prev(it)->second.writers |= kWrittenByCopy;
    // A copy for the CPU to read (readback_copies_): returned to the game's memory once this submit finishes.
    if (readback_copies_ && !write_back_on_)
        if (auto it = overlays_.upper_bound(dst); it != overlays_.begin()) {
            --it;
            Overlay& o = it->second;
            if (dst < it->first + o.size && o.size <= kWriteBackMax) {
                std::lock_guard lock(wb_mutex_);
                if (!wb_state_.count(it->first)) {
                    wb_reference(it->first, o.size);
                    static int logged = 0;
                    if (logged++ < 8)
                        NVN_LOG("copy for the CPU to read: %#llx+%llu goes back to the game's memory\n",
                                static_cast<unsigned long long>(it->first), static_cast<unsigned long long>(o.size));
                }
                o.cpu_reads = true;
                o.written = submitted_.load(std::memory_order_relaxed) + 1;
                any_cpu_reads_ = true;
            }
        }
    return BufferCopySource::Copied;
}

void Executor::note_gpu_write_vs_mirrors(uint64_t address, uint64_t size, const char* what) {
    if (mirrors_.empty() || !address || !size) return;
    static std::set<std::pair<uint64_t, const char*>> logged;  // (espejo, tipo): los tipos son literales
    auto it = mirrors_.upper_bound(address + size - 1);           // el primero que empieza después del rango
    for (int k = 0; k < 4 && it != mirrors_.begin(); k++) {       // los anteriores (pueden solaparse entre ellos)
        --it;
        if (it->first + it->second.size <= address) continue;
        // Los grandes (la caché de sombras) siempre; los pequeños (ventanas de índices…), 32 como mucho.
        if ((it->second.size < (16ull << 20) && logged.size() >= 32) || !logged.insert({it->first, what}).second) continue;
        NVN_LOG("espejo %#llx+%llu KB: la GPU escribe dentro (%s, %#llx+%llu, frame %llu): quien lea el espejo no lo ve\n",
                static_cast<unsigned long long>(it->first), static_cast<unsigned long long>(it->second.size >> 10), what,
                static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
                static_cast<unsigned long long>(frame_));
    }
}

// El espejo de [address, address + size) al día para este envío del juego, o nullptr (pequeño, sin memoria o sin
// alinear: se copia como siempre). Puede cerrar el render pass para subir páginas.
const Executor::Mirror* Executor::mirror(uint64_t address, uint64_t size, VkDeviceSize* offset, bool may_upload) {
    if (size < kMirrorMin || !address) return nullptr;
    VkContext* c = vk();
    if (size > c->props.limits.maxStorageBufferRange) return nullptr;
    const VkDeviceSize align = std::max<VkDeviceSize>(c->props.limits.minStorageBufferOffsetAlignment, 16);
    Mirror* m = nullptr;
    uint64_t base = 0;
    if (auto it = mirrors_.upper_bound(address); it != mirrors_.begin()) {
        --it;
        if (address + size <= it->first + it->second.size && (address - it->first) % align == 0) m = &it->second, base = it->first;
    }
    if (!may_upload && (!m || m->checked != submit_index_)) return nullptr;
    if (!m) {
        if (auto it = mirrors_.find(address); it != mirrors_.end()) {  // más pequeño que esto: se sustituye
            mirror_bytes_ -= it->second.size * (it->second.direct ? Mirror::kVersions : 1);
            retired_mirrors_.push_back(std::move(it->second));
            mirrors_.erase(it);
        }
        // Over the budget: let go of the mirrors no submit of this or the last frame read, the longest unread first (they
        // were only waiting for reap_mirrors' 300 frames), before giving up on this one
        const uint64_t budget = low_memory() ? kMirrorBudget / 2 : kMirrorBudget;
        while (mirror_bytes_ + size > budget) {
            auto oldest = mirrors_.end();
            for (auto it = mirrors_.begin(); it != mirrors_.end(); ++it)
                if (it->second.used_frame + 1 < frame_ && (oldest == mirrors_.end() || it->second.used_frame < oldest->second.used_frame))
                    oldest = it;
            if (oldest == mirrors_.end()) break;
            mirror_bytes_ -= oldest->second.size * (oldest->second.direct ? Mirror::kVersions : 1);
            retired_mirrors_.push_back(std::move(oldest->second));  // (destroyed once its last submit is done)
            mirrors_.erase(oldest);
        }
        if (mirror_bytes_ + size > budget) {
            static int logged = 0;
            if (logged++ < 4) NVN_LOG("espejos: sin presupuesto para %llu MB más\n", static_cast<unsigned long long>(size >> 20));
            return nullptr;
        }
        LoadTimer timer(*this, kLdMirrorNew, size);
        Mirror fresh;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        if (vkCreateBuffer(c->device, &bi, nullptr, &fresh.buffer) != VK_SUCCESS) return nullptr;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(c->device, fresh.buffer, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = find_memory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(c->device, &ai, nullptr, &fresh.memory) != VK_SUCCESS) {
            vkDestroyBuffer(c->device, fresh.buffer, nullptr);
            return nullptr;
        }
        vkBindBufferMemory(c->device, fresh.buffer, fresh.memory, 0);
        fresh.size = size;
        fresh.watched = nnsdk::watch::watchable(address, size);
        mirror_bytes_ += size;
        NVN_LOG("espejo nuevo %#llx+%llu KB%s (%llu MB en espejos)\n", static_cast<unsigned long long>(address),
                static_cast<unsigned long long>(size >> 10), fresh.watched ? " vigilado" : "",
                static_cast<unsigned long long>(mirror_bytes_ >> 20));
        m = &(mirrors_[address] = std::move(fresh));
        base = address;
        // DIAGNÓSTICO (ver note_gpu_write_vs_mirrors): overlays que la GPU ya escribió dentro del espejo nuevo.
        uint32_t written = 0;
        uint64_t first_at = 0, first_size = 0;
        auto o = overlays_.upper_bound(address);
        if (o != overlays_.begin()) --o;
        for (; o != overlays_.end() && o->first < address + size; ++o)
            if (o->second.writers && o->first + o->second.size > address && written++ == 0)
                first_at = o->first, first_size = o->second.size;
        if (written)
            NVN_LOG("espejo nuevo %#llx+%llu KB: %u overlays escritos por la GPU dentro (el primero %#llx+%llu): quien lea el "
                    "espejo no los ve\n", static_cast<unsigned long long>(address), static_cast<unsigned long long>(size >> 10),
                    written, static_cast<unsigned long long>(first_at), static_cast<unsigned long long>(first_size));
    }
    m->used_frame = frame_;
    m->used_submit = submitted_.load(std::memory_order_relaxed) + 1;
    *offset = address - base;
    m->unread = mirror_batch_;
    if (m->direct) m->version_submit[m->cur] = m->used_submit;
    if (m->checked == submit_index_) return m;
    // Revisión: las páginas que cambiaron (todas la primera vez), en tramos seguidos, por el staging. Vigilado, lo
    // dicen las escrituras que saltaron (y las páginas quedan cerradas antes de copiarlas); si no, el hash.
    std::vector<std::pair<size_t, size_t>> byte_runs;  // (offset, bytes)
    std::optional<LoadTimer> check_timer;
    check_timer.emplace(*this, kLdMirrorCheck);
    if (m->watched) {
        nnsdk::watch::collect(base, m->size, m->pages, &byte_runs);
    } else {
        const size_t n = static_cast<size_t>((m->size + kMirrorPage - 1) / kMirrorPage);
        const bool first = m->pages.size() != n;
        if (first) m->pages.assign(n, 0);
        for (size_t k = 0; k < n; k++) {
            const uint64_t at = uint64_t{k} * kMirrorPage;
            const uint64_t len = std::min(kMirrorPage, m->size - at);
            const uint64_t h = range_hash(reinterpret_cast<const uint8_t*>(base + at), static_cast<size_t>(len));
            if (!first && h == m->pages[k]) continue;
            m->pages[k] = h;
            if (!byte_runs.empty() && byte_runs.back().first + byte_runs.back().second == at)
                byte_runs.back().second += len;
            else
                byte_runs.push_back({at, len});
        }
    }
    check_timer.reset();
    uint64_t copy_bytes = 0;
    for (auto [run_at, run_len] : byte_runs) copy_bytes += run_len;
    if (!m->direct && copy_bytes) {
        m->hot = copy_bytes * 4 >= m->size * 3 ? m->hot + 1 : 0;
        if (m->hot >= 8 && !m->pages.empty() && !NVN_FLAG("no_direct_mirrors")) promote_mirror(*m);
    }
    if (m->direct) {
        // A copy no submit in flight reads (the in-use one read by this submit too: never that one); none free, the
        // oldest once its submit is done. Then all of it from the game's memory: a direct mirror changes whole.
        m->checked = submit_index_;
        if (!copy_bytes) return m;
        const uint64_t done = completed_.load(std::memory_order_acquire);
        int pick = -1;
        for (int k = 1; k <= Mirror::kVersions && pick < 0; k++) {
            const int v = (m->cur + k) % Mirror::kVersions;
            if (v != m->cur && m->version_submit[v] <= done) pick = v;
        }
        if (pick < 0) {
            int oldest = (m->cur + 1) % Mirror::kVersions;
            for (int v = 0; v < Mirror::kVersions; v++)
                if (v != m->cur && m->version_submit[v] < m->version_submit[oldest]) oldest = v;
            if (m->version_submit[oldest] > submitted_.load(std::memory_order_relaxed) ||
                !wait_submit_fence(m->version_submit[oldest], 250'000'000))
                return m;  // (nothing to write into: this submit reads the last copy, as a late upload would)
            pick = oldest;
        }
        std::optional<LoadTimer> copy_timer2;
        copy_timer2.emplace(*this, kLdMirrorCopy, m->size);
        memcpy(m->version_data[pick], reinterpret_cast<const void*>(base), m->size);
        prof_direct_bytes_ += m->size;
        prof_mirror_top_[base] += m->size;
        m->cur = pick;
        m->buffer = m->versions[pick];
        m->version_submit[pick] = submitted_.load(std::memory_order_relaxed) + 1;
        return m;
    }
    std::optional<LoadTimer> copy_timer;
    if (copy_bytes) copy_timer.emplace(*this, kLdMirrorCopy, copy_bytes);
    std::vector<std::pair<Staging::Alloc, VkBufferCopy>> copies;
    for (auto [run_at, run_len] : byte_runs)
        for (uint64_t at = run_at, end = run_at + run_len; at < end;) {
            const uint64_t len = std::min<uint64_t>(end - at, kStagingChunk - 256);
            Staging::Alloc a = staging_->alloc(len, 16);
            if (!a.data) {  // sin staging: que se revise entero la próxima vez
                m->pages.clear();
                return nullptr;
            }
            memcpy(a.data, reinterpret_cast<const void*>(base + at), len);
            copies.push_back({a, VkBufferCopy{a.offset, at, len}});
            prof_mirror_bytes_ += len;
            prof_mirror_top_[base] += len;
            at += len;
        }
    m->checked = submit_index_;
    if (!copies.empty() && mirror_batch_) {  // preload_mirrors: fuera de todo pase, la barrera de antes una vez para todos
        if (!mirror_batch_barrier_) full_barrier(), mirror_batch_barrier_ = true;
        note_gpu_work();
        for (auto& [a, region] : copies) vkCmdCopyBuffer(cmd_, a.buffer, m->buffer, 1, &region);
        prof_mirror_preloads_++;
        return m;
    }
    if (!copies.empty()) {
        split_why_ = "subida de espejo";
        end_render_pass();
        targets_changed_ = true;
        full_barrier();  // lo que lo leía antes, terminado
        note_gpu_work();
        for (auto& [a, region] : copies) vkCmdCopyBuffer(cmd_, a.buffer, m->buffer, 1, &region);
        full_barrier();
    }
    return m;
}

// Por defecto (<datos>/no_mirrors_at_start lo quita): los espejos que se leyeron en el último frame se revisan (y se suben sus páginas cambiadas)
// al empezar el envío, fuera de todo pase y con dos barreras completas para todos. Sin esto, el primer draw de cada envío
// que lee uno (los de 4 MB se reescriben enteros cada frame: ~18 MB/frame en una cacería) cierra el pase abierto (en la
// escena: el tiler guarda y vuelve a cargar sus adjuntos) y pone dos barreras completas por espejo. Lo que el juego
// escribió antes de enviar ya está en su memoria; uno que este envío no lea se revisa de más (memcpy de lo cambiado).
// Tras una misión (Adreno 825, 2026-11-19) el juego reescribía unos diez de 4 MB por frame y el pre-pase de profundidad
// salía partido en cinco por sus subidas: 10 ms de GPU en vez de 2,5.
void Executor::preload_mirrors() {
    if (in_render_pass_ || mirrors_.empty()) return;  // al empezar el envío no hay pase abierto
    mirror_batch_ = true;
    mirror_batch_barrier_ = false;
    for (auto& [base, m] : mirrors_) {
        if (m.used_frame + 1 < frame_ || m.checked == submit_index_) continue;
        const uint64_t used_frame = m.used_frame;
        VkDeviceSize offset = 0;
        mirror(base, m.size, &offset);
        m.used_frame = used_frame;  // si nadie lo lee, sigue envejeciendo (reap_mirrors); used_submit sí cuenta la copia
    }
    mirror_batch_ = false;
    if (mirror_batch_barrier_) full_barrier();  // las copias, visibles para todo lo que venga
}

bool Executor::pool_range(uint64_t address, uint64_t* start, uint64_t* size) {
    const bool memo = !NVN_FLAG("no_pool_memo");
    const uint64_t generation = g_memory_pools.generation();  // antes de recorrer: lo que cambie durante, invalida el memo
    if (memo && generation == pool_memo_generation_)
        for (const PoolMemo& m : pool_memo_)
            if (m.size && address >= m.start && address < m.start + m.size) {
                if (prof_timing_) fine_.pool_memo++;
                *start = m.start;
                *size = m.size;
                return true;
            }
    if (prof_timing_) fine_.pool_scans++;
    bool found = false;
    g_memory_pools.find_if([&](const void*, const MemoryPool& pool) {
        if (prof_timing_) fine_.pool_visits++;
        const auto s = reinterpret_cast<uint64_t>(pool.storage);
        if (address < s || address >= s + pool.size) return false;
        *start = s;
        *size = pool.size;
        found = true;
        return true;
    });
    if (!found || !memo) return found;
    if (generation != pool_memo_generation_) {
        for (PoolMemo& m : pool_memo_) m = {};
        pool_memo_generation_ = generation;
        pool_memo_next_ = 0;
    }
    pool_memo_[pool_memo_next_] = {*start, *size};
    pool_memo_next_ = (pool_memo_next_ + 1) % kPoolMemo;
    return true;
}

bool Executor::index_window(uint64_t address, uint64_t* base, uint64_t* size) {
    uint64_t start = 0, pool_size = 0;
    if (!pool_range(address, &start, &pool_size)) return false;
    *base = std::max<uint64_t>(start, address & ~(kIndexWindowAlign - 1));
    *size = std::min<uint64_t>(start + pool_size - *base, kIndexWindow);
    return *size >= kMirrorMin;
}

void Executor::destroy_mirror(Mirror& m) {
    VkContext* c = vk();
    if (m.direct) {
        for (int v = 0; v < Mirror::kVersions; v++) {
            if (m.versions[v]) vkDestroyBuffer(c->device, m.versions[v], nullptr);
            if (m.version_memory[v]) vkFreeMemory(c->device, m.version_memory[v], nullptr);  // (unmapped with it)
            m.versions[v] = VK_NULL_HANDLE;
            m.version_memory[v] = VK_NULL_HANDLE;
            m.version_data[v] = nullptr;
        }
        m.direct = false;
        m.buffer = VK_NULL_HANDLE;  // (one of the versions)
    }
    if (m.buffer) vkDestroyBuffer(c->device, m.buffer, nullptr);
    if (m.memory) vkFreeMemory(c->device, m.memory, nullptr);
    m.buffer = VK_NULL_HANDLE;
    m.memory = VK_NULL_HANDLE;
}

// See Mirror::direct. The device-local copy is retired (freed once the GPU is done with it). Host-visible memory is
// the GPU's own on these phones (one memory for both); without a host-coherent type the mirror stays as it was.
bool Executor::promote_mirror(Mirror& m) {
    VkContext* c = vk();
    if (mirror_bytes_ + m.size * Mirror::kVersions > (low_memory() ? kMirrorBudget / 2 : kMirrorBudget)) return false;
    Mirror fresh;
    for (int v = 0; v < Mirror::kVersions; v++) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = m.size;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                   VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        VkMemoryRequirements req;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        void* data = nullptr;
        bool ok = vkCreateBuffer(c->device, &bi, nullptr, &fresh.versions[v]) == VK_SUCCESS;
        if (ok) {
            vkGetBufferMemoryRequirements(c->device, fresh.versions[v], &req);
            ai.allocationSize = req.size;
            ai.memoryTypeIndex = find_memory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (ai.memoryTypeIndex == UINT32_MAX)
                ai.memoryTypeIndex = find_memory(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            ok = ai.memoryTypeIndex != UINT32_MAX &&
                 vkAllocateMemory(c->device, &ai, nullptr, &fresh.version_memory[v]) == VK_SUCCESS &&
                 vkBindBufferMemory(c->device, fresh.versions[v], fresh.version_memory[v], 0) == VK_SUCCESS &&
                 vkMapMemory(c->device, fresh.version_memory[v], 0, VK_WHOLE_SIZE, 0, &data) == VK_SUCCESS;
        }
        if (!ok) {
            fresh.direct = true;  // (destroy_mirror frees what was made)
            destroy_mirror(fresh);
            return false;
        }
        fresh.version_data[v] = static_cast<uint8_t*>(data);
    }
    Mirror old;  // the device-local copy, retired
    old.buffer = m.buffer;
    old.memory = m.memory;
    old.used_submit = submitted_.load(std::memory_order_relaxed) + 1;
    retired_mirrors_.push_back(std::move(old));
    for (int v = 0; v < Mirror::kVersions; v++) {
        m.versions[v] = fresh.versions[v];
        m.version_memory[v] = fresh.version_memory[v];
        m.version_data[v] = fresh.version_data[v];
        m.version_submit[v] = 0;
    }
    m.memory = VK_NULL_HANDLE;
    m.direct = true;
    m.cur = 0;
    m.buffer = m.versions[0];
    mirror_bytes_ += m.size * (Mirror::kVersions - 1);
    prof_mirrors_direct_++;
    NVN_LOG("espejo directo: %llu KB en 3 copias que lee la GPU (se reescribía entero en cada frame)\n",
            static_cast<unsigned long long>(m.size >> 10));
    return true;
}

// Los espejos sin usar en 10 s (o sustituidos) se sueltan cuando la GPU ya no los puede estar leyendo.
void Executor::reap_mirrors() {
    const uint64_t done = completed_.load(std::memory_order_acquire);
    for (size_t k = 0; k < retired_mirrors_.size();)
        if (retired_mirrors_[k].used_submit <= done) {
            destroy_mirror(retired_mirrors_[k]);
            retired_mirrors_[k] = std::move(retired_mirrors_.back());
            retired_mirrors_.pop_back();
        } else {
            k++;
        }
    for (auto it = mirrors_.begin(); it != mirrors_.end();)
        if (it->second.used_frame + 300 < frame_ && it->second.used_submit <= done) {
            mirror_bytes_ -= it->second.size * (it->second.direct ? Mirror::kVersions : 1);
            destroy_mirror(it->second);
            it = mirrors_.erase(it);
        } else {
            ++it;
        }
}

Staging::Alloc Executor::alloc_read_snapshot(uint64_t address, uint64_t size) {
    if (size > kStagingChunk) {
        uint64_t base = 0, bytes = 0;
        if (!pool_range(address, &base, &bytes) || size > bytes - (address - base)) {
            NVN_LOG("buffer read outside its memory pool: %#llx + %llu bytes\n",
                    static_cast<unsigned long long>(address), static_cast<unsigned long long>(size));
            return {};
        }
    }
    return staging_->alloc_contiguous(size, 16);
}

bool Executor::gather_overlays(uint64_t address, uint64_t size, VkBuffer* buffer, VkDeviceSize* offset, const uint8_t** host) {
    *buffer = VK_NULL_HANDLE;
    if (!size || overlays_.empty()) return true;
    if (size <= 65536 && !overlay_maybe_in(address, size)) return true;
    auto it = first_overlay_touching(address);
    std::vector<std::pair<const Overlay*, uint64_t>> parts;  // overlay e inicio de su rango
    while (it != overlays_.end() && it->first < address + size) {
        Overlay& o = it->second;
        // Como find_overlay: si la CPU reescribió el rango desde que lo escribió la GPU, manda la CPU.
        if (o.checked != submit_index_) {
            o.checked = submit_index_;
            if (!overlay_unchanged(it->first, o)) {
                erase_reason(rewrote_why(o, "la CPU reescribió su rango (su huella cambió), al juntar overlays"), address,
                             size);  // DEPURACIÓN
                it = erase_overlay(it);
                continue;
            }
        }
        parts.push_back({&o, it->first});
        ++it;
    }
    if (parts.empty()) return true;
    Staging::Alloc al = alloc_read_snapshot(address, size);
    if (!al.data) return false;
    {
        std::lock_guard lock(wb_mutex_);
        memcpy(al.data, reinterpret_cast<const void*>(address), size);
    }
    split_why_ = "unión de rangos de lectura";
    barrier();
    for (auto [o, start] : parts) {
        const uint64_t from = std::max(address, start), to = std::min(address + size, start + o->size);
        VkBufferCopy region{o->offset + (from - start), al.offset + (from - address), to - from};
        note_gpu_work();
        vkCmdCopyBuffer(cmd_, o->buffer, al.buffer, 1, &region);
    }
    barrier();
    *buffer = al.buffer;
    *offset = al.offset;
    if (host) *host = al.data;
    return true;
}

bool Executor::read_vertex_stream(BufferRange range, bool allow_mirror, VkBuffer& buffer, VkDeviceSize& offset,
                                  bool& gpu_vertices) {
    note_read(0, range.address, range.size);
    if (const Overlay* o = find_overlay(range.address, range.size, &offset)) {
        buffer = o->buffer;
    } else if (!gather_overlays(range.address, range.size, &buffer, &offset)) {
        return false;
    }
    if (buffer) {
        gpu_vertices = true;
        return true;
    }
    if ((allow_mirror || range.size > kStagingChunk) && range.size >= kMirrorMin) {
        if (const Mirror* m = mirror(range.address, range.size, &offset)) {
            buffer = m->buffer;
            return true;
        }
    }
    if (const auto* e = buffer_cache_.get(range.address, range.size, -1, submit_index_)) {
        buffer = e->buffer;
        offset = e->offset;
        return true;
    }
    auto copy = alloc_read_snapshot(range.address, range.size);
    if (!copy.data) return false;
    memcpy(copy.data, reinterpret_cast<const void*>(range.address), range.size);
    buffer = copy.buffer;
    offset = copy.offset;
    return true;
}

// Un hueco libre (de un overlay tirado cuyo último envío ya terminó) antes que la arena: el más justo en el que quepa,
// sin gastar uno de más del doble. Los huecos están alineados a 256 como las reservas de la arena.
bool Executor::alloc_overlay_slot(VkDeviceSize need, Staging::Alloc* a, VkDeviceSize* slot_size, uint64_t size) {
    *a = {VK_NULL_HANDLE, 0, nullptr};
    *slot_size = need;
    if (recycle_overlays_ && !overlay_free_.empty()) {
        const uint64_t done = completed_.load(std::memory_order_acquire);
        size_t best = overlay_free_.size();
        for (size_t k = 0; k < overlay_free_.size(); k++) {
            const FreeSlot& f = overlay_free_[k];
            if (f.free_after > done || f.size < need || f.size > 2 * need) continue;
            if (best == overlay_free_.size() || f.size < overlay_free_[best].size) best = k;
        }
        if (best < overlay_free_.size()) {
            const FreeSlot f = overlay_free_[best];
            overlay_free_[best] = overlay_free_.back();
            overlay_free_.pop_back();
            *a = {f.buffer, f.offset, f.data};
            *slot_size = f.size;
            prof_overlay_reused_bytes_ += size;
            return true;
        }
    }
    *a = overlay_arena_.alloc(need, 256);
    if (!a->data) return false;
    overlay_used_ += need;
    prof_overlay_new_bytes_ += size;
    return true;
}

// Antes un rango escrito nuevo que solapaba otros los sustituía: el nuevo salía de la memoria del juego y lo que la GPU
// había escrito en los sustituidos se perdía (en una cacería, 144 de 200 tiradas de lo de la GPU eran esto; el jugador
// veía geometría donde no tocaba). Ahora el overlay nuevo se llena con la memoria del juego y encima, en la GPU, lo de cada
// overlay que cae ENTERO dentro del rango y aún vale (la CPU no reescribió el suyo: manda la CPU, como siempre): lo que
// tendría la memoria en la Switch, donde la GPU escribe la memoria de verdad. La unión es justo el rango pedido. Uno que
// solo lo solapa en parte no se mete (se sustituye como antes): en este montón hay uniforms de la CPU de 256 bytes entre
// las mallas y ese solape suele ser memoria que el juego ya usa para otro buffer; meterlo haría crecer la unión y lo que
// la CPU escriba al lado la tiraría entera. Para una lectura (for_read: un SSBO o el origen de una copia que cae en parte
// en lo de la GPU), solo si ninguno de la GPU sobresale y hasta kMergeReadMax: una lectura no tira nada de la GPU.
const Executor::Overlay* Executor::merge_overlays(uint64_t address, uint64_t size, VkDeviceSize* offset, bool for_read) {
    if (!keep_gpu_overlays_ || !size || size + 256 > kStagingChunk || (for_read && (!merge_reads_ || size > kMergeReadMax)) ||
        !overlay_maybe_in(address, size))
        return nullptr;
    const uint64_t end = address + size;
    std::vector<uint64_t> keep_at;  // los que se conservan (por su inicio), en orden
    bool straddles = false;         // alguno de la GPU que sobresale del rango (y no se parte)
    std::vector<uint64_t> split_at;  // los que sobresalen y se parten (split_overlays_)
    for (auto it = first_overlay_touching(address); it != overlays_.end() && it->first < end;) {
        Overlay& o = it->second;
        if (!o.writers || !o.buffer) {
            ++it;
            continue;
        }
        if (o.checked != submit_index_) {  // como find_overlay: si la CPU reescribió su rango, manda la memoria del juego
            o.checked = submit_index_;
            if (!overlay_unchanged(it->first, o)) {
                erase_reason(rewrote_why(o, "la CPU reescribió su rango (su huella cambió), al unirlo a un rango nuevo"),
                             address, size);  // DEPURACIÓN
                it = erase_overlay(it);
                continue;
            }
        }
        if (it->first < address || it->first + o.size > end) {
            // The game carves the same memory into ranges with different edges from frame to frame (GPU-made vertices
            // and particles: A, then B over the end of A and the start of C). Throwing A and C away left their outside
            // parts with the game's stale memory, and whatever drew from there blew up for a frame (big black
            // triangles on the Adreno 830). Split instead: the inside part goes into the union like the others, the
            // outside parts stay overlays of their own. The union never grows past the range asked for.
            if (split_overlays_) split_at.push_back(it->first);
            else straddles = true;
            ++it;
            continue;
        }
        keep_at.push_back(it->first);
        ++it;
    }
    if (for_read && straddles) return nullptr;  // ese lector ve la memoria del juego (watch_nan lo dice: note_partial_read)
    if (keep_at.empty() && split_at.empty()) return nullptr;
    std::optional<LoadTimer> timer;  // la de una escritura ya la mide make_overlay
    if (for_read) timer.emplace(*this, kLdOverlay, size);
    Staging::Alloc a;
    VkDeviceSize slot_size = 0;
    if (!alloc_overlay_slot(size + 256, &a, &slot_size, size)) return nullptr;
    // Lo que se conserva (buffer y desfase de cada uno): sus huecos vuelven a overlay_free_ al tirarlos, libres cuando
    // acabe este envío, así que las copias de abajo (de este envío) aún los leen. Los demás que tocan el rango, fuera.
    struct Part {
        VkBuffer buffer;
        VkDeviceSize offset;
        uint64_t start, size;
    };
    std::vector<Part> parts;
    // The outside parts of split overlays: a new overlay each, filled with the game's memory and then, on the GPU, with
    // what the old one had there
    struct Piece {
        uint64_t start, size;
        VkBuffer from;
        VkDeviceSize from_offset;
        uint8_t writers;
        uint64_t written;
    };
    std::vector<Piece> pieces;
    uint8_t writers = 0;
    uint64_t written = 0;
    Provenance merged_prov;  // DEPURACIÓN (watch_nan): la procedencia de las partes, sumada
    for (auto it = first_overlay_touching(address); it != overlays_.end() && it->first < end;) {
        const Overlay& o = it->second;
        const bool part = std::find(keep_at.begin(), keep_at.end(), it->first) != keep_at.end();
        const bool split = std::find(split_at.begin(), split_at.end(), it->first) != split_at.end();
        // DEPURACIÓN: los que sobresalen (solo con un rango escrito) se sustituyen como antes; los de dentro sin nada de
        // la GPU son la copia de la memoria del juego (no se pierde nada).
        erase_reason(it->first < address || it->first + o.size > end ? kWhyStraddled : kWhyReplaced, address, size);
        if (split) {
            const uint64_t lo = std::max(it->first, address), hi = std::min(it->first + o.size, end);
            parts.push_back({o.buffer, o.offset + (lo - it->first), lo, hi - lo});
            if (it->first < address)
                pieces.push_back({it->first, address - it->first, o.buffer, o.offset, o.writers, o.written});
            if (it->first + o.size > end)
                pieces.push_back({end, it->first + o.size - end, o.buffer, o.offset + (end - it->first), o.writers, o.written});
            writers |= o.writers;
            written = std::max(written, o.written);
        }
        if (part) {
            parts.push_back({o.buffer, o.offset, it->first, o.size});
            writers |= o.writers;
            written = std::max(written, o.written);
            if (auto pv = provenance_.find(it->first); pv != provenance_.end()) {
                merged_prov.gpu_writes += pv->second.gpu_writes;
                merged_prov.cpu_patches += pv->second.cpu_patches;
                if (pv->second.last_frame >= merged_prov.last_frame)
                    merged_prov.last = pv->second.last, merged_prov.last_frame = pv->second.last_frame;
            }
        }
        it = erase_overlay(it, !part && !split);  // lo de las partes va a la unión (o a sus trozos): no se pierde
    }
    // The pieces, before the union (each one its own slot; one that doesn't get a slot is lost as before)
    struct PieceCopy {
        VkBuffer from, to;
        VkDeviceSize from_offset, to_offset, size;
    };
    std::vector<PieceCopy> piece_copies;
    for (const Piece& pc : pieces) {
        if (pc.size + 256 > kStagingChunk) continue;
        Staging::Alloc pa;
        VkDeviceSize piece_slot = 0;
        if (!alloc_overlay_slot(pc.size + 256, &pa, &piece_slot, pc.size)) continue;
        const VkDeviceSize pskew = pc.start & 255;
        std::unique_lock wb_lock(wb_mutex_);
        memcpy(pa.data + pskew, reinterpret_cast<const void*>(pc.start), pc.size);
        Overlay& po = overlays_[pc.start];
        po.buffer = pa.buffer;
        po.offset = pa.offset + pskew;
        po.data = pa.data + pskew;
        po.size = pc.size;
        po.slot_offset = pa.offset, po.slot_size = piece_slot, po.slot_data = pa.data;
        po.cpu_hash = range_hash(reinterpret_cast<const uint8_t*>(pc.start), pc.size);
        po.checked = submit_index_;
        po.writers = pc.writers;
        po.written = pc.written;
        wb_lock.unlock();
        filter_overlay(pc.start, pc.size, +1);
        piece_copies.push_back({pc.from, po.buffer, pc.from_offset, po.offset, pc.size});
    }
    if (!pieces.empty()) {
        static int logged = 0;
        if (logged++ < 16 || logged % 1000 == 0)
            NVN_LOG("overlays partidos: %#llx+%llu keeps what the GPU wrote around it in %zu piece(s) (%d times)\n",
                    static_cast<unsigned long long>(address), static_cast<unsigned long long>(size), piece_copies.size(),
                    logged);
        prof_split_ += pieces.size();
    }
    const VkDeviceSize skew = address & 255;
    Overlay* made = nullptr;
    {
        std::unique_lock wb_lock(wb_mutex_);  // sin una entrega a medias en lo que se copia
        memcpy(a.data + skew, reinterpret_cast<const void*>(address), size);
        if (size <= kWriteBackMax && write_back_on_) wb_reference(address, size);  // lo de la CPU al crearlo
        Overlay& o = overlays_[address];
        o.buffer = a.buffer;
        o.offset = a.offset + skew;
        o.data = a.data + skew;
        o.size = size;
        o.slot_offset = a.offset, o.slot_size = slot_size, o.slot_data = a.data;
        o.cpu_hash = range_hash(reinterpret_cast<const uint8_t*>(address), size);
        o.checked = submit_index_;
        o.writers = writers;  // sigue siendo de la GPU (otra unión, compactación)
        o.written = written;
        o.merged = true;
        made = &o;
    }
    filter_overlay(address, size, +1);
    // Lo de la GPU encima de lo de la CPU, en orden con lo que escribía los viejos y con lo que lea o escriba el nuevo.
    split_why_ = "unión de overlays";
    barrier();  // cierra el render pass: las copias no van dentro de uno
    for (const PieceCopy& pc : piece_copies) {
        VkBufferCopy region{pc.from_offset, pc.to_offset, pc.size};
        note_gpu_work();
        vkCmdCopyBuffer(cmd_, pc.from, pc.to, 1, &region);
    }
    for (const Part& p : parts) {
        VkBufferCopy region{p.offset, made->offset + (p.start - address), p.size};
        note_gpu_work();
        vkCmdCopyBuffer(cmd_, p.buffer, made->buffer, 1, &region);
    }
    barrier();
    prof_merged_++;
    prof_merged_parts_ += parts.size();
    prof_merged_bytes_ += size;
    if (vigilar_nan_) {  // DEPURACIÓN
        Provenance& p = provenance_[address];
        p = merged_prov;
        p.created = (writer_.empty() ? std::string("?") : writer_) + " (unión)";
        p.created_frame = frame_;
        p.submit = submitted_.load(std::memory_order_relaxed) + 1;
    }
    static int logged = 0;  // siempre (con tope): dónde se conserva lo de la GPU que antes se perdía
    if (logged++ < 16 || logged % 1000 == 0)
        NVN_LOG("overlays unidos: %#llx+%llu (%s) conserva lo de la GPU de %zu overlays de dentro%s (%d veces)\n",
                static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
                for_read ? "una lectura" : "un rango escrito", parts.size(),
                straddles ? "; tira los que sobresalen" : "", logged);
    *offset = made->offset;
    return made;
}

const Executor::Overlay* Executor::make_overlay(uint64_t address, uint64_t size, VkDeviceSize* offset, bool may_copy) {
    if (const Overlay* o = find_overlay(address, size, offset)) return o;
    if (size + 256 > kStagingChunk) return nullptr;
    LoadTimer timer(*this, kLdOverlay, size);  // con la unión, si la hay
    if (may_copy)
        if (const Overlay* o = merge_overlays(address, size, offset, false)) return o;
    // Un rango nuevo que solape otros los sustituye (sin unión: nada de la GPU dentro, solo solapados en parte, sin sitio,
    // con el pase abierto o no_keep_overlays).
    {  // DEPURACIÓN: lo que la GPU escribió en los sustituidos se pierde (el nuevo sale de la memoria del juego)
        static int logged = 0;
        auto it = overlays_.upper_bound(address);
        if (it != overlays_.begin()) --it;
        for (; it != overlays_.end() && it->first < address + size && logged < 16; ++it)
            if (it->first + it->second.size > address && it->second.writers)
                logged++, NVN_LOG("overlay nuevo %#llx+%llu pisa %#llx+%llu escrito por la GPU: se pierde lo suyo\n",
                                  static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
                                  static_cast<unsigned long long>(it->first), static_cast<unsigned long long>(it->second.size));
    }
    if (overlay_maybe_in(address, size))
        for (auto it = first_overlay_touching(address); it != overlays_.end() && it->first < address + size;) {
            // DEPURACIÓN: el motivo de cada uno (el perfil los cuenta por motivo): solapado en parte, la CPU lo había
            // reescrito, el pase abierto o sustituido sin más.
            const char* why = kWhyReplaced;
            if (Overlay& o = it->second; o.writers) {
                if (it->first < address || it->first + o.size > address + size)
                    why = kWhyStraddled;
                else if (!keep_gpu_overlays_)
                    why = kWhyReplaced;  // como antes, sin la huella de más
                else if (o.checked != submit_index_ && !overlay_unchanged(it->first, o))
                    why = rewrote_why(o, "la CPU reescribió su rango (su huella cambió), al crear un overlay que lo solapa");
                else if (!may_copy)
                    why = kWhyPassOpen;
            }
            erase_reason(why, address, size);
            it = erase_overlay(it);
        }
    Staging::Alloc a;
    VkDeviceSize slot_size = 0;
    if (!alloc_overlay_slot(size + 256, &a, &slot_size, size)) return nullptr;
    const VkDeviceSize skew = address & 255;
    std::unique_lock wb_lock(wb_mutex_);  // sin una entrega a medias en lo que se copia
    memcpy(a.data + skew, reinterpret_cast<const void*>(address), size);
    // La referencia para la devolución (copia de lo de la CPU y vigilancia de sus páginas) solo con use_return: sin
    // ella eran un memcpy y una reserva del tamaño del overlay y un mprotect de sus páginas por overlay nuevo, y luego
    // las escrituras de la CPU del juego en esas páginas saltaban (SIGSEGV) para nada. Los overlays creados con la
    // devolución apagada no se devuelven hasta que se rehagan.
    if (size <= kWriteBackMax && write_back_on_) wb_reference(address, size);  // lo de la CPU al crearlo
    Overlay& o = overlays_[address];
    o.buffer = a.buffer;
    o.offset = a.offset + skew;
    o.data = a.data + skew;
    o.size = size;
    o.slot_offset = a.offset, o.slot_size = slot_size, o.slot_data = a.data;
    filter_overlay(address, size, +1);
    o.cpu_hash = range_hash(reinterpret_cast<const uint8_t*>(address), size);
    o.checked = submit_index_;
    *offset = o.offset;
    if (vigilar_nan_) {  // DEPURACIÓN: quién lo crea y lo que copia de la memoria del juego
        Provenance& p = provenance_[address];
        p = {};
        p.created = writer_.empty() ? std::string("?") : writer_;
        p.created_frame = frame_;
        p.submit = submitted_.load(std::memory_order_relaxed) + 1;
        static int logged = 0;
        if (logged++ < 800) {
            std::string mem;
            char b[16];
            for (uint64_t i = 0; i < std::min<uint64_t>(size / 4, 8); i++)
                snprintf(b, sizeof b, " %08x", reinterpret_cast<const uint32_t*>(address)[i]), mem += b;
            NVN_LOG("[vigía] overlay nuevo %#llx+%llu en el frame %llu, de %s; memoria del juego:%s\n",
                    static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
                    static_cast<unsigned long long>(frame_), p.created.c_str(), mem.c_str());
        }
    }
    return &o;
}

// Ver la declaración. Mismos rangos y tamaños que los descriptores del draw (draw(), "3. Descriptores").
void Executor::prepare_draw_overlays() {
    if (!keep_gpu_overlays_ || overlays_.empty()) return;
    VkDeviceSize off = 0;
    for (int s = 0; s < kStages; s++) {
        if (!st_.stages[s]) continue;
        const uint32_t base = kGraphicsStorageBase + kGraphicsStorageStride * s;
        for (const Resource& r : st_.stages[s]->shader.resources) {
            if (r.kind != Resource::StorageBuffer || r.cbuf_index != 0 || r.cbuf_offset < base ||
                r.cbuf_offset >= base + 16 * kMaxStorage)
                continue;
            const BufferRange& range = st_.bindings[s].storage[(r.cbuf_offset - base) / 16];
            if (!range.address || !range.size) continue;
            const uint64_t size = std::min<uint64_t>(range.size, kStagingChunk);
            if (r.is_written) {
                if (vigilar_nan_) {  // DEPURACIÓN: como en el bucle de los descriptores
                    const ProgramStage* vsp = st_.stages[static_cast<int>(Stage::Vertex)];
                    const ProgramStage* fsp = st_.stages[static_cast<int>(Stage::Fragment)];
                    char b[128];
                    snprintf(b, sizeof b, "draw (vs %016llx ps %016llx, etapa %s), frame %llu",
                             static_cast<unsigned long long>(vsp ? vsp->print : 0),
                             static_cast<unsigned long long>(fsp ? fsp->print : 0), nvnshader::stage_name(static_cast<Stage>(s)),
                             static_cast<unsigned long long>(frame_));
                    writer_ = b;
                }
                // Marcado ya: otro SSBO escrito del mismo draw que lo contenga lo conserva al unirse.
                if (make_overlay(range.address, size, &off)) note_overlay_written(range.address, kWrittenByGraphics);
            } else if (!find_overlay(range.address, size, &off)) {
                merge_overlays(range.address, size, &off, true);
            }
        }
    }
}

// Splitting a writable binding is safe when the dispatch has no descriptor that crosses either edge.
// A wider alias needs a contiguous backing shared with the writer for the duration of the dispatch.
static bool compute_binding_crosses(const StageBindings& bindings, const ProgramStage& cs, const BufferRange& write) {
    auto crosses = [&](BufferRange other) {
        if (!other.address || !other.size) return false;
        if (other.address < write.address) return write.address - other.address < other.size;
        const uint64_t at = other.address - write.address;
        return at < write.size && other.size > write.size - at;
    };
    for (const Resource& r : cs.shader.resources) {
        if (r.kind == Resource::UniformBuffer) {
            if (r.cbuf_index == kShaderConstantsBank && crosses({cs.constants, cs.constants_size})) return true;
            if (r.cbuf_index >= kUniformBankBase && r.cbuf_index - kUniformBankBase < kMaxUniforms &&
                crosses(bindings.uniforms[r.cbuf_index - kUniformBankBase])) return true;
        } else if (r.kind == Resource::StorageBuffer && r.cbuf_index == kComputeDriverBank) {
            if (r.cbuf_offset >= kComputeStorageBase) {
                const uint32_t index = (r.cbuf_offset - kComputeStorageBase) / 16;
                if (index < kMaxStorage && crosses(bindings.storage[index])) return true;
            } else if (r.cbuf_offset >= kComputeUbeBase && r.cbuf_offset < kComputeUbeBase + 16 * kComputeUbeCount) {
                const uint32_t index = kComputeUbeFirst + (r.cbuf_offset - kComputeUbeBase) / 16;
                if (index < kMaxUniforms && crosses(bindings.uniforms[index])) return true;
            }
        }
    }
    return false;
}

// Ver la declaración. Mismos rangos y tamaños que los descriptores del dispatch (los SSBO de la tabla de c0; los UBE son
// uniform buffers aunque se lean como storage: no se unen).
uint32_t Executor::prepare_dispatch_overlays() {
    if (!keep_gpu_overlays_ || overlays_.empty()) return 0;
    const StageBindings& sb = st_.bindings[static_cast<int>(Stage::Compute)];
    VkDeviceSize off = 0;
    uint32_t snapshot_reads = 0;
    for (const Resource& r : st_.compute->shader.resources) {
        if (r.kind != Resource::StorageBuffer || r.cbuf_index != kComputeDriverBank || r.cbuf_offset < kComputeStorageBase)
            continue;
        const uint32_t index = (r.cbuf_offset - kComputeStorageBase) / 16;
        if (index >= static_cast<uint32_t>(kMaxStorage)) continue;
        const BufferRange& range = sb.storage[index];
        if (!range.address || !range.size) continue;
        const uint64_t size = std::min<uint64_t>(range.size, kStagingChunk);
        if (r.is_written) {
            // The guest can reuse the neighbouring bytes independently. Give this write its own
            // CPU-change check before dispatch, while a split can still copy the old GPU contents.
            const Overlay* o = find_overlay(range.address, size, &off);
            if (o && o->size > size && o->writers && split_overlays_ && !write_back_on_ && !o->cpu_reads &&
                !compute_binding_crosses(sb, *st_.compute, {range.address, size}))
                o = merge_overlays(range.address, size, &off, false);
            if (o || make_overlay(range.address, size, &off)) note_overlay_written(range.address, kWrittenByCompute);
        } else if (!find_overlay(range.address, std::min<uint64_t>(range.size, vk()->props.limits.maxStorageBufferRange), &off)) {
            const auto part = first_overlay_touching(range.address);
            if (part == overlays_.end() || part->first >= range.address + range.size) continue;
            // Aliases used by this dispatch must share a live backing. Other read-only
            // windows (including the face normal remap) only need the bytes at this point.
            const auto& resources = st_.compute->shader.resources;
            const bool aliases_writer = std::any_of(resources.begin(), resources.end(), [&](const Resource& w) {
                if (w.kind != Resource::StorageBuffer || !w.is_written || w.cbuf_index != kComputeDriverBank ||
                    w.cbuf_offset < kComputeStorageBase) return false;
                const uint32_t slot = (w.cbuf_offset - kComputeStorageBase) / 16;
                if (slot >= kMaxStorage) return false;
                const auto& other = sb.storage[slot];
                return other.address && other.size && (range.address >= other.address ?
                    range.address - other.address < other.size : other.address - range.address < range.size);
            });
            if (aliases_writer) merge_overlays(range.address, size, &off, true);
            else snapshot_reads |= 1u << index;
        }
    }
    return snapshot_reads;
}

// Ver la declaración. Antes de que la CPU escriba: los overlays que toca, con lo de la GPU y al día.
std::vector<uint64_t> Executor::overlays_before_cpu_write(uint64_t address, uint64_t size) {
    std::vector<uint64_t> kept;
    if (!keep_gpu_overlays_ || !patch_overlays_ || !size || !overlay_maybe_in(address, size)) return kept;
    for (auto it = first_overlay_touching(address); it != overlays_.end() && it->first < address + size; ++it)
        if (it->second.writers && it->second.buffer && overlay_unchanged(it->first, it->second)) kept.push_back(it->first);
    return kept;
}

// Ver la declaración. Lo que la CPU escribió, también en los que siguen valiendo (patch_overlay, la intersección); los que
// ya había reescrito la CPU (o sin nada de la GPU), fuera.
void Executor::cpu_wrote_partial(const std::vector<uint64_t>& kept, uint64_t address, uint64_t size) {
    if (!size || !overlay_maybe_in(address, size)) return;
    for (auto it = first_overlay_touching(address); it != overlays_.end() && it->first < address + size;) {
        const uint64_t start = it->first, end = start + it->second.size;
        if (std::find(kept.begin(), kept.end(), start) == kept.end()) {
            erase_reason(rewrote_why(it->second, "la CPU reescribió su rango (su huella cambió) antes de un comando que escribe "
                                                 "en una parte"),
                         address, size);  // DEPURACIÓN
            it = erase_overlay(it);
            continue;
        }
        const uint64_t from = std::max(address, start), to = std::min(address + size, end);
        patch_overlay(it, from, to - from);  // puede tirarlo (sin staging): se sigue desde su final
        prof_partial_patched_++;
        it = overlays_.lower_bound(end);
    }
}

bool Executor::trace_on_print(uint64_t print) {
    if (!trace_prints_loaded_) {  // una vez: la lista se lee al primer dispatch con el fichero puesto
        trace_prints_loaded_ = true;
        const char* data = getenv("NIER_DATA");
        if (FILE* f = fopen((std::string(data ? data : ".") + "/trace_print").c_str(), "r")) {
            char line[64];
            while (fgets(line, sizeof line, f)) {
                if (strstr(line, "completa")) trace_prints_full_ = true;
                else if (const uint64_t x = strtoull(line, nullptr, 16)) trace_prints_.push_back(x);
            }
            fclose(f);
        }
        NVN_LOG("trace_print: %zu huellas%s\n", trace_prints_.size(), trace_prints_full_ ? ", traza completa" : "");
    }
    for (size_t k = 0; k < trace_prints_.size(); k++)
        if (trace_prints_[k] == print) {
            trace_prints_.erase(trace_prints_.begin() + static_cast<std::ptrdiff_t>(k));  // una vez por sesión
            return true;
        }
    return false;
}

bool Executor::read_after_size(uint64_t size) {
    if (!read_after_loaded_) {  // una vez: la lista se lee al primer dispatch con el fichero puesto
        read_after_loaded_ = true;
        const char* data = getenv("NIER_DATA");
        if (FILE* f = fopen((std::string(data ? data : ".") + "/read_after_dispatch").c_str(), "r")) {
            char line[64];
            while (fgets(line, sizeof line, f))
                if (const uint64_t x = strtoull(line, nullptr, 10)) read_after_sizes_.push_back(x);
            fclose(f);
        }
        NVN_LOG("read_after_dispatch: %zu tamaños\n", read_after_sizes_.size());
    }
    for (uint64_t s : read_after_sizes_)
        if (s == size) return true;
    return false;
}

// DEPURACIÓN (read_after_dispatch): el compute que se acaba de grabar escribió los overlays de `ranges`. Se manda lo
// grabado, se espera a la GPU y se registra lo que dejó en ellos y lo que leyó (inputs, tal como lo vio la GPU: de un
// overlay, lo que tiene ahora; si no, la memoria del juego, que es lo que se copió al grabar). Luego se sigue en otro
// envío, como maybe_split. Cada una para la GPU: con tope.
void Executor::read_after_dispatch(const ProgramStage& cs, uint32_t x, uint32_t y, uint32_t z,
                                   const std::vector<std::pair<uint64_t, uint64_t>>& ranges,
                                   const std::vector<ReadInput>& inputs) {
    if (read_after_logged_ >= 512) return;
    read_after_logged_++;
    VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};  // lo que escribió, visible para la CPU
    hb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    note_gpu_work();
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0, nullptr, 0,
                         nullptr);
    splitting_ = true;  // como maybe_split: partir el envío a mitad no compacta los overlays
    finish(true);
    splitting_ = false;
    // Lo de la GPU en [address, address + size), sin tocar el mapa (find_overlay puede tirar el overlay).
    auto gpu = [&](uint64_t address, uint64_t size) -> const uint8_t* {
        auto it = overlays_.upper_bound(address);
        if (it == overlays_.begin()) return nullptr;
        --it;
        if (!it->second.data || address + size > it->first + it->second.size) return nullptr;
        return it->second.data + (address - it->first);
    };
    std::string in;
    char b[160];
    for (const ReadInput& r : inputs) {
        snprintf(b, sizeof b, " %s %#llx+%llu (%s):", r.label.c_str(), static_cast<unsigned long long>(r.address),
                 static_cast<unsigned long long>(r.size), r.from);
        in += b;
        const uint64_t words = std::min<uint64_t>(r.size / 4, r.words);
        const uint8_t* p = r.snapshot ? r.snapshot : !r.address ? nullptr
                           : strcmp(r.from, "overlay") == 0 ? gpu(r.address, words * 4)
                                                            : reinterpret_cast<const uint8_t*>(r.address);
        for (uint64_t i = 0; p && i < words; i++) {
            uint32_t w;
            memcpy(&w, p + 4 * i, 4);
            snprintf(b, sizeof b, " %x", w), in += b;
        }
    }
    // La huella 0 (sin c1) no distingue los compute: la del código entero (hasta control[0x700], donde acaba).
    uint32_t code_size = 0x100;
    if (cs.control) memcpy(&code_size, static_cast<const uint8_t*>(cs.control) + 0x700, 4);
    code_size = std::min<uint32_t>(code_size ? code_size : 0x100, 0x10000);
    uint64_t code_hash = 0xcbf29ce484222325ULL;
    for (uint32_t k = 0; cs.code && k < code_size; k++)
        code_hash = (code_hash ^ reinterpret_cast<const uint8_t*>(cs.code)[k]) * 0x100000001b3ULL;
    for (auto [address, size] : ranges) {
        const uint8_t* p = gpu(address, size);
        if (!p) {
            NVN_LOG("[leer] dispatch %llu: el overlay de %#llx+%llu ya no está\n", static_cast<unsigned long long>(dispatches_),
                    static_cast<unsigned long long>(address), static_cast<unsigned long long>(size));
            continue;
        }
        uint64_t bad = 0, zero = 0, first_bad = ~0ull;
        std::string first;
        for (uint64_t i = 0; i + 4 <= size; i += 4) {
            uint32_t w;
            memcpy(&w, p + i, 4);
            const bool nan = (w & 0x7f800000u) == 0x7f800000u;
            bad += nan, zero += w == 0;
            if (nan && first_bad == ~0ull) first_bad = i;
            if (i < 32) snprintf(b, sizeof b, " %08x", w), first += b;
        }
        NVN_LOG("[leer] dispatch %llu (compute %016llx, código %016llx de %u bytes, %u %u %u, frame %llu): el overlay "
                "%#llx+%llu queda con %llu de %llu palabras NaN/Inf (la primera en +%lld) y %llu a cero:%s | entradas:%s\n",
                static_cast<unsigned long long>(dispatches_), static_cast<unsigned long long>(cs.print),
                static_cast<unsigned long long>(code_hash), code_size, x, y, z, static_cast<unsigned long long>(frame_),
                static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
                static_cast<unsigned long long>(bad), static_cast<unsigned long long>(size / 4),
                first_bad == ~0ull ? -1ll : static_cast<long long>(first_bad), static_cast<unsigned long long>(zero),
                first.c_str(), in.c_str());
    }
    // El estado del overlay tras cada dispatch (las primeras 24 veces): el de antes del culpable es la salida del anterior.
    // En la memoria interna, como los demás volcados (crear carpetas en la SD falla).
    const std::string dir = "/storage/emulated/0/NieR-volcados/cara";
    mkdir("/storage/emulated/0/NieR-volcados", 0777);
    if (read_after_logged_ <= 24) {
        mkdir(dir.c_str(), 0777);
        for (auto [address, size] : ranges)
            if (const uint8_t* p = gpu(address, size)) {
                char fname[96];
                snprintf(fname, sizeof fname, "/salida_%llu_%llx.bin", static_cast<unsigned long long>(dispatches_),
                         static_cast<unsigned long long>(address));
                if (FILE* f = fopen((dir + fname).c_str(), "wb")) fwrite(p, 1, size, f), fclose(f);
            }
    }
    // El primer dispatch que deja NaN: cada entrada tal como la ve la GPU (copiada de su buffer: los espejos están en
    // memoria de la GPU) y la memoria del juego en la misma dirección, para comparar y para ejecutar el shader fuera.
    bool any_nan = false;
    for (auto [address, size] : ranges)
        if (const uint8_t* p = gpu(address, size))
            for (uint64_t i = 0; i + 4 <= size && !any_nan; i += 4) {
                uint32_t w;
                memcpy(&w, p + i, 4);
                any_nan = (w & 0x7f800000u) == 0x7f800000u;
            }
    if (any_nan && !read_after_dumped_) {
        read_after_dumped_ = true;
        mkdir(dir.c_str(), 0777);
        std::string index;
        for (const ReadInput& r : inputs) {
            std::string label = r.label;
            for (char& ch : label)
                if (ch == ' ' || ch == '(' || ch == ')') ch = '_';
            const uint64_t size = std::min<uint64_t>(r.bound.range ? r.bound.range : r.size, 32ull << 20);
            if (r.bound.buffer) dump_gpu_buffer(r.bound, size, dir + "/entrada_" + label + ".gpu.bin");
            if (r.address && r.size)
                if (FILE* f = fopen((dir + "/entrada_" + label + ".juego.bin").c_str(), "wb"))
                    fwrite(reinterpret_cast<const void*>(r.address), 1, std::min<uint64_t>(r.size, 32ull << 20), f), fclose(f);
            char line[192];
            snprintf(line, sizeof line, "%s %#llx %llu %s gpu_offset %llu gpu_range %llu\n", label.c_str(),
                     static_cast<unsigned long long>(r.address), static_cast<unsigned long long>(r.size), r.from,
                     static_cast<unsigned long long>(r.bound.offset), static_cast<unsigned long long>(r.bound.range));
            index += line;
        }
        if (FILE* f = fopen((dir + "/entradas.txt").c_str(), "w")) {
            fprintf(f, "dispatch %llu compute %016llx grupos %u %u %u\n%s", static_cast<unsigned long long>(dispatches_),
                    static_cast<unsigned long long>(cs.print), x, y, z, index.c_str());
            fclose(f);
        }
        NVN_LOG("[leer] entradas del dispatch %llu volcadas en %s\n", static_cast<unsigned long long>(dispatches_), dir.c_str());
        // lo que acaba de volcar dump_gpu_buffer se envió en otros envíos; se sigue en uno nuevo
    }
    char name[48];
    snprintf(name, sizeof name, "leer_cs%llu", static_cast<unsigned long long>(dispatches_));
    dump_stage_code(cs, name);
    begin();
}

// DEPURACIÓN (read_after_dispatch): copia [b.offset, b.offset + size) del buffer de la GPU al staging, espera y lo escribe
// en `file`. Por trozos que caben en el staging.
void Executor::dump_gpu_buffer(const VkDescriptorBufferInfo& b, uint64_t size, const std::string& file) {
    FILE* f = fopen(file.c_str(), "wb");
    if (!f) return;
    for (uint64_t done = 0; done < size;) {
        const uint64_t len = std::min<uint64_t>(size - done, kStagingChunk - 256);
        begin();
        Staging::Alloc a = staging_->alloc(len, 16);
        if (!a.data) {
            finish(true);
            break;
        }
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        note_gpu_work();
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        const VkBufferCopy region{b.offset + done, a.offset, len};
        note_gpu_work();
        vkCmdCopyBuffer(cmd_, b.buffer, a.buffer, 1, &region);
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        note_gpu_work();
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
        splitting_ = true;
        finish(true);
        splitting_ = false;
        fwrite(a.data, 1, len, f);
        done += len;
    }
    fclose(f);
}

const std::pair<const uint64_t, Executor::Overlay>* Executor::partial_overlay(uint64_t address, uint64_t size) {
    if (overlays_.empty() || !address || !size) return nullptr;
    bool any = false;
    for (uint64_t g = address >> 16, end = (address + size - 1) >> 16; g <= end && !any; g++)
        any = overlay_filter_[overlay_slot(g)] != 0;
    if (!any) return nullptr;
    auto it = overlays_.upper_bound(address);
    if (it != overlays_.begin() && std::prev(it)->first + std::prev(it)->second.size > address) --it;
    if (it == overlays_.end() || it->first >= address + size) return nullptr;
    return &*it;
}

// DEPURACIÓN (watch_nan): una lectura de [address, address + size) desde la memoria del juego o un espejo cuando la
// GPU escribió parte de ese rango en un overlay (lo de la GPU no se ve). Una vez por dirección, como mucho 64.
void Executor::note_partial_read(const char* who, uint64_t address, uint64_t size) {
    const auto* part = partial_overlay(address, size);
    if (!part || partial_logged_.size() >= 64 || !partial_logged_.insert(address).second) return;
    auto pv = provenance_.find(part->first);
    NVN_LOG("[vigía] %s lee %#llx+%llu de la memoria del juego (o de un espejo), pero dentro hay un overlay %#llx+%llu que "
            "no lo cubre entero: si lee esa parte, la lee vieja (sin lo de la GPU); %s, frame %llu; el overlay: %s\n",
            who, static_cast<unsigned long long>(address), static_cast<unsigned long long>(size),
            static_cast<unsigned long long>(part->first), static_cast<unsigned long long>(part->second.size),
            writer_.c_str(), static_cast<unsigned long long>(frame_),
            pv == provenance_.end() ? "de antes de vigilar" : pv->second.created.c_str());
}

// DEPURACIÓN (watch_nan): la primera vez que un draw lee posiciones float32 de un overlay que la GPU ya terminó de
// escribir: ¿NaN/Inf? Con quién lo creó y lo escribió, y la memoria del juego del mismo rango (si es igual, la GPU no
// escribió ahí: el overlay es la copia que hizo make_overlay al crearlo).
void Executor::check_overlay_vertices(uint64_t address, uint64_t size, uint32_t stride, uint32_t format, uint32_t stream) {
    const uint32_t comps = format == 0x0a ? 1 : format == 0x16 ? 2 : format == 0x22 ? 3 : format == 0x2e ? 4 : 0;
    if (!comps || !stride) return;  // solo posiciones float32 (R32F, RG32F, RGB32F, RGBA32F)
    auto it = overlays_.upper_bound(address);
    if (it == overlays_.begin()) return;
    --it;
    const Overlay& o = it->second;
    if (!o.data || address + size > it->first + o.size) return;
    Provenance& p = provenance_[it->first];
    const uint64_t done = completed_.load(std::memory_order_acquire);
    if (p.checked || o.written > done || p.submit > done) return;  // la GPU aún no acabó de escribirlo: luego
    p.checked = true;
    const uint8_t* gpu = o.data + (address - it->first);
    const auto* cpu = reinterpret_cast<const uint8_t*>(address);
    uint64_t words = 0, bad = 0, same = 0;
    std::string g, c;
    char b[16];
    for (uint64_t v = 0; v < 16 && uint64_t{v} * stride + comps * 4 <= size; v++)
        for (uint32_t k = 0; k < comps; k++) {
            uint32_t gw, cw;
            memcpy(&gw, gpu + v * stride + 4 * k, 4);
            memcpy(&cw, cpu + v * stride + 4 * k, 4);
            words++;
            bad += (gw & 0x7f800000u) == 0x7f800000u;
            same += gw == cw;
            if (words <= 8) snprintf(b, sizeof b, " %08x", gw), g += b, snprintf(b, sizeof b, " %08x", cw), c += b;
        }
    static int logged = 0;
    if (!bad || logged++ >= 32) return;
    const ProgramStage* vsp = st_.stages[static_cast<int>(Stage::Vertex)];
    const ProgramStage* fsp = st_.stages[static_cast<int>(Stage::Fragment)];
    NVN_LOG("[vigía] NaN/Inf en vértices: draw (vs %016llx ps %016llx, frame %llu) lee el stream %u (zancada %u, formato "
            "0x%x) %#llx del overlay %#llx+%llu: %llu de %llu componentes NaN/Inf, %llu iguales a la memoria del juego | "
            "GPU:%s | juego:%s | creado en el frame %llu por %s | %llu escrituras de la GPU, la última en el frame %llu por "
            "%s | parches de la CPU %llu (el último en el frame %llu)\n",
            static_cast<unsigned long long>(vsp ? vsp->print : 0), static_cast<unsigned long long>(fsp ? fsp->print : 0),
            static_cast<unsigned long long>(frame_), stream, stride, format, static_cast<unsigned long long>(address),
            static_cast<unsigned long long>(it->first), static_cast<unsigned long long>(o.size),
            static_cast<unsigned long long>(bad), static_cast<unsigned long long>(words), static_cast<unsigned long long>(same),
            g.c_str(), c.c_str(), static_cast<unsigned long long>(p.created_frame),
            p.created.empty() ? "? (de antes de vigilar)" : p.created.c_str(), static_cast<unsigned long long>(p.gpu_writes),
            static_cast<unsigned long long>(p.last_frame), p.last.empty() ? "-" : p.last.c_str(),
            static_cast<unsigned long long>(p.cpu_patches), static_cast<unsigned long long>(p.patch_frame));
}

// DEPURACIÓN (watch_nan): se tira un overlay con lo escrito por la GPU (un compute, un draw o una copia en la GPU):
// por qué (erase_reason) y qué lo pidió, sus primeras palabras y su procedencia. Si luego se lee, sale de la memoria
// del juego: así se ve si la cara del cazador se calcula bien al cargar y se pierde después. Con tope.
void Executor::note_erase(uint64_t start, const Overlay& o) {
    auto pv = provenance_.find(start);
    if (!o.writers && (pv == provenance_.end() || !pv->second.gpu_writes)) return;  // solo la copia de la memoria del juego
    static int logged = 0;
    if (logged++ >= 200 && logged % 1000) return;
    std::string gpu;
    char b[64];
    for (uint64_t i = 0; o.data && i < std::min<uint64_t>(o.size / 4, 4); i++)
        snprintf(b, sizeof b, " %08x", reinterpret_cast<const uint32_t*>(o.data)[i]), gpu += b;
    b[0] = 0;
    if (erase_by_size_)
        snprintf(b, sizeof b, " (%#llx+%llu)", static_cast<unsigned long long>(erase_by_), static_cast<unsigned long long>(erase_by_size_));
    NVN_LOG("[vigía] se tira el overlay %#llx+%llu escrito por la GPU (lo suyo se pierde), frame %llu: %s%s | GPU:%s | "
            "creado en el frame %llu por %s | %llu escrituras de la GPU, la última en el frame %llu por %s | en curso: %s "
            "(%d veces)\n",
            static_cast<unsigned long long>(start), static_cast<unsigned long long>(o.size), static_cast<unsigned long long>(frame_),
            erase_why_, b, gpu.c_str(), static_cast<unsigned long long>(pv == provenance_.end() ? 0 : pv->second.created_frame),
            pv == provenance_.end() || pv->second.created.empty() ? "? (de antes de vigilar)" : pv->second.created.c_str(),
            static_cast<unsigned long long>(pv == provenance_.end() ? 0 : pv->second.gpu_writes),
            static_cast<unsigned long long>(pv == provenance_.end() ? 0 : pv->second.last_frame),
            pv == provenance_.end() || pv->second.last.empty() ? "-" : pv->second.last.c_str(), writer_.c_str(), logged);
}

// La referencia de la devolución de un overlay (con wb_mutex_): lo que tiene ahora la CPU, para saber qué escribe la
// GPU, y la vigilancia de sus páginas desde aquí.
void Executor::wb_reference(uint64_t address, uint64_t size) {
    WriteBackState& w = wb_state_[address];
    w.generation = ++wb_generation_;
    w.base.assign(reinterpret_cast<const uint8_t*>(address), reinterpret_cast<const uint8_t*>(address) + size);
    w.owned.assign((size + kOwnChunk - 1) / kOwnChunk, 0);
    w.watched = !NVN_FLAG("return_by_value") && nnsdk::watch::watchable(address, size);
    w.keys.clear();
    w.orphan = false;
    if (w.watched) nnsdk::watch::collect(address, size, w.keys, nullptr);  // desde aquí, lo que escriba la CPU
}

// Antes se tiraba el overlay entero y los lectores volvían a la memoria del juego, que nunca tuvo lo de la GPU: en la
// aldea (traza t6) el juego copia 80 bytes en mitad de los dos buffers de 10 MB de argumentos de multi-draw cada frame;
// se perdían sus argumentos, se rehacían (10 MB copiados cada uno) y la arena se llenaba antes. Ahora el rango se copia
// también al overlay y el overlay sigue mandando. <datos>/no_patch_overlays: como antes.
std::map<uint64_t, Executor::Overlay>::iterator Executor::overlay_containing(uint64_t address, uint64_t size) {
    if (overlays_.empty() || !address || !size || !overlay_filter_[overlay_slot(address >> 16)]) return overlays_.end();
    if (!patch_overlays_) return overlays_.end();
    auto it = overlays_.upper_bound(address);
    if (it == overlays_.begin()) return overlays_.end();
    --it;
    Overlay& o = it->second;
    if (address + size > it->first + o.size || !o.buffer) return overlays_.end();
    if (!overlay_unchanged(it->first, o)) return overlays_.end();  // la CPU ya lo había reescrito: fuera, como antes
    return it;
}

void Executor::patch_overlay(std::map<uint64_t, Overlay>::iterator it, uint64_t address, uint64_t size) {
    Overlay& o = it->second;
    const uint64_t start = it->first;
    split_why_ = "parche de overlay (escritura de la CPU)";
    end_render_pass();  // las copias no van dentro de un pase
    targets_changed_ = true;
    full_barrier();  // lo que la GPU hiciera antes con el overlay, terminado
    for (uint64_t done = 0; done < size;) {
        const uint64_t len = std::min<uint64_t>(size - done, kStagingChunk - 256);
        Staging::Alloc a = staging_->alloc(len, 16);
        if (!a.data) {  // sin staging: como antes, el overlay fuera
            erase_reason("sin staging para copiarle lo que escribe la CPU", address, size);  // DEPURACIÓN
            erase_overlay(it);
            return;
        }
        memcpy(a.data, reinterpret_cast<const void*>(address + done), len);
        VkBufferCopy region{a.offset, o.offset + (address - start) + done, len};
        note_gpu_work();
        vkCmdCopyBuffer(cmd_, a.buffer, o.buffer, 1, &region);
        done += len;
    }
    full_barrier();
    {  // la devolución no debe tomar estos bytes por de la GPU: pasan a su referencia; la huella de ahora es la suya
        std::lock_guard lock(wb_mutex_);
        if (auto w = wb_state_.find(start); w != wb_state_.end() && w->second.base.size() == o.size)
            memcpy(w->second.base.data() + (address - start), reinterpret_cast<const void*>(address), size);
        delivered_hash_.erase(start);
        o.cpu_hash = range_hash(reinterpret_cast<const uint8_t*>(start), o.size);
    }
    o.checked = submit_index_;
    if (vigilar_nan_) {  // DEPURACIÓN
        Provenance& p = provenance_[start];
        p.cpu_patches++;
        p.patch_frame = frame_;
    }
    static int logged = 0;  // DEPURACIÓN
    if (logged++ < 8 || logged % 1000 == 0)
        NVN_LOG("overlay %#llx+%llu: la CPU escribe %llu bytes en %#llx; se copian también al overlay (%d veces)\n",
                static_cast<unsigned long long>(start), static_cast<unsigned long long>(o.size),
                static_cast<unsigned long long>(size), static_cast<unsigned long long>(address), logged);
}

// Compute: mismo esquema que un draw, con un c0 del driver sintético. Lo que escribe en storage buffers
// es memoria del juego (vértices, argumentos de draws indirectos…): se envía en el acto y se copia de
// vuelta antes de seguir, para que lo que venga después lo lea ya escrito.
void Executor::dispatch(const cmd::Dispatch& v) {
    ProgramStage* cs = st_.compute;
    uint32_t x = v.x, y = v.y, z = v.z;
    dispatches_++;
    maybe_split();
    // Indirecto: si los argumentos los escribió la GPU (un compute que cuenta el trabajo del siguiente), están en su
    // overlay y se usan desde ahí (vkCmdDispatchIndirect). La memoria del juego tiene lo que dejó la CPU o, con la
    // devolución, lo de un frame anterior: con eso el compute corría con el tamaño de otro frame.
    VkBuffer gpu_groups = VK_NULL_HANDLE;
    VkDeviceSize gpu_groups_offset = 0;
    if (v.indirect) {
        if (const Overlay* o = find_overlay(v.indirect, 12, &gpu_groups_offset)) gpu_groups = o->buffer;
        const auto* a = reinterpret_cast<const uint32_t*>(v.indirect);
        if (!gpu_groups) x = a[0], y = a[1], z = a[2];
        static int logged_source = 0;
        if (logged_source++ < 12)  // DEPURACIÓN
            NVN_LOG("dispatch indirecto %#llx: argumentos %s (memoria %u %u %u)\n", static_cast<unsigned long long>(v.indirect),
                    gpu_groups ? "de la GPU" : "de la CPU", a[0], a[1], a[2]);
        if (!gpu_groups && (uint64_t{x} * y * z > (1ull << 22) || x > 65535 || y > 65535 || z > 65535)) {
            static int logged = 0;  // imposible: colgaría la GPU
            if (logged++ < 16)
                NVN_LOG("dispatch indirecto imposible en %#llx: %u %u %u, devuelto %#llx\n",
                        static_cast<unsigned long long>(v.indirect), x, y, z,
                        static_cast<unsigned long long>(delivered_range(v.indirect)));
            return;
        }
        gpu_dispatches_ += gpu_groups != VK_NULL_HANDLE;
    }
    // DEPURACIÓN: la huella se gasta aunque ya se esté trazando este frame (si no, dispararía otra traza en otra carga).
    if (trace_on_print_ && cs && trace_on_print(cs->print) && frame_ != trace_frame_) {
        trace_frame_ = frame_;
        full_trace_ = trace_prints_full_;
        traced_draws_ = 0;
        memset(trace_counts_, 0, sizeof trace_counts_);
        NVN_LOG("traza del frame %llu desde el dispatch %llu (compute %016llx, de trace_print)\n",
                static_cast<unsigned long long>(frame_), static_cast<unsigned long long>(dispatches_),
                static_cast<unsigned long long>(cs->print));
    }
    if (frame_ == trace_frame_) {  // DEPURACIÓN
        std::string ssbos;
        char b[96];
        for (int k = 0; k < kMaxStorage; k++)
            if (st_.bindings[static_cast<int>(Stage::Compute)].storage[k].size)
                snprintf(b, sizeof b, " s%d %#llx+%llu", k,
                         (unsigned long long)st_.bindings[static_cast<int>(Stage::Compute)].storage[k].address,
                         (unsigned long long)st_.bindings[static_cast<int>(Stage::Compute)].storage[k].size), ssbos += b;
        ssbos += " |";  // DEPURACIÓN: sus uniform buffers (los 5..13 van por UBE)
        for (int k = 0; k < kMaxUniforms; k++)
            if (st_.bindings[static_cast<int>(Stage::Compute)].uniforms[k].size)
                snprintf(b, sizeof b, " u%d %#llx+%llu", k,
                         (unsigned long long)st_.bindings[static_cast<int>(Stage::Compute)].uniforms[k].address,
                         (unsigned long long)st_.bindings[static_cast<int>(Stage::Compute)].uniforms[k].size), ssbos += b;
        NVN_LOG("F dispatch %u %u %u%s |%s\n", x, y, z, cs && cs->module ? "" : " (sin shader)", ssbos.c_str());
        if (cs) dump_stage_code(*cs, "traza_cs" + std::to_string(dispatches_));
    }
    if (vigilar_nan_) {  // DEPURACIÓN: quién escribe lo que se cree o escriba en este dispatch (ver Provenance)
        const StageBindings& vb = st_.bindings[static_cast<int>(Stage::Compute)];
        char b[192];
        snprintf(b, sizeof b, "compute %016llx (%u %u %u%s), dispatch %llu, frame %llu;",
                 static_cast<unsigned long long>(cs ? cs->print : 0), x, y, z, gpu_groups ? " de la GPU" : "",
                 static_cast<unsigned long long>(dispatches_), static_cast<unsigned long long>(frame_));
        writer_ = b;
        // Un uniform buffer del juego (índice del banco 3 + i) con sus primeras 8 palabras.
        auto add_uniform = [&](uint32_t index, const char* via) {
            const BufferRange& u = vb.uniforms[index];
            snprintf(b, sizeof b, " c%u%s %#llx+%llu:", kUniformBankBase + index, via, static_cast<unsigned long long>(u.address),
                     static_cast<unsigned long long>(u.size));
            writer_ += b;
            for (uint64_t i = 0; u.address && i < std::min<uint64_t>(u.size / 4, 8); i++)
                snprintf(b, sizeof b, " %x", reinterpret_cast<const uint32_t*>(u.address)[i]), writer_ += b;
        };
        if (cs)
            for (const Resource& r : cs->shader.resources) {
                if (r.kind == Resource::UniformBuffer && r.cbuf_index >= kUniformBankBase &&
                    r.cbuf_index - kUniformBankBase < static_cast<uint32_t>(kMaxUniforms)) {
                    add_uniform(r.cbuf_index - kUniformBankBase, "");
                } else if (r.kind == Resource::StorageBuffer && r.cbuf_index == kComputeDriverBank &&
                           r.cbuf_offset >= kComputeUbeBase && r.cbuf_offset < kComputeUbeBase + 16 * kComputeUbeCount) {
                    // Los c5..c13 de compute van por UBE (c0[0x260…]): son uniform buffers aunque se lean como storage.
                    const uint32_t index = kComputeUbeFirst + (r.cbuf_offset - kComputeUbeBase) / 16;
                    if (index < static_cast<uint32_t>(kMaxUniforms)) add_uniform(index, " (UBE)");
                } else if (r.kind == Resource::StorageBuffer && r.cbuf_index == kComputeDriverBank &&
                           r.cbuf_offset >= kComputeStorageBase && (r.cbuf_offset - kComputeStorageBase) / 16 < kMaxStorage) {
                    const uint32_t k = (r.cbuf_offset - kComputeStorageBase) / 16;
                    snprintf(b, sizeof b, " s%u %#llx+%llu%s", k, static_cast<unsigned long long>(vb.storage[k].address),
                             static_cast<unsigned long long>(vb.storage[k].size), r.is_written ? " (escrito)" : "");
                    writer_ += b;
                }
            }
    }
    if (!cs || !cs->module || (!gpu_groups && (!x || !y || !z))) {
        static int logged = 0;  // DEPURACIÓN
        if (vigilar_nan_ && logged++ < 64)
            NVN_LOG("[vigía] dispatch descartado (%s): %s\n", !cs ? "sin programa" : !cs->module ? "sin shader" : "0 grupos",
                    writer_.c_str());
        return;
    }
    if (vigilar_nan_ && dumped_writers_.insert(cs->print).second) {  // DEPURACIÓN: su código, una vez por huella
        char name[48];
        snprintf(name, sizeof name, "vigia_cs_%016llx", static_cast<unsigned long long>(cs->print));
        dump_stage_code(*cs, name);
    }
    // Las uniones de overlays, antes de los descriptores (ver prepare_dispatch_overlays). Tras descartar los dispatch que
    // no se ejecutan (sin overlays para ellos) y con writer_ ya puesto (watch_nan). Los argumentos indirectos ya se
    // miraron: si una unión se traga su overlay, su hueco sigue intacto hasta que acabe este envío, y se leen antes.
    const uint32_t snapshot_reads = prepare_dispatch_overlays();
    auto skipped = [&](const char* why) {  // DEPURACIÓN
        static int logged = 0;
        if (vigilar_nan_ && logged++ < 64) NVN_LOG("[vigía] dispatch descartado (%s): %s\n", why, writer_.c_str());
    };
    end_render_pass();
    targets_changed_ = true;
    const int S = static_cast<int>(Stage::Compute);
    const StageBindings& sb = st_.bindings[S];
    // DEPURACIÓN (registro): un compute que escribe un SSBO de ≤ 4 KB (en MH Rise, el histograma y la exposición): sus
    // uniform buffers del juego al log, una vez cada 30 frames por shader (los parámetros de la medida, c3/c4).
    if (log_state_) {
        bool state_writer = false;
        for (const Resource& r : cs->shader.resources)
            if (r.kind == Resource::StorageBuffer && r.is_written && r.cbuf_index == kComputeDriverBank &&
                r.cbuf_offset >= kComputeStorageBase) {
                const uint32_t k = (r.cbuf_offset - kComputeStorageBase) / 16;
                if (k < kMaxStorage && sb.storage[k].size && sb.storage[k].size <= 4096) state_writer = true;
            }
        uint64_t& last = state_ubo_logged_[cs];
        if (state_writer && (!last || frame_ >= last + 30)) {
            last = frame_;
            for (int k = 0; k < kMaxUniforms; k++) {
                const BufferRange& u = sb.uniforms[k];
                if (!u.address || !u.size) continue;
                std::string vals;
                char b[24];
                const auto* f = reinterpret_cast<const float*>(u.address);
                for (uint64_t i = 0; i < std::min<uint64_t>(u.size / 4, 16); i++) snprintf(b, sizeof b, " %.6g", f[i]), vals += b;
                NVN_LOG("[estado] frame %llu: compute %p shader %016llx (%u %u %u) c%d (%llu bytes):%s\n",
                        (unsigned long long)frame_, static_cast<const void*>(cs),
                        static_cast<unsigned long long>(cs->print), x, y, z, k + static_cast<int>(kUniformBankBase),
                        (unsigned long long)u.size, vals.c_str());
            }
        }
    }

    std::vector<BoundImage> textures;
    AccessScope dispatch_access{VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0};
    for (const Resource& r : cs->shader.resources) {
        dispatch_access.access |= r.kind == Resource::UniformBuffer ? VK_ACCESS_UNIFORM_READ_BIT : VK_ACCESS_SHADER_READ_BIT;
        if (shader_writes_storage(r)) dispatch_access.access |= VK_ACCESS_SHADER_WRITE_BIT;
        if (r.kind != Resource::Texture && r.kind != Resource::Image) continue;
        if (r.set) continue;  // arrays bindless: bindless_set
        BoundImage b{&r, Stage::Compute};
        if (!prepare_shader_image(*cs, b)) return skipped("subida de textura");
        textures.push_back(b);
    }

    const ProgramStage* stages[kStages] = {};
    stages[S] = cs;
    // Arrays bindless (set 1): las sondas de reflexión que mezcla el compute 140589 (c6[0xc83c], sampler 263).
    Layout* early_layout = cs->shader.has_bindless_set && fast_bindless_ ? &layout(stages) : nullptr;  // como en draw()
    const VkDescriptorSet bindless = cs->shader.has_bindless_set ? [&] {
        LoadTimer timer(*this, kLdBindless);
        return bindless_set(stages, early_layout ? early_layout->bindless : VK_NULL_HANDLE);
    }() : VK_NULL_HANDLE;
    if (cs->shader.has_bindless_set && !bindless) return skipped("sin set bindless");
    Layout& l = early_layout ? *early_layout : layout(stages);
    VkPipeline p = compute_pipeline(*cs, l);
    if (!p) return skipped("sin pipeline");
    // c0 del driver: handles de textura y tabla de storage buffers (dirección real y tamaño; el shader
    // calcula cada acceso como dirección − base y el buffer enlazado empieza en esa dirección).
    uint8_t driver[kComputeDriverSize] = {};
    for (int k = 0; k < kMaxTextures && kTextureHandleBase + 8 * k + 8 <= kComputeDriverSize; k++)
        memcpy(driver + kTextureHandleBase + 8 * k, &sb.textures[k], 8);
    const bool ube = !NVN_FLAG("no_ube"), compute_c2 = !NVN_FLAG("no_c2_compute");
    // Handles de imagen (c2[0x120 + 8·i]: los compute los cargan para sus SUST/SULD; la imagen se resuelve por orden en
    // resolve_image) y los UBE (dirección y tamaño, como los storage buffers de 0x310).
    for (int k = 0; compute_c2 && k < kMaxImages && kComputeImageHandleBase + 8 * k + 8 <= kTextureHandleBase; k++)
        memcpy(driver + kComputeImageHandleBase + 8 * k, &sb.images[k], 8);
    for (uint32_t j = 0; ube && j < kComputeUbeCount && kComputeUbeFirst + j < static_cast<uint32_t>(kMaxUniforms); j++) {
        const BufferRange& u = sb.uniforms[kComputeUbeFirst + j];
        if (!u.address) continue;  // sin enlazar: queda lo que hubiera (0x260 es también la textura 31)
        const uint32_t size = static_cast<uint32_t>(u.size);
        memcpy(driver + kComputeUbeBase + 16 * j, &u.address, 8);
        memcpy(driver + kComputeUbeBase + 16 * j + 8, &size, 4);
    }
    for (int k = 0; k < kMaxStorage && kComputeStorageBase + 16 * k + 16 <= kComputeDriverSize; k++) {
        uint32_t size = static_cast<uint32_t>(sb.storage[k].size);
        memcpy(driver + kComputeStorageBase + 16 * k, &sb.storage[k].address, 8);
        memcpy(driver + kComputeStorageBase + 16 * k + 8, &size, 4);
    }

    struct Writeback {
        uint8_t* staging;
        uint64_t address;
        size_t size;
    };
    std::vector<Writeback> writebacks;
    VkDescriptorSet set = descriptor_set(l.set);
    if (!set) return skipped("sin descriptor set");
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> buffers;
    std::vector<VkDescriptorImageInfo> images;
    buffers.reserve(cs->shader.resources.size());
    images.reserve(cs->shader.resources.size());
    const VkPhysicalDeviceLimits& limits = vk()->props.limits;
    const VkDeviceSize ubo_align = std::max<VkDeviceSize>(limits.minUniformBufferOffsetAlignment, 16);
    const VkDeviceSize ssbo_align = std::max<VkDeviceSize>(limits.minStorageBufferOffsetAlignment, 16);
    std::vector<std::pair<uint64_t, uint64_t>> read_ranges;  // DEPURACIÓN (read_after_dispatch): overlays a leer
    VkDescriptorBufferInfo face_remap{};
    const bool trace_face_remap = full_trace_ && frame_ == trace_frame_ && cs->print == 0x525985a9ccdd7b21ULL;
    const std::string remap_file = trace_face_remap ? "/storage/emulated/0/NieR-volcados/normal-remap-f" +
        std::to_string(frame_) + "-cs" + std::to_string(dispatches_) : std::string{};
    std::vector<ReadInput> read_inputs;
    for (const Resource& r : cs->shader.resources) {
        if (r.kind != Resource::UniformBuffer && r.kind != Resource::StorageBuffer) continue;
        BufferRange range{};
        const void* src = nullptr;
        bool snapshot_read = false;
        if (r.kind == Resource::UniformBuffer) {
            if (r.cbuf_index == kComputeDriverBank || (r.cbuf_index == kTextureHandleBank && compute_c2)) {
                src = driver;
                range.size = sizeof driver;
            } else if (r.cbuf_index == kShaderConstantsBank) {
                range = {cs->constants, cs->constants_size};
            } else {
                int index = static_cast<int>(r.cbuf_index) - static_cast<int>(kUniformBankBase);
                if (index >= 0 && index < kMaxUniforms) range = sb.uniforms[index];
            }
        } else if (r.cbuf_index == kComputeDriverBank && ube && r.cbuf_offset >= kComputeUbeBase &&
                   r.cbuf_offset < kComputeUbeBase + 16 * kComputeUbeCount) {
            const uint32_t index = kComputeUbeFirst + (r.cbuf_offset - kComputeUbeBase) / 16;
            if (index < static_cast<uint32_t>(kMaxUniforms)) range = sb.uniforms[index];
            static int logged = 0;  // DEPURACIÓN: que el juego los enlaza de verdad
            if (logged++ < 16 || frame_ == trace_frame_)
                NVN_LOG("%scompute UBE c0[0x%x] -> uniform buffer %u %#llx+%llu\n", frame_ == trace_frame_ ? "F " : "",
                        r.cbuf_offset, index, static_cast<unsigned long long>(range.address),
                        static_cast<unsigned long long>(range.size));
        } else if (r.cbuf_index == kComputeDriverBank && r.cbuf_offset >= kComputeStorageBase) {
            int index = static_cast<int>((r.cbuf_offset - kComputeStorageBase) / 16);
            if (index < kMaxStorage) {
                range = sb.storage[index];
                snapshot_read = !r.is_written && (snapshot_reads & (1u << index));
            }
            if (range.address && !r.is_written) note_read(5, range.address, range.size);
        }
        if (!src && range.address) src = reinterpret_cast<const void*>(range.address);
        VkDeviceSize size = std::min<VkDeviceSize>(range.size ? range.size : 256,
                                                   r.kind == Resource::UniformBuffer ? 65536 : kStagingChunk);
        // Lo que escribe el compute va a su overlay; lo que lee, del overlay si otro compute lo escribió.
        VkDeviceSize ov_offset = 0;
        const Overlay* ov = nullptr;
        if (src == reinterpret_cast<const void*>(range.address) && range.address && range.size) {
            // Sin uniones aquí (las hizo prepare_dispatch_overlays): una tiraría un overlay cuyo descriptor ya está en buffers.
            const uint64_t read_size = snapshot_read ? std::min<uint64_t>(range.size, limits.maxStorageBufferRange) : size;
            ov = r.kind == Resource::StorageBuffer && r.is_written ? make_overlay(range.address, size, &ov_offset, false)
                                                                    : find_overlay(range.address, read_size, &ov_offset);
            if (ov && r.kind == Resource::StorageBuffer && r.is_written) {
                note_overlay_written(range.address, kWrittenByCompute);
                if (log_state_ && range.size <= 64 && (small_state_.size() < 32 || small_state_.count(range.address))) {
                    SmallState& st = small_state_[range.address];
                    st.size = range.size;
                    if (st.frame != frame_) st.prev_frame = st.frame, st.frame = frame_;
                    st.writes++;
                }
            }
        }
        if (frame_ == trace_frame_ && r.kind == Resource::StorageBuffer) {  // DEPURACIÓN
            std::string vals;
            char b[24];
            if (ov && ov->data)
                for (size_t i = 0; i < std::min<size_t>(size / 4, 64); i++)
                    snprintf(b, sizeof b, " %x", reinterpret_cast<const uint32_t*>(ov->data + (ov_offset - ov->offset))[i]), vals += b;
            NVN_LOG("F compute SSBO %s %#llx+%llu: %s%s\n", r.is_written ? "escrito" : "leído",
                    static_cast<unsigned long long>(range.address), static_cast<unsigned long long>(size),
                    ov ? "en el overlay, antes de ejecutarse este envío:" : "de la memoria", vals.c_str());
        }
        // Lo que solo se lee de la memoria del juego se copia una vez por envío, como en los draws: MH Rise lanza al
        // cargar varios compute por frame sobre los mismos SSBO de megas, y copiarlos enteros en cada uno se llevaba
        // dos tercios de NvnSubmit.
        const bool cacheable = src == reinterpret_cast<const void*>(range.address) && range.size &&
                               !(r.kind == Resource::StorageBuffer && r.is_written);
        VkDeviceSize mirror_offset = 0;
        VkBuffer snapshot_buffer = VK_NULL_HANDLE;
        VkDeviceSize snapshot_offset = 0;
        const uint8_t* snapshot_data = nullptr;
        const VkDeviceSize snapshot_size = std::min<VkDeviceSize>(range.size, limits.maxStorageBufferRange);
        if (!ov && snapshot_read && !gather_overlays(range.address, snapshot_size, &snapshot_buffer, &snapshot_offset, &snapshot_data)) {
            cpu_publication_ok_ = false;
            return skipped("mixed SSBO snapshot failed");
        }
        const Mirror* mir = !ov && !snapshot_buffer && cacheable && r.kind == Resource::StorageBuffer ? mirror(range.address, range.size, &mirror_offset)
                                                                                  : nullptr;
        if (vigilar_nan_ && !ov && !snapshot_buffer && r.kind == Resource::StorageBuffer && range.address && range.size)
            note_partial_read(r.is_written ? "un compute (SSBO escrito)" : "un compute", range.address, size);
        const char* from = "staging";  // DEPURACIÓN (read_after_dispatch)
        if (ov) {
            from = "overlay";
            buffers.push_back({ov->buffer, ov_offset, size});
        } else if (snapshot_buffer) {
            from = "snapshot";
            buffers.push_back({snapshot_buffer, snapshot_offset, snapshot_size});
        } else if (mir) {
            from = "espejo";
            buffers.push_back({mir->buffer, mirror_offset, range.size});
        } else if (const auto* hit = cacheable ? copied_.find(range.address, size) : nullptr) {
            from = "copia del envío";
            buffers.push_back({hit->first, hit->second, size});
        } else {
            Staging::Alloc a = staging_->alloc(size, r.kind == Resource::UniformBuffer ? ubo_align : ssbo_align);
            if (!a.data) return skipped("sin staging: los overlays que ya creó quedan con la memoria del juego");
            if (src && range.size)
                memcpy(a.data, src, size);
            else
                memset(a.data, 0, size);
            prof_copied_bytes_ += size;
            note_copy(range.address, size, r.kind == Resource::StorageBuffer, 5);
            if (cacheable) copied_.insert(range.address, size, a.buffer, a.offset);
            if (r.kind == Resource::StorageBuffer && r.is_written && range.address && range.size)
                writebacks.push_back({a.data, range.address, static_cast<size_t>(size)});
            buffers.push_back({a.buffer, a.offset, size});
        }
        if (trace_face_remap && r.kind == Resource::StorageBuffer && r.cbuf_index == kComputeDriverBank &&
            r.cbuf_offset == kComputeStorageBase + 16 && range.address && range.size) {
            // The shader masks each S1 word to 22 bits and indexes 12-byte entries in S0.
            // Snapshot both views so the first bad input can be assigned to the upload or the guest.
            face_remap = buffers.back();
            mkdir("/storage/emulated/0/NieR-volcados", 0775);
            std::ofstream guest(remap_file + ".guest.bin", std::ios::binary);
            guest.write(reinterpret_cast<const char*>(range.address), face_remap.range);
            guest.close();
            NVN_LOG("normal remap: %s guest=%s source=%s address=%#llx map_bytes=%llu sums_bytes=%llu\n",
                    remap_file.c_str(), guest ? "saved" : "failed", from, (unsigned long long)range.address,
                    (unsigned long long)face_remap.range, (unsigned long long)sb.storage[0].size);
        }
        if (read_after_) {  // DEPURACIÓN (read_after_dispatch): lo que recibe de verdad, para registrarlo tras ejecutarse
            char b[32];
            const bool driver_bank = r.cbuf_index == kComputeDriverBank;
            if (r.kind == Resource::UniformBuffer)
                snprintf(b, sizeof b, "c%u", r.cbuf_index);
            else if (driver_bank && r.cbuf_offset >= kComputeUbeBase && r.cbuf_offset < kComputeUbeBase + 16 * kComputeUbeCount)
                snprintf(b, sizeof b, "c%u (UBE)", kUniformBankBase + kComputeUbeFirst + (r.cbuf_offset - kComputeUbeBase) / 16);
            else if (driver_bank && r.cbuf_offset >= kComputeStorageBase)
                snprintf(b, sizeof b, "s%u%s", (r.cbuf_offset - kComputeStorageBase) / 16, r.is_written ? " (escrito)" : "");
            else
                snprintf(b, sizeof b, "ssbo c%u[0x%x]", r.cbuf_index, r.cbuf_offset);
            read_inputs.push_back({b, src == driver ? 0 : range.address, range.size, from,
                                   r.kind == Resource::UniformBuffer ? 16u : 4u, buffers.back(), snapshot_data});
            if (ov && r.kind == Resource::StorageBuffer && r.is_written && read_after_size(range.size))
                read_ranges.push_back({range.address, range.size});
        }
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set;
        w.dstBinding = r.binding;
        w.descriptorCount = 1;
        w.descriptorType = r.kind == Resource::UniformBuffer ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                                             : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &buffers.back();
        writes.push_back(w);
    }
    for (BoundImage& b : textures) {
        const bool image = b.r->kind == Resource::Image;
        Texture* t = b.tex;
        VkImageView iv = image ? storage_view(b.tex, b.view, b.r->texture_type, b.storage_target)
                               : t && t->image ? view(*t, b.view, ViewUse::Sampled) : VK_NULL_HANDLE;
        if (!image && !iv) iv = dummy_view(b.r->texture_type);
        if (!iv) return skipped("sin vista de imagen");
        images.push_back({image ? VK_NULL_HANDLE : sampler(b.smp), iv, VK_IMAGE_LAYOUT_GENERAL});
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set;
        w.dstBinding = b.r->binding;
        w.descriptorCount = 1;
        w.descriptorType = image ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &images.back();
        writes.push_back(w);
    }
    vkUpdateDescriptorSets(vk()->device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (gpu_groups) {
        dispatch_access.stage |= VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
        dispatch_access.access |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    }
    if (fill_quirk_ || crumbs_on_) {
        dispatch_access.stage |= transfer_write.stage;
        dispatch_access.access |= transfer_write.access;
    }
    sync_memory(cmd_, prior_writes, dispatch_access);
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, l.pipeline, 0, 1, &set, 0, nullptr);
    if (bindless) vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, l.pipeline, nvnshader::kBindlessSet, 1, &bindless, 0, nullptr);
    prof_gpu_mark("compute");
    if (crumbs_on_) {
        char label[64];
        snprintf(label, sizeof label, "dispatch %u %u %u%s", x, y, z, v.indirect ? " (indirecto)" : "");
        crumb_begin('c', label);
        crumb_stage(cs);
    } else if (fill_quirk_) {
        vkCmdFillBuffer(cmd_, crumb_buffer_, 8, 4, ++fill_count_);
    }
    note_gpu_work();
    if (loading_log_) {
        fprintf(loading_log_, "dispatch after_draw=%llu maxwell=%016llx groups=%u,%u,%u indirect=%llx\n",
                (unsigned long long)draws_,(unsigned long long)cs->print,x,y,z,(unsigned long long)v.indirect);
        for (const BoundImage& b : textures)
            if (b.tex) fprintf(loading_log_, "  compute_source binding=%u kind=%u write=%u serial=%llu\n",
                               b.r->binding,unsigned(b.r->kind),unsigned(b.r->is_written),(unsigned long long)b.tex->serial);
    }
    if (gpu_groups)
        vkCmdDispatchIndirect(cmd_, gpu_groups, gpu_groups_offset);
    else
        vkCmdDispatch(cmd_, x, y, z);
    publish_storage_writes(textures, "imagen de compute");
    const VkAccessFlags write_access = dispatch_access.access & (VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
    sync_memory(cmd_, {dispatch_access.stage, write_access}, later_access);
    if (crumbs_on_)
        crumb_end();
    else if (fill_quirk_)
        vkCmdFillBuffer(cmd_, crumb_buffer_, 12, 4, fill_count_);
    if (!writebacks.empty()) {
        VkMemoryBarrier hb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};  // las escrituras, visibles para la CPU
        hb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        note_gpu_work();
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hb, 0,
                             nullptr, 0, nullptr);
        prof_writebacks_++;
        finish(true);  // espera a la GPU; los datos siguen en el staging hasta que se reutilice el contexto
        for (const Writeback& w : writebacks) memcpy(reinterpret_cast<void*>(w.address), w.staging, w.size);
        begin();
    }
    if (!read_ranges.empty()) read_after_dispatch(*cs, x, y, z, read_ranges, read_inputs);  // DEPURACIÓN
    if (face_remap.buffer) {
        dump_gpu_buffer(face_remap, face_remap.range, remap_file + ".gpu.bin");
        begin();
    }
}

PicturePath Executor::blit(Texture& t, const VkRect2D& crop, VkImage dst, VkFormat format, VkExtent2D extent,
                    VkCommandBuffer cmd) {
    std::lock_guard lock(g_mutex);
    if ((!ready_ && !init()) || !t.image || !crop.extent.width || !crop.extent.height) return PicturePath::Clear;
    // FSR 1: más nítido que el blit lineal (no_fsr: el blit). Escala una zona que empieza en 0,0.
    // Guarda: el FSR cuesta ~4,4 ms de GPU por frame (en las ventanas limitadas por la GPU, 1000/fps menos la GPU del
    // frame da 4,4-4,8 ms de mediana en tf1, pk1, st1 y tk2; en cap1 el campamento a 27,2 fps con 32,6 ms de frame).
    // Con el frame del juego por encima de ~28,5 ms ya no caben los 33,3 ms: entonces el blit lineal (~0,5 ms), con
    // histéresis para no cambiar de nitidez a cada frame. <datos>/fsr_always: FSR siempre, como antes.
    bool fsr_ok = true;
    if (!NVN_FLAG("fsr_always")) {
        const int64_t gpu = frame_gpu_ns_.load(std::memory_order_relaxed);
        // (por lo que dura el frame ahora: 30 y 60 con sus medidas; otro límite del menú, dejando sitio a los ~4,8 ms del FSR)
        const int64_t period = frame_period_ns();
        const bool fps60 = period < 20'000'000;
        int64_t off_ns = fps60 ? kFsrOffNs60 : kFsrOffNs, on_ns = fps60 ? kFsrOnNs60 : kFsrOnNs;
        if (period > 17'000'000 && (period < 33'000'000 || period > 34'000'000))
            off_ns = period - 4'833'333, on_ns = off_ns - 3'500'000;
        if (!fsr_guard_off_) {
            fsr_guard_frames_ = gpu > off_ns ? fsr_guard_frames_ + 1 : 0;
            if (fsr_guard_frames_ >= 3) fsr_guard_off_ = true, fsr_guard_frames_ = 0;
        } else {
            fsr_guard_frames_ = gpu < on_ns ? fsr_guard_frames_ + 1 : 0;
            if (fsr_guard_frames_ >= 60) fsr_guard_off_ = false, fsr_guard_frames_ = 0;
        }
        fsr_ok = !fsr_guard_off_;
        prof_fsr_guard_frames_ += fsr_guard_off_;
    } else {
        fsr_guard_off_ = false, fsr_guard_frames_ = 0;
    }
    const VkExtent2D size{static_cast<uint32_t>(t.desc.width), static_cast<uint32_t>(t.desc.height)};
    const VkImageView fsr_view = fsr_ok && crop.offset.x == 0 && crop.offset.y == 0 ? view(t, {}, ViewUse::Sampled) : VK_NULL_HANDLE;
    return compose_picture(cmd, t.image, fsr_view, size, crop, dst, format, extent);
}

// DEPURACIÓN: copia una textura de color a RGBA8 en la CPU y la guarda como PPM en <datos>/volcados.
// DEPURACIÓN: profundidad de nivel 0 a gris (D32F tal cual; D24: los 24 bits bajos; D16: 2 bytes por texel), y la misma
// con toda la precisión en <name>_p16.pgm (P5 de 16 bits, big-endian). Antes el D16 se leía de 4 en 4 bytes (cada píxel
// del volcado era el byte bajo del texel impar y la mitad de abajo, basura del staging) y solo la capa 0: el mapa de
// sombras de MH Rise (512x512x32 D16, las 4 cascadas en las capas) no se podía leer. layer < 0: las primeras 4 capas.
void Executor::dump_depth(Texture& t, const std::string& name, int layer) {
    const uint32_t w = t.desc.width, h = t.desc.height;
    const bool d16 = t.vk_format == VK_FORMAT_D16_UNORM || t.vk_format == VK_FORMAT_D16_UNORM_S8_UINT;
    const bool f32 = t.vk_format == VK_FORMAT_D32_SFLOAT || t.vk_format == VK_FORMAT_D32_SFLOAT_S8_UINT;
    const uint32_t texel = d16 ? 2 : 4;  // lo que deja vkCmdCopyImageToBuffer del aspecto de profundidad
    const uint32_t first = layer >= 0 ? static_cast<uint32_t>(layer) : 0;
    const uint32_t count = layer >= 0 ? 1 : std::min<uint32_t>(std::max(t.layers, 1u), 4);
    if (first >= std::max(t.layers, 1u)) return;
    std::string dir = "/storage/emulated/0/NieR-volcados";
    mkdir(dir.c_str(), 0775);
    for (uint32_t l = first; l < first + count; l++) {
        const std::string file = layer < 0 && t.layers > 1 ? name + "_c" + std::to_string(l) : name;
        begin();
        flush_clears(&t);
        Staging::Alloc a = staging_->alloc(VkDeviceSize{w} * h * texel, 16);
        if (!a.data) return;
        barrier();
        VkBufferImageCopy r{};
        r.bufferOffset = a.offset;
        r.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, l, 1};
        r.imageExtent = {w, h, 1};
        note_gpu_work();
        vkCmdCopyImageToBuffer(cmd_, t.image, VK_IMAGE_LAYOUT_GENERAL, a.buffer, 1, &r);
        uint8_t* src = a.data;
        finish(true);
        std::vector<uint8_t> rgb(size_t{w} * h * 3), p16(size_t{w} * h * 2);
        double sum = 0, mn = 1e9, mx = -1e9;
        size_t zeros = 0;
        for (size_t k = 0; k < size_t{w} * h; k++) {
            float v;
            if (d16) {
                uint16_t u;
                memcpy(&u, src + k * 2, 2);
                v = u / 65535.0f;
            } else if (f32) {
                memcpy(&v, src + k * 4, 4);
            } else {
                uint32_t u;
                memcpy(&u, src + k * 4, 4);
                v = (u & 0xFFFFFF) / 16777215.0f;
            }
            sum += v, mn = std::min<double>(mn, v), mx = std::max<double>(mx, v);
            zeros += v == 0.0f;
            rgb[k * 3] = rgb[k * 3 + 1] = rgb[k * 3 + 2] = static_cast<uint8_t>(std::clamp(v, 0.0f, 1.0f) * 255);
            const uint16_t q = static_cast<uint16_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 65535.0f));
            p16[k * 2] = static_cast<uint8_t>(q >> 8), p16[k * 2 + 1] = static_cast<uint8_t>(q);
        }
        if (FILE* f = fopen((dir + "/" + file + ".ppm").c_str(), "wb")) {
            fprintf(f, "P6 %u %u 255\n", w, h);
            fwrite(rgb.data(), 1, rgb.size(), f);
            fclose(f);
        }
        if (FILE* f = fopen((dir + "/" + file + "_p16.pgm").c_str(), "wb")) {
            fprintf(f, "P5 %u %u 65535\n", w, h);
            fwrite(p16.data(), 1, p16.size(), f);
            fclose(f);
        }
        NVN_LOG("volcado %s: %ux%u capa %u de %u profundidad media %.4f mín %.4f máx %.4f, %.1f%% a cero\n", file.c_str(), w, h,
                l, std::max(t.layers, 1u), sum / (double(w) * h), mn, mx, 100.0 * zeros / (double(w) * h));
    }
}

// DEPURACIÓN: los bits tal cual del mip 0 (para ver NaN/infinitos de un R11G11B10F: el volcado a RGBA8 los pierde).
void Executor::check_loading_binding(const Texture& t, const TextureViewDesc& view, VkImageView image_view, uint32_t stage,
                                     const Resource& r, uint32_t element, const Sampler* smp) {
    // Two ASTC blocks in the first GOB of both Switch versions of Loading_IAM.tex. Check
    // the backing rather than a transient pool ID or the sampling view's reinterpretation.
    constexpr uint8_t at0[]{0x08, 0x01, 0xd9, 0x8a, 0x8a, 0xc1, 0x2d, 0x00, 0x04, 0x40, 0x00, 0x04, 0x40, 0x00, 0x58, 0xd5};
    constexpr uint8_t at48[]{0xde, 0x05, 0xa3, 0xa0, 0x8c, 0x8c, 0x6c, 0x68, 0x00, 0xb0, 0x2d, 0x45, 0x4a, 0x81, 0x52, 0xa4};
    if (t.desc.width != 2048 || t.desc.height != 4096 || (t.desc.format != 0x8b && t.desc.format != 0x7d) || !t.storage ||
        t.layout.total_size() < 64)
        return;
    if (!memcmp(t.storage, at0, sizeof at0) && !memcmp(t.storage + 48, at48, sizeof at48))
        loading_matches_.push_back({stage, r.set, r.binding, element, &t, smp, view, image_view});
}

void Executor::record_loading_draw(const cmd::Draw& d, VkPipeline pipeline, const cmd::SetRenderTargets& rt) {
    if (!loading_log_) {
        const char* data = getenv("NIER_DATA");
        loading_dir_ = std::string(data ? data : ".") + "/loading-f" + std::to_string(frame_) + "-XXXXXX";
        if (!mkdtemp(loading_dir_.data()) || !(loading_log_ = fopen((loading_dir_ + "/draws.txt").c_str(), "w"))) {
            NVN_LOG("cannot create loading capture in %s: %s\n", loading_dir_.c_str(), strerror(errno));
            loading_armed_ = false;
            return;
        }
        fprintf(loading_log_, "frame=%llu first_atlas_draw=%llu\n", (unsigned long long)frame_, (unsigned long long)draws_);
    }
    const auto& vp = st_.viewport;
    const auto& sc = st_.scissor;
    fprintf(loading_log_,
            "draw=%llu atlas_bindings=%zu pipeline=%p key=%016llx submit=%llu pass=%p fb=%p primitive=%d count=%d indirect=%llx "
            "viewport=%.2f,%.2f,%.2f,%.2f scissor=%d,%d,%u,%u enabled=%d area=%d,%d,%u,%u\n",
            (unsigned long long)draws_, loading_matches_.size(), (void*)pipeline, (unsigned long long)loading_pipeline_key_,
            (unsigned long long)submit_index_, (void*)rp_, (void*)cur_fb_, d.primitive, d.count, (unsigned long long)d.indirect,
            vp.x, vp.y, vp.width, vp.height, sc.offset.x, sc.offset.y, sc.extent.width, sc.extent.height, int(st_.scissor_set),
            rp_area_.offset.x, rp_area_.offset.y, rp_area_.extent.width, rp_area_.extent.height);
    for (int k = 0; k < rt.count; ++k) {
        if (const Texture* t = rt.colors[k]) {
            const BlendState& b = st_.blend[k];
            const auto& v = rt.color_views[k];
            fprintf(loading_log_,
                    "  rt%d serial=%llu addr=%p image=%p content=%llu vk=%u pass_vk=%u size=%dx%d level=%d layer=%d "
                    "mask=%x blend=%u rgb=%u,%u,%u alpha=%u,%u,%u\n",
                    k, (unsigned long long)t->serial, (void*)t->storage, (void*)t->image, (unsigned long long)t->content_version,
                    unsigned(t->vk_format), unsigned(rp_formats_[k]), t->desc.width, t->desc.height,
                    v.flags & kViewLevels ? v.base_level : 0, v.flags & kViewLayers ? v.base_layer : 0,
                    ((st_.masks.masks & rp_keep_masks_) >> (4 * k)) & 15, (loading_blend_enable_ >> k) & 1, b.src_rgb, b.dst_rgb,
                    b.eq_rgb, b.src_alpha, b.dst_alpha, b.eq_alpha);
        }
    }
    for (const AtlasBinding& b : loading_matches_) {
        const Texture& t = *b.texture;
        const auto& v = b.view;
        fprintf(loading_log_,
                "  atlas stage=%u set=%u binding=%u element=%u serial=%llu addr=%p nvn=%x vk=%u gpu_written=%d "
                "swizzle=%u,%u,%u,%u image_view=%p view_flags=%x view_format=%x view_swizzle=%u,%u,%u,%u "
                "levels=%d,%d layers=%d,%d",
                b.stage, b.set, b.binding, b.element, (unsigned long long)t.serial, (void*)t.storage, t.desc.format,
                unsigned(t.vk_format), int(t.gpu_levels != 0), t.desc.swizzle[0], t.desc.swizzle[1], t.desc.swizzle[2],
                t.desc.swizzle[3], (void*)b.image_view, v.flags, v.format, v.swizzle[0], v.swizzle[1], v.swizzle[2], v.swizzle[3],
                v.base_level, v.levels, v.base_layer, v.layers);
        if (b.sampler)
            fprintf(loading_log_, " min_mag=%u,%u wrap_uv=%u,%u", b.sampler->desc.min_filter, b.sampler->desc.mag_filter,
                    b.sampler->desc.wrap[0], b.sampler->desc.wrap[1]);
        fputc('\n', loading_log_);
    }
    for (const ProgramStage* stage : st_.stages) {
        if (!stage) continue;
        fprintf(loading_log_, "  stage=%s maxwell=%016llx base_spirv=%llu key=%016llx/%016llx\n",
                nvnshader::stage_name(stage->shader.stage), (unsigned long long)stage->print, (unsigned long long)stage->spirv_id,
                (unsigned long long)stage->cache_key.a, (unsigned long long)stage->cache_key.b);
        if (!loading_stages_.insert(stage->spirv_id).second) continue;
        const std::string path = loading_dir_ + "/base-" + std::to_string(stage->spirv_id) + ".spv";
        FILE* f = fopen(path.c_str(), "wb");
        bool ok = f && !stage->shader.spirv.empty() &&
                  fwrite(stage->shader.spirv.data(), 4, stage->shader.spirv.size(), f) == stage->shader.spirv.size();
        if (f && fclose(f)) ok = false;
        if (!ok) fprintf(loading_log_, "  error saving %s\n", path.c_str());
    }
    if (!loading_matches_.empty()) {
        NVN_LOG("loading capture %s draw=%llu vertex inputs below\n", loading_dir_.c_str(), (unsigned long long)draws_);
        trace_draw(d);
    }
}

void Executor::finish_loading_capture(ResourceRef<Texture> source, const VkRect2D& crop) {
    std::lock_guard lock(g_mutex);
    if (!loading_log_) return;
    Texture* t = source.get();
    bool saved = false;
    if (t) {
        fprintf(loading_log_, "present frame=%llu serial=%llu vk=%u size=%dx%d crop=%d,%d,%u,%u\n", (unsigned long long)frame_,
                (unsigned long long)t->serial, unsigned(t->vk_format), t->desc.width, t->desc.height, crop.offset.x,
                crop.offset.y, crop.extent.width, crop.extent.height);
        const FormatInfo* f = format_info(t->desc.format);
        if (t->image && t->aspect == VK_IMAGE_ASPECT_COLOR_BIT && t->desc.samples <= 1 && f && f->bytes == 4 && f->block == 1 &&
            t->vk_format == f->vk)
            saved = dump_raw(*t, "before-scale", loading_dir_);
    }
    fprintf(loading_log_, "raw=%s\n", saved ? "before-scale.raw" : "unavailable");
    const bool closed = fclose(loading_log_) == 0;
    NVN_LOG("loading capture %s: %s\n", loading_dir_.c_str(), saved && closed ? "saved" : "incomplete");
    loading_log_ = nullptr;
    loading_armed_ = false;
    loading_stages_.clear();
    loading_matches_.clear();
}

bool Executor::dump_raw(Texture& t, const std::string& name, const std::string& dir) {
    const uint32_t w = t.desc.width, h = t.desc.height;
    begin();
    flush_clears(&t);
    Staging::Alloc a = staging_->alloc(VkDeviceSize{w} * h * 4, 16);
    if (!a.data) {
        finish();
        return false;
    }
    // The writer can be in an earlier submission, even when this command buffer has no work yet.
    VkMemoryBarrier order{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    order.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    order.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &order, 0, nullptr, 0,
                         nullptr);
    VkBufferImageCopy r{};
    r.bufferOffset = a.offset;
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {w, h, 1};
    note_gpu_work();
    vkCmdCopyImageToBuffer(cmd_, t.image, VK_IMAGE_LAYOUT_GENERAL, a.buffer, 1, &r);
    uint8_t* src = a.data;
    order.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    order.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &order, 0, nullptr, 0, nullptr);
    const auto previous = submitted_.load();
    const VkFence fence = inflight_[cur_].fence;
    finish();
    if (submitted_.load() == previous || vkWaitForFences(vk()->device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        return false;
    mkdir(dir.c_str(), 0775);
    if (FILE* f = fopen((dir + "/" + name + ".raw").c_str(), "wb")) {
        uint32_t hdr[3] = {w, h, static_cast<uint32_t>(t.vk_format)};
        const size_t bytes = size_t{w} * h * 4;
        const bool ok = fwrite(hdr, sizeof hdr, 1, f) == 1 && fwrite(src, 1, bytes, f) == bytes;
        return fclose(f) == 0 && ok;
    }
    return false;
}

// DEPURACIÓN (NIER_VOLCAR_TEX): el nivel 0 entero de una 3D, array o cubemap de 4 bytes por texel (R11G11B10F, R32UI,
// RGBA8…), tal cual, con cabecera {ancho, alto, rebanadas o capas, VkFormat}: los volúmenes de luz que construye la GPU
// (sondas 64x32x384, rejilla 32x80x32, cubemap 256²) para mirarlos fuera.
void Executor::dump_volume(Texture& t, const std::string& name) {
    const FormatInfo* f = format_info(t.desc.format);
    if (!t.image || !f || f->bytes != 4 || f->block != 1 || !(t.aspect & VK_IMAGE_ASPECT_COLOR_BIT)) return;
    const uint32_t w = std::max(t.desc.width, 1), h = std::max(t.desc.height, 1);
    const bool volume = t.desc.target == kTarget3D;
    const uint32_t d = volume ? std::max(t.desc.depth, 1) : 1, layers = volume ? 1 : std::max(t.layers, 1u);
    const VkDeviceSize bytes = VkDeviceSize{w} * h * d * layers * 4;
    if (bytes > kStagingChunk) return;
    begin();
    flush_clears(&t);
    Staging::Alloc a = staging_->alloc(bytes, 16);
    if (!a.data) return;
    barrier();
    VkBufferImageCopy r{};
    r.bufferOffset = a.offset;
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers};
    r.imageExtent = {w, h, d};
    note_gpu_work();
    vkCmdCopyImageToBuffer(cmd_, t.image, VK_IMAGE_LAYOUT_GENERAL, a.buffer, 1, &r);
    uint8_t* src = a.data;
    finish(true);
    std::string dir = "/storage/emulated/0/NieR-volcados";
    mkdir(dir.c_str(), 0775);
    if (FILE* out = fopen((dir + "/" + name + ".raw").c_str(), "wb")) {
        const uint32_t hdr[4] = {w, h, d * layers, static_cast<uint32_t>(t.vk_format)};
        fwrite(hdr, sizeof hdr, 1, out);
        fwrite(src, 1, bytes, out);
        fclose(out);
    }
    NVN_LOG("volcado %s: %ux%ux%u formato %s (nivel 0 entero)\n", name.c_str(), w, h, d * layers, f->name);
}

void Executor::check_astc(Texture& t) {
    if (!t.image || t.desc.width <= 0 || t.desc.height <= 0) return;
    VkContext* c = vk();
    const uint32_t levels = std::min<uint32_t>(std::max(t.desc.levels, 1), 6);
    std::string report;
    for (uint32_t mip = 0; mip < levels; mip++) {
        const uint32_t w = std::max(uint32_t(t.desc.width) >> mip, 1u), h = std::max(uint32_t(t.desc.height) >> mip, 1u);
        if (uint64_t{w} * h * 4 > kStagingChunk - 256) continue;
        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = VK_IMAGE_TYPE_2D;
        ii.format = VK_FORMAT_R8G8B8A8_UNORM;
        ii.extent = {w, h, 1};
        ii.mipLevels = ii.arrayLayers = 1;
        ii.samples = VK_SAMPLE_COUNT_1_BIT;
        ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        VkImage tmp;
        if (vkCreateImage(c->device, &ii, nullptr, &tmp) != VK_SUCCESS) return;
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(c->device, tmp, &req);
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = find_memory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceMemory tmp_mem;
        if (vkAllocateMemory(c->device, &ai, nullptr, &tmp_mem) != VK_SUCCESS) {
            vkDestroyImage(c->device, tmp, nullptr);
            return;
        }
        vkBindImageMemory(c->device, tmp, tmp_mem, 0);
        begin();
        Staging::Alloc a = staging_->alloc(VkDeviceSize{w} * h * 4, 16);
        if (!a.data) return;
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = tmp;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        note_gpu_work();
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        VkImageBlit bl{};
        bl.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, 0, 1};
        bl.srcOffsets[1] = {static_cast<int32_t>(w), static_cast<int32_t>(h), 1};
        bl.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        bl.dstOffsets[1] = {static_cast<int32_t>(w), static_cast<int32_t>(h), 1};
        note_gpu_work();
        vkCmdBlitImage(cmd_, t.image, VK_IMAGE_LAYOUT_GENERAL, tmp, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_NEAREST);
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        note_gpu_work();
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        VkBufferImageCopy r{};
        r.bufferOffset = a.offset;
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.imageExtent = {w, h, 1};
        note_gpu_work();
        vkCmdCopyImageToBuffer(cmd_, tmp, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, a.buffer, 1, &r);
        finish(true);
        uint64_t magenta = 0;
        for (size_t k = 0; k < size_t{w} * h; k++) {
            const uint8_t* p = a.data + k * 4;
            magenta += p[0] == 255 && p[1] == 0 && p[2] == 255;
        }
        vkDestroyImage(c->device, tmp, nullptr);
        vkFreeMemory(c->device, tmp_mem, nullptr);
        char part[64];
        snprintf(part, sizeof part, " m%u %.0f%%", mip, 100.0 * double(magenta) / double(uint64_t{w} * h));
        report += part;
        if (magenta * 100 > uint64_t{w} * h) report += "!";
    }
    const bool bad = report.find('!') != std::string::npos;
    if (bad) {  // dónde está cada nivel según nuestra disposición, para leer la memoria desde fuera
        char part[64];
        snprintf(part, sizeof part, " | memoria %p:", static_cast<void*>(t.storage));
        report += part;
        for (size_t l = 0; l < t.layout.levels.size() && l < 8; l++) {
            snprintf(part, sizeof part, " L%zu +%#zx (%#zx, bh %u)", l, t.layout.levels[l].offset, t.layout.levels[l].size,
                     t.layout.levels[l].block_height_gobs);
            report += part;
        }
    }
    // Las que salieron magenta alguna vez se siguen registrando (¿se arreglan al terminar de cargarse?).
    static std::set<uint64_t> once_bad;
    const bool was_bad = once_bad.count(t.serial) != 0;
    if (bad) once_bad.insert(t.serial);
    static int good_logged = 0;
    if (bad || was_bad || good_logged++ < 10) {
        const uint8_t* p = t.desc.packaged_layout;
        const auto& l0 = t.layout.levels.empty() ? LevelLayout{} : t.layout.levels[0];
        NVN_LOG("astc %s (textura %llu): %dx%d formato 0x%x niveles %d flags 0x%x empaquetado %s%02x %02x %02x %02x %02x %02x %02x %02x "
                "bloque %ux%u:%s\n", bad ? "MAGENTA" : was_bad ? "ARREGLADA" : "bien", static_cast<unsigned long long>(t.serial), t.desc.width, t.desc.height, t.desc.format, t.desc.levels,
                t.desc.flags, t.desc.has_packaged_layout ? "" : "(no) ", p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
                l0.block_height_gobs, l0.block_depth, report.c_str());
    }
}

void Executor::dump(Texture& t, const std::string& name, uint32_t layer, int forced_mip) {
    // Profundidad: con un mip pedido (los volcados por capa de NIER_VOLCAR_TEX), esa capa; si no, las primeras 4.
    if (t.image && (t.aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && t.desc.width > 0 && t.desc.height > 0)
        return dump_depth(t, name, forced_mip >= 0 ? static_cast<int>(layer) : -1);
    if (t.image && t.vk_format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 && layer == 0 && forced_mip < 0) dump_raw(t, name);
    if (!t.image || (t.aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || t.desc.width <= 0 || t.desc.height <= 0) return;
    VkContext* c = vk();
    // DEPURACIÓN: las texturas grandes se vuelcan desde un mip más pequeño (que quepa en el staging).
    uint32_t mip = 0;
    while (mip + 1 < static_cast<uint32_t>(std::max(t.desc.levels, 1)) && (uint64_t{uint32_t(t.desc.width) >> mip} * (uint32_t(t.desc.height) >> mip)) > 2048ull * 2048)
        mip++;
    if (forced_mip >= 0) mip = std::min<uint32_t>(forced_mip, std::max(t.desc.levels, 1) - 1);
    if (layer >= t.layers) layer = 0;
    const uint32_t w = std::max(uint32_t(t.desc.width) >> mip, 1u), h = std::max(uint32_t(t.desc.height) >> mip, 1u);
    VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ii.imageType = VK_IMAGE_TYPE_2D;
    ii.format = VK_FORMAT_R8G8B8A8_UNORM;
    ii.extent = {w, h, 1};
    ii.mipLevels = ii.arrayLayers = 1;
    ii.samples = VK_SAMPLE_COUNT_1_BIT;
    ii.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VkImage tmp;
    if (vkCreateImage(c->device, &ii, nullptr, &tmp) != VK_SUCCESS) return;
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(c->device, tmp, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = find_memory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory tmp_mem;
    vkAllocateMemory(c->device, &ai, nullptr, &tmp_mem);
    vkBindImageMemory(c->device, tmp, tmp_mem, 0);
    begin();
    flush_clears(&t);
    Staging::Alloc a = staging_->alloc(VkDeviceSize{w} * h * 4, 16);
    if (!a.data) return;
    auto to = [&](VkImageLayout from, VkImageLayout layout) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        b.oldLayout = from;
        b.newLayout = layout;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = tmp;
        b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        note_gpu_work();
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &b);
    };
    to(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    {  // gris de control: si sobrevive, el blit no escribió
        VkClearColorValue gray{{0.5f, 0.5f, 0.5f, 1.0f}};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        note_gpu_work();
        vkCmdClearColorImage(cmd_, tmp, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &gray, 1, &range);
        to(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    }
    const bool control = name == "control";
    VkImageBlit bl{};
    bl.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, mip, layer, 1};
    bl.srcOffsets[1] = {static_cast<int32_t>(w), static_cast<int32_t>(h), 1};
    bl.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    bl.dstOffsets[1] = {static_cast<int32_t>(w), static_cast<int32_t>(h), 1};
    note_gpu_work();
    if (!control)
        vkCmdBlitImage(cmd_, t.image, VK_IMAGE_LAYOUT_GENERAL, tmp, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl,
                       VK_FILTER_NEAREST);
    to(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy r{};
    r.bufferOffset = a.offset;
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {w, h, 1};
    note_gpu_work();
    vkCmdCopyImageToBuffer(cmd_, tmp, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, a.buffer, 1, &r);
    std::vector<uint8_t> rgba(size_t{w} * h * 4);
    uint8_t* src = a.data;
    finish(true);  // espera a la GPU; los datos siguen en el staging hasta que se reutilice el contexto
    memcpy(rgba.data(), src, rgba.size());
    vkDestroyImage(c->device, tmp, nullptr);
    vkFreeMemory(c->device, tmp_mem, nullptr);

    // En la memoria interna: escribir megas en la SD (FUSE) a veces se bloquea segundos.
    std::string dir = "/storage/emulated/0/NieR-volcados";
    mkdir(dir.c_str(), 0775);
    if (FILE* f = fopen((dir + "/" + name + ".ppm").c_str(), "wb")) {
        fprintf(f, "P6 %u %u 255\n", w, h);
        std::vector<uint8_t> rgb(size_t{w} * h * 3);
        for (size_t k = 0; k < size_t{w} * h; k++) memcpy(&rgb[k * 3], &rgba[k * 4], 3);
        fwrite(rgb.data(), 1, rgb.size(), f);
        fclose(f);
    }
    if (FILE* f = fopen((dir + "/" + name + "_a.pgm").c_str(), "wb")) {  // el alfa, aparte
        fprintf(f, "P5 %u %u 255\n", w, h);
        std::vector<uint8_t> alpha(size_t{w} * h);
        for (size_t k = 0; k < alpha.size(); k++) alpha[k] = rgba[k * 4 + 3];
        fwrite(alpha.data(), 1, alpha.size(), f);
        fclose(f);
    }
    NVN_LOG("volcado %s: %ux%u formato %s\n", name.c_str(), w, h, format_info(t.desc.format) ? format_info(t.desc.format)->name : "?");
}

// Cada 300 frames: ms por frame dentro de run() (CPU del executor + espera a la GPU), de espera a la
// GPU, envíos y draws por frame.
namespace {
// MB que KGSL tiene asignados a este proceso (-1 si el kernel no lo expone).
long kgsl_process_mb() {
    char path[64];
    snprintf(path, sizeof path, "/sys/class/kgsl/kgsl/proc/%d/kernel", getpid());
    long long bytes = -1;
    if (FILE* f = fopen(path, "r")) {
        if (fscanf(f, "%lld", &bytes) != 1) bytes = -1;
        fclose(f);
    }
    return bytes < 0 ? -1 : static_cast<long>(bytes >> 20);
}
}  // namespace

// Un salto grande de memoria de la GPU deja constancia en el acto, con lo que cuento yo al lado.
void Executor::watch_kgsl() {
    auto now = std::chrono::steady_clock::now();
    if (now - kgsl_checked_ < std::chrono::milliseconds(500)) return;
    kgsl_checked_ = now;
    long mb = kgsl_process_mb();
    if (mb < 0 || mb < kgsl_logged_mb_ + 256) {
        if (mb >= 0 && mb + 256 < kgsl_logged_mb_) kgsl_logged_mb_ = mb;  // bajó: el siguiente salto se mide desde aquí
        return;
    }
    size_t pools = 0;
    VkDeviceSize staging = 0;
    for (const InFlight& f : inflight_) pools += f.pools.size(), staging += f.staging.bytes();
    size_t blocks = 0, dedicated = 0;
    VkDeviceSize block_bytes = 0;
    gpu_memory_stats(&blocks, &dedicated, &block_bytes);
    NVN_LOG("memoria de la GPU: %ld MB (antes %ld); míos: texturas %llu MB (%zu reservas propias), staging %llu MB, caché de "
            "vértices %llu MB, overlays %llu MB, %zu pools de descriptores, %zu pipelines, %zu vistas\n",
            mb, kgsl_logged_mb_, (unsigned long long)(g_texture_gpu_bytes.load() >> 20), dedicated,
            (unsigned long long)(staging >> 20), (unsigned long long)(buffer_cache_.bytes() >> 20),
            (unsigned long long)(overlay_arena_.bytes() >> 20), pools, pipelines_.size(), views_.size());
    kgsl_logged_mb_ = mb;
}


// --- Precalentado de pipelines ---
// Cada pipeline que crea el ejecutor se apunta (PipelineRecord) en <datos>/pipelines_estado.bin, en el orden en que sale
// la primera vez (título, aldea, …). Al arrancar, warm_up las vuelve a crear en un hilo propio, de prioridad baja, con el
// SPIR-V de la caché de SPIR-V y una caché del driver propia que empieza con pipelines.bin (cada una cuesta ~0,3 ms si el
// driver la tiene, 10-500 ms si no) y que al acabar se mezcla en la del ejecutor. El ejecutor las toma cuando el juego las
// pide (warm_take) en vez de crearlas en NvnSubmit con el frame parado. Partida del 4-11 sin esto: 9354 pipelines, 20,6 s
// creándolas en el driver (2,8 s en las que el driver ya tenía), más 5,7 s de traducciones.
// <datos>/no_warmup al arrancar: no se crea ninguna (se siguen apuntando).
namespace {
constexpr char kRecordsMagic[8] = {'N', 'V', 'P', 'I', 'P', 'E', 'S', '1'};
constexpr size_t kRecordsHeader = 32;
constexpr size_t kWarmMax = 4000;               // como mucho (lo de la aldea y la primera cacería está al principio)
constexpr long kWarmMinAvailableKb = 700 * 1024;  // con menos RAM libre en el sistema, se para
std::string pipeline_records_path() {
    const char* data = getenv("NIER_DATA");
    return std::string(data ? data : "/sdcard/NieR") + "/pipelines_estado.bin";
}
long available_kb() {
    long kb = -1;
    if (FILE* f = fopen("/proc/meminfo", "r")) {
        char line[128];
        while (fgets(line, sizeof line, f))
            if (sscanf(line, "MemAvailable: %ld kB", &kb) == 1) break;
        fclose(f);
    }
    return kb;
}
}  // namespace

// Claim the key before a foreground build. If warm-up already owns it, join its publication.
std::optional<Executor::PipelineBuild> Executor::warm_take(uint64_t h) {
    if (!warm_started_) return {};
    PipelineFuture build;
    {
        std::lock_guard lock(warm_mutex_);
        warm_done_.insert(h);
        auto it = warm_builds_.find(h);
        if (it == warm_builds_.end()) return {};
        build = std::move(it->second);
        warm_builds_.erase(it);
    }
    if (build.wait_for(std::chrono::seconds(0)) != std::future_status::ready) ++prof_warm_waits_;
    return await_pipeline(build);
}

Executor::PipelineBuild Executor::await_pipeline(const PipelineFuture& build) {
    const auto start = std::chrono::steady_clock::now();
    auto result = build.get();
    g_drop_stats.pipeline_wait_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count(), std::memory_order_relaxed);
    return result;
}

// Solo calienta la caché del driver: la pipeline se crea y se destruye. Guardarla y apuntarla para el
// precalentado tenía en memoria las de cientos de compute que el juego fija y nunca lanza (1257 en una partida del 5-11)
// y el precalentado las volvía a crear al arrancar. El ejecutor la crea al primer dispatch, con la caché ya caliente.
void Executor::prefetch_compute(const ProgramStage& cs) {
    constexpr int S = static_cast<int>(Stage::Compute);
    if (!warm_started_ || !cs.module || !cs.spirv_id) return;
    PipelineKey key;
    memset(&key, 0, sizeof key);
    key.stages[S] = cs.spirv_id;
    key.topology = kComputeTopology;
    const uint64_t h = hash_bytes(&key, sizeof key);  // la de compute_pipeline (sin pipelines_per_stage ni stripped)
    {
        std::lock_guard lock(warm_mutex_);
        if (warm_done_.count(h) || warm_builds_.count(h)) return;
    }
    const nvnshader::Translated* shaders[kStages] = {};
    shaders[S] = &cs.shader;
    Layout& l = layout_for(shaders, key.stages);
    VkPipeline p = VK_NULL_HANDLE;
    const auto t0 = std::chrono::steady_clock::now();
    if (driver_cache_.compile([&](VkPipelineCache cache) {
        return make_compute_pipeline(cs.module, l.pipeline, cache, p);
    }) != VK_SUCCESS) return;
    if (!p) return;
    g_drop_stats.prefetch_ns.fetch_add(
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()),
        std::memory_order_relaxed);
    vkDestroyPipeline(vk()->device, p, nullptr);
    driver_cache_.note_prefetch();
}

namespace {
int pipeline_worker_count() {
    static const int workers = NVN_FLAG("pipeline_threads_4") ? 4 : NVN_FLAG("pipeline_threads_3") ? 3
                               : NVN_FLAG("pipeline_threads_1") ? 1 : 2;
    return workers;
}
}  // namespace

Executor::PipelineFuture Executor::enqueue_pipeline(PipelineTask task, bool first_use) {
    std::call_once(pipeline_workers_once_, [this] {
        const int workers = pipeline_worker_count();
        NVN_LOG("pipelines: %d worker thread(s)\n", workers);
        for (int i = 0; i < workers; ++i) std::thread([this] { async_loop(); }).detach();
    });
    auto result = task.get_future().share();
    {
        std::lock_guard lock(async_mutex_);
        if (first_use) async_queue_.push_front(std::move(task));
        else async_queue_.push_back(std::move(task));
    }
    async_cv_.notify_one();
    return result;
}

void Executor::async_loop() {
    pthread_setname_np(pthread_self(), "PipelinesAsync");
    setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), 10);  // por debajo de los hilos del juego
    pin_background_thread();  // núcleos pequeños (lo crea NvnSubmit, que está en los grandes)
    for (;;) {
        PipelineTask job;
        {
            std::unique_lock lock(async_mutex_);
            async_cv_.wait(lock, [&] { return !async_queue_.empty(); });
            job = std::move(async_queue_.front());
            async_queue_.pop_front();
        }
        job();
    }
}

void Executor::record_pipeline(const PipelineRecord& r) {
    PipelineRecord rec = r;
    rec.magic = kRecordMagic;
    rec.sum = record_sum(rec);
    std::lock_guard lock(records_mutex_);
    if (records_pending_.size() < 20000) records_pending_.push_back(rec);
}

// Desde profile_frame (cada 30 o 300 frames): lo apuntado se añade al final del fichero en otro hilo. Son ~600 bytes por
// pipeline nueva: nada que ver con reescribir pipelines.bin entero.
void Executor::save_pipeline_records() {
    save_recent_pipelines();
    {
        std::lock_guard lock(records_mutex_);
        if (!records_loaded_ || records_writing_ || records_pending_.empty()) return;
        records_writing_ = true;
    }
    std::thread([this] {
        // Lo crea NvnSubmit (nice -10, ver kSubmitNice): unos KB a la SD, por debajo de los hilos del juego.
        setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), 19);
        std::vector<PipelineRecord> batch;
        {
            std::lock_guard lock(records_mutex_);
            batch.swap(records_pending_);
        }
        std::vector<PipelineRecord> fresh;
        std::vector<uint64_t> hashes;
        {
            std::lock_guard lock(records_mutex_);
            for (const PipelineRecord& r : batch) {
                const uint64_t h = hash_bytes(&r.key, sizeof r.key);
                if (records_known_.count(h)) continue;
                fresh.push_back(r);
                hashes.push_back(h);
            }
        }
        const bool ok =
            fresh.empty() || records_file::append(pipeline_records_path(), fresh.data(), fresh.size() * sizeof(PipelineRecord));
        std::lock_guard lock(records_mutex_);
        if (ok) {
            for (uint64_t h : hashes) records_known_.insert(h);
        } else {
            // A medias o sin fichero: lo que haya quedado se descarta al leer (suma que no cuadra). No se reintenta.
            NVN_LOG("precalentado: no se pudieron añadir %zu registros a %s\n", fresh.size(), pipeline_records_path().c_str());
        }
        records_writing_ = false;
    }).detach();
}

// <data>/pipelines_recent.bin: a header (magic, translator build) and the keys of the last session's pipelines, this
// session's first and then the ones from the previous file it didn't use (at most kRecentMax). Rewritten whole (about
// 60 KB, through a .tmp and a rename) when the executor has new pipelines; the warm-up starts with them.
namespace {
constexpr uint64_t kRecentMagic = 0x31544e4543455256ull;  // "VRECENT1"
constexpr size_t kRecentMax = 8000;
std::string recent_pipelines_path() {
    const char* data = getenv("NIER_DATA");
    return std::string(data ? data : "/sdcard/NieR") + "/pipelines_recent.bin";
}
}  // namespace

void Executor::save_recent_pipelines() {
    if (pipelines_.size() == recent_written_ || recent_writing_.load(std::memory_order_relaxed)) return;
    std::vector<uint64_t> hashes;
    hashes.reserve(pipelines_.size());
    for (const auto& [h, p] : pipelines_)
        if (p) hashes.push_back(h);
    recent_written_ = pipelines_.size();
    recent_writing_.store(true, std::memory_order_relaxed);
    std::thread([this, hashes = std::move(hashes)]() mutable {
        setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), 19);
        {
            std::lock_guard lock(records_mutex_);
            std::unordered_set<uint64_t> seen(hashes.begin(), hashes.end());
            for (uint64_t h : recent_prev_)
                if (hashes.size() < kRecentMax && seen.insert(h).second) hashes.push_back(h);
        }
        if (hashes.size() > kRecentMax) hashes.resize(kRecentMax);
        const std::string path = recent_pipelines_path(), tmp = path + ".tmp";
        bool ok = false;
        if (FILE* f = fopen(tmp.c_str(), "wb")) {
            const uint64_t header[2] = {kRecentMagic, static_cast<uint64_t>(NVN_TRADUCTOR_HUELLA)};
            ok = fwrite(header, sizeof header, 1, f) == 1 &&
                 (hashes.empty() || fwrite(hashes.data(), sizeof(uint64_t), hashes.size(), f) == hashes.size());
            ok = fclose(f) == 0 && ok;
        }
        if (ok) ok = rename(tmp.c_str(), path.c_str()) == 0;
        if (!ok) {
            unlink(tmp.c_str());
            static std::atomic<int> logged{0};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 3) NVN_LOG("warm-up: couldn't write %s\n", path.c_str());
        }
        recent_writing_.store(false, std::memory_order_relaxed);
    }).detach();
}

std::vector<uint64_t> Executor::load_recent_pipelines() {
    std::vector<uint64_t> hashes;
    if (FILE* f = fopen(recent_pipelines_path().c_str(), "rb")) {
        uint64_t header[2] = {};
        if (fread(header, sizeof header, 1, f) == 1 && header[0] == kRecentMagic &&
            header[1] == static_cast<uint64_t>(NVN_TRADUCTOR_HUELLA)) {
            uint64_t h;
            while (hashes.size() < kRecentMax && fread(&h, sizeof h, 1, f) == 1) hashes.push_back(h);
        }
        fclose(f);
    }
    std::lock_guard lock(records_mutex_);
    recent_prev_ = hashes;
    return hashes;
}

// Lee los registros válidos (cabecera de este formato y de este traductor; se para en el primero roto) y deja el fichero
// listo para añadir (records_file::load).
std::vector<PipelineRecord> Executor::load_pipeline_records() {
    const std::string path = pipeline_records_path();
    uint8_t header[kRecordsHeader] = {};
    memcpy(header, kRecordsMagic, 8);
    const uint32_t format = kRecordFormat, size = sizeof(PipelineRecord);
    const uint64_t build = NVN_TRADUCTOR_HUELLA;
    memcpy(header + 8, &format, 4);
    memcpy(header + 12, &size, 4);
    memcpy(header + 16, &build, 8);
    const records_file::Loaded file =
        records_file::load(path, header, sizeof header, sizeof(PipelineRecord), [](const uint8_t* p) {
            PipelineRecord r;
            memcpy(&r, p, sizeof r);
            return r.magic == kRecordMagic && r.kind <= 1 && r.sum == record_sum(r);
        });
    std::vector<PipelineRecord> records;
    std::unordered_set<uint64_t> known;
    for (size_t k = 0; k + sizeof(PipelineRecord) <= file.records.size(); k += sizeof(PipelineRecord)) {
        PipelineRecord r;
        memcpy(&r, file.records.data() + k, sizeof r);
        if (known.insert(hash_bytes(&r.key, sizeof r.key)).second) records.push_back(r);
    }
    NVN_LOG("precalentado: %zu pipelines apuntadas en %s (%zu KB%s%s)\n", records.size(), path.c_str(),
            file.file_bytes >> 10, file.reset && file.file_bytes ? ", de otro traductor o formato: vaciado" : "",
            file.cut ? ", final roto cortado" : "");
    std::lock_guard lock(records_mutex_);
    records_known_ = std::move(known);
    records_loaded_ = file.ready;
    return records;
}

void Executor::warm_up() {
    pthread_setname_np(pthread_self(), "Precalentado");
    setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), 10);  // este hilo, por debajo de los del juego
    const auto start = std::chrono::steady_clock::now();
    std::vector<PipelineRecord> records = load_pipeline_records();
    std::vector<uint8_t> initial = driver_cache_.take_warm_data();
    // The last session's first (pipelines_recent.bin), each group in its usual order. Only the first kWarmMax get made,
    // and in first-seen order that's the title screen and the village: loading a save mid-game found its area cold.
    if (const std::vector<uint64_t> recent = load_recent_pipelines(); !recent.empty()) {
        const std::unordered_set<uint64_t> set(recent.begin(), recent.end());
        const auto mid = std::stable_partition(records.begin(), records.end(), [&](const PipelineRecord& r) {
            return set.count(hash_bytes(&r.key, sizeof r.key)) != 0;
        });
        NVN_LOG("warm-up: %zu of the last session's %zu go first\n", static_cast<size_t>(mid - records.begin()), recent.size());
    }
    if (NVN_FLAG("no_warmup") || records.empty()) return;
    if (!shader_cache_ready(60000)) {
        NVN_LOG("precalentado: sin la caché de SPIR-V, no se crea nada\n");
        return;
    }
    const VkDevice device = vk()->device;
    VkPipelineCache cache = VK_NULL_HANDLE;
    {
        VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        ci.initialDataSize = initial.size();
        ci.pInitialData = initial.empty() ? nullptr : initial.data();
        if (vkCreatePipelineCache(device, &ci, nullptr, &cache) != VK_SUCCESS) {
            ci.initialDataSize = 0;
            ci.pInitialData = nullptr;
            if (vkCreatePipelineCache(device, &ci, nullptr, &cache) != VK_SUCCESS) cache = VK_NULL_HANDLE;
        }
        std::vector<uint8_t>().swap(initial);
    }
    // Lo de cada clave de la caché de SPIR-V, una vez: la traducción sin su SPIR-V (para el layout y las identidades) y,
    // si entra en alguna pipeline, su módulo (se destruyen al acabar: la pipeline ya no los necesita).
    struct Resolved {
        bool ok = false;
        nvnshader::Translated meta;  // sin spirv
        uint64_t id = 0;
        std::unordered_map<uint64_t,VkShaderModule> modules;
    };
    std::unordered_map<spirv_cache::Key, Resolved, spirv_cache::KeyHash> resolved;
    auto resolve = [&](const spirv_cache::Key& key, VkShaderModule* module, uint64_t fetch_types = 0) -> Resolved* {
        auto [it, fresh] = resolved.try_emplace(key);
        Resolved& r = it->second;
        nvnshader::Translated t;
        if (fresh) {
            r.ok = cached_translation(key, r.meta);
            if (r.ok) {
                r.id = spirv_identity(r.meta);
                t.spirv.swap(r.meta.spirv);
            }
        }
        if (!r.ok) return nullptr;
        if (module) {
            auto [m, missing] = r.modules.try_emplace(fetch_types, VK_NULL_HANDLE);
            if (missing && (fresh || cached_translation(key, t))) {
                create_fetch_module(device, t.spirv, fetch_types, &m->second);
            }
            *module = m->second;
            if (!*module) return nullptr;
        }
        return &r;
    };
    size_t created = 0, missing = 0, taken = 0, failed = 0, slow = 0, done = 0;
    double slow_ms = 0;
    bool low_memory = false;
    std::deque<PipelineFuture> in_flight;
    auto finish_oldest = [&] {
        const auto& result = in_flight.front().get();
        if (result.status == VK_SUCCESS) ++created;
        else ++failed;
        const double ms = result.driver_ns / 1e6;
        if (ms > 5) { ++slow; slow_ms += ms; }
        in_flight.pop_front();
    };
    for (PipelineRecord rec : records) {
        while (!in_flight.empty() &&
               (in_flight.size() >= size_t(pipeline_worker_count()) || created + in_flight.size() >= kWarmMax))
            finish_oldest();
        if (created >= kWarmMax) break;
        // (recorded with or without extended dynamic state: keyed as this run keys them, so the ones that now share a
        // pipeline make it once)
        if (rec.kind == 0 && vk()->extended_dynamic_state) normalize_dynamic(rec.key);
        if (rec.kind == 0 && vk()->vertex_input_dynamic_state) normalize_vertex_input(rec.key);
        if (done++ % 64 == 0) {
            const long kb = available_kb();
            if (kb >= 0 && kb < kWarmMinAvailableKb) {
                low_memory = true;
                break;
            }
        }
        const nvnshader::Translated* shaders[kStages] = {};
        VkShaderModule modules[kStages] = {};
        bool ok = true;
        for (int s = 0; s < kStages && ok; s++) {
            if (!rec.key.stages[s]) continue;
            const Resolved* base = resolve(rec.base[s], nullptr);
            ok = base && base->id == rec.key.stages[s];  // otra traducción con la misma clave: no vale
            if (!ok) break;
            shaders[s] = &base->meta;
            if (rec.module[s].a | rec.module[s].b) {
                const uint64_t types = s == static_cast<int>(Stage::Vertex) ? vertex_input_types(rec.key, base->meta.input_generics) : 0;
                ok = resolve(rec.module[s], &modules[s], types) != nullptr;
            }
        }
        // Sin el módulo de la etapa que manda (un registro de otra versión del ejecutor), nada.
        ok = ok && (rec.kind == 1 ? modules[static_cast<int>(Stage::Compute)] != VK_NULL_HANDLE
                                  : modules[static_cast<int>(Stage::Vertex)] != VK_NULL_HANDLE);
        if (!ok) {
            missing++;
            continue;
        }
        if (rec.kind == 0) apply_fragment_exports(rec.key, shaders[static_cast<int>(Stage::Fragment)]);
        const uint64_t h = hash_bytes(&rec.key, sizeof rec.key);
        const VkPipelineLayout layout = layout_for(shaders, rec.key.stages).pipeline;
        VkRenderPass rp = VK_NULL_HANDLE;
        if (rec.kind == 0) {
            VkFormat colors[8];
            for (int k = 0; k < 8; ++k) colors[k] = static_cast<VkFormat>(rec.key.pass_colors[k]);
            rp = render_pass(colors, rec.key.color_count, static_cast<VkFormat>(rec.key.pass_depth),
                             static_cast<VkSampleCountFlagBits>(rec.key.samples));
            if (!rp) { ++failed; continue; }
        }
        const auto* vs = shaders[static_cast<int>(Stage::Vertex)];
        const uint64_t inputs = vs ? vs->input_generics : 0;
        std::lock_guard lock(warm_mutex_);
        if (warm_done_.count(h) || warm_builds_.count(h)) { ++taken; continue; }
        PipelineTask task([rec, modules, inputs, layout, rp, cache] {
            PipelineBuild result;
            const auto start = std::chrono::steady_clock::now();
            result.status = rec.kind == 1
                ? make_compute_pipeline(modules[static_cast<int>(Stage::Compute)], layout, cache, result.pipeline)
                : make_graphics_pipeline(rec.key, modules, inputs, layout, rp, cache, result.pipeline);
            result.driver_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
            if (result.status != VK_SUCCESS) std::copy(std::begin(rec.module), std::end(rec.module), result.modules);
            return result;
        });
        auto build = enqueue_pipeline(std::move(task), false);
        warm_builds_.emplace(h, build);
        in_flight.push_back(std::move(build));
    }
    while (!in_flight.empty()) finish_oldest();  // jobs borrow the resolved modules and private driver cache
    for (auto& [key, r] : resolved)
        for (auto& [types, module] : r.modules)
            if (module) vkDestroyShaderModule(device, module, nullptr);
    if (cache) {
        driver_cache_.merge(cache, slow);
        vkDestroyPipelineCache(device, cache, nullptr);
    }
    NVN_LOG("precalentado: %zu pipelines creadas de %zu apuntadas en %.1f s (%zu compiladas por el driver, %.0f ms), "
            "%zu sin su SPIR-V en la caché, %zu ya las tenía el ejecutor, %zu fallidas%s\n",
            created, records.size(),
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), slow, slow_ms, missing, taken,
            failed, low_memory ? "; parado: poca RAM libre" : "");
}

void Executor::profile_frame() {
    // DEPURACIÓN: <datos>/profile_30 lo saca cada 30 frames (con el juego a pocos fps, 300 son minutos).
    const int period = NVN_FLAG("profile_30") ? 30 : 300;
    if (frame_ % period) return;
    std::lock_guard lock(g_mutex);
    // Sin <datos>/verbose_log, solo lo que hace falta (guardar la caché de pipelines): el perfil lee /proc, recorre la
    // memoria y escribe una docena de líneas en la SD cada 10 s.
    if (!NVN_FLAG("verbose_log") && !NVN_FLAG("perf_stats")) {  // (perf_stats: just the profile, without the slow log)
        driver_cache_.save(frame_);
        save_pipeline_records();
        load_period_ = LoadProf{};
        load_period_runs_ = load_period_slow_ = load_period_run_ns_ = load_period_worst_ns_ = 0;
        return;
    }
    NVN_LOG("resolución dinámica: GPU del frame %.1f ms (%s), a LSFG se le reservan %.1f ms; escena %dx%d (ancho mín %d)\n",
            frame_gpu_ns_.load(std::memory_order_relaxed) / 1e6, gpu_stamps_ ? "timestamps de la GPU" : "reloj de la CPU",
            stamp_extra_ / 0.6144 / 1e6, scene_vp_w_, scene_vp_h_, scene_vp_min_w_ == 1 << 30 ? 0 : scene_vp_min_w_);
    scene_vp_w_ = scene_vp_h_ = 0;
    scene_vp_min_w_ = 1 << 30;
    driver_cache_.save(frame_);
    save_pipeline_records();
    size_t blocks = 0, dedicated = 0;
    VkDeviceSize block_bytes = 0;
    gpu_memory_stats(&blocks, &dedicated, &block_bytes);
    {  // RAM: la "memoria de GPU" del Adreno es RAM del sistema
        VkDeviceSize staging = 0;
        for (const InFlight& f : inflight_) staging += f.staging.bytes();
        long pages = 0, resident = 0;
        if (FILE* f = fopen("/proc/self/statm", "r")) {
            if (fscanf(f, "%ld %ld", &pages, &resident) != 2) resident = 0;
            fclose(f);
        }
        long available_kb = 0;
        if (FILE* f = fopen("/proc/meminfo", "r")) {
            char line[128];
            while (fgets(line, sizeof line, f))
                if (sscanf(line, "MemAvailable: %ld kB", &available_kb) == 1) break;
            fclose(f);
        }
        NVN_LOG("perfil RAM: proceso %ld MB, GPU %ld MB, disponible en el sistema %ld MB; míos: staging %llu MB, caché de vértices %llu MB, "
                "overlays %llu MB, texturas %llu MB en bloques\n",
                resident * 4096 / (1 << 20), kgsl_process_mb(), available_kb / 1024, (unsigned long long)(staging >> 20),
                (unsigned long long)(buffer_cache_.bytes() >> 20), (unsigned long long)(overlay_arena_.bytes() >> 20),
                (unsigned long long)(block_bytes >> 20));
    }
    NVN_LOG("perfil memoria: %llu texturas en la GPU (%llu MB; %zu bloques de %llu MB y %zu reservas propias), %zu vistas, "
            "%zu framebuffers, %zu overlays\n",
            (unsigned long long)g_texture_gpu_images.load(), (unsigned long long)(g_texture_gpu_bytes.load() >> 20), blocks,
            (unsigned long long)(block_bytes >> 20), dedicated, views_.size(), framebuffers_.size(), overlays_.size());
    NVN_LOG("perfil cargas: %llu envíos (%llu de más de %.0f ms, el peor %.0f ms), %.0f ms en total: %s\n",
            static_cast<unsigned long long>(load_period_runs_), static_cast<unsigned long long>(load_period_slow_), kSlowRunMs,
            load_period_worst_ns_ / 1e6, load_period_run_ns_ / 1e6, load_summary(load_period_, load_period_run_ns_).c_str());
    {
        const uint64_t waits = g_executor_lock_waits.exchange(0, std::memory_order_relaxed);
        const uint64_t wait_ns = g_executor_lock_wait_ns.exchange(0, std::memory_order_relaxed);
        if (waits) NVN_LOG("perfil cargas: hilos del juego esperando al executor (texturas, samplers): %.0f ms en %llu\n", wait_ns / 1e6,
                           static_cast<unsigned long long>(waits));
    }
    load_period_ = LoadProf{};
    load_period_runs_ = load_period_slow_ = load_period_run_ns_ = load_period_worst_ns_ = 0;
    NVN_LOG("perfil: %.1f ms/frame en el executor (%.1f esperando a la GPU), %.1f envíos, %.0f draws y %.1f render passes por frame\n",
            prof_run_us_ / double(period) / 1000.0, prof_gpu_wait_us_ / double(period) / 1000.0, prof_submits_ / double(period),
            (draws_ - prof_draws_) / double(period), prof_render_passes_ / double(period));
    NVN_LOG("perfil compute: %.1f dispatches y %.1f esperas por write-back por frame; lecturas de lo escrito: vértices %llu, "
            "índices %llu, ubo %llu, ssbo %llu, indirecto %llu, ssbo de compute %llu; %zu rangos\n",
            (dispatches_ - prof_dispatches_) / double(period), prof_writebacks_ / double(period), (unsigned long long)prof_reads_[0],
            (unsigned long long)prof_reads_[1], (unsigned long long)prof_reads_[2], (unsigned long long)prof_reads_[3],
            (unsigned long long)prof_reads_[4], (unsigned long long)prof_reads_[5], gpu_written_ranges_.size());
    NVN_LOG("perfil compute: %llu lecturas solo parcialmente dentro; texturas subidas: %llu MB\n", (unsigned long long)prof_partial_,
            (unsigned long long)(prof_texture_bytes_ >> 20));
    NVN_LOG("texture checks: %llu full checks found changes; %.2f MB/frame hashed; %llu watched textures since startup\n",
            (unsigned long long)prof_audit_changes_,
            prof_full_hash_bytes_ / double(period) / 1048576.0, (unsigned long long)prof_watched_textures_);
    prof_audit_changes_ = prof_full_hash_bytes_ = 0;
    if (prof_astc_encoded_ || prof_astc_cached_ || prof_astc_empty_ || prof_astc_direct_ || prof_astc_deferred_ || prof_astc_applied_)
        NVN_LOG("perfil astc: %llu levels encoded, %llu from the cache, %llu BC1 converted directly, %llu empty (not streamed in yet); "
                "%llu sent to the background encoder, %llu back from it\n",
                (unsigned long long)prof_astc_encoded_, (unsigned long long)prof_astc_cached_,
                (unsigned long long)prof_astc_direct_, (unsigned long long)prof_astc_empty_,
                (unsigned long long)prof_astc_deferred_, (unsigned long long)prof_astc_applied_);
    prof_astc_encoded_ = prof_astc_cached_ = prof_astc_empty_ = prof_astc_direct_ = prof_astc_deferred_ = prof_astc_applied_ = 0;
    if (prof_count_checked_)
        NVN_LOG("perfil multi-draw: %.1f per frame with the draw count check\n", prof_count_checked_ / double(period));
    prof_count_checked_ = 0;
    NVN_LOG("perfil fsr: %llu de %d frames con el blit por la guarda (frame del juego > %.1f ms)\n",
            static_cast<unsigned long long>(prof_fsr_guard_frames_), period, (frame_period_ns() < 20'000'000 ? kFsrOffNs60 : kFsrOffNs) / 1e6);
    prof_fsr_guard_frames_ = 0;
    NVN_LOG("perfil overlays: %.2f MB nuevos de la arena y %.2f MB en huecos reutilizados por frame, %zu huecos libres, "
            "arena %llu MB; %llu compactaciones (%.1f ms esperando a la GPU y %.1f ms copiando cada una)\n",
            prof_overlay_new_bytes_ / double(period) / 1048576.0, prof_overlay_reused_bytes_ / double(period) / 1048576.0,
            overlay_free_.size(), static_cast<unsigned long long>(overlay_used_ >> 20), static_cast<unsigned long long>(prof_compactions_),
            prof_compactions_ ? prof_compact_wait_us_ / 1000.0 / double(prof_compactions_) : 0.0,
            prof_compactions_ ? prof_compact_copy_us_ / 1000.0 / double(prof_compactions_) : 0.0);
    {  // Lo escrito por la GPU que se tiró (ver note_lost), por motivo: con la cara a la vista o en una cacería, que la CPU
       // lo reescriba es lo esperado y lo solapado en parte suele ser memoria reutilizada; lo demás debería quedar a 0.
        uint64_t lost = 0, lost_bytes = 0;
        std::string reasons;
        char b[48];
        for (const LostCount& l : prof_lost_) {
            lost += l.count, lost_bytes += l.bytes;
            snprintf(b, sizeof b, ": %llu (%.1f KB)", static_cast<unsigned long long>(l.count), l.bytes / 1024.0);
            reasons += std::string(" | ") + l.why + b;
        }
        NVN_LOG("perfil overlays: lo escrito por la GPU tirado en %llu overlays (%.1f KB)%s; %llu uniones (%llu de la GPU "
                "conservados dentro, %.1f KB), %llu parches parciales de la CPU, %llu trozos de partidos%s\n",
                static_cast<unsigned long long>(lost), lost_bytes / 1024.0, reasons.c_str(),
                static_cast<unsigned long long>(prof_merged_), static_cast<unsigned long long>(prof_merged_parts_),
                prof_merged_bytes_ / 1024.0, static_cast<unsigned long long>(prof_partial_patched_),
                static_cast<unsigned long long>(prof_split_), !keep_gpu_overlays_ ? " (no_keep_overlays)" : !merge_reads_ ? " (no_merge_reads)" : "");
        prof_lost_.clear();
        prof_merged_ = prof_merged_parts_ = prof_merged_bytes_ = prof_partial_patched_ = prof_split_ = 0;
    }
    prof_overlay_new_bytes_ = prof_overlay_reused_bytes_ = prof_compactions_ = prof_compact_wait_us_ = prof_compact_copy_us_ = 0;
    prof_partial_ = 0;
    memset(prof_reads_, 0, sizeof prof_reads_);
    prof_dispatches_ = dispatches_;
    prof_render_passes_ = prof_writebacks_ = 0;
    // Los tramos por draw se midieron en uno de cada kProfileEvery frames: ms por frame medido.
    const double timed = std::max<uint64_t>(prof_timed_frames_, 1) * 1e6;
    prof_timed_frames_ = 0;
    NVN_LOG("perfil descriptores (ms/frame): buffers %.1f, texturas %.1f, actualizar %.1f, reservar %.1f; dispatch %.1f; draws en total %.1f\n",
            prof_sub_us_[0] / timed, prof_sub_us_[1] / timed, prof_sub_us_[2] / timed, prof_sub_us_[3] / timed,
            prof_dispatch_us_ / timed, prof_draw_total_us_ / timed);
    NVN_LOG("perfil: %.2f MB de uniform/storage buffers copiados por frame; espejos: %zu (%llu MB), %.2f MB subidos por frame "
            "(%.1f al empezar el envío); %llu directos, %.2f MB/frame escritos en ellos\n",
            prof_copied_bytes_ / double(period) / 1048576.0, mirrors_.size(), static_cast<unsigned long long>(mirror_bytes_ >> 20),
            prof_mirror_bytes_ / double(period) / 1048576.0, prof_mirror_preloads_ / double(period),
            static_cast<unsigned long long>(prof_mirrors_direct_), prof_direct_bytes_ / double(period) / 1048576.0);
    prof_mirror_bytes_ = 0;
    prof_direct_bytes_ = 0;
    prof_mirror_preloads_ = 0;
    size_t bindless_layouts = 0;
    {
        std::lock_guard objects_lock(objects_mutex_);
        bindless_layouts = bindless_layouts_.size();
    }
    NVN_LOG("perfil bindless: %.1f sets del set 1 armados por frame (%.2f ms cada uno), %.1f texturas del juego por set; "
            "%zu layouts; %.1f sets por frame reutilizados por contenido\n",
            prof_bindless_sets_ / double(period),
            prof_bindless_sets_ ? prof_bindless_us_ / 1000.0 / double(prof_bindless_sets_) : 0.0,
            prof_bindless_sets_ ? prof_bindless_textures_ / double(prof_bindless_sets_) : 0.0, bindless_layouts,
            prof_bindless_reused_ / double(period));
    prof_bindless_sets_ = prof_bindless_textures_ = prof_bindless_us_ = prof_bindless_reused_ = 0;
    {  // los espejos que más suben (en una cacería, ~26 MB/frame entre todos: ¿cuál y de qué tamaño?)
        std::vector<std::pair<uint64_t, uint64_t>> top;  // (bytes, dirección)
        for (auto& [a, b] : prof_mirror_top_) top.push_back({b, a});
        std::sort(top.rbegin(), top.rend());
        std::string list;
        for (size_t k = 0; k < std::min<size_t>(top.size(), 5); k++) {
            const auto it = mirrors_.find(top[k].second);
            char b[80];
            snprintf(b, sizeof b, " %#llx+%lluK:%.2fMB", static_cast<unsigned long long>(top[k].second),
                     static_cast<unsigned long long>(it != mirrors_.end() ? it->second.size >> 10 : 0), top[k].first / double(period) / 1048576.0);
            list += b;
        }
        if (!list.empty()) NVN_LOG("perfil espejos que más suben (MB/frame):%s\n", list.c_str());
        prof_mirror_top_.clear();
        NVN_LOG("perfil pipelines: %.2f nuevas por frame, %.1f ms/frame creándolas en el driver (%zu en total)\n",
                (driver_cache_.created() - prof_pipelines_logged_) / double(period), prof_pipeline_us_ / double(period) / 1000.0,
                pipelines_.size() + compute_pipelines_.size());
        size_t warm_ready = 0;
        {
            std::lock_guard warm_lock(warm_mutex_);
            warm_ready = warm_builds_.size();
        }
        // Con la clave por etapa de antes (dirección + generación) serían prof_stage_keys_ (solo las gráficas).
        NVN_LOG("perfil pipelines: %zu gráficas por lo traducido (por etapa serían %zu); precalentado: %llu tomadas, %llu "
                "esperas, %zu creadas sin pedir aún\n",
                pipelines_.size(), prof_stage_keys_.size(), static_cast<unsigned long long>(prof_warm_taken_),
                static_cast<unsigned long long>(prof_warm_waits_), warm_ready);
        prof_pipelines_logged_ = driver_cache_.created();
        prof_pipeline_us_ = 0;
    }
    {
        size_t written = 0, small = 0;
        uint64_t biggest = 0;
        for (auto& [a, o] : overlays_)
            if (o.written) written++, small += o.size <= kWriteBackMax, biggest = std::max(biggest, o.size);
        NVN_LOG("perfil: %.2f MB devueltos de overlays a la memoria del juego por frame; overlays %zu, escritos %zu "
                "(%zu pequeños, el mayor %llu KB), completado %llu; entregas %llu (hasta %llu, pendientes %llu desde %llu)\n",
                prof_written_back_ / double(period) / 1048576.0, overlays_.size(), written, small,
                static_cast<unsigned long long>(biggest >> 10), static_cast<unsigned long long>(completed_.load()),
                static_cast<unsigned long long>(prof_deliver_calls_), static_cast<unsigned long long>(prof_deliver_up_to_),
                static_cast<unsigned long long>(prof_deliver_pending_), static_cast<unsigned long long>(prof_deliver_min_));
        prof_deliver_calls_ = 0;
        std::lock_guard lock(wb_mutex_);
        std::string list;
        for (auto& [a, v] : prof_delivered_) {
            char b[64];
            snprintf(b, sizeof b, " %#llx+%lluK:%lluK", static_cast<unsigned long long>(a), static_cast<unsigned long long>(v.first >> 10),
                     static_cast<unsigned long long>(v.second >> 10));
            list += b;
        }
        if (!list.empty()) NVN_LOG("perfil: entregados (dirección+tamaño:bytes)%s\n", list.c_str());
        prof_delivered_.clear();
        list.clear();
        for (auto& [a, v] : prof_wb_seen_) {
            char b[64];
            snprintf(b, sizeof b, " %#llx+%lluK:%s", static_cast<unsigned long long>(a), static_cast<unsigned long long>(v.first >> 10),
                     v.second == kWrittenByGraphics ? "g" : v.second == kWrittenByCompute ? "c" : v.second ? "gc" : "?");
            list += b;
        }
        if (!list.empty()) NVN_LOG("perfil: copias de lectura (dirección+tamaño:quién)%s\n", list.c_str());
        prof_wb_seen_.clear();
    }
    prof_written_back_ = 0;
    {
        static uint64_t last_faults = 0, last_protects = 0;
        const uint64_t f = nnsdk::watch::faults(), pr = nnsdk::watch::protects();
        NVN_LOG("perfil: vigilancia %.1f escrituras que saltaron y %.1f tramos cerrados por frame\n",
                (f - last_faults) / double(period), (pr - last_protects) / double(period));
        last_faults = f, last_protects = pr;
    }
    {
        std::vector<std::pair<uint64_t, uint64_t>> top;  // (bytes, clave)
        for (auto& [k, v] : prof_copied_ranges_) top.push_back({v.first, k});
        std::sort(top.rbegin(), top.rend());
        for (size_t i = 0; i < std::min<size_t>(top.size(), 6); i++) {
            const CopyKey& key = prof_copied_keys_[top[i].second];
            const auto& v = prof_copied_ranges_[top[i].second];
            NVN_LOG("perfil:   %s %#llx+%llu KB (etapa %d): %.1f MB/frame en %.1f copias/frame\n", key.ssbo ? "SSBO" : "UBO",
                    static_cast<unsigned long long>(key.address), static_cast<unsigned long long>(key.size / 1024), key.stage,
                    v.first / double(period) / 1048576.0, v.second / double(period));
        }
        prof_copied_ranges_.clear();
        prof_copied_keys_.clear();
    }
    prof_copied_bytes_ = 0;
    memset(prof_sub_us_, 0, sizeof prof_sub_us_);
    prof_dispatch_us_ = prof_draw_total_us_ = 0;
    NVN_LOG("perfil draw (ms/frame): texturas %.1f, pipeline %.1f, descriptores %.1f, estado %.1f, vértices %.1f, dibujo %.1f\n",
            prof_section_us_[0] / timed, prof_section_us_[1] / timed, prof_section_us_[2] / timed,
            prof_section_us_[3] / timed, prof_section_us_[4] / timed, prof_section_us_[5] / timed);
    memset(prof_section_us_, 0, sizeof prof_section_us_);
    {  // Perfil fino, por frame medido (timed: ns -> ms por frame medido; frames: cuentas por frame medido)
        const double frames = timed / 1e6;
        const FineProf& f = fine_;
        NVN_LOG("perfil fino (ms/frame): pipeline = overlays y espejos %.2f + set 1 %.2f + render pass %.2f + layout y pipeline "
                "%.2f; vértices = índices %.2f + streams %.2f\n",
                prof_split_ns_[0] / timed, prof_split_ns_[1] / timed, prof_split_ns_[2] / timed, prof_split_ns_[3] / timed,
                prof_split_ns_[4] / timed, prof_split_ns_[5] / timed);
        NVN_LOG("perfil fino huellas (por frame): texturas %.0f revisadas, %.0f huellas de %.1f MB en %.2f ms; caché de vértices "
                "%.0f búsquedas, %.0f huellas de %.1f MB en %.2f ms (%.0f vigiladas sin huella); overlays %.0f huellas de %.1f MB "
                "en %.2f ms (%.0f vigilados sin huella); memory pools %.0f búsquedas (%.0f del memo) recorriendo %.0f\n",
                f.tex_checks / frames, f.tex_hashes / frames, f.tex_hash_bytes / frames / 1048576.0, f.tex_hash_ns / timed,
                f.vtx.lookups / frames, f.vtx.hashes / frames, f.vtx.hash_bytes / frames / 1048576.0, f.vtx.hash_ns / timed,
                f.vtx.watched / frames, f.ov_hashes / frames, f.ov_hash_bytes / frames / 1048576.0, f.ov_hash_ns / timed,
                f.ov_watched / frames, f.pool_scans / frames, f.pool_memo / frames, f.pool_visits / frames);
        NVN_LOG("perfil fino sets (por frame): %.0f del draw anterior, %.0f de la tabla, %.0f nuevos (%.0f descriptores, %.0f "
                "con plantilla): reservar %.2f ms, escribir %.2f ms, enlazar %.2f ms; set 1 sin reenlazar %.0f; pipelines %.0f "
                "búsquedas (%.0f del memo), %.0f enlaces; layouts del memo %.0f; set 1 con el layout ya hecho %.0f\n",
                f.sets_memo / frames, f.sets_map / frames, f.sets_new / frames, f.set_writes / frames, f.set_templates / frames,
                f.set_alloc_ns / timed, f.set_update_ns / timed, f.set_bind_ns / timed, f.set1_skipped / frames,
                f.pipe_lookups / frames, f.pipe_memo / frames, f.pipe_binds / frames, f.layout_memo / frames,
                f.bindless_fast / frames);
        fine_ = FineProf{};
        memset(prof_split_ns_, 0, sizeof prof_split_ns_);
    }
    NVN_LOG("perfil tiler (por frame): %.1f borrados como loadOp CLEAR y %.1f aparte, %.1f adjuntos sin guardar, %.1f pases "
            "con área recortada, %.1f barreras aplazadas (%.1f cortaron luego), %.1f re-enlaces sin cortar\n",
            prof_clear_loads_ / double(period), prof_clear_flushes_ / double(period), prof_store_none_ / double(period), prof_area_cut_ / double(period),
            prof_barriers_deferred_ / double(period), prof_barrier_splits_ / double(period), prof_rt_kept_ / double(period));
    NVN_LOG("perfil cortes: %.1f por draws que escriben storage tras una barrera, %.1f por muestrear un adjunto del pase\n",
            prof_split_storage_ / double(period), prof_split_sample_ / double(period));
    prof_split_storage_ = prof_split_sample_ = 0;
    {  // los cortes por tamaño del pase y motivo (ver split_why_), de más a menos: mismo framebuffer + cabían juntos
        std::vector<std::pair<uint64_t, const std::string*>> top;
        for (const auto& [key, c] : prof_cuts_) top.push_back({c.same + c.fits, &key});
        std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::string list;
        for (size_t k = 0; k < std::min<size_t>(top.size(), 16); k++) {
            const CutStat& c = prof_cuts_[*top[k].second];
            char b[160];
            snprintf(b, sizeof b, " %s %.1f+%.1f;", top[k].second->c_str(), c.same / double(period), c.fits / double(period));
            list += b;
        }
        NVN_LOG("perfil barreras (por frame): %.1f completas, %.1f sin nada grabado desde la anterior, %.1f solo tras cerrar un "
                "pase; %.1f saltadas (tight_barriers%s)\n", prof_full_barriers_ / double(period),
                prof_barriers_redundant_ / double(period), prof_barriers_after_pass_ / double(period),
                prof_barriers_skipped_ / double(period), pass_end_is_barrier_ ? " y _pase" : "");
        prof_full_barriers_ = prof_barriers_redundant_ = prof_barriers_after_pass_ = prof_barriers_skipped_ = 0;
        NVN_LOG("perfil merged_passes (por frame): %.1f pases con adjuntos de más, %.1f cambios de targets sin cortar%s\n",
                prof_passes_merged_ / double(period), prof_rt_merged_ / double(period), merge_passes_ ? "" : " (apagado)");
        prof_passes_merged_ = prof_rt_merged_ = 0;
        NVN_LOG("perfil cortes por motivo (por frame, mismo framebuffer + cabían juntos): %.1f + %.1f:%s\n",
                prof_passes_same_ / double(period), prof_passes_fit_ / double(period), list.c_str());
        prof_cuts_.clear();
        prof_passes_same_ = prof_passes_fit_ = 0;
    }
    {  // la escena por tamaño de la profundidad: si la resolución dinámica baja, baja el ancho mínimo del viewport
        std::vector<std::pair<uint64_t, uint32_t>> top;
        for (const auto& [key, d] : prof_depth_rts_) top.push_back({d.draws, key});
        std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::string list;
        for (size_t k = 0; k < std::min<size_t>(top.size(), 4); k++) {
            const DepthStat& d = prof_depth_rts_[top[k].second];
            char b[96];
            snprintf(b, sizeof b, " %ux%u: %.0f draws, viewport de %d a %d;", top[k].second >> 16, top[k].second & 0xFFFF,
                     d.draws / double(period), d.vp_min, d.vp_max);
            list += b;
        }
        NVN_LOG("perfil escena (por tamaño de la profundidad, por frame):%s\n", list.c_str());
        prof_depth_rts_.clear();
    }
    prof_clear_loads_ = prof_clear_flushes_ = prof_store_none_ = prof_area_cut_ = 0;
    prof_barriers_deferred_ = prof_barrier_splits_ = prof_rt_kept_ = 0;
    prof_run_us_ = prof_gpu_wait_us_ = prof_submits_ = 0;
    prof_draws_ = draws_;
}

void Executor::dump_frame(Texture* presented) {
    std::lock_guard lock(g_mutex);
    if (frame_ == trace_frame_ && ready_) {
        std::string f = std::to_string(frame_);
        if (presented) dump(*presented, "f" + f + "_ventana");
        finish(true);  // DEPURACIÓN: la GPU ha escrito ya las listas compactadas
        for (size_t k = 0; k < traced_gpu_draws_.size() && k < 120; k++) {
            const TracedGpuDraw& g = traced_gpu_draws_[k];
            const auto* args = reinterpret_cast<const uint32_t*>(g.args);
            const uint32_t count = args[0];
            auto idx = [&](const uint8_t* p, uint32_t i) {
                return g.type == 2 ? reinterpret_cast<const uint32_t*>(p)[i] : reinterpret_cast<const uint16_t*>(p)[i];
            };
            uint32_t mx = 0, outside = 0;
            auto it = prepass_args_.find(g.args_address);
            uint32_t orig_max = 0;
            if (it != prepass_args_.end())
                for (uint32_t i = 0; i < it->second.count; i++)
                    orig_max = std::max(orig_max, idx(reinterpret_cast<const uint8_t*>(it->second.indices), i));
            for (uint32_t i = 0; i < std::min<uint32_t>(count, 1 << 20); i++) {
                const uint32_t v = idx(g.indices, i);
                mx = std::max(mx, v);
                if (it != prepass_args_.end() && v > orig_max) outside++;
            }
            std::string first;
            for (uint32_t i = 0; i < std::min<uint32_t>(count, 9); i++) first += " " + std::to_string(idx(g.indices, i));
            NVN_LOG("F compactado %zu: %u índices (de %u originales, máx original %u), máx %u, fuera de rango %u:%s\n", k, count,
                    it != prepass_args_.end() ? it->second.count : 0, orig_max, mx, outside, first.c_str());
        }
        traced_gpu_draws_.clear();
        prepass_args_.clear();
        // DEPURACIÓN: solo la imagen final y el log, salvo con <datos>/full_trace
        const char* light = NVN_FLAG("full_trace") ? nullptr : getenv("NIER_TRAZA_LIGERA");
        for (size_t k = 0; k < frame_targets_.size() && k < 32 && (full_trace_ || !(light && *light)); k++)
            dump(*frame_targets_[k], "f" + f + "_rt" + std::to_string(k));
    }
    if (const int asked = g_trace_request.exchange(0, std::memory_order_relaxed)) {  // request_trace (detector)
        full_trace_ = asked == 2;
        trace_frame_ = frame_ + 1;
        traced_draws_ = 0;
        memset(trace_counts_, 0, sizeof trace_counts_);
        NVN_LOG("traza del frame %llu (pedida por el detector)\n", static_cast<unsigned long long>(trace_frame_));
    }
    if (frame_ % 30 == 0) {  // ¿piden una traza?
        const char* data = getenv("NIER_DATA");
        std::string flag = std::string(data ? data : "/sdcard/NieR") + "/trazar";
        const bool lequal = access((std::string(data ? data : "/sdcard/NieR") + "/profundidad_lequal").c_str(), F_OK) == 0;
        if (lequal != depth_lequal_) NVN_LOG("depth EQUAL como LEQUAL: %s\n", lequal ? "sí" : "no");
        depth_lequal_ = lequal;
        if (access(flag.c_str(), F_OK) == 0) {
            // "completa" dentro del archivo: también vuelca los render targets del frame (si no, solo log e imagen).
            char mode[16] = {};
            if (FILE* f = fopen(flag.c_str(), "r")) {
                fread(mode, 1, sizeof mode - 1, f);
                fclose(f);
            }
            unlink(flag.c_str());
            if (std::string_view(mode, strcspn(mode,"\r\n")) == "loading_iam") {
                loading_armed_ = true;
                NVN_LOG("armed loading illustration capture\n");
            } else {
                full_trace_ = strstr(mode, "completa") != nullptr;
                trace_frame_ = frame_ + 1;
                traced_draws_ = 0;
                memset(trace_counts_, 0, sizeof trace_counts_);
                NVN_LOG("traza del frame %llu\n", static_cast<unsigned long long>(trace_frame_));
            }
        }
    }
    if (frame_ == trace_frame_ + 1 && watch_addr_ && nnsdk::host().watch_writes && getenv("NIER_VIGILAR")) {
        nnsdk::host().watch_writes(watch_addr_);
        watch_addr_ = 0;
    }
    if (frame_ == trace_frame_) {
        std::string counts;
        if (full_trace_) NVN_LOG("face probe: frame %llu, matched %u, saved %u\n",
            static_cast<unsigned long long>(frame_), face_probe_matches_, face_probe_saved_);
        face_probe_matches_ = face_probe_saved_ = 0;
        for (size_t k = 0; k < std::size(trace_counts_); k++)
            if (trace_counts_[k]) counts += " " + std::to_string(k) + ":" + std::to_string(trace_counts_[k]);
        NVN_LOG("F comandos del frame (índice del variant:cuenta):%s; draws con render desactivado %llu\n", counts.c_str(),
                static_cast<unsigned long long>(prof_disabled_draws_));
    }
    // The log is written in blocks: after a capture, out to the file now, so a report saved from the menu has it.
    if (frame_ == trace_frame_ + 2) fflush(stdout);
    if (log_state_ && frame_ % 30 == 0 && !small_state_.empty()) log_small_state("cada 30 frames");
    frame_targets_.clear();
    frame_uploads_.clear();
    if (frame_ % 300 == 0)  // los args de frames viejos ya no sirven
        for (auto it = culled_draws_.begin(); it != culled_draws_.end();)
            it = frame_ - it->second.frame > 2 ? culled_draws_.erase(it) : std::next(it);
}

bool Executor::prepare_texel_readback(Texture& t) {
    begin();
    flush_clears(&t);
    if (__atomic_load_n(&t.mapping_changed, __ATOMIC_ACQUIRE) && !apply_mapping_changes(t)) return false;
    return apply_texel_writes(t);
}

bool Executor::publish_virtual_pool(void* key, Texture* only, Visibility visibility) {
    if (!virtual_pending_.load(std::memory_order_acquire)) return true;
    std::lock_guard lock(g_mutex);
    auto first = key ? virtual_writes_.find(key) : virtual_writes_.begin();
    if (first == virtual_writes_.end()) return true;
    const auto last = key ? std::next(first) : virtual_writes_.end();
    std::vector<TexelReadback> copies;
    std::vector<std::pair<void*, Texture*>> published;
    for (auto pool = first; pool != last; ++pool) {
        // gpu_dirty pins these mappings until the entire batch has reached guest memory.
        for (const auto& [t, dirty] : pool->second) {
            if (only && t != only) continue;
            if ((t->gpu_only_virtual || visibility == Visibility::Buffer) && visibility != Visibility::Storage) {
                const auto* backing = g_memory_pools.get(pool->first);
                if (backing) {
                    const uint64_t epoch = backing->buffer_epoch.load(std::memory_order_acquire);
                    if (epoch != t->buffer_epoch) {
                        t->buffer_overlap = backing->overlaps_buffer(t->desc.offset, t->layout.total_size());
                        t->buffer_epoch = epoch;
                    }
                }
                if (visibility == Visibility::Buffer && !t->buffer_overlap) continue;
                const bool raw = t->buffer_overlap || __atomic_load_n(&t->raw_alias, __ATOMIC_ACQUIRE) ||
                                 (backing && __atomic_load_n(&backing->mapped_by_cpu, __ATOMIC_ACQUIRE));
                if (!raw && (visibility == Visibility::Submission ||
                    (backing && !backing->has_cpu_backing(t->desc.offset, t->layout.total_size())))) continue;
            }
            published.emplace_back(pool->first, t);
            if (!prepare_texel_readback(*t)) { finish(); return false; }
            const uint32_t levels = upload_levels(*t);
            for (auto it = dirty.begin(); it != dirty.end();) {
                const auto [subresource, order] = *it++;
                const auto [mip, layer] = subresource;
                if (!t->owns_level(size_t{layer} * levels + mip)) continue;
                uint32_t count = 1;
                while (it != dirty.end() && it->second == order && it->first.first == mip &&
                       it->first.second == layer + count && t->owns_level(size_t{layer + count} * levels + mip)) {
                    ++it;
                    ++count;
                }
                const auto& lv = t->layout.levels[mip];
                copies.emplace_back(*t, Texture::TexelWrite{mip, layer, count, {0, 0, 0, lv.width_blocks, lv.height_blocks, lv.depth}});
                copies.back().order = order;
            }
        }
    }
    if (published.empty()) return true;
    // Different virtual images can cover the same physical bytes.
    std::stable_sort(copies.begin(), copies.end(), [](const auto& a, const auto& b) { return a.order < b.order; });
    if (!readback_texels(copies)) return false;
    finish(); // preparation can release the last image-owned mip
    for (const auto& [pool_key, texture] : published) {
        auto it = virtual_writes_.find(pool_key);
        it->second.erase(texture);
        if (!it->second.empty()) continue;
        if (auto* pool = g_memory_pools.get(pool_key)) {
            std::lock_guard mapping(pool->mapping_mutex);
            pool->gpu_dirty = false;
        }
        virtual_writes_.erase(it);
    }
    virtual_pending_.store(!virtual_writes_.empty(), std::memory_order_release);
    return true;
}

void Executor::TexelReadback::record(VkCommandBuffer cmd, const Staging::Alloc& allocation) {
    data = allocation.data;
    const auto& t = *texture;
    const auto& lv = t.layout.levels[region.level];
    const auto& box = region.box;
    const uint32_t x = box.x * t.layout.block_w, y = box.y * t.layout.block_h;
    VkBufferImageCopy copy{};
    copy.bufferOffset = allocation.offset;
    copy.imageSubresource = {t.aspect, region.level, region.layer, region.layers};
    copy.imageOffset = {int32_t(x), int32_t(y), int32_t(box.z)};
    copy.imageExtent = {std::min(box.width * t.layout.block_w, lv.width - x),
                        std::min(box.height * t.layout.block_h, lv.height - y), box.depth};
    vkCmdCopyImageToBuffer(cmd, t.image, VK_IMAGE_LAYOUT_GENERAL, allocation.buffer, 1, &copy);
}

void Executor::TexelReadback::publish() const {
    const auto& t = *texture;
    const auto& lv = t.layout.levels[region.level];
    const auto& box = region.box;
    for (uint32_t k = 0; k < region.layers; ++k) {
        auto* dst = t.storage + size_t{region.layer + k} * t.layout.layer_size + lv.offset;
        const auto* src = data + k * layer_bytes;
        if (t.layout.linear) {
            for (uint32_t j = 0; j < box.height; ++j)
                memcpy(dst + size_t{box.y + j} * t.layout.stride + size_t{box.x} * t.layout.bytes_per_block, src + j * row, row);
        } else {
            swizzle_region(lv, t.layout.bytes_per_block, box, src, row, slice, dst);
        }
    }
}

bool Executor::readback_texels(std::vector<TexelReadback>& copies) {
    size_t first = 0;
    while (first < copies.size()) {
        begin();
        end_render_pass();
        auto& slot = inflight_[cur_];
        size_t stop = first, bytes = 0;
        for (; stop < copies.size(); ++stop) {
            auto& copy = copies[stop];
            const size_t size = copy.layer_bytes * copy.region.layers;
            if (stop != first && bytes + size > kStagingSoftCap) break;
            const auto a = slot.texel_readback.alloc_contiguous(size, 4 * copy.texture->layout.bytes_per_block);
            if (!a.data) {
                if (stop != first) break; // submit this prefix and retry the next region in a fresh slot
                NVN_LOG("cannot allocate staging for %zu bytes of GPU texels\n", size);
                finish();
                return false;
            }
            if (stop == first) {
                note_gpu_work();
                sync_memory(cmd_, prior_writes, transfer_read);
            }
            copy.record(cmd_, a);
            bytes += size;
        }
        sync_memory(cmd_, transfer_write, {VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT});
        const auto previous = submitted_.load();
        finish();
        if (submitted_.load() == previous || vkWaitForFences(vk()->device, 1, &slot.fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) {
            NVN_LOG("GPU texture readback did not complete\n");
            return false;
        }
        for (; first < stop; ++first) copies[first].publish();
        // The fence and host copies are complete. Other slots can reuse these cached blocks.
        slot.texel_readback.reset();
        slot.texel_readback.trim(0);
    }
    return true;
}

bool Executor::sync_texture_storage(Texture& t, const Texture::TexelWrite& r) {
    std::lock_guard lock(g_mutex);
    const uint32_t levels = upload_levels(t);
    auto owned = [&](uint32_t layer) { return t.owns_level(size_t{layer} * levels + r.level); };
    uint32_t start = r.layer;
    const uint32_t end = r.layer + r.layers;
    while (start < end && !owned(start)) ++start;
    if (start == end) return true;
    const FormatInfo* f = format_info(t.desc.format);
    if (!t.image || !f || t.vk_format != f->vk || t.desc.samples > 1 || !t.aspect ||
        (t.aspect & (t.aspect - 1)) || t.layout.bytes_per_block != f->bytes) {
        NVN_LOG("cannot read GPU texels in guest format %#x (%d samples, aspects %#x)\n", t.desc.format, t.desc.samples, t.aspect);
        return false;
    }
    const auto* pool = g_memory_pools.get(t.desc.pool);
    std::unique_lock<std::mutex> mapping;
    for (;;) {
        if (!prepare_texel_readback(t)) { finish(); return false; }
        if (!pool || !pool->reserved) break;
        mapping = std::unique_lock(pool->mapping_mutex);
        if (!__atomic_load_n(&t.mapping_changed, __ATOMIC_ACQUIRE)) break;
        mapping.unlock();
    }
    std::vector<TexelReadback> copies;
    while (start < end) {
        if (!owned(start)) { ++start; continue; }
        uint32_t stop = start + 1;
        while (stop < end && owned(stop)) ++stop;
        copies.emplace_back(t, Texture::TexelWrite{r.level, start, stop - start, r.box});
        start = stop;
    }
    const bool ok = readback_texels(copies);
    finish(); // preparation may have released all of the requested layers
    return ok;
}

void Executor::return_linear_textures() {
    struct Copy {
        Texture* t;
        Staging::Alloc a;
        uint32_t row;
    };
    std::vector<Copy> copies;
    begin();
    for (Texture* t : linear_written_) {
        const FormatInfo* f = format_info(t->desc.format);
        if (!t->image || !t->storage || !f || f->block != 1 || f->bytes != t->layout.bytes_per_block || t->bcn || t->astc ||
            !(t->aspect & VK_IMAGE_ASPECT_COLOR_BIT) || t->layout.levels.empty() || t->desc.samples > 1)
            continue;
        const LevelLayout& l = t->layout.levels[0];
        const uint32_t row = l.width * f->bytes;
        if (t->layout.stride < row || VkDeviceSize{row} * l.height > kStagingChunk) continue;
        flush_clears(t);
        if (!t->owns_level(0) || !prepare_contents(*t, PriorContents::Preserve) || !t->owns_level(0)) continue;
        Staging::Alloc a = staging_->alloc(VkDeviceSize{row} * l.height, 16);
        if (!a.data) continue;
        copies.push_back({t, a, row});
    }
    linear_written_.clear();
    if (copies.empty()) {
        finish();
        return;
    }
    barrier();
    for (const Copy& c : copies) {
        VkBufferImageCopy r{};
        r.bufferOffset = c.a.offset;
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.imageExtent = {c.t->layout.levels[0].width, c.t->layout.levels[0].height, 1};
        note_gpu_work();
        vkCmdCopyImageToBuffer(cmd_, c.t->image, VK_IMAGE_LAYOUT_GENERAL, c.a.buffer, 1, &r);
    }
    finish(true);  // (the CPU reads it once its sync is signalled: it has to be there by then)
    for (const Copy& c : copies) {
        Texture& t = *c.t;
        std::unique_lock<std::mutex> mapping;
        if (const auto* pool = g_memory_pools.get(t.desc.pool); pool && pool->reserved) {
            mapping = std::unique_lock(pool->mapping_mutex);
            if (__atomic_load_n(&t.mapping_changed, __ATOMIC_ACQUIRE)) {
                linear_written_.push_back(&t);
                continue;
            }
        }
        const LevelLayout& l = t.layout.levels[0];
        uint8_t* dst = t.storage + l.offset;
        nnsdk::watch::before_kernel_write(dst, size_t{t.layout.stride} * l.height);
        for (uint32_t y = 0; y < l.height; y++) memcpy(dst + size_t{y} * t.layout.stride, c.a.data + size_t{y} * c.row, c.row);
        if (++t.returned == kReturnLinearMax)
            NVN_LOG("linear texture %dx%d went back to its memory %u times: no more (it isn't for the CPU)\n", t.desc.width,
                    t.desc.height, t.returned);
    }
}

// What holds a range of the game's memory on our side: a GPU buffer (overlay) and whether it goes back to the memory,
// or a texture's storage. For the album (album.cpp): a photo that comes out black says from where it was read.
std::string Executor::describe_memory(uint64_t address, uint64_t size) {
    std::lock_guard lock(g_mutex);
    std::string out;
    char line[384];
    for (auto it = overlays_.upper_bound(address + size); it != overlays_.begin();) {
        --it;
        const Overlay& o = it->second;
        if (it->first + o.size <= address) break;
        snprintf(line, sizeof line, " overlay %#llx+%llu (writers %u, written in submit %llu, cpu_reads %d, returned %llu)",
                 static_cast<unsigned long long>(it->first), static_cast<unsigned long long>(o.size), o.writers,
                 static_cast<unsigned long long>(o.written), o.cpu_reads, static_cast<unsigned long long>(o.written_back));
        out += line;
    }
    const auto* p = reinterpret_cast<const uint8_t*>(address);
    g_textures.for_each([&](Texture& t) {
        // And what the GPU wrote lately at that size or smaller, linear or not: where a photo could be read from
        if (t.storage && t.gpu_write_frame + 300 > frame_ && t.desc.width >= 174 && t.desc.width <= 1280 &&
            t.desc.height >= 174 && t.desc.height <= 960 && (t.desc.width == 1280 || t.desc.width == t.desc.height)) {
            snprintf(line, sizeof line, "\n  lately written by the GPU: %dx%d format 0x%x %s at %p (frame %llu of %llu, back to "
                     "memory %u times)", t.desc.width, t.desc.height, t.desc.format, t.layout.linear ? "linear" : "block-linear",
                     static_cast<void*>(t.storage), static_cast<unsigned long long>(t.gpu_write_frame),
                     static_cast<unsigned long long>(frame_), t.returned);
            out += line;
        }
        if (t.storage && p < t.storage + t.layout.total_size() && t.storage < p + size) {
            snprintf(line, sizeof line, " texture %dx%d format 0x%x %s%s at +%lld", t.desc.width, t.desc.height,
                     t.desc.format, t.layout.linear ? "linear" : "block-linear", t.gpu_levels ? ", written by the GPU" : "",
                     static_cast<long long>(p - t.storage));
            out += line;
        }
    });
    return out.empty() ? " nothing of ours (only the CPU's memory)" : out;
}

Executor g_executor;

}  // namespace

std::string describe_memory(const void* p, size_t size) {
    return g_executor.describe_memory(reinterpret_cast<uint64_t>(p), size);
}

DropStats g_drop_stats;
void prefetch_compute_pipeline(const ProgramStage& stage) { g_executor.prefetch_compute(stage); }

bool game_gpu_frame(uint64_t& serial, int64_t& ns) {
    const uint64_t now = g_game_gpu_serial.load(std::memory_order_acquire);
    if (now == serial) return false;
    serial = now;
    ns = g_executor.game_gpu_ns();
    return true;
}
uint64_t last_frame_run_ns() { return g_last_frame_run_ns.load(std::memory_order_relaxed); }
void request_trace(bool full) { g_trace_request.store(full ? 2 : 1, std::memory_order_relaxed); }

namespace {

// --- Envío asíncrono ---
// nvnQueueSubmitCommands solo encola: un hilo reproduce los comandos en orden mientras el juego graba el frame
// siguiente (antes el hilo del juego que envía pasaba ~20 ms por frame dentro de la llamada, y MainDraw
// esperaba). Cada envío del juego tiene un ticket: un NVNsync recuerda uno, y esperarlo es esperar a que se
// haya reproducido y a que la GPU haya terminado su último envío de Vulkan. Presentar también pasa por la cola
// (en orden), con como mucho un frame por delante. DEPURACIÓN: con <datos>/sincrono, como antes (sin hilo).
class SubmitQueue {
public:
    uint64_t submit(const std::vector<const CommandList*>& lists, void* host_sync = nullptr) {
        std::unique_lock lock(m_);
        const uint64_t ticket = ++queued_;
        if (host_sync) sync_signal_at(host_sync, ticket);
        for (const CommandList* list : lists)  // los NVNsync de dentro de las listas: este envío
            for (const Command& c : list->commands)
                if (const auto* f = std::get_if<cmd::FenceSync>(&c)) sync_signal_at(f->sync, ticket);
        if (!async()) {
            drain_locked(lock);
            lock.unlock();
            run(lists, ticket, host_sync);
            return ticket;
        }
        start_locked();
        jobs_.push_back({lists, nullptr, 0, ticket, host_sync});
        work_cv_.notify_one();
        return ticket;
    }
    void present(Window& w, int index) {
        std::unique_lock lock(m_);
        if (!async()) {
            drain_locked(lock);
            lock.unlock();
            nvn::present(w, index);
            return;
        }
        start_locked();
        done_cv_.wait(lock, [&] { return presents_pending_ < 2; });  // dos frames por delante como mucho
        presents_pending_++;
        jobs_.push_back({{}, &w, index, 0});
        work_cv_.notify_one();
    }
    uint64_t last_ticket() {
        std::lock_guard lock(m_);
        return queued_;
    }
    // true si el envío ya se reprodujo y la GPU lo terminó (o termina antes del plazo).
    bool wait(uint64_t ticket, uint64_t timeout_ns) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds(std::min<uint64_t>(timeout_ns, 1ull << 62));
        uint64_t gpu = 0;
        {
            std::unique_lock lock(m_);
            if (!done_cv_.wait_until(lock, deadline, [&] { return done_ >= ticket; })) return false;
            auto it = gpu_submit_.lower_bound(ticket);  // los viejos se olvidan: uno posterior también vale
            gpu = it != gpu_submit_.end() ? it->second : gpu_submit_.empty() ? 0 : gpu_submit_.rbegin()->second;
        }
        if (!gpu) return true;
        const auto now = std::chrono::steady_clock::now();
        const uint64_t left = now >= deadline ? 0 : std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now).count();
        return g_executor.wait_submit(gpu, left);
    }
    void drain() {
        std::unique_lock lock(m_);
        drain_locked(lock);
    }
    // Una lista que el juego recicla puede estar aún en la cola: se borra cuando se haya reproducido.
    void release(CommandList* list) {
        std::lock_guard lock(m_);
        if (done_ >= queued_ && !busy_) {
            recycle_command_list(list);
            return;
        }
        garbage_.push_back({queued_, list});
    }

private:
    struct Job {
        std::vector<const CommandList*> lists;
        Window* window;
        int index;
        uint64_t ticket;
        void* host_sync = nullptr;
    };
    bool async() {
        if (++checks_ % 64 == 1) {  // mirar el fichero de vez en cuando, no en cada envío
            sync_mode_ = nnsdk::ini::on("sincrono");
        }
        return !sync_mode_;
    }
    void start_locked() {
        if (started_) return;
        started_ = true;
        std::thread([this] { loop(); }).detach();  // vive lo que el proceso: sin join (al salir no hay terminate)
    }
    void drain_locked(std::unique_lock<std::mutex>& lock) {
        done_cv_.wait(lock, [&] { return jobs_.empty() && !busy_; });
    }
    void run(const std::vector<const CommandList*>& lists, uint64_t ticket, void* host_sync) {
        const bool ok = host_sync ? g_executor.publish_virtual_pool(nullptr, nullptr, Executor::Visibility::CpuFence)
                                  : g_executor.run(lists);
        if (!ok) {
            if (host_sync) sync_fail_at(host_sync, ticket);
            for (const auto* list : lists)
                for (const auto& c : list->commands)
                    if (const auto* f = std::get_if<cmd::FenceSync>(&c)) sync_fail_at(f->sync, ticket);
        }
        const uint64_t gpu = g_executor.last_submit();
        std::lock_guard lock(m_);
        gpu_submit_[ticket] = gpu;
        while (gpu_submit_.size() > 256) gpu_submit_.erase(gpu_submit_.begin());
        done_ = std::max(done_, ticket);
        collect_garbage_locked();
        done_cv_.notify_all();
    }
    void collect_garbage_locked() {
        while (!garbage_.empty() && garbage_.front().first <= done_) {
            recycle_command_list(garbage_.front().second);  // al pool de cmdbuf.cpp: no se vacía ni libera aquí
            garbage_.pop_front();
        }
    }
    void loop() {
        pthread_setname_np(pthread_self(), "NvnSubmit");
        pin_critical_thread();  // núcleos grandes (os.cpp)
        int nice = 0;
        std::unique_lock lock(m_);
        for (;;) {
            work_cv_.wait(lock, [&] { return !jobs_.empty(); });
            Job job = std::move(jobs_.front());
            jobs_.pop_front();
            busy_ = true;
            lock.unlock();
            if (const int want = submit_nice(); want != nice) {
                nice = want;
                set_thread_nice(nice, "NvnSubmit");
            }
            if (job.window) {
                nvn::present(*job.window, job.index);
                lock.lock();
                presents_pending_--;
            } else {
                run(job.lists, job.ticket, job.host_sync);
                lock.lock();
            }
            busy_ = false;
            collect_garbage_locked();
            done_cv_.notify_all();
        }
    }
    std::mutex m_;
    std::condition_variable work_cv_, done_cv_;
    std::deque<Job> jobs_;
    std::deque<std::pair<uint64_t, CommandList*>> garbage_;  // (último ticket encolado al soltarla, lista)
    std::map<uint64_t, uint64_t> gpu_submit_;                 // ticket -> envío de Vulkan
    uint64_t queued_ = 0, done_ = 0, checks_ = 0;
    int presents_pending_ = 0;
    bool busy_ = false, sync_mode_ = false, started_ = false;
};
SubmitQueue g_submit_queue;

}  // namespace

void execute(const std::vector<const CommandList*>& lists) { g_submit_queue.submit(lists); }
void queue_present(Window& w, int index) { g_submit_queue.present(w, index); }
void release_command_list(CommandList* list) { g_submit_queue.release(list); }

void next_frame() {
    g_drop_stats.run_ns.fetch_add(g_executor.frame_run_ns_, std::memory_order_relaxed);
    note_max(g_drop_stats.run_max_ns, g_executor.frame_run_ns_);
    g_last_frame_run_ns.store(g_executor.frame_run_ns_, std::memory_order_relaxed);
    g_executor.frame_run_ns_ = 0;
    g_executor.frame_++;
    g_executor.profile_frame();
}

void debug_dump_frame(Texture* presented) { g_executor.dump_frame(presented); }
void capture_loading_present(ResourceRef<Texture> presented, const VkRect2D& crop) { g_executor.finish_loading_capture(presented,crop); }

uint64_t last_submit() { return g_submit_queue.last_ticket(); }
std::recursive_mutex& executor_mutex() { return g_mutex; }
bool publish_virtual_pool(void* pool, Texture* only) { return g_executor.publish_virtual_pool(pool, only); }
bool publish_virtual_buffers(void* pool) {
    return g_executor.publish_virtual_pool(pool, nullptr, Executor::Visibility::Buffer);
}
void queue_cpu_fence(void* sync) { g_submit_queue.submit({}, sync); }
void note_executor_lock_wait(uint64_t ns) {
    g_executor_lock_waits.fetch_add(1, std::memory_order_relaxed);
    g_executor_lock_wait_ns.fetch_add(ns, std::memory_order_relaxed);
}
void retire_texture_gpu(Texture& t) { g_executor.retire_texture(t); }
bool sync_texture_storage(Texture& t, const Texture::TexelWrite& region) {
    g_submit_queue.drain();
    return g_executor.sync_texture_storage(t, region);
}
void retire_sampler(VkSampler sampler) { g_executor.retire_sampler(sampler); }
bool wait_submit(uint64_t submit, uint64_t timeout_ns) { return g_submit_queue.wait(submit, timeout_ns); }
void wait_idle() {
    g_submit_queue.drain();
    g_executor.wait_idle();
}

PicturePath blit_to_swapchain(Texture& texture, const VkRect2D& crop, VkImage image, VkFormat format, VkExtent2D extent,
                       VkCommandBuffer cmd) {
    return g_executor.blit(texture, crop, image, format, extent, cmd);
}

std::mutex& vk_queue_mutex() { return g_queue_mutex; }

}  // namespace nvn
