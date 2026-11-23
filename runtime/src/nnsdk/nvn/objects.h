#pragma once
#include <atomic>
// Objetos NVN inicializados que comparten varios archivos (pools, texturas, samplers, programas,
// ventana). Los objetos del juego son opacos: se asocian por dirección a estos.

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <map>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "nnsdk/nvn/gpu_memory.h"
#include "nnsdk/nvn/layout.h"
#include "nnsdk/nvn/nvn.h"
#include "nnsdk/nvn/present_image.h"
#include "nnsdk/nvn/resource_ref.h"
#include "nnsdk/nvn/spirv_cache.h"
#include "nnsdk/nvn/vk.h"
#include "nvn_shader.h"

namespace nvn {

constexpr int kStages = static_cast<int>(nvnshader::Stage::Compute) + 1;

struct MemoryPool {
    uint8_t* storage = nullptr;  // memoria del juego; también es la "dirección de GPU"
    size_t size = 0;
    int flags = 0;
    // Pool virtual (memory.cpp): sin memoria propia; storage es una reserva nuestra donde MapVirtual hace aparecer las
    // páginas de pools físicos (lo no mapeado se lee como ceros).
    bool reserved = false;
    bool mapped_by_cpu = false; // atomically set by an explicit CPU mapping request
    mutable std::mutex mapping_mutex;
    std::vector<bool> backed_pages;
    std::vector<bool> cpu_readable_pages;
    std::multimap<size_t, size_t> buffer_ranges; // offset, size of live raw buffers
    size_t max_buffer_size = 0;
    std::atomic<uint64_t> buffer_epoch{1};
    void register_buffer(size_t offset, size_t length);
    void unregister_buffer(size_t offset, size_t length);
    bool overlaps_buffer(size_t offset, size_t length) const;
    bool gpu_dirty = false; // a virtual texture still needs to publish into these physical pages
    bool covers(size_t offset, size_t bytes) const;
    bool has_cpu_backing(size_t offset, size_t bytes) const;
};
// NVNmemoryPoolFlags que cambian lo que hacemos (el resto son de caché y acceso).
constexpr int kPoolFlagPhysical = 0x100, kPoolFlagVirtual = 0x200;
constexpr int kPoolFlagCpuNoAccess = 0x1;

// NVNstorageClass is passed from the builder to the virtual allocator. The native
// class identifies storage owned by a VkImage; raw requests retain the guest layout.
constexpr int kStorageClassPitch = 0, kStorageClassNativeImage = 0xfd, kStorageClassBlockLinear = 0xfe;

// Estado de un NVNtextureBuilder; vive dentro del builder del juego (≤ 96 bytes).
struct TextureDesc {
    void* device;
    int target, flags, format;
    int width, height, depth, levels, samples;
    int stride;
    void* pool;
    int64_t offset;
    uint8_t depth_stencil_mode;  // NVNtextureDepthStencilMode: DEPTH 0, STENCIL 1 (lo que se muestrea)
    uint8_t swizzle[4];          // NVNtextureSwizzle (ZERO 0, ONE 1, R 2 … A 5); SetDefaults: R G B A
    // SetPackagedTextureLayout: la disposición con la que el empaquetador de texturas dejó los datos (8 bytes de NVN).
    uint8_t has_packaged_layout;
    uint8_t packaged_layout[8];
    bool operator==(const TextureDesc&) const = default;
};
static_assert(sizeof(TextureDesc) <= 96);

constexpr int kTextureFlagDisplay = 0x1, kTextureFlagImage = 0x4, kTextureFlagCompressible = 0x8, kTextureFlagLinear = 0x10;
constexpr int kTarget1D = 0, kTarget2D = 1, kTarget3D = 2, kTarget1DArray = 3, kTarget2DArray = 4,
              kTarget2DMultisample = 5, kTarget2DMultisampleArray = 6, kTargetRectangle = 7, kTargetCube = 8,
              kTargetCubeArray = 9, kTargetBuffer = 10;

// NVNtextureView: vive dentro del struct del juego (≤ 40 bytes). flags dice qué campos valen.
struct TextureViewDesc {
    uint16_t flags;
    uint8_t depth_stencil_mode, pad;
    int16_t base_level, levels;
    int32_t base_layer, layers;
    int32_t format, target;
    uint8_t swizzle[4];
};
static_assert(sizeof(TextureViewDesc) <= 40);
constexpr uint16_t kViewLevels = 1, kViewLayers = 2, kViewFormat = 4, kViewTarget = 8, kViewDepthStencil = 16,
                   kViewSwizzle = 32;

struct MipResidency {
    enum Issue : uint8_t { Unmapped = 1, InvalidAstc = 2, Encoding = 4 };
    uint32_t levels = 0, revision = 0;
    std::vector<uint8_t> issues; // layer * levels + mip; empty when every subresource is ready
    void publish(uint32_t count, std::vector<uint8_t> next);
    void encoded(uint32_t layer, uint32_t mip);
    void restrict_sample(VkImageSubresourceRange& range) const;
};

struct Texture {
    ResourceRef<Texture> ref;
    size_t handle_count = 0; // protected by TextureRegistry
    // Único en toda la ejecución (nunca se reutiliza, al contrario que la dirección del objeto): clave de
    // las cachés de vistas. Cambia al reinicializar o finalizar la textura.
    uint64_t serial = 0;
    TextureDesc desc{};
    TextureLayout layout;
    uint8_t* storage = nullptr;  // memoria del juego donde están sus texels (si los sube la CPU)
    bool virtual_storage = false;
    bool gpu_only_virtual = false;
    bool raw_alias = false; // atomic after registration; another resource may use the raw storage
    uint64_t buffer_epoch = 0;
    bool buffer_overlap = false;
    // Vulkan, creado cuando hace falta:
    VkImage image = VK_NULL_HANDLE;
    GpuAllocation memory;  // dentro de un bloque compartido o propia (gpu_memory.h)
    uint64_t gpu_bytes = 0;
    VkFormat vk_format = VK_FORMAT_UNDEFINED;
    uint8_t bcn = 0;  // bcn::Kind: la GPU no tiene ese BC, se guarda descomprimida (bcn.h); 0, como viene
    bool astc = false;  // with bcn: stored as ASTC 4x4 instead of RGBA8 (astc.h)
    VkImageAspectFlags aspect = 0;
    uint32_t layers = 1;
    bool vk_failed = false;    // no se pudo crear en Vulkan (formato sin soporte…): no reintentar
    bool mutable_format = true;  // imagen con MUTABLE_FORMAT: admite vistas con otro formato (ejecutor)
    uint64_t retry_frame = 0;  // sin memoria al crearla: no se reintenta antes de este frame
    // Layer-major mip slots whose Vulkan image may contain newer bytes than guest memory.
    // Single-subresource images use only the count.
    enum class LevelOwner : uint8_t { Guest, Image, Remapped };
    uint32_t gpu_levels = 0;
    std::vector<LevelOwner> level_owners;
    bool owns_level(size_t slot) const { return gpu_levels && (level_owners.empty() || level_owners[slot] == LevelOwner::Image); }
    void set_level_owner(size_t slot, size_t slots, bool gpu) {
        if (owns_level(slot) == gpu) return;
        if (slots > 1) {
            if (level_owners.empty()) level_owners.resize(slots);
            level_owners[slot] = gpu ? LevelOwner::Image : LevelOwner::Remapped;
        } else if (!gpu) level_hash_image = VK_NULL_HANDLE;
        if (gpu) ++gpu_levels;
        else --gpu_levels;
    }
    // Último frame del ejecutor en que fue render target, y si también lo fue en uno de los dos anteriores (se redibuja
    // cada frame: un draw que no se dibuje uno de ellos sale en el siguiente; ver las pipelines en segundo plano).
    uint64_t rt_frame = 0;
    uint64_t gpu_write_frame = 0;  // last executor frame the GPU wrote it (any way)
    uint64_t content_version = 1;  // image writes invalidate a sampled attachment snapshot
    uint32_t returned = 0;         // times its level 0 went back to its memory (linear: the CPU reads it, executor.cpp)
    bool rt_recurring = false;
    bool storage_image = false;  // la imagen se creó con STORAGE: se puede enlazar como imagen (ejecutor)
    bool uploaded = false;
    bool mapping_changed = false;  // Atomic access after registration: MapVirtual can run on the streaming thread.
    uint64_t content_hash = 0;   // de su memoria al subirla: si cambia, el juego la ha reescrito
    uint64_t checked_frame = ~uint64_t{0};
    float clear_color[4] = {};  // último ClearColor (para presentar antes de tener render real)
    bool cleared = false;
    // Last sampled descriptor for this allocation and resident mip.
    VkImageView view_memo = VK_NULL_HANDLE;
    uint32_t view_memo_revision = 0;
    // Outstanding BC-to-ASTC jobs keyed by uploaded subresource and source bytes.
    std::vector<uint64_t> astc_pending;
    // Unwatched changing textures are checked in full; static ones are audited on staggered frames.
    uint64_t full_check_until = 0;
    uint64_t next_full_check = 0;
    MipResidency residency;
    VkImage level_hash_image = VK_NULL_HANDLE;  // la imagen a la que corresponden (otra imagen: se sube entera)
    // Full digests of the bytes submitted for each layer and mip. Valid for level_hash_image only.
    std::vector<uint64_t> level_full_hash;
    // Vigilancia de escrituras de la CPU en su memoria (nnsdk/watch.h, como los espejos): las que sube la CPU sin
    // comprimir (el atlas de glifos) se revisan por páginas en cada envío en vez de por la huella muestreada una vez por
    // frame. Una clave por página de 4 KB; el envío de la última revisión.
    bool watched = false;
    // No se vigila: recibe WriteTexels/CopyBufferToTexture (sus cajas ya dicen qué cambia) o la CPU la reescribe entera una
    // y otra vez (vídeo: saltaría en cada página de cada frame; la huella muestreada ya lo ve).
    bool no_watch = false;
    uint8_t watch_rewrites = 0;  // cambios seguidos con más de media textura escrita
    std::vector<uint64_t> watch_keys;
    uint64_t watch_submit = ~uint64_t{0};
    uint64_t view_memo_serial = 0;
    TextureViewDesc view_memo_desc{};
    // Cajas que la CPU escribió con nvnTextureWriteTexels (texture.cpp) y que la imagen aún no tiene: el ejecutor las
    // sube al usar la textura. Con texel_writes_mutex(); g_texel_writes cuenta las pendientes de todas las texturas.
    struct TexelWrite {
        uint32_t level, layer, layers;
        BlockBox box;
    };
    std::vector<TexelWrite> texel_writes;
    using MappingRange = std::pair<size_t, size_t>;  // texture-relative [begin, end)
    std::vector<MappingRange> remapped_ranges;       // under the registry lock
    void note_mapping_change(size_t first, size_t last);
    struct MappedLevel {
        bool backed = false;
        std::vector<uint64_t> hashes;
    };
    std::vector<MappedLevel> mapped_levels;          // executor-owned, valid for level_hash_image
};

std::mutex& texel_writes_mutex();
extern std::atomic<uint32_t> g_texel_writes;
// Bumped whenever a texture's guest mapping changes (MapVirtual, any thread): the executor's bindless memo checks it
extern std::atomic<uint64_t> g_mapping_serial;

// Libera la imagen de Vulkan de una textura (texture.cpp); el objeto sigue siendo válido.
void release_texture_gpu(Texture& t);
uint64_t next_texture_serial();
// Memoria de GPU de las texturas vivas (DEPURACIÓN/perfil).
extern std::atomic<uint64_t> g_texture_gpu_bytes, g_texture_gpu_images;

struct SamplerDesc {  // NVNsamplerBuilder (≤ 96 bytes)
    void* device;
    uint8_t min_filter, mag_filter, wrap[3], compare_mode, compare_func, reduction;
    float max_anisotropy, lod_min, lod_max, lod_bias;
    float border[4];
};
static_assert(sizeof(SamplerDesc) <= 96);

struct Sampler {
    SamplerDesc desc{};
    VkSampler vk = VK_NULL_HANDLE;
    // El otro anisotrópico (ver Executor::sampler), en vivo. En los de comparación (sombras): el de antes, con el mínimo de
    // 4x forzado (<datos>/af_compare). En los demás: sin ese mínimo, el que pide el juego (<datos>/game_af).
    // Se retira con vk.
    VkSampler vk_af = VK_NULL_HANDLE;
    // Con las sombras apagadas (<datos>/no_shadows, en vivo; Executor::sampler): el de comparación que siempre pasa. Se
    // retira con vk.
    VkSampler vk_lit = VK_NULL_HANDLE;
};

struct TexturePoolEntry {
    ResourceRef<Texture> texture;
    TextureViewDesc view{};
};

struct DescriptorPool {
    void* memory_pool = nullptr;
    int64_t offset = 0;
    int count = 0;
    std::vector<TexturePoolEntry> textures;  // pool de texturas: id -> textura + vista
    std::vector<ResourceRef<Sampler>> samplers;
    // RegisterTexture may run on the resource thread while NvnSubmit resolves a handle.
    // A binding owns a copy of both the image identity and its registered view.
    TexturePoolEntry texture_at(uint32_t id) const {
        std::lock_guard lock(entries_mutex);
        return id < textures.size() ? textures[id] : TexturePoolEntry{};
    }
    void register_texture(uint32_t id, const TexturePoolEntry& entry) {
        std::lock_guard lock(entries_mutex);
        textures[id] = entry;
        texture_snapshot_.reset();
    }
    // One descriptor table per draw. Registrations publish a fresh snapshot on
    // the next read; each entry still checks the resource's current generation.
    auto texture_snapshot() const {
        std::lock_guard lock(entries_mutex);
        if (!texture_snapshot_) texture_snapshot_ = std::make_shared<const std::vector<TexturePoolEntry>>(textures);
        return texture_snapshot_;
    }
    auto sampler_snapshot() const {
        std::lock_guard lock(entries_mutex);
        if (!sampler_snapshot_) sampler_snapshot_ = std::make_shared<const std::vector<ResourceRef<Sampler>>>(samplers);
        return sampler_snapshot_;
    }
    Sampler* sampler_at(uint32_t id) const {
        std::lock_guard lock(entries_mutex);
        return id < samplers.size() ? samplers[id] : nullptr;
    }
    void register_sampler(uint32_t id, ResourceRef<Sampler> sampler);
    void register_sampler_builder(uint32_t id, const SamplerDesc& desc);
    void retire_sampler_builders();
private:
    mutable std::shared_ptr<const std::vector<TexturePoolEntry>> texture_snapshot_;
    mutable std::shared_ptr<const std::vector<ResourceRef<Sampler>>> sampler_snapshot_;
    struct BuilderSampler { Sampler value; ResourceRef<Sampler> ref; };
    std::map<uint32_t, BuilderSampler> builder_samplers;
    mutable std::mutex entries_mutex;
};

struct ProgramStage {
    // Geometry shader de la compactación de NieR (pre-pase de profundidad): escribe la lista de triángulos visibles y sus
    // argumentos indirectos, que ya no usamos (el pase de color dibuja la malla entera, ver culled_draws_).
    bool compaction_gs = false;
    uint64_t code = 0;  // dirección del código (NVN solo lo lee al dibujar: se traduce en el executor)
    uint64_t print = 0;  // huella del código (la de los logs), al traducirlo
    const void* control = nullptr;  // sección de control (memoria de CPU del juego)
    // Constantes del propio shader (su c1): el compilador las deja tras el código, en
    // code + control[0x700] hasta code + control[0x704].
    uint64_t constants = 0;
    uint32_t constants_size = 0;
    bool tried = false;
    nvnshader::Translated shader;
    VkShaderModule module = VK_NULL_HANDLE;
    // Compute de kUniformBarrierShaders (program.cpp), a los que se les dejan las barreras de sus bucles: la traducción
    // sin ellas, para sus interruptores en vivo (use_stripped_module). Los mismos recursos que module (el mismo layout).
    VkShaderModule stripped_module = VK_NULL_HANDLE;
    int8_t uniform_barrier_entry = -1;  // su índice en kUniformBarrierShaders (-1: no es de la lista)
    // Vertex: translated with the draw count check (count_check.h), for GPU-counted multi-draws on old drivers
    bool count_check = false;
    // Identidad de lo traducido (spirv_identity: el SPIR-V y lo que el layout toma de los recursos). Las pipelines y los
    // layouts del ejecutor van por ella y no por la dirección de la etapa: el juego crea varios programas con el mismo
    // código (MH Rise: 2200 etapas de 496 huellas en una partida) y cada uno llevaba sus propias pipelines. 0: sin traducir.
    uint64_t spirv_id = 0;
    // Clave con la que está en la caché de SPIR-V (program.cpp); a cero si no está (sin sección de control, caché
    // apagada). El ejecutor la apunta con cada pipeline nueva para crearla al arrancar la partida siguiente (precalentado).
    spirv_cache::Key cache_key;
    // Variantes: geometry por (genéricos que escribe el vertex shader, topología de entrada); pixel por los genéricos que
    // le llegan (los que no escribe la etapa anterior no se declaran); vertex por los tipos de sus atributos.
    struct Variant {
        VkShaderModule module = VK_NULL_HANDLE;
        spirv_cache::Key cache_key;
    };
    std::unordered_map<uint64_t, Variant> variants;
    // Traducciones "seguras" de la etapa (safe_variant): por (genéricos de la etapa anterior, nivel, tipos de atributos)
    std::map<std::tuple<uint64_t, int, uint64_t>, Variant> safe_variants;
};

struct Program {
    std::vector<ProgramStage> stages;
    uint64_t generation = 0;
};

struct Window {
    void* native_window = nullptr;
    int present_interval = 1;
    std::vector<ResourceRef<Texture>> textures;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkExtent2D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED;  // el del swapchain (el HUD lo necesita)
    std::vector<PresentImage> images;
    VkResult queue_image(uint32_t index, const VkPresentTimeGOOGLE* when);
    int next_texture = 0;
    uint64_t presented = 0;
    // nvnWindowSetCrop: la parte de la textura que va a la pantalla (ancho 0: entera). MH Rise dibuja a 1280x720 en
    // texturas de 1920x1080 y recorta.
    VkRect2D crop{};
    bool rebuild = false;
    bool capture_layers = false; // this swapchain was created with transfer source usage
};

extern Objects<MemoryPool> g_memory_pools;
// Each handle has a debug identity. Simultaneously live, identical descriptions of the same
// pool storage share the Vulkan image, deferred clears, views and pending texel writes.
class TextureRegistry {
public:
    Texture* get(const void* key);
    ResourceRef<Texture> reference(const void* key);
    std::shared_ptr<Texture> acquire(const void* key);
    uint64_t debug_id(const void* key);
    bool bind(const void* key, Texture texture);
    void unbind(const void* key);
    std::vector<Texture::MappingRange> take_mapping_changes(Texture& texture);
    void restore_mapping_changes(Texture& texture, const std::vector<Texture::MappingRange>& ranges);
    template <typename F> void for_each(F visit) {
        std::lock_guard lock(mutex_);
        for (auto& [storage, texture] : storage_) visit(*texture.lock());
    }
private:
    struct Handle { std::shared_ptr<Texture> texture; uint64_t id; };
    std::mutex mutex_;
    std::unordered_map<const void*, Handle> handles_;
    std::multimap<const uint8_t*, std::weak_ptr<Texture>> storage_;
    size_t max_virtual_size_ = 0;
    ResourceSlots<Texture> slots_;
    std::shared_ptr<Texture> detach(const void* key);
};
extern TextureRegistry g_textures;
extern Objects<Window> g_windows;
extern ResourceRegistry<DescriptorPool> g_texture_pools, g_sampler_pools;

Program* find_program(const void* program);
// Traduce la etapa la primera vez que se usa (program.cpp); false si no se pudo.
bool ensure_translated(ProgramStage& stage);
// The same, but a graphics stage that isn't in the SPIR-V cache yet goes to a background thread instead (program.cpp):
// 0, done here as ensure_translated (stage.module says if it worked); 1, queued (draws with it wait).
int translate_or_queue(ProgramStage& stage);
// Precarga (program.cpp): el hilo que traduce los shaders al fijarlos le pasa al ejecutor cada compute traducido, para
// crear su pipeline antes del primer dispatch (la toma warm_take).
void prefetch_compute_pipeline(const ProgramStage& stage);
// What has a range of the game's memory on our side (executor.cpp): for a log line
std::string describe_memory(const void* p, size_t size);

// Lo que pasó en el último tramo del log de ritmo (window.cpp lo escribe cada 10 s si hubo frames lentos y lo pone a
// cero): GPU del frame del juego, el ejecutor por frame, pipelines y traducciones en NvnSubmit y en segundo plano.
struct DropStats {
    std::atomic<uint64_t> gpu_sum_ns{0}, gpu_max_ns{0}, gpu_frames{0};
    std::atomic<uint64_t> run_ns{0}, run_max_ns{0}, wait_ns{0};  // run_max: el peor frame (suma de sus envíos)
    std::atomic<uint64_t> pipelines_live{0}, pipelines_live_ns{0};   // creadas en NvnSubmit con el frame parado
    std::atomic<uint64_t> pipelines_async{0}, draws_skipped{0};     // mandadas al hilo y draws sin dibujar esperándolas
    std::atomic<uint64_t> pipelines_known{0}, pipelines_known_ns{0};  // already recorded: made in NvnSubmit, not on the threads
    std::atomic<uint64_t> translated_live{0}, translated_live_ns{0};  // traducciones (no de la caché) en NvnSubmit
    std::atomic<uint64_t> prefetched{0}, prefetch_ns{0};             // las del hilo de la precarga
    std::mutex slow_mutex;      // el envío más lento del tramo (de 40 ms o más) y en qué se le fue el tiempo
    uint64_t slow_ns = 0;
    std::string slow_summary;
    std::atomic<uint64_t> pipeline_wait_ns{0};  // time spent joining worker or warm-up results at first use
};
extern DropStats g_drop_stats;
inline void note_max(std::atomic<uint64_t>& a, uint64_t v) {
    uint64_t now = a.load(std::memory_order_relaxed);
    while (v > now && !a.compare_exchange_weak(now, v, std::memory_order_relaxed)) {
    }
}
// true si hay que usar stripped_module en vez de module: un interruptor en vivo devuelve ese compute a la traducción
// sin las barreras de sus bucles (<datos>/no_exposure_barrier, no_reduction_barrier o no_uniform_barrier: todos).
bool use_stripped_module(const ProgramStage& stage);
// Geometry: el módulo para un vertex shader que escribe `previous_stores` y una topología de entrada de
// Eden; module a VK_NULL_HANDLE si no se puede traducir.
ProgramStage::Variant geometry_variant(ProgramStage& stage, uint64_t previous_stores, int input_topology);
// La misma etapa traducida con opciones más prudentes, para un driver que rechazó la pipeline (executor.cpp): nivel 1 sin
// la decoración NonUniform; 2, además pasada por spirv-opt; 3, además sin operaciones de subgrupo. Ninguna cambia los
// recursos (el layout es el mismo). previous_stores: el de la variante que se usaba (~0: la etapa tal cual).
constexpr int kSafeLevels = 3;
ProgramStage::Variant safe_variant(ProgramStage& stage, uint64_t previous_stores, int level, uint64_t input_types = 0);
// Vertex: el módulo con cada atributo declarado del tipo de su formato (nvnshader::Options::input_types); module a
// VK_NULL_HANDLE si no se puede traducir.
ProgramStage::Variant vertex_variant(ProgramStage& stage, uint64_t input_types);
// Identidad de una traducción (ProgramStage::spirv_id): huella del SPIR-V, de la etapa, de los genéricos y de lo que el
// layout toma de cada recurso (tipo, set, binding, número). Dos etapas con la misma comparten pipelines y layouts.
uint64_t spirv_identity(const nvnshader::Translated& t);
// vkCreateShaderModule con el SPIR-V que va al driver (program.cpp: sin las funciones que nadie llama)
VkResult create_shader_module(VkDevice device, const std::vector<uint32_t>& spirv, VkShaderModule* module);
VkResult create_fetch_module(VkDevice device, const std::vector<uint32_t>& spirv, uint64_t types, VkShaderModule* module);
// Precalentado (ejecutor): espera a que la caché de SPIR-V esté leída (la abre si hacía falta); false si está apagada
// o no llegó a tiempo. cached_translation: la traducción guardada con esa clave.
bool shader_cache_ready(int timeout_ms);
bool cached_translation(const spirv_cache::Key& key, nvnshader::Translated& out);
// DEPURACIÓN: guarda el código de la etapa en <datos>/shaders/<name>.code.
void dump_stage_code(const ProgramStage& stage, const std::string& name);
struct CopyRegion;
// Como nvnTextureWriteTexels: texels lineales (compactos si los strides son 0) a la memoria de la textura, en su
// disposición; la caja se sube a la imagen al usarla. Para CopyBufferToTexture, que en la Switch copia la GPU.
void write_texels_to(Texture& t, const TextureViewDesc* view, const CopyRegion* region, const void* data, int64_t row_stride,
                     int64_t image_stride);
Sampler* find_sampler(const void* sampler);

// Handles de NVN (nuestro formato): separados = id; combinado = sampler << 32 | textura.
inline uint32_t handle_texture_id(uint64_t h) { return static_cast<uint32_t>(h & 0xFFFFFFFF); }
inline uint32_t handle_sampler_id(uint64_t h) { return static_cast<uint32_t>(h >> 32); }

// Presenta en pantalla la textura index de la ventana (window.cpp).
void present(Window& window, int index);

}  // namespace nvn
