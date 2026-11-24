// nvnTextureBuilder*, nvnTexture*, nvnTextureView*, nvnTexturePool*, nvnSamplerBuilder*, nvnSampler*,
// nvnSamplerPool* y los handles del dispositivo. Las imágenes de Vulkan las crea el ejecutor cuando
// hacen falta; aquí solo se guarda la descripción y dónde están los texels en memoria del juego.

#include <algorithm>
#include <cstring>
#include <vector>
#include <memory>
#include <mutex>
#include <utility>

#include "nnsdk/nvn/commands.h"
#include "nnsdk/nvn/flags.h"
#include "nnsdk/nvn/formats.h"
#include "nnsdk/nvn/nvn.h"
#include "nnsdk/nvn/objects.h"
#include "nnsdk/nvn/texture_region.h"

using nvn::SamplerDesc;
using nvn::TextureDesc;
using nvn::TextureViewDesc;

namespace {

constexpr size_t kStorageAlignment = 512;

nvn::ResourceRegistry<nvn::Sampler> g_samplers;
nvn::ResourceSlots<nvn::Sampler> g_builder_sampler_slots; // serialized by the executor

nvn::TextureLayout layout_of(const TextureDesc& d) {
    const nvn::FormatInfo* f = nvn::format_info(d.format);
    uint32_t bpb = f && f->bytes ? f->bytes : 4, block_w = f ? f->block : 1, block_h = f ? f->block_height() : 1;
    bool array = d.target == nvn::kTarget1DArray || d.target == nvn::kTarget2DArray ||
                 d.target == nvn::kTarget2DMultisampleArray || d.target == nvn::kTargetCubeArray;
    bool cube = d.target == nvn::kTargetCube || d.target == nvn::kTargetCubeArray;
    bool is_3d = d.target == nvn::kTarget3D;
    // Un cubemap tiene 6 caras aunque el juego deje depth en 1 (NieR lo hace); un array de cubos, depth
    // caras (múltiplo de 6). Con 1 sola capa, GetStorageSize salía de una cara y los datos de las otras
    // 5 se escribían sobre lo que hubiera detrás (el mar del prólogo salía magenta: ASTC inválido).
    uint32_t layers = d.target == nvn::kTargetCube ? 6 : array || cube ? std::max(d.depth, 1) : 1;
    uint32_t depth = is_3d ? std::max(d.depth, 1) : 1;
    uint32_t height = d.target == nvn::kTarget1D || d.target == nvn::kTarget1DArray ? 1 : d.height;
    uint32_t samples = std::max(d.samples, 1);
    // NVNpackagedTextureLayout (SetPackagedTextureLayout): byte 0 = alto del bloque y byte 1 = alineación del pitch, en
    // log2 de GOBs. Las texturas sparse de MH Rise traen 03 04 (8 GOBs de alto, pitch a 16: tiles de 64 KB) y el juego
    // escribe sus mips ya swizzleados así, en los offsets que le da GetViewOffset: hay que usar los mismos.
    // <datos>/no_packed_layout (al arrancar: las texturas se crean al cargar): el bloque calculado, como antes.
    const bool packaged = d.has_packaged_layout && !(d.flags & nvn::kTextureFlagLinear) && !is_3d &&
                          d.packaged_layout[0] <= 5 && d.packaged_layout[1] <= 5 && !NVN_FLAG("no_packed_layout");
    return nvn::compute_layout(d.width * (samples > 1 ? 2 : 1), height * (samples > 2 ? 2 : 1), depth, layers,
                               d.levels, bpb * (samples > 4 ? 2 : 1), block_w, block_h, d.flags & nvn::kTextureFlagLinear,
                               d.stride, is_3d, packaged ? d.packaged_layout[0] : -1, packaged ? d.packaged_layout[1] : 0);
}

int storage_class_of(const TextureDesc& d, bool raw = false) {
    using namespace nvn;
    if (d.flags & kTextureFlagLinear) return kStorageClassPitch;
    if (raw || !(d.flags & kTextureFlagCompressible) || d.samples > 1 || d.target == kTargetBuffer) return kStorageClassBlockLinear;
    const auto* fmt = format_info(d.format);
    return fmt && fmt->block == 1 && fmt->bytes ? kStorageClassNativeImage : kStorageClassBlockLinear;
}

// Nivel y capa que nombra una vista (lo que no fija, el primero).
int view_level(const TextureViewDesc* v) { return v ? nvn::transfer_level(*v) : 0; }
int view_layer(const TextureViewDesc* v) { return v ? nvn::transfer_base_layer(*v) : 0; }

// Lo que tocan WriteTexels/ReadTexels: un nivel, unas capas y una caja en bloques dentro del nivel. La región va en
// texels; en los arrays su z (y en los 1D array su y) elige las capas, como en las copias de NVN. false si no cabe.
struct TexelRegion {
    const nvn::LevelLayout* level;
    uint32_t level_index, layer, layers, bpb;
    nvn::BlockBox box;
    size_t row_pitch, image_pitch;  // de la memoria lineal del juego (compacta si no da otros)
};
bool texel_region(const nvn::Texture& t, const TextureViewDesc* view, const nvn::CopyRegion* r, int64_t row_stride,
                  int64_t image_stride, TexelRegion& out) {
    const nvn::TextureLayout& L = t.layout;
    nvn::TextureRegion region;
    if (!r || !t.storage || !nvn::resolve_texture_region(t.desc.target, view ? *view : TextureViewDesc{}, *r,
                                                          L.levels.size(), L.layers, region)) return false;
    const nvn::LevelLayout& lv = L.levels[region.level];
    out.level = &lv;
    out.level_index = region.level;
    out.bpb = L.bytes_per_block;
    auto [x,y,z,w,h,d] = region.box;
    // Recortada al nivel: el juego no debería pasarse, y escribir fuera pisaría el siguiente nivel.
    w = std::min<int>(w, static_cast<int>(lv.width) - x), h = std::min<int>(h, static_cast<int>(lv.height) - y);
    d = std::min<int>(d, static_cast<int>(std::max(lv.depth, 1u)) - z);
    if (w <= 0 || h <= 0 || d <= 0) return false;
    out.layer = region.layer;
    out.layers = region.layers;
    const uint32_t bw = L.block_w, bh = L.block_h;
    out.box = {x / bw, y / bh, static_cast<uint32_t>(z), (x % bw + w + bw - 1) / bw, (y % bh + h + bh - 1) / bh,
               static_cast<uint32_t>(d)};
    out.box.width = std::min(out.box.width, lv.width_blocks - out.box.x);
    out.box.height = std::min(out.box.height, lv.height_blocks - out.box.y);
    out.row_pitch = row_stride > 0 ? static_cast<size_t>(row_stride) : size_t{out.box.width} * out.bpb;
    out.image_pitch = image_stride > 0 ? static_cast<size_t>(image_stride) : out.row_pitch * out.box.height;
    return true;
}

// Una capa o un corte de lo que tocan: dónde empieza el nivel en la memoria de la textura.
uint8_t* level_storage(const nvn::Texture& t, const TexelRegion& r, uint32_t layer) {
    return t.storage + size_t{layer} * t.layout.layer_size + r.level->offset;
}

// Lineal con stride: filas tal cual. Block-linear: la caja (ver layout.h).
void mark_texel_write(nvn::Texture& t, const TexelRegion& r) {
    std::lock_guard lock(nvn::texel_writes_mutex());
    if (!t.texel_writes.empty()) {
        const auto& last = t.texel_writes.back();
        if (last.level == r.level_index && last.layer == r.layer && last.layers == r.layers && last.box == r.box) return;
    }
    t.texel_writes.push_back({r.level_index, r.layer, r.layers, r.box});
    nvn::g_texel_writes++;
}

void write_texels(nvn::Texture& t, const TextureViewDesc* view, const nvn::CopyRegion* region, const void* data,
                  int64_t row_stride, int64_t image_stride) {
    TexelRegion r;
    if (!data || !texel_region(t, view, region, row_stride, image_stride, r)) {
        static int logged = 0;
        if (logged++ < 8)
            NVN_LOG("WriteTexels fuera de la textura %dx%dx%d (%d,%d,%d %dx%dx%d) o sin memoria\n", t.desc.width,
                    t.desc.height, t.desc.depth, region ? region->x : 0, region ? region->y : 0, region ? region->z : 0,
                    region ? region->width : 0, region ? region->height : 0, region ? region->depth : 0);
        return;
    }
    if (!nvn::before_guest_texture_access(t.storage, t.layout.total_size())) return;
    const auto* src = static_cast<const uint8_t*>(data);
    for (uint32_t k = 0; k < r.layers; k++) {
        uint8_t* level = level_storage(t, r, r.layer + k);
        const uint8_t* in = src + k * r.image_pitch;
        if (t.layout.linear) {
            for (uint32_t y = 0; y < r.box.height; y++)
                memcpy(level + size_t{r.box.y + y} * t.layout.stride + size_t{r.box.x} * r.bpb, in + y * r.row_pitch,
                       size_t{r.box.width} * r.bpb);
        } else {
            nvn::swizzle_region(*r.level, r.bpb, r.box, in, r.row_pitch, r.image_pitch, level);
        }
    }
    // La imagen se rehace desde la memoria al usarla (solo esta caja si la GPU ya escribió el resto).
    mark_texel_write(t, r);
}

// Las escrituras pendientes de una textura que se reinicia o se finaliza: fuera, y fuera de la cuenta (con ella a 0 el
// ejecutor ni toma el mutex).
void drop_texel_writes(nvn::Texture& t) {
    std::lock_guard lock(nvn::texel_writes_mutex());
    nvn::g_texel_writes -= static_cast<uint32_t>(t.texel_writes.size());
    t.texel_writes.clear();
}

void read_texels(nvn::Texture& t, const TextureViewDesc* view, const nvn::CopyRegion* region, void* data,
                 int64_t row_stride, int64_t image_stride) {
    TexelRegion r;
    if (!data || !texel_region(t, view, region, row_stride, image_stride, r)) return;
    if (!nvn::sync_texture_storage(t, {r.level_index, r.layer, r.layers, r.box})) return;
    auto* dst = static_cast<uint8_t*>(data);
    for (uint32_t k = 0; k < r.layers; k++) {
        const uint8_t* level = level_storage(t, r, r.layer + k);
        uint8_t* out = dst + k * r.image_pitch;
        if (t.layout.linear) {
            for (uint32_t y = 0; y < r.box.height; y++)
                memcpy(out + y * r.row_pitch, level + size_t{r.box.y + y} * t.layout.stride + size_t{r.box.x} * r.bpb,
                       size_t{r.box.width} * r.bpb);
        } else {
            nvn::deswizzle_region(*r.level, r.bpb, r.box, level, out, r.row_pitch, r.image_pitch);
        }
    }
}

// Getters del builder y del sampler (la misma descripción).
void sampler_min_mag(const SamplerDesc& d, int* min, int* mag) {
    if (min) *min = d.min_filter;
    if (mag) *mag = d.mag_filter;
}
void sampler_wrap(const SamplerDesc& d, int* s, int* t, int* r) {
    if (s) *s = d.wrap[0];
    if (t) *t = d.wrap[1];
    if (r) *r = d.wrap[2];
}
void sampler_lod_clamp(const SamplerDesc& d, float* min, float* max) {
    if (min) *min = d.lod_min;
    if (max) *max = d.lod_max;
}
void sampler_compare(const SamplerDesc& d, int* mode, int* func) {
    if (mode) *mode = d.compare_mode;
    if (func) *func = d.compare_func;
}
template <typename T>
void border_int(const SamplerDesc& d, T* color) {
    for (int k = 0; k < 4; k++) color[k] = static_cast<T>(d.border[k]);
}
SamplerDesc sampler_desc(const void* sampler) {
    auto s = g_samplers.acquire(sampler);
    return s ? s->desc : SamplerDesc{};
}

void release_sampler(nvn::Sampler& s) {
    for (auto* vk : {&s.vk, &s.vk_af, &s.vk_lit})
        if (*vk) nvn::retire_sampler(std::exchange(*vk, VK_NULL_HANDLE));
}

}  // namespace

void nvn::write_texels_to(Texture& t, const TextureViewDesc* view, const CopyRegion* region, const void* data,
                          int64_t row_stride, int64_t image_stride) {
    write_texels(t, view, region, data, row_stride, image_stride);
}

nvn::TextureRegistry nvn::g_textures;

void nvn::Texture::note_mapping_change(size_t first, size_t last) {
    auto it = std::lower_bound(remapped_ranges.begin(), remapped_ranges.end(), first,
        [](const MappingRange& r, size_t start) { return r.second < start; });
    if (it == remapped_ranges.end() || it->first > last) {
        remapped_ranges.emplace(it, first, last);
    } else {
        it->first = std::min(it->first, first);
        it->second = std::max(it->second, last);
        auto next = it + 1;
        while (next != remapped_ranges.end() && next->first <= it->second) {
            it->second = std::max(it->second, next->second);
            ++next;
        }
        remapped_ranges.erase(it + 1, next);
    }
    __atomic_store_n(&mapping_changed, true, __ATOMIC_RELEASE);
    g_mapping_serial.fetch_add(1, std::memory_order_release);
}

void nvn::TextureRegistry::restore_mapping_changes(Texture& t, const std::vector<Texture::MappingRange>& ranges) {
    std::lock_guard lock(mutex_);
    for (auto [first, last] : ranges) t.note_mapping_change(first, last);
}

std::vector<nvn::Texture::MappingRange> nvn::TextureRegistry::take_mapping_changes(Texture& t) {
    std::lock_guard lock(mutex_);
    __atomic_store_n(&t.mapping_changed, false, __ATOMIC_RELEASE);
    std::vector<Texture::MappingRange> ranges;
    ranges.swap(t.remapped_ranges);
    return ranges;
}

nvn::Texture* nvn::TextureRegistry::get(const void* key) {
    return reference(key).get();
}

nvn::ResourceRef<nvn::Texture> nvn::TextureRegistry::reference(const void* key) {
    std::lock_guard lock(mutex_);
    auto it = handles_.find(key);
    return it == handles_.end() ? ResourceRef<Texture>{} : it->second.texture->ref;
}

std::shared_ptr<nvn::Texture> nvn::TextureRegistry::acquire(const void* key) {
    std::lock_guard lock(mutex_);
    auto it = handles_.find(key);
    return it == handles_.end() ? nullptr : it->second.texture;
}

uint64_t nvn::TextureRegistry::debug_id(const void* key) {
    std::lock_guard lock(mutex_);
    auto it = handles_.find(key);
    return it == handles_.end() ? 0 : it->second.id;
}

// Caller holds the registry lock and, for a live key, the executor lock.
std::shared_ptr<nvn::Texture> nvn::TextureRegistry::detach(const void* key) {
    auto it = handles_.find(key);
    if (it == handles_.end()) return {};
    auto old = std::move(it->second.texture);
    handles_.erase(it);
    if (--old->handle_count) return {};
    auto [first, last] = storage_.equal_range(old->storage);
    for (auto s = first; s != last; ++s) {
        if (s->second.lock() == old) {
            storage_.erase(s);
            break;
        }
    }
    release_texture_gpu(*old);
    drop_texel_writes(*old);
    old->storage = nullptr;
    old->vk_failed = true;
    old->serial = next_texture_serial();
    return old;
}

bool nvn::TextureRegistry::bind(const void* key, Texture texture) {
    std::lock_guard lock(mutex_);
    auto reusable = detach(key);
    auto [first, last] = storage_.equal_range(texture.storage);
    if (texture.storage) {
        for (auto s = first; s != last; ++s) {
            auto shared = s->second.lock();
            if (shared->desc == texture.desc) {
                ++shared->handle_count;
                handles_.emplace(key, Handle{std::move(shared), texture.serial});
                if (reusable) slots_.erase(reusable->ref);
                return false;
            }
        }
    }
    // Reinitialization without Finalize keeps the address held by recorded commands
    // when this handle exclusively owns the old state.
    if (texture.virtual_storage && texture.storage) {
        const auto start = reinterpret_cast<uintptr_t>(texture.storage);
        const size_t size = texture.layout.total_size();
        auto it = storage_.lower_bound(reinterpret_cast<const uint8_t*>(start - std::min<size_t>(start, max_virtual_size_)));
        for (; it != storage_.end() && reinterpret_cast<uintptr_t>(it->first) < start + size; ++it) {
            auto other = it->second.lock();
            if (other->desc.pool != texture.desc.pool ||
                reinterpret_cast<uintptr_t>(it->first) + other->layout.total_size() <= start) continue;
            texture.raw_alias = true;
            __atomic_store_n(&other->raw_alias, true, __ATOMIC_RELEASE);
        }
        max_virtual_size_ = std::max(max_virtual_size_, size);
    }
    const bool aliases = texture.raw_alias;
    if (!reusable) {
        reusable = std::make_shared<Texture>();
        texture.ref = slots_.insert(reusable.get());
    } else {
        texture.ref = reusable->ref; // reinitializing a live handle keeps its bindings
    }
    texture.handle_count = 1;
    *reusable = std::move(texture);
    storage_.emplace(reusable->storage, reusable);
    handles_.emplace(key, Handle{reusable, reusable->serial});
    return aliases;
}

void nvn::TextureRegistry::unbind(const void* key) {
    std::lock_guard lock(mutex_);
    if (auto last = detach(key)) slots_.erase(last->ref);
}
std::atomic<uint32_t> nvn::g_texel_writes{0};
std::atomic<uint64_t> nvn::g_mapping_serial{0};

std::mutex& nvn::texel_writes_mutex() {
    static std::mutex m;
    return m;
}
std::atomic<uint64_t> nvn::g_texture_gpu_bytes{0}, nvn::g_texture_gpu_images{0};

uint64_t nvn::next_texture_serial() {
    static std::atomic<uint64_t> serial{1};
    return serial++;
}

// Destruirla ya podía pillar a la GPU leyéndola en un envío pendiente: la retira el ejecutor.
void nvn::release_texture_gpu(Texture& t) { nvn::retire_texture_gpu(t); }
nvn::ResourceRegistry<nvn::DescriptorPool> nvn::g_texture_pools, nvn::g_sampler_pools;

nvn::Sampler* nvn::find_sampler(const void* sampler) { return g_samplers.get(sampler); }

void nvn::DescriptorPool::register_sampler(uint32_t id, ResourceRef<Sampler> sampler) {
    decltype(builder_samplers)::node_type previous;
    {
        std::lock_guard lock(entries_mutex);
        samplers[id] = sampler;
        sampler_snapshot_.reset();
        previous = builder_samplers.extract(id);
    }
    if (previous) {
        std::lock_guard lock(executor_mutex());
        auto& builder = previous.mapped();
        g_builder_sampler_slots.erase(builder.ref);
        release_sampler(builder.value);
    }
}

// The caller holds the executor lock while replacing a builder-owned VkSampler.
void nvn::DescriptorPool::register_sampler_builder(uint32_t id, const SamplerDesc& desc) {
    std::lock_guard lock(entries_mutex);
    auto [it, inserted] = builder_samplers.try_emplace(id);
    auto& builder = it->second;
    if (!inserted) {
        g_builder_sampler_slots.erase(builder.ref);
        release_sampler(builder.value);
    }
    builder.value.desc = desc;
    builder.ref = g_builder_sampler_slots.insert(&builder.value);
    samplers[id] = builder.ref;
    sampler_snapshot_.reset();
}

void nvn::DescriptorPool::retire_sampler_builders() {
    std::lock_guard lock(entries_mutex);
    for (auto& [id, builder] : builder_samplers) {
        samplers[id] = nullptr;
        g_builder_sampler_slots.erase(builder.ref);
        release_sampler(builder.value);
    }
    builder_samplers.clear();
    sampler_snapshot_.reset();
}

extern "C" {

// --- Builder de texturas ---

void nvnTextureBuilderSetDefaults(TextureDesc* b) {
    memset(b, 0, sizeof *b);
    b->target = nvn::kTarget2D;
    b->width = b->height = b->depth = b->levels = 1;
    b->swizzle[0] = 2, b->swizzle[1] = 3, b->swizzle[2] = 4, b->swizzle[3] = 5;  // R G B A
}
void nvnTextureBuilderSetDevice(TextureDesc* b, void* device) { b->device = device; }
void nvnTextureBuilderSetFlags(TextureDesc* b, int flags) { b->flags = flags; }
void nvnTextureBuilderSetTarget(TextureDesc* b, int target) { b->target = target; }
void nvnTextureBuilderSetFormat(TextureDesc* b, int format) { b->format = format; }
void nvnTextureBuilderSetLevels(TextureDesc* b, int levels) { b->levels = levels; }
void nvnTextureBuilderSetSamples(TextureDesc* b, int samples) { b->samples = samples; }
void nvnTextureBuilderSetStride(TextureDesc* b, int64_t stride) { b->stride = static_cast<int>(stride); }
void nvnTextureBuilderSetWidth(TextureDesc* b, int w) { b->width = w; }
void nvnTextureBuilderSetHeight(TextureDesc* b, int h) { b->height = h; }
void nvnTextureBuilderSetDepth(TextureDesc* b, int d) { b->depth = d; }
void nvnTextureBuilderSetSize1D(TextureDesc* b, int w) { b->width = w, b->height = b->depth = 1; }
void nvnTextureBuilderSetSize2D(TextureDesc* b, int w, int h) { b->width = w, b->height = h, b->depth = 1; }
void nvnTextureBuilderSetSize3D(TextureDesc* b, int w, int h, int d) { b->width = w, b->height = h, b->depth = d; }
void nvnTextureBuilderSetStorage(TextureDesc* b, void* pool, int64_t offset) {
    b->pool = pool;
    b->offset = offset;
}
// El juego reserva esto y escribe ahí los texels en disposición Tegra: tiene que ser el tamaño real.
size_t nvnTextureBuilderGetStorageSize(const TextureDesc* b) {
    size_t size = (layout_of(*b).total_size() + kStorageAlignment - 1) & ~(kStorageAlignment - 1);
    if (size > (64u << 20))
        NVN_LOG("tamaño de textura enorme: %zu MB para %dx%dx%d formato 0x%x target %d niveles %d muestras %d flags 0x%x\n",
                size >> 20, b->width, b->height, b->depth, b->format, b->target, b->levels, b->samples, b->flags);
    return size;
}
size_t nvnTextureBuilderGetStorageAlignment(const TextureDesc*) { return kStorageAlignment; }
// Lo que se muestrea de una textura de profundidad+stencil si la vista no lo dice (MH Rise lo pone a 0, DEPTH).
void nvnTextureBuilderSetDepthStencilMode(TextureDesc* b, int mode) { b->depth_stencil_mode = static_cast<uint8_t>(mode); }
// La disposición del empaquetador (NVNpackagedTextureLayout, 8 bytes opacos): se guarda y se registra con lo que
// calculamos nosotros, para ver si coinciden.
void nvnTextureBuilderSetPackagedTextureLayout(TextureDesc* b, const uint8_t* layout) {
    b->has_packaged_layout = layout != nullptr;
    if (layout) memcpy(b->packaged_layout, layout, 8);
}
void nvnTextureBuilderSetSwizzle(TextureDesc* b, int r, int g, int bl, int a) {
    b->swizzle[0] = r, b->swizzle[1] = g, b->swizzle[2] = bl, b->swizzle[3] = a;
}
int nvnTextureBuilderGetStorageClass(const TextureDesc* b) { return storage_class_of(*b); }
int nvnTextureBuilderGetRawStorageClass(const TextureDesc* b) { return storage_class_of(*b, true); }
// Sin ZCull (la GPU lo hace a su manera): nada que reservar.
size_t nvnTextureBuilderGetZCullStorageSize(const TextureDesc*) { return 0; }
void* nvnTextureBuilderGetDevice(const TextureDesc* b) { return b->device; }
int nvnTextureBuilderGetFlags(const TextureDesc* b) { return b->flags; }
int nvnTextureBuilderGetTarget(const TextureDesc* b) { return b->target; }
int nvnTextureBuilderGetWidth(const TextureDesc* b) { return b->width; }
int nvnTextureBuilderGetHeight(const TextureDesc* b) { return b->height; }
int nvnTextureBuilderGetDepth(const TextureDesc* b) { return b->depth; }
int nvnTextureBuilderGetLevels(const TextureDesc* b) { return b->levels; }
int nvnTextureBuilderGetFormat(const TextureDesc* b) { return b->format; }
int nvnTextureBuilderGetSamples(const TextureDesc* b) { return b->samples; }
int64_t nvnTextureBuilderGetStride(const TextureDesc* b) { return b->stride; }
void* nvnTextureBuilderGetMemoryPool(const TextureDesc* b) { return b->pool; }
int64_t nvnTextureBuilderGetMemoryOffset(const TextureDesc* b) { return b->offset; }
int nvnTextureBuilderGetDepthStencilMode(const TextureDesc* b) { return b->depth_stencil_mode; }
void nvnTextureBuilderGetSwizzle(const TextureDesc* b, int* r, int* g, int* bl, int* a) {
    if (r) *r = b->swizzle[0];
    if (g) *g = b->swizzle[1];
    if (bl) *bl = b->swizzle[2];
    if (a) *a = b->swizzle[3];
}

// --- Texturas ---

nvn::Boolean nvnTextureInitialize(void* texture, const TextureDesc* b) {
    // Reinicializar sin Finalize: el mismo objeto (los comandos grabados y los pools de texturas guardan su
    // dirección), sin la imagen anterior.
    std::unique_lock<std::recursive_mutex> lock(nvn::executor_mutex(), std::defer_lock);
    auto existing = nvn::g_textures.acquire(texture);
    if (existing) {
        nvn::lock_executor(lock);
        if (existing->virtual_storage && !nvn::publish_virtual_pool(existing->desc.pool, existing.get())) return 0;
    }
    nvn::Texture t;
    t.serial = nvn::next_texture_serial();
    t.desc = *b;
    if (!t.desc.has_packaged_layout) memset(t.desc.packaged_layout, 0, sizeof t.desc.packaged_layout);
    if (!nvn::format_info(b->format) || !nvn::format_info(b->format)->bytes)
        NVN_LOG("textura con formato 0x%x sin mapear: %dx%dx%d target %d niveles %d flags 0x%x\n", b->format, b->width,
                b->height, b->depth, b->target, b->levels, b->flags);
    t.layout = layout_of(*b);
    if (b->flags & 0x20) {  // DEPURACIÓN: texturas sparse (streaming): ¿cómo las crea y re-crea?
        static int logged = 0;
        if (logged++ < 200)
            NVN_LOG("textura sparse %p%s: %dx%d formato 0x%x niveles %d pool %p +0x%llx -> %p\n", texture,
                    existing ? " (re-init)" : "", b->width, b->height, b->format, b->levels, b->pool,
                    static_cast<unsigned long long>(b->offset), static_cast<void*>(nvn::g_memory_pools.get(b->pool) ?
                    nvn::g_memory_pools.get(b->pool)->storage + b->offset : nullptr));
    }
    if (b->has_packaged_layout) {  // DEPURACIÓN: qué dice el empaquetador frente a nuestros bloques
        static int logged = 0;
        if (logged++ < 400) {
            const auto& l0 = t.layout.levels.empty() ? nvn::LevelLayout{} : t.layout.levels[0];
            const uint8_t* p = b->packaged_layout;
            NVN_LOG("layout empaquetado %02x %02x %02x %02x %02x %02x %02x %02x: %dx%dx%d target %d formato 0x%x niveles %d "
                    "flags 0x%x; nuestro bloque %ux%u GOBs, pitch %u GOBs, %zu KB, %s\n", p[0], p[1], p[2], p[3], p[4], p[5],
                    p[6], p[7], b->width, b->height, b->depth, b->target, b->format, b->levels, b->flags,
                    l0.block_height_gobs, l0.block_depth, l0.row_gobs, t.layout.total_size() >> 10,
                    t.layout.linear ? "lineal" : "block-linear");
        }
    }
    if (nvn::MemoryPool* pool = nvn::g_memory_pools.get(b->pool); pool && pool->storage) {
        t.storage = pool->storage + b->offset;
        t.virtual_storage = pool->reserved;
        t.gpu_only_virtual = pool->reserved && (pool->flags & nvn::kPoolFlagCpuNoAccess) && !t.layout.linear;
    }
    if ((b->width == 1600 && b->height == 904) || (b->width == 800 && b->height == 452))  // DEPURACIÓN: vídeo
        NVN_LOG("textura de vídeo %p: %dx%d formato 0x%x flags 0x%x stride %d pool %p +0x%llx -> %p\n", texture, b->width,
                b->height, b->format, b->flags, b->stride, b->pool, static_cast<unsigned long long>(b->offset),
                static_cast<void*>(t.storage));
    if (t.storage && !nvn::before_guest_texture_access(t.storage, t.layout.total_size())) return 0;
    if (nvn::g_textures.bind(texture, std::move(t)) && !nvn::publish_virtual_pool(b->pool)) {
        if (!lock.owns_lock()) nvn::lock_executor(lock);
        nvn::g_textures.unbind(texture);
        return 0;
    }
    return 1;
}
void nvnTextureFinalize(void* texture) {
    std::unique_lock<std::recursive_mutex> executor_lock(nvn::executor_mutex(), std::defer_lock);
    nvn::lock_executor(executor_lock);
    if (auto* t = nvn::g_textures.get(texture); t && t->virtual_storage && !nvn::publish_virtual_pool(t->desc.pool, t)) return;
    nvn::g_textures.unbind(texture);
}

#define TEXTURE_GETTER(name, field)                          \
    int nvnTextureGet##name(const void* t) {                \
        auto tex = nvn::g_textures.acquire(t);         \
        return tex ? tex->desc.field : 0;                   \
    }
TEXTURE_GETTER(Width, width)
TEXTURE_GETTER(Height, height)
TEXTURE_GETTER(Depth, depth)
TEXTURE_GETTER(Format, format)
TEXTURE_GETTER(Target, target)
TEXTURE_GETTER(Levels, levels)
TEXTURE_GETTER(Flags, flags)
TEXTURE_GETTER(Samples, samples)
TEXTURE_GETTER(DepthStencilMode, depth_stencil_mode)
#undef TEXTURE_GETTER
int nvnTextureGetStorageClass(const void* t) {
    auto tex = nvn::g_textures.acquire(t);
    return tex ? storage_class_of(tex->desc) : nvn::kStorageClassBlockLinear;
}
int nvnTextureGetRawStorageClass(const void* t) {
    auto tex = nvn::g_textures.acquire(t);
    return tex ? storage_class_of(tex->desc, true) : nvn::kStorageClassBlockLinear;
}
size_t nvnTextureGetZCullStorageSize(const void*) { return 0; }
void* nvnTextureGetMemoryPool(const void* t) {
    auto tex = nvn::g_textures.acquire(t);
    return tex ? tex->desc.pool : nullptr;
}
int64_t nvnTextureGetMemoryOffset(const void* t) {
    auto tex = nvn::g_textures.acquire(t);
    return tex ? tex->desc.offset : 0;
}
// Dirección de GPU = de CPU: la de sus texels.
uint64_t nvnTextureGetTextureAddress(const void* t) {
    auto tex = nvn::g_textures.acquire(t);
    if (tex && tex->gpu_only_virtual) {
        __atomic_store_n(&tex->raw_alias, true, __ATOMIC_RELEASE);
        if (!nvn::publish_virtual_pool(tex->desc.pool, tex.get())) return 0;
    }
    return tex ? reinterpret_cast<uint64_t>(tex->storage) : 0;
}
void nvnTextureGetSwizzle(const void* t, int* r, int* g, int* b, int* a) {
    auto tex = nvn::g_textures.acquire(t);
    const uint8_t identity[4] = {2, 3, 4, 5};
    const uint8_t* s = tex ? tex->desc.swizzle : identity;
    if (r) *r = s[0];
    if (g) *g = s[1];
    if (b) *b = s[2];
    if (a) *a = s[3];
}
int nvnTextureGetDebugID(const void* t) {
    return static_cast<int>(nvn::g_textures.debug_id(t));
}
void nvnTextureSetDebugLabel(void*, const char*) {}
// Dónde empieza en la memoria de la textura el primer nivel y la primera capa de la vista: capa · tamaño de capa +
// offset del nivel, con la misma disposición que GetStorageSize y la subida del ejecutor. MH Rise recorre cada
// (capa, nivel) con vistas de uno y copia en su staging, para cada uno, lo que va de su offset al del siguiente (el
// último, hasta GetStorageSize): con 0 copiaba la textura entera en cada nivel y se salía del staging
// (main+0x771da34 -> main+0x7723388).
int64_t nvnTextureGetViewOffset(const void* t, const TextureViewDesc* view) {
    auto tex = nvn::g_textures.acquire(t);
    if (!tex || tex->layout.levels.empty()) return 0;
    const nvn::TextureLayout& L = tex->layout;
    const size_t level = std::min<size_t>(std::max(view_level(view), 0), L.levels.size() - 1);
    // Las 3D no tienen capas (su profundidad va dentro de cada nivel).
    const size_t layer = tex->desc.target == nvn::kTarget3D ? 0 : std::min<size_t>(std::max(view_layer(view), 0), L.layers - 1);
    return static_cast<int64_t>(layer * L.layer_size + L.levels[level].offset);
}

// --- Texels desde la CPU: en la Switch van directos a la memoria de la textura (swizzleados si es block-linear) ---

void nvnTextureWriteTexels(const void* t, const TextureViewDesc* view, const nvn::CopyRegion* region, const void* data) {
    if (auto tex = nvn::g_textures.acquire(t)) write_texels(*tex, view, region, data, 0, 0);
}
void nvnTextureWriteTexelsStrided(const void* t, const TextureViewDesc* view, const nvn::CopyRegion* region,
                                  const void* data, int64_t row_stride, int64_t image_stride) {
    if (auto tex = nvn::g_textures.acquire(t)) write_texels(*tex, view, region, data, row_stride, image_stride);
}
void nvnTextureReadTexels(const void* t, const TextureViewDesc* view, const nvn::CopyRegion* region, void* data) {
    if (auto tex = nvn::g_textures.acquire(t)) read_texels(*tex, view, region, data, 0, 0);
}
void nvnTextureReadTexelsStrided(const void* t, const TextureViewDesc* view, const nvn::CopyRegion* region, void* data,
                                 int64_t row_stride, int64_t image_stride) {
    if (auto tex = nvn::g_textures.acquire(t)) read_texels(*tex, view, region, data, row_stride, image_stride);
}
void nvnTextureFlushTexels(const void* t, const TextureViewDesc* view, const nvn::CopyRegion* region) {
    TexelRegion r;
    if (auto tex = nvn::g_textures.acquire(t); tex && texel_region(*tex, view, region, 0, 0, r))
        mark_texel_write(*tex, r);
}
void nvnTextureInvalidateTexels(const void* t, const TextureViewDesc* view, const nvn::CopyRegion* region) {
    TexelRegion r;
    if (auto tex = nvn::g_textures.acquire(t); tex && texel_region(*tex, view, region, 0, 0, r))
        nvn::sync_texture_storage(*tex, {r.level_index, r.layer, r.layers, r.box});
}
// El juego escribe por CPU en texturas lineales (los frames de vídeo) con este stride.
int64_t nvnTextureGetStride(const void* t) {
    auto tex = nvn::g_textures.acquire(t);
    return tex && tex->layout.linear ? tex->layout.stride : 0;
}
size_t nvnTextureGetStorageSize(const void* t) {
    auto tex = nvn::g_textures.acquire(t);
    return tex ? tex->layout.total_size() : 0;
}

// --- Vistas ---

void nvnTextureViewSetDefaults(TextureViewDesc* v) { memset(v, 0, sizeof *v); }
void nvnTextureViewSetLevels(TextureViewDesc* v, int base, int count) {
    v->flags |= nvn::kViewLevels;
    v->base_level = base, v->levels = count;
}
void nvnTextureViewSetLayers(TextureViewDesc* v, int base, int count) {
    v->flags |= nvn::kViewLayers;
    v->base_layer = base, v->layers = count;
}
void nvnTextureViewSetFormat(TextureViewDesc* v, int format) {
    v->flags |= nvn::kViewFormat;
    v->format = format;
}
void nvnTextureViewSetTarget(TextureViewDesc* v, int target) {
    v->flags |= nvn::kViewTarget;
    v->target = target;
}
void nvnTextureViewSetDepthStencilMode(TextureViewDesc* v, int mode) {
    v->flags |= nvn::kViewDepthStencil;
    v->depth_stencil_mode = mode;
}
void nvnTextureViewSetSwizzle(TextureViewDesc* v, int r, int g, int b, int a) {
    static int logged = 0;  // DEPURACIÓN: qué combinaciones usa el juego
    static int seen[16][4];
    bool fresh = true;
    for (int k = 0; k < std::min(logged, 16); k++)
        if (seen[k][0] == r && seen[k][1] == g && seen[k][2] == b && seen[k][3] == a) fresh = false;
    if (fresh && logged < 16) {
        seen[logged][0] = r, seen[logged][1] = g, seen[logged][2] = b, seen[logged][3] = a;
        logged++;
        NVN_LOG("swizzle de vista: %d %d %d %d\n", r, g, b, a);
    }
    v->flags |= nvn::kViewSwizzle;
    v->swizzle[0] = r, v->swizzle[1] = g, v->swizzle[2] = b, v->swizzle[3] = a;
}

// Getters de la vista: NVN_TRUE si la vista fija ese campo (si no, no tocan la salida).
nvn::Boolean nvnTextureViewGetLevels(const TextureViewDesc* v, int* base, int* count) {
    if (!(v->flags & nvn::kViewLevels)) return 0;
    if (base) *base = v->base_level;
    if (count) *count = v->levels;
    return 1;
}
nvn::Boolean nvnTextureViewGetLayers(const TextureViewDesc* v, int* base, int* count) {
    if (!(v->flags & nvn::kViewLayers)) return 0;
    if (base) *base = v->base_layer;
    if (count) *count = v->layers;
    return 1;
}
nvn::Boolean nvnTextureViewGetFormat(const TextureViewDesc* v, int* format) {
    if (!(v->flags & nvn::kViewFormat)) return 0;
    if (format) *format = v->format;
    return 1;
}
nvn::Boolean nvnTextureViewGetTarget(const TextureViewDesc* v, int* target) {
    if (!(v->flags & nvn::kViewTarget)) return 0;
    if (target) *target = v->target;
    return 1;
}
nvn::Boolean nvnTextureViewGetDepthStencilMode(const TextureViewDesc* v, int* mode) {
    if (!(v->flags & nvn::kViewDepthStencil)) return 0;
    if (mode) *mode = v->depth_stencil_mode;
    return 1;
}
nvn::Boolean nvnTextureViewGetSwizzle(const TextureViewDesc* v, int* r, int* g, int* b, int* a) {
    if (!(v->flags & nvn::kViewSwizzle)) return 0;
    if (r) *r = v->swizzle[0];
    if (g) *g = v->swizzle[1];
    if (b) *b = v->swizzle[2];
    if (a) *a = v->swizzle[3];
    return 1;
}

// --- Pools de descriptores ---

nvn::Boolean nvnTexturePoolInitialize(void* pool, void* memory_pool, int64_t offset, int count) {
    std::lock_guard lock(nvn::executor_mutex());
    nvn::DescriptorPool& p = nvn::g_texture_pools.create(pool);
    p.memory_pool = memory_pool;
    p.offset = offset;
    p.count = count;
    p.textures.resize(count);
    return 1;
}
void nvnTexturePoolFinalize(void* pool) {
    std::lock_guard lock(nvn::executor_mutex());
    nvn::g_texture_pools.take(pool);
}
void nvnTexturePoolRegisterTexture(void* pool, int id, const void* texture, const TextureViewDesc* view) {
    auto p = nvn::g_texture_pools.acquire(pool);
    if (!p || id < 0 || id >= p->count) return;
    const auto owner = nvn::g_textures.acquire(texture);
    const auto entry = nvn::TexturePoolEntry{owner ? owner->ref : nvn::ResourceRef<nvn::Texture>{}, view ? *view : TextureViewDesc{}};
    p->register_texture(id, entry);
    {  // DEPURACIÓN: ¿limita el juego los niveles al registrar (streaming)?
        static std::atomic<unsigned> with_levels{}, total{};
        const unsigned registered = total.fetch_add(1, std::memory_order_relaxed) + 1;
        const nvn::Texture* t = owner.get();
        if (view && (view->flags & nvn::kViewLevels) && t && t->desc.levels > 1) {
            if (with_levels.fetch_add(1, std::memory_order_relaxed) < 30)
                NVN_LOG("registro con niveles: textura %llu %dx%d formato 0x%x niveles %d -> vista base %d cuenta %d (id %d)\n",
                        static_cast<unsigned long long>(t->serial), t->desc.width, t->desc.height, t->desc.format, t->desc.levels,
                        view->base_level, view->levels, id);
        }
        if (registered % 5000 == 0) NVN_LOG("registros de texturas: %u, con niveles %u\n", registered,
                                            with_levels.load(std::memory_order_relaxed));
    }
}
void nvnTexturePoolRegisterImage(void* pool, int id, const void* texture, const TextureViewDesc* view) {
    nvnTexturePoolRegisterTexture(pool, id, texture, view);
}

void nvnSamplerPoolFinalize(void* pool) {
    std::lock_guard lock(nvn::executor_mutex());
    if (auto old = nvn::g_sampler_pools.take(pool)) old->retire_sampler_builders();
}
nvn::Boolean nvnSamplerPoolInitialize(void* pool, void* memory_pool, int64_t offset, int count) {
    std::lock_guard lock(nvn::executor_mutex());
    nvnSamplerPoolFinalize(pool);
    nvn::DescriptorPool& p = nvn::g_sampler_pools.create(pool);
    p.memory_pool = memory_pool;
    p.offset = offset;
    p.count = count;
    p.samplers.resize(count);
    return 1;
}
void nvnSamplerPoolRegisterSampler(void* pool, int id, const void* sampler) {
    auto p = nvn::g_sampler_pools.acquire(pool);
    if (!p || id < 0 || id >= p->count) return;
    p->register_sampler(id, g_samplers.reference(sampler));
}
void nvnSamplerPoolRegisterSamplerBuilder(void* pool, int id, const SamplerDesc* b) {
    std::lock_guard lock(nvn::executor_mutex());
    nvn::DescriptorPool* p = nvn::g_sampler_pools.get(pool);
    if (!p || !b || id < 0 || id >= p->count) return;
    p->register_sampler_builder(id, *b);
}
void* nvnTexturePoolGetMemoryPool(const void* pool) {
    nvn::DescriptorPool* p = nvn::g_texture_pools.get(pool);
    return p ? p->memory_pool : nullptr;
}
int64_t nvnTexturePoolGetMemoryOffset(const void* pool) {
    nvn::DescriptorPool* p = nvn::g_texture_pools.get(pool);
    return p ? p->offset : 0;
}
int nvnTexturePoolGetSize(const void* pool) {
    nvn::DescriptorPool* p = nvn::g_texture_pools.get(pool);
    return p ? p->count : 0;
}
void* nvnSamplerPoolGetMemoryPool(const void* pool) {
    nvn::DescriptorPool* p = nvn::g_sampler_pools.get(pool);
    return p ? p->memory_pool : nullptr;
}
int64_t nvnSamplerPoolGetMemoryOffset(const void* pool) {
    nvn::DescriptorPool* p = nvn::g_sampler_pools.get(pool);
    return p ? p->offset : 0;
}
int nvnSamplerPoolGetSize(const void* pool) {
    nvn::DescriptorPool* p = nvn::g_sampler_pools.get(pool);
    return p ? p->count : 0;
}
void nvnTexturePoolSetDebugLabel(void*, const char*) {}
void nvnSamplerPoolSetDebugLabel(void*, const char*) {}

// --- Handles: separados = id; combinado = sampler << 32 | textura (los resuelve el ejecutor) ---

uint64_t nvnDeviceGetSeparateTextureHandle(const void*, int texture_id) { return static_cast<uint32_t>(texture_id); }
uint64_t nvnDeviceGetSeparateSamplerHandle(const void*, int sampler_id) { return static_cast<uint32_t>(sampler_id); }
uint64_t nvnDeviceGetTextureHandle(const void*, int texture_id, int sampler_id) {
    return uint64_t{static_cast<uint32_t>(sampler_id)} << 32 | static_cast<uint32_t>(texture_id);
}
uint64_t nvnDeviceGetImageHandle(const void*, int image_id) { return static_cast<uint32_t>(image_id); }
uint64_t nvnDeviceGetTexelFetchHandle(const void*, int texture_id) { return static_cast<uint32_t>(texture_id); }

// --- Samplers ---

void nvnSamplerBuilderSetDefaults(SamplerDesc* b) {
    memset(b, 0, sizeof *b);
    b->min_filter = 4;  // NEAREST_MIPMAP_LINEAR (el de OpenGL)
    b->mag_filter = 1;  // LINEAR
    b->wrap[0] = b->wrap[1] = b->wrap[2] = 1;  // REPEAT
    b->max_anisotropy = 1.0f;
    b->lod_min = -1000.0f;
    b->lod_max = 1000.0f;
    b->compare_func = 8;  // ALWAYS
}
void nvnSamplerBuilderSetDevice(SamplerDesc* b, void* device) { b->device = device; }
void nvnSamplerBuilderSetMinMagFilter(SamplerDesc* b, int min, int mag) { b->min_filter = min, b->mag_filter = mag; }
void nvnSamplerBuilderSetWrapMode(SamplerDesc* b, int s, int t, int r) { b->wrap[0] = s, b->wrap[1] = t, b->wrap[2] = r; }
void nvnSamplerBuilderSetLodClamp(SamplerDesc* b, float min, float max) {
    b->lod_min = min, b->lod_max = max;
    static int logged = 0;  // DEPURACIÓN
    if (min > 0 && logged++ < 20) NVN_LOG("sampler con LOD mínimo %.2f (máximo %.2f)\n", min, max);
}
void nvnSamplerBuilderSetLodBias(SamplerDesc* b, float bias) { b->lod_bias = bias; }
void nvnSamplerBuilderSetCompare(SamplerDesc* b, int mode, int func) { b->compare_mode = mode, b->compare_func = func; }
void nvnSamplerBuilderSetBorderColor(SamplerDesc* b, const float* color) { memcpy(b->border, color, sizeof b->border); }
void nvnSamplerBuilderSetMaxAnisotropy(SamplerDesc* b, float a) { b->max_anisotropy = a; }
void nvnSamplerBuilderSetReductionFilter(SamplerDesc* b, int filter) { b->reduction = filter; }
// Bordes enteros, guardados como float: el ejecutor solo mira si son negro, blanco o transparente (0 y 1).
void nvnSamplerBuilderSetBorderColori(SamplerDesc* b, const int* color) {
    for (int k = 0; k < 4; k++) b->border[k] = static_cast<float>(color[k]);
}
void nvnSamplerBuilderSetBorderColorui(SamplerDesc* b, const uint32_t* color) {
    for (int k = 0; k < 4; k++) b->border[k] = static_cast<float>(color[k]);
}

void nvnSamplerBuilderGetMinMagFilter(const SamplerDesc* b, int* min, int* mag) { sampler_min_mag(*b, min, mag); }
void nvnSamplerBuilderGetWrapMode(const SamplerDesc* b, int* s, int* t, int* r) { sampler_wrap(*b, s, t, r); }
void nvnSamplerBuilderGetLodClamp(const SamplerDesc* b, float* min, float* max) { sampler_lod_clamp(*b, min, max); }
float nvnSamplerBuilderGetLodBias(const SamplerDesc* b) { return b->lod_bias; }
void nvnSamplerBuilderGetCompare(const SamplerDesc* b, int* mode, int* func) { sampler_compare(*b, mode, func); }
void nvnSamplerBuilderGetBorderColor(const SamplerDesc* b, float* color) { memcpy(color, b->border, sizeof b->border); }
void nvnSamplerBuilderGetBorderColori(const SamplerDesc* b, int* color) { border_int(*b, color); }
void nvnSamplerBuilderGetBorderColorui(const SamplerDesc* b, uint32_t* color) { border_int(*b, color); }
float nvnSamplerBuilderGetMaxAnisotropy(const SamplerDesc* b) { return b->max_anisotropy; }
int nvnSamplerBuilderGetReductionFilter(const SamplerDesc* b) { return b->reduction; }
void* nvnSamplerBuilderGetDevice(const SamplerDesc* b) { return b->device; }

void nvnSamplerGetMinMagFilter(const void* s, int* min, int* mag) { sampler_min_mag(sampler_desc(s), min, mag); }
void nvnSamplerGetWrapMode(const void* s, int* ws, int* wt, int* wr) { sampler_wrap(sampler_desc(s), ws, wt, wr); }
void nvnSamplerGetLodClamp(const void* s, float* min, float* max) { sampler_lod_clamp(sampler_desc(s), min, max); }
float nvnSamplerGetLodBias(const void* s) { return sampler_desc(s).lod_bias; }
void nvnSamplerGetCompare(const void* s, int* mode, int* func) { sampler_compare(sampler_desc(s), mode, func); }
void nvnSamplerGetBorderColor(const void* s, float* color) { memcpy(color, sampler_desc(s).border, 16); }
void nvnSamplerGetBorderColori(const void* s, int* color) { border_int(sampler_desc(s), color); }
void nvnSamplerGetBorderColorui(const void* s, uint32_t* color) { border_int(sampler_desc(s), color); }
float nvnSamplerGetMaxAnisotropy(const void* s) { return sampler_desc(s).max_anisotropy; }
int nvnSamplerGetReductionFilter(const void* s) { return sampler_desc(s).reduction; }
void nvnSamplerSetDebugLabel(void*, const char*) {}

nvn::Boolean nvnSamplerInitialize(void* sampler, const SamplerDesc* b) {
    std::unique_lock<std::recursive_mutex> lock(nvn::executor_mutex(), std::defer_lock);
    nvn::lock_executor(lock);
    nvn::Sampler* s = g_samplers.get(sampler);
    if (!s) s = &g_samplers.create(sampler);
    release_sampler(*s);
    s->desc = *b;
    return 1;
}
void nvnSamplerFinalize(void* sampler) {
    std::unique_lock<std::recursive_mutex> lock(nvn::executor_mutex(), std::defer_lock);
    nvn::lock_executor(lock);
    if (auto s = g_samplers.take(sampler)) release_sampler(*s);
}

}  // extern "C"

NVN_PROC(nvnTextureBuilderSetDefaults);
NVN_PROC(nvnTextureBuilderSetDevice);
NVN_PROC(nvnTextureBuilderSetFlags);
NVN_PROC(nvnTextureBuilderSetTarget);
NVN_PROC(nvnTextureBuilderSetFormat);
NVN_PROC(nvnTextureBuilderSetLevels);
NVN_PROC(nvnTextureBuilderSetSamples);
NVN_PROC(nvnTextureBuilderSetStride);
NVN_PROC(nvnTextureBuilderSetWidth);
NVN_PROC(nvnTextureBuilderSetHeight);
NVN_PROC(nvnTextureBuilderSetDepth);
NVN_PROC(nvnTextureBuilderSetSize1D);
NVN_PROC(nvnTextureBuilderSetSize2D);
NVN_PROC(nvnTextureBuilderSetSize3D);
NVN_PROC(nvnTextureBuilderSetStorage);
NVN_PROC(nvnTextureBuilderGetStorageSize);
NVN_PROC(nvnTextureBuilderGetStorageAlignment);
NVN_PROC(nvnTextureInitialize);
NVN_PROC(nvnTextureFinalize);
NVN_PROC(nvnTextureGetWidth);
NVN_PROC(nvnTextureGetHeight);
NVN_PROC(nvnTextureGetDepth);
NVN_PROC(nvnTextureGetFormat);
NVN_PROC(nvnTextureGetTarget);
NVN_PROC(nvnTextureGetLevels);
NVN_PROC(nvnTextureGetFlags);
NVN_PROC(nvnTextureGetStorageSize);
NVN_PROC(nvnTextureGetStride);
NVN_PROC(nvnTextureGetSamples);
NVN_PROC(nvnTextureGetDepthStencilMode);
NVN_PROC(nvnTextureGetStorageClass);
NVN_PROC(nvnTextureGetRawStorageClass);
NVN_PROC(nvnTextureGetZCullStorageSize);
NVN_PROC(nvnTextureGetMemoryPool);
NVN_PROC(nvnTextureGetMemoryOffset);
NVN_PROC(nvnTextureGetTextureAddress);
NVN_PROC(nvnTextureGetSwizzle);
NVN_PROC(nvnTextureGetDebugID);
NVN_PROC(nvnTextureSetDebugLabel);
NVN_PROC(nvnTextureGetViewOffset);
NVN_PROC(nvnTextureWriteTexels);
NVN_PROC(nvnTextureWriteTexelsStrided);
NVN_PROC(nvnTextureReadTexels);
NVN_PROC(nvnTextureReadTexelsStrided);
NVN_PROC(nvnTextureFlushTexels);
NVN_PROC(nvnTextureInvalidateTexels);
NVN_PROC(nvnTextureBuilderSetDepthStencilMode);
NVN_PROC(nvnTextureBuilderSetSwizzle);
NVN_PROC(nvnTextureBuilderSetPackagedTextureLayout);
NVN_PROC(nvnTextureBuilderGetStorageClass);
NVN_PROC(nvnTextureBuilderGetRawStorageClass);
NVN_PROC(nvnTextureBuilderGetZCullStorageSize);
NVN_PROC(nvnTextureBuilderGetDevice);
NVN_PROC(nvnTextureBuilderGetFlags);
NVN_PROC(nvnTextureBuilderGetTarget);
NVN_PROC(nvnTextureBuilderGetWidth);
NVN_PROC(nvnTextureBuilderGetHeight);
NVN_PROC(nvnTextureBuilderGetDepth);
NVN_PROC(nvnTextureBuilderGetLevels);
NVN_PROC(nvnTextureBuilderGetFormat);
NVN_PROC(nvnTextureBuilderGetSamples);
NVN_PROC(nvnTextureBuilderGetStride);
NVN_PROC(nvnTextureBuilderGetMemoryPool);
NVN_PROC(nvnTextureBuilderGetMemoryOffset);
NVN_PROC(nvnTextureBuilderGetDepthStencilMode);
NVN_PROC(nvnTextureBuilderGetSwizzle);
NVN_PROC(nvnTextureViewSetDefaults);
NVN_PROC(nvnTextureViewSetLevels);
NVN_PROC(nvnTextureViewSetLayers);
NVN_PROC(nvnTextureViewSetFormat);
NVN_PROC(nvnTextureViewSetTarget);
NVN_PROC(nvnTextureViewSetDepthStencilMode);
NVN_PROC(nvnTextureViewSetSwizzle);
NVN_PROC(nvnTextureViewGetLevels);
NVN_PROC(nvnTextureViewGetLayers);
NVN_PROC(nvnTextureViewGetFormat);
NVN_PROC(nvnTextureViewGetTarget);
NVN_PROC(nvnTextureViewGetDepthStencilMode);
NVN_PROC(nvnTextureViewGetSwizzle);
NVN_PROC(nvnTexturePoolInitialize);
NVN_PROC(nvnTexturePoolFinalize);
NVN_PROC(nvnTexturePoolRegisterTexture);
NVN_PROC(nvnTexturePoolRegisterImage);
NVN_PROC(nvnSamplerPoolInitialize);
NVN_PROC(nvnSamplerPoolFinalize);
NVN_PROC(nvnSamplerPoolRegisterSampler);
NVN_PROC(nvnSamplerPoolRegisterSamplerBuilder);
NVN_PROC(nvnTexturePoolGetMemoryPool);
NVN_PROC(nvnTexturePoolGetMemoryOffset);
NVN_PROC(nvnTexturePoolGetSize);
NVN_PROC(nvnSamplerPoolGetMemoryPool);
NVN_PROC(nvnSamplerPoolGetMemoryOffset);
NVN_PROC(nvnSamplerPoolGetSize);
NVN_PROC(nvnTexturePoolSetDebugLabel);
NVN_PROC(nvnSamplerPoolSetDebugLabel);
NVN_PROC(nvnDeviceGetSeparateTextureHandle);
NVN_PROC(nvnDeviceGetSeparateSamplerHandle);
NVN_PROC(nvnDeviceGetTextureHandle);
NVN_PROC(nvnDeviceGetImageHandle);
NVN_PROC(nvnDeviceGetTexelFetchHandle);
NVN_PROC(nvnSamplerBuilderSetDefaults);
NVN_PROC(nvnSamplerBuilderSetDevice);
NVN_PROC(nvnSamplerBuilderSetMinMagFilter);
NVN_PROC(nvnSamplerBuilderSetWrapMode);
NVN_PROC(nvnSamplerBuilderSetLodClamp);
NVN_PROC(nvnSamplerBuilderSetLodBias);
NVN_PROC(nvnSamplerBuilderSetCompare);
NVN_PROC(nvnSamplerBuilderSetBorderColor);
NVN_PROC(nvnSamplerBuilderSetMaxAnisotropy);
NVN_PROC(nvnSamplerBuilderSetReductionFilter);
NVN_PROC(nvnSamplerBuilderSetBorderColori);
NVN_PROC(nvnSamplerBuilderSetBorderColorui);
NVN_PROC(nvnSamplerBuilderGetMinMagFilter);
NVN_PROC(nvnSamplerBuilderGetWrapMode);
NVN_PROC(nvnSamplerBuilderGetLodClamp);
NVN_PROC(nvnSamplerBuilderGetLodBias);
NVN_PROC(nvnSamplerBuilderGetCompare);
NVN_PROC(nvnSamplerBuilderGetBorderColor);
NVN_PROC(nvnSamplerBuilderGetBorderColori);
NVN_PROC(nvnSamplerBuilderGetBorderColorui);
NVN_PROC(nvnSamplerBuilderGetMaxAnisotropy);
NVN_PROC(nvnSamplerBuilderGetReductionFilter);
NVN_PROC(nvnSamplerBuilderGetDevice);
NVN_PROC(nvnSamplerGetMinMagFilter);
NVN_PROC(nvnSamplerGetWrapMode);
NVN_PROC(nvnSamplerGetLodClamp);
NVN_PROC(nvnSamplerGetLodBias);
NVN_PROC(nvnSamplerGetCompare);
NVN_PROC(nvnSamplerGetBorderColor);
NVN_PROC(nvnSamplerGetBorderColori);
NVN_PROC(nvnSamplerGetBorderColorui);
NVN_PROC(nvnSamplerGetMaxAnisotropy);
NVN_PROC(nvnSamplerGetReductionFilter);
NVN_PROC(nvnSamplerSetDebugLabel);
NVN_PROC(nvnSamplerInitialize);
NVN_PROC(nvnSamplerFinalize);
