// nvnMemoryPool*: la memoria la pone el juego (SetStorage). Como dirección de GPU usamos la
// misma dirección de CPU: así cualquier dirección que el juego pase en un comando se lee directo.
//
// Pools virtuales (MH Rise los usa para la memoria de texturas: trozos de 4 MB de pools físicos que mapea seguidos en
// un pool virtual con nvnMemoryPoolMapVirtual): no tienen memoria propia. Les damos una reserva de direcciones y
// MapVirtual hace que esas direcciones vean las mismas páginas que el trozo físico (el heap del juego es un memfd
// compartido, os_memory.cpp), así la CPU, las subidas del ejecutor y las texturas creadas en el pool virtual leen lo
// que el juego escribe por el físico. Lo no mapeado se lee como ceros.

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "nnsdk/watch.h"
#include "nnsdk/nvn/commands.h"
#include "nnsdk/nvn/nvn.h"
#include "nnsdk/nvn/objects.h"

namespace nnsdk {
// os_memory.cpp: [dst, dst + size) pasa a ver las páginas del heap que hay en src. false si no se puede (src fuera del
// heap, sin alinear a página o sin memfd); entonces dst no cambia.
bool alias_heap_pages(void* dst, const void* src, size_t size);
}  // namespace nnsdk

namespace {

// NVNmemoryPoolBuilder es opaco; guardamos aquí dentro solo estos campos (cabe sobrado).
struct MemoryPoolBuilder {
    void* device;
    void* storage;
    size_t size;
    int flags;
};

// NVNmappingRequest (40 bytes): main+0x7728f40 los rellena seguidos con zancada 0x28 y pone en +0x00 el pool físico (el
// que va dentro de cada trozo de 4 MB, en trozo+0x58), +0x08 el offset en él (0), +0x10 el offset en el virtual (base +
// i·4 MB), +0x18 el tamaño (4 MB) y en +0x20 la clase de almacenamiento (32 bits). main+0x7726b34 mapea partes de un
// trozo (offset físico múltiplo de 256, tamaño múltiplo de 64 KB). Sin pool físico, se desmapea.
struct MappingRequest {
    const void* physical_pool;
    int64_t physical_offset;
    int64_t virtual_offset;
    uint64_t size;
    int32_t storage_class;
    int32_t pad;
};
static_assert(sizeof(MappingRequest) == 40);

size_t page_size() {
    static const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    return page;
}

// Ceros en [p, p + size): páginas anónimas nuevas (lo que hubiera mapeado deja de verse).
bool map_zeros(void* p, size_t size) {
    if (mmap(p, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0) == MAP_FAILED)
        return false;
    nnsdk::watch::virtual_mapped(reinterpret_cast<uintptr_t>(p), size, 0);
    return true;
}

}  // namespace

nvn::Objects<nvn::MemoryPool> nvn::g_memory_pools;

bool nvn::MemoryPool::covers(size_t offset, size_t bytes) const {
    if (offset > size || bytes > size - offset) return false;
    if (!reserved || !bytes) return true;
    const size_t first = offset / page_size(), stop = (offset + bytes - 1) / page_size() + 1;
    std::lock_guard lock(mapping_mutex);
    return std::find(backed_pages.begin() + first, backed_pages.begin() + stop, false) == backed_pages.begin() + stop;
}

bool nvn::MemoryPool::has_cpu_backing(size_t offset, size_t bytes) const {
    if (!bytes || offset >= size) return false;
    const size_t end = offset + std::min(bytes, size - offset);
    std::lock_guard lock(mapping_mutex);
    const auto stop = cpu_readable_pages.begin() + (end - 1) / page_size() + 1;
    return std::find(cpu_readable_pages.begin() + offset / page_size(), stop, true) != stop;
}

void nvn::MemoryPool::register_buffer(size_t offset, size_t length) {
    std::lock_guard lock(mapping_mutex);
    buffer_ranges.emplace(offset, length);
    max_buffer_size = std::max(max_buffer_size, length);
    buffer_epoch.fetch_add(1, std::memory_order_release);
}

void nvn::MemoryPool::unregister_buffer(size_t offset, size_t length) {
    std::lock_guard lock(mapping_mutex);
    auto [first, last] = buffer_ranges.equal_range(offset);
    for (; first != last; ++first) {
        if (first->second != length) continue;
        buffer_ranges.erase(first);
        buffer_epoch.fetch_add(1, std::memory_order_release);
        break;
    }
}

bool nvn::MemoryPool::overlaps_buffer(size_t offset, size_t length) const {
    std::lock_guard lock(mapping_mutex);
    auto it = buffer_ranges.lower_bound(offset - std::min(offset, max_buffer_size));
    for (; it != buffer_ranges.end() && it->first < offset + length; ++it)
        if (it->first + it->second > offset) return true;
    return false;
}

extern "C" {

void nvnMemoryPoolBuilderSetDefaults(MemoryPoolBuilder* b) { memset(b, 0, sizeof *b); }
void nvnMemoryPoolBuilderSetDevice(MemoryPoolBuilder* b, void* device) { b->device = device; }
void nvnMemoryPoolBuilderSetStorage(MemoryPoolBuilder* b, void* storage, size_t size) {
    b->storage = storage;
    b->size = size;
}
void nvnMemoryPoolBuilderSetFlags(MemoryPoolBuilder* b, int flags) { b->flags = flags; }
void* nvnMemoryPoolBuilderGetMemory(const MemoryPoolBuilder* b) { return b->storage; }
size_t nvnMemoryPoolBuilderGetSize(const MemoryPoolBuilder* b) { return b->size; }
int nvnMemoryPoolBuilderGetFlags(const MemoryPoolBuilder* b) { return b->flags; }

void* nvnMemoryPoolBuilderGetDevice(const MemoryPoolBuilder* b) { return b->device; }

nvn::Boolean nvnMemoryPoolInitialize(void* pool, const MemoryPoolBuilder* b) {
    uint8_t* storage = static_cast<uint8_t*>(b->storage);
    bool reserved = false;
    // Virtual (o sin memoria, que en NVN solo vale para los virtuales): direcciones propias, a cero hasta MapVirtual.
    if ((b->flags & nvn::kPoolFlagVirtual) || (!storage && b->size)) {
        const size_t size = (b->size + page_size() - 1) & ~(page_size() - 1);
        void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) {
            NVN_LOG("memory pool virtual %p de %zu MB: sin direcciones\n", pool, b->size >> 20);
            return 0;
        }
        storage = static_cast<uint8_t*>(p);
        reserved = true;
        nnsdk::watch::add_virtual_range(reinterpret_cast<uintptr_t>(p), size);
    }
    static int logged = 0;  // DEPURACIÓN: los pools físicos y virtuales (para confirmar sus flags)
    if (((b->flags & (nvn::kPoolFlagPhysical | nvn::kPoolFlagVirtual)) || reserved) && logged++ < 12)
        NVN_LOG("memory pool %p %s: %zu KB, flags 0x%x, memoria %p%s\n", pool,
                reserved ? "virtual" : (b->flags & nvn::kPoolFlagPhysical) ? "físico" : "", b->size >> 10, b->flags,
                static_cast<void*>(storage), reserved && b->storage ? " (la del juego no se usa)" : "");
    nvn::MemoryPool& p = nvn::g_memory_pools.create(pool);
    p.storage = storage;
    p.size = b->size;
    p.flags = b->flags;
    p.reserved = reserved;
    if (reserved) {
        p.backed_pages.resize((p.size - 1) / page_size() + 1);
        p.cpu_readable_pages.resize(p.backed_pages.size());
    }
    return 1;
}
void nvnMemoryPoolFinalize(void* pool) {
    auto* current = nvn::g_memory_pools.get(pool);
    if (current && (current->flags & nvn::kPoolFlagPhysical) &&
        !nvn::before_guest_texture_access(current->storage, current->size)) return;
    if (current && current->reserved) {
        nvn::wait_idle();
        if (!nvn::publish_virtual_pool(pool)) return;
    }
    if (std::unique_ptr<nvn::MemoryPool> p = nvn::g_memory_pools.take(pool); p && p->reserved) {
        munmap(p->storage, (p->size + page_size() - 1) & ~(page_size() - 1));
    }
}

// Cada petición hace que [virtual + offset, + size) vea [físico + offset, + size), o ceros si no hay pool físico. Si
// algo no se puede ver (no está en el heap o sin alinear a página), se deja a cero y se avisa: esas direcciones siguen
// siendo válidas y se contesta que sí (solo una petición fuera del pool virtual da NVN_FALSE).
nvn::Boolean nvnMemoryPoolMapVirtual(void* pool, int count, const MappingRequest* requests) {
    nvn::MemoryPool* v = nvn::g_memory_pools.get(pool);
    if (!v || !v->reserved || count < 0 || (count && !requests)) {
        NVN_LOG("MapVirtual sobre %p, que no es un pool virtual\n", pool);
        return 0;
    }
    static int logged = 0, failed = 0;
    bool ok = true;
    std::vector<std::pair<size_t, size_t>> changed;
    // A new virtual address can import a physical page last written through a different pool.
    std::lock_guard executor(nvn::executor_mutex());
    if (count && !nvn::publish_virtual_pool(nullptr)) return 0;
    // Publish the new pages and their notifications together with respect to GPU-to-guest readback.
    std::unique_lock mapping(v->mapping_mutex);
    while (v->gpu_dirty) {
        mapping.unlock();
        if (!nvn::publish_virtual_pool(pool)) return 0;
        mapping.lock();
    }
    for (int k = 0; k < count; k++) {
        const MappingRequest& r = requests[k];
        if (r.virtual_offset < 0 || r.size > v->size || static_cast<uint64_t>(r.virtual_offset) > v->size - r.size) {
            NVN_LOG("MapVirtual: +0x%llx (%llu KB) fuera del pool virtual de %zu KB\n", static_cast<unsigned long long>(r.virtual_offset),
                    static_cast<unsigned long long>(r.size >> 10), v->size >> 10);
            ok = false;
            continue;
        }
        uint8_t* dst = v->storage + r.virtual_offset;
        const nvn::MemoryPool* phys = r.physical_pool ? nvn::g_memory_pools.get(r.physical_pool) : nullptr;
        bool mapped = false;
        if (!r.physical_pool) {
            mapped = map_zeros(dst, r.size);
        } else if (phys && r.physical_offset >= 0 && r.size <= phys->size &&
                   static_cast<uint64_t>(r.physical_offset) <= phys->size - r.size) {
            mapped = nnsdk::alias_heap_pages(dst, phys->storage + r.physical_offset, r.size);
        }
        bool replaced = mapped;
        if (!mapped) {
            replaced = map_zeros(dst, r.size);
            if (r.physical_pool && failed++ < 16)
                NVN_LOG("MapVirtual: no se puede ver %p+0x%llx (pool %s) en +0x%llx (%llu KB): se queda a cero\n",
                        phys ? static_cast<void*>(phys->storage) : nullptr, static_cast<unsigned long long>(r.physical_offset),
                        phys ? "fuera del heap o sin alinear" : "desconocido", static_cast<unsigned long long>(r.virtual_offset),
                        static_cast<unsigned long long>(r.size >> 10));
        }
        if (replaced && r.size) {
            const size_t begin = r.virtual_offset, end = begin + r.size;
            std::fill(v->backed_pages.begin() + begin / page_size(), v->backed_pages.begin() + (end - 1) / page_size() + 1,
                      mapped && r.physical_pool);
            std::fill(v->cpu_readable_pages.begin() + begin / page_size(), v->cpu_readable_pages.begin() + (end - 1) / page_size() + 1,
                      mapped && phys && !(phys->flags & nvn::kPoolFlagCpuNoAccess) && r.storage_class != nvn::kStorageClassNativeImage);
            if (!changed.empty() && changed.back().second == begin)
                changed.back().second = end;
            else
                changed.emplace_back(begin, end);
        }
        if (logged < 24) {  // DEPURACIÓN: las primeras (las del arranque: 6, 8 y 1 peticiones)
            logged++;
            NVN_LOG("MapVirtual %p: +0x%llx <- %p+0x%llx (%llu KB, clase 0x%x)%s\n", pool,
                    static_cast<unsigned long long>(r.virtual_offset), r.physical_pool,
                    static_cast<unsigned long long>(r.physical_offset), static_cast<unsigned long long>(r.size >> 10),
                    r.storage_class, mapped ? "" : " FALLO");
        }
    }
    if (!changed.empty()) {
        nvn::g_textures.for_each([&](nvn::Texture& t) {
            if (t.desc.pool != pool || t.desc.offset < 0) return;
            const size_t start = t.desc.offset;
            if (start >= v->size) return;
            const size_t end = start + std::min(t.layout.total_size(), v->size - start);
            if (start == end) return;
            for (auto [first, last] : changed) {
                if (first >= end || start >= last) continue;
                t.note_mapping_change(std::max(first, start) - start, std::min(last, end) - start);
            }
        });
    }
    return ok ? 1 : 0;
}

void nvnMemoryPoolSetDebugLabel(void*, const char*) {}

void* nvnMemoryPoolMap(const void* pool) {
    nvn::MemoryPool* p = nvn::g_memory_pools.get(pool);
    if (p && p->reserved) __atomic_store_n(&p->mapped_by_cpu, true, __ATOMIC_RELEASE);
    if (p && (p->reserved || (p->flags & nvn::kPoolFlagPhysical)) && !nvn::publish_virtual_pool(nullptr)) return nullptr;
    return p ? p->storage : nullptr;
}
uint64_t nvnMemoryPoolGetBufferAddress(const void* pool) {
    const auto* p = nvn::g_memory_pools.get(pool);
    return p ? reinterpret_cast<uintptr_t>(p->storage) : 0;
}
size_t nvnMemoryPoolGetSize(const void* pool) {
    nvn::MemoryPool* p = nvn::g_memory_pools.get(pool);
    return p ? p->size : 0;
}
int nvnMemoryPoolGetFlags(const void* pool) {
    nvn::MemoryPool* p = nvn::g_memory_pools.get(pool);
    return p ? p->flags : 0;
}
// La GPU lee la memoria del juego al ejecutar los comandos: no hay cachés que sincronizar.
void nvnMemoryPoolFlushMappedRange(const void*, int64_t, size_t) {}
void nvnMemoryPoolInvalidateMappedRange(const void* pool, int64_t, size_t size) {
    const auto* p = nvn::g_memory_pools.get(pool);
    if (size && p && (p->reserved || (p->flags & nvn::kPoolFlagPhysical))) nvn::publish_virtual_pool(nullptr);
}

}  // extern "C"

NVN_PROC(nvnMemoryPoolBuilderSetDefaults);
NVN_PROC(nvnMemoryPoolBuilderSetDevice);
NVN_PROC(nvnMemoryPoolBuilderSetStorage);
NVN_PROC(nvnMemoryPoolBuilderSetFlags);
NVN_PROC(nvnMemoryPoolBuilderGetMemory);
NVN_PROC(nvnMemoryPoolBuilderGetSize);
NVN_PROC(nvnMemoryPoolBuilderGetFlags);
NVN_PROC(nvnMemoryPoolBuilderGetDevice);
NVN_PROC(nvnMemoryPoolInitialize);
NVN_PROC(nvnMemoryPoolMapVirtual);
NVN_PROC(nvnMemoryPoolSetDebugLabel);
NVN_PROC(nvnMemoryPoolFinalize);
NVN_PROC(nvnMemoryPoolMap);
NVN_PROC(nvnMemoryPoolGetBufferAddress);
NVN_PROC(nvnMemoryPoolGetSize);
NVN_PROC(nvnMemoryPoolGetFlags);
NVN_PROC(nvnMemoryPoolFlushMappedRange);
NVN_PROC(nvnMemoryPoolInvalidateMappedRange);
