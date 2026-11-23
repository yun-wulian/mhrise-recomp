// Android offscreen regression fixture. Compile as a replacement executor translation unit
// with -fno-access-control in a test-only libnnsdk, then invoke the exported test entry.
#include "../../runtime/src/nnsdk/nvn/executor.cpp"
#include "nnsdk/types.h"

namespace nn::os {
Result SetMemoryHeapSize(size_t);
Result AllocateMemoryBlock(uintptr_t*, size_t);
void FreeMemoryBlock(uintptr_t, size_t);
}

struct TestPoolBuilder { void* device; void* storage; size_t size; int flags; };
struct TestMapping { const void* pool; int64_t physical, virt; uint64_t size; int32_t storage_class, pad; };
struct TestBufferBuilder { void* device; void* pool; int64_t offset; size_t size; };
extern "C" {
int nvnMemoryPoolInitialize(void*, const TestPoolBuilder*);
void nvnMemoryPoolFinalize(void*);
int nvnMemoryPoolMapVirtual(void*, int, const TestMapping*);
void nvnTextureBuilderSetDefaults(nvn::TextureDesc*);
int nvnTextureBuilderGetStorageClass(const nvn::TextureDesc*);
int nvnTextureBuilderGetRawStorageClass(const nvn::TextureDesc*);
uint64_t nvnTextureGetTextureAddress(const void*);
int nvnBufferInitialize(void*, const TestBufferBuilder*);
void* nvnBufferMap(const void*);
void nvnBufferFinalize(void*);
int nvnTextureInitialize(void*, const nvn::TextureDesc*);
void nvnTextureFinalize(void*);
}

static bool test_virtual_page_replacement(auto& executor, const auto& read_byte) {
    using namespace nvn;
    using Prior = typename std::remove_reference_t<decltype(executor)>::PriorContents;
    constexpr size_t tile = 64 * 1024;
    uintptr_t heap = 0;
    constexpr size_t pool_size = 16 * tile;
    if (nn::os::SetMemoryHeapSize(4 << 20).value || nn::os::AllocateMemoryBlock(&heap, pool_size).value) return false;
    auto* bytes = reinterpret_cast<uint8_t*>(heap);
    memset(bytes, 17, tile);
    memset(bytes + tile, 42, tile);
    memset(bytes + 2 * tile, 73, tile);
    memset(bytes + 3 * tile, 0, 3 * tile);
    memcpy(bytes + 6 * tile, bytes + 2 * tile, 4 * tile);
    memset(bytes + 10 * tile, 0x19, 6 * tile);
    int phys, virt, handles[2];
    TestPoolBuilder backing{nullptr, bytes, pool_size, kPoolFlagPhysical | 0x82};
    TestPoolBuilder virtual_pool{nullptr, nullptr, pool_size, kPoolFlagVirtual | 0xa1};
    if (!nvnMemoryPoolInitialize(&phys, &backing) || !nvnMemoryPoolInitialize(&virt, &virtual_pool)) return false;
    TestMapping initial{&phys, 0, 0, pool_size, 0xfe};
    if (!nvnMemoryPoolMapVirtual(&virt, 1, &initial)) return false;
    TextureDesc desc;
    nvnTextureBuilderSetDefaults(&desc);
    desc.width = desc.height = 128;
    desc.format = 0x25;  // RGBA8, exactly one 64 KiB tile per texture
    desc.flags = 0x20;
    desc.pool = &virt;
    Texture* textures[2];
    for (int i = 0; i < 2; ++i) {
        desc.offset = i * tile;
        if (!nvnTextureInitialize(&handles[i], &desc)) return false;
        textures[i] = g_textures.get(&handles[i]);
        if (!executor.ensure_image(*textures[i], executor.cmd_)) return false;
        if (!executor.prepare_contents(*textures[i], Prior::Preserve)) return false;
        if (read_byte(*textures[i], VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 0) != (i ? 42 : 17)) return false;
    }
    // Previous GPU work is complete. The next submission still belongs to the same frame.
    TestMapping replacement{&phys, 2 * tile, 0, tile, 0xfe};
    const auto other_version = textures[1]->content_version;
    if (!nvnMemoryPoolMapVirtual(&virt, 1, &replacement) ||
        !executor.prepare_contents(*textures[0], Prior::Preserve) ||
        !executor.prepare_contents(*textures[1], Prior::Preserve) ||
        other_version != textures[1]->content_version) return false;
    const bool visible = read_byte(*textures[0], VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 0) == 73 &&
                         read_byte(*textures[1], VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 0) == 42;
    int wide_handle;
    desc.offset = 2 * tile;
    desc.width = desc.height = 256;
    if (!nvnTextureInitialize(&wide_handle, &desc)) return false;
    auto& wide = *g_textures.get(&wide_handle);
    if (wide.layout.total_size() != 4 * tile || !executor.ensure_image(wide, executor.cmd_)) return false;
    if (!executor.prepare_contents(wide, Prior::Preserve)) return false;
    if (read_byte(wide, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 0) != 73) return false;
    TestMapping first{&phys, 2 * tile, 2 * tile, tile, 0xfe};
    TestMapping last{&phys, 5 * tile, 5 * tile, tile, 0xfe};
    TestMapping edges[]{last, first, last};
    TestMapping bridge{&phys, 3 * tile, 3 * tile, 2 * tile, 0xfe};
    auto checked = [&](const TestMapping* requests, int count, size_t expected) {
        const uint64_t before = executor.prof_full_hash_bytes_;
        const auto version = wide.content_version;
        return nvnMemoryPoolMapVirtual(&virt, count, requests) &&
               executor.prepare_contents(wide, Prior::Preserve) &&
               executor.prepare_contents(wide, Prior::Preserve) &&
               wide.content_version == version && executor.prof_full_hash_bytes_ - before == expected;
    };
    if (!checked(edges, 3, 2 * tile)) return false;
    TestMapping joined[]{last, first, bridge};
    if (!checked(joined, 3, 4 * tile)) return false;
    // Equal bytes, different residency: both dropping and restoring a zero page change the view calculation.
    TestMapping hole{nullptr, 0, 5 * tile, tile, 0xfe};
    if (!nvnMemoryPoolMapVirtual(&virt, 1, &hole) || !executor.prepare_contents(wide, Prior::Preserve) ||
        wide.residency.issues.empty() || !(wide.residency.issues[0] & MipResidency::Unmapped)) return false;
    if (read_byte(wide, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 0) != 73 ||
        !nvnMemoryPoolMapVirtual(&virt, 1, &last) || !executor.prepare_contents(wide, Prior::Preserve) ||
        (!wide.residency.issues.empty() && (wide.residency.issues[0] & MipResidency::Unmapped))) return false;
    cmd::ClearTexture white{&wide, {}, {0, 0, 0, 256, 256, 1},
                            {0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000}, 15};
    executor.clear_texture(white);
    TestMapping fresh{&phys, 6 * tile, 2 * tile, tile, 0xfe};
    if (read_byte(wide, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 0) != 255 ||
        !nvnMemoryPoolMapVirtual(&virt, 1, &fresh)) return false;
    executor.begin();
    Staging unavailable;
    unavailable.failed_ = true;
    auto* previous = executor.staging_;
    executor.staging_ = &unavailable;
    const bool premature = executor.prepare_contents(wide, Prior::Preserve);
    executor.staging_ = previous;
    if (premature || !executor.prepare_contents(wide, Prior::Preserve) ||
        read_byte(wide, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 0) != 73 ||
        read_byte(wide, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 240) != 255) return false;

    // Fresh pages with exactly the guest digest from before the GPU clear.
    TestMapping whole{&phys, 6 * tile, 2 * tile, 4 * tile, 0xfe};
    if (!nvnMemoryPoolMapVirtual(&virt, 1, &whole) || !executor.prepare_contents(wide, Prior::Preserve) ||
        read_byte(wide, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 0) != 73 ||
        read_byte(wide, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 240) != 0) return false;
    whole.physical = 2 * tile;
    if (!nvnMemoryPoolMapVirtual(&virt, 1, &whole) || !executor.prepare_contents(wide, Prior::Preserve) ||
        read_byte(wide, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 0) != 255 ||
        read_byte(wide, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 240) != 255) return false;
    int array_handle;
    desc.offset = 0;
    desc.width = desc.height = 128;
    desc.depth = 2;
    desc.target = kTarget2DArray;
    if (!nvnTextureInitialize(&array_handle, &desc)) return false;
    auto& array = *g_textures.get(&array_handle);
    white.texture = &array;
    white.region = {0, 0, 0, 128, 128, 1};
    executor.clear_texture(white);
    if (read_byte(array, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 0) != 255) return false;
    TestMapping sibling{&phys, 6 * tile, tile, tile, 0xfe};
    if (!nvnMemoryPoolMapVirtual(&virt, 1, &sibling)) return false;
    executor.begin();
    if (!executor.prepare_contents(array, Prior::Preserve) ||
        read_byte(array, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0) != 73 ||
        read_byte(array, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 0) != 255) return false;

    // Two virtual pools, disjoint layers and different mips in one guest submission.
    int second_pool, second_handle;
    virtual_pool.size = 6 * tile;
    TestMapping pages{&phys, 10 * tile, 0, 6 * tile, 0xfe};
    if (!nvnMemoryPoolInitialize(&second_pool, &virtual_pool) || !nvnMemoryPoolMapVirtual(&second_pool, 1, &pages)) return false;
    desc.pool = &second_pool;
    desc.depth = 3;
    desc.levels = 2;
    if (!nvnTextureInitialize(&second_handle, &desc)) return false;
    auto& second = *g_textures.get(&second_handle);
    std::vector<uint8_t> expected(6 * tile, 0x19);
    CommandList commands;
    white.view.flags = kViewLayers | kViewLevels;
    white.view.layers = white.view.levels = 1;
    white.view.base_layer = 1;
    commands.commands.push_back(white);
    auto add = [&](uint32_t layer, uint32_t mip, unsigned channel) {
        auto clear = white;
        clear.texture = &second;
        clear.view.base_layer = layer;
        clear.view.base_level = mip;
        const auto& lv = second.layout.levels[mip];
        clear.region.width = lv.width;
        clear.region.height = lv.height;
        for (unsigned i = 0; i < 4; ++i) clear.color[i] = i == channel ? 0x3f800000 : 0;
        commands.commands.push_back(clear);
        const size_t offset = layer * second.layout.layer_size + lv.offset;
        if (lv.size != size_t{lv.width} * lv.height * 4 || offset + lv.size > expected.size()) return false;
        for (size_t i = 0; i < lv.size; ++i) expected[offset + i] = i % 4 == channel ? 255 : 0;
        return true;
    };
    if (!add(0, 0, 0) || !add(2, 0, 1) || !add(2, 1, 2)) return false;
    commands.commands.emplace_back(cmd::FenceSync{nullptr, 1}); // make physical pages visible to the host
    const auto before = executor.last_submit();
    executor.run({&commands});
    if (executor.last_submit() - before != 1 || memcmp(bytes + 10 * tile, expected.data(), expected.size()) ||
        !std::all_of(bytes + 6 * tile, bytes + 7 * tile, [](uint8_t x) { return x == 255; })) return false;
    // The mapping stays pinned if staging cannot hold the first copy. Retrying must publish the pending GPU bytes.
    executor.begin();
    auto retry_clear = white;
    retry_clear.color[0] = 0;
    executor.clear_texture(retry_clear);
    auto& staging = executor.inflight_[executor.cur_].texel_readback;
    staging.trim(0);
    staging.failed_ = true;
    if (nvnMemoryPoolMapVirtual(&virt, 1, &sibling) || bytes[6 * tile] != 255 ||
        !nvnMemoryPoolMapVirtual(&virt, 1, &sibling) || bytes[6 * tile] != 0 || bytes[6 * tile + 1] != 255) return false;
    nvnTextureFinalize(&second_handle);
    nvnMemoryPoolFinalize(&second_pool);
    nvnTextureFinalize(&array_handle);
    nvnTextureFinalize(&wide_handle);
    for (int& handle : handles) nvnTextureFinalize(&handle);
    // Use the same allocation and shader-visible image with two distinct mapping contracts.
    nvnTextureBuilderSetDefaults(&desc);
    desc.pool = &virt;
    desc.width = desc.height = 128;
    desc.format = 0x25;
    desc.flags = kTextureFlagCompressible;
    const int native = nvnTextureBuilderGetStorageClass(&desc);
    const int raw = nvnTextureBuilderGetRawStorageClass(&desc);
    if (native == raw) return false;
    TestMapping native_page{&phys, 0, 0, tile, native};
    TestMapping raw_page{&phys, tile, 0, tile, raw};
    if (!nvnMemoryPoolMapVirtual(&virt, 1, &native_page) || !nvnTextureInitialize(&handles[0], &desc)) return false;
    cmd::ClearTexture red{g_textures.get(&handles[0]), {}, {0, 0, 0, 128, 128, 1}, {0x3f800000, 0, 0, 0x3f800000}, 15};
    CommandList fenced;
    fenced.commands = {red, cmd::FenceSync{nullptr, 1}};
    if (!executor.run({&fenced}) || !executor.virtual_pending_.load() ||
        !nvnMemoryPoolMapVirtual(&virt, 1, &raw_page) || bytes[0] != 255 || bytes[tile-2] != 0) return false;
    if (!executor.run({&fenced}) || executor.virtual_pending_.load() || bytes[tile] != 255 || bytes[2*tile-2] != 0) return false;
    if (!nvnMemoryPoolMapVirtual(&virt, 1, &native_page) || !executor.run({&fenced}) || !executor.virtual_pending_.load()) return false;
    int raw_buffer;
    TestBufferBuilder raw_view{nullptr, &virt, 0, tile};
    if (!nvnBufferInitialize(&raw_buffer, &raw_view) || executor.virtual_pending_.load()) return false;
    const void* exposed = nvnBufferMap(&raw_buffer);
    if (!exposed || memcmp(exposed, bytes, tile) || !executor.run({&fenced}) || executor.virtual_pending_.load()) return false;
    nvnBufferFinalize(&raw_buffer);
    TestMapping alias_page{&phys, 0, tile, tile, native};
    desc.offset = tile;
    if (!nvnMemoryPoolMapVirtual(&virt, 1, &alias_page) || !nvnTextureInitialize(&handles[1], &desc)) return false;
    auto blue = red;
    blue.texture = g_textures.get(&handles[1]);
    blue.color[0] = 0;
    blue.color[2] = 0x3f800000;
    for (bool red_last : {true, false}) {
        CommandList aliases;
        aliases.commands = {red_last ? blue : red, red_last ? red : blue, cmd::FenceSync{nullptr, 1}};
        if (!executor.run({&aliases}) || !executor.virtual_pending_.load() || !publish_virtual_pool(nullptr)) return false;
        for (size_t i = 0; i < tile; i += 4)
            if (bytes[i] != (red_last ? 255 : 0) || bytes[i + 2] != (red_last ? 0 : 255)) return false;
    }
    if (!executor.run({&fenced}) || !executor.virtual_pending_.load() ||
        !nvnTextureGetTextureAddress(&handles[0]) || executor.virtual_pending_.load()) return false;
    nvnTextureFinalize(&handles[1]);
    nvnTextureFinalize(&handles[0]);
    nvnMemoryPoolFinalize(&virt);
    nvnMemoryPoolFinalize(&phys);
    nn::os::FreeMemoryBlock(heap, pool_size);
    executor.begin();
    return visible;
}

static bool test_watched_handle_table(auto& e, nvn::Texture& source) {
    using namespace nvn;
    constexpr size_t allocation = 65536;
    uintptr_t heap = 0;
    if (nn::os::AllocateMemoryBlock(&heap, allocation).value) return false;
    std::array<uint32_t, 64> reference{};
    if (!nnsdk::watch::watchable(heap, sizeof reference)) return false;
    auto* watched = reinterpret_cast<uint32_t*>(heap);
    for (uint32_t i = 0; i < reference.size(); ++i) reference[i] = (i << 20) | (i + 8);
    reference[8] = (1u << 20) | 3;
    reference[9] = UINT32_MAX;
    memcpy(watched, reference.data(), sizeof reference);

    ProgramStage fs{};
    fs.shader.has_bindless_set = true;
    auto& binding = fs.shader.resources.emplace_back();
    binding.kind = Resource::Texture;
    binding.texture_type = 2;
    binding.set = nvnshader::kBindlessSet;
    binding.cbuf_index = kUniformBankBase;
    binding.count = reference.size();
    binding.size_shift = 2;
    const ProgramStage* stages[kStages]{};
    constexpr int fragment = static_cast<int>(Stage::Fragment);
    stages[fragment] = &fs;
    DescriptorPool pool;
    pool.textures.resize(8);
    pool.samplers.resize(2);
    auto& uniform = e.st_.bindings[fragment].uniforms[0];
    const auto saved_uniform = uniform;
    const auto saved_textures = e.st_.texture_pool, saved_samplers = e.st_.sampler_pool;
    e.st_.texture_pool = e.st_.sampler_pool = &pool;
    auto checked_set = [&]() -> VkDescriptorSet {
        uniform = {heap, sizeof reference};
        auto set = e.bindless_set(stages);
        uniform.address = reinterpret_cast<uintptr_t>(reference.data());
        return set == e.bindless_set(stages) ? set : VK_NULL_HANDLE;
    };
    const auto empty = checked_set();
    ResourceSlots<Texture> slots;
    const auto resource = slots.insert(&source);
    pool.register_texture(3, {resource, {}});
    const auto with_image = checked_set();
    Sampler sample{};
    sample.desc.max_anisotropy = 1;
    ResourceSlots<Sampler> sampler_slots;
    auto sampler_ref = sampler_slots.insert(&sample);
    pool.register_sampler(1, sampler_ref);
    const auto both = checked_set();
    const auto hits = e.prof_bindless_memo_;
    if (both != e.bindless_set(stages) ||
        e.prof_bindless_memo_ - hits != (NVN_FLAG("no_bindless_memo") ? 0u : 1u)) return false;
    sampler_slots.erase(sampler_ref);
    if (checked_set() != with_image) return false;
    pool.register_sampler(1, &sample);
    if (checked_set() != both) return false;
    slots.erase(resource);  // Same pool snapshot, different resource generation.
    const auto only_sampler = checked_set();
    pool.register_sampler(1, {});
    bool ok = empty && with_image && with_image != empty && both && both != with_image &&
              only_sampler && only_sampler != both && only_sampler != empty && checked_set() == empty;

    pool.register_texture(6, {&source, {}});
    pool.register_sampler(1, &sample);
    binding.handle_and_mask = 7;
    binding.shift_left = 1;
    binding.handle_or = 1u << 20;
    const auto transformed = checked_set();
    binding.shift_left = 0;
    const auto shifted = checked_set();
    // Reuse the same pages and pool with a different interpretation, then dirty one word.
    watched[9] = reference[9] = 6;
    const auto dirty = checked_set();
    ok &= transformed && shifted && shifted != transformed && dirty && dirty != shifted;
    e.st_.texture_pool = saved_textures;
    e.st_.sampler_pool = saved_samplers;
    uniform = saved_uniform;
    for (auto handle : {sample.vk, sample.vk_af, sample.vk_lit}) if (handle) e.retire_sampler(handle);
    nn::os::FreeMemoryBlock(heap, allocation);
    return ok;
}

extern "C" int test_texture_sync() {
    using namespace nvn;
    auto& e = g_executor;
    if (!e.init()) return 1;
    e.begin();
    e.watch_textures_ = false;
    Texture source{};
    source.serial = next_texture_serial();
    source.desc.target = kTarget2D;
    source.desc.format = 0x25;
    source.desc.width = source.desc.height = 256;
    source.desc.depth = source.desc.levels = 1;
    source.layout = compute_layout(256, 256, 1, 1, 1, 4, 1, 1, true, 1024, false);
    std::vector<uint8_t> bytes(source.layout.total_size(), 32);
    source.storage = bytes.data();
    if (!e.ensure_image(source, e.cmd_)) return 2;
    e.upload(source);

    auto read_byte = [&](Texture& t, VkImageAspectFlags aspect, uint32_t level, uint32_t layer, int x) {
        auto staging = e.staging_->alloc(16, 16);
        VkBufferImageCopy r{};
        r.bufferOffset = staging.offset;
        r.imageSubresource = {aspect, level, layer, 1};
        r.imageOffset.x = x;
        r.imageExtent = {1, 1, 1};
        vkCmdCopyImageToBuffer(e.cmd_, t.image, VK_IMAGE_LAYOUT_GENERAL, staging.buffer, 1, &r);
        VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(e.cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                             1, &host, 0, nullptr, 0, nullptr);
        e.finish(true);
        int value = staging.data[0];
        e.begin();
        return value;
    };

    if (!test_virtual_page_replacement(e, read_byte)) return 20;

    cmd::SetRenderTargets targets{};
    if (e.sample_source(source, {}, ViewUse::Sampled, targets) != &source) return 3;
    targets.count = 2;
    targets.colors[1] = &source;
    Texture* snap = e.sample_source(source, {}, ViewUse::Sampled, targets);
    if (!snap || snap->image == source.image || read_byte(*snap, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 64) != 32) return 4;
    bytes[256] = 91;  // between the first 64 bytes of each 4 KiB page
    if (content_hash(source) != source.content_hash) return 5;
    e.frame_ = source.next_full_check;
    if (!e.cpu_texture_changed(source)) return 6;
    e.upload(source);
    if (read_byte(source, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 64) != 91 ||
        read_byte(*snap, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 64) != 32) return 7;
    if (e.sample_source(source, {}, ViewUse::Sampled, targets) != snap || read_byte(*snap, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 64) != 91) return 8;
    bytes[256] = 23;
    ++e.frame_;
    if (!e.cpu_texture_changed(source)) return 9;
    e.upload(source);
    ++e.frame_;
    if (e.cpu_texture_changed(source)) return 10;
    // Failed uploads must not publish a new digest or prevent a retry.
    bytes[256] = 44;
    Staging unavailable;
    unavailable.failed_ = true;
    auto* previous = e.staging_;
    e.staging_ = &unavailable;
    e.upload(source);
    e.staging_ = previous;
    if (source.uploaded || !e.cpu_texture_changed(source)) return 14;
    e.upload(source);
    if (read_byte(source, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 64) != 44) return 15;

    // Read from an attachment after a real render-pass clear.
    e.st_.rt = targets;
    e.opt_.clear_diferido = false;
    e.clear_color({1, {1, 0, 0, 1}, 15});
    snap = e.sample_source(source, {}, ViewUse::Sampled, targets);
    if (!snap || read_byte(*snap, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 64) != 255) return 16;

    ProgramStage fragment{};
    fragment.shader.has_bindless_set = true;
    Resource binding{};
    binding.kind = Resource::Texture;
    binding.texture_type = 2;
    binding.set = nvnshader::kBindlessSet;
    binding.cbuf_index = kUniformBankBase;
    binding.count = 2;
    fragment.shader.resources.push_back(binding);
    const ProgramStage* stages[kStages]{};
    const int fs = static_cast<int>(Stage::Fragment);
    stages[fs] = &fragment;
    DescriptorPool pool{};
    pool.textures.resize(2);
    pool.textures[1].texture = &source;
    e.st_.texture_pool = &pool;
    uint32_t words[] = {1, 0, 1, 0};
    e.st_.bindings[fs].uniforms[0] = {reinterpret_cast<uint64_t>(words), sizeof words};
    if (!e.ensure_image(e.sampled_defaults_.two_d, e.cmd_)) return 17;
    auto normal = e.bindless_set(stages);
    auto feedback = e.bindless_set(stages, VK_NULL_HANDLE, &targets);
    if (!normal || !feedback || normal == feedback || e.bindless_set(stages, VK_NULL_HANDLE, &targets) != feedback) return 18;
    words[2] = UINT32_MAX;
    auto changed = e.bindless_set(stages, VK_NULL_HANDLE, &targets);
    if (!changed || changed == feedback) return 19;
    if (!test_watched_handle_table(e, source)) return 21;

    Texture depth{};
    depth.serial = next_texture_serial();
    depth.desc.target = kTarget2DArray;
    depth.desc.format = 0x35;
    depth.desc.width = depth.desc.height = 32;
    depth.desc.depth = depth.desc.levels = 2;
    if (!e.ensure_image(depth, e.cmd_)) return 11;
    VkImageSubresourceRange sub{depth.aspect, 1, 1, 1, 1};
    VkClearDepthStencilValue value{0.25f, 73};
    vkCmdClearDepthStencilImage(e.cmd_, depth.image, VK_IMAGE_LAYOUT_GENERAL, &value, 1, &sub);
    targets.depth = &depth;
    targets.depth_view.flags = kViewLayers | kViewLevels;
    targets.depth_view.base_level = targets.depth_view.base_layer = 1;
    snap = e.sample_source(depth, {}, ViewUse::Sampled, targets);
    if (!snap || read_byte(*snap, VK_IMAGE_ASPECT_STENCIL_BIT, 1, 1, 0) != 73) return 12;
    e.retire_texture(source);
    e.retire_texture(depth);
    return e.feedback_copies_.empty() ? 0 : 13;
}
