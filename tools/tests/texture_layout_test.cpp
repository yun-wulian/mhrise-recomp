#include "nnsdk/nvn/layout.h"
#include <cassert>
#include <utility>

static void check_swizzle_rows(const nvn::LevelLayout& level, const nvn::BlockBox& box) {
    constexpr uint32_t bytes = 4;
    const size_t pitch = size_t{box.width} * bytes + 13;
    const size_t slice = pitch * box.height + 7;
    std::vector<uint8_t> input(slice * box.depth);
    for (size_t i = 0; i < input.size(); ++i) input[i] = uint8_t(i * 37 + i / 251);
    std::vector<uint8_t> actual(level.size, 0xa5), expected = actual;
    nvn::swizzle_region(level, bytes, box, input.data(), pitch, slice, actual.data());
    for (uint32_t y = 0; y < box.height; ++y) {
        const nvn::BlockBox row{box.x, box.y + y, box.z, box.width, 1, box.depth};
        nvn::swizzle_region(level, bytes, row, input.data() + y * pitch, pitch, slice, expected.data());
    }
    assert(actual == expected); // includes untouched rows, columns, slices and tile padding
}

int main() {
    using nvn::compute_layout;
    // Both 8- and 16-GOB layouts occupy 256 KiB for Rise's 512x77 RGBA8 animation maps.
    // Check the actual texel addresses on either side of a GOB column and a tile row.
    const auto vat = compute_layout(512, 77, 1, 1, 2, 4, 1, 1, false, 0, false);
    const auto& level = vat.levels[0];
    assert(level.size == 262144);
    std::vector<uint8_t> tiled(level.size), pixels(512 * 77 * 4);
    tiled[4096] = 37;
    tiled[8192] = 89;
    tiled[131072] = 163;
    nvn::deswizzle_level(level, 4, tiled.data(), pixels.data());
    assert(pixels[16 * 4] == 37 && pixels[32 * 4] == 89 && pixels[64 * 512 * 4] == 163);
    assert(vat.levels[1].offset == level.size && vat.levels[1].block_height_gobs == 8);

    const std::pair<uint32_t, uint32_t> boundaries[] = {
        {1, 1}, {10, 1}, {11, 2}, {21, 2}, {22, 4}, {42, 4}, {43, 8}, {85, 8}, {86, 16}, {512, 16},
    };
    for (const auto& [rows, gobs] : boundaries) {
        for (uint32_t block : {1u, 4u, 6u}) {
            const auto h = (rows - 1) * block + 1;  // include the final partial compressed block
            const auto layout = compute_layout(512, h, 1, 2, 1, block == 1 ? 4 : 16, block, block, false, 0, false);
            assert(layout.levels[0].block_height_gobs == gobs);
        }
    }

    const auto packaged = compute_layout(512, 77, 1, 1, 1, 4, 1, 1, false, 0, false, 4, 0);
    assert(packaged.levels[0].block_height_gobs == 16);
    const auto volume = compute_layout(32, 77, 1, 1, 1, 4, 1, 1, false, 0, true);
    assert(volume.levels[0].block_height_gobs == 16);
    const auto preview = compute_layout(1136, 922, 1, 1, 1, 4, 1, 1, false, 0, false);
    check_swizzle_rows(preview.levels[0], {0, 0, 0, 1136, 922, 1});
    const auto partial = compute_layout(64, 40, 4, 1, 1, 4, 1, 1, false, 0, true);
    for (uint32_t y = 0; y < 8; ++y)
        for (uint32_t height = 1; height <= 25; ++height)
            check_swizzle_rows(partial.levels[0], {16, y, 1, 32, height, 2});
}
