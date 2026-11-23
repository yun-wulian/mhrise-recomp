#pragma once

#include <cstdint>
#include <vector>

namespace nvn {
inline constexpr uint64_t kFacePixelShader = 0xc87ca93efba78188ULL;
// Diagnostic copy only. Resource bindings, entry-point inputs and normal shader caches are untouched.
std::vector<uint32_t> instrument_face_nan(const std::vector<uint32_t>& source);
}
