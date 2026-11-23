#include "nnsdk/nvn/face_nan_probe.h"
#include <spirv/unified1/spirv.h>
#include <spirv/unified1/GLSL.std.450.h>

#include <array>
#include <cstring>
#include <map>

namespace nvn {
namespace {
using Words = std::vector<uint32_t>;

void emit(Words& to, SpvOp op, std::initializer_list<uint32_t> args) {
    to.push_back((uint32_t(args.size() + 1) << 16) | op);
    to.insert(to.end(), args);
}

struct Probe {
    explicit Probe(uint32_t bound) : next(bound) {}
    uint32_t next, u32 = 0, f32 = 0, boolean = 0, local_ptr = 0, output_ptr = 0;
    uint32_t entry = 0, glsl = 0, color = 0, first = 0;
    Words declarations, body;
    std::map<uint32_t, uint32_t> constants;

    uint32_t constant(uint32_t value) {
        auto [it, fresh] = constants.try_emplace(value, next);
        if (fresh) emit(declarations, SpvOpConstant, {u32, next++, value});
        return it->second;
    }
    uint32_t result(SpvOp op, uint32_t type, std::initializer_list<uint32_t> args) {
        uint32_t id = next++;
        body.push_back((uint32_t(args.size() + 3) << 16) | op);
        body.insert(body.end(), {type, id});
        body.insert(body.end(), args);
        return id;
    }
    uint32_t invalid(uint32_t scalar) {
        const auto bits = result(SpvOpBitcast, u32, {scalar});
        const auto abs = result(SpvOpBitwiseAnd, u32, {bits, constant(0x7fffffff)});
        return result(SpvOpUGreaterThanEqual, boolean, {abs, constant(0x7f800000)});
    }
    void observe(uint32_t id, unsigned lanes, unsigned kind) {
        uint32_t bad = 0;
        for (unsigned i = 0; i < lanes; ++i) {
            const auto lane = lanes == 1 ? id : result(SpvOpCompositeExtract, f32, {id, i});
            const auto flag = invalid(lane);
            bad = i ? result(SpvOpLogicalOr, boolean, {bad, flag}) : flag;
        }
        const auto old = result(SpvOpLoad, u32, {first});
        const auto empty = result(SpvOpIEqual, boolean, {old, constant(0)});
        const auto take = result(SpvOpLogicalAnd, boolean, {empty, bad});
        const auto updated = result(SpvOpSelect, u32, {take, constant((id << 3) | kind), old});
        emit(body, SpvOpStore, {first, updated});
    }
    // Only mark fragments whose actual RGB export is nonfinite. R,G encode the original
    // SPIR-V result ID in base 128; B is 16 + checkpoint kind. All are exact in R11G11B10F.
    void mark_output() {
        std::array<uint32_t, 3> addr{}, rgb{};
        uint32_t bad = 0;
        for (unsigned i = 0; i < 3; ++i) {
            addr[i] = result(SpvOpAccessChain, output_ptr, {color, constant(i)});
            rgb[i] = result(SpvOpLoad, f32, {addr[i]});
            const auto flag = invalid(rgb[i]);
            bad = i ? result(SpvOpLogicalOr, boolean, {bad, flag}) : flag;
        }
        const auto code = result(SpvOpLoad, u32, {first});
        const auto id = result(SpvOpShiftRightLogical, u32, {code, constant(3)});
        const auto low = result(SpvOpBitwiseAnd, u32, {id, constant(127)});
        const auto high = result(SpvOpShiftRightLogical, u32, {id, constant(7)});
        const auto kind = result(SpvOpBitwiseAnd, u32, {code, constant(7)});
        uint32_t tag[] = {low, high, kind};
        for (unsigned i = 0; i < 3; ++i) {
            const auto biased = result(SpvOpIAdd, u32, {tag[i], constant(i == 2 ? 16 : 1)});
            const auto value = result(SpvOpConvertUToF, f32, {biased});
            const auto out = result(SpvOpSelect, f32, {bad, value, rgb[i]});
            emit(body, SpvOpStore, {addr[i], out});
        }
    }
};
} // namespace

Words instrument_face_nan(const Words& s) {
    if (s.size() < 5 || s[0] != SpvMagicNumber || !s[3] || s[3] > 16384) return {};
    Probe p(s[3]);
    Words lanes(s[3]), locations(s[3], ~0u), storage(s[3], ~0u), literal(s[3], ~0u);
    Words root(s[3]), component(s[3], ~0u);
    size_t funcs = 0;
    for (size_t i = 5; i < s.size();) {
        const auto* w = s.data() + i;
        const unsigned n = w[0] >> 16, op = w[0] & 65535;
        if (!n || n > s.size() - i) return {};
        if (op == SpvOpEntryPoint && w[1] == SpvExecutionModelFragment) p.entry = w[2];
        if (op == SpvOpExtInstImport && n == 6 && !memcmp(w+2, "GLSL.std.450", 13)) p.glsl = w[1];
        if (op == SpvOpTypeInt && w[2] == 32 && w[3] == 0) p.u32 = w[1];
        if (op == SpvOpTypeFloat && w[2] == 32) {p.f32 = w[1]; lanes[w[1]] = 1;}
        if (op == SpvOpTypeVector && w[2] == p.f32) lanes[w[1]] = w[3];
        if (op == SpvOpTypeBool) p.boolean = w[1];
        if (op == SpvOpTypePointer && w[2] == SpvStorageClassFunction && w[3] == p.u32) p.local_ptr = w[1];
        if (op == SpvOpTypePointer && w[2] == SpvStorageClassOutput && w[3] == p.f32) p.output_ptr = w[1];
        if (op == SpvOpDecorate && w[2] == SpvDecorationLocation) locations[w[1]] = w[3];
        if (op == SpvOpConstant && n == 4) {
            literal[w[2]] = w[3];
            if (w[1] == p.u32) p.constants[w[3]] = w[2];
        }
        if (op == SpvOpVariable) {
            storage[w[2]] = w[3];
            root[w[2]] = w[2];
            if (w[3] == SpvStorageClassOutput && locations[w[2]] == 0) p.color = w[2];
        }
        if (op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain) {
            root[w[2]] = root[w[3]];
            if (n == 5) component[w[2]] = literal[w[4]];
        }
        if (op == SpvOpFunction && !funcs) funcs = i;
        i += n;
    }
    if (!funcs || !p.entry || !p.color || !p.output_ptr || !p.boolean || !p.u32 || !p.f32 || !p.glsl) return {};
    if (!p.local_ptr) {
        p.local_ptr = p.next++;
        emit(p.declarations, SpvOpTypePointer, {p.local_ptr, SpvStorageClassFunction, p.u32});
    }
    p.first = p.next++;
    bool main = false, first_block = false;
    unsigned exports = 0;
    for (size_t i = funcs; i < s.size();) {
        const auto* w = s.data() + i;
        const unsigned n = w[0] >> 16, op = w[0] & 65535;
        p.body.insert(p.body.end(), w, w+n);
        if (op == SpvOpFunction) first_block = main = w[2] == p.entry;
        if (main && op == SpvOpLabel && first_block) {
            emit(p.body, SpvOpVariable, {p.local_ptr, p.first, SpvStorageClassFunction, p.constant(0)});
            first_block = false;
        }
        if (main && n >= 4 && w[1] < lanes.size() && lanes[w[1]]) {
            unsigned kind = 0;
            if (op == SpvOpLoad && storage[root[w[3]]] == SpvStorageClassInput) kind = 1;
            if ((op >= SpvOpImageSampleImplicitLod && op <= SpvOpImageDrefGather) || op == SpvOpImageRead) kind = 2;
            if (op == SpvOpFDiv) kind = 3;
            if (op == SpvOpExtInst && w[3] == p.glsl &&
                (w[4] == GLSLstd450InverseSqrt || w[4] == GLSLstd450Sqrt || w[4] == GLSLstd450Log2 ||
                 w[4] == GLSLstd450Exp2 || w[4] == GLSLstd450Pow)) kind = 4;
            if (kind) p.observe(w[2], lanes[w[1]], kind);
        }
        if (main && op == SpvOpStore && root[w[1]] == p.color && component[w[1]] == 3) {
            p.mark_output();
            ++exports;
        }
        i += n;
    }
    if (exports != 1) return {};
    Words out(s.begin(), s.begin()+funcs);
    out[3] = p.next;
    out.insert(out.end(), p.declarations.begin(), p.declarations.end());
    out.insert(out.end(), p.body.begin(), p.body.end());
    return out;
}
} // namespace nvn
