// Binds (NVN_PLAN 7.5, 8.1): state objects, the program, handles and buffers.
//
// Recording: NVN reads state objects (and a graphics program's pre-built
// words) when it records, so the records keep copies. Binds for the compute
// stage (stage 5) are control commands in NVN, applied at submit: kinds 7
// (program, 16 bytes), 8-11 (texture, separate texture, separate sampler,
// image: 16 bytes), 12-13 (uniform, storage buffer: 24 bytes) — the layer
// writes control records of the same sizes, so the game's control memory is
// used as by NVN (sdk-rea shader follow-up F1). A compute program is read at
// submit, as NVN does.
//
// Records are packed to fit those sizes: a handle as its low 32 bits plus
// bit 32 (every NVN handle is a sign-extended 32-bit value, with bit 32 set
// for texture handles; anything else stops the run).
//
// Translation: each record updates the queue's RenderState (render.h); the
// draws and dispatches that use it come with Phase 8.2.
#include "render.h"

#include <cstring>

namespace rt::nvn {

namespace {

constexpr u32 kCompute = 5;
enum class HandleKind : u8 { kTexture, kSeparateTexture, kSeparateSampler, kImage };
enum class BufferKind : u8 { kUniform, kStorage, kVertex };
enum class StateKind : u32 { kDepthStencil, kPolygon, kBlend, kColor, kChannelMask };

struct HandleRecord { // 8 bytes: fits a 16-byte control entry
    u32 low;
    u8 kind, stage;
    u16 slot_flag; // slot in bits 0-14, bit 15: the handle's bit 32
};
struct BufferRecord { // 16 bytes: fits a 24-byte control entry
    u64 address;
    u32 size;
    u8 kind, stage;
    u16 slot;
};
struct StateRecord {
    StateKind kind;
    u32 size;
    u8 bytes[8];
};
struct StencilRecord {
    u32 which; // 0 reference, 1 write mask, 2 value mask
    u32 faces; // bit 0 front, bit 1 back
    u32 value;
};
struct ProgramHeader {
    u64 program;
    u32 stage_mask; // the stages bound
    u32 count;      // ProgramStage copies that follow (graphics)
};

void RecordHandle(GuestContext* c, u64 cmdbuf, HandleKind kind, u32 stage, u32 slot, u64 handle, u32 nvn_size) {
    const u32 low = static_cast<u32>(handle);
    const bool bit32 = (handle >> 32) & 1;
    const u64 rebuilt = static_cast<u64>(static_cast<s64>(static_cast<s32>(low))) | (u64{bit32} << 32);
    if (rebuilt != handle || slot >= 0x8000 || stage > kCompute) {
        Fatal("nvn: bind of handle 0x%llx (stage %u, slot %u): not an NVN handle the layer can keep",
              (unsigned long long)handle, stage, slot);
    }
    const HandleRecord r{low, static_cast<u8>(kind), static_cast<u8>(stage), static_cast<u16>(slot | (bit32 << 15))};
    if (stage == kCompute) ControlRecord(c, cmdbuf, Op::kBindHandle, r, nvn_size);
    else Record(c, cmdbuf, Op::kBindHandle, r);
}

void RecordBuffer(GuestContext* c, u64 cmdbuf, BufferKind kind, u32 stage, u32 slot, u64 address, u64 size) {
    if (size > 0xFFFFFFFFULL || slot > 0xFFFF || stage > kCompute) Fatal("nvn: buffer bind (stage %u slot %u size 0x%llx)", stage, slot, (unsigned long long)size);
    const BufferRecord r{address, static_cast<u32>(size), static_cast<u8>(kind), static_cast<u8>(stage), static_cast<u16>(slot)};
    if (kind != BufferKind::kVertex && stage == kCompute) ControlRecord(c, cmdbuf, Op::kBindBuffer, r, 24);
    else Record(c, cmdbuf, Op::kBindBuffer, r);
}

template <size_t N>
void RecordState(GuestContext* c, u64 cmdbuf, StateKind kind, u64 state) {
    StateRecord r{kind, N, {}};
    g_mem.Read(state, r.bytes, N);
    Record(c, cmdbuf, Op::kBindState, r);
}

void RecordStencil(GuestContext* c, u64 cmdbuf, u32 which, u32 faces, u32 value) {
    Record(c, cmdbuf, Op::kSetStencil, StencilRecord{which, faces, value});
}

} // namespace

// --- Recording ------------------------------------------------------------------------------------

NVN_PROC(nvnCommandBufferBindVertexAttribState, void, u64 cmdbuf, s32 count, u64 states) {
    if (count < 0 || count > 16) Fatal("nvn: BindVertexAttribState with %d attributes", count);
    std::vector<u8> payload(4 + 4 * static_cast<u32>(count));
    const u32 n = static_cast<u32>(count);
    std::memcpy(payload.data(), &n, 4);
    g_mem.Read(states, payload.data() + 4, 4 * n);
    Record(c, cmdbuf, Op::kBindVertexAttribs, payload.data(), static_cast<u32>(payload.size()));
}
NVN_PROC(nvnCommandBufferBindVertexStreamState, void, u64 cmdbuf, s32 count, u64 states) {
    if (count < 0 || count > 16) Fatal("nvn: BindVertexStreamState with %d streams", count);
    std::vector<u8> payload(4 + 8 * static_cast<u32>(count));
    const u32 n = static_cast<u32>(count);
    std::memcpy(payload.data(), &n, 4);
    g_mem.Read(states, payload.data() + 4, 8 * n);
    Record(c, cmdbuf, Op::kBindVertexStreams, payload.data(), static_cast<u32>(payload.size()));
}

// A null program unbinds the stages (BRO 0x582480 has its own path for it).
NVN_PROC(nvnCommandBufferBindProgram, void, u64 cmdbuf, u64 program, u32 stages) {
    const Program* p = program ? Get<Program>(program) : nullptr;
    const u32 bound = p ? stages & p->stage_mask : stages & 0x3F;
    if (bound & (1u << kCompute)) { // a control command, read at submit
        const u64 v = program;
        ControlRecord(c, cmdbuf, Op::kBindProgram, v, 16);
    }
    const u32 graphics = bound & ~(1u << kCompute);
    if (!graphics) return;
    std::vector<u8> payload(sizeof(ProgramHeader));
    ProgramHeader h{program, graphics, 0};
    for (u32 s = 0; s < kStages; ++s) {
        if (!p || !(graphics & (1u << s)) || !p->stages[s]) continue;
        const size_t at = payload.size();
        payload.resize(at + sizeof(ProgramStage));
        std::memcpy(payload.data() + at, &*p->stages[s], sizeof(ProgramStage));
        ++h.count;
    }
    std::memcpy(payload.data(), &h, sizeof h);
    Record(c, cmdbuf, Op::kBindProgram, payload.data(), static_cast<u32>(payload.size()));
}

NVN_PROC(nvnCommandBufferBindDepthStencilState, void, u64 cmdbuf, u64 state) { RecordState<8>(c, cmdbuf, StateKind::kDepthStencil, state); }
NVN_PROC(nvnCommandBufferBindPolygonState, void, u64 cmdbuf, u64 state) { RecordState<4>(c, cmdbuf, StateKind::kPolygon, state); }
NVN_PROC(nvnCommandBufferBindBlendState, void, u64 cmdbuf, u64 state) { RecordState<8>(c, cmdbuf, StateKind::kBlend, state); }
NVN_PROC(nvnCommandBufferBindColorState, void, u64 cmdbuf, u64 state) { RecordState<4>(c, cmdbuf, StateKind::kColor, state); }
NVN_PROC(nvnCommandBufferBindChannelMaskState, void, u64 cmdbuf, u64 state) { RecordState<4>(c, cmdbuf, StateKind::kChannelMask, state); }

NVN_PROC(nvnCommandBufferSetStencilRef, void, u64 cmdbuf, u32 faces, s32 ref) { RecordStencil(c, cmdbuf, 0, faces, static_cast<u32>(ref)); }
NVN_PROC(nvnCommandBufferSetStencilMask, void, u64 cmdbuf, u32 faces, s32 mask) { RecordStencil(c, cmdbuf, 1, faces, static_cast<u32>(mask)); }
NVN_PROC(nvnCommandBufferSetStencilValueMask, void, u64 cmdbuf, u32 faces, s32 mask) { RecordStencil(c, cmdbuf, 2, faces, static_cast<u32>(mask)); }
NVN_PROC(nvnCommandBufferSetPolygonOffsetClamp, void, u64 cmdbuf, float factor, float units, float clamp) {
    Record(c, cmdbuf, Op::kSetPolygonOffsetClamp, std::array<float, 3>{factor, units, clamp});
}

NVN_PROC(nvnCommandBufferBindTexture, void, u64 cmdbuf, u32 stage, s32 slot, u64 handle) {
    RecordHandle(c, cmdbuf, HandleKind::kTexture, stage, static_cast<u32>(slot), handle, 16);
}
NVN_PROC(nvnCommandBufferBindSeparateTexture, void, u64 cmdbuf, u32 stage, s32 slot, u64 handle) {
    RecordHandle(c, cmdbuf, HandleKind::kSeparateTexture, stage, static_cast<u32>(slot), handle, 16);
}
NVN_PROC(nvnCommandBufferBindSeparateSampler, void, u64 cmdbuf, u32 stage, s32 slot, u64 handle) {
    RecordHandle(c, cmdbuf, HandleKind::kSeparateSampler, stage, static_cast<u32>(slot), handle, 16);
}
NVN_PROC(nvnCommandBufferBindImage, void, u64 cmdbuf, u32 stage, s32 slot, u64 handle) {
    RecordHandle(c, cmdbuf, HandleKind::kImage, stage, static_cast<u32>(slot), handle, 16);
}
NVN_PROC(nvnCommandBufferBindUniformBuffer, void, u64 cmdbuf, u32 stage, s32 slot, u64 address, u64 size) {
    RecordBuffer(c, cmdbuf, BufferKind::kUniform, stage, static_cast<u32>(slot), address, size);
}
NVN_PROC(nvnCommandBufferBindStorageBuffer, void, u64 cmdbuf, u32 stage, s32 slot, u64 address, u64 size) {
    RecordBuffer(c, cmdbuf, BufferKind::kStorage, stage, static_cast<u32>(slot), address, size);
}
NVN_PROC(nvnCommandBufferBindVertexBuffer, void, u64 cmdbuf, s32 index, u64 address, u64 size) {
    RecordBuffer(c, cmdbuf, BufferKind::kVertex, 0, static_cast<u32>(index), address, size);
}

// --- Translation ----------------------------------------------------------------------------------

namespace {
template <typename T>
T Read(const u8* p, size_t size, size_t at = 0) {
    if (at + sizeof(T) > size) Fatal("nvn: a bind record is too short");
    T v;
    std::memcpy(&v, p + at, sizeof v);
    return v;
}
void Check(bool ok, const char* what, u32 value) {
    if (!ok) Fatal("nvn: bind: %s %u out of range", what, value);
}
} // namespace

bool TranslateBindRecord(Translator& t, Op op, const u8* p, size_t size) {
    RenderState& rs = *t.render;
    switch (op) {
    case Op::kBindVertexAttribs: {
        rs.vertex_attrib_count = Read<u32>(p, size);
        for (u32 i = 0; i < rs.vertex_attrib_count; ++i) rs.vertex_attribs[i] = Read<u32>(p, size, 4 + 4 * i);
        return true;
    }
    case Op::kBindVertexStreams: {
        rs.vertex_stream_count = Read<u32>(p, size);
        for (u32 i = 0; i < rs.vertex_stream_count; ++i) rs.vertex_streams[i] = Read<std::array<u32, 2>>(p, size, 4 + 8 * i);
        return true;
    }
    case Op::kBindProgram: {
        if (size == 8) { // compute: read at submit
            rs.compute_program = Read<u64>(p, size);
            rs.compute_stage.reset();
            if (rs.compute_program) rs.compute_stage = Get<Program>(rs.compute_program)->stages[kCompute];
            return true;
        }
        const auto h = Read<ProgramHeader>(p, size);
        rs.program = h.program;
        for (u32 s = 0; s < kCompute; ++s) {
            if (h.stage_mask & (1u << s)) rs.stages[s].reset(); // bound stages without code are unbound
        }
        for (u32 i = 0; i < h.count; ++i) {
            const auto st = Read<ProgramStage>(p, size, sizeof h + sizeof(ProgramStage) * i);
            rs.stages[static_cast<u32>(st.stage)] = st;
        }
        return true;
    }
    case Op::kBindState: {
        const auto r = Read<StateRecord>(p, size);
        switch (r.kind) {
        case StateKind::kDepthStencil: std::memcpy(rs.depth_stencil.data(), r.bytes, 8); break;
        case StateKind::kPolygon: std::memcpy(rs.polygon.data(), r.bytes, 4); break;
        case StateKind::kColor: std::memcpy(rs.color.data(), r.bytes, 4); break;
        case StateKind::kChannelMask: std::memcpy(rs.channel_mask.data(), r.bytes, 4); break;
        case StateKind::kBlend:
            Check(r.bytes[0] < kMaxColorTargets, "blend target", r.bytes[0]);
            std::memcpy(rs.blend[r.bytes[0]].data(), r.bytes, 8);
            break;
        }
        return true;
    }
    case Op::kSetStencil: {
        const auto r = Read<StencilRecord>(p, size);
        auto& v = r.which == 0 ? rs.stencil_ref : r.which == 1 ? rs.stencil_mask : rs.stencil_value_mask;
        if (r.faces & 1) v[0] = r.value;
        if (r.faces & 2) v[1] = r.value;
        return true;
    }
    case Op::kSetPolygonOffsetClamp: rs.polygon_offset = Read<std::array<float, 3>>(p, size); return true;
    case Op::kBindHandle: {
        const auto r = Read<HandleRecord>(p, size);
        const u32 slot = r.slot_flag & 0x7FFF;
        const u64 handle = static_cast<u64>(static_cast<s64>(static_cast<s32>(r.low))) | (u64{(r.slot_flag >> 15) & 1u} << 32);
        auto& b = rs.bindings[r.stage];
        switch (static_cast<HandleKind>(r.kind)) {
        case HandleKind::kTexture: Check(slot < b.textures.size(), "texture slot", slot), b.textures[slot] = handle; break;
        case HandleKind::kSeparateTexture:
            Check(slot < b.separate_textures.size(), "separate texture slot", slot), b.separate_textures[slot] = handle;
            break;
        case HandleKind::kSeparateSampler:
            Check(slot < b.separate_samplers.size(), "separate sampler slot", slot), b.separate_samplers[slot] = handle;
            break;
        case HandleKind::kImage: Check(slot < b.images.size(), "image slot", slot), b.images[slot] = handle; break;
        }
        { // c2 of the stage, as the shaders read their handles (render.h)
            static constexpr u32 base[] = {kC2Textures, kC2SeparateTextures, kC2SeparateSamplers, kC2Images};
            std::memcpy(rs.c2[C2Index(r.stage)].data() + base[r.kind] + 8 * slot, &handle, 8);
        }
        return true;
    }
    case Op::kBindBuffer: {
        const auto r = Read<BufferRecord>(p, size);
        const std::array<u64, 2> range{r.address, r.size};
        auto& b = rs.bindings[r.stage];
        switch (static_cast<BufferKind>(r.kind)) {
        case BufferKind::kUniform: Check(r.slot < b.uniform_buffers.size(), "uniform buffer slot", r.slot), b.uniform_buffers[r.slot] = range; break;
        case BufferKind::kStorage:
            Check(r.slot < b.storage_buffers.size(), "storage buffer slot", r.slot), b.storage_buffers[r.slot] = range;
            if (r.stage < kCompute) { // c0's table for the stage: {address low, high, size}
                const u32 entry[4] = {static_cast<u32>(r.address), static_cast<u32>(r.address >> 32), r.size, 0};
                std::memcpy(rs.c0.data() + kC0StorageBuffers + 0x100 * HwStage(r.stage) + 16 * r.slot, entry, 16);
            }
            break;
        case BufferKind::kVertex: Check(r.slot < rs.vertex_buffers.size(), "vertex buffer", r.slot), rs.vertex_buffers[r.slot] = range; break;
        }
        return true;
    }
    default: return false;
    }
}

} // namespace rt::nvn
