// Compute stages that drippu's translator rejects because a bindless image handle was spilled to memory.
//
// 22 of Brothership's 3,806 compute shaders (2026-10-24, the shader test over the game files) write an image
// whose handle the Switch's compiler stored to local memory, or to a per-thread slot of shared memory, and loaded
// back before the write. drippu's texture pass tracks a handle only through constant-buffer reads and simple
// arithmetic, so it throws "Failed to track bindless texture constant buffer"; the first of these in play
// (a new-game cutscene) crashed the run. TranslateComputeProgram calls drippu's TranslateProgram as before
// and, only when it throws that, translates again with drippu's own steps (Maxwell::TranslateProgram for a
// compute stage, translate_program.cpp; its helpers are local to that file, so they are repeated here) plus
// ForwardSpilledHandles before the texture pass. drippu stays unmodified.
//
// ForwardSpilledHandles: for each bindless instruction whose handle is such a load, every store that can reach
// the load (walking back through the blocks) must write to the same address, every other store or atomic on
// the way must provably not overlap it, and every reaching store must write the same constant-buffer word
// (GetCbufU32 with constant bank and offset). The handle is then that word, read again just before the
// instruction: the constant buffer does not change within a dispatch, so the value is the same.

#include "shaders.h"

#include <algorithm>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "common/settings.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/post_order.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/structured_control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/ir_opt/passes.h"

namespace rt::nvn {

namespace {

using Shader::IR::Inst;
using Shader::IR::Opcode;
using Shader::IR::Value;

enum class Space { kNone, kLocal, kShared };

/// The memory an instruction loads from (`load`) or writes to, and the bytes it covers (local addresses are
/// words, shared ones bytes); kNone for others.
Space SpaceOf(const Inst& inst, bool& load, u32& bytes) {
    load = false;
    bytes = 4;
    switch (inst.GetOpcode()) {
    case Opcode::LoadLocal: load = true; bytes = 1; return Space::kLocal;
    case Opcode::WriteLocal: bytes = 1; return Space::kLocal;
    case Opcode::LoadSharedU32: load = true; return Space::kShared;
    case Opcode::WriteSharedU8: bytes = 1; return Space::kShared;
    case Opcode::WriteSharedU16: bytes = 2; return Space::kShared;
    case Opcode::WriteSharedU32:
    case Opcode::SharedAtomicIAdd32:
    case Opcode::SharedAtomicSMin32:
    case Opcode::SharedAtomicUMin32:
    case Opcode::SharedAtomicSMax32:
    case Opcode::SharedAtomicUMax32:
    case Opcode::SharedAtomicInc32:
    case Opcode::SharedAtomicDec32:
    case Opcode::SharedAtomicAnd32:
    case Opcode::SharedAtomicOr32:
    case Opcode::SharedAtomicXor32:
    case Opcode::SharedAtomicExchange32: return Space::kShared;
    case Opcode::WriteSharedU64:
    case Opcode::SharedAtomicExchange64:
    case Opcode::SharedAtomicExchange32x2: bytes = 8; return Space::kShared;
    case Opcode::WriteSharedU128: bytes = 16; return Space::kShared;
    default: return Space::kNone;
    }
}

bool IsStore(Space s, const Inst& inst) {
    bool load;
    u32 bytes;
    return SpaceOf(inst, load, bytes) == s && !load;
}

/// Whether an address computation can be compared by its shape: pure arithmetic and the thread's ids.
bool Pure(Opcode op) {
    switch (op) {
    case Opcode::IAdd32:
    case Opcode::ISub32:
    case Opcode::IMul32:
    case Opcode::ShiftLeftLogical32:
    case Opcode::ShiftRightLogical32:
    case Opcode::BitwiseAnd32:
    case Opcode::BitwiseOr32:
    case Opcode::BitFieldUExtract:
    case Opcode::CompositeExtractU32x3:
    case Opcode::LocalInvocationId:
    case Opcode::WorkgroupId:
        return true;
    default:
        return false;
    }
}

/// Whether two values are the same: the same immediate, the same instruction, or the same pure computation of
/// the same values.
bool Same(const Value& a_in, const Value& b_in, int depth = 0) {
    const Value a = a_in.Resolve(), b = b_in.Resolve();
    if (a.IsImmediate() || b.IsImmediate()) {
        return a.IsImmediate() && b.IsImmediate() && a.Type() == Shader::IR::Type::U32 &&
               b.Type() == Shader::IR::Type::U32 && a.U32() == b.U32();
    }
    const Inst* x = a.InstRecursive();
    const Inst* y = b.InstRecursive();
    if (x == y) return true;
    if (depth > 16 || x->GetOpcode() != y->GetOpcode() || !Pure(x->GetOpcode()) || x->NumArgs() != y->NumArgs()) return false;
    for (size_t i = 0; i < x->NumArgs(); ++i) {
        if (!Same(x->Arg(i), y->Arg(i), depth + 1)) return false;
    }
    return true;
}

/// An address as base + constant (base empty for a constant address).
std::pair<std::optional<Value>, u32> Split(const Value& v_in) {
    const Value v = v_in.Resolve();
    if (v.IsImmediate()) {
        if (v.Type() == Shader::IR::Type::U32) return {std::nullopt, v.U32()};
        return {v, 0}; // not an address: compared as a value (never the same as a U32)
    }
    const Inst* i = v.InstRecursive();
    if (i->GetOpcode() == Opcode::IAdd32) {
        for (size_t k = 0; k < 2; ++k) {
            const Value c = i->Arg(k).Resolve();
            if (!c.IsImmediate() || c.Type() != Shader::IR::Type::U32) continue;
            auto [base, off] = Split(i->Arg(1 - k));
            return {base, off + c.U32()};
        }
    }
    return {v, 0};
}

/// Whether [a, a + a_bytes) and [b, b + b_bytes) provably do not overlap.
bool Disjoint(const Value& a, u32 a_bytes, const Value& b, u32 b_bytes) {
    const auto [ab, ao] = Split(a);
    const auto [bb, bo] = Split(b);
    if (ab.has_value() != bb.has_value() || (ab && !Same(*ab, *bb))) return false;
    return ao + a_bytes <= bo || bo + b_bytes <= ao;
}

/// The constant-buffer word (bank, offset) a value is, if it is one.
std::optional<std::pair<u32, u32>> CbufWord(const Value& v_in) {
    const Value v = v_in.Resolve();
    if (v.IsImmediate()) return std::nullopt;
    const Inst* i = v.InstRecursive();
    if (i->GetOpcode() != Opcode::GetCbufU32) return std::nullopt;
    const Value bank = i->Arg(0).Resolve(), offset = i->Arg(1).Resolve();
    if (!bank.IsImmediate() || !offset.IsImmediate() || bank.Type() != Shader::IR::Type::U32 ||
        offset.Type() != Shader::IR::Type::U32) {
        return std::nullopt;
    }
    return std::pair{bank.U32(), offset.U32()};
}

/// The constant-buffer word every store reaching `load` (in `block`) wrote to its address, or nothing.
std::optional<std::pair<u32, u32>> ReachingCbufWord(Shader::IR::Block* block, Inst& load, Space space) {
    const Value address = load.Arg(0);
    const u32 load_bytes = space == Space::kLocal ? 1 : 4;
    std::optional<std::pair<u32, u32>> word;
    bool ok = true;
    // Scans [begin, end) of a block backwards; true when a store ended this path (or it failed).
    const auto scan = [&](Shader::IR::Block* b, Shader::IR::Block::iterator end) {
        for (auto it = end; it != b->Instructions().begin();) {
            const Inst& i = *--it;
            bool is_load;
            u32 bytes;
            if (SpaceOf(i, is_load, bytes) != space || is_load) continue;
            if ((i.GetOpcode() == Opcode::WriteLocal || i.GetOpcode() == Opcode::WriteSharedU32) && Same(i.Arg(0), address)) {
                const auto w = CbufWord(i.Arg(1));
                if (!w || (word && *word != *w)) ok = false;
                else word = w;
                return true;
            }
            if (!Disjoint(i.Arg(0), bytes, address, load_bytes)) {
                ok = false; // may overlap: give up
                return true;
            }
        }
        return false;
    };
    std::set<Shader::IR::Block*> done;
    std::vector<Shader::IR::Block*> todo;
    const auto preds = [&](Shader::IR::Block* b) {
        if (b->ImmPredecessors().empty()) ok = false; // the entry, with no store: undefined
        for (Shader::IR::Block* p : b->ImmPredecessors()) {
            if (done.insert(p).second) todo.push_back(p);
        }
    };
    if (!scan(block, Shader::IR::Block::InstructionList::s_iterator_to(load))) preds(block);
    while (ok && !todo.empty()) {
        Shader::IR::Block* b = todo.back();
        todo.pop_back();
        if (!scan(b, b->Instructions().end())) preds(b);
    }
    return ok ? word : std::nullopt;
}

/// See the top of the file. Returns how many handles it forwarded.
size_t ForwardSpilledHandles(Shader::IR::Program& program) {
    size_t forwarded = 0;
    for (Shader::IR::Block* block : program.blocks) {
        for (Inst& inst : block->Instructions()) {
            if (!Shader::IR::NameOf(inst.GetOpcode()).starts_with("Bindless")) continue;
            const Value handle = inst.Arg(0).Resolve();
            if (handle.IsImmediate()) continue;
            Inst& load = *handle.InstRecursive();
            bool is_load;
            u32 bytes;
            const Space space = SpaceOf(load, is_load, bytes);
            if (space == Space::kNone || !is_load) continue;
            Shader::IR::Block* load_block = nullptr;
            for (Shader::IR::Block* b : program.blocks) {
                for (Inst& i : b->Instructions()) {
                    if (&i == &load) load_block = b;
                }
                if (load_block) break;
            }
            if (!load_block) continue;
            const auto word = ReachingCbufWord(load_block, load, space);
            if (!word) continue;
            Shader::IR::IREmitter ir{*block, Shader::IR::Block::InstructionList::s_iterator_to(inst)};
            inst.SetArg(0, ir.GetCbuf(ir.Imm32(word->first), ir.Imm32(word->second)));
            ++forwarded;
        }
    }
    return forwarded;
}

} // namespace

Shader::IR::Program ComputeFrontEnd(Shader::ObjectPool<Shader::IR::Inst>& insts,
                                    Shader::ObjectPool<Shader::IR::Block>& blocks, NvnEnvironment& env,
                                    Shader::Maxwell::Flow::CFG& cfg, const Shader::HostTranslateInfo& host) {
    Shader::IR::Program program;
    program.syntax_list = Shader::Maxwell::BuildASL(insts, blocks, env, cfg, host);
    u32 order = 0;
    for (const auto& node : program.syntax_list) {
        if (node.type != Shader::IR::AbstractSyntaxNode::Type::Block) continue;
        program.blocks.push_back(node.data.block);
        program.blocks.back()->SetOrder(order++);
    }
    program.post_order_blocks = Shader::IR::PostOrder(program.syntax_list.front());
    program.stage = env.ShaderStage();
    program.local_memory_size = env.LocalMemorySize();
    program.workgroup_size = env.WorkgroupSize();
    program.shared_memory_size = env.SharedMemorySize();
    if (program.blocks.size() != program.post_order_blocks.size()) { // RemoveUnreachableBlocks
        std::erase_if(program.blocks, [&](Shader::IR::Block* b) {
            return b != program.blocks.front() && b->ImmPredecessors().empty();
        });
    }
    namespace O = Shader::Optimization;
    if (!host.support_float64) O::LowerFp64ToFp32(program);
    if (!host.support_float16) O::LowerFp16ToFp32(program);
    if (!host.support_int64) O::LowerInt64ToInt32(program);
    if (!host.support_conditional_barrier) O::ConditionalBarrierPass(program);
    O::SsaRewritePass(program);
    O::ConstantPropagationPass(env, program);
    O::PositionPass(env, program);
    O::GlobalMemoryToStorageBufferPass(program, host);
    return program;
}

Shader::IR::Program TranslateComputeProgram(Shader::ObjectPool<Shader::IR::Inst>& insts,
                                            Shader::ObjectPool<Shader::IR::Block>& blocks, NvnEnvironment& env,
                                            Shader::Maxwell::Flow::CFG& cfg, const Shader::HostTranslateInfo& host_in) {
    try {
        return Shader::Maxwell::TranslateProgram(insts, blocks, env, cfg, host_in);
    } catch (const Shader::NotImplementedException& e) {
        if (!std::string_view(e.what()).starts_with("Failed to track bindless texture constant buffer")) throw;
    }
    Shader::HostTranslateInfo host{host_in};
    host.ApplyDescriptorLimitPolicy();
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_blocks;
    Shader::Maxwell::Flow::CFG fresh(env, flow_blocks, env.StartAddress(), false);
    Shader::IR::Program program = ComputeFrontEnd(insts, blocks, env, fresh, host);
    const size_t forwarded = ForwardSpilledHandles(program);
    namespace O = Shader::Optimization;
    O::TexturePass(env, program, host); // throws again if something is still untracked
    if (::Settings::values.resolution_info.active || ::Settings::values.rescale_hack.GetValue()) O::RescalingPass(program);
    O::DeadCodeEliminationPass(program);
    if (::Settings::values.renderer_debug) O::VerificationPass(program);
    O::CollectShaderInfoPass(env, program);
    O::LayerPass(program, host);
    O::VendorWorkaroundPass(program);
    // AddNVNStorageBuffers (translate_program.cpp), for a compute stage
    if (program.info.uses_global_memory) {
        auto& descs = program.info.storage_buffers_descriptors;
        for (u32 index = 0; index < 16; ++index) {
            if (!program.info.nvn_buffer_used[index]) continue;
            const u32 offset = 0x310u + index * 0x10u;
            if (const auto it = std::ranges::find(descs, offset, &Shader::StorageBufferDescriptor::cbuf_offset);
                it != descs.end()) {
                it->is_written |= program.info.stores_global_memory;
                continue;
            }
            descs.push_back({.cbuf_index = 0, .cbuf_offset = offset, .count = 1, .is_written = program.info.stores_global_memory});
        }
    }
    Log("nvn: compute stage: %zu bindless handle(s) tracked through a store to memory and back (compute_translate.cpp)",
        forwarded);
    return program;
}

} // namespace rt::nvn
