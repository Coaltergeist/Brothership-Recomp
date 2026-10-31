// Resolution scaling in the shaders, the factor read at run time (RESOLUTION_PLAN Phase 2).
//
// Adapted from drippu's ir_opt/rescaling_pass.cpp (yuzu's; GPL-2.0-or-later / GPL-3.0-or-later), which drippu runs
// inside TranslateProgram only when Settings::values.resolution_info is active and which writes the factor into
// the shader for integer coordinates (up_scale, down_shift as immediates): a translation per resolution, so a
// shader store and a seed per resolution. Here the same patches are made after TranslateProgram (shaders.h
// RescalingPass; drippu's own pass stays off), with the factor as data:
//
// - Integer coordinates (texel fetches, image reads, texture size queries of 2D textures and images whose
//   binding's bit is set): scaled by n / 4, n read from bits 124-127 of the texture bit mask (the push constants'
//   rescaling_textures, which every stage has; compute stages have no down_factor) - n = 2, 3, 4, 6, 8, 12 for
//   0.5x-3x (resolution.h), 0 read as 4. Integer arithmetic: exact as drippu's (v * n >> 2, v << 2 / n).
//   Each stage's bindings have bits of their own (shaders.h RescaleRange: EmitSPIRV adds the stage's base to every
//   index, so the factor bits are asked for at 124 minus the base); descriptors past the stage's range are never
//   scaled.
// - Fragment coordinates and point sizes (graphics): by down_factor, as drippu - per draw 1 / factor when the
//   draw's render targets are scaled, otherwise 1 (draws.cpp).
// - Fragment stages' integer fetches from scaled textures: drippu's SubScale (the pixel's offset within the
//   scaled texel kept), with the factor from n.
// As in drippu: image writes are not patched (a scaled image written by a shader is draws.cpp's rule to keep out).
#include "shaders.h"

#include <cstdlib>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/ir/basic_block.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/modifiers.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/ir/value.h"
#include "shader_recompiler/ir_opt/passes.h"
#include "shader_recompiler/shader_info.h"

namespace rt::nvn {
namespace {

using Shader::TextureType;
namespace IR = Shader::IR;

bool IsTextureTypeRescalable(TextureType type) {
    return type == TextureType::Color2D || type == TextureType::ColorArray2D || type == TextureType::Color2DRect;
}

/// The program's n (factor x 4), read once in the entry block on first use.
class Factor {
public:
    explicit Factor(IR::Program& program) : program_{program}, base_{RescaleRangeOf(program.stage).texture_base} {}

    IR::U32 N() {
        if (n_.IsEmpty()) {
            IR::Block& entry = *program_.blocks.front();
            auto at = entry.begin();
            while (at != entry.end() && at->GetOpcode() == IR::Opcode::Prologue) ++at;
            IR::IREmitter ir{entry, at};
            IR::U32 n{ir.Imm32(0u)};
            for (u32 bit = 0; bit < 4; ++bit) {
                const IR::U1 set{ir.IsTextureScaled(ir.Imm32(kRescaleFactorBit + bit - base_))}; // (EmitSPIRV adds base_)
                n = ir.BitwiseOr(n, IR::U32{ir.Select(set, ir.Imm32(1u << bit), ir.Imm32(0u))});
            }
            n_ = IR::U32{ir.Select(ir.IEqual(n, ir.Imm32(0u)), ir.Imm32(4u), n)};
        }
        return n_;
    }
    bool used = false;       // any patch made (the program then needs the push constants)
    bool frag_coord = false; // SubScale read the fragment coordinate (the program then declares it)

private:
    IR::Program& program_;
    u32 base_;
    IR::U32 n_;
};

IR::U32 Scale(IR::IREmitter& ir, Factor& f, const IR::U1& is_scaled, const IR::U32& value) {
    const IR::U32 scaled{ir.ShiftRightArithmetic(ir.IMul(value, f.N()), ir.Imm32(2u))};
    return IR::U32{ir.Select(is_scaled, scaled, value)};
}

IR::U32 DownScale(IR::IREmitter& ir, Factor& f, const IR::U1& is_scaled, const IR::U32& value) {
    const IR::U32 scaled{ir.IDiv(ir.ShiftLeftLogical(value, ir.Imm32(2u)), f.N())};
    return IR::U32{ir.Select(is_scaled, scaled, value)};
}

/// drippu's SubScale: the texel at the scaled size plus the pixel's offset within it.
IR::U32 SubScale(IR::IREmitter& ir, Factor& f, const IR::U1& is_scaled, const IR::U32& value, IR::Attribute attrib) {
    const IR::F32 up_factor{ir.FPMul(IR::F32{ir.ConvertUToF(32, 32, f.N())}, ir.Imm32(0.25f))};
    const IR::F32 down_factor{ir.FPRecip(up_factor)};
    const IR::F32 base{ir.FPMul(IR::F32{ir.ConvertUToF(32, 32, value)}, up_factor)};
    const IR::F32 frag_coord{ir.GetAttribute(attrib)};
    f.frag_coord = true;
    const IR::F32 floor{ir.FPMul(up_factor, IR::F32{ir.FPFloor(ir.FPMul(frag_coord, down_factor))})};
    const IR::F32 deviation{ir.FPAdd(base, ir.FPAdd(frag_coord, ir.FPNeg(floor)))};
    return IR::U32{ir.Select(is_scaled, IR::U32{ir.ConvertFToU(32, deviation)}, value)};
}

void PatchFragCoord(IR::Block& block, IR::Inst& inst) {
    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    const IR::F32 frag_coord{ir.GetAttribute(inst.Arg(0).Attribute())};
    inst.ReplaceUsesWith(ir.FPMul(frag_coord, ir.ResolutionDownFactor()));
}

void PatchPointSize(IR::Block& block, IR::Inst& inst) {
    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    const IR::F32 point_value{inst.Arg(1)};
    inst.SetArg(1, ir.FPMul(point_value, IR::F32{ir.FPRecip(ir.ResolutionDownFactor())}));
}

/// drippu's VisitMark (not its rescale_hack variant): a fragment coordinate passed through a shuffle is scaled
/// after the shuffle instead, and the coordinate read itself is left alone (flag 0xDEADBEEF).
void VisitMark(IR::Block& block, IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::ShuffleIndex:
    case IR::Opcode::ShuffleUp:
    case IR::Opcode::ShuffleDown:
    case IR::Opcode::ShuffleButterfly: break;
    default: return;
    }
    const IR::Value shfl_arg{inst.Arg(0)};
    if (shfl_arg.IsImmediate()) return;
    const IR::Inst* const arg_inst{shfl_arg.InstRecursive()};
    if (arg_inst->GetOpcode() != IR::Opcode::BitCastU32F32) return;
    const IR::Value bitcast_arg{arg_inst->Arg(0)};
    if (bitcast_arg.IsImmediate()) return;
    IR::Inst* const bitcast_inst{bitcast_arg.InstRecursive()};
    if (bitcast_inst->GetOpcode() != IR::Opcode::GetAttribute) return;
    const IR::Attribute attr{bitcast_inst->Arg(0).Attribute()};
    if (attr != IR::Attribute::PositionX && attr != IR::Attribute::PositionY) return;
    bitcast_inst->SetFlags<u32>(0xDEADBEEF);
    const auto it{IR::Block::InstructionList::s_iterator_to(inst)};
    IR::IREmitter ir{block, it};
    IR::Inst* const new_inst{&*block.PrependNewInst(it, inst)};
    const IR::F32 new_bitcast{ir.ConvertUToF(32, 32, IR::Value{new_inst})};
    inst.ReplaceUsesWith(ir.FPMul(new_bitcast, IR::F32{ir.FPRecip(ir.ResolutionDownFactor())}));
}

void PatchImageQueryDimensions(IR::Block& block, IR::Inst& inst, Factor& f, const RescaleRange& range) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    if (!IsTextureTypeRescalable(info.type) || info.descriptor_index >= range.textures) return;
    const auto it{IR::Block::InstructionList::s_iterator_to(inst)};
    IR::IREmitter ir{block, it};
    const IR::U1 is_scaled{ir.IsTextureScaled(ir.Imm32(info.descriptor_index))};
    const IR::Value new_inst{&*block.PrependNewInst(it, inst)};
    const IR::U32 width{DownScale(ir, f, is_scaled, IR::U32{ir.CompositeExtract(new_inst, 0)})};
    const IR::U32 height{DownScale(ir, f, is_scaled, IR::U32{ir.CompositeExtract(new_inst, 1)})};
    inst.ReplaceUsesWith(
        ir.CompositeConstruct(width, height, ir.CompositeExtract(new_inst, 2), ir.CompositeExtract(new_inst, 3)));
    f.used = true;
}

/// Argument `index` (x, y and, for 2D arrays at index 1, the layer) scaled.
void ScaleIntegerComposite(IR::IREmitter& ir, Factor& f, IR::Inst& inst, const IR::U1& is_scaled, size_t index,
                           bool offset) {
    const IR::Value composite{inst.Arg(index)};
    if (composite.IsEmpty()) return;
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    const IR::U32 x{Scale(ir, f, is_scaled, IR::U32{ir.CompositeExtract(composite, 0)})};
    const IR::U32 y{Scale(ir, f, is_scaled, IR::U32{ir.CompositeExtract(composite, 1)})};
    if (info.type == TextureType::ColorArray2D && !offset) {
        inst.SetArg(index, ir.CompositeConstruct(x, y, ir.CompositeExtract(composite, 2)));
    } else {
        inst.SetArg(index, ir.CompositeConstruct(x, y));
    }
}

void SubScaleCoord(IR::IREmitter& ir, Factor& f, IR::Inst& inst, const IR::U1& is_scaled) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    const IR::Value coord{inst.Arg(1)};
    const IR::U32 x{SubScale(ir, f, is_scaled, IR::U32{ir.CompositeExtract(coord, 0)}, IR::Attribute::PositionX)};
    const IR::U32 y{SubScale(ir, f, is_scaled, IR::U32{ir.CompositeExtract(coord, 1)}, IR::Attribute::PositionY)};
    if (info.type == TextureType::ColorArray2D) {
        inst.SetArg(1, ir.CompositeConstruct(x, y, ir.CompositeExtract(coord, 2)));
    } else {
        inst.SetArg(1, ir.CompositeConstruct(x, y));
    }
}

/// ImageFetch (textures) and ImageRead (images): fragment stages keep the pixel's offset (SubScale), others scale.
void PatchFetch(IR::Block& block, IR::Inst& inst, Factor& f, const RescaleRange& range, bool fragment, bool image) {
    const auto info{inst.Flags<IR::TextureInstInfo>()};
    if (!IsTextureTypeRescalable(info.type)) return;
    if (info.descriptor_index >= (image ? range.images : range.textures)) return;
    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    const IR::U1 is_scaled{image ? ir.IsImageScaled(ir.Imm32(info.descriptor_index))
                                 : ir.IsTextureScaled(ir.Imm32(info.descriptor_index))};
    if (fragment) {
        SubScaleCoord(ir, f, inst, is_scaled);
    } else {
        ScaleIntegerComposite(ir, f, inst, is_scaled, 1, false);
    }
    if (!image) ScaleIntegerComposite(ir, f, inst, is_scaled, 2, true); // ImageFetch's offset
    f.used = true;
}

} // namespace

void RescalingPass(IR::Program& program) {
    if (!RescalingPassOn()) return;
    const bool fragment = program.stage == Shader::Stage::Fragment;
    const bool graphics = program.stage != Shader::Stage::Compute;
    Factor f{program};
    const RescaleRange range = RescaleRangeOf(program.stage);
    if (fragment) {
        for (IR::Block* const block : program.post_order_blocks) {
            for (IR::Inst& inst : block->Instructions()) VisitMark(*block, inst);
        }
    }
    bool down_factor = false; // ResolutionDownFactor read (graphics only: compute stages have no down_factor)
    for (IR::Block* const block : program.post_order_blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            switch (inst.GetOpcode()) {
            case IR::Opcode::GetAttribute: {
                const IR::Attribute attr{inst.Arg(0).Attribute()};
                if (fragment && (attr == IR::Attribute::PositionX || attr == IR::Attribute::PositionY)) {
                    if (inst.Flags<u32>() != 0xDEADBEEF) PatchFragCoord(*block, inst);
                    down_factor = true;
                }
                break;
            }
            case IR::Opcode::SetAttribute:
                if (graphics && inst.Arg(0).Attribute() == IR::Attribute::PointSize && inst.Flags<u32>() != 0xDEADBEEF) {
                    PatchPointSize(*block, inst);
                    down_factor = true;
                }
                break;
            case IR::Opcode::ImageQueryDimensions: PatchImageQueryDimensions(*block, inst, f, range); break;
            case IR::Opcode::ImageFetch: PatchFetch(*block, inst, f, range, fragment, false); break;
            case IR::Opcode::ImageRead: PatchFetch(*block, inst, f, range, fragment, true); break;
            default: break;
            }
        }
    }
    if (!f.used && !down_factor) return;
    Shader::Optimization::DeadCodeEliminationPass(program);
    // CollectShaderInfoPass ran inside TranslateProgram: what the patches added is noted here
    program.info.uses_rescaling_uniform = true;
    if (f.frag_coord) {
        program.info.loads.mask[static_cast<size_t>(IR::Attribute::PositionX)] = true;
        program.info.loads.mask[static_cast<size_t>(IR::Attribute::PositionY)] = true;
    }
}

RescaleRange RescaleRangeOf(Shader::Stage stage) {
    switch (stage) {
    case Shader::Stage::Fragment: return {0, 64, 0, 32};
    case Shader::Stage::VertexA:
    case Shader::Stage::VertexB: return {64, 32, 32, 16};
    case Shader::Stage::Geometry: return {96, 16, 48, 8};
    case Shader::Stage::TessellationControl: return {112, 6, 56, 4};
    case Shader::Stage::TessellationEval: return {118, 6, 60, 4};
    case Shader::Stage::Compute: break;
    }
    return {0, kRescaleFactorBit, 0, 32 * Shader::Backend::SPIRV::NUM_IMAGE_SCALING_WORDS};
}

void SetRescaleBase(Shader::Backend::Bindings& bindings, Shader::Stage stage) {
    const RescaleRange r = RescaleRangeOf(stage);
    bindings.texture_scaling_index = r.texture_base, bindings.image_scaling_index = r.image_base;
}

bool RescalingPassOn() {
    static const bool on = [] {
        const char* e = std::getenv("RT_NVN_RESCALE_PASS");
        return !(e && *e == '0');
    }();
    return on;
}

} // namespace rt::nvn
