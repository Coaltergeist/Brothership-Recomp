// Shader translation: NVN's Maxwell shaders to SPIR-V with
// drippu's shader_recompiler.
#pragma once

#include "program.h"

#include <functional>
#include <string>
#include <vector>

#include "shader_recompiler/backend/bindings.h"
#include "shader_recompiler/stage.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/frontend/ir/program.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"
#include "shader_recompiler/shader_info.h"

namespace rt::nvn {

/// What a shader's translation asks about the state it runs with.
struct ShaderHooks {
    /// A constant-buffer word known at translation time: banks other than c1
    /// (c0 / c2: the driver's, c3 + n: the game's uniform buffers).
    std::function<u32(u32 bank, u32 offset)> cbuf;
    /// The type of the texture behind a bound handle (from c2).
    std::function<Shader::TextureType(u32 handle)> texture_type;
};

/// drippu's Environment for one NVN shader stage. `code` is the stage's code
/// memory from its GPU address (debug tag, SPH, instructions, compiler
/// constants: control +0x704 bytes); c1 reads come from it.
class NvnEnvironment final : public Shader::Environment {
public:
    NvnEnvironment(std::vector<u8> code, const ProgramStage& stage, ShaderHooks hooks);

    u64 ReadInstruction(u32 address) override;
    u32 ReadCbufValue(u32 cbuf_index, u32 cbuf_offset) override;
    Shader::TextureType ReadTextureType(u32 raw_handle) override;
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32 raw_handle) override;
    bool IsTexturePixelFormatInteger(u32 raw_handle) override;
    u32 ReadViewportTransformState() override;
    u32 TextureBoundBuffer() const override;
    u32 LocalMemorySize() const override;
    u32 SharedMemorySize() const override;
    std::array<u32, 3> WorkgroupSize() const override;
    bool HasHLEMacroState() const override;
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank, u32 offset) override;
    void Dump(u64 pipeline_hash, u64 shader_hash) override;

private:
    std::vector<u8> code_;
    ProgramStage s_;
    ShaderHooks hooks_;
    u32 local_memory_ = 0;
};

/// drippu's compiler settings for the layer's Vulkan device (as its pipeline cache makes them).
const Shader::Profile& ShaderProfile();
const Shader::HostTranslateInfo& ShaderHostInfo();

/// A translated stage.
struct TranslatedShader {
    std::vector<u32> spirv;
    Shader::Info info;
};

/// Translates one stage on its own (graphics stages without the previous
/// stage's outputs: the default runtime state). Throws Shader::Exception for
/// what the recompiler does not support.
TranslatedShader TranslateStage(NvnEnvironment& env);
/// The same with the given runtime state (graphics stages).
TranslatedShader TranslateStage(NvnEnvironment& env, Shader::RuntimeInfo runtime);

/// Resolution scaling with the factor read at run time (rescaling_pass.cpp): run on
/// every program right after TranslateProgram / TranslateComputeProgram. RT_NVN_RESCALE_PASS=0 leaves the
/// programs as drippu made them (A/B; the shader store is then off: its entries are made with the pass).
void RescalingPass(Shader::IR::Program& program);
bool RescalingPassOn();
/// Bits 124-127 of the texture bit mask (the push constants' rescaling_textures) hold n, the factor x 4
/// (resolution.h: 2, 3, 4, 6, 8, 12).
constexpr u32 kRescaleFactorBit = 124;
/// Each stage's part of the push constants' "binding is scaled" bits (texture bits 0-123 of 128, image bits 0-63):
/// fixed by stage, so a stage translated alone numbers its bits as in any pipeline, and the stages of a draw, which
/// share the push constants, never share a bit. EmitSPIRV adds the base to every index (Bindings'
/// texture_scaling_index / image_scaling_index: set by SetRescaleBase); descriptors past a stage's count are never
/// scaled. Fragment: textures 0-63, images 0-31; vertex 64-95 / 32-47; geometry 96-111 / 48-55; tessellation
/// control 112-117 / 56-59, evaluation 118-123 / 60-63; compute (always alone) 0-123 / 0-63.
struct RescaleRange {
    u32 texture_base, textures, image_base, images;
};
RescaleRange RescaleRangeOf(Shader::Stage stage);
/// Sets `bindings`' scaling bases to the stage's before EmitSPIRV.
void SetRescaleBase(Shader::Backend::Bindings& bindings, Shader::Stage stage);

/// drippu's Maxwell::TranslateProgram for a compute stage; when it fails to track a bindless handle, translated
/// again with handles spilled to local or shared memory followed back to their constant-buffer word
/// (compute_translate.cpp). Throws as drippu does otherwise.
Shader::IR::Program TranslateComputeProgram(Shader::ObjectPool<Shader::IR::Inst>& insts,
                                            Shader::ObjectPool<Shader::IR::Block>& blocks, NvnEnvironment& env,
                                            Shader::Maxwell::Flow::CFG& cfg, const Shader::HostTranslateInfo& host);
/// TranslateProgram's steps for a compute stage up to its texture pass (compute_translate.cpp; host already
/// normalised by ApplyDescriptorLimitPolicy): the IR the texture pass gets.
Shader::IR::Program ComputeFrontEnd(Shader::ObjectPool<Shader::IR::Inst>& insts,
                                    Shader::ObjectPool<Shader::IR::Block>& blocks, NvnEnvironment& env,
                                    Shader::Maxwell::Flow::CFG& cfg, const Shader::HostTranslateInfo& host);

/// Moves every resource of the SPIR-V from descriptor set 0 (drippu decorates them all so) to `set`.
void MoveDescriptorSet(std::vector<u32>& spirv, u32 set);

/// RT_NVN_SHADERTEST=<dir>: translates every shader dumped in tap mode
/// (<dir>/<hash>.ctrl, .code: research.cpp) and ends the run (shadertest.cpp).
[[noreturn]] void RunShaderTest(const char* dir);

} // namespace rt::nvn
