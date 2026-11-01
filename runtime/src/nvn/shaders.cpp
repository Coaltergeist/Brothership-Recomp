// Shader translation with drippu's shader_recompiler.
//
// The compiler settings (Profile, HostTranslateInfo) come from the layer's
// Vulkan device the way drippu's pipeline cache makes them
// (video_core/renderer_vulkan/vk_pipeline_cache.cpp). The Environment reads a
// stage as NVN lays it out (program.h): the SPH (graphics) or the
// instructions (compute) at the program offset (+0x708) of the code memory,
// c1 = the compiler constants at control +0x700 of the same memory;
// TEX_CB_INDEX = 2 (the driver's handle buffer c2), which is also drippu's
// sign of an NVN game (is_proprietary_driver).
#include "shaders.h"

#include "vk.h"

#include <cstring>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

namespace rt::nvn {

namespace {

constexpr u32 kTexCbIndex = 2;
constexpr u32 kConstantsBank = 1;

struct CompilerSettings {
    Shader::Profile profile;
    Shader::HostTranslateInfo host;
};

const CompilerSettings& Settings() {
    static const CompilerSettings s = [] {
        const auto& device = *Vk().device;
        const auto& fc = device.FloatControlProperties();
        const VkDriverId driver_id = device.GetDriverID();
        const VkShaderStageFlags subgroup_stages = device.GetSubgroupSupportedStages();
        auto bit = [&](VkShaderStageFlags flag, Shader::Stage stage) {
            return (subgroup_stages & flag) != 0 ? (1u << static_cast<u32>(stage)) : 0u;
        };
        CompilerSettings r;
        r.profile = Shader::Profile{
            .supported_spirv = device.SupportedSpirvVersion(),
            .unified_descriptor_binding = true,
            .support_descriptor_aliasing = device.IsDescriptorAliasingSupported(),
            .support_int8 = device.IsInt8Supported(),
            .support_uniform_and_storage_buffer_8bit = device.IsUniformAndStorageBuffer8BitAccessSupported(),
            .support_int16 = device.IsShaderInt16Supported(),
            .support_uniform_and_storage_buffer_16bit = device.IsUniformAndStorageBuffer16BitAccessSupported(),
            .support_int64 = device.IsShaderInt64Supported(),
            .support_vertex_instance_id = false,
            .support_float_controls = device.IsKhrShaderFloatControlsSupported(),
            .support_separate_denorm_behavior =
                fc.denormBehaviorIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
            .support_separate_rounding_mode = fc.roundingModeIndependence == VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
            .support_fp16_denorm_preserve = fc.shaderDenormPreserveFloat16 != VK_FALSE,
            .support_fp32_denorm_preserve = fc.shaderDenormPreserveFloat32 != VK_FALSE,
            .support_fp16_denorm_flush = fc.shaderDenormFlushToZeroFloat16 != VK_FALSE,
            .support_fp32_denorm_flush = fc.shaderDenormFlushToZeroFloat32 != VK_FALSE,
            .support_fp16_signed_zero_nan_preserve = fc.shaderSignedZeroInfNanPreserveFloat16 != VK_FALSE,
            .support_fp32_signed_zero_nan_preserve = fc.shaderSignedZeroInfNanPreserveFloat32 != VK_FALSE,
            .support_fp64_signed_zero_nan_preserve = fc.shaderSignedZeroInfNanPreserveFloat64 != VK_FALSE,
            .support_explicit_workgroup_layout = device.IsKhrWorkgroupMemoryExplicitLayoutSupported(),
            .support_vote = device.IsSubgroupFeatureSupported(VK_SUBGROUP_FEATURE_VOTE_BIT),
            .supported_subgroup_stages =
                bit(VK_SHADER_STAGE_VERTEX_BIT, Shader::Stage::VertexA) |
                bit(VK_SHADER_STAGE_VERTEX_BIT, Shader::Stage::VertexB) |
                bit(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, Shader::Stage::TessellationControl) |
                bit(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, Shader::Stage::TessellationEval) |
                bit(VK_SHADER_STAGE_GEOMETRY_BIT, Shader::Stage::Geometry) |
                bit(VK_SHADER_STAGE_FRAGMENT_BIT, Shader::Stage::Fragment) |
                bit(VK_SHADER_STAGE_COMPUTE_BIT, Shader::Stage::Compute),
            .support_viewport_index_layer_non_geometry = device.IsExtShaderViewportIndexLayerSupported(),
            .support_viewport_mask = device.IsNvViewportArray2Supported(),
            .support_typeless_image_loads = device.IsFormatlessImageLoadSupported(),
            .support_demote_to_helper_invocation = device.IsExtShaderDemoteToHelperInvocationSupported(),
            .support_int64_atomics = device.IsExtShaderAtomicInt64Supported(),
            .support_derivative_control = true,
            .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
            .support_native_ndc = device.IsExtDepthClipControlSupported(),
            .support_scaled_attributes = !device.MustEmulateScaledFormats(),
            .support_multi_viewport = device.SupportsMultiViewport(),
            .support_geometry_streams = device.AreTransformFeedbackGeometryStreamsSupported(),
            .support_sampled_image_array_nonuniform_indexing = device.IsSampledImageArrayNonUniformIndexingSupported(),
            .warp_size_potentially_larger_than_guest = device.IsWarpSizePotentiallyBiggerThanGuest(),
            .lower_left_origin_mode = false,
            .need_declared_frag_colors = false,
            .need_gather_subpixel_offset = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                           driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                           driver_id == VK_DRIVER_ID_MESA_RADV ||
                                           driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS ||
                                           driver_id == VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA,
            .has_broken_spirv_clamp = driver_id == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS,
            .has_broken_spirv_position_input = false,
            .has_broken_unsigned_image_offsets = false,
            .has_broken_signed_operations = false,
            .has_broken_fp16_float_controls = driver_id == VK_DRIVER_ID_NVIDIA_PROPRIETARY,
            .ignore_nan_fp_comparisons = false,
            .has_broken_spirv_subgroup_mask_vector_extract_dynamic = false,
            .has_broken_robust = false,
            .min_ssbo_alignment = device.GetStorageBufferAlignment(),
            .max_user_clip_distances = device.GetMaxUserClipDistances(),
        };
        r.host = Shader::HostTranslateInfo{
            .min_ssbo_alignment = device.GetStorageBufferAlignment(),
            .max_per_stage_descriptor_sampled_images = device.GetMaxPerStageDescriptorSampledImages(),
            .max_per_stage_resources = device.GetMaxPerStageResources(),
            .max_descriptor_set_samplers = device.GetMaxDescriptorSetSamplers(),
            .max_descriptor_set_uniform_buffers = device.GetMaxDescriptorSetUniformBuffers(),
            .max_descriptor_set_uniform_buffers_dynamic = device.GetMaxDescriptorSetUniformBuffersDynamic(),
            .max_descriptor_set_storage_buffers = device.GetMaxDescriptorSetStorageBuffers(),
            .max_descriptor_set_storage_buffers_dynamic = device.GetMaxDescriptorSetStorageBuffersDynamic(),
            .max_descriptor_set_sampled_images = device.GetMaxDescriptorSetSampledImages(),
            .max_descriptor_set_storage_images = device.GetMaxDescriptorSetStorageImages(),
            .max_descriptor_set_input_attachements = device.GetMaxDescriptorSetInputAttachments(),
            .support_float64 = device.IsFloat64Supported(),
            .support_float16 = device.IsFloat16Supported(),
            .support_int64 = device.IsShaderInt64Supported(),
            .needs_demote_reorder = driver_id == VK_DRIVER_ID_AMD_PROPRIETARY ||
                                    driver_id == VK_DRIVER_ID_AMD_OPEN_SOURCE ||
                                    driver_id == VK_DRIVER_ID_SAMSUNG_PROPRIETARY,
            .support_snorm_render_buffer = true,
            .support_viewport_index_layer = device.IsExtShaderViewportIndexLayerSupported(),
            .support_geometry_shader_passthrough = device.IsNvGeometryShaderPassthroughSupported(),
            .support_conditional_barrier = device.SupportsConditionalBarriers(),
        };
        r.host.ApplyDescriptorLimitPolicy();
        return r;
    }();
    return s;
}

} // namespace

const Shader::Profile& ShaderProfile() { return Settings().profile; }
const Shader::HostTranslateInfo& ShaderHostInfo() { return Settings().host; }

namespace {

Shader::Stage ToMaxwellStage(Stage s) {
    switch (s) {
    case Stage::kVertex: return Shader::Stage::VertexB;
    case Stage::kTessControl: return Shader::Stage::TessellationControl;
    case Stage::kTessEvaluation: return Shader::Stage::TessellationEval;
    case Stage::kGeometry: return Shader::Stage::Geometry;
    case Stage::kFragment: return Shader::Stage::Fragment;
    case Stage::kCompute: return Shader::Stage::Compute;
    }
    Fatal("nvn: shader stage %u", static_cast<u32>(s));
}

} // namespace

// --- Environment --------------------------------------------------------------------------------

NvnEnvironment::NvnEnvironment(std::vector<u8> code, const ProgramStage& stage, ShaderHooks hooks)
    : code_(std::move(code)), s_(stage), hooks_(std::move(hooks)) {
    this->stage = ToMaxwellStage(s_.stage);
    start_address = s_.program_offset;
    is_proprietary_driver = true; // TEX_CB_INDEX == 2, as drippu decides it
    if (s_.stage == Stage::kCompute) {
        local_memory_ = s_.local_memory; // control +0x710 (likely; the QMD's local memory is not traced yet)
    } else {
        if (code_.size() < start_address + sizeof sph) Fatal("nvn: shader code memory too small for its SPH");
        std::memcpy(&sph, code_.data() + start_address, sizeof sph);
        local_memory_ = static_cast<u32>(sph.LocalMemorySize()) + sph.common3.shader_local_memory_crs_size;
    }
}

u64 NvnEnvironment::ReadInstruction(u32 address) {
    if (u64{address} + 8 > code_.size()) {
        throw Shader::LogicError("instruction at 0x{:x} past the code memory (0x{:x} bytes)", address, code_.size());
    }
    u64 v;
    std::memcpy(&v, code_.data() + address, 8);
    return v;
}

u32 NvnEnvironment::ReadCbufValue(u32 bank, u32 offset) {
    if (bank == kConstantsBank) {
        const u64 at = u64{s_.constants_offset} + offset;
        if (offset + 4 > s_.constants_size || at + 4 > code_.size()) return 0; // past the constants: zero, as unused memory
        u32 v;
        std::memcpy(&v, code_.data() + at, 4);
        return v;
    }
    return hooks_.cbuf ? hooks_.cbuf(bank, offset) : 0;
}

Shader::TextureType NvnEnvironment::ReadTextureType(u32 handle) {
    return hooks_.texture_type ? hooks_.texture_type(handle) : Shader::TextureType::Color2D;
}
Shader::TexturePixelFormat NvnEnvironment::ReadTexturePixelFormat(u32) {
    return Shader::TexturePixelFormat::A8B8G8R8_UNORM; // only used for typeless image loads; 7.5 refines it
}
bool NvnEnvironment::IsTexturePixelFormatInteger(u32) { return false; }
u32 NvnEnvironment::ReadViewportTransformState() { return 1; } // NVN keeps the viewport transform on
u32 NvnEnvironment::TextureBoundBuffer() const { return kTexCbIndex; }
u32 NvnEnvironment::LocalMemorySize() const { return local_memory_; }
u32 NvnEnvironment::SharedMemorySize() const { return s_.shared_memory; }
std::array<u32, 3> NvnEnvironment::WorkgroupSize() const { return {s_.workgroup[0], s_.workgroup[1], s_.workgroup[2]}; }
bool NvnEnvironment::HasHLEMacroState() const { return false; }
std::optional<Shader::ReplaceConstant> NvnEnvironment::GetReplaceConstBuffer(u32, u32) { return std::nullopt; }
void NvnEnvironment::Dump(u64, u64) {}

// --- Translation ---------------------------------------------------------------------------------

/// Moves every resource of the SPIR-V from descriptor set 0 (drippu decorates them all so) to `set`.
void MoveDescriptorSet(std::vector<u32>& spirv, u32 set) {
    constexpr u32 kOpDecorate = 71, kDescriptorSet = 34;
    for (size_t i = 5; i < spirv.size();) {
        const u32 words = spirv[i] >> 16;
        if (words == 0) break;
        if ((spirv[i] & 0xFFFF) == kOpDecorate && words == 4 && spirv[i + 2] == kDescriptorSet) spirv[i + 3] = set;
        i += words;
    }
}

TranslatedShader TranslateStage(NvnEnvironment& env) { return TranslateStage(env, Shader::RuntimeInfo{}); }

TranslatedShader TranslateStage(NvnEnvironment& env, Shader::RuntimeInfo runtime) {
    const CompilerSettings& cs = Settings();
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_blocks;
    Shader::ObjectPool<Shader::IR::Inst> insts;
    Shader::ObjectPool<Shader::IR::Block> blocks;
    const bool compute = env.ShaderStage() == Shader::Stage::Compute;
    const u32 cfg_offset = env.StartAddress() + (compute ? 0 : static_cast<u32>(sizeof(Shader::ProgramHeader)));
    Shader::Maxwell::Flow::CFG cfg(env, flow_blocks, cfg_offset, false);
    Shader::IR::Program program = compute ? TranslateComputeProgram(insts, blocks, env, cfg, cs.host)
                                          : Shader::Maxwell::TranslateProgram(insts, blocks, env, cfg, cs.host);
    RescalingPass(program);
    TranslatedShader out;
    if (compute) {
        out.spirv = Shader::Backend::SPIRV::EmitSPIRV(cs.profile, program);
    } else {
        Shader::Maxwell::ConvertLegacyToGeneric(program, runtime);
        Shader::Backend::Bindings bindings;
        SetRescaleBase(bindings, program.stage);
        out.spirv = Shader::Backend::SPIRV::EmitSPIRV(cs.profile, runtime, program, bindings);
    }
    out.info = program.info;
    return out;
}

} // namespace rt::nvn
