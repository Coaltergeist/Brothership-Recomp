// Graphics pipelines from the queue's bound state, for draws.
#pragma once

#include "render.h"

#include <atomic>
#include <memory>
#include <stdexcept>
#include <vector>

#include "shader_recompiler/shader_info.h"

namespace rt::nvn {

/// State the draw path does not handle yet: with RT_NVN_SKIP_DRAWS=1 the draw is skipped (and the reason
/// logged once), otherwise the run stops with it.
struct NotYet : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// A draw whose pipeline is still being built on a worker (pipeline.cpp PipelineFor): skipped
/// whatever RT_NVN_SKIP_DRAWS says, and counted apart from the draws skipped for what the layer cannot do.
struct NotReady : NotYet {
    using NotYet::NotYet;
};

struct StageBuild;
struct PipelineLibrary;

/// A Vulkan graphics pipeline for one combination of programs, fixed-function state and target formats.
struct GraphicsPipeline {
    struct Stage {
        u32 nvn_stage; // Stage
        ProgramStage program;
        Shader::Info info;
        u32 set = 0;                             // its descriptor set (stages built alone: fragment 0, vertex 1, geometry 2)
        std::shared_ptr<const StageBuild> built; // its translation on its own (RT_NVN_STAGE_ALONE), or null
    };
    std::vector<Stage> stages; // in pipeline (and descriptor binding) order
    std::vector<Vulkan::vk::ShaderModule> modules;
    Vulkan::vk::DescriptorSetLayout set_layout;
    Vulkan::vk::PipelineLayout layout;
    Vulkan::vk::DescriptorUpdateTemplate update; // in drippu's DescriptorLayoutBuilder order
    Vulkan::vk::Pipeline pipeline;
    // Linked from pipeline libraries (RT_NVN_GPL, pipeline.cpp): the four parts, the state set at each draw
    // instead of baked in, and the optimised re-link once a worker made it (then drawn instead of `pipeline`).
    std::vector<std::shared_ptr<const PipelineLibrary>> libraries;
    struct Dynamic {
        VkCullModeFlags cull = 0;
        VkFrontFace front = VK_FRONT_FACE_CLOCKWISE;
        bool bias = false;
        float bias_constant = 0, bias_clamp = 0, bias_slope = 0;
        bool depth_test = false, depth_write = false, depth_bounds = false, stencil_test = false;
        VkCompareOp depth_compare = VK_COMPARE_OP_ALWAYS;
        VkStencilOpState front_ops{}, back_ops{};
    } dynamic;
    bool linked = false;
    Vulkan::vk::Pipeline optimised_owner;
    std::atomic<VkPipeline> optimised{VK_NULL_HANDLE};
    bool push = true; // descriptors pushed; false: more than the device pushes, a set per draw (DescriptorArena)
    bool uses_render_area = false;
    u32 vertex_streams = 0; // the streams of the attributes the vertex shader loads (bound at each draw)

    // What the translation read from the bound state (constant-buffer words, the types of bound textures):
    // the pipeline is reused while all of it still holds, as drippu's environments are replayed.
    struct CbufRead {
        u32 nvn_stage, bank, offset, value;
    };
    struct TypeRead {
        u32 nvn_stage, handle;
        Shader::TextureType type;
        // where the handle came from (TypeRecorder): `words` constant-buffer words, OR-ed as drippu combines a
        // texture and a sampler word; 0: not found, checked by `handle` itself
        u32 words = 0, bank = 0, offset = 0, bank2 = 0, offset2 = 0;
        bool registered = true; // the texture pool had the handle's texture when translated (else answered 2D)
    };
    std::vector<CbufRead> cbuf_reads;
    std::vector<TypeRead> type_reads;
    std::string failed; // why its build failed (NotYet): a remembered failure, not a pipeline
    // A few words of each stage's code memory when built (pipeline.cpp Fingerprint): pipelines are found by the
    // code's address, so other code loaded there later must not reuse this one.
    std::vector<std::array<u64, 4>> fingerprints;
    bool stale = false; // other code was found at its addresses: never used again (kept: the GPU may still use it)
    bool building = false; // on a worker (RT_NVN_PIPELINE_WAIT): its other fields are not there yet; under the cache's lock
};

/// One stage translated on its own (pipeline.cpp StageFor), shared by every pipeline that
/// uses it: its module and its own descriptor set. A fragment stage's set is pushed when it fits (set 0, with
/// `push_update` made for a layout of that set alone, compatible with every pipeline's); other sets come from
/// the submission's DescriptorArena (`set_update`).
struct StageBuild {
    u32 nvn_stage = 0;
    Shader::Info info;
    Vulkan::vk::ShaderModule module;
    Vulkan::vk::DescriptorSetLayout set_layout; // null: no descriptors
    Vulkan::vk::PipelineLayout push_layout;
    Vulkan::vk::DescriptorUpdateTemplate push_update, set_update;
    bool push = false;
    std::vector<GraphicsPipeline::CbufRead> cbuf_reads; // what its translation read (as GraphicsPipeline's)
    std::vector<GraphicsPipeline::TypeRead> type_reads;
};

/// A Vulkan compute pipeline for the bound compute program.
struct ComputePipeline {
    ProgramStage program;
    Shader::Info info;
    Vulkan::vk::ShaderModule module;
    Vulkan::vk::DescriptorSetLayout set_layout;
    Vulkan::vk::PipelineLayout layout;
    Vulkan::vk::DescriptorUpdateTemplate update;
    Vulkan::vk::Pipeline pipeline;
    bool push = true; // as GraphicsPipeline's
    std::vector<GraphicsPipeline::CbufRead> cbuf_reads; // as GraphicsPipeline's
    std::vector<GraphicsPipeline::TypeRead> type_reads;
    std::string failed; // as GraphicsPipeline's
    u64 hash = 0;       // its code's hash, as the "new compute pipeline" lines print it
};

/// The pipeline for the queue's bound compute program (rs.compute_stage), created at first use. Throws NotYet.
const ComputePipeline& ComputePipelineFor(const RenderState& rs);

/// The pipeline for the queue's bound state, primitive `mode` (NVNdrawPrimitive) and targets `targets`;
/// created (shaders translated by drippu's recompiler) at first use. Throws NotYet.
const GraphicsPipeline& PipelineFor(const RenderState& rs, u32 mode, const TargetSet& targets);

/// A stage's content hash, as the "new pipeline" log lines give it (pipeline.cpp StageHash).
u64 StageContentHash(const ProgramStage& s);

/// The type drippu's recompiler expects for the texture behind NVN handle `handle` in `rs`'s texture pool.
Shader::TextureType TextureTypeOf(const RenderState& rs, u32 handle, bool* registered = nullptr);

/// A word of constant buffer `bank` (0 / 2: the driver's shadows, 3 + n: the game's uniform buffer n) as
/// stage `nvn_stage` sees it now. Not for c1 (the stage's compiler constants).
u32 CbufWord(const RenderState& rs, u32 nvn_stage, u32 bank, u32 offset);

/// Compute's c0 for the queue's compute bindings and the group counts in rs.dispatch (render.h).
std::array<u8, kComputeC0Size> ComputeC0(const RenderState& rs);

/// The Vulkan format for NVN vertex attribute format `id` (LASRecomp's table), or VK_FORMAT_UNDEFINED.
VkFormat VertexFormat(u32 id);

} // namespace rt::nvn
