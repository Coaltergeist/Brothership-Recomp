// RT_NVN_SHADERTEST=<dir> (NVN_PLAN 7.2): translates every shader dumped in
// tap mode (research.cpp: <hash>.ctrl, <hash>.code) with the layer's own
// translation (shaders.cpp) and ends the run. Each stage is translated on its
// own, without the bound state: constant-buffer reads outside c1 answer 0 and
// every texture is taken as 2D; both are counted, as what 7.3 / 7.5 must
// supply. Summary per stage on the log ("shadertest:"); every failure's
// message, grouped. Compute shaders are also built into Vulkan compute
// pipelines on the layer's device (descriptor layouts by drippu's
// DescriptorLayoutBuilder, as its compute pipelines): the driver compiles the
// SPIR-V. RT_NVN_SHADERTEST_OUT=<dir> also writes <hash>.spv.
// drippu's headers first: vk.h brings X11's `Success` macro, which breaks its NvResult::Success.
#include "shader_recompiler/frontend/ir/program.h"
#include "video_core/renderer_vulkan/pipeline_helper.h"

#include "shaders.h"
#include "vk.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <thread>

namespace rt::nvn {

namespace {

std::vector<u8> ReadFile(const std::string& path) {
    std::vector<u8> v;
    if (FILE* f = std::fopen(path.c_str(), "rb")) {
        u8 buf[65536];
        for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) v.insert(v.end(), buf, buf + n);
        std::fclose(f);
    }
    return v;
}

/// RT_NVN_SHADERTEST_IR=<dir>: drippu's IR of a compute shader as its texture pass gets it (ComputeFrontEnd), for
/// reading why a bindless handle could not be tracked.
std::string ComputeIrBeforeTexturePass(NvnEnvironment& env) {
    Shader::HostTranslateInfo host = ShaderHostInfo();
    host.ApplyDescriptorLimitPolicy();
    Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_blocks;
    Shader::ObjectPool<Shader::IR::Inst> insts;
    Shader::ObjectPool<Shader::IR::Block> blocks;
    Shader::Maxwell::Flow::CFG cfg(env, flow_blocks, env.StartAddress(), false);
    return Shader::IR::DumpProgram(ComputeFrontEnd(insts, blocks, env, cfg, host));
}

const char* StageName(Stage s) {
    static const char* names[] = {"vertex", "fragment", "geometry", "tess control", "tess evaluation", "compute"};
    return names[static_cast<u32>(s)];
}

/// Builds a compute pipeline from the translation; the driver's error, or empty.
std::string BuildComputePipeline(const TranslatedShader& t) {
    try {
        const auto& device = *Vk().device;
        const auto& logical = device.GetLogical();
        Vulkan::vk::ShaderModule module = logical.CreateShaderModule({
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = t.spirv.size() * 4,
            .pCode = t.spirv.data(),
        });
        Vulkan::DescriptorLayoutBuilder builder{device};
        builder.Add(t.info, VK_SHADER_STAGE_COMPUTE_BIT);
        const bool push = builder.CanUsePushDescriptor();
        Vulkan::vk::DescriptorSetLayout set_layout = builder.CreateDescriptorSetLayout(push);
        Vulkan::vk::PipelineLayout layout = builder.CreatePipelineLayout(*set_layout);
        Vulkan::vk::Pipeline pipeline = logical.CreateComputePipeline(
            {
                .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .pNext = nullptr,
                .flags = 0,
                .stage{
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    .pNext = nullptr,
                    .flags = 0,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                    .module = *module,
                    .pName = "main",
                    .pSpecializationInfo = nullptr,
                },
                .layout = *layout,
                .basePipelineHandle = VK_NULL_HANDLE,
                .basePipelineIndex = 0,
            },
            VK_NULL_HANDLE);
        return {};
    } catch (const std::exception& e) {
        return e.what();
    }
}

// RT_NVN_SHADERTEST_GPL=1 (SHADER_PLAN: the measurements before choosing between the run-time route and the
// game files): every vertex and fragment stage is also built on its own into a Vulkan pipeline library
// (VK_EXT_graphics_pipeline_library) - vertex: pre-rasterisation shaders, fragment: fragment shader with its
// descriptors moved to set 1, so the two need no shared binding numbers (independent sets) - and fragments
// are translated with every generic input, no vertex stage being known. Then RT_NVN_SHADERTEST_LINKS pairs
// (default 200; vertex i with fragment i, not real pairs: only the link's time is measured) are linked with
// one vertex input and one fragment output library, fast and with link-time optimisation. Per-stage times on
// the log ("shadertest gpl:") and a summary. No pipeline cache: the driver's own disk cache still applies.
using Clock = std::chrono::steady_clock;

double MsSince(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

/// Mean, median, 95th percentile and maximum of `ms`.
std::string Spread(std::vector<double> ms) {
    if (ms.empty()) return "none";
    std::sort(ms.begin(), ms.end());
    double sum = 0;
    for (double m : ms) sum += m;
    char b[160];
    std::snprintf(b, sizeof b, "%zu: mean %.2f ms, median %.2f, p95 %.2f, max %.2f, total %.1f s", ms.size(),
                  sum / ms.size(), ms[ms.size() / 2], ms[ms.size() * 95 / 100], ms.back(), sum / 1000);
    return b;
}

class LibraryTest {
public:
    LibraryTest() : device_(*Vk().device), logical_(device_.GetLogical()) {
        if (!device_.IsGraphicsPipelineLibrarySupported()) Fatal("shadertest gpl: no graphics pipeline library with fast linking");
        const VkAttachmentDescription attachments[2] = {
            {0, VK_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE,
             VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL},
            {0, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE,
             VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL},
        };
        const VkAttachmentReference color{0, VK_IMAGE_LAYOUT_GENERAL}, depth{1, VK_IMAGE_LAYOUT_GENERAL};
        const VkSubpassDescription subpass{0, VK_PIPELINE_BIND_POINT_GRAPHICS, 0, nullptr, 1, &color, nullptr, &depth, 0, nullptr};
        pass_ = logical_.CreateRenderPass({
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .attachmentCount = 2,
            .pAttachments = attachments,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 0,
            .pDependencies = nullptr,
        });
        // RT_NVN_SHADERTEST_PASS_CHECK=1: each stage library is built twice more - for a render pass of other
        // formats (three colour targets, other depth) and for the first pass again - to see whether the driver's
        // cache of a library's compile depends on the render pass (SHADER_PLAN Phase 5: the files do not say
        // which pass a stage is drawn into)
        const VkAttachmentDescription other[4] = {
            {0, VK_FORMAT_R16G16B16A16_SFLOAT, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE,
             VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL},
            {0, VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE,
             VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL},
            {0, VK_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE,
             VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL},
            {0, VK_FORMAT_D24_UNORM_S8_UINT, VK_SAMPLE_COUNT_1_BIT, VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE,
             VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL},
        };
        const VkAttachmentReference other_colors[3] = {{0, VK_IMAGE_LAYOUT_GENERAL}, {1, VK_IMAGE_LAYOUT_GENERAL}, {2, VK_IMAGE_LAYOUT_GENERAL}};
        const VkAttachmentReference other_depth{3, VK_IMAGE_LAYOUT_GENERAL};
        const VkSubpassDescription other_subpass{0, VK_PIPELINE_BIND_POINT_GRAPHICS, 0, nullptr, 3, other_colors, nullptr, &other_depth, 0, nullptr};
        other_pass_ = logical_.CreateRenderPass({
            .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .attachmentCount = 4,
            .pAttachments = other,
            .subpassCount = 1,
            .pSubpasses = &other_subpass,
            .dependencyCount = 0,
            .pDependencies = nullptr,
        });
        link_layout_ = Layout(VK_NULL_HANDLE, VK_NULL_HANDLE); // for the two state-only parts
        // the two state-only parts every link takes
        const VkPipelineVertexInputStateCreateInfo vertex_input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        const VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                              nullptr, 0, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_FALSE};
        input_ = Library(VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT, [&](VkGraphicsPipelineCreateInfo& ci) {
            ci.pVertexInputState = &vertex_input;
            ci.pInputAssemblyState = &assembly;
        });
        const VkPipelineColorBlendAttachmentState blend{VK_FALSE, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD,
                                                        VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD, 15};
        const VkPipelineColorBlendStateCreateInfo blending{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                                           nullptr, 0, VK_FALSE, VK_LOGIC_OP_COPY, 1, &blend, {}};
        output_ = Library(VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT, [&](VkGraphicsPipelineCreateInfo& ci) {
            ci.pColorBlendState = &blending;
            ci.pMultisampleState = &multisample_;
        });
    }

    /// Builds a vertex or fragment stage into a library; its driver time in `ms`.
    void Add(Stage stage, TranslatedShader t, double& ms) {
        Part part;
        Vulkan::DescriptorLayoutBuilder builder{device_};
        const bool vertex = stage == Stage::kVertex;
        builder.Add(t.info, vertex ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT);
        part.set = builder.CreateDescriptorSetLayout(false);
        if (!vertex) MoveDescriptorSet(t.spirv, 1);
        part.layout = vertex ? Layout(*part.set, VK_NULL_HANDLE) : Layout(VK_NULL_HANDLE, *part.set);
        part.module = logical_.CreateShaderModule({
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .codeSize = t.spirv.size() * 4,
            .pCode = t.spirv.data(),
        });
        const VkPipelineShaderStageCreateInfo shader{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                                                     vertex ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT,
                                                     *part.module, "main", nullptr};
        const VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, nullptr, 0,
                                                         1, nullptr, 1, nullptr};
        const VkPipelineRasterizationStateCreateInfo raster{
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, nullptr, 0, VK_FALSE, VK_FALSE,
            VK_POLYGON_MODE_FILL, VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE, VK_FALSE, 0, 0, 0, 1.0f};
        const VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        const auto fill = [&](VkGraphicsPipelineCreateInfo& ci) {
                                    ci.stageCount = 1;
                                    ci.pStages = &shader;
                                    ci.layout = *part.layout;
                                    if (vertex) {
                                        ci.pViewportState = &viewport;
                                        ci.pRasterizationState = &raster;
                                    } else {
                                        ci.pDepthStencilState = &depth;
                                        ci.pMultisampleState = &multisample_;
                                    }
                                };
        const VkGraphicsPipelineLibraryFlagsEXT kind = vertex ? VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT
                                                              : VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT;
        // RT_NVN_SHADERTEST_OTHER_PASS=1: the first build too for the other render pass (a later process: whether
        // the driver's disk cache of the first pass's builds serves another pass)
        static const bool other_first = std::getenv("RT_NVN_SHADERTEST_OTHER_PASS") != nullptr;
        const auto t0 = Clock::now();
        part.pipeline = Library(kind, fill, other_first ? *other_pass_ : VK_NULL_HANDLE);
        ms = MsSince(t0);
        if (std::getenv("RT_NVN_SHADERTEST_PASS_CHECK")) {
            const auto t1 = Clock::now();
            Vulkan::vk::Pipeline other_pass = Library(kind, fill, *other_pass_);
            const double other_ms = MsSince(t1);
            const auto t2 = Clock::now();
            Vulkan::vk::Pipeline again = Library(kind, fill);
            const double again_ms = MsSince(t2);
            std::lock_guard lock(mu_);
            other_pass_ms_.push_back(other_ms);
            same_pass_ms_.push_back(again_ms);
        }
        std::lock_guard lock(mu_); // Add may run on several threads (RT_NVN_SHADERTEST_THREADS)
        (vertex ? vertex_ : fragment_).push_back(std::move(part));
    }

    /// Links up to `count` vertex/fragment pairs, fast and optimised; logs both spreads.
    void Link(size_t count) {
        count = std::min({count, vertex_.size(), fragment_.size()});
        std::vector<double> fast, optimised;
        for (size_t i = 0; i < count; ++i) {
            const Vulkan::vk::PipelineLayout layout = Layout(*vertex_[i].set, *fragment_[i].set); // the union
            for (const bool lto : {false, true}) {
                const VkPipeline parts[4] = {*input_, *vertex_[i].pipeline, *fragment_[i].pipeline, *output_};
                const VkPipelineLibraryCreateInfoKHR libraries{VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR, nullptr, 4, parts};
                const VkGraphicsPipelineCreateInfo ci{
                    .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    .pNext = &libraries,
                    .flags = lto ? VkPipelineCreateFlags{VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT} : 0u,
                    .layout = *layout,
                    .renderPass = *pass_,
                };
                const auto t0 = Clock::now();
                Vulkan::vk::Pipeline linked = logical_.CreateGraphicsPipeline(ci, VK_NULL_HANDLE);
                (lto ? optimised : fast).push_back(MsSince(t0));
            }
        }
        if (!other_pass_ms_.empty()) {
            Log("shadertest gpl: the same stage libraries for another render pass %s", Spread(other_pass_ms_).c_str());
            Log("shadertest gpl: ... and for the first render pass again %s", Spread(same_pass_ms_).c_str());
        }
        Log("shadertest gpl: fast links %s", Spread(fast).c_str());
        Log("shadertest gpl: optimised links %s", Spread(optimised).c_str());
    }

private:
    struct Part {
        Vulkan::vk::DescriptorSetLayout set;
        Vulkan::vk::PipelineLayout layout;
        Vulkan::vk::ShaderModule module;
        Vulkan::vk::Pipeline pipeline;
    };

    /// A layout for independent sets: set 0 the vertex stage's, set 1 the fragment stage's; drippu's push
    /// constants (as DescriptorLayoutBuilder::CreatePipelineLayout).
    Vulkan::vk::PipelineLayout Layout(VkDescriptorSetLayout vertex, VkDescriptorSetLayout fragment) {
        using Shader::Backend::SPIRV::RenderAreaLayout;
        using Shader::Backend::SPIRV::RescalingLayout;
        const VkDescriptorSetLayout sets[2] = {vertex, fragment};
        const VkPushConstantRange range{VK_SHADER_STAGE_ALL_GRAPHICS, 0,
                                        static_cast<u32>(sizeof(RescalingLayout) + sizeof(RenderAreaLayout))};
        return logical_.CreatePipelineLayout({
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT,
            .setLayoutCount = 2,
            .pSetLayouts = sets,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        });
    }

    /// Creates a library of the given part; `fill` adds that part's state.
    template <typename F>
    Vulkan::vk::Pipeline Library(VkGraphicsPipelineLibraryFlagsEXT flags, F&& fill, VkRenderPass pass = VK_NULL_HANDLE) {
        const VkGraphicsPipelineLibraryCreateInfoEXT info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT, nullptr, flags};
        const VkDynamicState states[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        const VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, nullptr, 0, 2, states};
        VkGraphicsPipelineCreateInfo ci{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = &info,
            .flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR | VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT,
            .pDynamicState = &dynamic,
            .layout = *link_layout_,
            .renderPass = pass ? pass : *pass_,
        };
        fill(ci);
        return logical_.CreateGraphicsPipeline(ci, VK_NULL_HANDLE);
    }

    const Vulkan::Device& device_;
    const Vulkan::vk::Device& logical_;
    Vulkan::vk::RenderPass pass_, other_pass_;
    std::vector<double> other_pass_ms_, same_pass_ms_; // RT_NVN_SHADERTEST_PASS_CHECK
    Vulkan::vk::PipelineLayout link_layout_;
    Vulkan::vk::Pipeline input_, output_;
    std::mutex mu_;
    std::vector<Part> vertex_, fragment_;
    const VkPipelineMultisampleStateCreateInfo multisample_{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                                            nullptr, 0, VK_SAMPLE_COUNT_1_BIT};
};

} // namespace

void RunShaderTest(const char* dir) {
    std::vector<std::string> hashes;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        const std::string n = e.path().filename().string();
        if (n.size() > 5 && n.ends_with(".ctrl")) hashes.push_back(n.substr(0, n.size() - 5));
    }
    std::sort(hashes.begin(), hashes.end());
    if (hashes.empty()) Fatal("shadertest: no <hash>.ctrl files in %s", dir);
    const char* out_dir = std::getenv("RT_NVN_SHADERTEST_OUT");
    Log("shadertest: %zu shaders from %s", hashes.size(), dir);

    struct Count {
        u32 total = 0, ok = 0;
        u64 spirv_words = 0;
    };
    std::map<Stage, Count> per_stage;
    std::map<std::string, std::vector<std::string>> failures; // message -> hashes
    std::map<u32, u32> cbuf_banks;                          // bank -> shaders reading it at translation
    u32 textured = 0, nvn_storage = 0, compiled = 0;
    u64 texture_handles = 0;
    std::optional<LibraryTest> gpl;
    if (std::getenv("RT_NVN_SHADERTEST_GPL")) gpl.emplace();
    std::map<Stage, std::vector<double>> translate_ms, driver_ms; // per stage (GPL mode)
    // RT_NVN_SHADERTEST_THREADS=<n> (GPL mode): the driver's builds after all translations (which stay on
    // this thread), on n threads, with the wall time of the whole batch - what preparing at start would take
    const char* threads_env = std::getenv("RT_NVN_SHADERTEST_THREADS");
    const u32 threads = gpl && threads_env ? std::max(1ul, std::strtoul(threads_env, nullptr, 10)) : 0;
    struct Pending {
        Stage stage;
        TranslatedShader t;
    };
    std::vector<Pending> pending;
    for (const std::string& h : hashes) {
        const std::vector<u8> ctrl = ReadFile(std::string(dir) + "/" + h + ".ctrl");
        std::vector<u8> code = ReadFile(std::string(dir) + "/" + h + ".code");
        std::string why;
        if (ctrl.size() < kControlSize) {
            failures["control section shorter than 0x800 bytes"].push_back(h);
            continue;
        }
        const auto s = ParseControlBytes(ctrl.data(), 0, why);
        if (!s) {
            failures["not parsed: " + why].push_back(h);
            continue;
        }
        Count& count = per_stage[s->stage];
        ++count.total;
        std::set<u32> banks;
        std::set<u32> handles;
        ShaderHooks hooks{
            .cbuf = [&](u32 bank, u32) -> u32 {
                banks.insert(bank);
                return 0;
            },
            .texture_type = [&](u32 handle) {
                handles.insert(handle);
                return Shader::TextureType::Color2D;
            },
        };
        try {
            NvnEnvironment env(std::move(code), *s, hooks);
            Shader::RuntimeInfo runtime{};
            if (gpl) runtime.previous_stage_stores.mask.set(); // no previous stage known: every input
            const auto t0 = Clock::now();
            const TranslatedShader t = TranslateStage(env, runtime);
            const double translated = MsSince(t0);
            double driver = -1;
            if (threads && (s->stage == Stage::kVertex || s->stage == Stage::kFragment || s->stage == Stage::kCompute)) {
                pending.push_back({s->stage, t}); // built below
            } else if (s->stage == Stage::kCompute) {
                const auto t1 = Clock::now();
                if (const std::string err = BuildComputePipeline(t); !err.empty()) {
                    failures["compute pipeline: " + err].push_back(h);
                    continue;
                }
                driver = MsSince(t1);
                ++compiled;
            } else if (gpl && (s->stage == Stage::kVertex || s->stage == Stage::kFragment)) {
                try {
                    gpl->Add(s->stage, t, driver);
                } catch (const std::exception& e) {
                    failures[std::string(StageName(s->stage)) + " library: " + e.what()].push_back(h);
                    continue;
                }
            }
            if (gpl) {
                translate_ms[s->stage].push_back(translated);
                if (driver >= 0) driver_ms[s->stage].push_back(driver);
                Log("shadertest gpl: %s %s translation %.2f ms, driver %.2f ms, %zu SPIR-V words", h.c_str(),
                    StageName(s->stage), translated, driver, t.spirv.size());
            }
            ++count.ok;
            count.spirv_words += t.spirv.size();
            if (!t.info.texture_descriptors.empty()) ++textured;
            if (t.info.nvn_buffer_used.any()) ++nvn_storage;
            texture_handles += handles.size();
            for (u32 b : banks) ++cbuf_banks[b];
            if (out_dir) {
                if (FILE* f = std::fopen((std::string(out_dir) + "/" + h + ".spv").c_str(), "wb")) {
                    std::fwrite(t.spirv.data(), 4, t.spirv.size(), f);
                    std::fclose(f);
                }
            }
        } catch (const std::exception& e) {
            failures[std::string(StageName(s->stage)) + ": " + e.what()].push_back(h);
            if (const char* ir_dir = std::getenv("RT_NVN_SHADERTEST_IR"); ir_dir && s->stage == Stage::kCompute) {
                try {
                    NvnEnvironment env(ReadFile(std::string(dir) + "/" + h + ".code"), *s, hooks);
                    std::filesystem::create_directories(ir_dir);
                    if (FILE* f = std::fopen((std::string(ir_dir) + "/" + h + ".ir.txt").c_str(), "w")) {
                        const std::string ir = ComputeIrBeforeTexturePass(env);
                        std::fwrite(ir.data(), 1, ir.size(), f);
                        std::fclose(f);
                    }
                } catch (const std::exception& e2) {
                    Log("shadertest: %s: no IR: %s", h.c_str(), e2.what());
                }
            }
        }
    }
    u32 total = 0, ok = 0;
    for (const auto& [stage, n] : per_stage) {
        Log("shadertest: %-15s %4u of %4u translated, %llu SPIR-V words on average", StageName(stage), n.ok, n.total,
            (unsigned long long)(n.ok ? n.spirv_words / n.ok : 0));
        total += n.total, ok += n.ok;
    }
    for (const auto& [msg, hs] : failures) {
        Log("shadertest: FAILED %zux: %s (e.g. %s)", hs.size(), msg.c_str(), hs.front().c_str());
        for (size_t i = 0; i < hs.size() && i < 100; ++i) Log("shadertest: failed: %s", hs[i].c_str());
    }
    std::string banks;
    for (const auto& [b, n] : cbuf_banks) banks += " c" + std::to_string(b) + ":" + std::to_string(n);
    Log("shadertest: translated shaders reading constant buffers at translation time (bank:shaders):%s",
        banks.empty() ? " none" : banks.c_str());
    Log("shadertest: %u with textures (%llu texture handles asked), %u using NVN storage buffers", textured,
        (unsigned long long)texture_handles, nvn_storage);
    Log("shadertest: %u compute pipelines built by the Vulkan driver", compiled);
    if (threads) {
        std::atomic<size_t> next{0};
        std::mutex mu;
        u32 failed = 0;
        const auto t0 = Clock::now();
        std::vector<std::thread> workers;
        for (u32 i = 0; i < threads; ++i) {
            workers.emplace_back([&] {
                for (size_t j; (j = next.fetch_add(1)) < pending.size();) {
                    Pending& job = pending[j];
                    double ms = -1;
                    try {
                        if (job.stage == Stage::kCompute) {
                            const auto t1 = Clock::now();
                            if (!BuildComputePipeline(job.t).empty()) throw std::runtime_error("compute pipeline");
                            ms = MsSince(t1);
                        } else {
                            gpl->Add(job.stage, std::move(job.t), ms);
                        }
                    } catch (const std::exception&) {
                        std::lock_guard lock(mu);
                        ++failed;
                        continue;
                    }
                    std::lock_guard lock(mu);
                    driver_ms[job.stage].push_back(ms);
                }
            });
        }
        for (auto& w : workers) w.join();
        Log("shadertest gpl: %zu driver builds on %u threads in %.1f s (wall), %u failed", pending.size(), threads,
            MsSince(t0) / 1000, failed);
    }
    if (gpl) {
        for (const auto& [stage, ms] : translate_ms) {
            Log("shadertest gpl: %s translation %s", StageName(stage), Spread(ms).c_str());
            Log("shadertest gpl: %s driver (%s) %s", StageName(stage),
                stage == Stage::kCompute ? "whole pipeline" : "library", Spread(driver_ms[stage]).c_str());
        }
        const char* links = std::getenv("RT_NVN_SHADERTEST_LINKS");
        gpl->Link(links ? std::strtoul(links, nullptr, 10) : 200);
    }
    Log("shadertest: %u of %u translated", ok, total);
    Exit(ok == total ? 0 : 1);
}

} // namespace rt::nvn
