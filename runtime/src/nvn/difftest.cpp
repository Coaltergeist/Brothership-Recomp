// RT_NVN_DIFFTEST=1 (layer.h): the layer's pure functions against the game's
// own SDK. For each state object and builder: one object in scratch memory for
// the SDK's functions and one for the layer's; SetDefaults on both, then a
// list of setter calls, and after each step every getter is called on both
// and the results compared (return value and out parameters). Runs on the
// thread of the game's first NVN call, logs each difference, and ends the run
// (exit status 1 if any). Enum values below are NVN's public ones.
#include "images.h"
#include "nvn.h"

#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <vector>

namespace rt::nvn {

namespace {

struct Step {
    const char* fn;            // a setter (or SetDefaults again)
    std::vector<u64> args;     // after the object
    std::vector<float> fargs{}; // in v0, v1 ...
};

struct Getter {
    const char* fn;
    std::vector<u64> args;  // after the object, before the out parameters
    int outs = 0;           // out parameters ...
    int out_size = 4;       // ... of this size each
    int ret_size = 0;       // bytes of x0 that count (0: void)
    bool ret_float = false; // the result is a float in v0 instead
};

struct Suite {
    const char* name;
    const char* defaults;
    std::vector<Step> setup; // before SetDefaults (e.g. the device a builder needs)
    std::vector<Step> steps;
    std::vector<Getter> getters;
};

constexpr u64 kObjSize = 0x200;
constexpr u64 kSdkObj = Scratch::kObjects, kOurObj = Scratch::kObjects + kObjSize;
constexpr u64 kSdkOut = Scratch::kOut, kOurOut = Scratch::kOut + 0x100;
constexpr u8 kFill = 0xA5; // out parameters a getter leaves untouched read back as this

u64 g_compared = 0, g_differences = 0;

u64 Call(GuestContext* c, bool sdk, const char* fn, u64 obj, const std::vector<u64>& args, u64 outs = 0, int n_outs = 0,
         int out_size = 0, const std::vector<float>& fargs = {}, u64* v0 = nullptr) {
    std::vector<u64> a{obj};
    a.insert(a.end(), args.begin(), args.end());
    for (int i = 0; i < n_outs; ++i) a.push_back(outs + static_cast<u64>(i * out_size));
    if (a.size() > 8) Fatal("difftest: %s has more than 8 arguments", fn);
    const u64 target = sdk ? RealProc(c, fn) : Lookup(fn);
    if (!target) Fatal("difftest: %s has no %s function", fn, sdk ? "SDK" : "layer");
    u64 x[8] = {}, f[8] = {};
    for (size_t i = 0; i < a.size(); ++i) x[i] = a[i];
    if (fargs.size() > 8) Fatal("difftest: %s has more than 8 float arguments", fn);
    for (size_t i = 0; i < fargs.size(); ++i) f[i] = std::bit_cast<u32>(fargs[i]);
    return hle::CallGuestFp(c, target, {x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]},
                            {f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]}, v0, sdk);
}

std::string Show(const char* fn, const std::vector<u64>& args, const std::vector<float>& fargs = {}) {
    std::string s = fn;
    s += "(";
    for (size_t i = 0; i < args.size(); ++i) {
        char b[24];
        std::snprintf(b, sizeof b, "%s0x%llx", i ? ", " : "", (unsigned long long)args[i]);
        s += b;
    }
    for (size_t i = 0; i < fargs.size(); ++i) {
        char b[32];
        std::snprintf(b, sizeof b, "%s%g", i || !args.empty() ? ", " : "", static_cast<double>(fargs[i]));
        s += b;
    }
    return s + ")";
}

std::string Result(u64 ret, int ret_size, const u8* out, int outs, int out_size) {
    std::string s;
    char b[40];
    if (ret_size) {
        const u64 v = ret_size == 8 ? ret : ret & ((1ULL << (8 * ret_size)) - 1);
        std::snprintf(b, sizeof b, "ret 0x%llx", (unsigned long long)v);
        s += b;
    }
    for (int i = 0; i < outs; ++i) {
        if (out_size > 8) { // bytes, in memory order
            std::snprintf(b, sizeof b, "%sout%d ", s.empty() ? "" : " ", i);
            s += b;
            for (int k = 0; k < out_size; ++k) {
                std::snprintf(b, sizeof b, "%02x", out[i * out_size + k]);
                s += b;
            }
            continue;
        }
        u64 v = 0;
        std::memcpy(&v, out + i * out_size, static_cast<size_t>(out_size));
        std::snprintf(b, sizeof b, "%sout%d 0x%llx", s.empty() ? "" : " ", i, (unsigned long long)v);
        s += b;
    }
    return s;
}

void CompareGetters(GuestContext* c, const Suite& s, const std::string& after) {
    for (const Getter& g : s.getters) {
        std::string res[2];
        for (int sdk = 1; sdk >= 0; --sdk) {
            const u64 out = sdk ? kSdkOut : kOurOut;
            std::vector<u8> fill(0x100, kFill);
            g_mem.Write(out, fill.data(), fill.size());
            u64 v0 = 0;
            const u64 ret = Call(c, sdk, g.fn, sdk ? kSdkObj : kOurObj, g.args, out, g.outs, g.out_size, {}, &v0);
            std::vector<u8> got(0x100);
            g_mem.Read(out, got.data(), got.size());
            res[sdk] = Result(g.ret_float ? v0 : ret, g.ret_float ? 4 : g.ret_size, got.data(), g.outs, g.out_size);
        }
        ++g_compared;
        if (res[0] != res[1]) {
            ++g_differences;
            Log("difftest: %s after %s: %s: SDK %s, layer %s", s.name, after.c_str(), Show(g.fn, g.args).c_str(),
                res[1].c_str(), res[0].c_str());
        }
    }
}

void RunSuite(GuestContext* c, const Suite& s) {
    const u64 before = g_differences;
    const std::vector<u8> zero(kObjSize, 0);
    g_mem.Write(kSdkObj, zero.data(), zero.size());
    g_mem.Write(kOurObj, zero.data(), zero.size());
    for (const Step& st : s.setup) {
        Call(c, true, st.fn, kSdkObj, st.args);
        Call(c, false, st.fn, kOurObj, st.args);
    }
    Call(c, true, s.defaults, kSdkObj, {});
    Call(c, false, s.defaults, kOurObj, {});
    CompareGetters(c, s, "SetDefaults");
    for (const Step& st : s.steps) {
        Call(c, true, st.fn, kSdkObj, st.args, 0, 0, 0, st.fargs);
        Call(c, false, st.fn, kOurObj, st.args, 0, 0, 0, st.fargs);
        CompareGetters(c, s, Show(st.fn, st.args, st.fargs));
    }
    Log("difftest: %s: %llu differences", s.name, (unsigned long long)(g_differences - before));
}

std::vector<Suite> Suites() {
    const u64 dev = Scratch::kDevice; // a zeroed stand-in where an object takes a device
    std::vector<Suite> v;

    std::vector<Getter> blend = {
        {"nvnBlendStateGetBlendTarget", {}, 0, 4, 4},      {"nvnBlendStateGetBlendFunc", {}, 4, 4, 0},
        {"nvnBlendStateGetBlendEquation", {}, 2, 4, 0},    {"nvnBlendStateGetAdvancedMode", {}, 0, 4, 4},
        {"nvnBlendStateGetAdvancedOverlap", {}, 0, 4, 4},  {"nvnBlendStateGetAdvancedPremultipliedSrc", {}, 0, 4, 1},
        {"nvnBlendStateGetAdvancedNormalizedDst", {}, 0, 4, 1},
    };
    v.push_back({"BlendState", "nvnBlendStateSetDefaults", {},
                 {{"nvnBlendStateSetBlendTarget", {3}},
                  {"nvnBlendStateSetBlendTarget", {7}},
                  {"nvnBlendStateSetBlendFunc", {5, 6, 0xA, 0x13}},
                  {"nvnBlendStateSetBlendFunc", {0x61, 0x62, 0x63, 0x64}},
                  {"nvnBlendStateSetBlendEquation", {2, 5}},
                  {"nvnBlendStateSetAdvancedMode", {0x2A}},
                  {"nvnBlendStateSetAdvancedOverlap", {2}},
                  {"nvnBlendStateSetAdvancedPremultipliedSrc", {0}},
                  {"nvnBlendStateSetAdvancedNormalizedDst", {0}},
                  {"nvnBlendStateSetAdvancedMode", {0}}},
                 blend});

    std::vector<Getter> color = {{"nvnColorStateGetLogicOp", {}, 0, 4, 4}, {"nvnColorStateGetAlphaTest", {}, 0, 4, 4}};
    for (u64 i = 0; i < 8; ++i) color.push_back({"nvnColorStateGetBlendEnable", {i}, 0, 4, 1});
    v.push_back({"ColorState", "nvnColorStateSetDefaults", {},
                 {{"nvnColorStateSetBlendEnable", {0, 1}},
                  {"nvnColorStateSetBlendEnable", {5, 1}},
                  {"nvnColorStateSetBlendEnable", {0, 0}},
                  {"nvnColorStateSetLogicOp", {0xF}},
                  {"nvnColorStateSetAlphaTest", {4}}},
                 color});

    std::vector<Getter> mask;
    for (u64 i = 0; i < 8; ++i) mask.push_back({"nvnChannelMaskStateGetChannelMask", {i}, 4, 1, 0});
    v.push_back({"ChannelMaskState", "nvnChannelMaskStateSetDefaults", {},
                 {{"nvnChannelMaskStateSetChannelMask", {2, 1, 0, 1, 0}},
                  {"nvnChannelMaskStateSetChannelMask", {7, 0, 0, 0, 1}},
                  {"nvnChannelMaskStateSetChannelMask", {0, 0, 0, 0, 0}}},
                 mask});

    std::vector<Getter> ds = {
        {"nvnDepthStencilStateGetDepthTestEnable", {}, 0, 4, 1},
        {"nvnDepthStencilStateGetDepthWriteEnable", {}, 0, 4, 1},
        {"nvnDepthStencilStateGetStencilTestEnable", {}, 0, 4, 1},
        {"nvnDepthStencilStateGetDepthFunc", {}, 0, 4, 4},
    };
    for (u64 face = 1; face <= 3; ++face) {
        ds.push_back({"nvnDepthStencilStateGetStencilFunc", {face}, 0, 4, 4});
        ds.push_back({"nvnDepthStencilStateGetStencilOp", {face}, 3, 4, 0});
    }
    v.push_back({"DepthStencilState", "nvnDepthStencilStateSetDefaults", {},
                 {{"nvnDepthStencilStateSetDepthTestEnable", {1}},
                  {"nvnDepthStencilStateSetDepthWriteEnable", {0}},
                  {"nvnDepthStencilStateSetDepthFunc", {4}},
                  {"nvnDepthStencilStateSetStencilTestEnable", {1}},
                  {"nvnDepthStencilStateSetStencilFunc", {1, 3}},
                  {"nvnDepthStencilStateSetStencilFunc", {2, 6}},
                  {"nvnDepthStencilStateSetStencilFunc", {3, 5}},
                  {"nvnDepthStencilStateSetStencilOp", {1, 2, 3, 4}},
                  {"nvnDepthStencilStateSetStencilOp", {2, 8, 7, 6}},
                  {"nvnDepthStencilStateSetStencilOp", {3, 5, 1, 2}}},
                 ds});

    v.push_back({"MultisampleState", "nvnMultisampleStateSetDefaults", {},
                 {{"nvnMultisampleStateSetMultisampleEnable", {0}},
                  {"nvnMultisampleStateSetSamples", {4}},
                  {"nvnMultisampleStateSetRasterSamples", {8}},
                  {"nvnMultisampleStateSetAlphaToCoverageEnable", {1}},
                  {"nvnMultisampleStateSetAlphaToCoverageDither", {0}},
                  {"nvnMultisampleStateSetCoverageToColorEnable", {1}},
                  {"nvnMultisampleStateSetCoverageToColorOutput", {3}},
                  {"nvnMultisampleStateSetCoverageModulationMode", {2}},
                  {"nvnMultisampleStateSetSampleLocationsEnable", {1}},
                  {"nvnMultisampleStateSetSampleLocationsGridEnable", {1}}},
                 {{"nvnMultisampleStateGetMultisampleEnable", {}, 0, 4, 1},
                  {"nvnMultisampleStateGetSamples", {}, 0, 4, 4},
                  {"nvnMultisampleStateGetRasterSamples", {}, 0, 4, 4},
                  {"nvnMultisampleStateGetAlphaToCoverageEnable", {}, 0, 4, 1},
                  {"nvnMultisampleStateGetAlphaToCoverageDither", {}, 0, 4, 1},
                  {"nvnMultisampleStateGetCoverageToColorEnable", {}, 0, 4, 1},
                  {"nvnMultisampleStateGetCoverageToColorOutput", {}, 0, 4, 4},
                  {"nvnMultisampleStateGetCoverageModulationMode", {}, 0, 4, 4},
                  {"nvnMultisampleStateGetSampleLocationsEnable", {}, 0, 4, 1},
                  {"nvnMultisampleStateGetSampleLocationsGridEnable", {}, 0, 4, 1}}});

    v.push_back({"PolygonState", "nvnPolygonStateSetDefaults", {},
                 {{"nvnPolygonStateSetCullFace", {2}},
                  {"nvnPolygonStateSetCullFace", {3}},
                  {"nvnPolygonStateSetFrontFace", {0}},
                  {"nvnPolygonStateSetPolygonMode", {1}},
                  {"nvnPolygonStateSetPolygonOffsetEnables", {7}}},
                 {{"nvnPolygonStateGetCullFace", {}, 0, 4, 4},
                  {"nvnPolygonStateGetFrontFace", {}, 0, 4, 4},
                  {"nvnPolygonStateGetPolygonMode", {}, 0, 4, 4},
                  {"nvnPolygonStateGetPolygonOffsetEnables", {}, 0, 4, 4}}});

    v.push_back({"VertexAttribState", "nvnVertexAttribStateSetDefaults", {},
                 {{"nvnVertexAttribStateSetFormat", {0x25, 12}},
                  {"nvnVertexAttribStateSetStreamIndex", {15}},
                  {"nvnVertexAttribStateSetFormat", {0x4D, 0x7FF}},
                  {"nvnVertexAttribStateSetStreamIndex", {3}}},
                 {{"nvnVertexAttribStateGetFormat", {}, 2, 8, 0},
                  {"nvnVertexAttribStateGetStreamIndex", {}, 0, 4, 4}}});

    v.push_back({"VertexStreamState", "nvnVertexStreamStateSetDefaults", {},
                 {{"nvnVertexStreamStateSetStride", {64}},
                  {"nvnVertexStreamStateSetStride", {0x800}},
                  {"nvnVertexStreamStateSetDivisor", {5}}},
                 {{"nvnVertexStreamStateGetStride", {}, 0, 4, 8}, {"nvnVertexStreamStateGetDivisor", {}, 0, 4, 4}}});

    v.push_back({"DeviceBuilder", "nvnDeviceBuilderSetDefaults", {},
                 {{"nvnDeviceBuilderSetFlags", {0x5}}, {"nvnDeviceBuilderSetFlags", {0x1F}}},
                 {{"nvnDeviceBuilderGetFlags", {}, 0, 4, 4}}});

    v.push_back({"MemoryPoolBuilder", "nvnMemoryPoolBuilderSetDefaults", {{"nvnMemoryPoolBuilderSetDevice", {dev}}},
                 {{"nvnMemoryPoolBuilderSetFlags", {0xe2}},
                  {"nvnMemoryPoolBuilderSetStorage", {0x40001000, 0x200000}},
                  {"nvnMemoryPoolBuilderSetDefaults", {}},
                  {"nvnMemoryPoolBuilderSetFlags", {0x18a}},
                  {"nvnMemoryPoolBuilderSetStorage", {0x40800000, 0x1000}}},
                 {{"nvnMemoryPoolBuilderGetDevice", {}, 0, 4, 8},
                  {"nvnMemoryPoolBuilderGetFlags", {}, 0, 4, 4},
                  {"nvnMemoryPoolBuilderGetMemory", {}, 0, 4, 8},
                  {"nvnMemoryPoolBuilderGetSize", {}, 0, 4, 8}}});

    v.push_back({"BufferBuilder", "nvnBufferBuilderSetDefaults", {{"nvnBufferBuilderSetDevice", {dev}}},
                 {{"nvnBufferBuilderSetStorage", {dev, 0x100, 0x2000}},
                  {"nvnBufferBuilderSetDefaults", {}},
                  {"nvnBufferBuilderSetStorage", {dev + 0x800, 0, 0x40}}},
                 {{"nvnBufferBuilderGetDevice", {}, 0, 4, 8},
                  {"nvnBufferBuilderGetMemoryPool", {}, 0, 4, 8},
                  {"nvnBufferBuilderGetMemoryOffset", {}, 0, 4, 8},
                  {"nvnBufferBuilderGetSize", {}, 0, 4, 8}}});

    v.push_back({"QueueBuilder", "nvnQueueBuilderSetDefaults", {{"nvnQueueBuilderSetDevice", {dev}}},
                 {{"nvnQueueBuilderSetQueuePriority", {2}},
                  {"nvnQueueBuilderSetDefaults", {}},
                  {"nvnQueueBuilderSetFlags", {3}},
                  {"nvnQueueBuilderSetCommandMemorySize", {0x20000}},
                  {"nvnQueueBuilderSetComputeMemorySize", {0}},
                  {"nvnQueueBuilderSetControlMemorySize", {0x1000}},
                  {"nvnQueueBuilderSetCommandFlushThreshold", {0x4000}},
                  {"nvnQueueBuilderSetQueuePriority", {2}},
                  {"nvnQueueBuilderSetQueueMemory", {0x50000000, 0x10000}}},
                 {{"nvnQueueBuilderGetDevice", {}, 0, 4, 8},
                  {"nvnQueueBuilderGetFlags", {}, 0, 4, 4},
                  {"nvnQueueBuilderGetCommandMemorySize", {}, 1, 4, 1},
                  {"nvnQueueBuilderGetComputeMemorySize", {}, 1, 4, 1},
                  {"nvnQueueBuilderGetControlMemorySize", {}, 1, 4, 1},
                  {"nvnQueueBuilderGetCommandFlushThreshold", {}, 1, 4, 1},
                  {"nvnQueueBuilderGetQueuePriority", {}, 0, 4, 4},
                  {"nvnQueueBuilderGetMemory", {}, 0, 4, 8}}});
    // Enum values: NVN's (min filter 0-5, mag 0-1, wrap 0-7, compare mode 0-1,
    // compare function 1-8, reduction 0-2); border colours as four words.
    constexpr u64 border = Scratch::kOut + 0x200;
    g_mem.Write(border, std::array<u32, 4>{0x3F800000, 0x3F000000, 0, 0x3F800000}.data(), 16);
    v.push_back({"SamplerBuilder", "nvnSamplerBuilderSetDefaults", {{"nvnSamplerBuilderSetDevice", {dev}}},
                 {{"nvnSamplerBuilderSetMinMagFilter", {5, 0}},
                  {"nvnSamplerBuilderSetMinMagFilter", {2, 1}},
                  {"nvnSamplerBuilderSetWrapMode", {7, 5, 6}},
                  {"nvnSamplerBuilderSetWrapMode", {0, 3, 4}},
                  {"nvnSamplerBuilderSetLodClamp", {}, {0.5f, 12.0f}},
                  {"nvnSamplerBuilderSetLodBias", {}, {-1.25f}},
                  {"nvnSamplerBuilderSetCompare", {1, 8}},
                  {"nvnSamplerBuilderSetCompare", {0, 3}},
                  {"nvnSamplerBuilderSetMaxAnisotropy", {}, {16.0f}},
                  {"nvnSamplerBuilderSetReductionFilter", {2}},
                  {"nvnSamplerBuilderSetLodSnap", {}, {0.25f}},
                  {"nvnSamplerBuilderSetBorderColor", {border}},
                  {"nvnSamplerBuilderSetDefaults", {}},
                  {"nvnSamplerBuilderSetBorderColori", {border}},
                  {"nvnSamplerBuilderSetBorderColorui", {border + 4}}},
                 {{"nvnSamplerBuilderGetDevice", {}, 0, 4, 8},
                  {"nvnSamplerBuilderGetMinMagFilter", {}, 2, 4},
                  {"nvnSamplerBuilderGetWrapMode", {}, 3, 4},
                  {"nvnSamplerBuilderGetLodClamp", {}, 2, 4},
                  {"nvnSamplerBuilderGetLodBias", {}, 0, 4, 0, true},
                  {"nvnSamplerBuilderGetCompare", {}, 2, 4},
                  {"nvnSamplerBuilderGetMaxAnisotropy", {}, 0, 4, 0, true},
                  {"nvnSamplerBuilderGetReductionFilter", {}, 0, 4, 4},
                  {"nvnSamplerBuilderGetLodSnap", {}, 0, 4, 0, true},
                  {"nvnSamplerBuilderGetBorderColor", {}, 1, 16},
                  {"nvnSamplerBuilderGetBorderColori", {}, 1, 16},
                  {"nvnSamplerBuilderGetBorderColorui", {}, 1, 16}}});
    // Window builder: the texture array is read by GetTexture (index < count only).
    constexpr u64 window_textures = Scratch::kOut + 0x240;
    g_mem.Write(window_textures, std::array<u64, 4>{0x1110, 0x2220, 0x3330, 0x4440}.data(), 32);
    v.push_back({"WindowBuilder", "nvnWindowBuilderSetDefaults", {{"nvnWindowBuilderSetDevice", {dev}}},
                 {{"nvnWindowBuilderSetTextures", {3, window_textures}},
                  {"nvnWindowBuilderSetPresentInterval", {2}},
                  {"nvnWindowBuilderSetNumActiveTextures", {2}},
                  {"nvnWindowBuilderSetNativeWindow", {0x12345678}},
                  {"nvnWindowBuilderSetPresentInterval", {0}},
                  {"nvnWindowBuilderSetDefaults", {}},
                  {"nvnWindowBuilderSetTextures", {2, window_textures + 8}}},
                 {{"nvnWindowBuilderGetDevice", {}, 0, 4, 8},
                  {"nvnWindowBuilderGetNumTextures", {}, 0, 4, 4},
                  {"nvnWindowBuilderGetTexture", {0}, 0, 4, 8},
                  {"nvnWindowBuilderGetTexture", {1}, 0, 4, 8},
                  {"nvnWindowBuilderGetTexture", {2}, 0, 4, 8},
                  {"nvnWindowBuilderGetPresentInterval", {}, 0, 4, 4},
                  {"nvnWindowBuilderGetNumActiveTextures", {}, 0, 4, 4},
                  {"nvnWindowBuilderGetNativeWindow", {}, 0, 4, 8}}});
    // Texture builder: everything but the size/alignment queries (they read
    // the device; checked against the running driver in tap mode instead).
    constexpr u64 layout = Scratch::kOut + 0x220;
    hle::WriteGuest<u32>(layout, 0xABCD5432); // bits above the low 12 are dropped by NVN
    v.push_back({"TextureBuilder", "nvnTextureBuilderSetDefaults", {{"nvnTextureBuilderSetDevice", {dev}}},
                 {{"nvnTextureBuilderSetFlags", {0x28}},
                  {"nvnTextureBuilderSetTarget", {10}},
                  {"nvnTextureBuilderSetSize3D", {640, 360, 6}},
                  {"nvnTextureBuilderSetSize2D", {128, 64}},
                  {"nvnTextureBuilderSetSize1D", {4096}},
                  {"nvnTextureBuilderSetWidth", {33}},
                  {"nvnTextureBuilderSetHeight", {17}},
                  {"nvnTextureBuilderSetDepth", {5}},
                  {"nvnTextureBuilderSetLevels", {9}},
                  {"nvnTextureBuilderSetFormat", {0x41}},
                  {"nvnTextureBuilderSetSamples", {4}},
                  {"nvnTextureBuilderSetSwizzle", {5, 4, 3, 2}},
                  {"nvnTextureBuilderSetDepthStencilMode", {1}},
                  {"nvnTextureBuilderSetStride", {5120}},
                  {"nvnTextureBuilderSetStorage", {0x12345000, 0x2000}},
                  {"nvnTextureBuilderSetPackagedTextureData", {0x55550000}},
                  {"nvnTextureBuilderSetPackagedTextureLayout", {layout}},
                  {"nvnTextureBuilderSetGLTextureName", {77}},
                  {"nvnTextureBuilderSetDefaults", {}}},
                 {{"nvnTextureBuilderGetDevice", {}, 0, 4, 8},
                  {"nvnTextureBuilderGetFlags", {}, 0, 4, 4},
                  {"nvnTextureBuilderGetTarget", {}, 0, 4, 4},
                  {"nvnTextureBuilderGetWidth", {}, 0, 4, 4},
                  {"nvnTextureBuilderGetHeight", {}, 0, 4, 4},
                  {"nvnTextureBuilderGetDepth", {}, 0, 4, 4},
                  {"nvnTextureBuilderGetLevels", {}, 0, 4, 4},
                  {"nvnTextureBuilderGetFormat", {}, 0, 4, 4},
                  {"nvnTextureBuilderGetSamples", {}, 0, 4, 4},
                  {"nvnTextureBuilderGetDepthStencilMode", {}, 0, 4, 4},
                  {"nvnTextureBuilderGetStride", {}, 0, 4, 4},
                  {"nvnTextureBuilderGetGLTextureName", {}, 0, 4, 4},
                  {"nvnTextureBuilderGetMemoryPool", {}, 0, 4, 8},
                  {"nvnTextureBuilderGetMemoryOffset", {}, 0, 4, 8},
                  {"nvnTextureBuilderGetPackagedTextureData", {}, 0, 4, 8},
                  {"nvnTextureBuilderGetSwizzle", {}, 4, 4},
                  {"nvnTextureBuilderGetPackagedTextureLayout", {}, 1, 4, 4}}});
    v.push_back({"TextureView", "nvnTextureViewSetDefaults", {},
                 {{"nvnTextureViewSetLevels", {2, 5}},
                  {"nvnTextureViewSetLayers", {1, 6}},
                  {"nvnTextureViewSetFormat", {0x25}},
                  {"nvnTextureViewSetSwizzle", {5, 4, 3, 2}},
                  {"nvnTextureViewSetDepthStencilMode", {1}},
                  {"nvnTextureViewSetTarget", {4}},
                  {"nvnTextureViewSetLevels", {0, 1}},
                  {"nvnTextureViewSetDefaults", {}},
                  {"nvnTextureViewSetDepthStencilMode", {0}},
                  {"nvnTextureViewSetLayers", {3, 1}}},
                 {{"nvnTextureViewGetLevels", {}, 2, 4, 1},
                  {"nvnTextureViewGetLayers", {}, 2, 4, 1},
                  {"nvnTextureViewGetFormat", {}, 1, 4, 1},
                  {"nvnTextureViewGetSwizzle", {}, 4, 4, 1},
                  {"nvnTextureViewGetDepthStencilMode", {}, 1, 4, 1},
                  {"nvnTextureViewGetTarget", {}, 1, 4, 1}}});
    return v;
}

// --- Command buffers -----------------------------------------------------------------------
// Control memory is used as NVN uses it (cmdbuf.cpp), so the SDK and the layer
// get the same small control chunks from a callback and must agree on every
// getter, every callback (event, size) and every handle (as an offset into the
// chunks). Command memory needs a real NVN memory pool on the SDK's side, so
// that part runs on the layer alone: records written through tiny chunks
// (callbacks, merged and separate segments, control jumps) must read back in
// order.

struct CallbackLog {
    u64 arena = 0, next = 0;  // control chunks handed out from `arena`
    u64 cmd_next = 0;         // command chunks (offsets into the test pool)
    u32 cmd_chunks = 0;
    std::vector<std::pair<u32, u64>> calls; // (event, minimum size)
};
CallbackLog g_cb[3]; // [0] layer, [1] SDK, [2] layer command-memory test
u64 g_test_pool = 0;

constexpr u64 kControlChunks[] = {0x30, 0x48, 0x40, 0x60, 0x38, 0x80};
constexpr u64 kCmdPoolStorage = Scratch::kCommands + 0x8000, kCmdPoolSize = 0x8000;
constexpr u64 kPoolObj = Scratch::kObjects + 2 * kObjSize, kPoolBuilder = Scratch::kObjects + 3 * kObjSize;

void TestCallback(GuestContext* c, u64 cmdbuf, u32 event, u64 min_size, u64 data) {
    CallbackLog& l = g_cb[data];
    l.calls.push_back({event, min_size});
    if (event == 1) {
        const u64 size = kControlChunks[l.calls.size() % std::size(kControlChunks)];
        Call(c, data == 1, "nvnCommandBufferAddControlMemory", cmdbuf, {l.next, size});
        l.next += size;
    } else {
        // 60-byte chunks follow each other (their segments merge), 64-byte ones leave a gap
        const u64 size = (l.cmd_chunks++ % 4) < 2 ? 60 : 64;
        Call(c, false, "nvnCommandBufferAddCommandMemory", cmdbuf, {g_test_pool, l.cmd_next, size});
        l.cmd_next += size + (size == 64 ? 8 : 0);
    }
}

bool Expect(bool ok, const char* what, unsigned long long got, unsigned long long want) {
    ++g_compared;
    if (!ok) {
        ++g_differences;
        Log("difftest: %s: 0x%llx, expected 0x%llx", what, got, want);
    }
    return ok;
}

void RunCommandBufferTest(GuestContext* c) {
    const u64 before = g_differences;
    static const u64 callback = hle::RegisterHostFunction("difftest.cmdbufCallback", hle::Bind<&TestCallback>());
    const std::vector<u8> zero(kObjSize, 0);
    const u64 arena[2] = {Scratch::kCommands + 0x4000, Scratch::kCommands}; // layer, SDK
    u64 obj[2] = {kOurObj, kSdkObj};
    for (int side = 0; side < 2; ++side) {
        const bool sdk = side == 1;
        g_cb[side] = {arena[side], arena[side] + 0x40, 0, 0, {}};
        g_mem.Write(obj[side], zero.data(), zero.size());
        Call(c, sdk, "nvnCommandBufferInitialize", obj[side], {Scratch::kDevice});
        Call(c, sdk, "nvnCommandBufferSetMemoryCallback", obj[side], {callback});
        Call(c, sdk, "nvnCommandBufferSetMemoryCallbackData", obj[side], {static_cast<u64>(side)});
        Call(c, sdk, "nvnCommandBufferAddControlMemory", obj[side], {arena[side], 0x40});
    }
    auto compare = [&](const char* after) {
        for (const char* fn : {"nvnCommandBufferGetControlMemorySize", "nvnCommandBufferGetControlMemoryUsed",
                               "nvnCommandBufferGetControlMemoryFree", "nvnCommandBufferGetMemoryCallback"}) {
            const u64 sdk = Call(c, true, fn, kSdkObj, {}), ours = Call(c, false, fn, kOurObj, {});
            if (!Expect(sdk == ours, (std::string("command buffers: ") + fn).c_str(), ours, sdk)) Log("difftest:   (after %s)", after);
        }
        Expect(g_cb[0].calls == g_cb[1].calls, "command buffers: control callbacks (count)", g_cb[0].calls.size(), g_cb[1].calls.size());
    };
    compare("AddControlMemory");
    for (int list = 0; list < 16; ++list) {
        for (int side = 0; side < 2; ++side) Call(c, side == 1, "nvnCommandBufferBeginRecording", obj[side], {});
        compare("BeginRecording");
        u64 handle[2];
        for (int side = 0; side < 2; ++side) {
            handle[side] = Call(c, side == 1, "nvnCommandBufferEndRecording", obj[side], {}) - arena[side];
        }
        compare("EndRecording");
        Expect(handle[0] == handle[1], "command buffers: handle (offset)", handle[0], handle[1]);
    }
    Log("difftest: command buffers: control memory: %zu callbacks each", g_cb[0].calls.size());

    // Command memory, layer only: a pool on scratch memory, tiny chunks.
    const u64 builder = kPoolBuilder;
    Call(c, false, "nvnMemoryPoolBuilderSetDefaults", builder, {});
    Call(c, false, "nvnMemoryPoolBuilderSetDevice", builder, {Scratch::kDevice});
    Call(c, false, "nvnMemoryPoolBuilderSetFlags", builder, {0x24}); // CPU_CACHED | GPU_CACHED
    Call(c, false, "nvnMemoryPoolBuilderSetStorage", builder, {kCmdPoolStorage, kCmdPoolSize});
    if (!Call(c, false, "nvnMemoryPoolInitialize", kPoolObj, {builder})) Fatal("difftest: the test pool failed");
    g_test_pool = kPoolObj;
    g_cb[2] = {Scratch::kCommands + 0x6000, Scratch::kCommands + 0x6000, 0, 0, {}};
    Call(c, false, "nvnCommandBufferInitialize", kOurObj, {Scratch::kDevice});
    Call(c, false, "nvnCommandBufferSetMemoryCallback", kOurObj, {callback});
    Call(c, false, "nvnCommandBufferSetMemoryCallbackData", kOurObj, {2});
    constexpr u32 kLists = 4, kPerList = 60;
    u64 handles[kLists];
    u32 n = 0;
    for (u32 l = 0; l < kLists; ++l) {
        Call(c, false, "nvnCommandBufferBeginRecording", kOurObj, {});
        for (u32 i = 0; i < kPerList; ++i, ++n) {
            Call(c, false, "nvnCommandBufferReportCounter", kOurObj, {n, 0x1'0000'0000ULL + 0x100ULL * n});
        }
        handles[l] = Call(c, false, "nvnCommandBufferEndRecording", kOurObj, {});
    }
    u32 cmd_calls = 0, ctl_calls = 0;
    for (const auto& [event, size] : g_cb[2].calls) {
        if (event == 0) ++cmd_calls, Expect(size == 512, "command buffers: command callback size", size, 512);
        else ++ctl_calls, Expect(size == 256, "command buffers: control callback size", size, 256);
    }
    n = 0;
    size_t segments = 0;
    for (u32 l = 0; l < kLists; ++l) {
        segments += Segments(handles[l]).size();
        u32 got = 0;
        ForEachRecord(handles[l], [&](Op op, const u8* p, u32 size) {
            u32 counter;
            u64 address;
            std::memcpy(&counter, p, 4);
            std::memcpy(&address, p + 8, 8);
            Expect(op == Op::kReportCounter && size == 16, "command buffers: record op", static_cast<u32>(op), 1);
            Expect(counter == n && address == 0x1'0000'0000ULL + 0x100ULL * n, "command buffers: record counter", counter, n);
            ++n, ++got;
        });
        Expect(got == kPerList, "command buffers: records in a list", got, kPerList);
    }
    Log("difftest: command buffers: %u records in %u lists, %zu segments, %u command and %u control callbacks", n,
        kLists, segments, cmd_calls, ctl_calls);
    Call(c, false, "nvnCommandBufferFinalize", kOurObj, {});
    Call(c, false, "nvnMemoryPoolFinalize", kPoolObj, {});
    Log("difftest: command buffers: %llu differences", (unsigned long long)(g_differences - before));
}

// Queues and syncs, on the layer alone (the SDK's need the GPU): sync
// results, counter reports written before their sync signals (NVN_PLAN 5.0
// rule 3) and in GPU order, many submissions in flight.
void RunQueueTest(GuestContext* c) {
    const u64 before = g_differences;
    const u64 obj = Scratch::kObjects;
    const u64 device = obj + 4 * kObjSize, device_builder = obj + 5 * kObjSize, queue = obj + 6 * kObjSize,
              queue_builder = obj + 7 * kObjSize, sync_a = obj + 8 * kObjSize, sync_b = obj + 9 * kObjSize,
              cmdbuf = obj + 10 * kObjSize, reports = obj + 11 * kObjSize; // 0x200: 32 reports
    auto call = [&](const char* fn, u64 o, std::vector<u64> args = {}) { return Call(c, false, fn, o, args); };
    call("nvnDeviceBuilderSetDefaults", device_builder);
    if (!call("nvnDeviceInitialize", device, {device_builder})) Fatal("difftest: the test device failed");
    call("nvnQueueBuilderSetDevice", queue_builder, {device});
    call("nvnQueueBuilderSetDefaults", queue_builder);
    if (!call("nvnQueueInitialize", queue, {queue_builder})) Fatal("difftest: the test queue failed");
    call("nvnSyncInitialize", sync_a, {device});
    call("nvnSyncInitialize", sync_b, {device});
    Expect(call("nvnSyncWait", sync_a, {0}) == 3, "queue: wait on a sync never fenced", call("nvnSyncWait", sync_a, {0}), 3);
    call("nvnQueueFenceSync", queue, {sync_a, 0, 0});
    Expect(call("nvnSyncWait", sync_a, {~0ULL}) == 0, "queue: a fence with nothing submitted", 0, 0);

    // A pool for command memory, control memory from the scratch area, as the command-buffer test.
    Call(c, false, "nvnMemoryPoolBuilderSetDefaults", kPoolBuilder, {});
    Call(c, false, "nvnMemoryPoolBuilderSetDevice", kPoolBuilder, {device});
    Call(c, false, "nvnMemoryPoolBuilderSetFlags", kPoolBuilder, {0x24});
    Call(c, false, "nvnMemoryPoolBuilderSetStorage", kPoolBuilder, {kCmdPoolStorage, kCmdPoolSize});
    if (!Call(c, false, "nvnMemoryPoolInitialize", kPoolObj, {kPoolBuilder})) Fatal("difftest: the test pool failed");
    auto list = [&](u32 counters, u32 first_report, u64 fence, u64 wait) {
        call("nvnCommandBufferInitialize", cmdbuf, {device});
        call("nvnCommandBufferAddCommandMemory", cmdbuf, {kPoolObj, 0, 0x1000});
        call("nvnCommandBufferAddControlMemory", cmdbuf, {Scratch::kCommands, 0x400});
        call("nvnCommandBufferBeginRecording", cmdbuf);
        if (wait) call("nvnCommandBufferWaitSync", cmdbuf, {wait});
        for (u32 i = 0; i < counters; ++i) {
            call("nvnCommandBufferReportCounter", cmdbuf, {0, reports + 16ULL * ((first_report + i) % 32)});
            if (i % 3 == 1) call("nvnCommandBufferBarrier", cmdbuf, {0x100});
        }
        if (fence) call("nvnCommandBufferFenceSync", cmdbuf, {fence, 0, 0});
        const u64 handle = call("nvnCommandBufferEndRecording", cmdbuf);
        hle::WriteGuest<u64>(Scratch::kOut, handle);
        call("nvnQueueSubmitCommands", queue, {1, Scratch::kOut});
        call("nvnCommandBufferFinalize", cmdbuf);
    };
    auto report = [&](u32 i) { return hle::ReadGuest<std::array<u64, 2>>(reports + 16ULL * i); };
    const std::vector<u8> fill(0x200, 0xFF);
    g_mem.Write(reports, fill.data(), fill.size());
    list(3, 0, sync_a, 0);
    const u64 r = call("nvnSyncWait", sync_a, {~0ULL});
    Expect(r == 0 || r == 1, "queue: wait for a submitted fence", r, 1);
    for (u32 i = 0; i < 3; ++i) {
        const auto rep = report(i);
        Expect(rep[0] == 0 && rep[1] != ~0ULL, "queue: counter report written before its sync signaled", rep[1], 0);
        if (i) Expect(rep[1] >= report(i - 1)[1], "queue: counter order", rep[1], report(i - 1)[1]);
    }
    list(1, 3, 0, sync_a); // waits for a sync of its own queue
    call("nvnQueueFenceSync", queue, {sync_b, 0, 0});
    Expect(call("nvnSyncWait", sync_b, {~0ULL}) <= 1, "queue: queue fence after a submission", 0, 0);
    Expect(report(3)[1] >= report(2)[1], "queue: counters of a later submission", report(3)[1], report(2)[1]);

    // Many submissions in flight, 1 to 20 counters each; then all reports in order.
    g_mem.Write(reports, fill.data(), fill.size());
    u32 at = 0;
    u64 last = 0;
    for (u32 k = 0; k < 50; ++k) {
        const u32 n = 1 + (k * 7) % 20;
        list(n, at, k == 49 ? sync_b : 0, 0);
        at += n;
    }
    call("nvnQueueFinish", queue);
    Expect(call("nvnSyncWait", sync_b, {0}) == 0, "queue: after QueueFinish", 1, 0);
    for (u32 i = at - 32; i < at; ++i) { // the last 32 reports (the area wraps)
        const auto rep = report(i % 32);
        Expect(rep[1] != ~0ULL && rep[1] >= last, "queue: reports of 50 submissions in order", rep[1], last);
        last = rep[1];
    }
    // SAMPLES_PASSED (no draws: 0 samples) around a reset, then buffer copies: of CPU-written bytes with a
    // pending fill over part of them, and of that copy's result in the same submission (a GPU part).
    auto record = [&](auto&& commands) {
        call("nvnCommandBufferInitialize", cmdbuf, {device});
        call("nvnCommandBufferAddCommandMemory", cmdbuf, {kPoolObj, 0, 0x1000});
        call("nvnCommandBufferAddControlMemory", cmdbuf, {Scratch::kCommands, 0x400});
        call("nvnCommandBufferBeginRecording", cmdbuf);
        commands();
        hle::WriteGuest<u64>(Scratch::kOut, call("nvnCommandBufferEndRecording", cmdbuf));
        call("nvnQueueSubmitCommands", queue, {1, Scratch::kOut});
        call("nvnCommandBufferFinalize", cmdbuf);
    };
    g_mem.Write(reports, fill.data(), fill.size());
    record([&] {
        call("nvnCommandBufferReportCounter", cmdbuf, {1, reports});
        call("nvnCommandBufferResetCounter", cmdbuf, {1});
        call("nvnCommandBufferReportCounter", cmdbuf, {1, reports + 16});
        call("nvnCommandBufferReportCounter", cmdbuf, {0, reports + 32});
    });
    record([&] { call("nvnCommandBufferReportCounter", cmdbuf, {1, reports + 48}); });
    call("nvnQueueFinish", queue);
    for (u32 i = 0; i < 4; ++i) {
        const auto rep = report(i);
        Expect(rep[0] == 0 && rep[1] != ~0ULL, "queue: SAMPLES_PASSED report without draws", rep[0], 0);
        if (i) Expect(rep[1] >= report(i - 1)[1], "queue: SAMPLES_PASSED report order", rep[1], report(i - 1)[1]);
    }
    const u64 src = kCmdPoolStorage + 0x2000, dst = kCmdPoolStorage + 0x3000, dst2 = kCmdPoolStorage + 0x4000;
    std::vector<u8> bytes(0x100), expect(0x100);
    for (u32 i = 0; i < 0x100; ++i) bytes[i] = static_cast<u8>(i * 7 + 3);
    g_mem.Write(src, bytes.data(), bytes.size());
    g_mem.Write(dst, fill.data(), 0x100);
    g_mem.Write(dst2, fill.data(), 0x100);
    expect = bytes;
    for (u32 i = 0x40; i < 0x60; i += 4) std::memcpy(expect.data() + i, "\xDD\xCC\xBB\xAA", 4);
    record([&] {
        call("nvnCommandBufferClearBuffer", cmdbuf, {src + 0x40, 0x20, 0xAABBCCDD});
        call("nvnCommandBufferCopyBufferToBuffer", cmdbuf, {src, dst, 0x100, 0});
        call("nvnCommandBufferCopyBufferToBuffer", cmdbuf, {dst + 0x10, dst2 + 0x10, 0xF0, 0});
    });
    call("nvnQueueFinish", queue);
    std::vector<u8> got(0x100), got2(0x100);
    g_mem.Read(dst, got.data(), got.size());
    g_mem.Read(dst2, got2.data(), got2.size());
    for (u32 i = 0; i < 0x100; ++i) {
        if (got[i] != expect[i]) { Expect(false, "queue: CopyBufferToBuffer over a pending fill", got[i], expect[i]); break; }
        const u8 want2 = i < 0x10 ? 0xFF : expect[i];
        if (got2[i] != want2) { Expect(false, "queue: CopyBufferToBuffer of a copy in the same submission", got2[i], want2); break; }
    }
    call("nvnSyncFinalize", sync_a);
    call("nvnSyncFinalize", sync_b);
    call("nvnQueueFinalize", queue);
    Call(c, false, "nvnMemoryPoolFinalize", kPoolObj, {});
    call("nvnDeviceFinalize", device);
    Log("difftest: queue: %llu differences", (unsigned long long)(g_differences - before));
}

// Copies into textures (NVN_PLAN 6.3), layer only: known bytes copied through
// nvnCommandBufferCopyBufferToTexture into levels, layers, 3D slices, a
// compressed and a depth texture, then the Vulkan image read back.
void RunCopyTest(GuestContext* c) {
    const u64 before = g_differences;
    const u64 obj = Scratch::kObjects;
    const u64 device = obj + 12 * kObjSize, device_builder = obj + 13 * kObjSize, queue = obj + 14 * kObjSize,
              queue_builder = obj + 15 * kObjSize, cmdbuf = obj + 16 * kObjSize, builder = obj + 17 * kObjSize,
              texture = obj + 18 * kObjSize, view = obj + 19 * kObjSize, region = obj + 20 * kObjSize,
              pool = obj + 21 * kObjSize, pool_builder = obj + 22 * kObjSize;
    constexpr u64 kPoolSize = 0x80000;
    auto call = [&](const char* fn, u64 o, std::vector<u64> args = {}) { return Call(c, false, fn, o, args); };
    call("nvnDeviceBuilderSetDefaults", device_builder);
    if (!call("nvnDeviceInitialize", device, {device_builder})) Fatal("difftest: the test device failed");
    call("nvnQueueBuilderSetDevice", queue_builder, {device});
    call("nvnQueueBuilderSetDefaults", queue_builder);
    if (!call("nvnQueueInitialize", queue, {queue_builder})) Fatal("difftest: the test queue failed");
    call("nvnMemoryPoolBuilderSetDefaults", pool_builder);
    call("nvnMemoryPoolBuilderSetDevice", pool_builder, {device});
    call("nvnMemoryPoolBuilderSetFlags", pool_builder, {0x24}); // CPU_CACHED | GPU_CACHED
    call("nvnMemoryPoolBuilderSetStorage", pool_builder, {Scratch::kTextures, kPoolSize});
    if (!call("nvnMemoryPoolInitialize", pool, {pool_builder})) Fatal("difftest: the test pool failed");

    struct Case {
        const char* name;
        u32 target, w, h, d, levels, format;
        s32 view_level, view_layer; // -1: not set in the view
        CopyRegion r;
    };
    const Case cases[] = {
        {"2D level 1 through a view", 1, 64, 32, 1, 3, 0x25, 1, -1, {4, 2, 0, 16, 8, 1}},
        {"2D array, layers 1-2", 4, 16, 16, 4, 1, 0x25, -1, -1, {0, 0, 1, 16, 16, 2}},
        {"2D array, view base layer 1", 4, 16, 16, 4, 1, 0x25, -1, 1, {2, 3, 1, 5, 7, 2}},
        {"3D slices 2-4", 2, 8, 8, 8, 1, 0x01, -1, -1, {2, 1, 2, 4, 4, 3}},
        {"BC1, blocks (1,1) 2x2", 1, 16, 16, 1, 1, 0x42, -1, -1, {4, 4, 0, 8, 8, 1}},
        {"cube face 3", 8, 8, 8, 1, 1, 0x25, -1, -1, {0, 0, 3, 8, 8, 1}},
        {"depth 32F", 1, 16, 8, 1, 1, 0x34, -1, -1, {0, 0, 0, 16, 8, 1}},
    };
    u64 at = 0x1000; // after the command memory
    for (const Case& k : cases) {
        call("nvnTextureBuilderSetDevice", builder, {device});
        call("nvnTextureBuilderSetDefaults", builder);
        call("nvnTextureBuilderSetTarget", builder, {k.target});
        call("nvnTextureBuilderSetSize3D", builder, {k.w, k.h, k.d});
        call("nvnTextureBuilderSetLevels", builder, {k.levels});
        call("nvnTextureBuilderSetFormat", builder, {k.format});
        const u64 align = call("nvnTextureBuilderGetStorageAlignment", builder);
        const u64 tex_size = call("nvnTextureBuilderGetStorageSize", builder);
        at = (at + align - 1) / align * align;
        const u64 tex_at = at;
        at += tex_size;
        call("nvnTextureBuilderSetStorage", builder, {pool, tex_at});
        if (!call("nvnTextureInitialize", texture, {builder})) Fatal("difftest: copy test: %s: texture failed", k.name);
        call("nvnTextureViewSetDefaults", view);
        if (k.view_level >= 0) call("nvnTextureViewSetLevels", view, {static_cast<u64>(k.view_level), 1});
        if (k.view_layer >= 0) call("nvnTextureViewSetLayers", view, {static_cast<u64>(k.view_layer), 1});
        hle::WriteGuest(region, k.r);

        const FormatInfo* f = FindFormat(k.format);
        auto cdiv = [](u32 a, u32 b) { return (a + b - 1) / b; };
        const u32 xb = cdiv(k.r.x, f->block_w), yb = cdiv(k.r.y, f->block_h);
        const u32 wb = cdiv(k.r.width, f->block_w), hb = cdiv(k.r.height, f->block_h);
        const u64 row = u64{wb} * f->bytes, size = row * hb * k.r.depth;
        std::vector<u8> src(size);
        for (u64 i = 0; i < size; ++i) src[i] = static_cast<u8>(i * 131 + 7 * (&k - cases) + 1);
        if (k.format == 0x34) { // depth values in [0, 1]
            for (u64 i = 0; i < size / 4; ++i) {
                const float v = static_cast<float>(i % 1000) / 1000.0f;
                std::memcpy(&src[4 * i], &v, 4);
            }
        }
        at = (at + 255) / 256 * 256;
        const u64 src_at = at;
        at += size;
        if (at > kPoolSize) Fatal("difftest: copy test: the pool is too small");
        g_mem.Write(Scratch::kTextures + src_at, src.data(), size);

        call("nvnCommandBufferInitialize", cmdbuf, {device});
        call("nvnCommandBufferAddCommandMemory", cmdbuf, {pool, 0, 0x1000}); // the pool's first 4 KB
        call("nvnCommandBufferAddControlMemory", cmdbuf, {Scratch::kCommands, 0x400});
        call("nvnCommandBufferBeginRecording", cmdbuf);
        call("nvnCommandBufferCopyBufferToTexture", cmdbuf,
             {Scratch::kTextures + src_at, texture, k.view_level >= 0 || k.view_layer >= 0 ? view : 0, region, 0});
        hle::WriteGuest<u64>(Scratch::kOut, call("nvnCommandBufferEndRecording", cmdbuf));
        call("nvnQueueSubmitCommands", queue, {1, Scratch::kOut});
        call("nvnQueueFinish", queue);
        call("nvnCommandBufferFinalize", cmdbuf);

        const u32 level = k.view_level >= 0 ? k.view_level : 0;
        const TextureState state = TextureStateOf(texture);
        const std::vector<u8> img = DownloadImage(state, level);
        const u32 lw = std::max(1u, k.w >> level), lh = std::max(1u, k.h >> level);
        const u64 lrow = u64{cdiv(lw, f->block_w)} * f->bytes, limage = lrow * cdiv(lh, f->block_h);
        const u32 first = k.r.z + (k.view_layer >= 0 ? k.view_layer : 0);
        u64 bad = 0;
        for (u32 s_ = 0; s_ < static_cast<u32>(k.r.depth); ++s_) {
            for (u32 y = 0; y < hb; ++y) {
                const u64 img_at = (first + s_) * limage + (yb + y) * lrow + u64{xb} * f->bytes;
                const u64 src_row = (u64{s_} * hb + y) * row;
                if (img_at + row > img.size() || std::memcmp(&img[img_at], &src[src_row], row) != 0) ++bad;
            }
        }
        Expect(bad == 0, (std::string("copies: ") + k.name + ": rows different").c_str(), bad, 0);
        call("nvnTextureFinalize", texture);
    }
    call("nvnMemoryPoolFinalize", pool);
    call("nvnQueueFinalize", queue);
    call("nvnDeviceFinalize", device);
    Log("difftest: copies: %zu cases, %llu differences", std::size(cases), (unsigned long long)(g_differences - before));
}

// Windows (NVN_PLAN 9.1, 9.2), layer only: textures handed out in turn, the
// acquire sync signaled, and the pacing of the display model (interval 2:
// 33 ms a frame, interval 1: 17 ms; measured over 12 frames, with margins).
void RunWindowTest(GuestContext* c) {
    const u64 before = g_differences;
    const u64 obj = Scratch::kObjects;
    const u64 device = obj + 23 * kObjSize, device_builder = obj + 24 * kObjSize, queue = obj + 25 * kObjSize,
              queue_builder = obj + 26 * kObjSize, pool = obj + 27 * kObjSize, pool_builder = obj + 28 * kObjSize,
              tex_builder = obj + 29 * kObjSize, window = obj + 30 * kObjSize, window_builder = obj + 31 * kObjSize,
              sync = obj + 32 * kObjSize, textures = obj + 33 * kObjSize; // 3 textures, kObjSize apart
    const u64 tex_list = Scratch::kOut + 0x280, index = Scratch::kOut + 0x2a0;
    auto call = [&](const char* fn, u64 o, std::vector<u64> args = {}) { return Call(c, false, fn, o, args); };
    call("nvnDeviceBuilderSetDefaults", device_builder);
    if (!call("nvnDeviceInitialize", device, {device_builder})) Fatal("difftest: the test device failed");
    call("nvnQueueBuilderSetDevice", queue_builder, {device});
    call("nvnQueueBuilderSetDefaults", queue_builder);
    if (!call("nvnQueueInitialize", queue, {queue_builder})) Fatal("difftest: the test queue failed");
    call("nvnMemoryPoolBuilderSetDefaults", pool_builder);
    call("nvnMemoryPoolBuilderSetDevice", pool_builder, {device});
    call("nvnMemoryPoolBuilderSetFlags", pool_builder, {0x24});
    call("nvnMemoryPoolBuilderSetStorage", pool_builder, {Scratch::kTextures, 0x80000});
    if (!call("nvnMemoryPoolInitialize", pool, {pool_builder})) Fatal("difftest: the test pool failed");
    for (u32 i = 0; i < 3; ++i) {
        call("nvnTextureBuilderSetDevice", tex_builder, {device});
        call("nvnTextureBuilderSetDefaults", tex_builder);
        call("nvnTextureBuilderSetTarget", tex_builder, {1});
        call("nvnTextureBuilderSetSize2D", tex_builder, {64, 64});
        call("nvnTextureBuilderSetFormat", tex_builder, {0x25});
        call("nvnTextureBuilderSetStorage", tex_builder, {pool, 0x10000ULL * i});
        if (!call("nvnTextureInitialize", textures + i * kObjSize, {tex_builder})) Fatal("difftest: window texture");
        hle::WriteGuest<u64>(tex_list + 8 * i, textures + i * kObjSize);
    }
    call("nvnWindowBuilderSetDevice", window_builder, {device});
    call("nvnWindowBuilderSetDefaults", window_builder);
    call("nvnWindowBuilderSetTextures", window_builder, {3, tex_list});
    if (!call("nvnWindowInitialize", window, {window_builder})) Fatal("difftest: the test window failed");
    call("nvnSyncInitialize", sync, {device});
    Expect(call("nvnWindowGetPresentInterval", window) == 1, "window: default present interval",
           call("nvnWindowGetPresentInterval", window), 1);
    for (u32 interval : {2u, 1u}) {
        call("nvnWindowSetPresentInterval", window, {interval});
        constexpr u32 kFrames = 12;
        auto t0 = std::chrono::steady_clock::now();
        for (u32 f = 0; f < kFrames + 3; ++f) {
            if (f == 3) t0 = std::chrono::steady_clock::now(); // after the first frames fill the textures
            Expect(call("nvnWindowAcquireTexture", window, {sync, index}) == 0, "window: acquire result", 1, 0);
            const u32 got = hle::ReadGuest<u32>(index);
            static u32 expect = 0;
            Expect(got == expect, "window: acquired texture", got, expect);
            expect = (expect + 1) % 3;
            Expect(call("nvnSyncWait", sync, {0}) == 0, "window: acquire sync signaled", 1, 0);
            call("nvnQueuePresentTexture", queue, {window, got});
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const double want = kFrames * interval * 1000.0 / 60.0;
        Log("difftest: window: interval %u: %u frames in %.1f ms (%.1f expected)", interval, kFrames, ms, want);
        Expect(ms > want * 0.8 && ms < want * 1.5, "window: frame pacing (ms)", static_cast<u64>(ms), static_cast<u64>(want));
    }
    call("nvnSyncFinalize", sync);
    call("nvnWindowFinalize", window);
    for (u32 i = 0; i < 3; ++i) call("nvnTextureFinalize", textures + i * kObjSize);
    call("nvnMemoryPoolFinalize", pool);
    call("nvnQueueFinalize", queue);
    call("nvnDeviceFinalize", device);
    Log("difftest: window: %llu differences", (unsigned long long)(g_differences - before));
}

} // namespace

void RunDiffTest(GuestContext* c) {
    auto lock = LockScratch();
    Log("difftest: the layer's state objects and builders against the game's SDK");
    for (const Suite& s : Suites()) RunSuite(c, s);
    RunCommandBufferTest(c);
    RunQueueTest(c);
    RunCopyTest(c);
    RunWindowTest(c);
    Log("difftest: %llu comparisons, %llu differences", (unsigned long long)g_compared,
        (unsigned long long)g_differences);
    Exit(g_differences ? 1 : 0);
}

} // namespace rt::nvn
