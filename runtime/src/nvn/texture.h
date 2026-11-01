// Textures, texture views and their storage (texture.cpp), for the files that
// record and translate texture commands.
#pragma once

#include "nvn.h"

#include <optional>
#include <string>
#include <vector>

namespace rt::nvn {

/// NVNtextureBuilder, inline in the game's storage, in the layer's layout (88
/// bytes; SDK 3.5.0 to 12.3.2 use at least 100).
struct TextureBuilder {
    u64 device, pool;
    s64 offset;
    u64 packaged_data, layout; // layout: the low 12 bits of the NVNpackagedTextureLayout word, as NVN keeps
    u32 flags, target, width, height, depth, levels, format, samples, ds_mode;
    s32 stride;
    u32 gl_name;
    u8 swizzle[4];
    u8 layout_set, pad[3];
};
static_assert(sizeof(TextureBuilder) <= 96);

/// NVNtextureView, inline in the game's storage (16 bytes; SDK 3.5.0 to 12.3.2
/// use 16 to 36). Each property counts only when its bit in `set` is on.
struct TextureView {
    int16_t base_layer, num_layers; // signed, as NVN reads them
    u8 base_level, num_levels;
    u8 target;
    u8 set;     // kViewLevels ... kViewTarget in bits 0-5; bits 6-7: the depth-stencil mode
    u32 format;
    u8 swizzle[4];
};
constexpr u8 kViewLevels = 1, kViewLayers = 2, kViewFormat = 4, kViewSwizzle = 8, kViewDsMode = 16, kViewTarget = 32;
static_assert(sizeof(TextureView) == 16);

struct Storage {
    u64 size;
    u32 alignment;
};

/// Where levels and layers lie in a texture's storage: level l of layer k at
/// k x layer_size + level_offsets[l]; one more offset than levels: their end
/// (checked against the driver's nvnTextureGetViewOffset in tap mode).
struct StorageLayout {
    Storage storage;
    std::vector<u64> level_offsets;
    u64 layer_size;
};
std::optional<StorageLayout> ComputeLayout(const TextureBuilder& b, std::string& why);

/// NVN's storage size and alignment for the builder's settings, or nothing
/// (and `why`) for settings the layer does not handle yet.
std::optional<Storage> ComputeStorage(const TextureBuilder& b, std::string& why);

/// What an initialised NVNtexture is: its builder's settings, its storage and
/// GPU address. Commands and pool entries keep copies (NVN reads a texture
/// when the command is recorded or the entry registered).
struct TextureState {
    TextureBuilder settings;
    Storage storage;
    u64 address; // GPU address (= guest address) of its storage
};

/// The state of the game's initialised NVNtexture at `texture`.
TextureState TextureStateOf(u64 texture);

/// A view as passed by the game: its contents, or none for a null pointer.
struct ViewArg {
    TextureView view;
    u8 present;
    u8 pad[7];
};
ViewArg ReadView(u64 view);

/// A texture pool entry as registered (RegisterTexture / RegisterImage).
struct PoolTexture {
    TextureState texture;
    ViewArg view;
    bool image;
};
/// Entry `id` of the game's NVNtexturePool at `pool`, or nothing if the pool is gone or the id unregistered.
std::optional<PoolTexture> TexturePoolEntry(u64 pool, u32 id);

/// NVNcopyRegion.
struct CopyRegion {
    s32 x, y, z, width, height, depth;
};

/// Op::kCopyBufferToTexture: the texture and view as they were when recorded.
/// Op::kCopyTextureToTexture: both textures and views as they were when recorded.
struct CopyTextureToTextureRecord {
    TextureState src;
    ViewArg src_view;
    CopyRegion src_region;
    TextureState dst;
    ViewArg dst_view;
    CopyRegion dst_region;
    u32 flags;
    u32 pad;
};

/// Op::kClearTexture (ClearTexture / ClearTexturei / ClearTextureui): the colour
/// as four 32-bit words (float, int or uint bits, as passed).
struct ClearTextureRecord {
    TextureState texture;
    ViewArg view;
    CopyRegion region;
    u32 color[4];
    u32 mask; // channels: bit 0 red ... bit 3 alpha
    u32 pad;
};

struct CopyBufferToTextureRecord {
    u64 buffer; // GPU address
    TextureState texture;
    ViewArg view;
    CopyRegion region;
    u32 flags;
};

} // namespace rt::nvn
