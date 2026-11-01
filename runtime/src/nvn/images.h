// Texture contents on the Vulkan side and the per-submission
// upload ring (5.0 rule 1), for queue.cpp's translation of records.
#pragma once

#include "texture.h"
#include "vk.h"

#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

namespace rt::nvn {

/// The Vulkan format with the same bytes per block and component layout as
/// the NVN format `id`, or VK_FORMAT_UNDEFINED where the layer has none yet.
VkFormat VulkanFormat(u32 id);

/// The authoritative copy of a texture's contents (6.3): one image per
/// texture storage address, in VK_IMAGE_LAYOUT_GENERAL once initialised.
struct TextureImage {
    Vulkan::vk::Image image;
    TextureState state; // the settings it was made for
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspect = 0;
    bool initialised = false; // its layout transition has been recorded
    bool storage = false;     // created for storage-image use (the device can store to its format)
    // the factor x 4 the image was made at (4: the game's size; ScaleN() when MakeImage's
    // rule scales it). Sizes and coordinates in the game's units become the image's by Up / UpSize.
    u32 scale_n = 4;
    bool Scaled() const { return scale_n != 4; }
    s32 Up(s32 v) const { return static_cast<s32>(static_cast<s64>(v) * scale_n / 4); }
    u32 UpSize(u32 v) const { return std::max(1u, static_cast<u32>(u64{v} * scale_n / 4)); }
    struct Attachment { // render.cpp: a view of the image used as a render target
        Vulkan::vk::ImageView view;
        Vulkan::vk::Framebuffer framebuffer; // for the single-attachment render pass of clears
        VkExtent2D extent;
        u32 layers;
    };
    std::map<std::array<u32, 3>, Attachment> attachments; // by (level, base layer, layers)
    std::map<std::array<u32, 7>, Vulkan::vk::ImageView> sampled; // SampledView, by its parameters
};

/// The run's factor x 4 (resolution.h: 2, 3, 4, 6, 8, 12); 4 at 720p.
u32 ScaleN();

/// The image for texture `t` (created at first use; a new one replaces it when
/// the storage gets other settings), its layout transition recorded into `cmd`
/// at first use.
std::shared_ptr<TextureImage> ImageFor(const TextureState& t, const Vulkan::vk::CommandBuffer& cmd);

/// A view of `img` (the image of texture `t`) as a shader reads it: of `type`, restricted by `view` (levels,
/// layers, swizzle, depth-stencil mode). Created at first use, kept with the image.
VkImageView SampledView(TextureImage& img, const TextureState& t, const ViewArg& view, VkImageViewType type);

/// A view of `img` as a storage image (one level; no swizzle), of `type`, restricted by `view`.
VkImageView StorageView(TextureImage& img, const TextureState& t, const ViewArg& view, VkImageViewType type);

/// The Vulkan sampler for entry `id` of the game's NVNsamplerPool at `pool` (sampler.cpp); stops the run
/// for an unregistered entry.
VkSampler SamplerFor(u64 pool, u32 id);

/// Guest memory read by a submission, copied at submit (5.0 rule 1). Chunks
/// of host-visible memory, reused once the submission has retired.
class UploadRing {
public:
    explicit UploadRing(Vulkan::MemoryUsage usage = Vulkan::MemoryUsage::Upload) : usage_(usage) {}
    struct Slice {
        VkBuffer buffer;
        u64 offset;
        u8* data; // nullptr in a device ring the CPU cannot map
    };
    /// `size` bytes at an `alignment` multiple, at least `min_offset` into its buffer (for a copy of part of a
    /// guest buffer bound at offset - min_offset, so the part keeps its place in the binding).
    Slice Allocate(u64 size, u64 alignment, u64 min_offset = 0);
    /// A texel-buffer view of `range` bytes at `slice` (of this ring), created once and kept with the ring.
    /// vkCreateBufferView took 20-30 ms on lavapipe while the queue was busy (it waits for the queue, it
    /// seems), and a submission's slices repeat frame after frame, so after the first frames none is created.
    VkBufferView TexelView(const Slice& slice, u64 range, VkFormat format);
    void Reset();
    /// Makes the GPU's writes visible to the CPU (a download ring, after the submission completed).
    void Invalidate();

private:
    Vulkan::MemoryUsage usage_;
    std::vector<Vulkan::vk::Buffer> chunks_;
    std::vector<u64> sizes_; // of chunks_ (a device ring's chunks have no mapping to measure)
    std::map<std::tuple<VkBuffer, u64, u64, VkFormat>, Vulkan::vk::BufferView> views_;
    size_t current_ = 0;
    u64 used_ = 0;
};

/// Descriptor sets for the pipelines with more descriptors than the device pushes (pipeline.h `push`), one
/// per draw or dispatch, from pools of the submission that records them; reused once it has retired.
class DescriptorArena {
public:
    VkDescriptorSet Allocate(VkDescriptorSetLayout layout);
    void Reset();

private:
    std::vector<Vulkan::vk::DescriptorPool> pools_;
    size_t current_ = 0;
};

/// An indexed draw drawn with parameters only the GPU had (draws.cpp): `indices` were copied from the index
/// buffer's start; checked at retirement against the parameters' first index + count (queue.cpp).
struct IndirectCheck {
    u64 params, indices;
};

/// What the translation of one submission's records writes into.
struct RenderState;

/// A GPU write to guest memory, applied when the submission retires, in command order (5.0 rules 2-3).
struct WriteBack {
    enum class Kind : u8 { kRing, kBytes, kFill, kEventWrite, kEventAdd, kCounter } kind;
    u64 address, size;
    const u8* ring = nullptr; // kRing: the data in the submission's download ring
    std::vector<u8> bytes;    // kBytes
    u32 value = 0;            // kFill (per word), kEvent*; kCounter: the report's index in its submission (queue.cpp)
    VkBuffer ring_buffer = VK_NULL_HANDLE; // kRing: where the GPU has the data (later reads copy from it)
    u64 ring_offset = 0;
    // kRing: the GPU command that wrote it (named in skip reasons, draws.cpp WriterName).
    enum class Origin : u8 { kUnknown, kStorageBuffer, kBufferTexture, kCopyBuffer, kCopyTexture } origin = Origin::kUnknown;
    u8 stage = 0xFF; // kStorageBuffer, kBufferTexture: the shader's NVN stage (5 = compute)
    std::string dump_name; // RT_NVN_DUMP_F9 (draws.cpp): the file its data goes to at retirement
};

/// RT_TRACE_WAITS (queue.cpp): the bytes a submission's draws and dispatches move, for a slow submission's report.
struct TraceBytes {
    u64 write_fills = 0;  // ShaderWrite: memory a shader writes, filled with what the GPU sees before it runs
    u32 writes = 0;       // ShaderWrite calls
    u64 largest_write = 0;
    u64 read_fills = 0;   // Snapshot: reads copied for the GPU (not reused within the submission)
    u64 gpu_parts = 0;    // of both fills: bytes the GPU copied from earlier results instead of the CPU
    u64 to_device = 0;    // compute writes: the filled slice into device memory before the dispatch
    u64 from_device = 0;  // and back into the download ring after it
};

/// The buffer cache (draws.cpp): guest memory kept on the GPU across a queue's submissions, refreshed
/// page by page when write tracking (writetrack.h) saw a CPU write. One per queue (queue.cpp); null when off
/// (RT_NVN_BUFFER_CACHE=0).
struct BufferCache;
std::shared_ptr<BufferCache> MakeBufferCache();
/// A page copy into the cache, recorded by the translation and executed before the submission's commands.
struct CacheCopy {
    VkBuffer src;
    u64 src_offset, dst_offset, size;
};
/// RT_NVN_BUFFER_CACHE_CHECK: a served read as the GPU found it in the cache (`got`, in the submission's download
/// ring, valid at retirement) against the bytes it should be (`expected`); compared when the submission retires.
struct CacheVerify {
    u64 address;
    u32 phase;
    const u8* got;
    std::vector<u8> expected;
    const u8* ref = nullptr; // RT_NVN_PAGE_COPY_CHECK: compare `got` with this GPU copy instead (size: expected.size())
    bool gpu_parts = false;  // the reference copy took parts only the GPU had
    std::string note;        // RT_NVN_PAGE_COPY_CHECK: how the page copy served it (draws.cpp PageCopy)
};
/// The cache's device buffer (the destination of every CacheCopy).
VkBuffer CacheBuffer(const BufferCache& cache);

/// The write index (draws.cpp IndexedRead): the latest write of each byte in one submission's list of
/// writes, by address - disjoint [lo, hi) ranges naming the write's index in the list. It catches up with the list
/// when read (`indexed`: the writes seen so far; the list is only appended to until the submission retires).
struct WriteIndex {
    std::map<u64, std::pair<u64, u32>> ranges; // lo -> {hi, index of the write}
    size_t indexed = 0;
    void Reset() { ranges.clear(), indexed = 0; }
};

struct Translator {
    const Vulkan::vk::CommandBuffer& cmd;
    UploadRing& upload;
    std::vector<std::shared_ptr<const void>>& keep; // alive until the submission retires
    RenderState* render = nullptr;                  // the queue's (render.h)
    UploadRing* download = nullptr;
    std::vector<WriteBack>* writebacks = nullptr;
    DescriptorArena* sets = nullptr;
    std::vector<IndirectCheck>* indirect_checks = nullptr;
    UploadRing* device = nullptr; // device memory for what compute shaders write (draws.cpp DispatchWrite)
    /// Copies of the dispatch being translated to record after it: its device-memory results into the
    /// download ring (draws.cpp DispatchWrite).
    std::vector<std::pair<UploadRing::Slice, UploadRing::Slice>> after_dispatch{};
    std::vector<u64> after_dispatch_sizes{};
    const void* dynamic_of = nullptr; // draws.cpp: the linked pipeline whose draw-time state `cmd` holds now
    /// RT_TRACE_WAITS (queue.cpp): called right before and after each vkCmdDispatch with its compute code's hash,
    /// to time the dispatch on the GPU.
    std::function<void(u64 code_hash, bool after)> time_dispatch{};
    TraceBytes* bytes = nullptr; // RT_TRACE_WAITS: counted here (draws.cpp)
    /// The queue's earlier submissions still in flight (their writes not in guest memory yet), oldest first.
    /// `used`: a read copied GPU data out of its download ring (queue.cpp keeps that ring until this
    /// submission retires).
    struct Earlier {
        const std::vector<WriteBack>* writes;
        bool used;
        WriteIndex* index = nullptr; // of `writes` (null: reads scan the list)
    };
    std::vector<Earlier> earlier{};
    WriteIndex* index = nullptr; // of `writebacks` (null: reads scan the list)
    /// Guest memory already copied for this submission (draws.cpp), by (address, size, GPU writes so far).
    /// draws.cpp Snapshot: the copies of guest memory made in this submission, by {address, size}, each with the
    /// number of writebacks it has been checked against (a later write over its bytes makes it stale).
    struct SnapshotEntry {
        UploadRing::Slice slice;
        size_t writes;
    };
    std::map<std::array<u64, 2>, SnapshotEntry> snapshots{};
    /// draws.cpp PageCopy: blocks of guest memory with room reserved in this
    /// submission's upload ring, by their first address: up to `hi`, in `slice` (the first byte at slice.offset, a
    /// multiple of 4096); pages are copied in when a read first needs them (`filled`: the count of writes recorded
    /// then plus one, 0 = not copied), and the recorded writes over the block seen so far (`scanned` of t.writebacks)
    /// are kept as {lo, hi, index} for the staleness check.
    struct Span {
        u64 hi;
        UploadRing::Slice slice;
        std::vector<size_t> filled;
        size_t scanned;
        std::vector<std::array<u64, 3>> written;
    };
    std::multimap<u64, Span> spans{};
    u64 longest_span = 0;
    /// draws.cpp CachedRead (Phase 2): the queue's cache, the copies into it this submission needs (queue.cpp
    /// records them first), this submission's value and the queue's retired value then.
    BufferCache* cache = nullptr;
    std::vector<CacheCopy>* cache_copies = nullptr;
    std::vector<CacheVerify>* cache_verify = nullptr;
    u64 submission = 0, retired = 0;
    /// draws.cpp Draw / Dispatch: the writebacks from this index on belong to the draw or dispatch being translated
    /// (its shader writes, recorded when its descriptors are set up - before it runs): their bytes change only once it
    /// has run, so a copy made meanwhile counts as made before them (PageCopy, Snapshot). SIZE_MAX: none.
    size_t pending_from = SIZE_MAX;
    /// Linear textures uploaded from guest memory in this submission (draws.cpp), by image.
    std::set<const TextureImage*> linear_uploads{};
};

/// Op::kCopyBufferToTexture.
void TranslateCopyBufferToTexture(Translator& t, const CopyBufferToTextureRecord& r);
/// Op::kCopyTextureToBuffer (the same record layout).
void TranslateCopyTextureToBuffer(Translator& t, const CopyBufferToTextureRecord& r);
/// Op::kCopyTextureToTexture.
void TranslateCopyTextureToTexture(Translator& t, const CopyTextureToTextureRecord& r);

struct CopyBufferRecord { // Op::kCopyBufferToBuffer
    u64 src, dst, size;
};
/// Op::kCopyBufferToBuffer (draws.cpp): the source as the GPU sees it at this point, copied on the GPU into
/// the download ring; the destination is a write-back from there.
void TranslateCopyBufferToBuffer(Translator& t, const CopyBufferRecord& r);
/// Op::kClearTexture.
void TranslateClearTexture(Translator& t, const ClearTextureRecord& r);

/// A read-back in flight (StartDownload): a level of an image copied into a host buffer by a submission
/// that signals `done`, queued after everything submitted before it. Destroy it on a thread that may use
/// the allocator (drippu's VMA is externally synchronized): the presenting thread, for captures.
struct ImageDownload {
    Vulkan::vk::Buffer buffer;
    Vulkan::vk::CommandPool pool;
    Vulkan::vk::CommandBuffers buffers; // (freed before the pool)
    Vulkan::vk::Fence done;
    std::shared_ptr<const TextureImage> image; // alive until the copy is done
    u64 size = 0;
    u32 width = 0, height = 0; // of the level read (a scaled image's: its own size)

    bool Ready() const { return done.GetStatus() == VK_SUCCESS; }
    /// The bytes, once Ready (waits for the copy otherwise).
    std::vector<u8> Take();
};
/// Starts reading level `level` of the image holding texture `t` (every layer or 3D slice, tightly packed
/// in format blocks) back without waiting for it. Null if the texture has no image.
std::unique_ptr<ImageDownload> StartDownload(const TextureState& t, u32 level);
/// Tests: StartDownload, then waits for it. Empty if the texture has no image.
std::vector<u8> DownloadImage(const TextureState& t, u32 level);

/// RT_NVN_LOG_TARGETS=1 (a test aid): how each image is used, rewritten into
/// targets.csv (the run folder) every 5 s - one row per image (address and settings): its uses by kind, the
/// levels drawn to, and the smallest and largest viewport of the draws into it (the game's dynamic resolution
/// shows as viewports smaller than the target). kToBuffer and kCapture read the image back into memory.
enum class ImageUse : u8 { kColor, kDepth, kClear, kSampled, kStorage, kCopySrc, kCopyDst, kUpload, kToBuffer, kPresent, kCapture, kCount };
void NoteImageUse(const TextureState& t, ImageUse use, u32 level = 0);
void NoteTargetViewport(const TextureState& t, float width, float height);

/// A full memory barrier, as every NVN Barrier is translated for now. Ends the open draw pass first.
void FullBarrier(const Vulkan::vk::CommandBuffer& cmd);

/// Draws share render passes: a draw keeps the pass the previous draw on `cmd` left open when the targets
/// are the same (pass, framebuffer, extent), otherwise it ends that one and begins its own. NVN orders a
/// draw's reads of what earlier draws wrote with a Barrier, which ends the pass.
void BeginDrawPass(const Vulkan::vk::CommandBuffer& cmd, const VkRenderPassBeginInfo& begin);
/// Ends the pass draws left open on `cmd`, if any. Everything that records commands outside a draw calls
/// it first (FullBarrier does), and the submission before ending the command buffer.
bool EndOpenPass(const Vulkan::vk::CommandBuffer& cmd); // true if it ended one (and barriered)
bool PassOpen(const Vulkan::vk::CommandBuffer& cmd);    // a render pass is open on it (RT_PROFILE_HEAVY)

/// Sample counting on a command buffer being translated (queue.cpp): one Vulkan occlusion query per
/// stretch of commands, as a query may not cross the begin or end of a render pass. BeginDrawPass and
/// EndOpenPass call Pause before they begin or end a pass and Resume after, so each query lies wholly inside
/// one pass or wholly outside passes (where it may span the passes other translations begin and end).
struct SampleCounting {
    virtual void Pause() = 0;
    virtual void Resume() = 0;

protected:
    ~SampleCounting() = default;
};
/// Registers `counting` for `cmd` (null: none).
void SetSampleCounting(const Vulkan::vk::CommandBuffer& cmd, SampleCounting* counting);

} // namespace rt::nvn
