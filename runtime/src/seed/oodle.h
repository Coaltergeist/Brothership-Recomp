// Oodle decompression for the shader seed (RELEASE_PLAN Phase 3). The game's archives - the .pak's global shader
// cache, the IoStore container's packages - are Oodle-compressed; they are decoded with ooz (third_party/ooz, powzix,
// GPL-3.0-or-later), checked byte-identical to Epic's oo2core on the game's GlobalShaderCache (39 Kraken blocks,
// 2026-11-01). Epic's library is never needed or shipped.
#pragma once

#include <cstddef>

namespace rt::seed {

/// The room `dst` needs past its end: the decoder writes up to this far beyond the output.
constexpr size_t kOodleSlack = 64;

/// Decompresses one Oodle block (Kraken, Mermaid, Selkie, Leviathan, LZNA, Bitknit) into exactly `size` bytes at
/// `dst` (which has `size + kOodleSlack` bytes); false when it does not decode to that.
bool OodleDecompress(const void* src, size_t src_size, void* dst, size_t size);

} // namespace rt::seed
