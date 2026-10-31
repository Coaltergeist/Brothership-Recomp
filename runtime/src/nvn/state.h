// The small state objects' layouts (state.cpp), for the files that read them
// from bind records (render.h's RenderState keeps them as bound).
#pragma once

#include "nvn.h"

namespace rt::nvn {

struct BlendState { // 8 bytes
    u8 target, src, dst, src_alpha, dst_alpha;
    u8 equations;     // color in bits 0-3, alpha in 4-7
    u8 advanced_mode; // < 256
    u8 advanced;      // overlap in bits 0-1, premultiplied source bit 2, normalized destination bit 3
};
static_assert(sizeof(BlendState) == 8);

struct ColorState { // 4 bytes
    u8 blend_enable; // one bit per target
    u8 logic_op, alpha_test, pad;
};
static_assert(sizeof(ColorState) == 4);

struct ChannelMaskState { // 4 bytes: 4 bits (r, g, b, a) per target
    u32 masks;
};

struct DepthStencilState { // 8 bytes
    u8 flags;          // depth test bit 0, depth write bit 1, stencil test bit 2
    u8 depth_func;
    u8 stencil_funcs;  // front in bits 0-3, back in 4-7
    u8 pad;
    u16 front_ops, back_ops; // fail, depth fail, depth pass: 4 bits each
};
static_assert(sizeof(DepthStencilState) == 8);

struct MultisampleState { // 24 bytes
    u8 flags; // enable 0, alpha-to-coverage 1, its dither 2, coverage-to-color 3, sample locations 4, their grid 5
    u8 samples, raster_samples, coverage_to_color_output;
    u8 coverage_modulation, pad[3];
    u8 sample_locations[16]; // reserved
};
static_assert(sizeof(MultisampleState) == 24);

struct PolygonState { // 4 bytes
    u8 cull_face, front_face, polygon_mode, offset_enables;
};

struct VertexAttribState { // 4 bytes: format bits 0-7, stream bits 8-12, relative offset bits 13-31
    u32 bits;
};

struct VertexStreamState { // 8 bytes
    u32 stride, divisor;
};

} // namespace rt::nvn
