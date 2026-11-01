// ooz (third_party/ooz, unmodified) built into the runtime, behind oodle.h. Its kraken.cpp holds the decoders and then
// its command-line program, whose main is renamed here and never called; lzna.cpp and bitknit.cpp are compiled as
// they are (CMake). Its stdafx.h asks for Windows' headers (Windows.h, intrin.h, tchar.h): the real ones on Windows,
// our stand-ins in ooz_posix/ elsewhere.
#include "oodle.h"

#define main ooz_command_line_main
#include "kraken.cpp"
#undef main

namespace rt::seed {

bool OodleDecompress(const void* src, size_t src_size, void* dst, size_t size) {
    return Kraken_Decompress(static_cast<const byte*>(src), src_size, static_cast<byte*>(dst), size) ==
           static_cast<int>(size);
}

} // namespace rt::seed
