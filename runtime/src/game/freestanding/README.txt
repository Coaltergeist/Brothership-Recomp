Stand-ins for the C library headers the lifted code includes. The game module is built with only
clang's own headers (-ffreestanding -nostdlibinc) plus these, so a player's machine needs no Windows SDK, MinGW or
glibc headers: tools/build_game.sh. Maths and memory functions map to the compiler's built-ins; the rest is declared
here and defined in ../libc.c (module builds only), where only drippu's standalone code paths - never taken under
our runtime - reach them. The static build (NXRT_GAME=static) uses the real headers instead.
