/* freestanding stand-in (README.txt): the compiler's built-ins */
#pragma once
#define fabs __builtin_fabs
#define fabsf __builtin_fabsf
#define sqrt __builtin_sqrt
#define sqrtf __builtin_sqrtf
#define floor __builtin_floor
#define floorf __builtin_floorf
#define ceil __builtin_ceil
#define ceilf __builtin_ceilf
#define trunc __builtin_trunc
#define truncf __builtin_truncf
#define copysign __builtin_copysign
#define copysignf __builtin_copysignf
#define fma __builtin_fma
#define fmaf __builtin_fmaf
#define isnan __builtin_isnan
#define isinf __builtin_isinf
#define signbit __builtin_signbit
