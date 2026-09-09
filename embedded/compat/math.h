#pragma once
#if defined(__GNUC__) || defined(__clang__)
#define sqrtf __builtin_sqrtf
#define sqrt __builtin_sqrt
#define log10f __builtin_log10f
#define log10 __builtin_log10
#define fabsf __builtin_fabsf
#define fabs __builtin_fabs
#define expf __builtin_expf
#define exp __builtin_exp
#define sinf __builtin_sinf
#define cosf __builtin_cosf
#endif
