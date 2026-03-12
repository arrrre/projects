#include <math.h>
#include <string.h>

#include "fft.h"
#include "arena.h"

#define PI 3.14159265359

static inline cf32 cf32_mul(cf32 a, cf32 b) {
    return (cf32){ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}

b32 _fft_helper(cf32* out, cf32* in, u64 n, b32 invert) {
    b32 is_power_of_two = (n > 0) && ((n & (n - 1)) == 0);
    if (!is_power_of_two || !out || !in) { return false; }

    if (out != in) { memcpy(out, in, n * sizeof(cf32)); }

    u32 j = 0;
    for (u32 i = 0; i < n; i++) {
        if (i < j) {
            cf32 temp = out[i];
            out[i] = out[j];
            out[j] = temp;
        }
        u32 bit = (u32)n >> 1;
        for (; j & bit; bit >>= 1) { j ^= bit; }
        j ^= bit;
    }

    for (u64 len = 2; len <= n; len <<= 1) {
        // e^(-j * 2 * PI / len)
        f32 ang = 2.0f * PI / len * (invert ? -1 : 1);
        cf32 wlen = { cosf(ang), sinf(ang) };
        
        for (u64 i = 0; i < n; i += len) {
            cf32 w = { 1.0f, 0.0f };
            for (u32 k = 0; k < len / 2; k++) {
                cf32 u = out[i + k];
                cf32 v = cf32_mul(out[i + k + len / 2], w);
                
                out[i + k].re = u.re + v.re;
                out[i + k].im = u.im + v.im;
                out[i + k + len / 2].re = u.re - v.re;
                out[i + k + len / 2].im = u.im - v.im;
                
                w = cf32_mul(w, wlen);
            }
        }
    }

    if (invert) {
        f32 scale = 1.0f / (f32)n;
        for (u64 i = 0; i < n; i++) {
            out[i].re *= scale;
            out[i].im *= scale;
        }
    }

    return true;
}

b32 fft(cf32* out, cf32* in, u64 n) {
    return _fft_helper(out, in, n, false);
}

b32 ifft(cf32* out, cf32* in, u64 n) {
    return _fft_helper(out, in, n, true);
}

b32 _fftshift_helper(cf32* s, u64 n, u64 shift) {
    if (!s || n == 0) { return false; }

    mem_arena_temp scratch = arena_scratch_get(NULL, 0);
    
    cf32* temp = PUSH_ARRAY(scratch.arena, cf32, n);

    for (u64 i = 0; i < n; i++) {
        u64 target_idx = (i + shift) % n;
        temp[target_idx] = s[i];
    }

    memcpy(s, temp, n * sizeof(cf32));

    arena_scratch_release(scratch);

    return true;
}

b32 fftshift(cf32* s, u64 n) {
    return _fftshift_helper(s, n, (n + 1) / 2);
}

b32 ifftshift(cf32* s, u64 n) {
    return _fftshift_helper(s, n, n / 2);
}
