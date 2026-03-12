#include <math.h>
#include <stdio.h>
#include <time.h>

#include "base.h"
#include "arena.h"
#include "prng.h"
#include "fft.h"

#define PI 3.14159265359

// fft -> fftshift -> ifftshift -> ifft
// Want signal centered at zero: ifftshift -> fft -> fftshift

void print_array(cf32* arr, u64 n, b32 real);
void test_fft(mem_arena* arena);

int main(void) {
    prng_seed(time(NULL), 42);

    mem_arena* perm_arena = arena_create(GiB(1), MiB(1));

    const u64 N = 32;

    cf32* s = PUSH_ARRAY(perm_arena, cf32, N);

    f32 lower = -0.5f, upper = 0.5f;
    for (u64 i = 0; i < N; i++) {
        s[i].re = prng_randf() * (upper - lower) + lower;
        s[i].im = 0.0f;
    }

    cf32* S = PUSH_ARRAY(perm_arena, cf32, N);

    printf("s:\n");
    print_array(s, N, true);

    fft(S, s, N);
    fftshift(S, N);

    printf("S:\n");
    print_array(S, N, false);

    // test_fft(perm_arena);

    arena_destroy(perm_arena);

    return 0;
}

void print_array(cf32* s, u64 n, b32 real) {
    for (u64 i = 0; i < n; i++) {
        f32 m = real ? s[i].re : sqrtf(s[i].re * s[i].re + s[i].im * s[i].im);
        printf("%.3f ", m);
    }
    printf("\n");
}

void test_fft(mem_arena* arena) {
    u64 N = 64; // Must be a power of 2 for this algorithm
    cf32* s = PUSH_ARRAY(arena, cf32, N);
    cf32* S = PUSH_ARRAY(arena, cf32, N);

    // Fill with a sine wave: sin(2 * PI * frequency * t)
    f32 frequency = 8.0f; 
    for (u64 i = 0; i < N; i++) {
        f32 t = (f32)i / N;
        s[i].re = sinf(2.0f * PI * frequency * t);
        s[i].im = 0.0f; // Input is real-only for this test
    }

    // Run the FFT
    fft(S, s, N);

    // Print the magnitudes to see the "spike"
    printf("Index | Magnitude\n----------------\n");
    for (u64 i = 0; i < N / 2; i++) { // Only need first half (Nyquist)
        f32 mag = sqrtf(S[i].re * S[i].re + S[i].im * S[i].im);
        if (mag > 0.1f) { 
            printf("%3llu   | %f <--- SPIKE!\n", i, mag);
        } else {
            printf("%3llu   | %f\n", i, mag);
        }
    }
}
