#ifndef FFT_H
#define FFT_H

#include "base.h"

b32 fft(cf32* out, cf32* in, u64 n);
b32 ifft(cf32* out, cf32* in, u64 n);
b32 fftshift(cf32* s, u64 n);
b32 ifftshift(cf32* s, u64 n);

#endif
