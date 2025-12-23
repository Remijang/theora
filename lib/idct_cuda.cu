#include <stdlib.h>
#include <string.h>
#include "state.h"
#include "dct.h"
#include <cuda_runtime.h>

__global__ void idct8x8_batch_kernel(
    const int16_t* in, 
    int16_t* out, 
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const int16_t* p_in = in + (idx * 64);
    int16_t* p_out = out + (idx * 64);

    int32_t block[64];

    // horizontal
    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        int32_t t[8], r;
        const int16_t* _x = p_in + (i << 3);

        t[0] = (OC_C4S4 * (int16_t)(_x[0] + _x[4])) >> 16;
        t[1] = (OC_C4S4 * (int16_t)(_x[0] - _x[4])) >> 16;
        t[2] = (OC_C6S2 * _x[2] >> 16) - (OC_C2S6 * _x[6] >> 16);
        t[3] = (OC_C2S6 * _x[2] >> 16) + (OC_C6S2 * _x[6] >> 16);
        t[4] = (OC_C7S1 * _x[1] >> 16) - (OC_C1S7 * _x[7] >> 16);
        t[5] = (OC_C3S5 * _x[5] >> 16) - (OC_C5S3 * _x[3] >> 16);
        t[6] = (OC_C5S3 * _x[5] >> 16) + (OC_C3S5 * _x[3] >> 16);
        t[7] = (OC_C1S7 * _x[1] >> 16) + (OC_C7S1 * _x[7] >> 16);

        r = t[4] + t[5];
        t[5] = (OC_C4S4 * (int16_t)(t[4] - t[5])) >> 16;
        t[4] = r;
        r = t[7] + t[6];
        t[6] = (OC_C4S4 * (int16_t)(t[7] - t[6])) >> 16;
        t[7] = r;

        r = t[0] + t[3];
        t[3] = t[0] - t[3];
        t[0] = r;
        r = t[1] + t[2];
        t[2] = t[1] - t[2];
        t[1] = r;
        r = t[6] + t[5];
        t[5] = t[6] - t[5];
        t[6] = r;

        block[(0 << 3) + i] = t[0] + t[7];
        block[(1 << 3) + i] = t[1] + t[6];
        block[(2 << 3) + i] = t[2] + t[5];
        block[(3 << 3) + i] = t[3] + t[4];
        block[(4 << 3) + i] = t[3] - t[4];
        block[(5 << 3) + i] = t[2] - t[5];
        block[(6 << 3) + i] = t[1] - t[6];
        block[(7 << 3) + i] = t[0] - t[7];
    }

    // vertical
    #pragma unroll 8
    for (int i = 0; i < 8; i++) {
        int32_t t[8], r;
        int32_t* _x = block + (i << 3);

        t[0] = (OC_C4S4 * (int16_t)(_x[0] + _x[4])) >> 16;
        t[1] = (OC_C4S4 * (int16_t)(_x[0] - _x[4])) >> 16;
        t[2] = (OC_C6S2 * _x[2] >> 16) - (OC_C2S6 * _x[6] >> 16);
        t[3] = (OC_C2S6 * _x[2] >> 16) + (OC_C6S2 * _x[6] >> 16);
        t[4] = (OC_C7S1 * _x[1] >> 16) - (OC_C1S7 * _x[7] >> 16);
        t[5] = (OC_C3S5 * _x[5] >> 16) - (OC_C5S3 * _x[3] >> 16);
        t[6] = (OC_C5S3 * _x[5] >> 16) + (OC_C3S5 * _x[3] >> 16);
        t[7] = (OC_C1S7 * _x[1] >> 16) + (OC_C7S1 * _x[7] >> 16);

        r = t[4] + t[5];
        t[5] = (OC_C4S4 * (int16_t)(t[4] - t[5])) >> 16;
        t[4] = r;
        r = t[7] + t[6];
        t[6] = (OC_C4S4 * (int16_t)(t[7] - t[6])) >> 16;
        t[7] = r;

        r = t[0] + t[3];
        t[3] = t[0] - t[3];
        t[0] = r;
        r = t[1] + t[2];
        t[2] = t[1] - t[2];
        t[1] = r;
        r = t[6] + t[5];
        t[5] = t[6] - t[5];
        t[6] = r;

        int32_t y0 = t[0] + t[7];
        int32_t y1 = t[1] + t[6];
        int32_t y2 = t[2] + t[5];
        int32_t y3 = t[3] + t[4];
        int32_t y4 = t[3] - t[4];
        int32_t y5 = t[2] - t[5];
        int32_t y6 = t[1] - t[6];
        int32_t y7 = t[0] - t[7];
        // scalar fixing
        p_out[(0 << 3) + i] = (int16_t)((y0 + 8) >> 4);
        p_out[(1 << 3) + i] = (int16_t)((y1 + 8) >> 4);
        p_out[(2 << 3) + i] = (int16_t)((y2 + 8) >> 4);
        p_out[(3 << 3) + i] = (int16_t)((y3 + 8) >> 4);
        p_out[(4 << 3) + i] = (int16_t)((y4 + 8) >> 4);
        p_out[(5 << 3) + i] = (int16_t)((y5 + 8) >> 4);
        p_out[(6 << 3) + i] = (int16_t)((y6 + 8) >> 4);
        p_out[(7 << 3) + i] = (int16_t)((y7 + 8) >> 4);
    }
}

extern "C" void idct8x8_batch(
    const ogg_int16_t *d_in,
    ogg_int16_t *d_out,
    int n
){
    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    idct8x8_batch_kernel<<<blocks, threads>>>(d_in, d_out, n);
}