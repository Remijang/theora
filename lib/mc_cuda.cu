#include <cuda_runtime.h>
#include <stdint.h>
#include "state.h"
#include "mc_cuda.h"

__global__ void mc_kernel(const mc_pack p) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= p.nfrags) return;

  int mv1 = p.mv1[idx];
  int mv2 = p.mv2[idx];
  ptrdiff_t offset = p.frag_offsets[idx];
  int plane_idx = p.frag_plane[idx];
  int ystride = p.ystride[plane_idx];
  int reference_idx = p.refi[idx];

  const uint8_t *base = NULL;
  if (reference_idx == OC_FRAME_PREV)
    base = p.ref0[plane_idx];
  else if (reference_idx == OC_FRAME_GOLD)
    base = p.ref1[plane_idx];

  const uint8_t *src1 = base ? base + offset + mv1 : NULL;
  const uint8_t *src2 = base ? base + offset + mv2 : NULL;
  const int16_t *res = p.residue + idx * 64;
  uint8_t *dst = p.dst[plane_idx] + offset;
  bool use_avg = src1 && src2 && mv1 != mv2;

  for (int y = 0; y < 8; ++y) {
    #pragma unroll
    for (int x = 0; x < 8; ++x) {
      // intra predictor
      int pred = 128;
      if (base) {
        int tmp = src1[y * ystride + x];
        // half-pixel predictor
        if (use_avg) pred = (tmp + src2[y * ystride + x]) >> 1;
        // whole-pixel predictor
        else pred = tmp;
      }
      // PRED + RES
      int v = pred + res[y * 8 + x];
      // clamp
      v = v < 0 ? 0 : (v > 255 ? 255 : v);
      
      dst[y * ystride + x] = (uint8_t)v;
    }
  }
}

extern "C" void mc_cuda_launch(const mc_pack params) {
  int threads = 128;
  int blocks = (params->nfrags + threads - 1) / threads;
  mc_kernel<<<blocks, threads>>>(params);
}


