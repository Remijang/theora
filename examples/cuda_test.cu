#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "../lib/mc_cuda.h"

// from lib/idct_cuda.cu
extern "C" void idct8x8_batch(const int16_t *d_in, int16_t *d_out, int n);

static void check(cudaError_t err) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s\n", cudaGetErrorString(err));
    exit(1);
  }
}

// from lib/idct.c
static void idct8_host(int16_t *y, int16_t *x) {
  const int32_t C1 = 64277;
  const int32_t C2 = 60547;
  const int32_t C3 = 54491;
  const int32_t C4 = 46341;
  const int32_t C5 = 36410;
  const int32_t C6 = 25080;
  const int32_t C7 = 12785;
  
  int32_t t[8], r, w[64];
  
  // horizontal
  for (int i = 0; i < 8; ++i) {
    const int16_t *_x = x + (i << 3);
    t[0] = (C4 * (int16_t) (_x[0] + _x[4])) >> 16;
    t[1] = (C4 * (int16_t) (_x[0] - _x[4])) >> 16;
    t[2] = ((C6 * _x[2]) >> 16) - ((C2 * _x[6]) >> 16);
    t[3] = ((C2 * _x[2]) >> 16) + ((C6 * _x[6]) >> 16);
    t[4] = ((C7 * _x[1]) >> 16) - ((C1 * _x[7]) >> 16);
    t[5] = ((C3 * _x[5]) >> 16) - ((C5 * _x[3]) >> 16);
    t[6] = ((C5 * _x[5]) >> 16) + ((C3 * _x[3]) >> 16);
    t[7] = ((C1 * _x[1]) >> 16) + ((C7 * _x[7]) >> 16);
    r = t[4] + t[5];
    t[5] = (C4 * (int16_t) (t[4] - t[5])) >> 16;
    t[4] = r;
    r = t[7] + t[6];
    t[6] = (C4 * (int16_t) (t[7] - t[6])) >> 16;
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
    w[(0 << 3) + i] = t[0] + t[7];
    w[(1 << 3) + i] = t[1] + t[6];
    w[(2 << 3) + i] = t[2] + t[5];
    w[(3 << 3) + i] = t[3] + t[4];
    w[(4 << 3) + i] = t[3] - t[4];
    w[(5 << 3) + i] = t[2] - t[5];
    w[(6 << 3) + i] = t[1] - t[6];
    w[(7 << 3) + i] = t[0] - t[7];
  }

  // vertical
  for (int i = 0; i < 8; ++i) {
    int32_t *_x = w + (i << 3);
    t[0] = (C4 * (int16_t)(_x[0] + _x[4])) >> 16;
    t[1] = (C4 * (int16_t)(_x[0] - _x[4])) >> 16;
    t[2] = ((C6 * _x[2]) >> 16) - ((C2 * _x[6]) >> 16);
    t[3] = ((C2 * _x[2]) >> 16) + ((C6 * _x[6]) >> 16);
    t[4] = ((C7 * _x[1]) >> 16) - ((C1 * _x[7]) >> 16);
    t[5] = ((C3 * _x[5]) >> 16) - ((C5 * _x[3]) >> 16);
    t[6] = ((C5 * _x[5]) >> 16) + ((C3 * _x[3]) >> 16);
    t[7] = ((C1 * _x[1]) >> 16) + ((C7 * _x[7]) >> 16);
    r = t[4] + t[5];
    t[5] = (C4 * (int16_t) (t[4] - t[5])) >> 16;
    t[4] = r;
    r = t[7] + t[6];
    t[6] = (C4 * (int16_t) (t[7] - t[6])) >> 16;
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
    y[(0 << 3) + i] = (int16_t)((y0 + 8) >> 4);
    y[(1 << 3) + i] = (int16_t)((y1 + 8) >> 4);
    y[(2 << 3) + i] = (int16_t)((y2 + 8) >> 4);
    y[(3 << 3) + i] = (int16_t)((y3 + 8) >> 4);
    y[(4 << 3) + i] = (int16_t)((y4 + 8) >> 4);
    y[(5 << 3) + i] = (int16_t)((y5 + 8) >> 4);
    y[(6 << 3) + i] = (int16_t)((y6 + 8) >> 4);
    y[(7 << 3) + i] = (int16_t)((y7 + 8) >> 4);
  }
}

static void mc_intra_whole_pixel(uint8_t *dst, uint8_t *ref, int16_t *res, int stride){
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 8; ++x) {
      // intra predictor
      int v = 128;
      // PRED + RES
      if (ref) v = ref[y * stride + x];
      v += res[y * 8 + x];
      // clamp
      v = v < 0 ? 0 : (v > 255 ? 255 : v);

      dst[y * stride + x] = (uint8_t) v;
    }
  }
}

int main(void) {
  // dct coefficient
  int16_t h_in[64];
  // idct result
  int16_t h_idct[64];
  int16_t h_idct_gpu[64];
  // reference frame
  uint8_t h_ref[64];
  // mc result
  uint8_t h_dst[64];
  uint8_t h_dst_cpu[64];
  // 
  int h_mv1[1] = {0};
  int h_mv2[1] = {0};
  int h_refi[1] = {2};
  int h_plane[1] = {0};
  int h_ystride[3] = {8, 8, 8};
  ptrdiff_t h_off[1] = {0};

  std::mt19937 rng(48763);
  std::uniform_int_distribution<int> rng1(-256, 256);
  std::uniform_int_distribution<int> rng2(0, 255);

  for (int i = 0; i < 64; ++i)
    h_in[i] = (int16_t) rng1(rng);
  h_in[0] = 512;

  for (int i = 0; i < 64; ++i) {
    if (h_refi[0] != 2) h_ref[i] = (uint8_t) rng2(rng);
    else h_ref[i] = 0;
  }

  int16_t *d_in = NULL;
  int16_t *d_out = NULL;
  uint8_t *d_ref = NULL;
  uint8_t *d_dst = NULL;
  int *d_mv1 = NULL;
  int *d_mv2 = NULL;
  int *d_refi = NULL;
  int *d_plane = NULL;
  int *d_ystride = NULL;
  ptrdiff_t *d_off = NULL;

  check( cudaMalloc(&d_in, 64 * sizeof(int16_t)) );
  check( cudaMalloc(&d_out, 64 * sizeof(int16_t)) );
  check( cudaMalloc(&d_ref, 64 * sizeof(uint8_t)) );
  check( cudaMalloc(&d_dst, 64 * sizeof(uint8_t)) );
  check( cudaMalloc(&d_mv1, 1 * sizeof(int)) );
  check( cudaMalloc(&d_mv2, 1 * sizeof(int)) );
  check( cudaMalloc(&d_refi, 1 * sizeof(int)) );
  check( cudaMalloc(&d_plane, 1 * sizeof(int)) );
  check( cudaMalloc(&d_ystride, 3 * sizeof(int)) );
  check( cudaMalloc(&d_off, 1 * sizeof(ptrdiff_t)) );

  mc_pack p = {
    .ref0 = {d_ref, NULL, NULL},
    .ref1 = {NULL, NULL, NULL},
    .dst = {d_dst, NULL, NULL},
    .residue = d_out,
    .mv1 = d_mv1,
    .mv2 = d_mv2,
    .refi = d_refi,
    .frag_offsets = d_off,
    .frag_plane = d_plane,
    .ystride = d_ystride,
    .nfrags = 1
  };

  // idct gpu
  check( cudaMemcpy(d_in, h_in, 64 * sizeof(int16_t), cudaMemcpyHostToDevice) );
  idct8x8_batch(d_in, d_out, 1);
  check( cudaMemcpy(h_idct_gpu, d_out, 64 * sizeof(int16_t), cudaMemcpyDeviceToHost) );

  // mc gpu
  check( cudaMemcpy(d_ref, h_ref, 64, cudaMemcpyHostToDevice) );
  check( cudaMemcpy(d_mv1, h_mv1, 1 * sizeof(int), cudaMemcpyHostToDevice) );
  check( cudaMemcpy(d_mv2, h_mv2, 1 * sizeof(int), cudaMemcpyHostToDevice) );
  check( cudaMemcpy(d_refi, h_refi, 1 * sizeof(int), cudaMemcpyHostToDevice) );
  check( cudaMemcpy(d_plane, h_plane, 1 * sizeof(int), cudaMemcpyHostToDevice) );
  check( cudaMemcpy(d_ystride, h_ystride, 3 * sizeof(int), cudaMemcpyHostToDevice) );
  check( cudaMemcpy(d_off, h_off, 1 * sizeof(ptrdiff_t), cudaMemcpyHostToDevice) );
  mc_cuda_launch(p);
  check( cudaMemcpy(h_dst, d_dst, 64 * sizeof(uint8_t), cudaMemcpyDeviceToHost) );

  // idct cpu + mc cpu
  idct8_host(h_idct, h_in);
  mc_intra_whole_pixel(h_dst_cpu, (h_refi[0] == 2) ? NULL : h_ref, h_idct, 8);

  // compare results
  int mis_count = 0;
  for (int i = 0; i < 64; ++i) {
    if (h_dst[i] != h_dst_cpu[i]) ++mis_count;
    if (h_idct_gpu[i] != h_idct[i]) ++mis_count;
  }
  printf("mismatch count: %d\n", mis_count);

  // calulate time
  const int iters = 1000000;
  cudaEvent_t start_gpu, stop_gpu;
  cudaEventCreate(&start_gpu);
  cudaEventCreate(&stop_gpu);

  cudaEventRecord(start_gpu);
  for(int i = 0; i < iters; ++i){
    idct8x8_batch(d_in, d_out, 1);
    mc_cuda_launch(p);
  }
  cudaEventRecord(stop_gpu);
  cudaEventSynchronize(stop_gpu);
  
  float total_time = 0;
  cudaEventElapsedTime(&total_time, start_gpu, stop_gpu);
  cudaEventDestroy(start_gpu);
  cudaEventDestroy(stop_gpu);

  auto start_cpu = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    idct8_host(h_idct, h_in);
    mc_intra_whole_pixel(h_dst_cpu, (h_refi[0] == 2) ? NULL : h_ref, h_idct, 8);
  }
  auto end_cpu = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> diff_cpu = end_cpu - start_cpu;

  printf("gpu average time: %.3f us\n", (total_time * 1000.0f) / iters);
  printf("cpu average time: %.3f us\n", diff_cpu.count() / iters);
  
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_ref);
  cudaFree(d_dst);
  cudaFree(d_mv1);
  cudaFree(d_mv2);
  cudaFree(d_refi);
  cudaFree(d_plane);
  cudaFree(d_ystride);
  cudaFree(d_off);
  return 0;
}

