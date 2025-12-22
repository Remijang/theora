#include <stdio.h>
#include <cuda_runtime.h>

extern "C" {
#include <theora/theora.h>
}

__global__ void test(unsigned char *dst) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *dst = 42;
  }
}

int main(void) {
  unsigned char *d_c;
  cudaMalloc(&d_c, 1);
  test<<<1, 1>>>(d_c);
  cudaDeviceSynchronize();
  unsigned char h_c;
  cudaMemcpy(&h_c, d_c, 1, cudaMemcpyDeviceToHost);
  cudaFree(d_c);

  theora_comment comment;
  theora_comment_init(&comment);
  theora_comment_clear(&comment);

  printf("test: comment %zu, char val %u\n", sizeof(comment), h_c);
  return 0;
}

