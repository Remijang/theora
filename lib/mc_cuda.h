#ifndef MC_CUDA_H_
#define MC_CUDA_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mc_pack {
  uint8_t *ref0[3];
  uint8_t *ref1[3];
  uint8_t *dst[3];
  int16_t *residue;
  int *mv1;
  int *mv2;
  int *refi;
  ptrdiff_t *frag_offsets;
  int *frag_plane;
  int *ystride;
  int nfrags;
} mc_pack;

void mc_cuda_launch(const mc_pack params);

#ifdef __cplusplus
}
#endif

#endif

