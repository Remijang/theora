#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include "decint.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void cuda_init(oc_dec_ctx *_dec, oc_dec_pipeline_state *_pipe);

void cuda_clear(oc_dec_ctx *_dec);

void cuda_transfer_to_device(oc_dec_ctx *_dec);

void cuda_transfer_to_host(oc_dec_ctx *_dec);

void cuda_process(
  oc_dec_ctx *_dec,
  oc_dec_pipeline_state *_pipe,
  int _pli
);

void cuda_enqueue(
  oc_dec_ctx *_dec,
  oc_dec_pipeline_state *_pipe,
  int _pli,
  ptrdiff_t _fragi,
  int16_t _dct_coeffs[128],
  int _last_zzi,
  uint16_t _dc_quant
);

#ifdef __cplusplus
}
#endif

#endif
