#include <cuda_runtime.h>
#include <ogg/ogg.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

/*Ensure C linkage for the C headers when compiling with NVCC (C++).*/
extern "C" {
#include "state.h"
#include "decint.h"
}

#include "mc_cuda.h"
#include "cuda_helper.h"

extern "C" void idct8x8_batch(
  const int16_t *in,
  int16_t *out,
  int n
);

// from state.c
static const signed char OC_MVMAP[2][64]={
  {
        -15,-15,-14,-14,-13,-13,-12,-12,-11,-11,-10,-10, -9, -9, -8,
     -8, -7, -7, -6, -6, -5, -5, -4, -4, -3, -3, -2, -2, -1, -1,  0,
      0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,
      8,  8,  9,  9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15
  },
  {
         -7, -7, -7, -7, -6, -6, -6, -6, -5, -5, -5, -5, -4, -4, -4,
     -4, -3, -3, -3, -3, -2, -2, -2, -2, -1, -1, -1, -1,  0,  0,  0,
      0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2,  3,  3,  3,  3,
      4,  4,  4,  4,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,  7
  }
};
static const signed char OC_MVMAP2[2][64]={
  {
      -1, 0,-1,  0,-1, 0,-1,  0,-1, 0,-1,  0,-1, 0,-1,
    0,-1, 0,-1,  0,-1, 0,-1,  0,-1, 0,-1,  0,-1, 0,-1,
    0, 1, 0, 1,  0, 1, 0, 1,  0, 1, 0, 1,  0, 1, 0, 1,
    0, 1, 0, 1,  0, 1, 0, 1,  0, 1, 0, 1,  0, 1, 0, 1
  },
  {
      -1,-1,-1,  0,-1,-1,-1,  0,-1,-1,-1,  0,-1,-1,-1,
    0,-1,-1,-1,  0,-1,-1,-1,  0,-1,-1,-1,  0,-1,-1,-1,
    0, 1, 1, 1,  0, 1, 1, 1,  0, 1, 1, 1,  0, 1, 1, 1,
    0, 1, 1, 1,  0, 1, 1, 1,  0, 1, 1, 1,  0, 1, 1, 1
  }
};

extern "C" {

void cuda_init(oc_dec_ctx *_dec, oc_dec_pipeline_state *_pipe) {
  _pipe->cuda_n = 0;
  _pipe->cuda_count = 0;
  _pipe->cuda_in = NULL;
  _pipe->cuda_out = NULL;
  _pipe->cuda_in_dev = NULL;
  _pipe->cuda_out_dev = NULL;
  _pipe->cuda_fragment_indices = NULL;
  _pipe->cuda_plane_indices = NULL;

  _pipe->use_cuda_mc = 1;
  for (int i = 0; i < 3; ++i) {
    _pipe->cuda_ref_dev[i] = NULL;
    _pipe->cuda_ref_gold_dev[i] = NULL;
    _pipe->cuda_dst_dev[i] = NULL;
  }
  _pipe->cuda_mv1_dev = NULL;
  _pipe->cuda_mv2_dev = NULL;
  _pipe->cuda_refi_dev = NULL;
  _pipe->cuda_plane_dev = NULL;
  _pipe->cuda_ystride_dev = NULL;
  _pipe->cuda_off_dev = NULL;
  _pipe->cuda_mv1_host = NULL;
  _pipe->cuda_mv2_host = NULL;
  _pipe->cuda_refi_host = NULL;
  _pipe->cuda_plane_host = NULL;
  _pipe->cuda_off_host = NULL;

  size_t blocks = (size_t) _dec->state.nfrags;
  if(blocks > 0) {
    _pipe->cuda_n = blocks;
    _pipe->cuda_in = (int16_t *) _ogg_malloc(blocks * 64 * sizeof(int16_t));
    _pipe->cuda_out = (int16_t *) _ogg_malloc(blocks * 64 * sizeof(int16_t));
    cudaMalloc(&_pipe->cuda_in_dev, blocks * 64 * sizeof(int16_t));
    cudaMalloc(&_pipe->cuda_out_dev, blocks * 64 * sizeof(int16_t));
    _pipe->cuda_fragment_indices = (ptrdiff_t *) _ogg_malloc(blocks * sizeof(ptrdiff_t));
    _pipe->cuda_plane_indices = (unsigned char *) _ogg_malloc(blocks * sizeof(unsigned char));
    
    for (int i = 0; i < 3; ++i) {
      ptrdiff_t stride =_dec->state.ref_ystride[i];
      size_t plane_bytes = (size_t) (stride < 0 ? -stride : stride) * (_dec->state.fplanes[i].nvfrags << 3);
      cudaMalloc(&_pipe->cuda_ref_dev[i], plane_bytes);
      cudaMalloc(&_pipe->cuda_ref_gold_dev[i], plane_bytes);
      cudaMalloc(&_pipe->cuda_dst_dev[i], plane_bytes);
    }
    cudaMalloc(&_pipe->cuda_mv1_dev, blocks * sizeof(int));
    cudaMalloc(&_pipe->cuda_mv2_dev, blocks * sizeof(int));
    cudaMalloc(&_pipe->cuda_refi_dev, blocks * sizeof(int));
    cudaMalloc(&_pipe->cuda_plane_dev, blocks * sizeof(int));
    cudaMalloc(&_pipe->cuda_ystride_dev, 3 * sizeof(int));
    cudaMalloc(&_pipe->cuda_off_dev, blocks * sizeof(ptrdiff_t));
    _pipe->cuda_mv1_host = (int *) _ogg_malloc(blocks * sizeof(int));
    _pipe->cuda_mv2_host = (int *) _ogg_malloc(blocks * sizeof(int));
    _pipe->cuda_refi_host = (int *) _ogg_malloc(blocks * sizeof(int));
    _pipe->cuda_plane_host = (int *) _ogg_malloc(blocks * sizeof(int));
    _pipe->cuda_off_host = (ptrdiff_t *) _ogg_malloc(blocks * sizeof(ptrdiff_t));



    if(
      !_pipe->cuda_in || !_pipe->cuda_out || !_pipe->cuda_in_dev || !_pipe->cuda_out_dev ||
      !_pipe->cuda_fragment_indices || !_pipe->cuda_plane_indices ||
      !_pipe->cuda_ref_dev[0] || !_pipe->cuda_ref_dev[1] || !_pipe->cuda_ref_dev[2] ||
      !_pipe->cuda_ref_gold_dev[0] || !_pipe->cuda_ref_gold_dev[1] || !_pipe->cuda_ref_gold_dev[2] ||
      !_pipe->cuda_dst_dev[0] || !_pipe->cuda_dst_dev[1] || !_pipe->cuda_dst_dev[2] ||
      !_pipe->cuda_mv1_dev || !_pipe->cuda_mv2_dev || !_pipe->cuda_refi_dev ||
      !_pipe->cuda_plane_dev || !_pipe->cuda_ystride_dev || !_pipe->cuda_off_dev ||
      !_pipe->cuda_mv1_host || !_pipe->cuda_mv2_host ||
      !_pipe->cuda_refi_host || !_pipe->cuda_plane_host || !_pipe->cuda_off_host
    ){
      _ogg_free(_pipe->cuda_in);
      _ogg_free(_pipe->cuda_out);
      if(_pipe->cuda_in_dev) cudaFree(_pipe->cuda_in_dev);
      if(_pipe->cuda_out_dev) cudaFree(_pipe->cuda_out_dev);
      _ogg_free(_pipe->cuda_fragment_indices);
      _ogg_free(_pipe->cuda_plane_indices);

      for (int i = 0; i < 3; ++i) {
        if(_pipe->cuda_ref_dev[i])cudaFree(_pipe->cuda_ref_dev[i]);
        if(_pipe->cuda_ref_gold_dev[i])cudaFree(_pipe->cuda_ref_gold_dev[i]);
        if(_pipe->cuda_dst_dev[i])cudaFree(_pipe->cuda_dst_dev[i]);
      }
      if(_pipe->cuda_mv1_dev) cudaFree(_pipe->cuda_mv1_dev);
      if(_pipe->cuda_mv2_dev) cudaFree(_pipe->cuda_mv2_dev);
      if(_pipe->cuda_refi_dev) cudaFree(_pipe->cuda_refi_dev);
      if(_pipe->cuda_plane_dev) cudaFree(_pipe->cuda_plane_dev);
      if(_pipe->cuda_ystride_dev) cudaFree(_pipe->cuda_ystride_dev);
      if(_pipe->cuda_off_dev) cudaFree(_pipe->cuda_off_dev);
      _ogg_free(_pipe->cuda_mv1_host);
      _ogg_free(_pipe->cuda_mv2_host);
      _ogg_free(_pipe->cuda_refi_host);
      _ogg_free(_pipe->cuda_plane_host);
      _ogg_free(_pipe->cuda_off_host);

      _pipe->cuda_n = 0;
      _pipe->cuda_count = 0;
      _pipe->cuda_in = NULL;
      _pipe->cuda_out = NULL;
      _pipe->cuda_in_dev = NULL;
      _pipe->cuda_out_dev = NULL;
      _pipe->cuda_fragment_indices = NULL;
      _pipe->cuda_plane_indices = NULL;

      _pipe->use_cuda_mc = 0;
      for (int i = 0; i < 3; ++i) {
        _pipe->cuda_ref_dev[i] = NULL;
        _pipe->cuda_ref_gold_dev[i] = NULL;
        _pipe->cuda_dst_dev[i] = NULL;
      }
      _pipe->cuda_mv1_dev = NULL;
      _pipe->cuda_mv2_dev = NULL;
      _pipe->cuda_refi_dev = NULL;
      _pipe->cuda_plane_dev = NULL;
      _pipe->cuda_ystride_dev = NULL; 
      _pipe->cuda_off_dev = NULL;
      _pipe->cuda_mv1_host = NULL;
      _pipe->cuda_mv2_host = NULL;
      _pipe->cuda_refi_host = NULL;
      _pipe->cuda_plane_host = NULL;
      _pipe->cuda_off_host = NULL;
    }
  }
}

void cuda_clear(oc_dec_ctx *_dec) {
  _dec->pipe.cuda_n = 0;
  _dec->pipe.cuda_count = 0;
  _ogg_free(_dec->pipe.cuda_in);
  _ogg_free(_dec->pipe.cuda_out);
  if (_dec->pipe.cuda_in_dev) {
    cudaFree(_dec->pipe.cuda_in_dev);
    _dec->pipe.cuda_in_dev = NULL;
  }
  if (_dec->pipe.cuda_out_dev) {
    cudaFree(_dec->pipe.cuda_out_dev);
    _dec->pipe.cuda_out_dev = NULL;
  }
  _ogg_free(_dec->pipe.cuda_fragment_indices);
  _ogg_free(_dec->pipe.cuda_plane_indices);

  _dec->pipe.use_cuda_mc = 1;
  for (int i = 0; i < 3; ++i) {
    if (_dec->pipe.cuda_ref_dev[i]) {
      cudaFree(_dec->pipe.cuda_ref_dev[i]);
      _dec->pipe.cuda_ref_dev[i] = NULL;
    }
    if (_dec->pipe.cuda_ref_gold_dev[i]) {
      cudaFree(_dec->pipe.cuda_ref_gold_dev[i]);
      _dec->pipe.cuda_ref_gold_dev[i] = NULL;
    }
    if (_dec->pipe.cuda_dst_dev[i]) {
      cudaFree(_dec->pipe.cuda_dst_dev[i]);
      _dec->pipe.cuda_dst_dev[i] = NULL;
    }
  }
  if (_dec->pipe.cuda_mv1_dev) {
    cudaFree(_dec->pipe.cuda_mv1_dev);
    _dec->pipe.cuda_mv1_dev = NULL;
  }
  if (_dec->pipe.cuda_mv2_dev) {
    cudaFree(_dec->pipe.cuda_mv2_dev);
    _dec->pipe.cuda_mv2_dev = NULL;
  }
  if (_dec->pipe.cuda_refi_dev) {
    cudaFree(_dec->pipe.cuda_refi_dev);
    _dec->pipe.cuda_refi_dev = NULL;
  }
  if (_dec->pipe.cuda_plane_dev) {
    cudaFree(_dec->pipe.cuda_plane_dev);
    _dec->pipe.cuda_plane_dev = NULL;
  }
  if (_dec->pipe.cuda_ystride_dev) {
    cudaFree(_dec->pipe.cuda_ystride_dev);
    _dec->pipe.cuda_ystride_dev = NULL;
  }
  _ogg_free(_dec->pipe.cuda_mv1_host);
  _ogg_free(_dec->pipe.cuda_mv2_host);
  _ogg_free(_dec->pipe.cuda_refi_host);
  _ogg_free(_dec->pipe.cuda_plane_host);
  _ogg_free(_dec->pipe.cuda_off_host);
}

void cuda_transfer_to_device(oc_dec_ctx *_dec){
  int ystride_host[3];
  int prev_idx = _dec->state.ref_frame_idx[OC_FRAME_PREV];
  int gold_idx = _dec->state.ref_frame_idx[OC_FRAME_GOLD];
  int dst_idx = _dec->state.ref_frame_idx[OC_FRAME_SELF];
  if (prev_idx < 0 || gold_idx < 0 || dst_idx < 0)
    return;

  for (int i = 0; i < 3; ++i) {
    unsigned char *prev_data = _dec->state.ref_frame_bufs[prev_idx][i].data;
    unsigned char *gold_data = gold_idx >= 0 ? _dec->state.ref_frame_bufs[gold_idx][i].data : NULL;
    unsigned char *dst_data = _dec->state.ref_frame_bufs[dst_idx][i].data;
    
    ptrdiff_t stride = _dec->state.ref_ystride[i];
    size_t plane_bytes = (size_t) (stride < 0 ? -stride : stride) * (_dec->state.fplanes[i].nvfrags << 3);

    cudaMemcpy(_dec->pipe.cuda_ref_dev[i], prev_data, plane_bytes, cudaMemcpyHostToDevice);
    if (gold_data) cudaMemcpy(_dec->pipe.cuda_ref_gold_dev[i], gold_data, plane_bytes, cudaMemcpyHostToDevice);
    else cudaMemcpy(_dec->pipe.cuda_ref_gold_dev[i], prev_data, plane_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(_dec->pipe.cuda_dst_dev[i], dst_data, plane_bytes, cudaMemcpyHostToDevice);
    ystride_host[i] = _dec->state.ref_ystride[i];
  }
  cudaMemcpy(_dec->pipe.cuda_ystride_dev, ystride_host, sizeof(ystride_host), cudaMemcpyHostToDevice);
}

void cuda_transfer_to_host(oc_dec_ctx *_dec){
  int dst_idx = _dec->state.ref_frame_idx[OC_FRAME_SELF];
  for (int i = 0; i < 3; ++i) {
    ptrdiff_t stride = _dec->state.ref_ystride[i];
    size_t plane_bytes = (size_t) (stride < 0 ? -stride : stride) * (_dec->state.fplanes[i].nvfrags << 3);
    cudaMemcpy(_dec->state.ref_frame_bufs[dst_idx][i].data, _dec->pipe.cuda_dst_dev[i], plane_bytes, cudaMemcpyDeviceToHost);
  }
}

int cuda_mv_off(const oc_theora_state *_state, int _offsets[2], int _pli, oc_mv _mv){
  int qpx = _pli != 0 && !(_state->info.pixel_fmt & 2);
  int qpy = _pli != 0 && !(_state->info.pixel_fmt & 1);
  int mx = OC_MVMAP[qpx][OC_MV_X(_mv) + 31];
  int my = OC_MVMAP[qpy][OC_MV_Y(_mv) + 31];
  int mx2 = OC_MVMAP2[qpx][OC_MV_X(_mv) + 31];
  int my2 = OC_MVMAP2[qpy][OC_MV_Y(_mv) + 31];
  int offs = my * _state->ref_ystride[_pli] + mx;
  if (mx2 || my2) {
    _offsets[1] = offs + my2 * _state->ref_ystride[_pli] + mx2;
    _offsets[0] = offs;
    return 2;
  }
  _offsets[0] = offs;
  return 1;
}

void cuda_process(
  oc_dec_ctx *_dec,
  oc_dec_pipeline_state *_pipe,
  int _pli
) {
  int count = _pipe->cuda_count;
  if (count <= 0 || _pipe->cuda_n <= 0) return;

  size_t bytes = (size_t) count * 64 * sizeof(int16_t);
  cudaMemcpy(_pipe->cuda_in_dev, _pipe->cuda_in, bytes, cudaMemcpyHostToDevice);
  idct8x8_batch(_pipe->cuda_in_dev, _pipe->cuda_out_dev, count);

  cuda_transfer_to_device(_dec);
  cudaMemcpy(_pipe->cuda_mv1_dev, _pipe->cuda_mv1_host, count * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(_pipe->cuda_mv2_dev, _pipe->cuda_mv2_host, count * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(_pipe->cuda_plane_dev, _pipe->cuda_plane_host, count * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(_pipe->cuda_refi_dev, _pipe->cuda_refi_host, count * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(_pipe->cuda_off_dev, _pipe->cuda_off_host, count * sizeof(ptrdiff_t), cudaMemcpyHostToDevice);
  mc_pack p = {
    .ref0 = {_pipe->cuda_ref_dev[0], _pipe->cuda_ref_dev[1], _pipe->cuda_ref_dev[2]},
    .ref1 = {_pipe->cuda_ref_gold_dev[0], _pipe->cuda_ref_gold_dev[1], _pipe->cuda_ref_gold_dev[2]},
    .dst = {_pipe->cuda_dst_dev[0], _pipe->cuda_dst_dev[1], _pipe->cuda_dst_dev[2]},
    .residue = _pipe->cuda_out_dev,
    .mv1 = _pipe->cuda_mv1_dev,
    .mv2 = _pipe->cuda_mv2_dev,
    .refi = _pipe->cuda_refi_dev,
    .frag_offsets = _pipe->cuda_off_dev,
    .frag_plane = _pipe->cuda_plane_dev,
    .ystride = _pipe->cuda_ystride_dev,
    .nfrags = count
  };
  mc_cuda_launch(p);
  cuda_transfer_to_host(_dec);
  _pipe->cuda_count = 0;
}

void cuda_enqueue(
  oc_dec_ctx *_dec,
  oc_dec_pipeline_state *_pipe,
  int _pli,
  ptrdiff_t _fragi,
  int16_t _dct_coeffs[128],
  int _last_zzi,
  uint16_t _dc_quant
) {
  if (_pipe->cuda_n <= 0) return;

  int16_t tmp[64];
  int idx = _pipe->cuda_count++;

  tmp[0] = (int16_t) (_dct_coeffs[0] * (int32_t) _dc_quant);
  for(int i = 1; i < 64; ++i)
    tmp[i] = _dct_coeffs[i];

  memcpy(_pipe->cuda_in + idx * 64, tmp, 64 * sizeof(int16_t));
  _pipe->cuda_fragment_indices[idx] = _fragi;
  _pipe->cuda_plane_indices[idx] = (unsigned char) _pli;

  int mv_off[2];
  int opt = cuda_mv_off(&_dec->state, mv_off, _pli, _dec->state.frag_mvs[_fragi]);
  _pipe->cuda_mv1_host[idx] = mv_off[0];
  _pipe->cuda_mv2_host[idx] = (opt == 2) ? mv_off[1] : mv_off[0];
  
  _pipe->cuda_refi_host[idx] = _dec->state.frags[_fragi].refi;
  _pipe->cuda_plane_host[idx] = _pli;
  
  int ref_idx_self = _dec->state.ref_frame_idx[OC_FRAME_SELF];
  ptrdiff_t plane_off = _dec->state.ref_frame_bufs[ref_idx_self][_pli].data - _dec->state.ref_frame_data[ref_idx_self];
  _pipe->cuda_off_host[idx] = _dec->state.frag_buf_offs[_fragi]-plane_off;
}

}