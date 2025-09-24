#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include "common.h"


#if defined(__GFX11__)

using float_t_8 = float __attribute__((ext_vector_type(8)));
using half_t_16 = _Float16 __attribute__((ext_vector_type(16)));
using bf16_t_16 = __bf16 __attribute__((ext_vector_type(16)));
using int32_t_8 = int32_t __attribute__((ext_vector_type(8)));
using int8_t_16 = int8_t __attribute__((ext_vector_type(16)));
using int32_t_2 = int32_t __attribute__((ext_vector_type(2)));


__device__ void mma_f32f16_16_16_16_gfx11_(float* data) {
    half_t_16 am = { 0.01 };
    half_t_16 bm = { 0.02 };
    float_t_8 cm = { 0 };

    for (int k = 0; k < N_LOOP_INTERNAL; k++) {
        cm = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(am, bm, cm);
    }

    int idx = (threadIdx.y * blockDim.x + threadIdx.x) * sizeof(cm) / sizeof(cm[0]);
    reinterpret_cast<decltype(cm)&>(data[idx]) = cm;
}

__device__ void mma_f32bf16_16_16_16_gfx11_(float* data) {
    bf16_t_16 am = { 0.01 };
    bf16_t_16 bm = { 0.02 };
    float_t_8 cm = { 0 };

    for (int k = 0; k < N_LOOP_INTERNAL; k++) {
        cm = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(am, bm, cm);
    }

    int idx = (threadIdx.y * blockDim.x + threadIdx.x) * sizeof(cm) / sizeof(cm[0]);
    reinterpret_cast<decltype(cm)&>(data[idx]) = cm;
}

__device__ void mma_f16f16_16_16_16_gfx11_(half2* data) {
    half_t_16 am = { 0.01 };
    half_t_16 bm = { 0.02 };
    half_t_16 cm = { 0 };

    for (int k = 0; k < N_LOOP_INTERNAL; k++) {
        cm = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(am, bm, cm, false);
    }

    int idx = (threadIdx.y * blockDim.x + threadIdx.x) * sizeof(cm) / sizeof(cm[0]);
    reinterpret_cast<decltype(cm)&>(data[idx]) = cm;
}

__device__ void mma_bf16bf16_16_16_16_gfx11_(__hip_bfloat162* data) {
    half_t_16 am = { 0.01 };
    half_t_16 bm = { 0.02 };
    half_t_16 cm = { 0 };

    for (int k = 0; k < N_LOOP_INTERNAL; k++) {
        cm = __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32(am, bm, cm, false);
    }

    int idx = (threadIdx.y * blockDim.x + threadIdx.x) * sizeof(cm) / sizeof(cm[0]);
    reinterpret_cast<decltype(cm)&>(data[idx]) = cm;
}

__device__ void mma_i32i8_16_16_16_gfx11_(int32_t* data) {
    int8_t_16 am = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
    int8_t_16 bm = { 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1 };
    int32_t_8 cm = { 0 };

    for (int k = 0; k < N_LOOP_INTERNAL; k++) {
        cm = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, am, true, bm, cm, true);
    }

    int idx = (threadIdx.y * blockDim.x + threadIdx.x) * sizeof(cm) / sizeof(cm[0]);
    reinterpret_cast<decltype(cm)&>(data[idx]) = cm;
}

__device__ void mma_i32i4_16_16_16_gfx11_(int32_t* data) {
    int32_t_2 am = { 1, 1 };
    int32_t_2 bm = { 1, -1 };
    int32_t_8 cm = { 0 };

    for (int k = 0; k < N_LOOP_INTERNAL; k++) {
        cm = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, am, true, bm, cm, true);
    }

    int idx = (threadIdx.y * blockDim.x + threadIdx.x) * sizeof(cm) / sizeof(cm[0]);
    reinterpret_cast<decltype(cm)&>(data[idx]) = cm;
}

#endif

#if defined(__GFX11__)

__global__ void mma_f32f16_16_16_16_gfx11(float* data, int* rc) {
    mma_f32f16_16_16_16_gfx11_(data);
    *rc = 0;
}

__global__ void mma_f32bf16_16_16_16_gfx11(float* data, int* rc) {
    mma_f32bf16_16_16_16_gfx11_(data);
    *rc = 0;
}

__global__ void mma_f16f16_16_16_16_gfx11(half2* data, int* rc) {
    mma_f16f16_16_16_16_gfx11_(data);
    *rc = 0;
}

__global__ void mma_bf16bf16_16_16_16_gfx11(__hip_bfloat162* data, int* rc) {
    mma_bf16bf16_16_16_16_gfx11_(data);
    *rc = 0;
}

__global__ void mma_i32i8_16_16_16_gfx11(int32_t* data, int* rc) {
    mma_i32i8_16_16_16_gfx11_(data);
    *rc = 0;
}

__global__ void mma_i32i4_16_16_16_gfx11(int32_t* data, int* rc) {
    mma_i32i4_16_16_16_gfx11_(data);
    *rc = 0;
}

#else

__global__ void mma_f32f16_16_16_16_gfx11(float* data, int* rc) {
    *rc = 1;
}

__global__ void mma_f32bf16_16_16_16_gfx11(float* data, int* rc) {
    *rc = 1;
}

__global__ void mma_f16f16_16_16_16_gfx11(half2* data, int* rc) {
    *rc = 1;
}

__global__ void mma_bf16bf16_16_16_16_gfx11(__hip_bfloat162* data, int* rc) {
    *rc = 1;
}

__global__ void mma_i32i8_16_16_16_gfx11(int32_t* data, int* rc) {
    *rc = 1;
}

__global__ void mma_i32i4_16_16_16_gfx11(int32_t* data, int* rc) {
    *rc = 1;
}

#endif
