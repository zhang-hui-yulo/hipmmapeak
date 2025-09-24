#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#if defined(__GFX12__)
#include <hip/hip_fp8.h>
#endif
#include "common.h"


#if defined(__GFX12__)

using float_t_8 = float __attribute__((ext_vector_type(8)));
using half_t_8 = _Float16 __attribute__((ext_vector_type(8)));
using bf16_t_8 = __bf16 __attribute__((ext_vector_type(8)));
using int32_t_8 = int32_t __attribute__((ext_vector_type(8)));
using int8_t_8 = int8_t __attribute__((ext_vector_type(8)));
using int32_t_2 = int32_t __attribute__((ext_vector_type(2)));
using fp8_t_8 = __hip_fp8_storage_t __attribute__((ext_vector_type(8)));


__device__ void mma_f32f16_16_16_16_gfx12_(float* data) {
    half_t_8 am = { 0.01 };
    half_t_8 bm = { 0.02 };
    float_t_8 cm = { 0 };

    for (int k = 0; k < N_LOOP_INTERNAL; k++) {
        cm = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(am, bm, cm);
    }

    int idx = (threadIdx.y * blockDim.x  + threadIdx.x) * sizeof(cm) / sizeof(cm[0]);
    reinterpret_cast<decltype(cm)&>(data[idx]) = cm;
}

__device__ void mma_f32bf16_16_16_16_gfx12_(float* data) {
    bf16_t_8 am = { 0.01 };
    bf16_t_8 bm = { 0.02 };
    float_t_8 cm = { 0 };

    for (int k = 0; k < N_LOOP_INTERNAL; k++) {
        cm = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(am, bm, cm);
    }

    int idx = (threadIdx.y * blockDim.x + threadIdx.x) * sizeof(cm) / sizeof(cm[0]);
    reinterpret_cast<decltype(cm)&>(data[idx]) = cm;
}

__device__ void mma_f16f16_16_16_16_gfx12_(half* data) {
    half_t_8 am = { 0.01 };
    half_t_8 bm = { 0.02 };
    half_t_8 cm = { 0 };

    for (int k = 0; k < N_LOOP_INTERNAL; k++) {
        cm = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(am, bm, cm);
    }

    int idx = (threadIdx.y * blockDim.x + threadIdx.x) * sizeof(cm) / sizeof(cm[0]);
    reinterpret_cast<decltype(cm)&>(data[idx]) = cm;
}

__device__ void mma_bf16bf16_16_16_16_gfx12_(__hip_bfloat16* data) {
    bf16_t_8 am = { 0.01 };
    bf16_t_8 bm = { 0.02 };
    bf16_t_8 cm = { 0 };

    for (int k = 0; k < N_LOOP_INTERNAL; k++) {
        cm = __builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12(am, bm, cm);
    }

    int idx = (threadIdx.y * blockDim.x + threadIdx.x) * sizeof(cm) / sizeof(cm[0]);
    reinterpret_cast<decltype(cm)&>(data[idx]) = cm;
}

__device__ void mma_i32i8_16_16_16_gfx12_(int32_t* data) {
    int8_t_8 am = { 1, 1, 1, 1, 1, 1, 1, 1 };
    int8_t_8 bm = { 1, -1, 1, -1, 1, -1, 1, -1 };
    int32_t_8 cm = { 0 };

    for (int k = 0; k < N_LOOP_INTERNAL; k++) {
        cm = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, am, true, bm, cm, true);
    }

    int idx = (threadIdx.y * blockDim.x + threadIdx.x) * sizeof(cm) / sizeof(cm[0]);
    reinterpret_cast<decltype(cm)&>(data[idx]) = cm;
}

__device__ void mma_i32i4_16_16_16_gfx12_(int32_t* data) {
    int32_t am = { 1 };
    int32_t bm = { -1 };
    int32_t_8 cm = { 0 };

    for (int k = 0; k < N_LOOP_INTERNAL; k++) {
        cm = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, am, true, bm, cm, true);
    }

    int idx = (threadIdx.y * blockDim.x + threadIdx.x) * sizeof(cm) / sizeof(cm[0]);
    reinterpret_cast<decltype(cm)&>(data[idx]) = cm;
}

__device__ void mma_f32f8_16_16_16_gfx12_(float* data) {
    fp8_t_8 am = { (__hip_fp8_storage_t)__hip_fp8_e4m3(0.01) };
    fp8_t_8 bm = { (__hip_fp8_storage_t)__hip_fp8_e4m3(0.02) };
    float_t_8 cm = { 0 };

    for (int k = 0; k < N_LOOP_INTERNAL; k++) {
        cm = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(am, bm, cm);
    }

    int idx = (threadIdx.y * blockDim.x + threadIdx.x) * sizeof(cm) / sizeof(cm[0]);
    reinterpret_cast<decltype(cm)&>(data[idx]) = cm;
}

#endif

#if defined(__GFX12__)

__global__ void mma_f32f16_16_16_16_gfx12(float* data, int* rc) {
    mma_f32f16_16_16_16_gfx12_(data);
    *rc = 0;
}

__global__ void mma_f32bf16_16_16_16_gfx12(float* data, int* rc) {
    mma_f32bf16_16_16_16_gfx12_(data);
    *rc = 0;
}

__global__ void mma_f16f16_16_16_16_gfx12(half* data, int* rc) {
    mma_f16f16_16_16_16_gfx12_(data);
    *rc = 0;
}

__global__ void mma_bf16bf16_16_16_16_gfx12(__hip_bfloat16* data, int* rc) {
    mma_bf16bf16_16_16_16_gfx12_(data);
    *rc = 0;
}

__global__ void mma_i32i8_16_16_16_gfx12(int32_t* data, int* rc) {
    mma_i32i8_16_16_16_gfx12_(data);
    *rc = 0;
}

__global__ void mma_i32i4_16_16_16_gfx12(int32_t* data, int* rc) {
    mma_i32i4_16_16_16_gfx12_(data);
    *rc = 0;
}

__global__ void mma_f32f8_16_16_16_gfx12(float* data, int* rc) {
    mma_f32f8_16_16_16_gfx12_(data);
    *rc = 0;
}

#else

__global__ void mma_f32f16_16_16_16_gfx12(float* data, int* rc) {
    *rc = 1;
}

__global__ void mma_f32bf16_16_16_16_gfx12(float* data, int* rc) {
    *rc = 1;
}

__global__ void mma_f16f16_16_16_16_gfx12(half* data, int* rc) {
    *rc = 1;
}

__global__ void mma_bf16bf16_16_16_16_gfx12(__hip_bfloat16* data, int* rc) {
    *rc = 1;
}

__global__ void mma_i32i8_16_16_16_gfx12(int32_t* data, int* rc) {
    *rc = 1;
}

__global__ void mma_i32i4_16_16_16_gfx12(int32_t* data, int* rc) {
    *rc = 1;
}

__global__ void mma_f32f8_16_16_16_gfx12(float* data, int* rc) {
    *rc = 1;
}

#endif
