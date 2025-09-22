#include <hip/hip_runtime.h>
#include "gfx11.h"
#include "gfx12.h"


#define hipCheckError() hipCheckError_(__FILE__, __LINE__)
inline void hipCheckError_(const char* file, int line) {
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        printf("%s in %s at line %d\n", hipGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

template <typename OutputType, unsigned M, unsigned N, unsigned K>
void run(void* kernel, float targetTime) {
    const int num_tb = 512;
    const int num_warps_per_tb = 4;
    const int warp_size = 32;
    dim3 grid(num_tb);
    dim3 block(warp_size, num_warps_per_tb);
    hipStream_t stream;
    hipStreamCreate(&stream);
    hipCheckError();
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipCheckError();
    void* data = nullptr;
    size_t nbytes = num_warps_per_tb * M * N * sizeof(OutputType);
    hipMalloc(&data, nbytes);
    hipCheckError();

    int* d_rc = nullptr;
    hipMalloc(&d_rc, sizeof(int));
    ((void (*)(void*, int*))kernel) <<< grid, block, 0, stream >>> (data, d_rc);
    int h_rc = 0;
    hipMemcpy(&h_rc, d_rc, sizeof(int), hipMemcpyDeviceToHost);
    if (h_rc != 0) {
        printf("not supported\n");
    } else {
        int n_loop = N_LOOP_CALIB;
        hipCheckError();
        hipEventRecord(start, stream);
        for (int i = 0; i < n_loop; i++) {
            ((void (*)(void*, int*))kernel) <<< grid, block, 0, stream >>> (data, d_rc);
        }
        hipCheckError();
        hipEventRecord(stop, stream);
        hipEventSynchronize(stop);
        float ms = 0;
        hipEventElapsedTime(&ms, start, stop);

        n_loop = (int)(targetTime * 1000 / ms * n_loop);
        n_loop = n_loop > 0 ? n_loop : N_LOOP_CALIB;

        hipEventRecord(start, stream);
        for (int i = 0; i < n_loop; i++)
        {
            ((void (*)(void*, int*))kernel) <<< grid, block, 0, stream >>> (data, d_rc);
        }
        hipCheckError();
        hipEventRecord(stop, stream);
        hipEventSynchronize(stop);
        hipStreamDestroy(stream);
        hipCheckError();
        hipEventElapsedTime(&ms, start, stop);
        float ops = 1.0f * num_tb * num_warps_per_tb * n_loop * N_LOOP_INTERNAL * M * N * K * 2;
        printf("%s: %.1f ms %.1f T(fl)ops\n", __func__, ms, ops / ms / 1.0e9f);
    }
    hipFree(d_rc);
    hipFree(data);
    hipCheckError();
}

void print_usage() {
    printf("Usage: mmapeak [options]\n");
    printf("Options:\n");
    printf("  -t <seconds>   Set target time for benchmarks in seconds (default: %.1f)\n", DEFAULT_TARGET_TIME);
    printf("  -h, --help     Show this help message\n");
}

int main(int argc, char** argv) {
    float targetTime = DEFAULT_TARGET_TIME;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            targetTime = atof(argv[++i]);
            if (targetTime <= 0) {
                printf("Error: Target time must be positive\n");
                print_usage();
                return 1;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage();
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            print_usage();
            return 1;
        }
    }

    int deviceCount = 0;
    hipGetDeviceCount(&deviceCount);
    hipCheckError();
    if (deviceCount == 0) {
        printf("No ROCm devices found\n");
        return 1;
    }

    for (int i = 0; i < deviceCount; i++) {
        printf("----------------------------------------\n");

        hipDeviceProp_t prop{};
        hipGetDeviceProperties(&prop, i);
        hipCheckError();
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %.1f GiB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessor count: %d\n", prop.multiProcessorCount);

        hipSetDevice(i);
        hipCheckError();

        printf("Running benchmarks with target time: %.1f seconds\n", targetTime);

        printf("\nRunning GFX12\n");
        printf("mma_f32f16_16_16_16_gfx12\n");
        run<float, 16, 16, 16>((void*)mma_f32f16_16_16_16_gfx12, targetTime);
        //printf("mma_f32bf16_16_16_16_gfx12\n");
        //run<float, 16, 16, 16>((void*)mma_f32bf16_16_16_16_gfx12, targetTime);
        printf("mma_f16f16_16_16_16_gfx12\n");
        run<half, 16, 16, 16>((void*)mma_f16f16_16_16_16_gfx12, targetTime);
        printf("mma_bf16bf16_16_16_16_gfx12\n");
        run<__hip_bfloat16, 16, 16, 16>((void*)mma_bf16bf16_16_16_16_gfx12, targetTime);
        printf("mma_i32i8_16_16_16_gfx12\n");
        run<int32_t, 16, 16, 16>((void*)mma_i32i8_16_16_16_gfx12, targetTime);
        printf("mma_i32i4_16_16_16_gfx12\n");
        run<int32_t, 16, 16, 16>((void*)mma_i32i4_16_16_16_gfx12, targetTime);
        printf("mma_f32f8_16_16_16_gfx12\n");
        run<int32_t, 16, 16, 16>((void*)mma_f32f8_16_16_16_gfx12, targetTime);

        printf("\nRunning GFX11\n");
        printf("mma_f32f16_16_16_16_gfx11\n");
        run<float, 16, 16, 16>((void*)mma_f32f16_16_16_16_gfx11, targetTime);
        printf("mma_i32i8_16_16_16_gfx11\n");
        run<float, 16, 16, 16>((void*)mma_i32i8_16_16_16_gfx11, targetTime);
    }
    return 0;
}
