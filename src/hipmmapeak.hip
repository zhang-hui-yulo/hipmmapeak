#include <sstream>
#include <unordered_set>
#include <hip/hip_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "gfx11.h"
#include "gfx12.h"


/**
 * \brief Perror macro with exit
 */
#if !defined(PEAK_ERROR_EXIT)
#  define PEAK_ERROR_EXIT(e)                                         \
      do {                                                           \
        hipError_t code = (e);                                      \
        if (code != hipSuccess) {                                   \
          fprintf(stderr, "<%s:%d> %s:\n    %s: %s\n",               \
                  __FILE__, __LINE__, #e,                            \
                  hipGetErrorName(code), hipGetErrorString(code)); \
          fflush(stderr);                                            \
          exit(1);                                                   \
        }                                                            \
      } while (0)
#endif

#if !defined(PEAK_CHECK_LAST)
#  define PEAK_CHECK_LAST() PEAK_ERROR_EXIT(hipPeekAtLastError()); PEAK_ERROR_EXIT(hipDeviceSynchronize())
#endif

#if !defined(PEAK_CHECK_ERROR)
#  define PEAK_CHECK_ERROR(e) PEAK_ERROR_EXIT(e)
#endif


class GPUTimer {
public:
    GPUTimer() {
        PEAK_CHECK_ERROR(hipEventCreate(&_start));
        PEAK_CHECK_ERROR(hipEventCreate(&_stop));
    }

    ~GPUTimer() {
        if (_start) {
            PEAK_CHECK_ERROR(hipEventDestroy(_start));
        }

        if (_stop) {
            PEAK_CHECK_ERROR(hipEventDestroy(_stop));
        }
    }

    void start() {
        PEAK_CHECK_ERROR(hipEventRecord(_start));
    }

    void stop() {
        PEAK_CHECK_ERROR(hipEventRecord(_stop));
        PEAK_CHECK_ERROR(hipEventSynchronize(_stop));
    }

    float elapsed() const {
        float ms = 0;
        PEAK_CHECK_ERROR(hipEventElapsedTime(&ms, _start, _stop));
        return ms;
    }

private:
    hipEvent_t _start = nullptr;
    hipEvent_t _stop = nullptr;
};

template <typename OutputType, unsigned M, unsigned N, unsigned K>
void run(void* kernel, float targetTime) {
    constexpr int num_tb = 512;
    constexpr int num_warps_per_tb = 4;
    constexpr int warp_size = 32;
    dim3 grid(num_tb);
    dim3 block(warp_size, num_warps_per_tb);
    
    thrust::device_vector<OutputType> output(num_warps_per_tb * M * N);
    thrust::device_vector<int> d_rc(1, 0);
    ((void (*)(void*, int*))kernel) <<< grid, block >>> (output.data().get(), d_rc.data().get());
    thrust::host_vector<int> h_rc(1, 0);

    if (h_rc[0] != 0) {
        printf("not supported\n");
    } else {
        GPUTimer timer;
        int n_loop = N_LOOP_CALIB;
        
        timer.start();
        for (int i = 0; i < n_loop; i++) {
            ((void (*)(void*, int*))kernel) <<< grid, block >>> (output.data().get(), d_rc.data().get());
        }
        timer.stop();
        float ms = timer.elapsed();

        n_loop = (int)(targetTime * 1000 / ms * n_loop);
        n_loop = n_loop > 0 ? n_loop : N_LOOP_CALIB;

        timer.start();
        for (int i = 0; i < n_loop; i++)
        {
            ((void (*)(void*, int*))kernel) <<< grid, block >>> (output.data().get(), d_rc.data().get());
        }
        timer.stop();
        ms = timer.elapsed();

        float ops = 1.0f * num_tb * num_warps_per_tb * n_loop * N_LOOP_INTERNAL * M * N * K * 2;
        printf("%s: %.1f ms %.1f T(fl)ops\n", __func__, ms, ops / ms / 1.0e9f);
    }
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
    PEAK_CHECK_ERROR(hipGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("No ROCm devices found\n");
        return 1;
    }

    std::unordered_set<std::string> arch_table;
    {
        std::stringstream archs(REQUIRED_HIP_ARCHITECTURES);
        std::string arch;
        while (archs >> arch) {
            arch_table.emplace(arch);
        }
    }

    for (int i = 0; i < deviceCount; i++) {
        printf("----------------------------------------\n");

        hipDeviceProp_t prop{};
        PEAK_CHECK_ERROR(hipGetDeviceProperties(&prop, i));

        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %.1f GiB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Multiprocessor count: %d\n", prop.multiProcessorCount);

        if (arch_table.count(prop.gcnArchName)) {
            PEAK_CHECK_ERROR(hipSetDevice(i));

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
        } else {
            printf("No GPU binary for %s, please add this architecture into REQUIRED_HIP_ARCHITECTURES.", prop.gcnArchName);
        }
    }
    return 0;
}
