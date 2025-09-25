# HIPMMAPEAK - HIP Matrix Multiply Performance Benchmark

HIPMMAPEAK is a HIP-based benchmarking tool designed to measure the peak performance of matrix multiplication operations across various data types and tensor core configurations on AMD GPUs.

## Overview

This tool measures the throughput of AMD's Tensor Core dense operations using different precision formats:
- 4-bit integer (Int4)
- 8-bit integer (INT8)
- 8-bit floating point (FP8)
- 16-bit floating point (FP16, BF16)

## Building

### Using CMake

#### For Windows

```bash
set PATH=%HIP_PATH%bin;%PATH%
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DMMA_HIP_ARCHITECTURES=gfx1100;gfx1201
cmake --build build -j
```

#### For Linux

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMMA_HIP_ARCHITECTURES=gfx1100;gfx1201
cmake --build build -j
```

#### Note

Please use ROCm Toolkit version 6.4.2 (or later).

## Usage

```bash
./build/hipmmapeak [options]
```

### Options

- `-t <seconds>`: Set target time for benchmarks in seconds (default: 3.0)
- `-h, --help`: Show help message

## Compatibility

MMA operations that are not supported on your hardware will display "not supported", suggest to clean reinstall the driver with DDU to avoid weird bugs in driver cache.

## Architecture Support

- RDNA4 (9070XT, etc.): gfx12 family
- RDNA3 (7900XTX, etc.): gfx11 family

## License

This project is provided as-is.
