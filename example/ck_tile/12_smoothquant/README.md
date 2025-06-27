# CK Tile SmoothQuant Implementation: Comprehensive Documentation

This document provides an in-depth analysis of the CK Tile SmoothQuant implementation, covering the mathematical foundations, instance configurations, source code architecture, and technical implementation details.

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Instance Configurations and Shapes](#instance-configurations-and-shapes)
4. [Source Code Architecture](#source-code-architecture)
5. [Pipeline Implementation Details](#pipeline-implementation-details)
6. [Key Functions and Data Flow](#key-functions-and-data-flow)
7. [Build and Usage](#build-and-usage)
8. [Performance Considerations](#performance-considerations)

## Overview

SmoothQuant is a quantization technique designed to reduce the computational and memory requirements of large language models while maintaining accuracy. This CK Tile implementation provides an optimized GPU kernel for performing SmoothQuant operations on AMD GPUs using the Composable Kernel framework.

### What SmoothQuant Does

SmoothQuant performs the following operations:
1. **Channel-wise Smoothing**: Applies per-channel scaling factors to reduce outliers
2. **Row-wise Quantization**: Quantizes the smoothed values using row-wise scaling factors
3. **Dynamic Scale Computation**: Computes quantization scales based on the maximum absolute values

## Mathematical Foundation

### Core Mathematical Operations

The SmoothQuant algorithm can be expressed as a sequence of mathematical operations:

#### 1. Channel-wise Smoothing
Given input matrix `X ∈ ℝ^(M×N)` and smooth scale vector `s ∈ ℝ^N`:

```
Y[i,j] = X[i,j] × s[j]
```

Where:
- `X` is the input activation matrix
- `s` is the per-channel smooth scale factor
- `Y` is the smoothed intermediate result

#### 2. Row-wise Scale Computation
For each row `i`, compute the maximum absolute value:

```
amax[i] = max_j |Y[i,j]|
```

Then compute the quantization scale:

```
yscale[i] = amax[i] / Q_max
```

Where `Q_max` is the maximum value of the target quantized type (e.g., 127 for int8).

#### 3. Quantization
The final quantized output is computed as:

```
QY[i,j] = saturate(round(Y[i,j] / yscale[i]))
```

Where `saturate()` clips values to the valid range of the target data type.

### Numerical Precision Considerations

- **Input Data Types**: FP16, BF16
- **Smooth Scale Data Type**: FP32 for numerical stability
- **Compute Data Type**: FP32 for intermediate calculations
- **Output Scale Data Type**: FP32
- **Quantized Output**: INT8

The use of FP32 for computations ensures numerical stability during the reduction operations and scale calculations.

## Instance Configurations and Shapes

The implementation supports various configurations optimized for different matrix dimensions and hardware characteristics.

### Supported Data Types

| Input Type | Smooth Scale | Compute Type | Output Scale | Quantized Output |
|------------|--------------|--------------|--------------|------------------|
| FP16       | FP32         | FP32         | FP32         | INT8             |
| BF16       | FP32         | FP32         | FP32         | INT8             |

### Shape Configurations

The kernel is instantiated for specific shape configurations optimized for common transformer model dimensions:

#### Common Dimensions (N dimension)
- **N=64, N=128**: Small embedding dimensions
- **N=256**: Common for smaller models
- **N=512**: Medium model sizes
- **N=768**: BERT-base dimension
- **N=1024**: GPT-2 dimension
- **N=1536**: Medium-large models
- **N=2048**: Large models (GPT-2 large)
- **N=3072**: Very large models
- **N=4096**: Extra large models (GPT-3 style)
- **N=8192**: Extremely large models

#### Block Shape Parameters

Each instance is configured with specific block shape parameters:

```cpp
template<
    typename DataType,
    index_t Repeat_M,         // Thread repetition along M dimension
    index_t Repeat_N,         // Thread repetition along N dimension  
    index_t ThreadPerBlock_M, // Number of threads along M dimension
    index_t ThreadPerBlock_N, // Number of threads along N dimension
    index_t Vector_N,         // Vector size for memory access
    bool kPadN,              // Whether to pad N dimension
    bool kTwoPass            // Use two-pass algorithm
>
```

#### Example Configuration Analysis

For `smoothquant_fp16_n4096_instance.cpp`:

```cpp
// rm rn tm  tn   vn   pd    2p
//  1, 2, 1, 256, 8,  true, false  // Config 1
//  1, 4, 1, 256, 4,  true, false  // Config 2
//  1, 2, 1,1024, 2,  true, false  // Config 3
//  1, 4, 1,1024, 1,  true, false  // Config 4
```

**Configuration 1**: `1, 2, 1, 256, 8, true, false`
- Block size: `1×512` (1×2×256×1 for Repeat_M×Repeat_N×ThreadPerBlock_N×Vector_N)
- 256 threads per block along N dimension
- Vector size 8 for coalesced memory access
- Single-pass algorithm

**Configuration 2**: `1, 4, 1, 256, 4, true, false`
- Block size: `1×1024`
- Higher repeat count along N for better compute utilization
- Smaller vector size due to higher repeat count

## Source Code Architecture

### Directory Structure

```
12_smoothquant/
├── smoothquant.hpp           # Main header with type configurations and traits
├── smoothquant.cpp          # Runtime implementation and validation
├── example_smoothquant.cpp  # Simple example with hardcoded configuration
├── instances/               # Pre-compiled kernel instances
│   ├── smoothquant_instance_common.hpp  # Common instance utilities
│   ├── smoothquant_*_instance.cpp       # Specific shape instances
│   └── smoothquant_fwd_api.cpp         # Public API implementation
└── script/                  # Test and benchmark scripts
```

### Core Include Hierarchy

```
include/ck_tile/ops/smoothquant.hpp
├── kernel/smoothquant_kernel.hpp           # Main kernel implementation
├── pipeline/smoothquant_pipeline_problem.hpp   # Problem definition
├── pipeline/smoothquant_pipeline_one_pass.hpp  # Single-pass algorithm
├── pipeline/smoothquant_pipeline_two_pass.hpp  # Two-pass algorithm
└── pipeline/smoothquant_pipeline_default_policy.hpp # Memory/compute policies
```

## Pipeline Implementation Details

### Two Pipeline Variants

#### 1. One-Pass Pipeline (`SmoothquantPipelineOnePass`)

**Advantages**:
- Lower memory bandwidth requirements
- Faster for smaller matrices
- Simpler control flow

**Process**:
1. Load input data and smooth scales
2. Compute smoothed values: `Y = X * SmoothScale`
3. Perform reduction to find row-wise maximum
4. Compute quantization scales
5. Quantize and store results

**Suitable for**: Cases where the entire row fits in registers/shared memory.

#### 2. Two-Pass Pipeline (`SmoothquantPipelineTwoPass`)

**Advantages**:
- Handles arbitrary matrix widths
- Better cache reuse
- More efficient for very wide matrices

**Process**:
1. **First Pass**: Iterate through matrix width, computing smoothed values and accumulating row-wise maximum
2. **Second Pass**: Re-read data and perform quantization using computed scales

**Suitable for**: Large N dimensions that don't fit in a single block.

### Memory Access Patterns

#### Coalesced Memory Access
The implementation uses vector loads/stores to ensure coalesced memory access:

```cpp
static constexpr index_t Vector_N = 8; // For fp16/bf16
```

#### Shared Memory Usage
Shared memory is used for:
- Cross-warp reduction synchronization
- Temporary storage during two-pass execution

### Reduction Operations

#### Cross-Lane Reduction
Uses warp-level primitives for efficient reduction within warps:

```cpp
auto reduce_absmax_func = ReduceOp::AbsMax{};
```

#### Cross-Warp Reduction
For blocks with multiple warps, uses shared memory for cross-warp synchronization:

```cpp
block_reduce2d_cross_warp_sync(absmax, smem, reduce_max_func);
```

#### Optimized Max3 Operation
For FP32 computations, uses hardware-accelerated `v_max3_f32` instruction:

```cpp
auto reduce_absmax3_func = [](auto acc_, auto v_0_, auto v_1_) {
    float rtn;
    asm volatile("v_max3_f32 %0, %1, abs(%2), abs(%3)"
                 : "=v"(rtn) : "v"(acc_), "v"(v_0_), "v"(v_1_));
    return rtn;
};
```

## Key Functions and Data Flow

### 1. Kernel Entry Point

```cpp
template<typename Pipeline_>
struct Smoothquant {
    CK_TILE_DEVICE void operator()(Kargs kargs) const;
};
```

**Responsibilities**:
- Set up tensor windows for input/output data
- Configure shared memory
- Invoke the pipeline

### 2. Pipeline Execution

```cpp
template<typename XWindow, typename SmoothScaleWindow, 
         typename QYWindow, typename YScaleWindow>
CK_TILE_DEVICE auto operator()(
    const XWindow& x_window,
    const SmoothScaleWindow& smscale_window,
    YScaleWindow& yscale_window,
    QYWindow& qy_window,
    index_t row_size,
    void* smem) const;
```

**Data Flow**:
1. **Load Phase**: Load input data and smooth scales using tile operations
2. **Compute Phase**: Apply smooth scaling: `Y = X * SmoothScale`
3. **Reduce Phase**: Compute row-wise absolute maximum values
4. **Scale Phase**: Compute quantization scales: `yscale = absmax / 127`
5. **Quantize Phase**: Apply quantization: `QY = saturate(Y / yscale)`
6. **Store Phase**: Write quantized results and scales to global memory

### 3. Host-Side Validation

```cpp
template<typename DataType>
bool run(const ArgParser& arg_parser);
```

**Validation Process**:
1. **Reference Computation**: Compute expected results on CPU
2. **GPU Execution**: Run optimized GPU kernel
3. **Comparison**: Compare results with configurable tolerance
4. **Performance Measurement**: Measure execution time and bandwidth

### 4. Memory Layout and Striding

The implementation supports non-contiguous memory layouts:

```cpp
struct SmoothquantHostArgs {
    const void* p_x;       // [M, N] input with stride
    const void* p_smscale; // [N] smooth scales
    void* p_yscale;        // [M] output scales  
    void* p_qy;            // [M, N] quantized output with stride
    index_t x_stride;      // Input row stride
    index_t y_stride;      // Output row stride
};
```

### 5. Type Configuration System

```cpp
template<typename DataType>
struct SmoothquantTypeConfig;

template<>
struct SmoothquantTypeConfig<half_t> {
    using XDataType = half_t;
    using SmoothScaleDataType = float;
    using YScaleDataType = float;
    using QYDataType = int8_t;
    using ComputeDataType = float;
};
```

This system ensures type safety and optimal precision for each component.

## Build and Usage

### Building the Example

```bash
# From CK root directory
mkdir build && cd build
sh ../script/cmake-ck-dev.sh ../ <arch>  # e.g., gfx90a, gfx942
make tile_smoothquant -j
```

### Command Line Usage

```bash
./build/bin/tile_smoothquant [options]

Options:
  -m <value>      M dimension (default: 3328)
  -n <value>      N dimension (default: 4096)  
  -v <0|1>        CPU validation (default: 1)
  -prec <type>    Precision: fp16, bf16 (default: fp16)
  -x_stride <value>  Input stride per row (default: -1, auto)
  -y_stride <value>  Output stride per row (default: -1, auto)
  -warmup <value>    Warmup iterations (default: 5)
  -repeat <value>    Timing iterations (default: 20)
```

### Example Usage

```bash
# Test with FP16, default size
./tile_smoothquant -prec fp16

# Test with BF16, custom size
./tile_smoothquant -prec bf16 -m 2048 -n 2048

# Performance test without validation
./tile_smoothquant -v 0 -warmup 10 -repeat 50
```

## Performance Considerations

### Memory Bandwidth Optimization

1. **Coalesced Access**: Vector loads/stores ensure optimal memory bandwidth
2. **Stride Support**: Handles non-contiguous layouts efficiently
3. **Cache Reuse**: Two-pass algorithm reuses L2 cache effectively

### Compute Optimization

1. **Hardware Instructions**: Uses `v_max3_f32` for efficient reductions
2. **Warp-Level Primitives**: Minimizes shared memory synchronization
3. **Register Pressure**: Carefully balanced to avoid spilling

### Scalability

1. **Block Shapes**: Multiple configurations for different matrix sizes
2. **Two-Pass Algorithm**: Handles arbitrary matrix widths
3. **Cross-Warp Sync**: Efficiently handles large blocks

### Expected Performance

- **Memory Bandwidth**: Near peak for coalesced access patterns
- **Compute Utilization**: High for reduction-heavy workloads
- **Latency**: Minimal due to optimized synchronization

The implementation achieves optimal performance through careful balance of memory access patterns, compute utilization, and hardware-specific optimizations, making it suitable for production deployment in large-scale transformer inference workloads.
