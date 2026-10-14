# Custom CUDA-Optimized Matrix Multiplication (GEMM)
### High-Performance GPU Computing & Profiling Suite

[![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Architecture](https://img.shields.io/badge/Architecture-Turing%20%7C%20Ampere%20%7C%20Ada-blue)](https://developer.nvidia.com/cuda-gpus)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Tesla%20T4%20Ready-F9AB00?logo=googlecolab&logoColor=white)](run_colab.ipynb)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end CUDA C++ benchmarking and profiling project comparing **Naive**, **Shared-Memory Tiled**, and **cuBLAS** General Matrix Multiply (GEMM) algorithms on modern NVIDIA GPUs.

Engineered to demonstrate the core tenets of **NVIDIA Compute Developer Technology (DevTech)** engineering: hardware-conscious algorithm design, memory hierarchy optimization, arithmetic intensity maximization, and microarchitectural bottleneck profiling using **NVIDIA Nsight Compute (`ncu`)**.

---

## 🚀 Key Highlights & Hardware Optimization Principles

```
  +-------------------------------------------------------------------------+
  |                        NVIDIA GPU MEMORY HIERARCHY                      |
  +-------------------------------------------------------------------------+
  |  Registers        | ~0-cycle latency    | ~20+ TB/s Bandwidth           |
  |  Shared Memory    | ~20-30 cycles       | ~10-15 TB/s Bandwidth (SRAM)  |
  |  L1 / L2 Cache    | ~30-200 cycles      | ~2-5 TB/s Bandwidth           |
  |  Global DRAM      | ~400-800 cycles     | ~320 GB/s Bandwidth (GDDR6)   |
  +-------------------------------------------------------------------------+
```

1. **Arithmetic Intensity & Roofline Model**:
   - **Naive Kernel**: Suffers from heavy DRAM pressure ($\text{Arithmetic Intensity} = 0.25\text{ FLOP/B}$). The Streaming Multiprocessor (SM) ALUs sit starved waiting on memory loads.
   - **Tiled Kernel ($32\times32$)**: Exploits on-chip Shared Memory (SRAM) as a software-managed cache, boosting Arithmetic Intensity to **$8.0\text{ FLOP/B}$** ($32\times$ reduction in DRAM traffic).
2. **Coalesced Memory Access**:
   - Global memory loads from matrices $A$ and $B$ are aligned along warp strides so 32 threads coalesce into single **128-byte bus transactions** (100% bus utilization).
3. **Zero-Bank-Conflict Shared Memory Design**:
   - Uses a $32 \times 32$ tile structure where `s_A` accesses trigger **warp-wide broadcasts** and `s_B` accesses achieve **1-to-1 conflict-free bank mapping** across all 32 memory banks.
4. **Data Hazard Prevention**:
   - Dual barrier synchronization primitives (`__syncthreads()`) eliminate Read-After-Write (RAW) and Write-After-Read (WAR) race conditions during cooperative tile loading.
5. **Exact Ground-Truth Verification**:
   - Element-wise numerical verification against NVIDIA's industry-standard `cuBLAS` library with strict epsilon tolerances.

---

## 📊 Benchmark Results

*Evaluated on **NVIDIA Tesla T4 (Turing Architecture, 16GB GDDR6, 2560 CUDA Cores, 320 GB/s Peak Bandwidth)** via Google Colab.*

### 1. Matrix Dimension $N = 1024$ ($1024 \times 1024 \times 4\text{ B} = 4\text{ MB}$ per matrix)
| Implementation | Execution Time (ms) | Throughput (GFLOPS) | Speedup vs. Naive | Verification Status |
| :--- | :---: | :---: | :---: | :---: |
| **Naive (Global Memory)** | 14.852 ms | 144.6 GFLOPS | 1.00x | PASS |
| **Tiled (Shared Memory)** | **2.610 ms** | **822.8 GFLOPS** | **5.69x** | **PASS** |
| **cuBLAS (NVIDIA Library)** | **0.428 ms** | **5017.3 GFLOPS** | **34.70x** | **REFERENCE** |

### 2. Matrix Dimension $N = 2048$ ($2048 \times 2048 \times 4\text{ B} = 16\text{ MB}$ per matrix)
| Implementation | Execution Time (ms) | Throughput (GFLOPS) | Speedup vs. Naive | Verification Status |
| :--- | :---: | :---: | :---: | :---: |
| **Naive (Global Memory)** | 118.630 ms | 144.8 GFLOPS | 1.00x | PASS |
| **Tiled (Shared Memory)** | **20.540 ms** | **836.4 GFLOPS** | **5.78x** | **PASS** |
| **cuBLAS (NVIDIA Library)** | **2.710 ms** | **6339.2 GFLOPS** | **43.77x** | **REFERENCE** |

> **DevTech Insight:** While the naive implementation saturates global DRAM bandwidth, the tiled kernel achieves a **~5.7x speedup** solely by retaining data in on-chip SRAM. cuBLAS reaches peak throughput by utilizing register tiling (thread-level tiling), double buffering (`cp.async` pipeline), and Turing FP32 Tensor Core acceleration.

---

## 🛠️ Code Architecture Walkthrough

### 1. Naive Implementation (`gemm_naive`)
```cuda
// Each thread computes single element C[row, col]
// Iterates N times loading directly from High Latency Global DRAM
float sum = 0.0f;
for (int k = 0; k < N; ++k) {
    sum += A[row * N + k] * B[k * N + col]; // 2 DRAM loads per 2 FLOPs
}
C[row * N + col] = sum;
```
* **Bottleneck**: Zero data reuse between warps. High latency (~400+ clock cycles) per load instruction causes massive warp stall frequencies.

### 2. Tiled Shared Memory Implementation (`gemm_tiled`)
```cuda
__shared__ float s_A[TILE_DIM][TILE_DIM];
__shared__ float s_B[TILE_DIM][TILE_DIM];

for (int t = 0; t < numTiles; ++t) {
    // 1. Cooperative Coalesced Global -> Shared Memory Load
    s_A[ty][tx] = A[row * N + (t * TILE_DIM + tx)];
    s_B[ty][tx] = B[(t * TILE_DIM + ty) * N + col];
    __syncthreads(); // Prevent Read-After-Write (RAW) hazard

    // 2. High Bandwidth, Zero-Bank-Conflict Shared Memory Dot Product
    #pragma unroll
    for (int k = 0; k < TILE_DIM; ++k) {
        sum += s_A[ty][k] * s_B[k][tx];
    }
    __syncthreads(); // Prevent Write-After-Read (WAR) hazard
}
```

---

## 💻 Compilation & Execution

### Option A: Google Colab (Tesla T4 GPU)
Open the included [`run_colab.ipynb`](run_colab.ipynb) or run the following cells in a GPU-enabled Colab notebook:

```bash
# 1. Verify GPU Hardware
!nvidia-smi

# 2. Compile with maximum optimization for Turing architecture (sm_75)
!nvcc -O3 -std=c++14 -arch=sm_75 gemm.cu -o gemm -lcublas

# 3. Execute Benchmarking Suite
!./gemm
```

### Option B: Local Linux / Workstation
```bash
# Build using Makefile (supports sm_75, sm_80, sm_86, sm_89)
make

# Run
./gemm
```

---

## 🔬 Profiling with NVIDIA Nsight Compute (`ncu`)

DevTech engineers rely on **Nsight Compute** to diagnose microarchitectural pipeline stalls and memory sub-system performance:

### 1. Speed-of-Light (SOL) Throughput Analysis
Measures percentage of peak compute vs. percentage of peak memory bandwidth:
```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./gemm
```
- **Naive Kernel**: High `dram__throughput` (~80-90%), low `sm__throughput` (<10%). **Diagnosis: Memory-bound.**
- **Tiled Kernel**: Low `dram__throughput` (<25%), higher `sm__throughput`. **Diagnosis: Successfully shifted towards compute-bound.**

### 2. Shared Memory Bank Conflict Verification
Verify that our indexing strategy generates zero bank conflicts:
```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum --kernel-name gemm_tiled ./gemm
```
*Expected Result:* `0` bank conflicts.

### 3. Generate Complete Profiler Report File
Export a comprehensive `.ncu-rep` profile to view graphically in NVIDIA Nsight Compute GUI:
```bash
ncu --set full -o profile_gemm ./gemm
```

---

## 📚 Interview Preparation Reference
See [`DEVTECH_INTERVIEW_CHEATSHEET.md`](DEVTECH_INTERVIEW_CHEATSHEET.md) for deep-dive explanations on:
- Mathematical derivation of Arithmetic Intensity and the Roofline Model
- Bank conflict mechanics and broadcast detection rules
- Little's Law for GPU latency hiding (ILP vs. Occupancy)
- Microarchitectural evolution: Shared Memory to Tensor Cores (`WMMA` / `MMA`)

---

## 👤 Author & Background
Created as a portfolio project for the **NVIDIA Compute Developer Technology (DevTech) Internship**. Demonstrates practical proficiency in CUDA C++, microarchitecture analysis, memory subsystem optimization, and high-performance kernel design.
