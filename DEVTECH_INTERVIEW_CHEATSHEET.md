# NVIDIA DevTech Technical Interview Cheat Sheet: High-Performance GEMM & CUDA Optimization

This guide is structured for candidates interviewing for the **Compute Developer Technology (DevTech)** role at NVIDIA. DevTech engineers specialize in analyzing customer workloads, finding architectural bottlenecks, and maximizing hardware utilization on NVIDIA GPUs.

---

## 1. Core Architectural Concepts

### A. Arithmetic Intensity (AI) & The Roofline Model
* **Definition:** $\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes transferred across DRAM bus}}$.
* **Single Precision GEMM Operations:** For square matrices of dimension $N$:
  $$\text{FLOPs} = 2 \times N^3 \quad (\text{1 multiply + 1 accumulate per element})$$
* **Naive GEMM Memory Traffic:**
  - Every thread computes 1 element of $C$, performing $N$ loads of $A$ and $N$ loads of $B$ directly from Global Memory.
  - Total DRAM Reads: $2 \times N^3 \times 4\text{ bytes}$.
  - Arithmetic Intensity: $\frac{2N^3 \text{ FLOPs}}{8N^3 \text{ bytes}} = 0.25 \text{ FLOP/byte}$.
  - **Result:** Severely **Memory-Bound**. On a Tesla T4 (peak bandwidth 320 GB/s), theoretical throughput is capped at $0.25 \times 320 \times 10^9 \approx 80 \text{ GFLOPS}$ (out of 8.1 TFLOPS FP32 peak!).
* **Tiled GEMM Memory Traffic (Tile dimension $B = 32$):**
  - Blocks load a $B \times B$ tile of $A$ and $B$ into Shared Memory (SRAM) once, then reuse it $B$ times in inner product calculations.
  - DRAM Reads: $\frac{2 \times N^3}{B} \times 4\text{ bytes}$.
  - Arithmetic Intensity: $\frac{2N^3}{\frac{8N^3}{B}} = \frac{B}{4} = \frac{32}{4} = 8.0 \text{ FLOP/byte}$.
  - **Result:** $32\times$ reduction in DRAM traffic! Shifting kernel performance up the Roofline curve toward the compute-bound ceiling.

---

### B. Memory Coalescing
* **Hardware Mechanism:** The memory controller serves Global Memory accesses in contiguous 32-byte, 64-byte, or 128-byte transactions.
* **Coalescing Condition:** When the 32 threads in a warp access contiguous 4-byte memory addresses (e.g., `threadIdx.x` indexing consecutive elements in row-major order: `row * N + (col_base + threadIdx.x)`), the GPU hardware collapses the 32 individual 4-byte requests into a single 128-byte bus transaction.
* **Non-Coalesced / Strided Access:** If threads access memory with a large stride (e.g., column-wise in row-major matrix: `(row_base + threadIdx.x) * N + col`), each thread hits a different cache line. A single instruction then triggers 32 separate 32-byte transactions (1024 bytes fetched for 128 bytes needed $\rightarrow$ **12.5% bus efficiency**).

---

### C. Shared Memory Architecture & Bank Conflicts
* **Physical Structure:** Shared Memory is on-chip SRAM organized into **32 independent memory banks** (4 bytes per bank).
* **Bank Mapping Formula:**
  $$\text{Bank ID} = \left(\frac{\text{Byte Address}}{4}\right) \bmod 32$$
* **Conflict Condition:** A bank conflict occurs when two or more threads within the same warp request different addresses residing in the same memory bank. The memory controller must serialize these accesses, multiplying latency.
* **Special Cases:**
  - **Bank Conflict Free:** All 32 threads access distinct banks ($1\times$ latency).
  - **Broadcast:** All 32 threads access the exact same address in a bank. The hardware broadcasts the single 32-bit word to all requesting threads simultaneously ($1\times$ latency).
* **Our Tiled Kernel Analysis:**
  - `s_A[ty][k]`: All threads in a warp share identical `ty` and `k`. This is a 100% **Broadcast** (0 conflicts).
  - `s_B[k][tx]`: Thread `tx` (ranging 0 to 31) accesses index `tx`. Bank ID is $tx \bmod 32$, so each thread hits a unique bank (Banks 0 to 31). This is 100% **Conflict-Free**!

---

### D. Thread Synchronization (`__syncthreads()`)
* **What it does:** Hardware execution barrier for all threads within a CTA (Cooperative Thread Array / Thread Block). No thread in the block passes the barrier until all threads reach it.
* **Hazard Prevention in GEMM:**
  1. **RAW (Read-After-Write) Hazard:** Barrier 1 ensures all threads have finished loading tiles `s_A` and `s_B` into shared memory before any thread begins computing the dot product.
  2. **WAR (Write-After-Read) Hazard:** Barrier 2 ensures all threads have finished consuming the current tile from shared memory before any thread advances to the next loop iteration and overwrites `s_A` and `s_B`.
* **Cost:** `__syncthreads()` causes warp stalls if threads in the block have imbalanced workloads. In our tiled GEMM, all threads perform uniform computation, minimizing synchronization latency.

---

## 2. DevTech Interview Q&A (Rapid Fire)

### Q1: Why does cuBLAS perform significantly faster than a basic shared memory tiled kernel?
**Answer:**
1. **Thread Tiling / Register Tiling:** cuBLAS does not compute a single element per thread. Each thread computes a $4\times4$, $8\times8$, or $16\times16$ sub-tile stored in SM registers, maximizing register reuse and reducing Shared Memory loads.
2. **Double Buffering / Software Pipelining:** cuBLAS pipelines asynchronous memory loads (`cp.async` on Ampere+) so global memory fetches for tile $t+1$ overlap with math execution on tile $t$.
3. **Tensor Cores (HMMA / IMMA):** Modern GPUs (Volta, Turing, Ampere, Hopper, Blackwell) contain dedicated matrix math hardware units. cuBLAS utilizes mixed-precision Tensor Cores executing matrix multiply-accumulate instructions in hardware.
4. **Epilogue Vectorization:** cuBLAS loads and stores using 128-bit vectorized instructions (`float4` / `ld.global.v4.f32`).

---

### Q2: How do you choose the tile size (`TILE_DIM`)?
**Answer:**
- Must balance **Occupancy**, **Shared Memory capacity**, and **Register pressure**:
  - `TILE_DIM = 16`: $16 \times 16 = 256$ threads/block. $2 \times 16 \times 16 \times 4\text{ B} = 2\text{ KB}$ shared memory. Good occupancy, but lower arithmetic intensity ($4 \text{ FLOP/B}$).
  - `TILE_DIM = 32`: $32 \times 32 = 1024$ threads/block (max threads per block on modern architectures). $2 \times 32 \times 32 \times 4\text{ B} = 8\text{ KB}$ shared memory. Higher arithmetic intensity ($8 \text{ FLOP/B}$), but 1024 threads/block limits the number of active blocks per SM to 1 or 2 depending on SM thread limits (max 1024 or 2048 threads/SM).
  - For advanced kernels, DevTech typically uses 2D thread tiling (e.g., $128 \times 128$ block tile with $8 \times 8$ register tile per thread, using 256 threads).

---

### Q3: What is Little's Law in GPU Computing, and how does it relate to latency hiding?
**Answer:**
$$\text{Concurrency Required} = \text{Throughput} \times \text{Latency}$$
To hide a Global Memory access latency of ~400-800 cycles, the GPU needs sufficient independent active warps (via high occupancy) OR sufficient independent instructions per warp (Instruction-Level Parallelism / ILP, e.g. unrolling loops and register tiling).

---

### Q4: How would you profile this GEMM code using NVIDIA Nsight Compute (`ncu`)?
**Answer:**
1. **Speed of Light Analysis:**
   ```bash
   ncu --section SpeedOfLight ./gemm
   ```
   Checks whether the kernel is memory-bound (`dram__throughput`) or compute-bound (`sm__throughput`).
2. **Memory Workload Analysis:**
   ```bash
   ncu --section MemoryWorkloadAnalysis ./gemm
   ```
   Reveals L1/Shared Memory hit rates, shared memory bank conflicts, and global memory coalescing efficiency.
3. **Warp State & Stall Reasons:**
   ```bash
   ncu --section WarpStateStats ./gemm
   ```
   Identifies whether warps are stalling on memory throttles, barrier synchronization (`sync`), or math pipeline dependencies.
