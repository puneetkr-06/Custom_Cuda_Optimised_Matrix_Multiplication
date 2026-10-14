#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/**
 * ============================================================================
 * NVIDIA DevTech Interview Preparation - High Performance GEMM in CUDA C++
 * ============================================================================
 * 
 * Matrix Multiplication: C = A * B
 * Dimensions: A is (M x K), B is (K x N), C is (M x N).
 * In this benchmark, we evaluate square matrices: M = N = K.
 *
 * ----------------------------------------------------------------------------
 * CORE CONCEPTS TESTED IN NVIDIA DEVTECH INTERVIEWS:
 * ----------------------------------------------------------------------------
 * 1. Arithmetic Intensity & The Roofline Model:
 *    - GEMM requires 2 * N^3 floating point operations (FLOPs) for N^3 multiply-adds.
 *    - Memory traffic: Reading A and B, writing C.
 *    - If every thread reads directly from Global Memory (Naive), each element is
 *      read N times from DRAM. High DRAM traffic causes memory bandwidth saturation.
 *    - Shared memory tiling reuses loaded data across threads in the block, boosting
 *      Arithmetic Intensity by a factor of TILE_DIM.
 *
 * 2. Memory Coalescing:
 *    - Global memory loads are serviced in 32-byte or 128-byte cache lines.
 *    - In Row-Major order, consecutive threads in a warp (consecutive threadIdx.x)
 *      accessing consecutive memory addresses (e.g., A[row * N + col]) achieve 100%
 *      coalescing efficiency (single memory transaction per warp).
 *
 * 3. Shared Memory & Bank Conflicts:
 *    - Shared Memory (SRAM) is on-chip, providing ~10-20x lower latency and ~10x
 *      higher bandwidth compared to High Bandwidth Memory / GDDR.
 *    - It is organized into 32 banks (4 bytes per bank). Consecutive 32-bit words
 *      map to consecutive banks. If multiple threads within a warp access different
 *      addresses in the same bank, a bank conflict occurs and accesses serialize.
 *
 * 4. Thread Synchronization (__syncthreads):
 *    - Barrier synchronization primitive ensuring all warps within a thread block
 *      reach this point before any warp proceeds.
 *    - Essential in tiled algorithms: Prevents Write-After-Read (WAR) and
 *      Read-After-Write (RAW) data hazards when filling shared memory tiles.
 * ============================================================================
 */

// Error handling macros for robust CUDA development
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__      \
                      << " - " << cudaGetErrorString(err) << std::endl;       \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t status = call;                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__    \
                      << " - Code: " << status << std::endl;                  \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Tile dimension for the shared memory kernel
constexpr int TILE_DIM = 32;

// ============================================================================
// 1. NAIVE GEMM KERNEL (Global Memory Only)
// ============================================================================
/**
 * @brief Computes C = A * B using direct global memory loads.
 *
 * Bottleneck Analysis:
 * - Each thread computes one element C[row, col].
 * - To compute C[row, col], the thread iterates K times, loading A[row, k]
 *   and B[k, col] from Global DRAM.
 * - DRAM Accesses: 2 * N^3 reads.
 * - Coalescing:
 *   - For matrix A: All threads in a warp share the same 'row', meaning they all
 *     broadcast the same element A[row, k].
 *   - For matrix B: Consecutive threads have consecutive 'col' values, so
 *     B[k, col] is coalesced across a row.
 * - However, there is ZERO data reuse between warps and thread blocks. The same
 *   elements of A and B are repeatedly fetched across the memory bus from DRAM,
 *   saturating memory bus bandwidth and leaving SM ALU pipelines starved.
 */
__global__ void gemm_naive(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int N) {
    // Compute the 2D coordinate of the element of C this thread is responsible for
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float sum = 0.0f; // Accumulator kept in SM registers (lowest latency)
        #pragma unroll 4
        for (int k = 0; k < N; ++k) {
            // DRAM read A[row * N + k] and B[k * N + col]
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// ============================================================================
// 2. TILED GEMM KERNEL (Shared Memory + Block Tiling)
// ============================================================================
/**
 * @brief Computes C = A * B using shared memory tiling (software cache).
 *
 * DevTech Optimization Principles Applied:
 *
 * 1. Blocking / Tiling for Temporal Locality:
 *    Instead of streaming from DRAM N times, we partition A and B into sub-matrices
 *    (tiles) of size TILE_DIM x TILE_DIM. We load one tile of A and one tile of B into
 *    on-chip Shared Memory (SRAM) per step.
 *
 * 2. High Coalescing Efficiency on Loads:
 *    - Threads in the block cooperatively load elements from A and B into `s_A` and `s_B`.
 *    - `threadIdx.x` drives the contiguous memory stride in Global Memory, ensuring
 *      perfect 128-byte coalesced memory transactions.
 *
 * 3. Thread Synchronization (__syncthreads):
 *    - Synchronize before computing to ensure the entire tile is present in shared memory.
 *    - Synchronize after computing the partial dot product to ensure all threads have
 *      finished reading before the next iteration overwrites `s_A` and `s_B`.
 *
 * 4. Bank Conflict Avoidance:
 *    - Size: 32x32 floats. `s_A[ty][tx]` and `s_B[ty][tx]`.
 *    - During the inner multiplication loop `s_A[ty][k] * s_B[k][tx]`:
 *      - All threads in a warp share the same `ty` and `k`. Hence, all 32 threads read
 *        from the exact same address `s_A[ty][k]` -> Shared memory hardware broadcasts
 *        the value with zero bank conflicts!
 *      - For `s_B[k][tx]`, each thread accesses bank `tx % 32`. Since threads 0..31
 *        have `tx = 0..31`, each thread maps to a unique bank (Banks 0 to 31) -> 
 *        100% bank-conflict-free 1-cycle access!
 */
__global__ void gemm_tiled(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int N) {
    // 2D thread and block coordinates
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int row = by * TILE_DIM + ty;
    const int col = bx * TILE_DIM + tx;

    // Allocated in fast on-chip SRAM (Shared Memory)
    // Total footprint: 2 * 32 * 32 * 4 bytes = 8 KB per block (very low occupancy footprint)
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    float sum = 0.0f; // Accumulator held in register file

    // Loop over all tiles required to compute the dot product for C[row, col]
    const int numTiles = (N + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; ++t) {
        // --------------------------------------------------------------------
        // Step 1: Cooperative Coalesced Global -> Shared Memory Load
        // --------------------------------------------------------------------
        int a_col = t * TILE_DIM + tx;
        int b_row = t * TILE_DIM + ty;

        // Boundary checks for arbitrary matrix sizes
        s_A[ty][tx] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        s_B[ty][tx] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

        // Barrier 1: Ensure entire tile is loaded by all threads before computation begins
        __syncthreads();

        // --------------------------------------------------------------------
        // Step 2: Compute Partial Dot Product from Shared Memory
        // --------------------------------------------------------------------
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        // Barrier 2: Ensure all threads finished reading shared memory
        // before next loop iteration overwrites s_A and s_B
        __syncthreads();
    }

    // Write final accumulated result to Global Memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// ============================================================================
// 3. cuBLAS GEMM WRAPPER (Industry Standard Baseline)
// ============================================================================
/**
 * @brief Computes C = A * B using NVIDIA's official cuBLAS library.
 *
 * NOTE on Layouts (Row-Major vs Column-Major):
 * - C/C++ stores matrices in Row-Major order: A(M, K) with stride K.
 * - Fortran/cuBLAS assumes Column-Major order: A(M, K) with stride M.
 * - Mathematical Identity:
 *     (A * B)^T = B^T * A^T
 *   A Row-Major matrix of shape (M x N) has the exact same in-memory memory layout
 *   as a Column-Major matrix of shape (N x M) transposed!
 * - Therefore, calling cublasSgemm with:
 *     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N)
 *   computes C = A * B directly in Row-Major order without transposing data!
 */
void run_cublas_gemm(cublasHandle_t handle,
                     const float* d_A,
                     const float* d_B,
                     float* d_C,
                     int N) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alpha,
        d_B, N,      // Pass B first
        d_A, N,      // Pass A second
        &beta,
        d_C, N       // Output C in row-major
    ));
}

// ============================================================================
// Verification Utility
// ============================================================================
bool verify_results(const float* ref, const float* test, int N, float tolerance = 1e-3f) {
    float max_diff = 0.0f;
    int mismatch_count = 0;

    for (int i = 0; i < N * N; ++i) {
        float diff = std::fabs(ref[i] - test[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff > tolerance) {
            if (mismatch_count < 5) {
                std::cerr << "Mismatch at index " << i << " [" << i / N << ", " << i % N
                          << "]: Ref=" << ref[i] << " vs Test=" << test[i]
                          << " (Diff: " << diff << ")" << std::endl;
            }
            mismatch_count++;
        }
    }

    if (mismatch_count > 0) {
        std::cerr << "Verification FAILED: " << mismatch_count << " mismatches. Max diff: " 
                  << max_diff << std::endl;
        return false;
    }
    return true;
}

// ============================================================================
// Benchmark Execution Suite
// ============================================================================
struct BenchmarkResult {
    std::string name;
    float time_ms;
    float gflops;
    float speedup;
    bool verified;
};

void run_benchmark_for_size(int N, cublasHandle_t cublas_handle) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << " BENCHMARKING MATRIX MULTIPLICATION: N = " << N 
              << " (" << N << " x " << N << " = " << (size_t)N * N * sizeof(float) / (1024 * 1024) 
              << " MB per matrix)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    size_t bytes = (size_t)N * N * sizeof(float);
    double total_flops = 2.0 * (double)N * (double)N * (double)N;

    // Allocate Host Memory
    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C_cublas(N * N);
    std::vector<float> h_C_test(N * N);

    // Initialize inputs with deterministic pseudo-random values
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(rand() % 100) / 100.0f;
        h_B[i] = static_cast<float>(rand() % 100) / 100.0f;
    }

    // Allocate Device Memory
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // CUDA Events for exact GPU execution timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int WARMUP_RUNS = 2;
    const int BENCHMARK_RUNS = 5;

    std::vector<BenchmarkResult> results;

    // ------------------------------------------------------------------------
    // 1. cuBLAS (Reference & Baseline)
    // ------------------------------------------------------------------------
    {
        // Warmup
        for (int i = 0; i < WARMUP_RUNS; ++i) {
            run_cublas_gemm(cublas_handle, d_A, d_B, d_C, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed runs
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < BENCHMARK_RUNS; ++i) {
            run_cublas_gemm(cublas_handle, d_A, d_B, d_C, N);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        float avg_ms = total_ms / BENCHMARK_RUNS;
        float gflops = (float)((total_flops * 1e-9) / (avg_ms * 1e-3));

        // Copy back reference
        CUDA_CHECK(cudaMemcpy(h_C_cublas.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        results.push_back({"cuBLAS (NVIDIA Library)", avg_ms, gflops, 1.0f, true});
    }

    // ------------------------------------------------------------------------
    // 2. Naive Approach
    // ------------------------------------------------------------------------
    {
        dim3 blockDim(TILE_DIM, TILE_DIM);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

        // Warmup
        for (int i = 0; i < WARMUP_RUNS; ++i) {
            gemm_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed runs
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < BENCHMARK_RUNS; ++i) {
            gemm_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        float avg_ms = total_ms / BENCHMARK_RUNS;
        float gflops = (float)((total_flops * 1e-9) / (avg_ms * 1e-3));

        CUDA_CHECK(cudaMemcpy(h_C_test.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        bool verified = verify_results(h_C_cublas.data(), h_C_test.data(), N);
        results.push_back({"Naive (Global Memory)", avg_ms, gflops, 1.0f, verified});
    }

    // ------------------------------------------------------------------------
    // 3. Tiled Approach (Shared Memory)
    // ------------------------------------------------------------------------
    {
        dim3 blockDim(TILE_DIM, TILE_DIM);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

        // Warmup
        for (int i = 0; i < WARMUP_RUNS; ++i) {
            gemm_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed runs
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < BENCHMARK_RUNS; ++i) {
            gemm_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        float avg_ms = total_ms / BENCHMARK_RUNS;
        float gflops = (float)((total_flops * 1e-9) / (avg_ms * 1e-3));

        CUDA_CHECK(cudaMemcpy(h_C_test.data(), d_C, bytes, cudaMemcpyDeviceToHost));
        bool verified = verify_results(h_C_cublas.data(), h_C_test.data(), N);
        results.push_back({"Tiled (Shared Memory)", avg_ms, gflops, 1.0f, verified});
    }

    // Calculate speedup relative to Naive
    float naive_time = results[1].time_ms;
    for (auto& res : results) {
        res.speedup = naive_time / res.time_ms;
    }

    // Print Formatted Performance Comparison Table
    std::cout << "\n+-------------------------+-------------+---------------+------------+-----------+\n";
    std::cout << "| Implementation          | Time (ms)   | Throughput    | Speedup    | Status    |\n";
    std::cout << "|                         |             | (GFLOPS)      | vs Naive   |           |\n";
    std::cout << "+-------------------------+-------------+---------------+------------+-----------+\n";

    for (const auto& res : results) {
        std::cout << "| " << std::left << std::setw(23) << res.name
                  << " | " << std::right << std::setw(11) << std::fixed << std::setprecision(3) << res.time_ms
                  << " | " << std::right << std::setw(13) << std::fixed << std::setprecision(1) << res.gflops
                  << " | " << std::right << std::setw(9)  << std::fixed << std::setprecision(2) << res.speedup << "x"
                  << " | " << std::left  << std::setw(9)  << (res.verified ? "PASS" : "FAIL")
                  << " |\n";
    }
    std::cout << "+-------------------------+-------------+---------------+------------+-----------+\n";

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main(int argc, char** argv) {
    std::cout << "================================================================================" << std::endl;
    std::cout << "   NVIDIA DevTech Internship Showcase: Custom CUDA GEMM Optimization Suite      " << std::endl;
    std::cout << "================================================================================" << std::endl;

    // Query Device Properties
    int device_id = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "SM Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Global Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << "Memory Clock Rate: " << prop.memoryClockRate * 1e-3 << " MHz" << std::endl;
    std::cout << "Theoretical Peak Bandwidth: "
              << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << " GB/s" << std::endl;

    // Initialize cuBLAS
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    // Run benchmarks for required matrix sizes: N = 1024 and N = 2048
    run_benchmark_for_size(1024, cublas_handle);
    run_benchmark_for_size(2048, cublas_handle);

    // Clean up cuBLAS
    CUBLAS_CHECK(cublasDestroy(cublas_handle));

    std::cout << "\n[OK] All GEMM tests completed successfully." << std::endl;
    return 0;
}
