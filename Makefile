# ==============================================================================
# Makefile for CUDA GEMM Benchmark Suite (NVIDIA DevTech Preparation)
# ==============================================================================

NVCC        := nvcc
NVCC_FLAGS  := -O3 -std=c++14 -lineinfo --use_fast_math
ARCH_FLAGS  := -gencode arch=compute_75,code=sm_75 \
               -gencode arch=compute_80,code=sm_80 \
               -gencode arch=compute_86,code=sm_86 \
               -gencode arch=compute_89,code=sm_89
LIBS        := -lcublas

TARGET      := gemm
SRC         := gemm.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) $(SRC) -o $(TARGET) $(LIBS)

# Single target for Tesla T4 (Google Colab default: sm_75)
colab: $(SRC)
	$(NVCC) $(NVCC_FLAGS) -arch=sm_75 $(SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET) profile_gemm.*

.PHONY: all colab clean
