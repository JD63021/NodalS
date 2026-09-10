NVCC ?= nvcc
ARCH ?= sm_86
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -Xcompiler=-Wall,-Wextra -Xcompiler=-pthread

H8_DEPS = h8_main.cu h8_precomputed_b.cuh h7_momentum_assembly.cuh \
          h6_coarse_spmv.cuh h4_kernels.cuh h2b_fine_csr.cuh \
          g5e_fp32_kernels.cuh g4_host.hpp g4_refresh_host.hpp \
          g5_sa_host.hpp g3_host.hpp g2_host.hpp foam_mesh_g2.hpp \
          cuda_runtime.hpp precision.hpp vector_kernels.cuh

all: nodals_gpu_h8_fp32
fp32: nodals_gpu_h8_fp32
fp64: nodals_gpu_h8_fp64

nodals_gpu_h8_fp32: $(H8_DEPS)
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -DNODALS_GPU_PRECISION_MODE=1 h8_main.cu -o $@

nodals_gpu_h8_fp64: $(H8_DEPS)
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -DNODALS_GPU_PRECISION_MODE=0 h8_main.cu -o $@

clean:
	rm -f nodals_gpu_h8_fp32 nodals_gpu_h8_fp64

.PHONY: all fp32 fp64 clean
