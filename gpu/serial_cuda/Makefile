NVCC ?= nvcc
ARCH ?= sm_86
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -Xcompiler=-Wall,-Wextra

all: nodals_gpu_h4_fp32

nodals_gpu_h4_fp32: h4_main.cu h4_kernels.cuh h2b_fine_csr.cuh g5e_fp32_kernels.cuh g4_host.hpp g4_refresh_host.hpp g5_sa_host.hpp g3_host.hpp g2_host.hpp foam_mesh_g2.hpp cuda_runtime.hpp precision.hpp vector_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -DNODALS_GPU_PRECISION_MODE=1 h4_main.cu -o $@

clean:
	rm -f nodals_gpu_h4_fp32
