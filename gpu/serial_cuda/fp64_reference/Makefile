NVCC ?= nvcc
ARCH ?= sm_86
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -Xcompiler=-Wall,-Wextra
all: nodals_gpu_g5d
nodals_gpu_g5d: g4_main.cu g4_kernels.cuh g4_refresh_host.hpp g5_sa_host.hpp g4_host.hpp g3_host.hpp g2_host.hpp foam_mesh_g2.hpp cuda_runtime.hpp precision.hpp vector_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) -DNODALS_GPU_PRECISION_MODE=0 g4_main.cu -o $@
clean:
	rm -f nodals_gpu_g5d
