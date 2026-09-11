NVCC ?= nvcc
ARCH ?= sm_86
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -Xcompiler=-Wall,-Wextra

all: nodals_hxt5b_fp64 nodals_hxt5b_fp32

nodals_hxt5b_fp64: hxt5b_main.cu hxt5b_boundary.cuh hxt5a_mixlen.cuh hybrid_amg.cuh hybrid_cf_overload.inc
	$(NVCC) $(NVCCFLAGS) -DNODALS_GPU_PRECISION_MODE=0 -I.. -arch=$(ARCH) hxt5b_main.cu -o $@

nodals_hxt5b_fp32: hxt5b_main.cu hxt5b_boundary.cuh hxt5a_mixlen.cuh hybrid_amg.cuh hybrid_cf_overload.inc
	$(NVCC) $(NVCCFLAGS) -DHXT4A_USE_FLOAT=1 -DNODALS_GPU_PRECISION_MODE=1 -I.. -arch=$(ARCH) hxt5b_main.cu -o $@

clean:
	rm -f nodals_hxt5b_fp64 nodals_hxt5b_fp32

.PHONY: all clean
