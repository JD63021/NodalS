NVCC ?= nvcc
ARCH ?= sm_86
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -Xcompiler=-Wall,-Wextra
all: nodals_hxt4b_fp64 nodals_hxt4b_fp32
nodals_hxt4b_fp64: hxt4b_main.cu ../hybrid_hxt4a/hxt4a_main.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) hxt4b_main.cu -o $@
nodals_hxt4b_fp32: hxt4b_main.cu ../hybrid_hxt4a/hxt4a_main.cu
	$(NVCC) $(NVCCFLAGS) -DHXT4A_USE_FLOAT=1 -arch=$(ARCH) hxt4b_main.cu -o $@
clean:
	rm -f nodals_hxt4b_fp64 nodals_hxt4b_fp32
.PHONY: all clean
