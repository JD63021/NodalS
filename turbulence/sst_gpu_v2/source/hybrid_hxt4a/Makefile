NVCC ?= nvcc
ARCH ?= sm_86
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -Xcompiler=-Wall,-Wextra

all: nodals_hxt4a_fp64 nodals_hxt4a_fp32

nodals_hxt4a_fp64: hxt4a_main.cu ../hybrid_hxt2/hxt2_main.cu ../hybrid_hxt3a/hxt3a_main.cu ../hybrid_hxt1/hxt1_mesh.hpp
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) hxt4a_main.cu -o $@

nodals_hxt4a_fp32: hxt4a_main.cu ../hybrid_hxt2/hxt2_main.cu ../hybrid_hxt3a/hxt3a_main.cu ../hybrid_hxt1/hxt1_mesh.hpp
	$(NVCC) $(NVCCFLAGS) -DHXT4A_USE_FLOAT=1 -arch=$(ARCH) hxt4a_main.cu -o $@

clean:
	rm -f nodals_hxt4a_fp64 nodals_hxt4a_fp32

.PHONY: all clean
