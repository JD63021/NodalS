NVCC ?= nvcc
ARCH ?= sm_86
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -Xcompiler=-Wall,-Wextra

all: nodals_hxt2_gpu

nodals_hxt2_gpu: hxt2_main.cu ../hybrid_hxt1/hxt1_mesh.hpp
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) hxt2_main.cu -o $@

clean:
	rm -f nodals_hxt2_gpu

.PHONY: all clean
