NVCC ?= nvcc
ARCH ?= sm_86
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -Xcompiler=-Wall,-Wextra

all: nodals_hxt1_gpu

nodals_hxt1_gpu: hxt1_main.cu hxt1_mesh.hpp
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) hxt1_main.cu -o $@

clean:
	rm -f nodals_hxt1_gpu

.PHONY: all clean
