# NodalS SST CUDA/FP64: ordinary, incremental developer build.
# Numerical .cu/.inc sources are unmodified from the 2026-11-04 freeze.
# Host changes relink only; CUDA changes recompile the single unchanged CUDA TU.
.DEFAULT_GOAL := all
ROOT := source/puretet_profile5_eq5_hxt5b
BUILD ?= build
NVCC ?= /usr/local/cuda-12.2/bin/nvcc
CXX ?= g++
AR ?= ar
CUDA_ARCH ?= sm_86
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -Xcompiler=-Wall,-Wextra
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra
CUDAFLAGS := $(NVCCFLAGS) -DNODALS_GPU_PRECISION_MODE=0 -I source -arch=$(CUDA_ARCH)

HOST_SRC := $(ROOT)/modules/physics/sst2003m_host.cpp \
            $(ROOT)/modules/wall/portable/wall_host.cpp \
            $(ROOT)/modules/transport/portable/transport_host.cpp \
            $(ROOT)/modules/coupling/portable/coupling_host.cpp \
            $(ROOT)/modules/runtime/compiled_gate.cpp \
            $(ROOT)/modules/runtime/model_dispatch.cpp
HOST_OBJ := $(patsubst $(ROOT)/%.cpp,$(BUILD)/host/%.o,$(HOST_SRC))
CUDA_OBJ := $(BUILD)/nodals_sst_cuda.o
LIB := $(BUILD)/libnodals_sst_runtime.a
BIN := $(BUILD)/nodals_sst

.PHONY: all clean FORCE print-config help
all: $(BIN)
help:
	@echo 'Build: make -j4 [NVCC=/usr/local/cuda-12.2/bin/nvcc] [CUDA_ARCH=sm_86]'
	@echo 'Run:   python3 nodals.py run --case examples/ubend_200k.case --steps 10'
print-config:
	@echo 'CUDA=$(NVCC), CUDA_ARCH=$(CUDA_ARCH), CXX=$(CXX), BUILD=$(BUILD)'

# Change the stamp only when the actual toolchain / code generation flags change.
$(BUILD)/compile.options: FORCE
	@mkdir -p $(BUILD)
	@printf 'nvcc=%s\narch=%s\nflags=%s\n' '$(NVCC)' '$(CUDA_ARCH)' '$(CUDAFLAGS)' > $@.tmp
	@if ! cmp -s $@.tmp $@; then mv -f $@.tmp $@; else rm -f $@.tmp; fi
$(BUILD)/host.options: FORCE
	@mkdir -p $(BUILD)
	@printf 'cxx=%s\nflags=%s\n' '$(CXX)' '$(CXXFLAGS)' > $@.tmp
	@if ! cmp -s $@.tmp $@; then mv -f $@.tmp $@; else rm -f $@.tmp; fi

# No copying to a stage app and no install hash/manifest check.
$(CUDA_OBJ): $(ROOT)/nodals_sst_main.cu $(BUILD)/compile.options
	@mkdir -p $(dir $@)
	$(NVCC) $(CUDAFLAGS) -MMD -MF $(BUILD)/nodals_sst_cuda.d -MT $@ -c $< -o $@

$(BUILD)/host/%.o: $(ROOT)/%.cpp $(BUILD)/host.options
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MMD -MF $(@:.o=.d) -MT $@ -c $< -o $@

$(LIB): $(HOST_OBJ)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $(HOST_OBJ)

$(BIN): $(CUDA_OBJ) $(LIB)
	$(NVCC) $(CUDAFLAGS) $(CUDA_OBJ) $(LIB) -o $@
	@echo 'Built $(BIN) (no stage installation or hash registration)'

clean:
	rm -rf $(BUILD)
-include $(BUILD)/nodals_sst_cuda.d $(HOST_OBJ:.o=.d)
