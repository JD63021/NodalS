# Message 1 multicore foundation: compact independent MPI build.
# Does NOT modify or rebuild the numerical GPU SST solver.
.DEFAULT_GOAL := all
MPI_CXX ?= mpicxx
AR ?= ar
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wno-unknown-pragmas
BUILD ?= build
GPU_ROOT ?= ../sst_gpu_v2/source/puretet_profile5_eq5_hxt5b
# Auto-generated local source binding, preserving the original GPU modules.
-include source.mk
MOD := $(GPU_ROOT)/modules
CPPFLAGS += -I$(MOD)/physics -I$(MOD)/wall/portable \
            -I$(MOD)/transport/portable -I$(MOD)/coupling/portable \
            -I$(MOD)/runtime

HOST_SOURCES := \
  $(MOD)/physics/sst2003m_host.cpp \
  $(MOD)/wall/portable/wall_host.cpp \
  $(MOD)/transport/portable/transport_host.cpp \
  $(MOD)/coupling/portable/coupling_host.cpp \
  $(MOD)/runtime/compiled_gate.cpp \
  $(MOD)/runtime/model_dispatch.cpp
HOST_OBJECTS := $(patsubst $(GPU_ROOT)/%.cpp,$(BUILD)/portable/%.o,$(HOST_SOURCES))
LOCAL_SOURCES := src/bridge/portable_probe.cpp src/app/mpi_probe.cpp
LOCAL_OBJECTS := $(patsubst %.cpp,$(BUILD)/%.o,$(LOCAL_SOURCES))

.PHONY: all clean print-config
all: $(BUILD)/nodals_sst_cpu_mc_probe
print-config:
	@echo 'MPI_CXX=$(MPI_CXX)'
	@echo 'GPU_ROOT=$(abspath $(GPU_ROOT))'
	@echo 'CXXFLAGS=$(CXXFLAGS)'

$(BUILD)/portable/%.o: $(GPU_ROOT)/%.cpp
	@mkdir -p $(dir $@)
	$(MPI_CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/src/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(MPI_CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/libnodals_sst_portable.a: $(HOST_OBJECTS)
	$(AR) rcs $@ $^

$(BUILD)/nodals_sst_cpu_mc_probe: $(LOCAL_OBJECTS) $(BUILD)/libnodals_sst_portable.a
	$(MPI_CXX) $(CXXFLAGS) -o $@ $(LOCAL_OBJECTS) $(BUILD)/libnodals_sst_portable.a
	@echo 'NODALS_CPU_SST_M1_BUILD status=PASS executable=$@ scope=MPI_PORTABLE_PROBE'

clean:
	rm -rf $(BUILD)

-include $(HOST_OBJECTS:.o=.d) $(LOCAL_OBJECTS:.o=.d)

# NODALS_CPU_SST_M2_TARGET_BEGIN -- separate operator-algebra executable.
# The live M1 probe and original CPU/GPU solvers are not modified.
M2_SOURCES := src/mesh/tet_mesh.cpp src/parallel/rcb.cpp \
  src/operators/p1bf3_p0.cpp src/app/mpi_operator_check.cpp
M2_OBJECTS := $(patsubst %.cpp,$(BUILD)/%.o,$(M2_SOURCES))
.PHONY: m2-operators
m2-operators: $(BUILD)/nodals_sst_cpu_mc_m2
$(BUILD)/nodals_sst_cpu_mc_m2: $(M2_OBJECTS)
	$(MPI_CXX) $(CXXFLAGS) -o $@ $^
	@echo 'NODALS_CPU_SST_M2_BUILD status=PASS executable=$@ scope=MPI_OPERATOR_CHECK_NOT_FLOW_SOLVER'
-include $(M2_OBJECTS:.o=.d)
# NODALS_CPU_SST_M2_TARGET_END
# NODALS_CPU_SST_M3_BUILD_ONLY_BEGIN
include Makefile.m3
# NODALS_CPU_SST_M3_BUILD_ONLY_END

# CPU_SST_MESSAGE4_INCLUDE (separate from M1-M3)
include Makefile.m4
