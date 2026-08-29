.DEFAULT_GOAL := all

PETSC_DIR ?= $(HOME)/petsc-main
PETSC_ARCH ?= arch-cuda-debug

CXXFLAGS += -O3 -std=c++17 -Wall -Wextra

# CUDA 12.9+ removed the legacy NVTX v2 libnvToolsExt shared library.
# Preserve the working Gate9N link workaround from the frozen Makefile.
NODALS_PETSC_KSP_LIB = $(filter-out -lnvToolsExt,$(PETSC_KSP_LIB))

NODALS_FRAGMENTS := \
  src/00_core/preamble.inc \
  src/00_core/mesh_runtime.inc \
  src/10_fem/basis_quadrature.inc \
  src/20_boundary/problem_boundary.inc \
  src/30_momentum/custom_momentum.inc \
  src/35_stabilization/supg_dynamic_assembly.inc \
  src/40_pressure/pressure_assembly.inc \
  src/50_parallel/ownership_solve.inc \
  src/60_output/diagnostics_vtu.inc \
  src/70_simple/simple_support_experimental.inc \
  src/80_app/main_solver.inc

all: nodals_solver

app/nodals_main.o: app/nodals_main.cpp $(NODALS_FRAGMENTS)

nodals_solver: app/nodals_main.o
	@echo "NODALS_LINK target=$@ PETSC_ARCH=$(PETSC_ARCH) removed=-lnvToolsExt"
	${CXXLINKER} -o $@ $^ ${NODALS_PETSC_KSP_LIB}

# Same flags, compiled directly from the authoritative FULLFAST-FP64-RICH-SCALE
# oracle (SHA256 0de1a338...). Used for workstation parity regression.
reference/p1bf3_simple_foam_mpi.o: reference/p1bf3_simple_foam_mpi.cpp

nodals_reference_solver: reference/p1bf3_simple_foam_mpi.o
	@echo "NODALS_LINK target=$@ reference=frozen_gate9n PETSC_ARCH=$(PETSC_ARCH) removed=-lnvToolsExt"
	${CXXLINKER} -o $@ $^ ${NODALS_PETSC_KSP_LIB}

reference: nodals_reference_solver

clean::
	${RM} nodals_solver nodals_reference_solver app/nodals_main.o reference/p1bf3_simple_foam_mpi.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
