.DEFAULT_GOAL := all

PETSC_DIR ?= $(HOME)/src/petsc
PETSC_ARCH ?= arch-linux-cuda-opt

CXXFLAGS += -O3 -std=c++17 -Wall -Wextra

# Match the root NodalS build: PETSc's included rules provide the C++ compile
# recipe.  This also avoids depending on a non-existent CXXCOMPILE variable.
NODALS_TURB_PETSC_KSP_LIB = $(filter-out -lnvToolsExt,$(PETSC_KSP_LIB))

NODALS_TURB_FRAGMENTS := \
  src/00_core/preamble_mesh.inc \
  src/10_problem/problem_config.inc \
  src/20_discretization/discrete_boundary_fem.inc \
  src/30_momentum/custom_momentum_core.inc \
  src/40_pressure/effective_pressure.inc \
  src/50_runtime/dynamic_runtime_plan.inc \
  src/55_turbulence/mixing_length_spalding.inc \
  src/60_momentum_dynamic/dynamic_operators.inc \
  src/70_pressure_assembly/pressure_assembly.inc \
  src/75_pressure_amg/custom_pressure_amg.inc \
  src/80_app/main_solver.inc

all: nodals_turbulence_cpu_fp64

# Deliberately no recipe here. PETSc conf/rules supplies the implicit .cpp -> .o
# compile command, exactly as in the main NodalS Makefile.
app/nodals_turbulence_main.o: app/nodals_turbulence_main.cpp $(NODALS_TURB_FRAGMENTS)

nodals_turbulence_cpu_fp64: app/nodals_turbulence_main.o
	@echo "NODALS_TURB_LINK target=$@ PETSC_ARCH=$(PETSC_ARCH) removed=-lnvToolsExt"
	${CXXLINKER} -o $@ $^ ${NODALS_TURB_PETSC_KSP_LIB}

clean::
	${RM} nodals_turbulence_cpu_fp64 app/nodals_turbulence_main.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
