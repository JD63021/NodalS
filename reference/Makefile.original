.DEFAULT_GOAL := all

PETSC_DIR ?= $(HOME)/petsc-main
PETSC_ARCH ?= arch-cuda-debug

CXXFLAGS += -O3 -std=c++17 -Wall -Wextra

# CUDA 12.9+ removed the legacy NVTX v2 libnvToolsExt shared library.
# Some older PETSc configurations retain -lnvToolsExt in PETSC_KSP_LIB even
# when libpetsc itself has no runtime dependency on it.  Filter only that
# stale application-link token; BUILD_M7A_FP32.sh performs a hard ldd check.
P1BF3_PETSC_KSP_LIB = $(filter-out -lnvToolsExt,$(PETSC_KSP_LIB))

all: p1bf3_simple_foam_mpi

p1bf3_simple_foam_mpi: p1bf3_simple_foam_mpi.o
	@echo "P1BF3_M7A_LINK_SANITIZE removed=-lnvToolsExt reason=legacy_NVTX_v2"
	${CXXLINKER} -o $@ $^ ${P1BF3_PETSC_KSP_LIB}

clean::
	${RM} p1bf3_simple_foam_mpi p1bf3_simple_foam_mpi.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
