# M3 recovery: full matched-geometry momentum/pressure solver (one Picard correction).
# Uses unchanged GPU native header mathematics through GPU_ROOT.
M3_SOURCES := src/mesh/tet_mesh.cpp src/parallel/rcb.cpp src/operators/p1bf3_p0.cpp \
 src/case/case_config.cpp src/wall/bf3_flux.cpp src/wall/wall_sampler.cpp \
 src/pressure/constrained_schur.cpp src/pressure/pressure_amg.cpp \
 src/initialization/potential_flow.cpp src/momentum/m3_predictor.cpp \
 src/app/mpi_picard.cpp
M3_OBJECTS := $(patsubst %.cpp,$(BUILD)/%.o,$(M3_SOURCES))
CPPFLAGS += -I$(GPU_ROOT)/modules/initialization
.PHONY: m3-picard
m3-picard: $(BUILD)/nodals_sst_cpu_mc_m3
$(BUILD)/nodals_sst_cpu_mc_m3: $(M3_OBJECTS)
	$(MPI_CXX) $(CXXFLAGS) -o $@ $^
	@echo 'NODALS_CPU_SST_M3_BUILD status=PASS executable=$@ scope=ONE_REAL_PICARD_NOT_FULL_SST'
-include $(M3_OBJECTS:.o=.d)
