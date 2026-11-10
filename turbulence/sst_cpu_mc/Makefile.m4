# Message 4: dedicated, standalone CPU SST source tree. M3 sources unmodified.
M4_SOURCES := $(wildcard m4/src/*/*.cpp)
M4_OBJECTS := $(patsubst %.cpp,$(BUILD)/%.o,$(M4_SOURCES))
CPPFLAGS += -I$(GPU_ROOT)/modules/initialization -I$(GPU_ROOT)/modules/wall/portable \
  -I$(GPU_ROOT)/modules/geometry -I$(GPU_ROOT)/modules/physics \
  -I$(GPU_ROOT)/modules/transport/portable
.PHONY: m4-sst
m4-sst: $(BUILD)/nodals_sst_cpu_mc_m4
$(BUILD)/m4/src/%.o: m4/src/%.cpp
	@mkdir -p $(dir $@)
	$(MPI_CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@
$(BUILD)/nodals_sst_cpu_mc_m4: $(M4_OBJECTS)
	$(MPI_CXX) $(CXXFLAGS) -o $@ $^
	@echo 'NODALS_CPU_SST_M4_BUILD status=PASS executable=$@ scope=REPEATED_SIMPLE_SST_CPU_M4_ISOLATED'
-include $(M4_OBJECTS:.o=.d)
