// NodalS tetrahedral SST: Stage 10 flat CUDA application entry point.
// Historical numerical foundation is retained, but never included through .cu main files.
// This entry is installed at the root of the generated solver source tree.

#include <cuda_runtime.h>
#include "stage6_mem_audit.cuh"
#define cudaMalloc(ptr, bytes) stage6_mem::alloc((void**)(ptr),(size_t)(bytes),__FILE__,__LINE__)
#define cudaFree(ptr) stage6_mem::free((void*)(ptr),__FILE__,__LINE__)
#define HXT4B_EMBED_MAIN hxt4b_embedded_main
#define HXT4A_EMBED_MAIN hxt4a_embedded_gate_main
#define main hxt2_embedded_gate_main
#include "modules/foundation/hxt2_foundation.inc"
#undef main
#define main hxt3a_embedded_gate_main
#include "modules/foundation/hxt3a_foundation.inc"
#undef main
#include "modules/foundation/hxt4a_foundation.inc"
#undef HXT4A_EMBED_MAIN
#include "modules/foundation/hxt4b_foundation.inc"
#include <cstdlib>
#undef HXT4B_EMBED_MAIN

#include "hybrid_amg.cuh"
#include "hxt5a_mixlen.cuh"
#include "hxt5b_boundary.cuh"
#include "modules/boundary/stage27_inlet_profile.cuh"
#include "hxt5e_fgmres.cuh"
#include "hxt5c_sst_g1.cuh"
#include "hxt5d_sst_g2.cuh"
#if defined(NODALS_P2_NATIVE_PILOT)
#include "modules/coupling/bernstein_p2_native_scalar_pilot.inc"
#endif
#include <sstream>
#include <chrono>

#include "modules/coupling/portable/coupling_core.hpp"
#include "modules/app/sst_application.inc"
