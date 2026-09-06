#pragma once
#include <cstdint>
#include <type_traits>

namespace nodals_gpu {

enum class PrecisionMode : int { FP64 = 0, FP32 = 1, AMG_FP32 = 2 };

#ifndef NODALS_GPU_PRECISION_MODE
#define NODALS_GPU_PRECISION_MODE 0
#endif

#if NODALS_GPU_PRECISION_MODE == 0
using StateReal = double;
using OperatorReal = double;
using AMGReal = double;
constexpr PrecisionMode kPrecisionMode = PrecisionMode::FP64;
constexpr const char* kPrecisionName = "fp64";
#elif NODALS_GPU_PRECISION_MODE == 1
using StateReal = float;
using OperatorReal = float;
using AMGReal = float;
constexpr PrecisionMode kPrecisionMode = PrecisionMode::FP32;
constexpr const char* kPrecisionName = "fp32";
#elif NODALS_GPU_PRECISION_MODE == 2
using StateReal = double;
using OperatorReal = double;
using AMGReal = float;
constexpr PrecisionMode kPrecisionMode = PrecisionMode::AMG_FP32;
constexpr const char* kPrecisionName = "amg_fp32";
#else
#error "Unsupported NODALS_GPU_PRECISION_MODE"
#endif

// Reductions and convergence decisions intentionally remain FP64 in all modes.
using AccumReal = double;
using Index32 = std::int32_t;
using Offset64 = std::int64_t;

static_assert(sizeof(AccumReal) == 8, "NodalS GPU reductions must remain FP64");
static_assert(sizeof(Index32) == 4, "Index32 contract");
static_assert(sizeof(Offset64) == 8, "Offset64 contract");

} // namespace nodals_gpu
