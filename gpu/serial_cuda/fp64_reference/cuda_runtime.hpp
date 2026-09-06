#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstddef>
#include <utility>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace nodals_gpu {

inline void cuda_check(cudaError_t e, const char* expr, const char* file, int line) {
  if (e != cudaSuccess) {
    char buf[1024];
    std::snprintf(buf, sizeof(buf), "CUDA failure %s at %s:%d: %s", expr, file, line, cudaGetErrorString(e));
    throw std::runtime_error(buf);
  }
}
#define NODALS_CUDA(call) ::nodals_gpu::cuda_check((call), #call, __FILE__, __LINE__)

struct DeviceMemoryInfo {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
};

inline DeviceMemoryInfo device_memory_info() {
  DeviceMemoryInfo m;
  NODALS_CUDA(cudaMemGetInfo(&m.free_bytes, &m.total_bytes));
  return m;
}

inline void print_device_memory(const char* label, std::size_t explicit_bytes = 0) {
  const auto m = device_memory_info();
  const double mib = 1024.0 * 1024.0;
  std::printf("NODALS_GPU_MEMORY label=%s freeMiB=%.3f usedMiB=%.3f totalMiB=%.3f explicitMiB=%.3f\n",
              label, m.free_bytes / mib, (m.total_bytes - m.free_bytes) / mib,
              m.total_bytes / mib, explicit_bytes / mib);
}

template<class T>
class DeviceBuffer {
public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t n) { allocate(n); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept { swap(other); }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) { reset(); swap(other); }
    return *this;
  }
  ~DeviceBuffer() { reset(); }

  void allocate(std::size_t n) {
    reset();
    n_ = n;
    if (n_) NODALS_CUDA(cudaMalloc(reinterpret_cast<void**>(&p_), n_ * sizeof(T)));
  }
  void reset() noexcept {
    if (p_) cudaFree(p_);
    p_ = nullptr; n_ = 0;
  }
  T* data() { return p_; }
  const T* data() const { return p_; }
  std::size_t size() const { return n_; }
  std::size_t bytes() const { return n_ * sizeof(T); }

  void upload(const T* h, std::size_t n) {
    if (n != n_) throw std::runtime_error("DeviceBuffer upload size mismatch");
    if (n) NODALS_CUDA(cudaMemcpy(p_, h, bytes(), cudaMemcpyHostToDevice));
  }
  void download(T* h, std::size_t n) const {
    if (n != n_) throw std::runtime_error("DeviceBuffer download size mismatch");
    if (n) NODALS_CUDA(cudaMemcpy(h, p_, bytes(), cudaMemcpyDeviceToHost));
  }
private:
  void swap(DeviceBuffer& o) noexcept { std::swap(p_, o.p_); std::swap(n_, o.n_); }
  T* p_ = nullptr;
  std::size_t n_ = 0;
};

class CudaEventTimer {
public:
  CudaEventTimer() { NODALS_CUDA(cudaEventCreate(&a_)); NODALS_CUDA(cudaEventCreate(&b_)); }
  ~CudaEventTimer() { cudaEventDestroy(a_); cudaEventDestroy(b_); }
  void start(cudaStream_t s = 0) { NODALS_CUDA(cudaEventRecord(a_, s)); }
  float stop(cudaStream_t s = 0) {
    NODALS_CUDA(cudaEventRecord(b_, s));
    NODALS_CUDA(cudaEventSynchronize(b_));
    float ms = 0.0f; NODALS_CUDA(cudaEventElapsedTime(&ms, a_, b_)); return ms;
  }
private:
  cudaEvent_t a_{}, b_{};
};

} // namespace nodals_gpu
