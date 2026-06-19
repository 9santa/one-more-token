#pragma once

#include "llm/cuda/cuda_check.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <driver_types.h>
#include <stdexcept>
#include <vector>

namespace llm {

// RAII container/smart pointer for GPU memory
template<typename T>
class DeviceBuffer {
private:
    T* ptr_ = nullptr;
    size_t size_ = 0;

public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(size_t size) {
        allocate(size);
    }

    ~DeviceBuffer() {
        reset();
    }

    // Delete copy constuctor and assignment operator
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_),
          size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this == &other) return *this;

        reset();

        ptr_ = other.ptr_;
        size_ = other.size_;

        other.ptr_ = nullptr;
        other.size_ = 0;

        return *this;
    }

    void allocate(size_t size) {
        reset();

        if (size == 0) return;

        size_ = size;
        LLM_CUDA_CHECK(cudaMalloc(&ptr_, size_ * sizeof(T)));
    }

    void reset() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    T* data() {
        return ptr_;
    }

    const T* data() const {
        return ptr_;
    }

    size_t size() const {
        return size_;
    }

    // From CPU to GPU
    void copyFromHost(const std::vector<T>& host) {
        if (host.size() != size_) {
            throw std::runtime_error("DeviceBuffer copyFromHost size mismatch");
        }

        if (size_ == 0) return;

        LLM_CUDA_CHECK(cudaMemcpy(
            ptr_,
            host.data(),
            size_ * sizeof(T),
            cudaMemcpyHostToDevice
        ));
    }

    // From GPU to CPU
    std::vector<T> copyToHost() const {
        std::vector<T> host(size_);

        if (size_ == 0) return host;

        LLM_CUDA_CHECK(cudaMemcpy(
            host.data(),
            ptr_,
            size_ * sizeof(T),
            cudaMemcpyDeviceToHost
        ));

        return host;
    }
};


} // namespace llm
