#include "llm/cuda/vector_add.h"

#include "llm/cuda/cuda_check.h"
#include "llm/cuda/device_buffer.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <vector>

namespace llm {

__global__ void vectorAddKernel(const float* a,
                                const float* b,
                                float* out,
                                int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

std::vector<float> cudaVectorAdd(const std::vector<float>& a,
                                 const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("cudaVectorAdd vector sizes mismatch");
    }

    const int n = static_cast<int>(a.size());

    DeviceBuffer<float> dA(a.size());
    DeviceBuffer<float> dB(b.size());
    DeviceBuffer<float> dOut(a.size());

    dA.copyFromHost(a);
    dB.copyFromHost(b);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    vectorAddKernel<<<blocks, threads>>>(dA.data(), dB.data(), dOut.data(), n);

    LLM_CUDA_CHECK(cudaGetLastError());
    LLM_CUDA_CHECK(cudaDeviceSynchronize());

    return dOut.copyToHost();
}


} // namespace llm
