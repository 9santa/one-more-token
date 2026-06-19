#include "llm/core/token.h"
#include "llm/model/cuda_mock_model.h"

#include "llm/cuda/cuda_check.h"
#include "llm/cuda/device_buffer.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdexcept>


namespace llm {

__device__ int chooseNextToken(int lastToken,
                               int currentLength,
                               int vocabSize,
                               int eosTokenId) {
    int next = lastToken + 1;

    if (next >= vocabSize) next = 3;

    if (next < 3) next = 3;

    if (currentLength >= 12) next = eosTokenId;

    return next;
}

__global__ void optimizedLogitsKernel(const int* lastTokens,
                                      const int* positions,
                                      float* logits,
                                      int batchSize,
                                      int vocabSize,
                                      int eosTokenId) {
    // 2D Grid Mapping: Y is the row (batch), X is the column (vocab item, token)
    int row = blockIdx.y;
    int vocab = blockIdx.x * blockDim.x + threadIdx.x;

    // Safety check: discard the whole block if the row is out of bounds
    if (row >= batchSize) return;

    // The Shared Board: Allocate fast memory physically on the GPU chip
    __shared__ int shared_next_token;

    // The Leader: ONLY the first thread in this block calculates the next token
    if (threadIdx.x == 0) {
        shared_next_token = chooseNextToken(
                lastTokens[row],
                positions[row],
                vocabSize,
                eosTokenId
        );
    }

    // Make all threads pause here until the Leader is done
    __syncthreads();

    // Safety check: ensure we don't write outside the matrix
    // This has to be AFTER __syncthreads()!!!
    if (vocab >= vocabSize) return;

    // All threads read the answer from the shared board
    int next = shared_next_token;

    // Map to flat 1D memory address
    int idx = row * vocabSize + vocab;
    logits[idx] = (vocab == next) ? 10.0f : -10.0f;
}

CudaMockModel::CudaMockModel(size_t vocabSize, TokenId eosTokenId)
    : vocabSize_(vocabSize),
      eosTokenId_(eosTokenId) {
    if (vocabSize_ == 0) {
        throw std::runtime_error("CudaMockModel vocab size must be positive");
    }

    if (vocabSize_ > static_cast<size_t>(INT_MAX)) {
        throw std::runtime_error("CudaMockModel vocab size too large (INT_MAX)");
    }
}

size_t CudaMockModel::vocabSize() const {
    return vocabSize_;
}

std::vector<std::vector<float>> CudaMockModel::runKernel(
    const std::vector<TokenId>& lastTokens,
    const std::vector<int>& positions
) const {
    if (lastTokens.size() != positions.size()) {
        throw std::runtime_error("CudaMockModel runKernel size mismatch");
    }

    const int batchSize = static_cast<int>(lastTokens.size());
    const int vocabSize = static_cast<int>(vocabSize_);

    if (batchSize == 0) return {};

    std::vector<int> hostLastTokens;
    hostLastTokens.reserve(lastTokens.size());

    for (TokenId token : lastTokens) {
        hostLastTokens.push_back(static_cast<int>(token));
    }

    const size_t flatLogitsSize =
        static_cast<size_t>(batchSize) * static_cast<size_t>(vocabSize_);

    DeviceBuffer<int> dLastTokens(hostLastTokens.size());
    DeviceBuffer<int> dPositions(positions.size());
    DeviceBuffer<float> dLogits(flatLogitsSize);

    dLastTokens.copyFromHost(hostLastTokens);
    dPositions.copyFromHost(positions);

    const int threads = 256;
    const int total = batchSize * vocabSize;
    const int blocks = (total + threads - 1) / threads;

    optimizedLogitsKernel<<<blocks, threads>>>(
        dLastTokens.data(),
        dPositions.data(),
        dLogits.data(),
        batchSize,
        vocabSize,
        static_cast<int>(eosTokenId_)
    );

    LLM_CUDA_CHECK(cudaGetLastError());
    LLM_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> flat = dLogits.copyToHost();

    std::vector<std::vector<float>> batchLogits;
    batchLogits.reserve(batchSize);

    for (size_t row = 0; row < batchSize; row++) {
        std::vector<float> logits(vocabSize);

        for (int v = 0; v < vocabSize; v++) {
            logits[v] = flat[row * vocabSize + v];
        }

        batchLogits.push_back(std::move(logits));
    }

    return batchLogits;
}

std::vector<std::vector<float>> CudaMockModel::prefillBatch(
        const std::vector<std::vector<TokenId>>& batchPromptTokens
) {
    std::vector<TokenId> lastTokens;
    std::vector<int> positions;
    lastTokens.reserve(batchPromptTokens.size());
    positions.reserve(batchPromptTokens.size());

    for (const auto& prompt : batchPromptTokens) {
        if (prompt.empty()) {
            throw std::runtime_error("CudaMockModel prefill received empty prompt");
        }

        lastTokens.push_back(prompt.back());
        positions.push_back(static_cast<int>(prompt.size()));
    }

    return runKernel(lastTokens, positions);
}

std::vector<std::vector<float>> CudaMockModel::decodeBatch(
    const std::vector<TokenId>& lastTokens,
    const std::vector<int>& positions
) {
    return runKernel(lastTokens, positions);
}


} // namespace llm
