#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace llm {

inline void cudaCheck(cudaError_t result,
                      const char* expression,
                      const char* file,
                      int line) {
    if (result == cudaSuccess) {
        return;
    }

    std::string message;
    message += "CUDA error: ";
    message += cudaGetErrorString(result);
    message += "\nExpression: ";
    message += expression;
    message += "\nLocation: ";
    message += file;
    message += ":";
    message += std::to_string(line);

    throw std::runtime_error(message);
}


} // namespace llm

#define LLM_CUDA_CHECK(expr) \
    ::llm::cudaCheck((expr), #expr, __FILE__, __LINE__)
