#pragma once

#include "llm/core/token.h"
#include "llm/core/sequence.h"

#include <stdexcept>
#include <vector>

namespace llm {

struct DecodeBatch {
    std::vector<RequstId> requestIds;
    // This is simple ragged contexts as 2D vector
    // Later need to replace with packed tensors + metadata
    std::vector<std::vector<TokenId>> contexts;

    void validate() const {
        if (requestIds.size() != contexts.size()) {
            throw std::runtime_error("DecodeBatch requestIds/context size mismatch");
        }

        for (const auto& ctx : contexts) {
            if (ctx.empty()) {
                throw std::runtime_error("DecodeBatch contains empty context");
            }
        }
    }

    size_t size() const {
        return requestIds.size();
    }

    bool empty() const {
        return requestIds.empty();
    }
};


} // namespace llm
