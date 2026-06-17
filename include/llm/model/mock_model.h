#pragma once

#include "llm/model/model_backend.h"

#include <stdexcept>
#include <vector>

namespace llm {

// This fake model produces deterministic logits.
// It is intentionally simple just so the engine has something model-shaped to call.
class MockModel : public ModelBackend {
private:
    size_t vocabSize_;
    TokenId eosTokenId_;

public:
    MockModel(size_t vocabSize, TokenId eosTokenId)
        : vocabSize_(vocabSize),
          eosTokenId_(eosTokenId) {
        if (vocabSize_ == 0) {
            throw std::runtime_error("MockModel vocab size must be positive");
        }
    }

    size_t vocabSize() const override {
        return vocabSize_;
    }

    std::vector<float> forwardNextToken(
        const std::vector<TokenId>& contextTokens
    ) override {
        if (contextTokens.empty()) {
            throw std::runtime_error("MockModel requires non-empty context");
        }

        std::vector<float> logits(vocabSize_, -10.0f);

        TokenId last = contextTokens.back();

        TokenId next = last + 1;

        if (next >= static_cast<TokenId>(vocabSize_)) {
            next = 3;
        }

        // Avoid generating special tokens too early.
        if (next < 3) {
            next = 3;
        }

        // End after the context becomes long enough.
        if (contextTokens.size() >= 12) {
            next = eosTokenId_;
        }

        logits[next] = 10.0f;

        return logits;
    }

    std::vector<std::vector<float>> forwardNextTokenBatch(
        const std::vector<std::vector<TokenId>>& batchContexts
    ) override {
        std::vector<std::vector<float>> batchLogits;
        batchLogits.reserve(batchContexts.size());

        for (const auto& context : batchContexts) {
            batchLogits.push_back(forwardNextToken(context));
        }

        return batchLogits;
    }
};


} // namespace llm
