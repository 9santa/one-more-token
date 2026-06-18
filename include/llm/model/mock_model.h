#pragma once

#include "llm/core/token.h"
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

    std::vector<float> makeLogits(TokenId lastToken,
                                  int currentLength) const {
        std::vector<float> logits(vocabSize_, -10.0f);

        TokenId next = lastToken + 1;

        if (next >= static_cast<TokenId>(vocabSize_)) {
            next = 3;
        }

        if (next < 3) next = 3;

        // Stop after sequence becomes too long
        // currentLength means: number of tokens already in the sequence
        // before producing the next one
        if (currentLength >= 12) next = eosTokenId_;

        logits[next] = 10.0f;
        return logits;
    }

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

    // prefill: sees full prompt
    std::vector<std::vector<float>> prefillBatch(
        const std::vector<std::vector<TokenId>>& batchPromptTokens
    ) override {
        std::vector<std::vector<float>> batchLogits;
        batchLogits.reserve(batchPromptTokens.size());

        for (const auto& prompt : batchPromptTokens) {
            if (prompt.empty()) {
                throw std::runtime_error("MockModel prefill received empty prompt");
            }

            TokenId last = prompt.back();
            int currentLength = static_cast<int>(prompt.size());

            batchLogits.push_back(makeLogits(last, currentLength));
        }

        return batchLogits;
    }

    // decode: sees only last token + position
    std::vector<std::vector<float>> decodeBatch(
        const std::vector<TokenId>& lastTokens,
        const std::vector<int>& positions
    ) override {
        if (lastTokens.size() != positions.size()) {
            throw std::runtime_error("MockModel decode size mismatch");
        }

        std::vector<std::vector<float>> batchLogits;
        batchLogits.reserve(lastTokens.size());

        for (size_t i = 0; i < lastTokens.size(); i++) {
            batchLogits.push_back(makeLogits(lastTokens[i], positions[i]));
        }

        return batchLogits;
    }
};


} // namespace llm
