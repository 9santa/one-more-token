#pragma once

#include "llm/core/token.h"
#include "llm/core/generation_config.h"
#include "llm/kv/kv_cache.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace llm {

using RequstId = uint64_t;

enum class SequenceStatus {
    Waiting,
    Prefill,
    Decode,
    Finished,
    Cancelled
};

enum class FinishReason {
    None,
    EosToken,
    MaxNewTokens,
    Cancelled
};

struct Sequence {
    RequstId id = 0;

    KVCacheHandle kvCache;

    std::string prompt;

    std::vector<TokenId> promptsTokens;
    std::vector<TokenId> generatedTokens;

    GenerationConfig config;

    SequenceStatus status = SequenceStatus::Waiting;
    FinishReason finishReason = FinishReason::None;

    int generatedLength() const {
        return static_cast<int>(generatedTokens.size());
    }

    int totalLength() const {
        return static_cast<int>(promptsTokens.size() + generatedTokens.size());
    }

    bool isWaiting() const {
        return status == SequenceStatus::Waiting;
    }

    bool isPrefill() const {
        return status == SequenceStatus::Prefill;
    }

    bool isDecode() const {
        return status == SequenceStatus::Decode;
    }

    bool isFinished() const {
        return status == SequenceStatus::Finished ||
               status == SequenceStatus::Cancelled;
    }

    TokenId lastToken() const {
        if (!generatedTokens.empty()) {
            return generatedTokens.back();
        }

        if (!promptsTokens.empty()) {
            return promptsTokens.back();
        }

        throw std::runtime_error("Sequence has no tokens");
    }

    // Right contextTokens() copies, which is bad. Will change this later.
    std::vector<TokenId> contextTokens() const {
        std::vector<TokenId> context;
        context.reserve(promptsTokens.size() + generatedTokens.size());

        context.insert(context.end(), promptsTokens.begin(), promptsTokens.end());
        context.insert(context.end(), generatedTokens.begin(), generatedTokens.end());

        return context;
    }
};


} // namespace llm
