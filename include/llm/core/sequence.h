#pragma once

#include "llm/core/token.h"
#include "llm/core/generation_config.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llm {

using RequstId = uint64_t;

enum class SequenceStatus {
    Waiting,
    Running,
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

    std::string prompt;

    std::vector<TokenId> promptsTokens;
    std::vector<TokenId> generatedTokens;

    GenerationConfig config;

    SequenceStatus status = SequenceStatus::Running;
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

    bool isRunning() const {
        return status == SequenceStatus::Running;
    }

    bool isFinished() const {
        return status == SequenceStatus::Finished ||
               status == SequenceStatus::Cancelled;
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
