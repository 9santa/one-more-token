#pragma once

#include "llm/core/token.h"
#include "llm/core/sequence.h"

#include <string>

namespace llm {

// This is what the engine returns after each decoding step.
// A token event means: request X produced token Y
struct TokenEvent {
    RequstId requestId = 0;

    TokenId tokenId = -1;
    std::string text;

    bool hasToken = false;
    bool finished = false;

    FinishReason finishReason = FinishReason::None;
};


} // namespace llm
