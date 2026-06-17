#pragma once

#include "llm/core/token.h"

namespace llm {

struct GenerationConfig {
    int maxNewTokens = 32;

    TokenId eosTokenId = 2;

    // Will be used later
    double temperature = 1.0;
};

} // namespace llm
