#pragma once

#include "llm/core/token.h"

#include <vector>
#include <string>

namespace llm {

struct GenerateResult {
    std::vector<TokenId> promptTokens;
    std::vector<TokenId> generatedTokens;

    std::string text;
    bool stoppedByEos = false;
};

} // namespace llm
