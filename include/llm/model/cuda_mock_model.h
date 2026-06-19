#pragma once

#include "llm/core/token.h"
#include "llm/model/model_backend.h"

#include <vector>

namespace llm {

class CudaMockModel : public ModelBackend {
private:
    size_t vocabSize_;
    TokenId eosTokenId_;

    std::vector<std::vector<float>> runKernel(
        const std::vector<TokenId>& lastTokens,
        const std::vector<int>& positions
    ) const;

public:
    CudaMockModel(size_t vocabSize, TokenId eosTokenId);

    size_t vocabSize() const override;

    std::vector<std::vector<float>> prefillBatch(
        const std::vector<std::vector<TokenId>>& batchPromptTOkens
    ) override;

    std::vector<std::vector<float>> decodeBatch(
        const std::vector<TokenId>& lastTokens,
        const std::vector<int>& positions
    ) override;
};


} // namespace llm
