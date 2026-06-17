#pragma once

#include "llm/core/token.h"
#include "llm/core/generate_result.h"
#include "llm/core/generation_config.h"
#include "llm/model/model_backend.h"
#include "llm/sampling/sampler.h"
#include "llm/tokenizer/simple_tokenizer.h"

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace llm {

class Engine {
private:
    ModelBackend& model_;
    SimpleTokenizer& tokenizer_;
    Sampler sampler_;

public:
    Engine(ModelBackend& model,
           SimpleTokenizer& tokenizer)
        : model_(model),
          tokenizer_(tokenizer) {}

    GenerateResult generate(const std::string& prompt,
                            const GenerationConfig& config,
                            const std::function<void(TokenId)>& onToken = nullptr) {
        if (config.maxNewTokens <= 0) {
            throw std::runtime_error("maxNewTokens must be positive");
        }

        std::vector<TokenId> promptTokens = tokenizer_.encode(prompt);

        std::vector<TokenId> context = promptTokens;
        std::vector<TokenId> generated;

        bool stoppedByEos = false;

        for (int step = 0; step < config.maxNewTokens; step++) {
            std::vector<float> logits = model_.forwardNextToken(context);

            if (logits.size() != model_.vocabSize()) {
                throw std::runtime_error("Model returned logits with wrong vocab size");
            }

            TokenId next = sampler_.sample(logits, config);

            if (next == config.eosTokenId) {
                stoppedByEos = true;
                break;
            }

            generated.push_back(next);
            context.push_back(next);

            if (onToken) {
                onToken(next);
            }
        }

        GenerateResult result;
        result.promptTokens = promptTokens;
        result.generatedTokens = generated;
        result.text = tokenizer_.decode(generated);
        result.stoppedByEos = stoppedByEos;

        return result;
    }
};


} // namespace llm
