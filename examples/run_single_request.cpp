#include "llm/core/generation_config.h"
#include "llm/engine/engine.h"
#include "llm/model/mock_model.h"
#include "llm/tokenizer/simple_tokenizer.h"

#include <iostream>
#include <string>

int main() {
    llm::SimpleTokenizer tokenizer;

    llm::GenerationConfig config;
    config.maxNewTokens = 16;
    config.eosTokenId = tokenizer.eosTokenId();

    llm::MockModel model(tokenizer.vocabSize(), tokenizer.eosTokenId());

    llm::Engine engine(model, tokenizer);

    std::string prompt = "hello";

    std::cout << "Prompt: " << prompt << "\n";
    std::cout << "Generated tokens: ";

    llm::GenerateResult result = engine.generate(
        prompt,
        config,
        [&](llm::TokenId token) {
            std::cout << token << " ";
        }
    );

    std::cout << "\n";
    std::cout << "Generated text: " << result.text << "\n";
    std::cout << "Stopped by EOS: " << (result.stoppedByEos ? "yes" : "no") << "\n";

    return 0;
}
