#include "llm/core/generation_config.h"
#include "llm/core/sequence.h"
#include "llm/core/token_event.h"
#include "llm/engine/multi_request_engine.h"
#include "llm/model/mock_model.h"
#include "llm/tokenizer/simple_tokenizer.h"

#include <iostream>
#include <string>
#include <vector>

static std::string finishReasonToString(llm::FinishReason reason) {
    switch (reason) {
        case llm::FinishReason::None:
            return "none";
        case llm::FinishReason::EosToken:
            return "eos";
        case llm::FinishReason::MaxNewTokens:
            return "max_new_tokens";
        case llm::FinishReason::Cancelled:
            return "cancelled";
    }

    return "unknown";
}

int main() {
    llm::SimpleTokenizer tokenizer;

    llm::GenerationConfig config;
    config.maxNewTokens = 16;
    config.eosTokenId = tokenizer.eosTokenId();

    llm::MockModel model(tokenizer.vocabSize(), tokenizer.eosTokenId());

    llm::MultiRequstEngine engine(model, tokenizer);

    std::vector<llm::RequstId> ids;

    ids.push_back(engine.submit("hello", config));
    ids.push_back(engine.submit("mock model", config));
    ids.push_back(engine.submit("I am", config));

    int tick = 0;

    while (engine.hasActive()) {
        ++tick;

        std::cout << "\n=== tick " << tick << " ===\n";

        std::vector<llm::TokenEvent> events = engine.step();

        for (const llm::TokenEvent& event : events) {
            std::cout << "request " << event.requestId << ": ";

            if (event.hasToken) {
                std::cout << "token=" << event.tokenId
                          << " text=\"" << event.text << "\"";
            } else {
                std::cout << "no emitted token";
            }

            if (event.finished) {
                std::cout << " finished="
                          << finishReasonToString(event.finishReason);
            }

            std::cout << "\n";
        }
    }

    std::cout << "\n=== final results ===\n";

    for (llm::RequstId id : ids) {
        llm::GenerateResult result = engine.result(id);

        std::cout << "request " << id
                  << " output: \"" << result.text << "\""
                  << " stopped_by_eos="
                  << (result.stoppedByEos ? "yes" : "no")
                  << "\n";
    }

    return 0;
}
