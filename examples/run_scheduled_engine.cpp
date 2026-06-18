#include "llm/core/generation_config.h"
#include "llm/core/sequence.h"
#include "llm/core/token_event.h"

#include "llm/engine/scheduled_engine.h"
#include "llm/engine/scheduler.h"

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

    llm::SchedulerConfig schedulerConfig;
    schedulerConfig.maxActiveSequences = 2;

    llm::ScheduledEngine engine(model, tokenizer, schedulerConfig);

    std::vector<llm::RequstId> ids;

    ids.push_back(engine.submit("hello", config));
    ids.push_back(engine.submit("mock model", config));
    ids.push_back(engine.submit("I am", config));

    int tick = 0;

    while (engine.hasWork()) {
        ++tick;

        if (tick == 3) {
            llm::RequstId lateId = engine.submit("kv cache", config);
            ids.push_back(lateId);

            std::cout << "\nsubmitted late request " << lateId << "\n";
        }

        std::cout << "\n=== tick " << tick << " ===\n";
        std::cout << "before step: active=" << engine.activeCount()
                  << " waiting=" << engine.waitingCount()
                  << "\n";

        std::vector<llm::TokenEvent> events = engine.step();

        std::cout << "after step:  active=" << engine.activeCount()
                  << " waiting=" << engine.waitingCount()
                  << "\n";

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
