#pragma once

#include "llm/core/generate_result.h"
#include "llm/core/generation_config.h"
#include "llm/core/sequence.h"
#include "llm/core/token.h"
#include "llm/core/token_event.h"
#include "llm/model/model_backend.h"
#include "llm/sampling/sampler.h"
#include "llm/tokenizer/simple_tokenizer.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace llm {

class MultiRequstEngine {
private:
    ModelBackend& model_;
    SimpleTokenizer& tokenizer_;
    Sampler sampler_;

    RequstId nextRequestId_ = 1;

    std::unordered_map<RequstId, Sequence> sequences_;
    std::vector<RequstId> activeIds_;

    Sequence& getSequenceOrThrow(RequstId id) {
        auto it = sequences_.find(id);

        if (it == sequences_.end()) {
            throw std::runtime_error("Unknown request id");
        }

        return it->second;
    }

    const Sequence& getSequenceOrThrow(RequstId id) const {
        auto it = sequences_.find(id);

        if (it == sequences_.end()) {
            throw std::runtime_error("Unknown request id");
        }

        return it->second;
    }

    static bool isFinishedStatus(SequenceStatus status) {
        return status == SequenceStatus::Finished ||
               status == SequenceStatus::Cancelled;
    }

    void removeInactiveFromActiveList() {
        std::vector<RequstId> stillActive;
        stillActive.reserve(activeIds_.size());

        for (RequstId id : activeIds_) {
            const Sequence& seq = getSequenceOrThrow(id);

            if (!isFinishedStatus(seq.status)) {
                stillActive.push_back(id);
            }
        }

        activeIds_ = std::move(stillActive);
    }

public:
    MultiRequstEngine(ModelBackend& model,
                      SimpleTokenizer& tokenizer)
        : model_(model),
          tokenizer_(tokenizer) {}

    RequstId submit(const std::string& prompt,
                    const GenerationConfig& config) {
        if (config.maxNewTokens <= 0) {
            throw std::runtime_error("maxNewTokens must be positive");
        }

        Sequence seq;
        seq.id = nextRequestId_++;
        seq.prompt = prompt;
        seq.promptsTokens = tokenizer_.encode(prompt);
        seq.config = config;
        seq.status = SequenceStatus::Running;
        seq.finishReason = FinishReason::None;

        RequstId id = seq.id;

        sequences_.emplace(id, std::move(seq));
        activeIds_.push_back(id);

        return id;
    }

    bool hasActive() const {
        return !activeIds_.empty();
    }

    size_t activeCount() const {
        return activeIds_.size();
    }

    void cancel(RequstId id) {
        Sequence& seq = getSequenceOrThrow(id);

        if (seq.status == SequenceStatus::Finished) {
            return;
        }

        seq.status = SequenceStatus::Cancelled;
        seq.finishReason = FinishReason::Cancelled;

        removeInactiveFromActiveList();
    }

    std::vector<TokenEvent> step() {
        std::vector<TokenEvent> events;

        if (activeIds_.empty()) {
            return events;
        }

        std::vector<RequstId> idsThisStep = activeIds_;

        for (RequstId id : idsThisStep) {
            Sequence& seq = getSequenceOrThrow(id);

            if (!seq.isRunning()) {
                continue;
            }

            std::vector<TokenId> context = seq.contextTokens();

            std::vector<float> logits = model_.forwardNextToken(context);

            if (logits.size() != model_.vocabSize()) {
                throw std::runtime_error("Model returned logits with wrong vocab size");
            }

            TokenId nextToken = sampler_.sample(logits, seq.config);

            TokenEvent event;
            event.requestId = id;
            event.tokenId = nextToken;
            event.hasToken = false;
            event.finished = false;
            event.finishReason = FinishReason::None;

            if (nextToken == seq.config.eosTokenId) {
                seq.status = SequenceStatus::Finished;
                seq.finishReason = FinishReason::EosToken;

                event.finished = true;
                event.finishReason = FinishReason::EosToken;

                events.push_back(event);
                continue;
            }

            seq.generatedTokens.push_back(nextToken);

            event.hasToken = true;
            event.text = tokenizer_.decode(std::vector<TokenId>{nextToken}); // a bit clunky

            if (seq.generatedLength() >= seq.config.maxNewTokens) {
                seq.status = SequenceStatus::Finished;
                seq.finishReason = FinishReason::MaxNewTokens;

                event.finished = true;
                event.finishReason = FinishReason::MaxNewTokens;
            }

            events.push_back(event);
        }

        removeInactiveFromActiveList();

        return events;
    }

    GenerateResult result(RequstId id) const {
        const Sequence& seq = getSequenceOrThrow(id);

        GenerateResult result;
        result.promptTokens = seq.promptsTokens;
        result.generatedTokens = seq.generatedTokens;
        result.text = tokenizer_.decode(seq.generatedTokens);
        result.stoppedByEos = seq.finishReason == FinishReason::EosToken;

        return result;
    }

    const Sequence& sequence(RequstId id) const {
        return getSequenceOrThrow(id);
    }
};


} // namespace llm
