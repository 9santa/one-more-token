#pragma once

#include "llm/core/generate_result.h"
#include "llm/core/generation_config.h"
#include "llm/core/sequence.h"
#include "llm/core/token.h"
#include "llm/core/token_event.h"

#include "llm/engine/decode_batch.h"
#include "llm/engine/scheduler.h"

#include "llm/model/model_backend.h"
#include "llm/sampling/sampler.h"
#include "llm/tokenizer/simple_tokenizer.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace llm {

class ScheduledEngine {
private:
    ModelBackend& model_;
    SimpleTokenizer& tokenizer_;
    Sampler sampler_;
    Scheduler scheduler_;

    RequstId nextRequestId_ = 1;

    std::unordered_map<RequstId, Sequence> sequences_;

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

    void admitWaitingRequsts() {
        std::vector<RequstId> admitted = scheduler_.admitWaiting();

        for (RequstId id : admitted) {
            Sequence& seq = getSequenceOrThrow(id);

            if (!seq.isWaiting()) {
                throw std::runtime_error("Scheduler admitted non-waiting sequence");
            }

            seq.status = SequenceStatus::Running;
        }
    }

    DecodeBatch buildDecodeBatch() const {
        DecodeBatch batch;

        const std::vector<RequstId>& activeIds = scheduler_.activeIds();

        batch.requestIds.reserve(activeIds.size());
        batch.contexts.reserve(activeIds.size());

        for (RequstId id : activeIds) {
            const Sequence& seq = getSequenceOrThrow(id);

            if (!seq.isRunning()) {
                continue;
            }

            batch.requestIds.push_back(id);
            batch.contexts.push_back(seq.contextTokens());
        }

        if (!batch.empty()) {
            batch.validate();
        }

        return batch;
    }

public:
    ScheduledEngine(ModelBackend& model,
                    SimpleTokenizer& tokenizer,
                    const SchedulerConfig& schedulerConfig)
        : model_(model),
          tokenizer_(tokenizer),
          scheduler_(schedulerConfig) {}

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
        seq.status = SequenceStatus::Waiting;
        seq.finishReason = FinishReason::None;

        RequstId id = seq.id;

        sequences_.emplace(id, std::move(seq));
        scheduler_.addWaiting(id);

        return id;
    }

    void cancel(RequstId id) {
        Sequence& seq = getSequenceOrThrow(id);

        if (seq.isFinished()) {
            return;
        }

        seq.status = SequenceStatus::Cancelled;
        seq.finishReason = FinishReason::Cancelled;

        scheduler_.remove(id);
    }

    bool hasWork() const {
        return scheduler_.hasWork();
    }

    size_t activeCount() const {
        return scheduler_.activeCount();
    }

    size_t waitingCount() const {
        return scheduler_.waitingCount();
    }

    std::vector<TokenEvent> step() {
        std::vector<TokenEvent> events;

        // Important:
        // Admit waiting requests before building the batch.
        // This fills free capacity.
        admitWaitingRequsts();

        DecodeBatch batch = buildDecodeBatch();

        if (batch.empty()) {
            return events;
        }

        std::vector<std::vector<float>> batchLogits =
            model_.forwardNextTokenBatch(batch.contexts);

        if (batchLogits.size() != batch.size()) {
            throw std::runtime_error("Model returned wrong number of batch rows");
        }

        std::vector<RequstId> finishedThisStep;

        for (size_t row = 0; row < batch.size(); ++row) {
            RequstId id = batch.requestIds[row];
            Sequence& seq = getSequenceOrThrow(id);

            if (!seq.isRunning()) {
                continue;
            }

            const std::vector<float>& logits = batchLogits[row];

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

                finishedThisStep.push_back(id);
                events.push_back(event);
                continue;
            }

            seq.generatedTokens.push_back(nextToken);

            event.hasToken = true;
            event.text = tokenizer_.decode(std::vector<TokenId>{nextToken});

            if (seq.generatedLength() >= seq.config.maxNewTokens) {
                seq.status = SequenceStatus::Finished;
                seq.finishReason = FinishReason::MaxNewTokens;

                event.finished = true;
                event.finishReason = FinishReason::MaxNewTokens;

                finishedThisStep.push_back(id);
            }

            events.push_back(event);
        }

        for (RequstId id : finishedThisStep) {
            scheduler_.removeActive(id);
        }

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
