#pragma once

#include "llm/core/generate_result.h"
#include "llm/core/generation_config.h"
#include "llm/core/sequence.h"
#include "llm/core/token.h"
#include "llm/core/token_event.h"

#include "llm/engine/batch.h"
#include "llm/engine/scheduler.h"

#include "llm/kv/kv_cache.h"
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
    KVCacheManager kvCacheManager_;

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

            if (seq.kvCache.valid()) {
                throw std::runtime_error("Sequence already had KV cache before prefill");
            }

            seq.kvCache = kvCacheManager_.allocate(
                seq.id,
                static_cast<int>(seq.promptsTokens.size())
            );

            seq.status = SequenceStatus::Prefill;
        }
    }

    PrefillBatch buildPrefillBatch() const {
        PrefillBatch batch;

        const std::vector<RequstId>& activeIds = scheduler_.activeIds();

        for (RequstId id : activeIds) {
            const Sequence& seq = getSequenceOrThrow(id);

            if (!seq.isPrefill()) continue;

            batch.requestIds.push_back(id);
            batch.promptTokens.push_back(seq.promptsTokens);
        }

        if (!batch.empty()) {
            batch.validate();
        }

        return batch;
    }

    DecodeBatch buildDecodeBatch() const {
        DecodeBatch batch;

        const std::vector<RequstId>& activeIds = scheduler_.activeIds();

        for (RequstId id : activeIds) {
            const Sequence& seq = getSequenceOrThrow(id);

            if (!seq.isDecode()) continue;

            batch.requestIds.push_back(id);
            batch.lastTokens.push_back(seq.lastToken());
            batch.positions.push_back(seq.totalLength());
        }

        if (!batch.empty()) {
            batch.validate();
        }

        return batch;
    }

    TokenEvent applyLogitsToSequence(RequstId id,
                                     const std::vector<float>& logits,
                                     std::vector<RequstId>& finishedThisStep) {
        if (logits.size() != model_.vocabSize()) {
            throw std::runtime_error("Model returned logits with wrong vocab size");
        }

        Sequence& seq = getSequenceOrThrow(id);

        if (!seq.kvCache.valid()) {
            throw std::runtime_error("Sequence has no KV cache");
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

            kvCacheManager_.free(seq.kvCache);
            seq.kvCache = KVCacheHandle{};

            event.finished = true;
            event.finishReason = FinishReason::EosToken;

            finishedThisStep.push_back(id);
            return event;
        }

        seq.generatedTokens.push_back(nextToken);

        // Generated one real token, so the KV cache sequence length grows by 1
        kvCacheManager_.appendToken(seq.kvCache);

        event.hasToken = true;
        event.text = tokenizer_.decode(std::vector<TokenId>{nextToken});

        if (seq.generatedLength() >= seq.config.maxNewTokens) {
            seq.status = SequenceStatus::Finished;
            seq.finishReason = FinishReason::MaxNewTokens;

            kvCacheManager_.free(seq.kvCache);
            seq.kvCache = KVCacheHandle{};

            event.finished = true;
            event.finishReason = FinishReason::MaxNewTokens;

            finishedThisStep.push_back(id);
            return event;
        }

        // After prefill emits the first token, the sequence moves into decode
        // Decode sequences remain decode sequences
        seq.status = SequenceStatus::Decode;

        return event;
    }

    std::vector<TokenEvent> runPrefillBatch(const PrefillBatch& batch) {
        std::vector<TokenEvent> events;

        std::vector<std::vector<float>> batchLogits =
            model_.prefillBatch(batch.promptTokens);

        if (batchLogits.size() != batch.size()) {
            throw std::runtime_error("Model returned wrong prefill batch size");
        }

        std::vector<RequstId> finishedThisStep;

        for (size_t row = 0; row < batch.size(); row++) {
            RequstId id = batch.requestIds[row];

            TokenEvent event =
                applyLogitsToSequence(id, batchLogits[row], finishedThisStep);

            events.push_back(event);
        }

        for (RequstId id : finishedThisStep) {
            scheduler_.removeActive(id);
        }

        return events;
    }

    std::vector<TokenEvent> runDecodeBatch(const DecodeBatch& batch) {
        std::vector<TokenEvent> events;

        std::vector<std::vector<float>> batchLogits =
            model_.decodeBatch(batch.lastTokens, batch.positions);

        if (batchLogits.size() != batch.size()) {
            throw std::runtime_error("Model returned wrong decode batch size");
        }

        std::vector<RequstId> finishedThisStep;

        for (size_t row = 0; row < batch.size(); row++) {
            RequstId id = batch.requestIds[row];

            TokenEvent event =
                applyLogitsToSequence(id, batchLogits[row], finishedThisStep);

            events.push_back(event);
        }

        for (RequstId id : finishedThisStep) {
            scheduler_.removeActive(id);
        }

        return events;
    }

    size_t countStatus(SequenceStatus status) const {
        size_t count = 0;

        for (RequstId id : scheduler_.activeIds()) {
            const Sequence& seq = getSequenceOrThrow(id);

            if (seq.status == status) count++;
        }

        return count;
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

        if (seq.kvCache.valid()) {
            kvCacheManager_.free(seq.kvCache);
            seq.kvCache = KVCacheHandle{};
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

    size_t prefillCount() const {
        return countStatus(SequenceStatus::Prefill);
    }

    size_t decodeCount() const {
        return countStatus(SequenceStatus::Decode);
    }

    size_t activeKVCacheCount() const {
        return kvCacheManager_.activeEntries();
    }

    std::vector<TokenEvent> step() {
        admitWaitingRequsts();

        PrefillBatch prefillBatch = buildPrefillBatch();

        if (!prefillBatch.empty()) {
            return runPrefillBatch(prefillBatch);
        }

        DecodeBatch decodeBatch = buildDecodeBatch();

        if (!decodeBatch.empty()) {
            return runDecodeBatch(decodeBatch);
        }

        return {};
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
