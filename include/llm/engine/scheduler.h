#pragma once

#include "llm/core/sequence.h"

#include <algorithm>
#include <deque>
#include <pstl/glue_algorithm_defs.h>
#include <stdexcept>

namespace llm {

struct SchedulerConfig {
    size_t maxActiveSequences = 4;
};

class Scheduler {
private:
    SchedulerConfig config_;

    std::deque<RequstId> waiting_;
    std::vector<RequstId> active_;

    bool containsActive(RequstId id) const {
        return std::find(active_.begin(), active_.end(), id) != active_.end();
    }

    bool containsWaiting(RequstId id) const {
        return std::find(waiting_.begin(), waiting_.end(), id) != waiting_.end();
    }

public:
    explicit Scheduler(const SchedulerConfig& config)
        : config_(config) {
        if (config_.maxActiveSequences == 0) {
            throw std::runtime_error("Scheduler maxActiveSequences must be positive");
        }
    }

    void addWaiting(RequstId id) {
        if (containsActive(id) || containsWaiting(id)) {
            throw std::runtime_error("Request already known to scheduler");
        }

        waiting_.push_back(id);
    }

    std::vector<RequstId> admitWaiting() {
        std::vector<RequstId> admitted;

        while (!waiting_.empty() &&
               active_.size() < config_.maxActiveSequences) {
            RequstId id = waiting_.front();
            waiting_.pop_front();

            active_.push_back(id);
            admitted.push_back(id);
        }

        return admitted;
    }

    void removeActive(RequstId id) {
        auto it = std::remove(active_.begin(), active_.end(), id);

        if (it == active_.end()) {
            return;
        }

        active_.erase(it, active_.end());
    }

    void removeWaiting(RequstId id) {
        auto it = std::remove(waiting_.begin(), waiting_.end(), id);

        if (it == waiting_.end()) {
            return;
        }

        waiting_.erase(it, waiting_.end());
    }

    void remove(RequstId id) {
        removeActive(id);
        removeWaiting(id);
    }

    const std::vector<RequstId>& activeIds() const {
        return active_;
    }

    bool hasWork() const {
        return !waiting_.empty() || !active_.empty();
    }

    size_t activeCount() const {
        return active_.size();
    }

    size_t waitingCount() const {
        return waiting_.size();
    }

    size_t maxActiveSequences() const {
        return config_.maxActiveSequences;
    }
};


} // namespace llm
