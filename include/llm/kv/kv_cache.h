#pragma once

#include <cstdint>
#include <stdexcept>
#include <unordered_map>

using u64 = uint64_t;
using RequstId = uint64_t;

namespace llm {

struct KVCacheHandle {
    u64 id = 0;

    bool valid() const {
        return id != 0;
    }
};

struct KVCacheEntry {
    RequstId requestId = 0;
    int currentLength = 0;
};

class KVCacheManager {
private:
    u64 nextId_ = 1;
    std::unordered_map<u64, KVCacheEntry> entries_;

public:
    KVCacheHandle allocate(RequstId requestId, int initialLength) {
        if (initialLength <= 0) {
            throw std::runtime_error("KV cache initial length must be positive");
        }

        KVCacheHandle handle;
        handle.id = nextId_++;

        KVCacheEntry entry;
        entry.requestId = requestId;
        entry.currentLength = initialLength;

        entries_.emplace(handle.id, entry);

        return handle;
    }

    void appendToken(KVCacheHandle handle) {
        auto it = entries_.find(handle.id);

        if (it == entries_.end()) {
            throw std::runtime_error("Invalid KV cache handle in appendToken");
        }

        it->second.currentLength += 1;
    }

    int currentLength(KVCacheHandle handle) const {
        auto it = entries_.find(handle.id);

        if (it == entries_.end()) {
            throw std::runtime_error("Invalid KV cache handle in currentLength");
        }

        return it->second.currentLength;
    }

    void free(KVCacheHandle handle) {
        if (!handle.valid()) return;

        entries_.erase(handle.id);
    }

    size_t activeEntries() const {
        return entries_.size();
    }
};


} // namespace llm
