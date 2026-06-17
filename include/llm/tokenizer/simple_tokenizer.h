#pragma once

#include "llm/core/token.h"

#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <string>

namespace llm {

class SimpleTokenizer {
private:
    std::vector<std::string> idToToken_;
    std::unordered_map<std::string, TokenId> tokenToId_;

public:
    SimpleTokenizer() {
        idToToken_ = {
            "<pad>",     // 0
            "<bos>",     // 1
            "<eos>",     // 2
            "hello",     // 3
            "world",     // 4
            "I",         // 5
            "am",        // 6
            "a",         // 7
            "mock",      // 8
            "model",     // 9
            "token",     // 10
            "engine",    // 11
            "batching",  // 12
            "kv",        // 13
            "cache",     // 14
            ".",         // 15
            "<unk>"      // 16
        };

        for (TokenId i = 0; i < static_cast<TokenId>(idToToken_.size()); i++) {
            tokenToId_[idToToken_[i]] = i;
        }
    }

    std::vector<TokenId> encode(const std::string& text) const {
        std::vector<TokenId> ids;

        std::istringstream iss(text);
        std::string word;

        while (iss >> word) {
            auto it = tokenToId_.find(word);

            if (it == tokenToId_.end()) {
                ids.push_back(unkTokenId());
            } else {
                ids.push_back(it->second);
            }
        }

        if (ids.empty()) {
            ids.push_back(bosTokenId());
        }

        return ids;
    }

    std::string decode(const std::vector<TokenId>& ids) const {
        std::string out;

        for (TokenId id : ids) {
            if (id == eosTokenId() || id == padTokenId() || id == bosTokenId()) {
                continue;
            }

            if (id < 0 || id >= static_cast<TokenId>(idToToken_.size())) {
                throw std::runtime_error("Token id out of range in decode");
            }

            if (!out.empty()) {
                out += " ";
            }

            out += idToToken_[id];
        }

        return out;
    }

    size_t vocabSize() const {
        return idToToken_.size();
    }

    TokenId padTokenId() const { return 0; }
    TokenId bosTokenId() const { return 1; }
    TokenId eosTokenId() const { return 2; }
    TokenId unkTokenId() const { return 16; }
};


} // namespace llm
