#include "languagemodel.h"

#include <algorithm>
#include <cstdint>
#include <limits>

LanguageModel::LanguageModel() : 
    mMaxSpanLen(32),
    mSpanStride(16) {}

void LanguageModel::ProcessSequence(std::vector<int> tokens, float learningRate) {
    if (tokens.size() < 2) {
        return;
    }
    
    const int N = (int)tokens.size();
    
    // Store sliding-window spans.
    for (int start = 0; start < N; start += mSpanStride) {
        const int end = std::min(start + mMaxSpanLen, N);
        const int len = end - start;
        if (len < 2) {
            break;
        }
        
        std::vector<int> span;
        span.reserve((size_t)len);
        for (int i = start; i < end; ++i) {
            span.push_back(tokens[(size_t)i]);
        }
        
        mModel.push_back(span);
        
        if (end == N) {
            break;
        }
    }
}

// Extract relevant spans from the model.
// "Relevant" = contains topic tokens; returns top spans sorted by match strength.
std::vector<std::vector<int>> LanguageModel::Extract(std::vector<int> topic) {
    std::vector<std::vector<int>> out;
    if (topic.empty() || mModel.empty()) {
        return out;
    }
    
    // Build a set of topic tokens (dedupe).
    std::unordered_set<int> topicSet;
    topicSet.reserve(topic.size() * 2);
    for (size_t i = 0; i < topic.size(); ++i) {
        topicSet.insert(topic[i]);
    }
    
    // Score spans by:
    //  - number of unique topic tokens present (primary)
    //  - total occurrences of topic tokens (secondary)
    struct ScoredSpan {
        int scoreUnique;
        int scoreHits;
        size_t index;
    };
    
    std::vector<ScoredSpan> scored;
    scored.reserve(mModel.size() / 8 + 1);
    
    for (size_t s = 0; s < mModel.size(); ++s) {
        const std::vector<int>& span = mModel[s];
        
        int hits = 0;
        int unique = 0;
        
        // Track which topic tokens we saw in this span.
        // Using unordered_set would be heavier; this map is small in practice,
        // but we can do a cheap bitmap-like approach via unordered_set on demand.
        std::unordered_set<int> seen;
        seen.reserve(16);
        
        for (size_t i = 0; i < span.size(); ++i) {
            const int t = span[i];
            if (topicSet.find(t) != topicSet.end()) {
                ++hits;
                if (seen.insert(t).second) {
                    ++unique;
                }
            }
        }
        
        if (hits > 0) {
            ScoredSpan ss;
            ss.scoreUnique = unique;
            ss.scoreHits   = hits;
            ss.index       = s;
            scored.push_back(ss);
        }
    }
    
    if (scored.empty()) {
        return out;
    }
    
    std::sort(scored.begin(), scored.end(),
        [](const ScoredSpan& a, const ScoredSpan& b) -> bool {
            if (a.scoreUnique != b.scoreUnique) return a.scoreUnique > b.scoreUnique;
            if (a.scoreHits   != b.scoreHits)   return a.scoreHits   > b.scoreHits;
            return a.index < b.index;
        });
    
    // Return the best matches (cap to avoid huge output).
    const size_t maxOut = 2048;
    const size_t take = std::min(maxOut, scored.size());
    out.reserve(take);
    
    for (size_t i = 0; i < take; ++i) {
        out.push_back(mModel[scored[i].index]);
    }
    
    return out;
}

bool LanguageModel::SaveToFile(const std::string& filename) const {
    std::ofstream ofs(filename.c_str(), std::ios::binary | std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        return false;
    }

    // --- Tokenizer vocab (tokenToWord is authoritative, ordered) ---
    const std::uint32_t vocabCount = (std::uint32_t)tok.tokenToWord.size();
    ofs.write((const char*)&vocabCount, sizeof(vocabCount));
    if (!ofs.good()) {
        return false;
    }

    for (std::uint32_t i = 0; i < vocabCount; ++i) {
        const std::string& s = tok.tokenToWord[(size_t)i];

        if (s.size() > (size_t)std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }

        const std::uint32_t len = (std::uint32_t)s.size();
        ofs.write((const char*)&len, sizeof(len));
        if (len > 0) {
            ofs.write(s.data(), (std::streamsize)len);
        }

        if (!ofs.good()) {
            return false;
        }
    }

    // --- Model spans ---
    if (mModel.size() > (size_t)std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    const std::uint32_t spanCount = (std::uint32_t)mModel.size();
    ofs.write((const char*)&spanCount, sizeof(spanCount));
    if (!ofs.good()) {
        return false;
    }

    const bool int32FastPath = (sizeof(int) == sizeof(std::int32_t));

    for (std::uint32_t s = 0; s < spanCount; ++s) {
        const std::vector<int>& span = mModel[(size_t)s];

        if (span.size() > (size_t)std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }
        const std::uint32_t len = (std::uint32_t)span.size();
        ofs.write((const char*)&len, sizeof(len));

        if (len > 0) {
            if (int32FastPath) {
                // Write the vector in one go (much faster than per-token writes).
                ofs.write((const char*)span.data(), (std::streamsize)len * (std::streamsize)sizeof(std::int32_t));
            } else {
                for (std::uint32_t i = 0; i < len; ++i) {
                    const std::int32_t tok32 = (std::int32_t)span[(size_t)i];
                    ofs.write((const char*)&tok32, sizeof(tok32));
                }
            }
        }

        if (!ofs.good()) {
            return false;
        }
    }

    return true;
}

bool LanguageModel::LoadFromFile(const std::string& filename) {
    std::ifstream ifs(filename.c_str(), std::ios::binary | std::ios::in);
    if (!ifs.is_open()) {
        return false;
    }
    
    // Reset current state
    tok.wordToToken.clear();
    tok.tokenToWord.clear();
    tok.mCurrentToken = 0;
    
    mModel.clear();
    
    // --- Tokenizer vocab ---
    std::uint32_t vocabCount = 0;
    ifs.read((char*)&vocabCount, sizeof(vocabCount));
    if (!ifs.good()) {
        return false;
    }
    
    tok.tokenToWord.reserve((size_t)vocabCount);
    tok.wordToToken.reserve((size_t)vocabCount);
    
    for (std::uint32_t i = 0; i < vocabCount; ++i) {
        std::uint32_t len = 0;
        ifs.read((char*)&len, sizeof(len));
        if (!ifs.good()) {
            return false;
        }
        
        std::string s;
        if (len > 0) {
            s.resize((size_t)len);
            ifs.read(&s[0], (std::streamsize)len);
            if (!ifs.good()) {
                return false;
            }
        }
        
        if (tok.wordToToken.find(s) != tok.wordToToken.end()) {
            return false; // reject duplicates for determinism
        }
        
        tok.wordToToken[s] = (int)i;
        tok.tokenToWord.push_back(s);
    }
    
    // Finalize tokenizer internal counter:
    // Next new token should be appended after the last loaded token.
    tok.mCurrentToken = vocabCount;
    
    // --- Model spans ---
    std::uint32_t spanCount = 0;
    ifs.read((char*)&spanCount, sizeof(spanCount));
    if (!ifs.good()) {
        return false;
    }
    
    mModel.reserve((size_t)spanCount);
    
    const bool int32FastPath = (sizeof(int) == sizeof(std::int32_t));

    for (std::uint32_t s = 0; s < spanCount; ++s) {
        std::uint32_t len = 0;
        ifs.read((char*)&len, sizeof(len));
        if (!ifs.good()) {
            return false;
        }
        
        std::vector<int> span;
        span.resize((size_t)len);

        if (len > 0) {
            if (int32FastPath) {
                // Read the token IDs in one go.
                ifs.read((char*)span.data(), (std::streamsize)len * (std::streamsize)sizeof(std::int32_t));
                if (!ifs.good()) {
                    return false;
                }

                // Validate token IDs.
                for (std::uint32_t i = 0; i < len; ++i) {
                    const std::int32_t t = (std::int32_t)span[(size_t)i];
                    if (t < 0 || (std::uint32_t)t >= vocabCount) {
                        return false;
                    }
                }
            } else {
                for (std::uint32_t i = 0; i < len; ++i) {
                    std::int32_t t = 0;
                    ifs.read((char*)&t, sizeof(t));
                    if (!ifs.good()) {
                        return false;
                    }

                    // Safety: reject corrupted models with invalid token ids
                    if (t < 0 || (std::uint32_t)t >= vocabCount) {
                        return false;
                    }

                    span[(size_t)i] = (int)t;
                }
            }
        }
        
        mModel.push_back(span);
    }
    
    return true;
}
