#include "languagemodel.h"

#include <fstream>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

LanguageModel::LanguageModel(Tokenizer* tokenizer) : 
    tok(tokenizer) {}

bool LanguageModel::GetRelevantContext(AttentionSystem& attention,
                                       const std::vector<int>& context,
                                       std::vector<int>& focus) {
    if (context.empty() || mModel.empty()) 
        return false;
    
    std::vector<int> content;
    for (unsigned int i=0; i < context.size(); i++) {
        int token = context[i];
        if (attention.IsContentLike(token)) 
            content.push_back( token );
    }
    
    // Build a set of tokens from the current context for quick lookup.
    std::unordered_set<int> contextTokens;
    contextTokens.reserve(content.size());
    for (std::size_t i = 0; i < content.size(); ++i) 
        contextTokens.insert(content[i]);
    
    if (contextTokens.empty()) 
        return false;
    
    // Use the existing GetContext to get span-level neighborhoods.
    std::vector<std::vector<int>> baseFocus;
    if (!GetContext(content, baseFocus, 1)) {
        return false;
    }
    
    bool foundAny = false;
    
    // De-duplicate tokens we add to the flat focus list.
    std::unordered_set<int> addedTokens;
    
    for (unsigned int si = 0; si < baseFocus.size(); si++) {
        const std::vector<int>& span = baseFocus[si];
        unsigned int spanSize = span.size();
        if (spanSize == 0) 
            continue;
        
        for (unsigned int s = 0; s < spanSize; s++) {
            int token = span[s];
            
            // Only keep clearly content-like neighbors.
            if (!attention.IsContentLike(token)) 
                continue;
            
            // Add to flat focus list once per token.
            if (addedTokens.insert(token).second) {
                focus.push_back(token);
                foundAny = true;
            }
        }
    }
    
    return foundAny;
}

bool LanguageModel::GetContext(const std::vector<int>& context, std::vector<std::vector<int>>& focus, unsigned int range) {
    if (context.empty() || mModel.empty())
        return false;
    
    if (range < 1) range = 1;
    unsigned int pre = range-1;
    unsigned int post = range;
    
    const unsigned int focusMaxSz = 1024 * 740;
    const unsigned int modelSize  = static_cast<unsigned int>(mModel.size());
    bool foundAny = false;
    
    if (focus.size() > focusMaxSz) {
        focus.erase(focus.begin(), focus.begin() + focusMaxSz / 4);
    }
    
    // If the context is only one token long, fall back to simple single-token matching
    if (context.size() < 2) {
        std::unordered_set<int> contextTokens;
        contextTokens.reserve(context.size());
        for (std::size_t i = 0; i < context.size(); ++i) {
            contextTokens.insert(context[i]);
        }
    
        std::vector<bool> spanAdded(modelSize, false);
        
        for (unsigned int si = 0; si < modelSize; ++si) {
            const std::vector<int>& span = mModel[si];
            
            for (std::size_t ti = 0; ti < span.size(); ++ti) {
                int token = span[ti];
                
                if (contextTokens.find(token) != contextTokens.end()) {
                    // Add this span and its neighbors [si-pre, si+post]
                    int start = static_cast<int>(si) - pre;
                    if (start < 0) {
                        start = 0;
                    }
                    int end = static_cast<int>(si) + post;
                    if (end >= static_cast<int>(modelSize)) {
                        end = static_cast<int>(modelSize) - 1;
                    }
                    
                    for (int idx = start; idx <= end; ++idx) {
                        if (!spanAdded[static_cast<unsigned int>(idx)]) {
                            focus.push_back(mModel[static_cast<unsigned int>(idx)]);
                            spanAdded[static_cast<unsigned int>(idx)] = true;
                            foundAny = true;
                            
                            if (focus.size() > focusMaxSz) 
                                break;
                        }
                    }
                    // no need to keep scanning this span
                    break;
                }
            }
        }
        
        return foundAny;
    }
    
    // Build a set of all adjacent token pairs in the context for bi-gram matching.
    // We encode each pair (a,b) into a 64-bit integer key.
    std::unordered_set<std::uint64_t> contextPairs;
    contextPairs.reserve(context.size());
    
    for (std::size_t i = 0; i + 1 < context.size(); ++i) {
        int a = context[i];
        int b = context[i + 1];
        
        std::uint64_t key =
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) ^
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(b));
        
        contextPairs.insert(key);
    }
    
    std::vector<bool> spanAdded(modelSize, false);
    
    // Scan each span in the model for any adjacent token pair that matches
    // one of the bigrams from the context. If it matches, we add the span
    // and its neighbors to focus.
    for (unsigned int si = 0; si < modelSize; ++si) {
        const std::vector<int>& span = mModel[si];
        // Not enough tokens to form a pair
        if (span.size() < 2) 
            continue;
        
        for (std::size_t ti = 0; ti + 1 < span.size(); ++ti) {
            int a = span[ti];
            int b = span[ti + 1];
            
            std::uint64_t key =
                (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) ^
                static_cast<std::uint64_t>(static_cast<std::uint32_t>(b));
            
            if (contextPairs.find(key) != contextPairs.end()) {
                // Add this span and its neighbors [si-pre, si+post]
                int start = static_cast<int>(si) - pre;
                if (start < 0) {
                    start = 0;
                }
                int end = static_cast<int>(si) + post;
                if (end >= static_cast<int>(modelSize)) {
                    end = static_cast<int>(modelSize) - 1;
                }
                
                for (int idx = start; idx <= end; ++idx) {
                    if (!spanAdded[static_cast<unsigned int>(idx)]) {
                        focus.push_back(mModel[static_cast<unsigned int>(idx)]);
                        spanAdded[static_cast<unsigned int>(idx)] = true;
                        foundAny = true;
                        
                        if (focus.size() > focusMaxSz) 
                            break;
                    }
                }
                
                // Once we know this span matches at least one bigram (and we've
                // added its neighborhood), we don't need to keep scanning it.
                break;
            }
        }
    }
    
    return foundAny;
}

void LanguageModel::AddContext(const std::vector<int>& context) {
    if (context.empty()) 
        return;
    
    mModel.push_back(context);
}


bool LanguageModel::SaveToFile(const std::string& filename) const {
    if (tok == nullptr) 
        return false;
    
    std::ofstream out(filename.c_str(), std::ios::binary);
    if (!out.is_open()) 
        return false;
    
    // Save tokenizer vocabulary
    std::uint32_t vocabSize = static_cast<std::uint32_t>(tok->tokenToWord.size());
    out.write(reinterpret_cast<const char*>(&vocabSize), sizeof(vocabSize));
    if (!out.good()) 
        return false;
    
    for (std::uint32_t i = 0; i < vocabSize; i++) {
        const std::string& word = tok->tokenToWord[static_cast<std::size_t>(i)];
        
        // FIX: index should be the token index (i), not word.size()
        std::uint32_t index = i;
        std::uint32_t len   = static_cast<std::uint32_t>(word.size());
        
        out.write(reinterpret_cast<const char*>(&index), sizeof(index));
        out.write(reinterpret_cast<const char*>(&len),   sizeof(len));
        if (!out.good()) 
            return false;
        
        if (len > 0) {
            out.write(word.data(), static_cast<std::streamsize>(len));
            if (!out.good()) {
                return false;
            }
        }
    }
    
    // Save model spans
    std::uint32_t spanCount = static_cast<std::uint32_t>(mModel.size());
    out.write(reinterpret_cast<const char*>(&spanCount), sizeof(spanCount));
    if (!out.good()) 
        return false;
    
    for (std::uint32_t i = 0; i < spanCount; i++) {
        const std::vector<int>& span = mModel[static_cast<std::size_t>(i)];
        std::uint32_t spanLen = static_cast<std::uint32_t>(span.size());
        
        out.write(reinterpret_cast<const char*>(&spanLen), sizeof(spanLen));
        if (!out.good()) 
            return false;
        
        for (std::uint32_t j = 0; j < spanLen; j++) {
            std::int32_t tokVal = static_cast<std::int32_t>(span[static_cast<std::size_t>(j)]);
            out.write(reinterpret_cast<const char*>(&tokVal), sizeof(tokVal));
            if (!out.good()) {
                return false;
            }
        }
    }
    
    return out.good();
}


bool LanguageModel::LoadFromFile(const std::string& filename) {
    if (tok == nullptr) 
        return false;
    
    std::ifstream in(filename.c_str(), std::ios::binary);
    if (!in.is_open()) 
        return false;
    
    mModel.clear();
    
    // Load tokenizer vocabulary
    std::uint32_t vocabSize = 0;
    in.read(reinterpret_cast<char*>(&vocabSize), sizeof(vocabSize));
    if (!in.good()) 
        return false;
    
    // Reset tokenizer data
    tok->tokenToWord.clear();
    tok->wordToToken.clear();
    tok->tokenToWord.resize(static_cast<std::size_t>(vocabSize));
    
    for (std::uint32_t n = 0; n < vocabSize; n++) {
        std::uint32_t index = 0;
        std::uint32_t len   = 0;
        
        in.read(reinterpret_cast<char*>(&index), sizeof(index));
        in.read(reinterpret_cast<char*>(&len),   sizeof(len));
        if (!in.good()) {
            tok->tokenToWord.clear();
            tok->wordToToken.clear();
            return false;
        }
        
        std::string word;
        word.resize(static_cast<std::size_t>(len));
        
        if (len > 0) {
            in.read(&word[0], static_cast<std::streamsize>(len));
            if (!in.good()) {
                tok->tokenToWord.clear();
                tok->wordToToken.clear();
                return false;
            }
        }
        
        if (index >= vocabSize) {
            // Corrupt file; bail out
            tok->tokenToWord.clear();
            tok->wordToToken.clear();
            return false;
        }
        
        tok->tokenToWord[static_cast<std::size_t>(index)] = word;
        tok->wordToToken[word] = static_cast<int>(index);
    }
    
    // Load model spans
    std::uint32_t spanCount = 0;
    in.read(reinterpret_cast<char*>(&spanCount), sizeof(spanCount));
    if (!in.good()) {
        tok->tokenToWord.clear();
        tok->wordToToken.clear();
        return false;
    }
    
    mModel.reserve(static_cast<std::size_t>(spanCount));
    
    for (std::uint32_t i = 0; i < spanCount; i++) {
        std::uint32_t spanLen = 0;
        in.read(reinterpret_cast<char*>(&spanLen), sizeof(spanLen));
        if (!in.good()) {
            mModel.clear();
            tok->tokenToWord.clear();
            tok->wordToToken.clear();
            return false;
        }
        
        std::vector<int> span;
        span.reserve(static_cast<std::size_t>(spanLen));
        
        for (std::uint32_t j = 0; j < spanLen; j++) {
            std::int32_t tok32 = 0;
            in.read(reinterpret_cast<char*>(&tok32), sizeof(tok32));
            if (!in.good()) {
                mModel.clear();
                tok->tokenToWord.clear();
                tok->wordToToken.clear();
                return false;
            }
            span.push_back(static_cast<int>(tok32));
        }
        
        mModel.push_back(span);
    }
    
    return true;
}

unsigned int LanguageModel::size(void) const {
    return mModel.size();
}
