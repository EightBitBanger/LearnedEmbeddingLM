#include "context.h"
#include "string.h"

Context::Context(Tokenizer* tokenizer) : 
    mTokenizer(tokenizer) {}

Context& Context::operator=(const std::vector<int>& tokens) {
    mContext = tokens;
    return *this;
}

// Assign from a vector of words
Context& Context::operator=(const std::vector<std::string>& words) {
    if (!mTokenizer) {
        mContext.clear();
        return *this;
    }
    std::vector<std::string> additive = words;
    for (unsigned int i=0; i < additive.size(); i++) 
        StringCaseLowerAll(additive[i]);
    
    // Ensure all words exist in the tokenizer vocabulary
    mTokenizer->AddTokens(additive);
    
    // Convert words to tokens
    mContext.clear();
    mContext.reserve(additive.size());
    for (std::size_t i = 0; i < additive.size(); ++i) {
        int token = mTokenizer->GetToken(additive[i]);
        if (token != -1) {
            mContext.push_back(token);
        }
        // If token == -1, the word somehow didn't make it into the vocab;
        // we simply skip it to avoid storing invalid tokens.
    }
    
    return *this;
}

// Get context as token IDs (read-only)
const std::vector<int>& Context::GetTokens() const {
    return mContext;
}

// Get context as words
std::vector<std::string> Context::GetWords() const {
    std::vector<std::string> result;
    if (!mTokenizer) return result;
    
    result.reserve(mContext.size());
    for (std::size_t i = 0; i < mContext.size(); ++i) {
        result.push_back(mTokenizer->GetWord(mContext[i]));
    }
    return result;
}

// Optional convenience
void Context::Clear() {
    mContext.clear();
}
