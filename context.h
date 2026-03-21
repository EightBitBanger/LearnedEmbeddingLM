#ifndef _CONTEXT__
#define _CONTEXT__

#include <vector>
#include <string>
#include "tokenizer.h"

class Context {
public:
    // Require a tokenizer at construction time
    Context(Tokenizer* tokenizer);
    
    // Assign from a vector of token IDs
    Context& operator=(const std::vector<int>& tokens);
    
    // Assign from a vector of words
    Context& operator=(const std::vector<std::string>& words);
    
    // Get context as token IDs (read-only)
    const std::vector<int>& GetTokens() const;
    
    // Get context as words
    std::vector<std::string> GetWords() const;
    
    // Optional convenience
    void Clear();
    
private:
    Tokenizer* mTokenizer;        // not owned, just referenced
    std::vector<int> mContext;    // current context as token IDs
};

#endif
