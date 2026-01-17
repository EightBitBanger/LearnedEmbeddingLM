#ifndef _TOKENIZER__
#define _TOKENIZER__

#include <unordered_map>
#include <algorithm>
#include <vector>
#include <string>

#include "tokenizer.h"

class Tokenizer {
    
public:
    // Constructor
    Tokenizer();
    
    // Method to add tokens
    void AddTokens(std::vector<std::string> additive);
    
    bool CheckWordExists(std::string word);
    
    bool CheckTokenExists(int token);
    
    std::string GetWord(int token);
    
    int GetToken(const std::string& word);
    
    // Greedy BPE-style vocab decomposition (longest-match) -> token IDs.
    // - If the full word exists, returns that token.
    // - Otherwise splits into longest vocab pieces.
    // - For non-initial pieces, can prefer continuationPrefix + piece (e.g. "##ing").
    // - If it cannot segment, falls back to unkToken if present; otherwise returns false.
    bool TokenizeWordBPE(const std::string& word, 
                         std::vector<int>& outTokens, 
                         const std::string& unkToken = "<unk>", 
                         bool useContinuationPrefix = true, 
                         const std::string& continuationPrefix = "##") const;
    
    // Save the vocabulary to a file.
    bool SaveToFile(const std::string& filename) const;
    
    // Load a vocabulary from a file.
    bool LoadFromFile(const std::string& filename);
    
    std::unordered_map<std::string, int> wordToToken;
    std::vector<std::string> tokenToWord; 
    
    unsigned int mCurrentToken;
};

#endif
