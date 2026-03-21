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
    Tokenizer() : mCurrentToken(0) {}
    
    // Method to add tokens
    void AddTokens(std::vector<std::string> additive);
    
    bool CheckWordExists(std::string word);
    
    bool CheckTokenExists(int token);
    
    std::string GetWord(int token);
    
    int GetToken(const std::string& word);
    
    std::unordered_map<std::string, int> wordToToken;
    std::vector<std::string> tokenToWord; 
    
private:
    
    unsigned int mCurrentToken;
};

#endif
