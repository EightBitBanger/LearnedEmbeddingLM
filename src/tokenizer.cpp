#include "tokenizer.h"

void Tokenizer::AddTokens(std::vector<std::string> additive) {
    std::sort(additive.begin(), additive.end());
    for (unsigned int i = 0; i < additive.size(); i++) {
        if (wordToToken.find(additive[i]) == wordToToken.end()) {
            wordToToken[additive[i]] = mCurrentToken;
            tokenToWord.push_back(additive[i]);
            mCurrentToken++;
        }
    }
}

bool Tokenizer::CheckWordExists(std::string word) {
    if (wordToToken.find(word) == wordToToken.end())
        return false;
    return true;
}

bool Tokenizer::CheckTokenExists(int token) {
    if (token >= 0 && token < static_cast<int>(tokenToWord.size()))
        return true;
    return false;
}

std::string Tokenizer::GetWord(int token) {
    if (CheckTokenExists(token))
        return tokenToWord[token];
    return ""; 
}

int Tokenizer::GetToken(const std::string& word) {
    if (CheckWordExists(word))
        return wordToToken[word];
    return -1; 
}

