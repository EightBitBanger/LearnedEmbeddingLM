#include <fstream>
#include <limits>

#include "tokenizer.h"

Tokenizer::Tokenizer() : 
    mCurrentToken(0) {
    AddTokens({"<pad>",    // Filler non-space token
               "<unk>",    // Unknown to the vocabulary
               "<bos>",    // Beginning of a sentence
               "<eos>",    // End of a sentence
               "<tool>"}); // Agent function call
}

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



bool Tokenizer::SaveToFile(const std::string& filename) const {
    std::ofstream ofs(filename.c_str(), std::ios::binary | std::ios::out | std::ios::trunc);
    if (!ofs.is_open())
        return false;

    // Header
    const char magic[8] = { 'T','O','K','V','O','C','B','\0' }; // 8 bytes
    const std::uint32_t version = 1;
    const std::uint32_t count = (std::uint32_t)tokenToWord.size();

    ofs.write(magic, sizeof(magic));
    ofs.write((const char*)&version, sizeof(version));
    ofs.write((const char*)&count, sizeof(count));
    if (!ofs.good())
        return false;

    // Body: [u32 byteLen][bytes...]
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::string& s = tokenToWord[(size_t)i];

        // Guard: keep lengths representable in u32
        if (s.size() > (size_t)std::numeric_limits<std::uint32_t>::max())
            return false;

        const std::uint32_t len = (std::uint32_t)s.size();
        ofs.write((const char*)&len, sizeof(len));
        if (len > 0) {
            ofs.write(s.data(), (std::streamsize)len);
        }

        if (!ofs.good())
            return false;
    }

    return true;
}

bool Tokenizer::LoadFromFile(const std::string& filename) {
    std::ifstream ifs(filename.c_str(), std::ios::binary | std::ios::in);
    if (!ifs.is_open())
        return false;
    
    // Header
    char magic[8];
    std::uint32_t version = 0;
    std::uint32_t count = 0;
    
    ifs.read(magic, sizeof(magic));
    ifs.read((char*)&version, sizeof(version));
    ifs.read((char*)&count, sizeof(count));
    if (!ifs.good())
        return false;
    
    const char expected[8] = { 'T','O','K','V','O','C','B','\0' };
    for (int i = 0; i < 8; ++i) {
        if (magic[i] != expected[i])
            return false;
    }
    if (version != 1)
        return false;
    
    // Reset
    wordToToken.clear();
    tokenToWord.clear();
    mCurrentToken = 0;
    
    tokenToWord.reserve((size_t)count);
    wordToToken.reserve((size_t)count);
    
    // Body
    for (std::uint32_t i = 0; i < count; ++i) {
        std::uint32_t len = 0;
        ifs.read((char*)&len, sizeof(len));
        if (!ifs.good())
            return false;
        
        std::string s;
        if (len > 0) {
            s.resize((size_t)len);
            ifs.read(&s[0], (std::streamsize)len);
            if (!ifs.good())
                return false;
        }
        
        // Reject duplicates so GetToken() stays deterministic
        if (wordToToken.find(s) != wordToToken.end())
            return false;
        
        wordToToken[s] = (int)mCurrentToken;
        tokenToWord.push_back(s);
        mCurrentToken++;
    }
    
    return true;
}

bool Tokenizer::TokenizeWordBPE(const std::string& word, 
                                std::vector<int>& outTokens, 
                                const std::string& unkToken, 
                                bool useContinuationPrefix, 
                                const std::string& continuationPrefix) const {
    outTokens.clear();
    
    if (word.empty()) 
        return true;
    
    // Fast path - check whole word exists
    std::unordered_map<std::string, int>::const_iterator it = wordToToken.find(word);
    if (it != wordToToken.end()) {
        outTokens.push_back(it->second);
        return true;
    }
    
    const unsigned int n = (unsigned int)word.size();
    unsigned int i = 0;
    
    while (i < n) {
        bool found = false;
        
        int bestToken = -1;
        unsigned int bestJ = i;
        
        // Try longest substring first: word[i..j)
        for (unsigned int j = n; j > i; --j) {
            const unsigned int len = j - i;
            std::string piece = word.substr((size_t)i, (size_t)len);
            
            // Prefer continuation-prefix for non-initial pieces
            if (i != 0 && useContinuationPrefix) {
                std::string contPiece = continuationPrefix + piece;
                std::unordered_map<std::string, int>::const_iterator itCont = wordToToken.find(contPiece);
                if (itCont != wordToToken.end()) {
                    bestToken = itCont->second;
                    bestJ = j;
                    found = true;
                    break; // longest-first
                }
            }
            
            std::unordered_map<std::string, int>::const_iterator it = wordToToken.find(piece);
            if (it != wordToToken.end()) {
                bestToken = it->second;
                bestJ = j;
                found = true;
                break; // longest-first
            }
        }
        
        if (!found) {
            // Fallback to unk token if available
            if (!unkToken.empty()) {
                std::unordered_map<std::string, int>::const_iterator itUnk = wordToToken.find(unkToken);
                if (itUnk != wordToToken.end()) {
                    outTokens.clear();
                    outTokens.push_back(itUnk->second);
                    return true;
                }
            }
            return false;
        }
        
        outTokens.push_back(bestToken);
        i = bestJ;
    }
    
    return true;
}

