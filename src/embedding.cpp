#include "embedding.h"

#include <fstream>
#include <cstdint>

EmbeddingSystem::EmbeddingSystem() {
}

void EmbeddingSystem::Clear(void) {
    mEmbeddings.clear();
}

void EmbeddingSystem::AddEmbedding(int token, const Embedding& emb) {
    mEmbeddings[token] = emb;
}

void EmbeddingSystem::AddEmbedding(int token) {
    // If we already have an embedding for this token, don't overwrite it.
    if (mEmbeddings.find(token) != mEmbeddings.end()) {
        return;
    }
    
    Embedding emb;
    
    // Simple random init in a small range, e.g. [-0.1, 0.1]
    for (int i = 0; i < EMBEDDING_WIDTH; ++i) {
        float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        emb.v[i] = r * 0.2f - 0.1f;              // -0.1 .. 0.1
    }
    
    mEmbeddings[token] = emb;
}

void EmbeddingSystem::TrainOnSentence(const std::vector<int>& tokens, int windowSize, float strength) {
    if (tokens.empty())
        return;
    if (windowSize <= 0)
        return;
    if (strength <= 0.0f)
        return;
    
    const int n = static_cast<int>(tokens.size());
    
    for (int i = 0; i < n; ++i) {
        int centerToken = tokens[static_cast<std::size_t>(i)];
        
        // Ensure the center token has an embedding.
        AddEmbedding(centerToken);
        
        Embedding& centerEmb = mEmbeddings[centerToken];
        
        int start = i - windowSize;
        if (start < 0) start = 0;
        int end = i + windowSize;
        if (end >= n) end = n - 1;
        
        for (int j = start; j <= end; ++j) {
            if (j == i) continue;
            
            int neighborToken = tokens[static_cast<std::size_t>(j)];
            
            if (neighborToken < 0) {
                continue;
            }
            
            int dim = neighborToken % EMBEDDING_WIDTH;
            if (dim < 0) {
                dim += EMBEDDING_WIDTH;
            }
            
            centerEmb.v[dim] += strength;
        }
    }
}

bool EmbeddingSystem::HasEmbedding(int token) const {
    return mEmbeddings.find(token) != mEmbeddings.end();
}

bool EmbeddingSystem::GetEmbedding(int token, Embedding& outEmbedding) const {
    std::unordered_map<int, Embedding>::const_iterator it = mEmbeddings.find(token);
    if (it == mEmbeddings.end()) {
        return false;
    }
    outEmbedding = it->second;
    return true;
}

const Embedding* EmbeddingSystem::GetEmbeddingPtr(int token) const {
    std::unordered_map<int, Embedding>::const_iterator it = mEmbeddings.find(token);
    if (it == mEmbeddings.end()) {
        return NULL;
    }
    return &it->second;
}

std::size_t EmbeddingSystem::size(void) const {
    return mEmbeddings.size();
}

bool EmbeddingSystem::SaveToFile(const std::string& filename) const {
    std::ofstream out(filename.c_str(), std::ios::binary);
    if (!out.is_open()) {
        return false;
    }
    
    std::uint32_t count = static_cast<std::uint32_t>(mEmbeddings.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    if (!out.good()) {
        return false;
    }
    
    for (std::unordered_map<int, Embedding>::const_iterator it = mEmbeddings.begin();
         it != mEmbeddings.end(); ++it) {
        
        std::int32_t tokenId = static_cast<std::int32_t>(it->first);
        out.write(reinterpret_cast<const char*>(&tokenId), sizeof(tokenId));
        if (!out.good()) {
            return false;
        }
        
        const Embedding& emb = it->second;
        out.write(reinterpret_cast<const char*>(emb.v),
                  sizeof(float) * static_cast<std::size_t>(EMBEDDING_WIDTH));
        if (!out.good()) {
            return false;
        }
    }
    
    return out.good();
}

bool EmbeddingSystem::LoadFromFile(const std::string& filename) {
    std::ifstream in(filename.c_str(), std::ios::binary);
    if (!in.is_open()) {
        return false;
    }
    
    mEmbeddings.clear();
    
    std::uint32_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!in.good()) {
        return false;
    }
    
    for (std::uint32_t i = 0; i < count; ++i) {
        std::int32_t tokenId = 0;
        in.read(reinterpret_cast<char*>(&tokenId), sizeof(tokenId));
        if (!in.good()) {
            mEmbeddings.clear();
            return false;
        }
        
        Embedding emb;
        in.read(reinterpret_cast<char*>(emb.v),
                sizeof(float) * static_cast<std::size_t>(EMBEDDING_WIDTH));
        if (!in.good()) {
            mEmbeddings.clear();
            return false;
        }
        
        mEmbeddings[static_cast<int>(tokenId)] = emb;
    }
    
    return true;
}
