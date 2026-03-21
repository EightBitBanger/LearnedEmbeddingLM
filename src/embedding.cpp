#include "embedding.h"

#include <fstream>
#include <cstdint>
#include <math.h>

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
    if (mEmbeddings.find(token) != mEmbeddings.end()) 
        return;
    
    Embedding embedding;
    
    // Simple random init in a small range  (-0.1 - 0.1)
    for (int i = 0; i < EMBEDDING_WIDTH; ++i) {
        float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
        embedding.v[i] = r * 0.2f - 0.1f;
    }
    
    mEmbeddings[token] = embedding;
}

void EmbeddingSystem::TrainOnSentence(const std::vector<int>& tokens, int windowSize, float strength) {
    if (tokens.size() < 2) return;
    for (int i = 0; i < (int)tokens.size(); ++i) {
        int targetToken = tokens[i];
        
        // Ensure the target has an embedding base to work from
        if (!HasEmbedding(targetToken)) {
            AddEmbedding(targetToken); 
        }
        
        Embedding& targetEmb = mEmbeddings[targetToken];
        
        // Define the local window
        int start = std::max(0, i - windowSize);
        int end = std::min((int)tokens.size() - 1, i + windowSize);
        
        for (int j = start; j <= end; ++j) {
            if (i == j) continue; // Don't train on yourself
            
            int neighborToken = tokens[j];
            
            // Closer words have more semantic influence
            // Linear decay: 1.0 for immediate neighbors, decreasing as distance increases
            float distance = (float)std::abs(i - j);
            float weight = strength * (1.0f - (distance / (float)(windowSize + 1)));
            
            // High-Frequency Dimension Mapping via a simple hash with a secondary salt to reduce collisions
            unsigned int dim = (unsigned int)(neighborToken * 2654435761u) % EMBEDDING_WIDTH;
            
            // 3. The "Bumping" Logic we increment the dimension
            targetEmb.v[dim] += weight;
            
            // optional; Slightly penalize the neighbor's dimension in "rival" categories to keep the vectors sparse and distinct.
            unsigned int antiDim = (dim + (EMBEDDING_WIDTH / 2)) % EMBEDDING_WIDTH;
            targetEmb.v[antiDim] -= (weight * 0.2f);
        }
    }
    
    // Post-Training Normalization
    for (int i = 0; i < (int)tokens.size(); ++i) {
        Normalize(tokens[i]);
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

void EmbeddingSystem::Normalize(int token) {
    if (!HasEmbedding(token)) return;
    Embedding& emb = mEmbeddings[token];
    
    float magSq = 0.0f;
    for (int d = 0; d < EMBEDDING_WIDTH; ++d) magSq += (emb.v[d] * emb.v[d]);
    
    if (magSq > 0.00001f) {
        float invMag = 1.0f / std::sqrt(magSq);
        for (int d = 0; d < EMBEDDING_WIDTH; ++d) emb.v[d] *= invMag;
    }
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
