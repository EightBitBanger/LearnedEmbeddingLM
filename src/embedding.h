#ifndef _EMBEDDING__
#define _EMBEDDING__

#define EMBEDDING_WIDTH  128

#include <string>
#include <vector>
#include <unordered_map>

struct Embedding {
    
    float v[EMBEDDING_WIDTH];
    
};

class EmbeddingSystem {
public:
    
    EmbeddingSystem();
    
    // Remove all embeddings.
    void Clear(void);
    
    // Add or replace an embedding for a token.
    void AddEmbedding(int token, const Embedding& emb);
    
    // Add a random embedding for a token if its unknown.
    void AddEmbedding(int token);
    
    // Simple analytic "training": for each token, we bump dimensions in its
    // embedding based on nearby tokens in the sentence.
    // Each neighbor token hashes to a dimension: neighborToken % EMBEDDING_WIDTH.
    void TrainOnSentence(const std::vector<int>& tokens, int windowSize, float strength);
    
    // Check if we have an embedding for this token.
    bool HasEmbedding(int token) const;
    
    // Copy embedding out; returns false if not found.
    bool GetEmbedding(int token, Embedding& outEmbedding) const;
    
    // Pointer access; returns NULL if not found.
    const Embedding* GetEmbeddingPtr(int token) const;
    
    // Number of stored embeddings.
    std::size_t size(void) const;
    
    // Load embeddings from a binary file.
    bool LoadFromFile(const std::string& filename);
    
    // Save embeddings to a binary file.
    bool SaveToFile(const std::string& filename) const;
    
private:
    
    std::unordered_map<int, Embedding> mEmbeddings;
    
};

#endif
