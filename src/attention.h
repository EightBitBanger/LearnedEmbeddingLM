#ifndef _ATTENTION__
#define _ATTENTION__

#include <unordered_map>
#include <vector>
#include <string>

// Key for a single edge (anchor, neighbor, offset)
struct AttentionKey {
    int anchor;
    int neighbor;
    int offset;
    
    bool operator==(const AttentionKey& other) const noexcept {
        return anchor  == other.anchor &&
               neighbor == other.neighbor &&
               offset   == other.offset;
    }
};

// Hash functor for the unordered_map
struct AttentionKeyHash {
    std::size_t operator()(const AttentionKey& k) const noexcept {
        std::size_t h1 = std::hash<int>()(k.anchor);
        std::size_t h2 = std::hash<int>()(k.neighbor);
        std::size_t h3 = std::hash<int>()(k.offset);
        
        // Standard hash-combine pattern.
        std::size_t seed = h1;
        seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};

struct AttentionEdge {
    float        weight;
    unsigned int count;
    unsigned int lastUpdateStep;
    
    AttentionEdge() : 
        weight(0.0f),
        count(0u),
        lastUpdateStep(0u) {}
};

struct TokenState {
    unsigned int asAnchorCount;
    unsigned int asNeighborCount;
    unsigned int totalEdges;
    
    float degree;        // approximate graph degree (unique neighbors)
    float relationScore; // how "relation/glue-like" this token behaves
    float contentScore;  // how "content/entity-like" this token behaves
    
    TokenState() : 
        asAnchorCount(0u),
        asNeighborCount(0u),
        totalEdges(0u),
        degree(0.0f),
        relationScore(0.0f),
        contentScore(0.0f) {}
};

struct TokenInfo {
    bool hasStats;        // false if token not found in tokenStats
    
    // Soft probabilities (heuristic) that the token is function/content-like.
    float pFunction;      // in [0,1]
    float pContent;       // in [0,1], and pFunction + pContent ~= 1
    
    // How confident we are in those probabilities, based on totalEdges.
    // 0.0  -> no data (both probs ~0.5)
    // 1.0  -> plenty of data (probs fully reflect scores)
    float confidence;     
    
    // Raw stats (for inspection / debugging / tuning).
    float degree;
    float relationScore;
    float contentScore;
    unsigned int totalEdges;
    
    TokenInfo() : 
        hasStats(false),
        pFunction(0.5f),
        pContent(0.5f),
        confidence(0.0f),
        degree(0.0f),
        relationScore(0.0f),
        contentScore(0.0f),
        totalEdges(0u)
    {}
};


class AttentionSystem {
public:
    
    float        baseWeight;
    float        falloff;
    
    // Flatter representation: each edge is keyed by (anchor, neighbor, offset).
    std::unordered_map<AttentionKey, AttentionEdge, AttentionKeyHash> attention;
    
    // Per-token usage stats (for role inference).
    std::unordered_map<int, TokenState> tokenStats;
    
    // Simple learning step counter (for optional aging/decay).
    unsigned int updateStep;
    
    AttentionSystem();
    
    // Learn from a sequence of tokens.
    void ProcessSequence(const std::vector<int>& tokens);
    
    // Normalize all the attention scores
    void RenormalizeAll(float weightScale);
    
    // Clear out the attention scores and role stats.
    void Clear();
    
    // Score a specific (anchor, candidate, offset) triple.
    float GetScore(int anchor, int candidate, int offset) const;
    
    // Aggregate score over all offsets for (anchor, candidate).
    float GetScore(int anchor, int candidate) const;
    
    // Score a candidate next token using the full context, using
    // the proper offset (next position index - anchor index).
    float GetScore(const std::vector<int>& context, int token_j) const;
    
    // Pick highest-scoring candidate from a list.
    int GetNextToken(const std::vector<int>& context,
                     const std::vector<int>& allTokens);
    
    // Normalizes each weight per anchor adding up to 1
    void NormalizeWeightsPerAnchor();
    
    // Set the score for a specific (tokenA, tokenB, offset).
    void SetScore(int tokenA, int tokenB, int offset, float score);
    
    // Set the *aggregate* score for (tokenA, tokenB), distributing across offsets.
    void SetScore(int tokenA, int tokenB, float score);
    
    // Scale a specific (tokenA, tokenB, offset) association.
    void AdjustScore(int tokenA, int tokenB, int offset, float multiplier);
    
    // Scale all offsets for (tokenA, tokenB).
    void AdjustScore(int tokenA, int tokenB, float multiplier);
    
    // Weighted average offset of tokenB relative to tokenA.
    float GetAverageOffset(int tokenA, int tokenB) const;
    
    // Recompute per-token role scores (degree / relationScore / contentScore)
    // from the current attention graph and tokenStats.
    void RecomputeRoleScores(void);
    
    // Access per-token role stats; may return NULL if token is unknown.
    const TokenState* GetTokenStats(int token) const;
    
    // Return soft role probabilities and raw stats for a token.
    // edgesForFullConfidence is how many edges we consider "enough data"
    // to trust the scores fully (above that, confidence ~1).
    TokenInfo GetTokenInfo(int token) const;
    
    // Get the score relating to the word being a content word.
    float GetContentScore(int token) const;
    
    // Cull tokens that have low usage.
    unsigned int PruneLowInteractionTokens(unsigned int minTotalEdges, float minAbsWeightSum, bool requireBoth);
    
    // Save the attention scoring data to a file.
    bool SaveToFile(const std::string& filename) const;
    
    // Load the attention scoring data from a file.
    bool LoadFromFile(const std::string& filename);
};

#endif
