#ifndef _SAMPLER__
#define _SAMPLER__

#include "embedding.h"
#include "attention.h"
#include <unordered_map>
#include <vector>

struct VocalIntention;

struct SamplerParameters {
    
    float temperatureHigh; // Loosen sampling to broaden the search
    float temperatureLow;  // Tighten sampling onto a specific span
    float attentionRate;   // Strength of attention
    float embeddingRate;   // Strength of embedding
    
    SamplerParameters() : 
        temperatureHigh(1.2f), 
        temperatureLow(0.3f),
        attentionRate(0.1f),
        embeddingRate(0.3f) {}
    
};

struct TokenDistribution {
    std::vector<int> tokens;
    std::vector<double> weights;
};


class SamplerSystem {
public:
    // Positional attention
    AttentionSystem* attention;
    // Token embeddings
    EmbeddingSystem* embedding;
    
    int SampleNextToken(const std::vector<int>& context,
                        const std::vector<std::vector<int>>& focus,
                        const SamplerParameters& params);
    
    TokenDistribution SampleNextTokenDistribution(const std::vector<int>& context,
                                                  const std::vector<std::vector<int>>& focus,
                                                  const SamplerParameters& params, int topk);
    
    SamplerSystem(AttentionSystem* attentionPtr, EmbeddingSystem* embeddingPtr);
    
private:
    // High-level steps of the sampler:
    int  GetSentenceStart(int contextSize, int maxSentenceLen) const;
    
    void ComputeSpanBestMatches(const std::vector<int>& context,
                                int sentenceStart,
                                int maxSentenceLen,
                                const std::vector<std::vector<int>>& focus,
                                std::vector<int>& spanBestLen,
                                int& globalBestLen,
                                int& globalBestSpan) const;
    
    void BuildScoreMaps(const std::vector<int>& context,
                        int sentenceStart,
                        int maxSentenceLen,
                        const std::vector<std::vector<int>>& focus,
                        const std::vector<int>& spanBestLen,
                        int globalBestLen,
                        int globalBestSpan,
                        std::unordered_map<int, double>& lockedScores,
                        std::unordered_map<int, double>& allScores) const;
    
    void FallbackToFrequencyScores(const std::vector<std::vector<int>>& focus,
                                   std::unordered_map<int, double>& allScores,
                                   int& globalBestLen) const;
    
    void ChooseScoreSource(int globalBestLen,
                           const std::unordered_map<int, double>& lockedScores,
                           const std::unordered_map<int, double>& allScores,
                           const SamplerParameters& params,
                           bool& useLockedScores,
                           float& effectiveTemp) const;
    
    void BuildTokenDistribution(const std::vector<int>& context,
                                const std::unordered_map<int, double>& baseScores,
                                const SamplerParameters& params,
                                float effectiveTemp,
                                std::vector<int>& tokens,
                                std::vector<double>& weights,
                                double& totalWeight) const;
    
    int  SampleFromDistribution(const std::vector<int>& tokens,
                                const std::vector<double>& weights,
                                double totalWeight) const;
};

#endif
