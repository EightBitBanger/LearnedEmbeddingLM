#ifndef GLOBAL_SEMANTIC_ALIGNMENT_H
#define GLOBAL_SEMANTIC_ALIGNMENT_H

#include "tokenizer.h"
#include "attention.h"
#include "sampler.h"

#include <string>
#include <vector>


class GlobalSemanticAlignment {
public:
    GlobalSemanticAlignment(Tokenizer* tokPtr, SamplerSystem* samplerPtr, AttentionSystem* attentionPtr);
    
    int SampleAligned(const std::vector<int>& context, const std::vector<std::vector<int>>& focus, const SamplerParameters& params);
    
    // Soft question detector: returns [0,1] where 1 is a strong question.
    float GetQuestionScore(const std::vector<int>& context);
    
    // Extracts subject/topic tokens from the most recent question-like sentence.
    std::vector<int> GetQuestionSubject(const std::vector<int>& context, unsigned int maxTokens = 8);
    
    
private:
    
    void ResolveWordList(const std::vector<std::string>& words, std::vector<int>& outTokens);
    void RefreshLexicons();
    
    std::vector<int> questionWords;
    std::vector<int> negationWords;
    std::vector<int> futureMarkers;
    std::vector<int> pastMarkers;
    std::vector<int> beAux;
    std::vector<int> haveAux;
    std::vector<int> doAux;
    std::vector<int> prepositionWords;
    std::vector<int> intensityUpWords;
    std::vector<int> intensityDownWords;
    std::vector<int> pronounWords;
    
    Tokenizer* tok;
    SamplerSystem* sampler;
    AttentionSystem* attention;
};

#endif
