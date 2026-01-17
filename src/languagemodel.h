#ifndef LANGUAGE_MODEL_H
#define LANGUAGE_MODEL_H

#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <limits>
#include <cstdint>

#include "tokenizer.h"
#include "attention.h"
#include "embedding.h"

class LanguageModel {
public:
    
    Tokenizer tok;
    
    AttentionSystem attention;
    EmbeddingSystem embedding;
    
    std::vector<std::vector<int>> mModel;
    
    int mMaxSpanLen;
    int mSpanStride;
    
    // Add new source corpus to the model.
    void ProcessSequence(std::vector<int> tokens, float learningRate);
    
    // Extract relevant spans from the model.
    // "Relevant" = contains topic tokens; returns top spans sorted by match strength.
    std::vector<std::vector<int>> Extract(std::vector<int> topic);
    
    // Load a model file.
    bool LoadFromFile(const std::string& filename);
    
    // Save the model to a file.
    bool SaveToFile(const std::string& filename) const;
    
    LanguageModel();
};

#endif
